# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

# External imports
import time
import sys
import os
import glob
import copy
import uuid
import json
import cProfile
import gzip
import tracemalloc
import pickle
from typing import Optional, Any
import goexplore_py.globals as global_const
import tensorflow as tf

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)

import horovod.tensorflow as hvd
import numpy as np
from PIL import Image
import imageio
import tempfile
import errno
import shutil
import logging
import contextlib

# Go-Explore imports
from goexplore_py.utils import get_code_hash
from goexplore_py.experiment_settings import hrv_and_tf_init, parse_arguments, setup, process_defaults, \
    del_out_of_setup_args
from goexplore_py.logger import SimpleLogger
from goexplore_py.profiler import display_top
import goexplore_py.mpi_support as mpi
from atari_reset.atari_reset.ppo import flatten_lists
from goexplore_py.data_classes import LogParameters
local_logger = logging.getLogger(__name__)

compress = gzip
compress_suffix = '.gz'
compress_kwargs = {'compresslevel': 1}

MODEL_POSTFIX = '_model.joblib'
ARCHIVE_POSTFIX = '_arch'
TRAJ_POSTFIX = '_traj.tfrecords'

CHECKPOINT_ABBREVIATIONS = {
    'model': MODEL_POSTFIX,
    'archive': ARCHIVE_POSTFIX + compress_suffix,
    'trajectory': TRAJ_POSTFIX
}


def save_state(state, filename):
    p = pickle.Pickler(compress.open(filename + compress_suffix, 'wb', **compress_kwargs))
    p.fast = True
    p.dump(state)


VERSION = 1
PROFILER: Optional[Any] = None


class SumTwo:
    def __call__(self, a, b):
        return a + b


class GetWeight:
    def __init__(self, expl, total):
        self.expl = expl
        self.total = total

    def __call__(self, x):
        cell_key = x
        cell = self.expl.archive.archive[x]
        archive = self.expl.archive.archive
        weight = self.expl.archive.cell_selector.get_weight(cell_key, cell, archive)
        prob = weight / self.total
        return prob


def render_pictures(log_par: LogParameters, expl, filename, prev_checkpoint, screenshots, sil_trajectories):
    # Synchronize screenshots
    expl.trajectory_gatherer.env.recursive_setattr('rooms', screenshots)
    env_render = expl.trajectory_gatherer.env.recursive_getattr('render_with_known')[0]
    archive = expl.archive.archive
    all_cells = list(archive)
    x_res = expl.trajectory_gatherer.env.recursive_getattr('x_res')[0]
    y_res = expl.trajectory_gatherer.env.recursive_getattr('y_res')[0]

    def render(local_name, get_val, log_scale=False):
        local_file_name = filename + '_' + local_name + '.png'
        env_render(all_cells, x_res, y_res, filename=local_file_name, get_val=get_val, log_scale=log_scale)

    names = set('')

    # Show cells. Colors indicate how many different cells where mapped to the same location in the picture.
    name = 'cells'
    if log_par.should_render(name):
        render(name, lambda x: 1)
        names.add(name)

    # Show probability grid
    name = 'sel_prob'
    if log_par.should_render(name):
        expl.archive.cell_selector.update_all = True
        prob_dict = expl.archive.cell_selector.get_probabilities_dict(expl.archive.archive)
        render(name, lambda x: prob_dict[x], log_scale=True)
        names.add(name)

    # Show how many times we have reached a cell
    name = 'reached'
    if log_par.should_render(name):
        render(name, lambda x: archive[x].nb_reached)
        names.add(name)

    # Show how many times we have chosen a cell
    name = 'chosen'
    if log_par.should_render(name):
        render(name, lambda x: archive[x].nb_chosen)
        names.add(name)

    # Show how many times we have chosen a cell
    name = 'sub_goal_fail'
    if log_par.should_render(name):
        render(name, lambda x: archive[x].nb_sub_goal_failed)
        names.add(name)

    # Show the return probability of this cell
    name = 'ret_prob'
    if log_par.should_render(name):
        def local_get_weight(x):
            if x in expl.archive.cells_reached_dict:
                if len(expl.archive.cells_reached_dict[x]) > 0:
                    return sum(expl.archive.cells_reached_dict[x]) / len(expl.archive.cells_reached_dict[x])
            return 0
        render(name, local_get_weight)
        names.add(name)

    # Show trajectories currently being imitated
    name = 'sil_traj'
    if log_par.should_render(name):
        for i, traj_id in enumerate(sil_trajectories):
            if traj_id in expl.archive.cell_trajectory_manager.cell_trajectories:
                trajectory = expl.archive.cell_trajectory_manager.get_trajectory(traj_id, -1,
                                                                                 expl.archive.cell_id_to_key_dict)
                cells = dict()
                cells_count = dict()
                for j, traj_elem in enumerate(trajectory):
                    cell_key, _nb_actions = traj_elem
                    if cell_key not in cells_count:
                        cells_count[cell_key] = 0
                    cells_count[cell_key] += 1
                    if len(trajectory) > 1:
                        val = j / (len(trajectory)-1)
                    else:
                        val = 0
                    if cell_key not in cells:
                        cells[cell_key] = val
                    else:
                        cells[cell_key] = (cells[cell_key] + val) / cells_count[cell_key]
                render(str(i) + '_' + name, lambda x: cells.get(x, -1))
        names.add(name)

    for picture in log_par.save_pictures:
        assert picture in names, 'Can not render unknown picture type: ' + picture

    if prev_checkpoint:
        for picture in log_par.clear_pictures:
            pattern = prev_checkpoint + '*' + '_' + picture + '.png'
            paths = glob.glob(pattern)
            for path in paths:
                os.remove(path)


class CheckpointTracker:
    def __init__(self, log_par, expl):
        self.log_par = log_par
        self.expl = expl
        self.old = None
        self.old_compute = None
        self.old_it = None
        self.old_len_grid = None
        self.old_max_score = None
        self.cur_time = None
        self.start_time = time.time()
        self.last_time = np.round(self.start_time)
        self.n_iters = 0
        self.log_warmup = False
        self.will_log_warmup = False
        self.old_time_passed = 0
        self.current_time_passed = 0
        self.pre_cycle_time = None
        self._should_write_checkpoint = False

    def pre_cycle(self):
        self.old_compute = self.expl.frames_compute
        self.old_it = self.expl.cycles
        self.old_len_grid = len(self.expl.archive.archive)
        self.old_max_score = self.expl.archive.max_score
        self.pre_cycle_time = time.time()
        self.will_log_warmup = False

    def post_cycle(self):
        self.old_time_passed = self.current_time_passed
        self.current_time_passed += time.time() - self.pre_cycle_time
        self.cur_time = np.round(time.time())
        self.last_time = self.cur_time
        self.n_iters += 1
        # After the first cycle, if log_warmup is True (meaning we just completed the warmup phase), we will promise
        # to perform the warmup by setting will_log_warmup to True, and then we can set log_warmup to False for the next
        # iteration.
        if self.log_warmup:
            self.will_log_warmup = True
            self.log_warmup = False

    def calc_write_checkpoint(self):
        if self.log_par.checkpoint_compute is not None:
            passed_compute_thresh = (self.old_compute // self.log_par.checkpoint_compute !=
                                     self.expl.frames_compute // self.log_par.checkpoint_compute)
        else:
            passed_compute_thresh = False
        if self.log_par.checkpoint_it is not None:
            passed_it_thresh = (self.old_it // self.log_par.checkpoint_it !=
                                self.expl.cycles // self.log_par.checkpoint_it)
        else:
            passed_it_thresh = False
        if self.log_par.checkpoint_first_iteration:
            first_it = self.old_it == 0
        else:
            first_it = False
        if self.log_par.checkpoint_first_iteration:
            last_it = not self.should_continue()
        else:
            last_it = False
        if self.log_par.checkpoint_time is not None:
            passed_time_thresh = (self.old_time_passed // self.log_par.checkpoint_time !=
                                  self.current_time_passed // self.log_par.checkpoint_time)
        else:
            passed_time_thresh = False

        return (first_it or passed_compute_thresh or passed_it_thresh or last_it or
                self.will_log_warmup or passed_time_thresh)

    def set_should_write_checkpoint(self, write_checkpoint):
        self._should_write_checkpoint = write_checkpoint

    def should_write_checkpoint(self):
        return self._should_write_checkpoint

    def should_continue(self):
        if self.log_par.max_time is not None and time.time() - self.start_time >= self.log_par.max_time:
            return False
        if self.log_par.max_compute_steps is not None and self.expl.frames_compute >= self.log_par.max_compute_steps:
            return False
        if self.log_par.max_iterations is not None and self.n_iters >= self.log_par.max_iterations:
            return False
        if self.log_par.max_cells is not None and len(self.expl.archive.archive) >= self.log_par.max_cells:
            return False
        if self.log_par.max_score is not None and self.expl.archive.max_score >= self.log_par.max_score:
            return False
        return True


def _run(**kwargs):
    # Make sure that, if one worker crashes, the entire MPI process is aborted
    def handle_exception(exc_type, exc_value, exc_traceback):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        sys.stderr.flush()
        if hvd.size() > 1:
            mpi.COMM_WORLD.Abort(1)
    sys.excepthook = handle_exception

    track_memory = kwargs['trace_memory']
    disable_logging = bool(kwargs['disable_logging'])
    warm_up_cycles = kwargs['warm_up_cycles']
    log_after_warm_up = kwargs['log_after_warm_up']
    screenshot_merge = kwargs['screenshot_merge']
    clear_checkpoints = list(filter(None, kwargs['clear_checkpoints'].split(':')))

    if 'all' in clear_checkpoints:
        clear_checkpoints = CHECKPOINT_ABBREVIATIONS.keys()

    if track_memory:
        tracemalloc.start(25)

    kwargs = del_out_of_setup_args(kwargs)
    expl, log_par = setup(**kwargs)
    local_logger.info('setup done')

    # We only need one MPI worker to log the results
    local_logger.info('Initializing logger')
    logger = None
    traj_logger = None
    if hvd.rank() == 0 and not disable_logging:
        logger = SimpleLogger(log_par.base_path + '/log.txt')
        traj_logger = SimpleLogger(log_par.base_path + '/traj_log.txt')

    ########################
    # START THE EXPERIMENT #
    ########################

    local_logger.info('Starting experiment')
    checkpoint_tracker = CheckpointTracker(log_par, expl)
    prev_checkpoint = None
    merged_dict = {}
    sil_trajectories = []
    if screenshot_merge[0:9] == 'from_dir:':
        screen_shot_dir = screenshot_merge[9:]
    else:
        screen_shot_dir = f'{log_par.base_path}/screen_shots'

    local_logger.info('Initiate cycle')
    expl.init_cycle()
    local_logger.info('Initiating Cycle done')

    if kwargs['expl_state'] is not None:
        local_logger.info('Performing warm up cycles...')
        expl.start_warm_up()
        for i in range(warm_up_cycles):
            if hvd.rank() == 0:
                local_logger.info(f'Running warm up cycle: {i}')
            expl.run_cycle()
        expl.end_warm_up()
        checkpoint_tracker.n_iters = expl.cycles
        checkpoint_tracker.log_warmup = log_after_warm_up
        local_logger.info('Performing warm up cycles... done')
    
    while checkpoint_tracker.should_continue():
        # Run one iteration
        if hvd.rank() == 0:
            local_logger.info(f'Running cycle: {checkpoint_tracker.n_iters}')

        checkpoint_tracker.pre_cycle()
        expl.run_cycle()
        checkpoint_tracker.post_cycle()

        write_checkpoint = None
        if hvd.rank() == 0:
            write_checkpoint = checkpoint_tracker.calc_write_checkpoint()
        write_checkpoint = mpi.get_comm_world().bcast(write_checkpoint, root=0)
        checkpoint_tracker.set_should_write_checkpoint(write_checkpoint)

        # Code that should be executed by all workers at a checkpoint generation
        if checkpoint_tracker.should_write_checkpoint():
            local_logger.debug(f'Rank: {hvd.rank()} is exchanging screenshots for checkpoint: {expl.frames_compute}')
            screenshots = expl.trajectory_gatherer.env.recursive_getattr('rooms')
            if screenshot_merge == 'mpi':
                screenshots = flatten_lists(mpi.COMM_WORLD.allgather(screenshots))
            merged_dict = {}
            for screenshot_dict in screenshots:
                for key, value in screenshot_dict.items():
                    if key not in merged_dict:
                        merged_dict[key] = value
                    else:
                        after_threshold_screenshot_taken_merged = merged_dict[key][0]
                        after_threshold_screenshot_taken_current = screenshot_dict[key][0]
                        if after_threshold_screenshot_taken_current and not after_threshold_screenshot_taken_merged:
                            merged_dict[key] = value

            if screenshot_merge == 'disk':
                for key, value in merged_dict.items():
                    filename = f'{screen_shot_dir}/{key}_{hvd.rank()}.png'
                    os.makedirs(screen_shot_dir, exist_ok=True)
                    if not os.path.isfile(filename):
                        im = Image.fromarray(value[1])
                        im.save(filename)
                        im_array = imageio.imread(filename)
                        assert (im_array == value[1]).all()

                mpi.COMM_WORLD.barrier()

            local_logger.debug('Merging SIL trajectories')
            sil_trajectories = [expl.prev_selected_traj]
            if hvd.size() > 1:
                sil_trajectories = flatten_lists(mpi.COMM_WORLD.allgather(sil_trajectories))
            local_logger.debug(f'Rank: {hvd.rank()} is done merging trajectories for checkpoint: {expl.frames_compute}')

            expl.sync_before_checkpoint()
            local_logger.debug(f'Rank: {hvd.rank()} is done synchronizing for checkpoint: {expl.frames_compute}')

        # Code that should be executed only by the master
        if hvd.rank() == 0 and not disable_logging:
            gatherer = expl.trajectory_gatherer
            return_success_rate = -1
            if gatherer.nb_return_goals_chosen > 0:
                return_success_rate = gatherer.nb_return_goals_reached / gatherer.nb_return_goals_chosen
            exploration_success_rate = -1
            if gatherer.nb_exploration_goals_chosen > 0:
                exploration_success_rate = gatherer.nb_exploration_goals_reached / gatherer.nb_exploration_goals_chosen

            cum_success_rate = 0
            for reached in expl.archive.cells_reached_dict.values():
                success_rate = sum(reached) / len(reached)
                cum_success_rate += success_rate
            mean_success_rate = cum_success_rate / len(expl.archive.archive)

            logger.write('it', checkpoint_tracker.n_iters)
            logger.write('score', expl.archive.max_score)
            logger.write('cells', len(expl.archive.archive))
            logger.write('ret_suc', return_success_rate)
            logger.write('exp_suc', exploration_success_rate)
            logger.write('rew_mean', gatherer.reward_mean)
            logger.write('len_mean', gatherer.length_mean)
            logger.write('ep', gatherer.nb_of_episodes)
            logger.write('arch_suc', mean_success_rate)
            logger.write('cum_suc', cum_success_rate)
            logger.write('frames', expl.frames_compute)

            if len(gatherer.loss_values) > 0:
                loss_values = np.mean(gatherer.loss_values, axis=0)
                assert len(loss_values) == len(gatherer.model.loss_names)
                for (loss_value, loss_name) in zip(loss_values, gatherer.model.loss_names):
                    logger.write(loss_name, loss_value)

            stored_frames = 0
            for traj in expl.archive.cell_trajectory_manager.full_trajectories.values():
                stored_frames += len(traj)

            logger.write('sil_frames', stored_frames)

            nb_no_score_cells = len(expl.archive.archive)
            for weight in expl.archive.cell_selector.selector_weights:
                if hasattr(weight, 'max_score_dict'):
                    nb_no_score_cells = len(weight.max_score_dict)
            logger.write('no_score_cells', nb_no_score_cells)

            cells_found_ret = 0
            cells_found_rand = 0
            cells_found_policy = 0
            for cell_key in expl.archive.archive:
                cell_info = expl.archive.archive[cell_key]
                if cell_info.ret_discovered == global_const.EXP_STRAT_NONE:
                    cells_found_ret += 1
                elif cell_info.ret_discovered == global_const.EXP_STRAT_RAND:
                    cells_found_rand += 1
                elif cell_info.ret_discovered == global_const.EXP_STRAT_POLICY:
                    cells_found_policy += 1

            logger.write('cells_found_ret', cells_found_ret)
            logger.write('cells_found_rand', cells_found_rand)
            logger.write('cells_found_policy', cells_found_policy)
            logger.flush()

            traj_manager = expl.archive.cell_trajectory_manager
            new_trajectories = sorted(traj_manager.new_trajectories,
                                      key=lambda t: traj_manager.cell_trajectories[t].frame_finished)
            for traj_id in new_trajectories:
                traj_info = traj_manager.cell_trajectories[traj_id]
                traj_logger.write('it', checkpoint_tracker.n_iters)
                traj_logger.write('frame', traj_info.frame_finished)
                traj_logger.write('exp_strat', traj_info.exp_strat)
                traj_logger.write('exp_new_cells', traj_info.exp_new_cells)
                traj_logger.write('ret_new_cells', traj_info.ret_new_cells)
                traj_logger.write('score', traj_info.score)
                traj_logger.write('total_actions', traj_info.total_actions)
                traj_logger.write('id', traj_info.id)
                traj_logger.flush()

            # Code that should be executed by only the master at a checkpoint generation
            if checkpoint_tracker.should_write_checkpoint():
                local_logger.info(f'Rank: {hvd.rank()} is writing checkpoint: {expl.frames_compute}')
                filename = f'{log_par.base_path}/{expl.frames_compute:0{log_par.n_digits}}'

                # Save pictures
                if len(log_par.save_pictures) > 0:
                    if screenshot_merge == 'disk':
                        for file_name in os.listdir(screen_shot_dir):
                            if file_name.endswith('.png'):
                                room = int(file_name.split('_')[0])
                                if room not in merged_dict:
                                    screen_shot = imageio.imread(f'{screen_shot_dir}/{file_name}')
                                    merged_dict[room] = (True, screen_shot)

                    elif screenshot_merge[0:9] == 'from_dir:':
                        for file_name in os.listdir(screen_shot_dir):
                            if file_name.endswith('.png'):
                                room = int(file_name.split('.')[0])
                                if room not in merged_dict:
                                    screen_shot = imageio.imread(f'{screen_shot_dir}/{file_name}')
                                    merged_dict[room] = (True, screen_shot)

                    render_pictures(log_par, expl, filename, prev_checkpoint, merged_dict, sil_trajectories)

                # Save archive state
                if log_par.save_archive:
                    save_state(expl.get_state(), filename + ARCHIVE_POSTFIX)
                    expl.archive.cell_trajectory_manager.dump(filename + TRAJ_POSTFIX)

                # Save model
                if log_par.save_model:
                    expl.trajectory_gatherer.save_model(filename + MODEL_POSTFIX)

                # Clean up previous checkpoint.
                if prev_checkpoint:
                    for checkpoint_type in clear_checkpoints:
                        if checkpoint_type in CHECKPOINT_ABBREVIATIONS:
                            postfix = CHECKPOINT_ABBREVIATIONS[checkpoint_type]
                        else:
                            postfix = checkpoint_type
                        with contextlib.suppress(FileNotFoundError):
                            local_logger.debug(f'Removing old checkpoint: {prev_checkpoint + postfix}')
                            os.remove(prev_checkpoint + postfix)
                prev_checkpoint = filename

                if track_memory:
                    snapshot = tracemalloc.take_snapshot()
                    display_top(snapshot)

                if PROFILER:
                    local_logger.info(f'ITERATION: {checkpoint_tracker.n_iters}')
                    PROFILER.disable()
                    PROFILER.dump_stats(filename + '.stats')
                    PROFILER.enable()

    local_logger.info(f'Rank {hvd.rank()} finished experiment')
    mpi.get_comm_world().barrier()


def find_checkpoint(base_path):
    for path_to_load in sorted(glob.glob(base_path + '/*'), reverse=True):
        local_logger.debug(f'path_to_load: {path_to_load}')
        for job_lib_file in sorted(glob.glob(path_to_load + '/*' + MODEL_POSTFIX), reverse=True):
            local_logger.debug(f'job_lib_file: {job_lib_file}')
            num = str(os.path.basename(job_lib_file).split('_')[0])
            local_logger.debug(f'Looking for: {os.path.join(path_to_load, num + ARCHIVE_POSTFIX + compress_suffix)}')
            arch_exists = os.path.exists(os.path.join(path_to_load, num + ARCHIVE_POSTFIX + compress_suffix))
            local_logger.debug(f'Looking for: {os.path.join(path_to_load, num + TRAJ_POSTFIX)}')
            traj_exists = os.path.exists(os.path.join(path_to_load, num + TRAJ_POSTFIX))
            if arch_exists and traj_exists:
                return path_to_load, num
    return None, None


def run(kwargs):
    global PROFILER
    mpi.init_mpi()
    np.set_printoptions(threshold=sys.maxsize)
    process_defaults(kwargs)

    log_info = kwargs['log_info']
    log_files = kwargs['log_files']

    if log_info != '':
        log_files = list(filter(None, log_files.split(':')))

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Configure root logger
        root_logger = logging.getLogger()
        numeric_level = getattr(logging, log_info.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % log_info)
        root_logger.setLevel(numeric_level)

        # Create handlers
        for log_file in log_files:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            handler.addFilter(logging.Filter(name=log_file))
            root_logger.addHandler(handler)

    base_path = kwargs['base_path']
    fail_on_duplicate = kwargs['fail_on_duplicate']
    continue_run = bool(kwargs['continue'])

    kwargs['cell_trajectories_file'] = ''
    if continue_run:
        assert kwargs['expl_state'] is None
        assert kwargs['load_path'] == ''
        assert kwargs['cell_trajectories_file'] == ''
        if os.path.exists(base_path):
            path_to_load, num = find_checkpoint(base_path)
            if path_to_load is not None and num is not None:
                kwargs['expl_state'] = os.path.join(path_to_load, num + ARCHIVE_POSTFIX + compress_suffix)
                kwargs['load_path'] = os.path.join(path_to_load, num + MODEL_POSTFIX)
                kwargs['cell_trajectories_file'] = os.path.join(path_to_load, num + TRAJ_POSTFIX)
                local_logger.info(f'Successfully loading from checkpoint: {kwargs["expl_state"]} {kwargs["load_path"]} '
                                  f'{kwargs["cell_trajectories_file"]}')
        if kwargs['expl_state'] is None or kwargs['load_path'] == '':
            kwargs['expl_state'] = None
            kwargs['load_path'] = ''
            kwargs['cell_trajectories_file'] = ''
            local_logger.warning(f'No checkpoint found in: {kwargs["base_path"]} starting new run.')

    if os.path.exists(base_path) and fail_on_duplicate:
        raise Exception('Experiment: ' + base_path + ' already exists!')

    # We need to setup the MPI environment before performing any data processing
    nb_cpu = 2
    session, master_seed = hrv_and_tf_init(nb_cpu, kwargs['nb_envs'],  kwargs['seed'])
    with session.as_default():
        worker_seed_start = master_seed + 1
        kwargs['seed'] = worker_seed_start

        # Process load path
        kwargs['model_path'] = kwargs['load_path']

        # Process profile
        profile = kwargs['profile']

        # Only one process should write information about our experiment
        if hvd.rank() == 0:
            cur_id = 0
            if os.path.exists(base_path):
                current = glob.glob(base_path + '/*')
                for c in current:
                    try:
                        idx, _ = c.split('/')[-1].split('_')
                        idx = int(idx)
                        if idx >= cur_id:
                            cur_id = idx + 1
                    except ValueError:
                        pass
                    except IndexError:
                        pass
            base_path = f'{base_path}/{cur_id:04d}_{uuid.uuid4().hex}/'
            os.makedirs(base_path, exist_ok=True)

            info = copy.copy(kwargs)
            info['version'] = VERSION
            code_hash = get_code_hash()
            info['code_hash'] = code_hash
            local_logger.info(f'Code hash: {code_hash}')
            json.dump(info, open(base_path + '/kwargs.json', 'w'), indent=4)
            kwargs['base_path'] = base_path
            local_logger.info(f'Experiment running in {base_path}')
        else:
            base_path = None
        base_path = mpi.COMM_WORLD.bcast(base_path, root=0)
        kwargs['base_path'] = base_path

        if profile:
            PROFILER = cProfile.Profile()
            PROFILER.enable()

        temp_dir = tempfile.mkdtemp(prefix='tmp_goexplore_')
        kwargs['temp_dir'] = temp_dir
        try:
            _run(**kwargs)
            if PROFILER is not None:
                PROFILER.disable()
        finally:
            try:
                # delete directory
                shutil.rmtree(temp_dir)
            except OSError as exc:
                # ENOENT - no such file or directory
                if exc.errno != errno.ENOENT:
                    raise
            if PROFILER is not None:
                PROFILER.print_stats()


def main():
    args = parse_arguments()
    run(vars(args))


if __name__ == '__main__':
    main()
