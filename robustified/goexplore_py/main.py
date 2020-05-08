
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from sys import platform
from goexplore_py.randselectors import *
from goexplore_py.goexplore import *
import goexplore_py.goexplore
import goexplore_py.montezuma_env as montezuma_env
import goexplore_py.pitfall_env as pitfall_env
import goexplore_py.generic_atari_env as generic_atari_env
import goexplore_py.generic_goal_conditioned_env as generic_goal_conditioned_env
import goexplore_py.complex_fetch_env as complex_fetch_env
import cProfile
import gzip

import tracemalloc, linecache


VERSION = 1

THRESH_TRUE = 20_000_000_000
THRESH_COMPUTE = 1_000_000
MAX_FRAMES = None
MAX_FRAMES_COMPUTE = None
MAX_ITERATIONS = None
MAX_TIME = 12 * 60 * 60
MAX_CELLS = None
MAX_SCORE = None

PROFILER = None

def track_performance(perf_array, sleep_time, exp_coef):
    import time
    import psutil
    while True:
        vmem = psutil.virtual_memory().used
        cpu = psutil.cpu_percent()

        perf_array[MemInfo.E_VIRT_USE_MEAN] = exp_coef * vmem + (1 - exp_coef) * perf_array[MemInfo.E_VIRT_USE_MEAN]
        perf_array[MemInfo.E_CPU_MEAN] = exp_coef * cpu + (1 - exp_coef) * perf_array[MemInfo.E_CPU_MEAN]

        perf_array[MemInfo.E_VIRT_USE_CUR] = vmem
        perf_array[MemInfo.E_CPU_CUR] = cpu

        perf_array[MemInfo.E_VIRT_USE_MAX] = max(vmem, perf_array[MemInfo.E_VIRT_USE_MAX])
        perf_array[MemInfo.E_CPU_MAX] = max(cpu, perf_array[MemInfo.E_CPU_MAX])

        time.sleep(sleep_time)

def _run(
        base_path,
        args
        ):

    goexplore_py.goexplore.perf_array = multiprocessing.Array('d', [0.0] * MemInfo.ARRAY_SIZE)
    perf_process = multiprocessing.Process(target=track_performance, args=(goexplore_py.goexplore.perf_array, 1, 0.1))
    perf_process.start()

    if 'robot' in args.game:
        if args.explorer_type == 'drift':
            explorer = RandomDriftExplorerRobot(args.repeat_action)
        else:
            explorer = RepeatedRandomExplorerRobot(args.repeat_action)
    elif 'fetch' in args.game:
        if args.explorer_type == 'drift':
            explorer = RandomDriftExplorerFetch(args.repeat_action)
        else:
            explorer = RepeatedRandomExplorerFetch(args.repeat_action)
    elif args.explorer_type == 'repeated':
        explorer = RepeatedRandomExplorer(args.repeat_action)
    else:
        explorer = RandomExplorer()

    if args.use_real_pos:
        args.target_shape = None
        args.max_pix_value = None

    if args.dynamic_state:
        args.target_shape = (-1, -1)
        args.max_pix_value = -1

    IMPORTANT_ATTRS = []
    if args.game == 'montezuma':
        game_class = MyMontezuma
        game_class.TARGET_SHAPE = args.target_shape
        game_class.MAX_PIX_VALUE = args.max_pix_value
        game_args = dict(
        score_objects=args.use_objects, x_repeat=args.x_repeat,
        objects_from_pixels=args.objects_from_pixels,
        objects_remember_rooms=args.remember_rooms,
        only_keys=args.only_keys
        )
        IMPORTANT_ATTRS = ['level', 'score']
        grid_resolution = (
            GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
            GridDimension('x', args.resolution), GridDimension('y', args.resolution)
        )
    elif args.game == 'pitfall':
        game_class = pitfall_env.MyPitfall
        game_class.TARGET_SHAPE = args.target_shape
        game_class.MAX_PIX_VALUE = args.max_pix_value
        game_args = dict(x_repeat=args.x_repeat, treasure_type=args.pitfall_treasure_type)
        IMPORTANT_ATTRS = ['score']
        grid_resolution = (
            GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
            GridDimension('x', args.resolution), GridDimension('y', args.resolution)
        )
    elif 'generic' in args.game:
        if len(args.game.split('_')) > 2:
            resize_shape = args.game.split('_')[2]
            x, y, p = resize_shape.split('x')
            target_shape = (int(x), int(y))
            max_pix_value = int(p)

        game_class = generic_atari_env.MyAtari
        game_class.TARGET_SHAPE = args.target_shape
        game_class.MAX_PIX_VALUE = args.max_pix_value
        game_args = dict(name=args.game.split('_')[1])
        grid_resolution = (
            GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
            GridDimension('x', args.resolution), GridDimension('y', args.resolution)
        )
    elif 'robot' in args.game:
        game_class = generic_goal_conditioned_env.MyRobot
        game_args = dict(env_name=args.game.split('_')[1], interval_size=args.interval_size, seed_low=args.seed, seed_high=args.seed)
        grid_resolution = (
            GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
            GridDimension('x', args.resolution), GridDimension('y', args.resolution)
        )
    elif 'fetch' in args.game:
        game_class = complex_fetch_env.MyComplexFetchEnv

        model_file = f'teleOp_{args.fetch_type}.xml'

        if args.target_location == 'None':
            args.target_location = None

        game_args = dict(
            nsubsteps=args.nsubsteps,
            min_grip_score=args.min_grip_score,
            max_grip_score=args.max_grip_score,
            model_file=model_file,
            target_single_shelf=args.target_single_shelf,
            combine_table_shelf_box=args.combine_table_shelf_box,
            ordered_grip=args.fetch_ordered,
            target_location=args.target_location,
            timestep=args.timestep,
            force_closed_doors=args.fetch_force_closed_doors
        )

        door1_dists_ignore = GridDimension('door1_dists', 1000, 500)
        door1_dists_use = GridDimension('door1_dists', args.door_resolution, args.door_offset)
        if args.target_location == '0010' or (args.target_location is None and not args.target_single_shelf):  # Lower door
            door1_dists = door1_dists_use
        elif args.target_single_shelf:
            door1_dists = FetchConditionalObject(
                '0010', door1_dists_use, door1_dists_ignore
            )
        else:
            door1_dists = door1_dists_ignore


        door_dists_ignore = GridDimension('door_dists', 1000, 500)
        door_dists_use = GridDimension('door_dists', args.door_resolution, args.door_offset)
        if args.target_location == '0001' or (args.target_location is None and not args.target_single_shelf):  # Lower door
            door_dists = door_dists_use
        elif args.target_single_shelf:
            door_dists = FetchConditionalObject(
                '0001', door_dists_use, door_dists_ignore
            )
        else:
            door_dists = door_dists_ignore

        target_grid = GridDimension('object_pos', 1, sort=args.conflate_objects)
        if args.target_location is not None:
            target_grid = GridEquality('object_pos', args.target_location, sort=args.conflate_objects)

        grid_resolution = (
            door_dists, door1_dists,
            GridDimension('gripped_info', 1), GridDimension('gripped_pos', 1000, 500),
            target_grid, GridDimension('gripper_pos', args.gripper_pos_resolution)
        )
        if args.fetch_single_cell:
            grid_resolution = (
                SingleCell('door_dists', (0,)), SingleCell('door1_dists', (0,)), SingleCell('gripped_info', None),
                SingleCell('gripped_pos', None), SingleCell('object_pos', None), SingleCell('gripper_pos', None)
            )
        IMPORTANT_ATTRS = ['door_dists', 'door1_dists', 'gripped_info', 'object_pos']
    else:
        raise NotImplementedError("Unknown game: " + args.game)

    selector = WeightedSelector(game_class,
                                seen=Weight(args.seen_weight, args.seen_power),
                                chosen=Weight(args.chosen_weight, args.chosen_power),
                                action=Weight(args.action_weight, args.action_power),
                                room_cells=Weight(0.0),
                                dir_weights=DirWeights(args.horiz_weight, args.vert_weight, args.low_score_weight, args.high_score_weight),
                                chosen_since_new_weight=Weight(args.chosen_since_new_weight, args.chosen_since_new_power),
                                low_level_weight=args.low_level_weight,
                                grip_weight=args.grip_weight,
                                door_weight=args.door_weight
    )

    pool_cls = multiprocessing.get_context(args.start_method).Pool
    if args.pool_class == 'torch':
        pool_cls = torch.multiprocessing.Pool
    elif args.pool_class == 'loky':
        pool_cls = LPool
    elif args.pool_class == 'sync':
        pool_cls = SyncPool

    pool_cls = seed_pool_wrapper(pool_cls)

    expl = Explore(
        explorer,
        selector,
        (game_class, game_args),
        grid_resolution,
        pool_class=pool_cls,
        args=args,
        important_attrs=IMPORTANT_ATTRS
    )

    if args.seed_path is not None:
        # First we check that the arguments match
        orig_kwargs = json.load(open(args.seed_path + '/kwargs.json'))
        new_kwargs = json.load(open(base_path + '/kwargs.json'))
        for k in set(list(orig_kwargs.keys()) + list(new_kwargs.keys())):
            # Note: we include seed_path because of course the original had a different seed_path,
            # we include base_path for the same reason. code_hash is riskier but makes the assumption
            # that the original may have crashed due to a non-performance affecting bug that was fixed
            # in the new version. Seed is included because it technically doesn't make a difference
            # if the new version is started with the same or a different seed.
            if k in ('code_hash', 'seed', 'seed_path', 'base_path'):
                continue
            assert orig_kwargs.get(k) == new_kwargs.get(k)

        # Now we find the SECOND TO LAST grid, which will be our starting point,
        # as well as the maximum number of compute and game steps
        # Note, very importantly, that we are resuming from the SECOND TO LAST: the
        # reason is that we can't know for sure that the last one isn't buggy (because
        # the original run might have crashed while creating it)
        grid_idx = 0
        while os.path.exists(f'{args.seed_path}/__grid_{grid_idx + 2}.pickle.bz2'):
            grid_idx += 1
        last_recompute_exp = sorted(glob.glob(f'{args.seed_path}/*pre_recompute_experience.bz2'))[grid_idx]
        expl.frames_true, expl.frames_compute = [int(e) for e in last_recompute_exp.split('/')[-1].split('_')[:2]]

        # Now we create links to all the files that precede the grid change
        os.remove(f'{args.base_path}/__grid_0.pickle.bz2')
        for e in tqdm(glob.glob(f'{args.seed_path}/*.bz2'), desc='symlinks'):
            filename = e.split('/')[-1]
            if 'thisisfake' in filename:
                continue
            if filename.startswith('__grid'):
                idx = int(filename.split('_')[3].split('.')[0])
                if idx < grid_idx:
                    os.system(f'ln -s -r "{e}" "{base_path}/{filename}"')
            else:
                game_frames, compute_frames = [int(e.split('.')[0]) for e in filename.split('_')[:2]]
                if game_frames < expl.frames_true:
                    os.system(f'ln -s -r "{e}" "{base_path}/{filename}"')


        # Now we set all the relevent attributes
        expl.grid = pickle.load(bz2.open(f'{args.seed_path}/__grid_{grid_idx}.pickle.bz2', 'rb'))
        expl.former_grids.cur_length = grid_idx
        exp_data = pickle.load(bz2.open(last_recompute_exp))
        # Note: expl.cycles cannot be set, this is probably ok because we don't generally use this value
        # Note: expl.seen_level_1 could be set but not worth it because this is not intended for domain knowledge
        # Note: expl.dynamic_state_frame_sets is unused
        # Note: expl.last_recompute_dynamic_state probably can be left in its initial state. The only thing we want
        #       is for the dynamic state to be recomputed immediately, which should happen.
        expl.max_score = max([e.score for e in expl.grid.values()])
        # Note: expl.prev_len_grid is unused
        # Note: expl.real_grid doesn't really matter except for domain knowledge, so ignoring here
        # Note: expl.pos_cache can be ignored
        # Note: expl.last_added_cell is set but not used, ignoring
        # Note: expl.real_cell can be ignored
        (
            expl.experience_prev_ids, expl.experience_actions, expl.experience_rewards,
            expl.experience_cells, expl.experience_scores, expl.experience_lens
        ) = exp_data
        expl.dynamic_state_split_rules = [
            e[0] for e in expl.experience_cells
            if isinstance(e, tuple)
        ][-1]

        expl.selector.clear_all_cache()
        for k, v in expl.grid.items():
            expl.selector.cell_update(k, v)

        with tqdm(desc='Sample Frames', total=args.max_recent_frames) as t_sample:
            while len(expl.random_recent_frames) < args.max_recent_frames:
                old = len(expl.random_recent_frames)
                expl.sample_only_cycle()
                t_sample.update(len(expl.random_recent_frames) - old)

    with tqdm(desc='Time (seconds)', smoothing=0, total=MAX_TIME) as t_time, \
            tqdm(desc='Iterations', total=MAX_ITERATIONS) as t_iter, \
            tqdm(desc='Compute steps', total=MAX_FRAMES_COMPUTE) as t_compute, \
            tqdm(desc='Game step', total=MAX_FRAMES) as t, \
            tqdm(desc='Max score', total=MAX_SCORE) as t_score, \
            tqdm(desc='Done score', total=MAX_SCORE) as t_done_score, \
            tqdm(desc='Cells', total=MAX_CELLS) as t_cells:
        t_compute.update(expl.frames_compute)
        t.update(expl.frames_true)
        start_time = time.time()
        last_time = np.round(start_time)
        seen_level_1 = False
        n_iters = 0
        prev_checkpoint = None

        def should_continue():
            if MAX_TIME is not None and time.time() - start_time >= MAX_TIME:
                return False
            if MAX_FRAMES is not None and expl.frames_true >= MAX_FRAMES:
                return False
            if MAX_FRAMES_COMPUTE is not None and expl.frames_compute >= MAX_FRAMES_COMPUTE:
                return False
            if MAX_ITERATIONS is not None and n_iters >= MAX_ITERATIONS:
                return False
            if MAX_CELLS is not None and len(expl.grid) >= MAX_CELLS:
                return False
            if MAX_SCORE is not None and expl.max_score >= MAX_SCORE:
                return False
            return True

        while should_continue():
            # Run one iteration
            old = expl.frames_true
            old_compute = expl.frames_compute
            old_len_grid = len(expl.grid)
            old_max_score = expl.max_score

            expl.run_cycle()

            t.update(expl.frames_true - old)
            t_score.update(expl.max_score - old_max_score)
            t_done_score.n = expl.grid[DONE].score
            t_done_score.refresh()
            t_compute.update(expl.frames_compute - old_compute)
            t_iter.update(1)
            # Note: due to the archive compression that can happen with dynamic cell representation,
            # we need to do this so that tqdm doesn't complain about negative updates.
            t_cells.n = len(expl.grid)
            t_cells.refresh()

            cur_time = np.round(time.time())
            t_time.update(int(cur_time - last_time))
            last_time = cur_time
            n_iters += 1

            # In some circumstances (see comments), save a checkpoint and some pictures
            if ((not seen_level_1 and expl.seen_level_1) or  # We have solved level 1
                    old == 0 or  # It is the first iteration
                    old // THRESH_TRUE != expl.frames_true // THRESH_TRUE or  # We just passed the THRESH_TRUE threshold
                    old_compute // THRESH_COMPUTE != expl.frames_compute // THRESH_COMPUTE or  # We just passed the THRESH_COMPUTE threshold
                    not should_continue()):  # This is the last iteration

                seen_level_1 = getattr(expl, 'seen_level_1', False)
                expl.save_checkpoint()

                if PROFILER:
                    print("ITERATION:", n_iters)
                    PROFILER.disable()
                    PROFILER.dump_stats(filename + '.stats')
                    PROFILER.enable()

    perf_process.terminate()

def display_top(snapshot, key_type='traceback', limit=20):
    top_stats = snapshot.statistics(key_type)

    tqdm.write("Top %s lines" % limit)
    for index, stat in reversed(list(enumerate(top_stats[:limit], 1))):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        # filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        filename = frame.filename
        tqdm.write("#%s: %s:%s: %.1f MB"
              % (index, filename, frame.lineno, stat.size / (1024*1024)))
        line = linecache.getline(frame.filename, frame.lineno).strip()

        for frame in stat.traceback.format():
            tqdm.write(str(frame))

        if line:
            tqdm.write('    %s' % line)

        tqdm.write('\n')

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        tqdm.write("%s other: %.1f MB" % (len(other), size / (1024*1024)))
    total = sum(stat.size for stat in top_stats)
    tqdm.write('\n')
    tqdm.write("Total allocated size: %.1f MB" % (total / (1024*1024)))
    tqdm.write('\n')

class Tee(object):
    def __init__(self, name, output):
        self.file = open(name, 'w')
        self.stdout = getattr(sys, output)
        self.output = output
        setattr(sys, self.output, self)
    def __del__(self):
        setattr(sys, self.output, self.stdout)
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
    
def run(base_path, args):
    cur_id = 0
    if os.path.exists(base_path):
        current = glob.glob(base_path + '/*')
        for c in current:
            try:
                idx, _ = c.split('/')[-1].split('_')
                idx = int(idx)
                if idx >= cur_id:
                    cur_id = idx + 1

                if args.seed is not None:
                    if os.path.exists(c + '/has_died'):
                        continue
                    other_kwargs = json.load(open(c + '/kwargs.json'))
                    is_same_kwargs = True
                    for k, v in vars(args).items():
                        def my_neq(a, b):
                            if isinstance(a, tuple):
                                a = list(a)
                            if isinstance(b, tuple):
                                b = list(b)
                            return a != b
                        if k != 'base_path' and my_neq(other_kwargs[k], v):
                            is_same_kwargs = False
                            break
                    if is_same_kwargs:
                        try:
                            last_exp = sorted([e for e in glob.glob(c + '/*_experience.bz2') if 'thisisfake' not in e])[-1]
                        except IndexError:
                            continue
                        mod_time = os.path.getmtime(last_exp)
                        if time.time() - mod_time < 3600:
                            print('Another run is already running at', c, 'exiting.')
                            return
                        compute = int(last_exp.split('/')[-1].split('_')[1])
                        if compute >= args.max_compute_steps:
                            print('A completed equivalent run already exists at', c, 'exiting.')
                            return
            except Exception:
                pass

    base_path = f'{base_path}/{cur_id:04d}_{uuid.uuid4().hex}/'
    args.base_path = base_path
    os.makedirs(base_path, exist_ok=True)
    open(f'{base_path}/thisisfake_{args.max_compute_steps}_experience.bz2', 'w')
    info = copy.copy(vars(args))
    info['version'] = VERSION
    info['code_hash'] = get_code_hash()
    print('Code hash:', info['code_hash'])
    del info['base_path']
    json.dump(info, open(base_path + '/kwargs.json', 'w'), sort_keys=True, indent=2)

    code_path = base_path + '/code'
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(cur_dir, code_path, ignore=shutil.ignore_patterns('*.png', '*.stl', '*.JPG', '__pycache__', 'LICENSE*', 'README*'))

    teeout = Tee(args.base_path + '/log.out', 'stdout')
    teeerr = Tee(args.base_path + '/log.err', 'stderr')
    
    print('Experiment running in', base_path)

    try:
        _run(base_path, args)
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()
        import signal
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGTERM)
        open(base_path + 'has_died', 'w')
        os._exit(1)

if __name__ == '__main__':
    if os.path.exists('/home/udocker/deeplearning_goexplore_adrienle/'):
        os.makedirs('/mnt/phx4', exist_ok=True)
        os.system('/opt/michelangelo/mount.nfs -n -o nolock qstore1-phx4:/share /mnt/phx4')

    if platform == "darwin":
        # Circumvents the following issue on Mac OS:
        # https://github.com/opencv/opencv/issues/5150
        cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()

    current_group = parser

    def boolarg(arg, *args, default=False, help='', neg=None, dest=None):
        def extract_name(a):
            dashes = ''
            while a[0] == '-':
                dashes += '-'
                a = a[1:]
            return dashes, a

        if dest is None:
            _, dest = extract_name(arg)

        group = current_group.add_mutually_exclusive_group()
        group.add_argument(arg, *args, dest=dest, action='store_true', help=help + (' (DEFAULT)' if default else ''), default=default)
        not_args = []
        for a in [arg] + list(args):
            dashes, name = extract_name(a)
            not_args.append(f'{dashes}no_{name}')
        if isinstance(neg, str):
            not_args[0] = neg
        if isinstance(neg, list):
            not_args = neg
        group.add_argument(*not_args, dest=dest, action='store_false', help=f'Opposite of {arg}' + (' (DEFAULT)' if not default else ''), default=default)

    def add_argument(*args, **kwargs):
        if 'help' in kwargs and kwargs.get('default') is not None:
            kwargs['help'] += f' (default: {kwargs.get("default")})'

        current_group.add_argument(*args, **kwargs)

    current_group = parser.add_argument_group('General Go-Explore')
    add_argument('--game', '-g', type=str, default='montezuma', help='Determines the game to which apply goexplore.')
    add_argument('--repeat_action', '--ra', type=float, default=20, help='The average number of times that actions will be repeated in the exploration phase.')
    add_argument('--explore_steps', type=int, default=100, help='Maximum number of steps in the explore phase.')
    add_argument('--ignore_death', type=int, default=0, help='Number of steps immediately before death to ignore.')
    boolarg('--optimize_score', default=True, help='Optimize for score (only speed). Will use fewer "game frames" and come up with faster trajectories with lower scores. If not combined with --remember_rooms and --objects_from_ram is not enabled, things should run much slower.')
    add_argument('--prob_override', type=float, default=0.0, help='Probability that the newly found cells will randomly replace the current cell.')
    add_argument('--batch_size', type=int, default=100, help='Number of worker threads to spawn')
    boolarg('--reset_cell_on_update', '--rcou', help='Reset the times-chosen and times-chosen-since when a cell is updated.')
    add_argument('--explorer_type', type=str, default='repeated', help='The type of explorer. repeated, drift or random.')
    add_argument('--seed', type=int, default=None, help='The random seed.')

    current_group = parser.add_argument_group('Checkpointing')
    add_argument('--base_path', '-p', type=str, default='./results/', help='Folder in which to store results')
    add_argument('--path_postfix', '--pf', type=str, default='', help='String appended to the base path.')
    add_argument('--seed_path', type=str, default=None, help='Path from which to load existing results.')
    add_argument('--checkpoint_game', type=int, default=20_000_000_000_000_000_000, help='Save a checkpoint every this many GAME frames (note: recommmended to ignore, since this grows very fast at the end).')
    add_argument('--checkpoint_compute', type=int, default=1_000_000, help='Save a checkpoint every this many COMPUTE frames.')
    boolarg('--pictures', dest='save_pictures', help='Save pictures of the pyramid every checkpoint (uses more space).')
    boolarg('--prob_pictures', '--pp', dest='save_prob_pictures', help='Save pictures of showing probabilities.')
    boolarg('--item_pictures', '--ip', dest='save_item_pictures', help='Save pictures of showing items collected.')
    boolarg('--clear_old_checkpoints', neg='--keep_checkpoints', default=True,
            help='Clear large format checkpoints. Checkpoints aren\'t necessary for view folder to work. They use a lot of space.')
    boolarg('--keep_prob_pictures', '--kpp', help='Keep old pictures showing probabilities.')
    boolarg('--keep_item_pictures', '--kip', help='Keep old pictures showing items collected.')
    boolarg('--warn_delete', default=True, help='Warn before deleting the existing directory, if any.')
    boolarg('--save_cells', default=False, help='Save exact cells produced by Go-Explore instead of just hints as to whether they are done or not.')

    current_group = parser.add_argument_group('Runtime')
    add_argument('--max_game_steps', type=int, default=None, help='Maximum number of GAME frames.')
    add_argument('--max_compute_steps', '--mcs', type=int, default=None, help='Maximum number of COMPUTE frames.')
    add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations.')
    add_argument('--max_hours', '--mh', type=float, default=12, help='Maximum number of hours to run this for.')
    add_argument('--max_cells', type=int, default=None, help='The maximum number of cells before stopping.')
    add_argument('--max_score', type=float, default=None, help='Stop when this score (or more) has been reached in the archive.')

    current_group = parser.add_argument_group('General Selection Probability')
    add_argument('--seen_weight', '--sw', type=float, default=0.0, help='The weight of the "seen" attribute in cell selection.')
    add_argument('--seen_power', '--sp', type=float, default=0.5, help='The power of the "seen" attribute in cell selection.')
    add_argument('--chosen_weight', '--cw', type=float, default=0.0, help='The weight of the "chosen" attribute in cell selection.')
    add_argument('--chosen_power', '--cp', type=float, default=0.5, help='The power of the "chosen" attribute in cell selection.')
    add_argument('--chosen_since_new_weight', '--csnw', type=float, default=0.0, help='The weight of the "chosen since new" attribute in cell selection.')
    add_argument('--chosen_since_new_power', '--csnp', type=float, default=0.5, help='The power of the "chosen since new" attribute in cell selection.')
    add_argument('--action_weight', '--aw', type=float, default=0.0, help='The weight of the "action" attribute in cell selection.')
    add_argument('--action_power', '--ap', type=float, default=0.5, help='The power of the "action" attribute in cell selection.')

    current_group = parser.add_argument_group('Atari Selection Probability')
    add_argument('--horiz_weight', '--hw', type=float, default=0.0, help='Weight of not having one of the two possible horizontal neighbors.')
    add_argument('--vert_weight', '--vw', type=float, default=0.0, help='Weight of not having one of the two possible vertical neighbors.')
    add_argument('--low_score_weight', type=float, default=0.0, help='Weight of not having a neighbor with a lower score/object number.')
    add_argument('--high_score_weight', type=float, default=0.0, help='Weight of not having a neighbor with a higher score/object number.')
    add_argument('--low_level_weight', type=float, default=1.0, help='Weight of cells in levels lower than the current max. If this is non-zero, lower levels will keep getting optimized, potentially leading to better solutions overall. Setting this to greater than 1 is possible but nonsensical since it means putting a larger weight on low levels than higher levels.')

    current_group = parser.add_argument_group('Atari Domain Knowledge')
    add_argument('--resolution', '--res', type=float, default=16, help='Length of the side of a grid cell.')
    boolarg('--use_objects', neg='--use_scores', default=True, help='Use objects in the cell description. Otherwise scores will be used.')
    add_argument('--x_repeat', type=int, default=2, help='How much to duplicate pixels along the x direction. 2 is closer to how the games were meant to be played, but 1 is the original emulator resolution. NOTE: affects the behavior of GoExplore.')
    boolarg('--objects_from_pixels', neg='--objects_from_ram', default=True, help='Get the objects from pixels instead of RAM.')
    boolarg('--only_keys', neg='--all_objects', default=True, help='Use only the keys instead of all objects.')
    boolarg('--remember_rooms', help='Remember which room the objects picked up came from. Makes it easier to solve the game (because the state encodes the location of the remaining keys anymore), but takes more time/memory space, which in practice makes it worse quite often. Using this is better if running with --no_optimize_score')
    add_argument('--pitfall_treasure_type', type=str, default='none', help='How to include treasures in the cell description of Pitfall: none (don\'t include treasures), count (include treasure count), score (include sum of positive rewards) or location (include the specific location the treasures were found).')

    current_group = parser.add_argument_group('Atari No Domain Knowledge')
    boolarg('--use_real_pos', neg=['--state_is_pixels', '--pix'], default=True, help='If this is on, the state will be resized pixels, not human prior.')
    add_argument('--resize_x', '--rx', type=int, default=11, help='What to resize the pixels to in the x direction for use as a state.')
    add_argument('--resize_y', '--ry', type=int, default=8, help='What to resize the pixels to in the y direction for use as a state.')
    add_argument('--max_pix_value', '--mpv', type=int, default=8, help='The range of pixel values when resizing will be rescaled to from 0 to this value. Lower means fewer possible states in states_is_pixels.')
    add_argument('--resize_shape', type=str, default=None, help='Shortcut for passing --resize_x (0), --resize_y (1) and --max_pix_value (2) all at the same time: 0x1x2')

    boolarg('--dynamic_state', help='Dynamic downscaling of states. Ignores --resize_x, --resize_y, --max_pix_value and --resize_shape.')

    add_argument('--first_compute_dynamic_state', type=int, default=100_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')
    add_argument('--first_compute_archive_size', type=int, default=10_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')
    add_argument('--recompute_dynamic_state_every', type=int, default=5_000_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')
    add_argument('--max_archive_size', type=int, default=1_000_000_000, help='Number of steps before recomputing the dynamic state representation (ignored if negative).')

    add_argument('--cell_split_factor', type=float, default=0.03, help='The factor by which we try to split frames when recomputing the representation. 1 -> each frame is its own cell. 0 -> all frames are in the same cell.')
    add_argument('--split_iterations', type=int, default=100, help='The number of iterations when recomputing the representation. A higher number means a more accurate (but less stochastic) results, and a lower number means a more stochastic and less accurate result. Note that stochasticity can be a good thing here as it makes it harder to get stuck.')
    add_argument('--max_recent_frames', type=int, default=5_000, help='The number of recent frames to use in recomputing the representation. A higher number means slower recomputation but more accuracy, a lower number is faster and more stochastic.')
    add_argument('--recent_frame_add_prob', type=float, default=0.1, help='The probability for a frame to be added to the list of recent frames.')

    current_group = parser.add_argument_group('OpenAI Robotics')
    add_argument('--interval_size', type=float, default=0.1, help='The interval size for robotics envs.')

    current_group = parser.add_argument_group('Fetch Robotics')
    add_argument('--fetch_type', type=str, default='boxes', help='The type of fetch environment (boxes, cubes, objects...)')
    add_argument('--nsubsteps', type=int, default=20, help='The number of substeps in mujoco between each action (each substep takes 0.002 seconds).')
    add_argument('--target_location', type=str, default=None, help='The target location for fetch envs.')
    add_argument('--min_grip_score', type=int, default=0, help='The minimum grip score (inclusive) for a fetch grip to be included in the archive.\n0: at least 1 finger touching, 1: 2 fingers touching, 3: 2 fingers touching AND not touching the table (gripping and lifting).')
    add_argument('--max_grip_score', type=int, default=3, help='The maximum grip score (inclusive). All grips with higher scores will be given this score instead.')
    add_argument('--minmax_grip_score', type=str, default=None, help='Shortcut to set both the min and max grip score. The first digit is the min and second is the max.')
    add_argument('--door_resolution', type=float, default=0.2, help='Number by which to divide the door distance.')
    old_group = current_group
    # current_group = current_group.add_mutually_exclusive_group()
    add_argument('--timestep', type=float, default=0.002, help='The size of a mujoco timestep.')
    add_argument('--total_timestep', type=float, default=None, help='The total timestep length (if included, timestep is ignored from the command line and instead set to total_timestep / nsubsteps). A reasonable value is 0.08')
    current_group = old_group
    add_argument('--door_offset', type=float, default=None, help='Number to add to the door distance before dividing.')
    add_argument('--gripper_pos_resolution', type=float, default=0.5, help='Number by which to divide the gripper position.')
    add_argument('--door_weight', type=float, default=1.0, help='Weight of different door positions.')
    add_argument('--grip_weight', type=float, default=1.0, help='Weight of different grip positions.')
    boolarg('--fetch_uniform', help='Select uniformly for fetch. Shurtcut for --door_weight=0 --grip_weight=0 --low_level_weight=1.')
    boolarg('--conflate_objects', help='Conflate objects when getting their positions. With this, there is no difference between object 1 being in shelf 0001 and object 2 being in shelf 0001.')
    boolarg('--target_single_shelf', help='As soon as a shelf is reached, only target that one shelf going forward.')
    boolarg('--fetch_ordered', help='Whether to put objects in the shelves in a specific order or not.')
    boolarg('--combine_table_shelf_box', help='Combine the table and shelf box for determining death by being outside the table-shelf box.')
    boolarg('--fetch_force_closed_doors', help='Only give rewards for fetch if the doors are closed.')
    boolarg('--fetch_single_cell', help='Only one cell when doing fetch.', default=False)

    current_group = parser.add_argument_group('Performance')
    add_argument('--n_cpus', type=int, default=None, help='Number of worker threads to spawn')
    add_argument('--pool_class', type=str, default='loky', help='The multiprocessing pool class (py or torch or loky).')
    add_argument('--start_method', type=str, default='fork', help='The process start method.')
    boolarg('--reset_pool', help='The pool should be reset every 100 iterations.')
    boolarg('--profile', help='Whether or not to enable a profiler.')

    args = parser.parse_args()
    if args.door_offset is None:
        args.door_offset = args.door_resolution - 0.005
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed + 1)

    if args.total_timestep is not None:
        args.timestep = args.total_timestep / args.nsubsteps
    del args.total_timestep

    if args.fetch_uniform:
        args.door_weight = 0
        args.grip_weight = 0
        args.low_level_weight = 1
    del args.fetch_uniform

    if args.minmax_grip_score:
        args.min_grip_score = int(args.minmax_grip_score[0])
        args.max_grip_score = int(args.minmax_grip_score[1])
    del args.minmax_grip_score

    if args.resize_shape:
        x, y, p = args.resize_shape.split('x')
        args.resize_x = int(x)
        args.resize_y = int(y)
        args.max_pix_value = int(p)

    args.target_shape = (args.resize_x, args.resize_y)
    del args.resize_shape
    del args.resize_x
    del args.resize_y

    if args.start_method == 'fork' and args.pool_class == 'torch':
        raise Exception('Fork start method not supported by torch.multiprocessing.')

    THRESH_TRUE = args.checkpoint_game
    THRESH_COMPUTE = args.checkpoint_compute
    MAX_FRAMES = args.max_game_steps
    MAX_FRAMES_COMPUTE = args.max_compute_steps
    MAX_TIME = args.max_hours * 3600
    MAX_ITERATIONS = args.max_iterations
    MAX_CELLS = args.max_cells
    MAX_SCORE = args.max_score

    assert args.pitfall_treasure_type in ('none', 'count', 'score', 'location')

    if args.profile:
        PROFILER = cProfile.Profile()
        PROFILER.enable()
    try:
        run(args.base_path, args)
        if PROFILER is not None:
            PROFILER.disable()
    finally:
        if PROFILER is not None:
            PROFILER.print_stats()
