# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
import gym
import argparse
import cv2
import gzip
import tensorflow as tf
import horovod.tensorflow as hvd
from sys import platform
from typing import Any
import os
import random
import numpy as np
import logging

import atari_reset.atari_reset.ppo as ppo
import atari_reset.atari_reset.wrappers as wrappers

import goexplore_py.ge_wrappers as ge_wrappers
import goexplore_py.ge_policies as ge_policies
import goexplore_py.ge_models as ge_models
import goexplore_py.ge_runners as ge_runners
import goexplore_py.trajectory_trackers as trajectory_trackers
import goexplore_py.randselectors as randselectors
import goexplore_py.archives as archives
import goexplore_py.goal_representations as goal_rep
import goexplore_py.trajectory_gatherers as trajectory_gatherers
import goexplore_py.montezuma_env as montezuma_env
import goexplore_py.pitfall_env as pitfall_env
import goexplore_py.generic_atari_env as generic_atari_env
import goexplore_py.generic_goal_conditioned_env as generic_goal_conditioned_env
import goexplore_py.explorers as explorers
import goexplore_py.cell_representations as cell_representations
from goexplore_py.data_classes import GridDimension, LogParameters
from goexplore_py.goexplore import Explore
from goexplore_py.globals import set_action_meanings, set_master_pid
from goexplore_py.trajectory_manager import CellTrajectoryManager
import goexplore_py.mpi_support as mpi

PROFILER = None

master_pid = None
logger = logging.getLogger(__name__)


def get_game(game,
             target_shape,
             max_pix_value,
             score_objects,
             objects_from_pixels,
             objects_remember_rooms,
             only_keys,
             x_res,
             y_res,
             interval_size,
             seed_low,
             seed_high,
             cell_representation,
             end_on_death):
    game_lowered = game.lower()
    logger.info(f'Loading game: {game_lowered}')
    if game_lowered == 'montezuma' or game_lowered == 'montezumarevenge':
        game_name = 'MontezumaRevenge'
        game_class = montezuma_env.MyMontezuma
        game_class.TARGET_SHAPE = target_shape
        game_class.MAX_PIX_VALUE = max_pix_value
        game_args = dict(
            check_death=end_on_death,
            score_objects=score_objects,
            objects_from_pixels=objects_from_pixels,
            objects_remember_rooms=objects_remember_rooms,
            only_keys=only_keys,
            cell_representation=cell_representation
        )
        grid_resolution = (
            GridDimension('level', 1), GridDimension('objects', 1), GridDimension('room', 1),
            GridDimension('x', x_res), GridDimension('y', y_res)
        )
        cell_representation.set_grid_resolution(grid_resolution)
    elif game_lowered == 'pitfall':
        game_name = 'Pitfall'
        game_class = pitfall_env.MyPitfall
        game_class.TARGET_SHAPE = target_shape
        game_class.MAX_PIX_VALUE = max_pix_value
        game_args = dict(
            cell_representation=cell_representation
        )
        grid_resolution = (
            GridDimension('level', 1), GridDimension('objects', 1), GridDimension('room', 1),
            GridDimension('x', x_res), GridDimension('y', y_res)
        )
        cell_representation.set_grid_resolution(grid_resolution)
    elif 'generic' in game_lowered:
        game_name = game.split('_')[1]
        game_class = generic_atari_env.MyAtari
        game_class.TARGET_SHAPE = target_shape
        game_class.MAX_PIX_VALUE = max_pix_value
        game_args = dict(name=game.split('_')[1])
        grid_resolution = (
            GridDimension('level', 1), GridDimension('objects', 1), GridDimension('room', 1),
            GridDimension('x', x_res), GridDimension('y', y_res)
        )
    elif 'robot' in game_lowered:
        game_name = game.split('_')[1]
        game_class = generic_goal_conditioned_env.MyRobot
        game_args = dict(env_name=game.split('_')[1],
                         interval_size=interval_size,
                         seed_low=seed_low,
                         seed_high=seed_high)
        grid_resolution = (
            GridDimension('level', 1), GridDimension('objects', 1), GridDimension('room', 1),
            GridDimension('x', x_res), GridDimension('y', y_res)
        )
    else:
        raise NotImplementedError("Unknown game: " + game)
    return game_name, game_class, game_args, grid_resolution


def get_frame_wrapper(frame_resize):
    if frame_resize == "RectColorFrame":
        frame_resize_wrapper = ge_wrappers.RectColorFrame
        new_height = 105
        new_width = 80
    elif frame_resize == "RectGreyFrame":
        frame_resize_wrapper = ge_wrappers.RectGreyFrame
        new_height = 105
        new_width = 80
    elif frame_resize == "SquareGreyFrame":
        frame_resize_wrapper = ge_wrappers.SquareGreyFrame
        new_width = 84
        new_height = 84
    elif frame_resize == "RectColorFrameWithBug":
        frame_resize_wrapper = ge_wrappers.RectColorFrameWithBug
        new_height = 105
        new_width = 80
    else:
        raise NotImplementedError("No such frame-size wrapper: " + frame_resize)
    return frame_resize_wrapper, new_height, new_width


def set_global_seeds(i):
    try:
        import tensorflow as local_tf
        local_tf.set_random_seed(i)
    except ImportError:
        # noinspection PyUnusedLocal
        local_tf = None
    np.random.seed(i)
    random.seed(i)


def hrv_and_tf_init(nb_cpu, nb_envs, seed_offset):
    hvd.init()
    master_seed = hvd.rank() * (nb_envs + 1) + seed_offset
    logger.info(f'initialized worker {hvd.rank()} with seed {master_seed}')
    set_global_seeds(master_seed)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=nb_cpu,
                            inter_op_parallelism_threads=nb_cpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    session = tf.Session(config=config)
    return session, master_seed


def get_archive(archive_names,
                optimize_score,
                grid_resolution,
                pre_fill_archive: str = None,
                selector=None,
                cell_trajectory_manager=None,
                max_failed: int = None,
                reset_on_update: bool = False):
    local_archives = []

    domain_knowledge_archive = None
    for archive_name in archive_names.split(':'):
        if archive_name.lower() == 'domainknowledge':
            domain_knowledge_archive = archives.DomainKnowledgeArchive(
                optimize_score,
                selector,
                cell_trajectory_manager,
                grid_resolution,
                max_failed,
                reset_on_update)
            local_archives.append(domain_knowledge_archive)
        elif archive_name.lower() == 'firstroomonly':
            domain_knowledge_archive = archives.FirstRoomOnlyArchive(
                optimize_score,
                selector,
                cell_trajectory_manager,
                grid_resolution,
                max_failed,
                reset_on_update)
            local_archives.append(domain_knowledge_archive)

    if pre_fill_archive is not None:
        if 'file:' in pre_fill_archive:
            file_name = pre_fill_archive.split(':')[1]
            with gzip.open(file_name, 'rb') as file_handle:
                archive = pickle.load(file_handle)

            # Clear some, but not all information
            logger.info('Using pre-filled archive.')
            for key in archive:
                archive[key].score = -float('inf')
                archive[key].nb_seen = 0
                archive[key].nb_chosen = 0
                archive[key].nb_chosen_since_update = 0
                archive[key].nb_chosen_since_to_new = 0
                archive[key].nb_chosen_since_to_update = 0
                archive[key].nb_actions = 0
                archive[key].nb_chosen_for_exploration = 0
                archive[key].nb_reached_for_exploration = 0
                archive[key].reached.clear()
            domain_knowledge_archive.archive = archive
        else:
            raise NotImplementedError("Unknown archive pre-fill: " + pre_fill_archive)

    if len(local_archives) > 1:
        archive_collection = archives.ArchiveCollection()
        for archive in local_archives:
            archive_collection.add_archive(archive)
        archive = archive_collection
    else:
        archive = local_archives[0]
    return archive


def get_goal_rep(goal_representation_name: str,
                 cell_representation: Any,
                 new_width: int,
                 new_height: int,
                 x_res: int,
                 y_res: int,
                 goal_value: int,
                 rep_type: str,
                 rel_final_goal: bool,
                 rel_sub_goal: bool):
    if goal_representation_name == 'raw':
        goal_representation = goal_rep.ScaledGoalRep(
            rep_type,
            rel_final_goal,
            rel_sub_goal,
            cell_representation.array_length)
        policy_name = 'gru_simple_goal'
    elif goal_representation_name == 'scaled_1_1_1':
        goal_representation = goal_rep.ScaledGoalRep(
            rep_type,
            rel_final_goal,
            rel_sub_goal,
            cell_representation.array_length,
            norm_const=[1, 1, 1, 20, 10])
        policy_name = 'gru_simple_goal'
    elif goal_representation_name == 'scaled':
        goal_representation = goal_rep.ScaledGoalRep(
            rep_type,
            rel_final_goal,
            rel_sub_goal,
            cell_representation.array_length,
            norm_const=[3, 3, 24, 20, 10])
        policy_name = 'gru_simple_goal'
    elif goal_representation_name == 'onehot_r25':
        goal_representation = goal_rep.OneHotGoalRep(rep_type, rel_final_goal, rel_sub_goal, [3, 3, 25, 20, 10])
        policy_name = 'gru_simple_goal'
    elif goal_representation_name == 'onehot_r24':
        goal_representation = goal_rep.OneHotGoalRep(rep_type, rel_final_goal, rel_sub_goal, [3, 3, 24, 20, 10])
        policy_name = 'gru_simple_goal'
    elif goal_representation_name == 'onehot':
        logger.debug(f'max values: {cell_representation.get_max_values()}')
        goal_representation = goal_rep.OneHotGoalRep(rep_type, rel_final_goal, rel_sub_goal,
                                                     cell_representation.get_max_values())
        policy_name = 'gru_simple_goal'
    elif goal_representation_name == 'filter_pos_only':
        original_width = 320.0  # This includes the pixel-doubling that is done before the agent location is determined
        original_height = 210.0
        y_offset = 50  # This accounts for the offset that the MR environment uses to determine the location of a cell
        x_fac = new_width / original_width
        y_fac = new_height / original_height
        goal_representation = goal_rep.PosFilterGoalRep(
            (int(new_width), int(new_height), 1),
            int(x_res*x_fac),
            int(y_res*y_fac),
            y_offset=int(y_offset*y_fac),
            goal_value=goal_value,
            norm_const=[],
            pos_only=True)
        policy_name = 'gru_filter_goal'
    elif goal_representation_name == 'filter':
        original_width = 320.0  # This includes the pixel-doubling that is done before the agent location is determined
        original_height = 210.0
        y_offset = 50  # This accounts for the offset that the MR environment uses to determine the location of a cell
        x_fac = new_width / original_width
        y_fac = new_height / original_height
        goal_representation = goal_rep.PosFilterGoalRep(
            (int(new_width), int(new_height), 4),
            int(x_res*x_fac),
            int(y_res*y_fac),
            y_offset=int(y_offset*y_fac),
            goal_value=goal_value,
            norm_const=[3, 3, 24])
        policy_name = 'gru_filter_goal'
    else:
        raise NotImplementedError("No such goal representation: " + goal_representation_name)
    return goal_representation, policy_name


def get_env(game_name,
            game_class,
            game_args,
            clip_rewards,
            frame_resize_wrapper,
            scale_rewards,
            ignore_negative_rewards,
            sticky,
            archive_collection,
            goal_representation,
            done_on_return,
            nb_envs,
            goal_representation_name,
            x_res,
            y_res,
            make_video,
            save_path,
            plot_goal,
            plot_return_prob,
            plot_archive,
            one_vid_per_goal,
            skip,
            goal_explorer,
            seed,
            trajectory_tracker,
            max_exploration_steps,
            max_episode_steps,
            entropy_manager,
            on_done_reward,
            no_exploration_gradients,
            frame_history,
            pixel_repetition,
            sil,
            gamma,
            noops,
            game_reward_factor,
            goal_reward_factor,
            clip_game_reward,
            rew_clip_range,
            max_actions_to_goal,
            max_actions_to_new_cell,
            plot_grid,
            plot_sub_goal,
            cell_reached,
            start_method,
            cell_selection_modifier,
            traj_modifier,
            fail_ent_inc,
            final_goal_reward
            ):
    logger.info(f'Creating environment for game: {game_name}')
    temp_env = gym.make(game_name + 'NoFrameskip-v4')
    set_action_meanings(temp_env.unwrapped.get_action_meanings())

    def make_env(rank):
        def env_fn():
            logger.debug(f'Process seed set to: {rank} seed: {seed + rank}')
            set_global_seeds(seed + rank)
            env_id = game_name + 'NoFrameskip-v4'
            if max_episode_steps is not None:
                gym.spec(env_id).max_episode_steps = max_episode_steps
            local_env = gym.make(env_id)
            set_action_meanings(local_env.unwrapped.get_action_meanings())
            local_env = game_class(local_env, **game_args)
            # Even if make video is true, only define it for one of our environments
            if make_video and rank % nb_envs == 0 and hvd.local_rank() == 0:
                make_video_local = True
            else:
                make_video_local = False
            video_file_prefix = save_path + '/vids/' + game_name
            video_writer = wrappers.VideoWriter(
                local_env,
                video_file_prefix,
                plot_goal=plot_goal,
                x_res=x_res,
                y_res=y_res,
                plot_archive=plot_archive,
                plot_return_prob=plot_return_prob,
                one_vid_per_goal=one_vid_per_goal,
                make_video=make_video_local,
                directory=save_path + '/vids',
                pixel_repetition=pixel_repetition,
                plot_grid=plot_grid,
                plot_sub_goal=plot_sub_goal)
            local_env = video_writer
            local_env = wrappers.my_wrapper(
                local_env,
                clip_rewards=clip_rewards,
                frame_resize_wrapper=frame_resize_wrapper,
                scale_rewards=scale_rewards,
                ignore_negative_rewards=ignore_negative_rewards,
                sticky=sticky,
                skip=skip,
                noops=noops)
            local_env = ge_wrappers.GoalConGoExploreEnv(
                env=local_env,
                archive=archive_collection,
                goal_representation=goal_representation,
                done_on_return=done_on_return,
                video_writer=video_writer,
                goal_explorer=goal_explorer,
                trajectory_tracker=trajectory_tracker,
                max_exploration_steps=max_exploration_steps,
                entropy_manager=entropy_manager,
                on_done_reward=on_done_reward,
                no_exploration_gradients=no_exploration_gradients,
                game_reward_factor=game_reward_factor,
                goal_reward_factor=goal_reward_factor,
                clip_game_reward=clip_game_reward,
                clip_range=rew_clip_range,
                max_actions_to_goal=max_actions_to_goal,
                max_actions_to_new_cell=max_actions_to_new_cell,
                cell_reached=cell_reached,
                cell_selection_modifier=cell_selection_modifier,
                traj_modifier=traj_modifier,
                fail_ent_inc=fail_ent_inc,
                final_goal_reward=final_goal_reward
            )
            video_writer.goal_conditioned_wrapper = local_env
            if sil != 'none':
                local_env = ge_wrappers.SilEnv(
                    env=local_env,
                    goal_representation=goal_representation,
                    trajectory_tracker=trajectory_tracker,
                    gamma=gamma,
                    sil_invalid=(sil == 'sil')
                )
            return local_env
        return env_fn
    logger.info(f'Creating: {nb_envs} environments.')
    env_factories = [make_env(i + nb_envs * hvd.rank()) for i in range(nb_envs)]
    env = ge_wrappers.GoalConSubprocVecEnv(env_factories, start_method)
    env = ge_wrappers.GoalConVecFrameStack(env, frame_history)
    if 'filter' in goal_representation_name:
        env = ge_wrappers.GoalConVecGoalStack(env, goal_representation)
    return env


def get_policy(policy_name):
    policy = {'gru_simple_goal': ge_policies.GRUPolicyGoalConSimpleFlexEnt}[policy_name]
    return policy


def process_defaults(kwargs):
    for key in kwargs:
        if isinstance(kwargs[key], DefaultArg):
            kwargs[key] = kwargs[key].v
    return kwargs


def setup(resolution,
          score_objects,
          base_path,
          game,
          objects_from_pixels,
          objects_remember_rooms,
          only_keys,
          optimize_score,
          use_real_pos,
          resize_x,
          resize_y,
          max_pix_value,
          interval_size,
          seed_low,
          seed_high,
          goal_rep_names,
          pre_fill_archive,
          archive_to_load,
          done_on_return,
          nb_envs,
          goal_value,
          inc_ent_fac,
          end_on_death,
          archive_names,
          selector_name,
          frame_resize,
          clip_rewards,
          scale_rewards,
          ignore_negative_rewards,
          sticky,
          num_steps,
          lam,
          gamma,
          entropy_coef,
          vf_coef,
          l2_coef,
          clip_range,
          model_path,
          test_mode,
          nb_of_epochs,
          learning_rate,
          seed,
          make_video,
          skip,
          freeze_network,
          n_digits,
          checkpoint_game,
          checkpoint_compute,
          max_game_steps,
          max_compute_steps,
          max_hours,
          max_iterations,
          max_cells,
          max_score,
          save_pictures,
          clear_pictures,
          trajectory_tracker_name,
          checkpoint_it,
          selector_weights_str,
          special_attribute_str,
          max_exploration_steps,
          ret_inc_ent_thresh,
          expl_inc_ent_thresh,
          entropy_strategy,
          ent_inc_power,
          ret_inc_ent_fac,
          rep_type,
          rel_final_goal,
          rel_sub_goal,
          on_done_reward,
          no_exploration_gradients,
          frame_history,
          expl_ent_reset,
          pixel_repetition,
          max_episode_steps,
          sil,
          sil_coef,
          sil_vf_coef,
          sil_ent_coef,
          max_traj_candidates,
          exchange_sil_traj,
          random_exp_prob,
          mean_repeat,
          checkpoint_first_iteration,
          checkpoint_last_iteration,
          save_archive,
          save_model,
          disable_hvd,
          noops,
          cell_representation_name,
          game_reward_factor,
          goal_reward_factor,
          clip_game_reward,
          one_vid_per_goal,
          rew_clip_range,
          max_actions_to_goal,
          max_actions_to_new_cell,
          plot_archive,
          plot_goal,
          plot_grid,
          plot_sub_goal,
          cell_reached_name,
          soft_traj_track_window_size,
          expl_state,
          cell_trajectories_file,
          start_method,
          cell_selection_modifier,
          traj_modifier,
          checkpoint_time,
          base_weight,
          delay,
          max_failed,
          legacy_entropy,
          fail_ent_inc,
          temp_dir,
          final_goal_reward,
          low_prob_traj_tresh,
          reset_on_update,
          weight_based_skew):
    global master_pid
    logger.info('Starting setup')
    set_master_pid(os.getpid())
    res = [int(x) for x in resolution.split(',')]
    if len(res) == 2:
        x_res, y_res = res
    elif len(res) == 1:
        x_res = res
        y_res = res
    else:
        raise ValueError('Invalid grid resolution: ' + resolution + '. Valid formats are: x,y and x_and_y.')

    clip_game_reward = bool(clip_game_reward)
    one_vid_per_goal = bool(one_vid_per_goal)
    legacy_entropy = bool(legacy_entropy)
    reset_on_update = bool(reset_on_update)
    weight_based_skew = bool(weight_based_skew)
    rew_clip_range = [float(x) for x in rew_clip_range.split(',')]

    if max_hours:
        max_time = max_hours * 3600
    else:
        max_time = None

    logger.info('Creating log-parameters')
    save_pictures = [x for x in save_pictures.split(':') if x != '']
    clear_pictures = [x for x in clear_pictures.split(':') if x != '']
    log_par = LogParameters(n_digits,
                            checkpoint_game,
                            checkpoint_compute,
                            bool(checkpoint_first_iteration),
                            bool(checkpoint_last_iteration),
                            max_game_steps,
                            max_compute_steps,
                            max_time,
                            max_iterations,
                            max_cells,
                            max_score,
                            save_pictures,
                            clear_pictures,
                            base_path,
                            checkpoint_it,
                            bool(save_archive),
                            bool(save_model),
                            checkpoint_time)

    target_shape = (resize_x, resize_y)

    if use_real_pos:
        target_shape = None
        max_pix_value = None

    # Get the cell representation
    logger.info('Creating cell representation')
    if cell_representation_name == 'level_room_keys_x_y':
        cell_representation = cell_representations.CellRepresentationFactory(cell_representations.MontezumaPosLevel)
        assert cell_representation.supported(game.lower()), cell_representation_name + ' does not support ' + game
    elif cell_representation_name == 'level_room_keys_x_y_score':
        cell_representation = cell_representations.CellRepresentationFactory(cell_representations.LevelKeysRoomXYScore)
        assert cell_representation.supported(game.lower()), cell_representation_name + ' does not support ' + game
    elif cell_representation_name == 'room_treasure_x_y':
        cell_representation = cell_representations.CellRepresentationFactory(cell_representations.PitfallPosLevel)
        assert cell_representation.supported(game.lower()), cell_representation_name + ' does not support ' + game
    elif cell_representation_name == 'room_x_y':
        cell_representation = cell_representations.CellRepresentationFactory(cell_representations.RoomXY)
        assert cell_representation.supported(game.lower()), cell_representation_name + ' does not support ' + game
    else:
        raise NotImplementedError('Unknown cell representation: ' + cell_representation_name)

    # Get game
    game_name, game_class, game_args, grid_resolution = get_game(game=game,
                                                                 target_shape=target_shape,
                                                                 max_pix_value=max_pix_value,
                                                                 score_objects=score_objects,
                                                                 objects_from_pixels=objects_from_pixels,
                                                                 objects_remember_rooms=objects_remember_rooms,
                                                                 only_keys=only_keys,
                                                                 x_res=x_res,
                                                                 y_res=y_res,
                                                                 interval_size=interval_size,
                                                                 seed_low=seed_low,
                                                                 seed_high=seed_high,
                                                                 cell_representation=cell_representation,
                                                                 end_on_death=end_on_death)

    logger.info('Obtaining selector special attributes')
    selector_special_attribute_list = []
    selector_weights_list = []
    for special_attribute in special_attribute_str.split(':'):
        params = special_attribute.split(',')
        special_attribute_name = params[0]
        if special_attribute_name == '':
            pass
        elif special_attribute_name == randselectors.WeightedSumAttribute.get_name():
            weight = randselectors.WeightedSumAttribute(*params[1:])
            selector_special_attribute_list.append(weight)
        elif special_attribute_name == randselectors.SubGoalFailAttribute.get_name():
            weight = randselectors.SubGoalFailAttribute()
            selector_special_attribute_list.append(weight)
        else:
            raise NotImplementedError('Unknown special attribute: ' + special_attribute_name)

    logger.info('Obtaining selector weights')
    for selector_weight in selector_weights_str.split(':'):
        params = selector_weight.split(',')
        selector_weight_name = params[0]
        if selector_weight_name == '':
            pass
        elif selector_weight_name == 'sub_goal_fail':
            assert len(params) == 1, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.SubGoalFailWeight()
            selector_weights_list.append(weight)
        elif selector_weight_name == 'attr':
            name = params[1]
            weight = float(params[2])
            power = float(params[3])
            scalar = float(params[4])
            assert len(params) == 5, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.AttrWeight(name, weight, power, scalar)
            selector_weights_list.append(weight)
        elif selector_weight_name == 'mult':
            name = params[1]
            power = float(params[2])
            scalar = float(params[3])
            assert len(params) == 4, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.MultWeight(name, scalar, power)
            selector_weights_list.append(weight)
        elif selector_weight_name == 'level':
            low_level_weight = float(params[1])
            assert len(params) == 2, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.LevelWeights(low_level_weight)
            selector_weights_list.append(weight)
        elif selector_weight_name == 'neighbor':
            horiz = float(params[1])
            vert = float(params[2])
            score_low = float(params[3])
            score_high = float(params[4])
            assert len(params) == 5, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.NeighborWeights(game_class, horiz, vert, score_low, score_high)
            selector_weights_list.append(weight)
        elif selector_weight_name == 'max_score_reset':
            assert len(params) == 1, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.MaxScoreReset()
            selector_weights_list.append(weight)
        elif selector_weight_name == 'max_score_only':
            assert len(params) == 2, 'Incorrect number of selector-weight parameters provided'
            if hasattr(cell_representation, 'score'):
                weight = randselectors.MaxScoreOnly(str(params[1]))
                selector_weights_list.append(weight)
                selector_special_attribute_list.append(weight)
            else:
                weight = randselectors.MaxScoreOnlyNoScore(str(params[1]))
                selector_special_attribute_list.append(weight)
        elif selector_weight_name == 'max_score_cell':
            assert len(params) == 1, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.MaxScoreCell()
            selector_weights_list.append(weight)
        elif selector_weight_name == 'target_cell':
            assert len(params) % 2 == 1, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.TargetCell()
            attribute = None
            for par in params[1:]:
                if attribute is None:
                    attribute = par
                else:
                    weight.desired_attr[attribute] = int(par)
                    attribute = None
            selector_weights_list.append(weight)
        elif selector_weight_name == 'max_score_and_done':
            assert len(params) == 1, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.MaxScoreAndDone()
            selector_weights_list.append(weight)
        elif selector_weight_name == 'score_based_filter':
            assert len(params) == 1, 'Incorrect number of selector-weight parameters provided'
            weight = randselectors.ScoreBasedFilter()
            selector_weights_list.append(weight)
        else:
            raise NotImplementedError('Unknown selector weight: ' + selector_weight)

    # Get selector
    logger.info('Obtaining random selector')
    if selector_name == 'weighted':
        selector = randselectors.WeightedSelector(selector_weights_list,
                                                  selector_special_attribute_list,
                                                  base_weight,
                                                  weight_based_skew)
    elif selector_name == 'random':
        selector = randselectors.RandomSelector()
    elif selector_name == 'iterative':
        selector = randselectors.IterativeSelector()
    else:
        raise NotImplementedError('Unknown selector: ' + selector_name)

    logger.info('Creating random explorer')
    random_explorer = explorers.RepeatedRandomExplorer(mean_repeat)

    # Get goal explorer
    logger.info('Creating goal explorer')
    goal_explorer = ge_wrappers.DomKnowNeighborGoalExplorer(x_res, y_res, random_exp_prob, random_explorer)

    # Get frame wrapper
    logger.info('Obtaining frame wrapper')
    frame_resize_wrapper, new_height, new_width = get_frame_wrapper(frame_resize)

    logger.info('Obtaining cell trajectory manager')
    cell_trajectory_manager = CellTrajectoryManager(sil,
                                                    cell_representation,
                                                    temp_dir=temp_dir,
                                                    low_prob_traj_tresh=low_prob_traj_tresh)

    # Get the archive
    logger.info('Obtaining archive')
    archive = get_archive(archive_names=archive_names,
                          optimize_score=optimize_score,
                          grid_resolution=grid_resolution,
                          pre_fill_archive=pre_fill_archive,
                          selector=selector,
                          cell_trajectory_manager=cell_trajectory_manager,
                          max_failed=max_failed,
                          reset_on_update=reset_on_update)

    logger.info('Defining cell reached operator')

    def cell_equals(x, y):
        return x == y

    def score_goe(my_cell, archive_cell):
        my_cell_score = my_cell.score
        archive_cell_score = archive_cell.score
        my_cell.score = 0
        archive_cell.score = 0
        result = (my_cell == archive_cell and my_cell_score >= archive_cell_score)
        my_cell.score = my_cell_score
        archive_cell.score = archive_cell_score
        return result

    if cell_reached_name == 'equal':
        cell_reached = cell_equals
    elif cell_reached_name == 'score_goe':
        cell_reached = score_goe
    else:
        raise NotImplementedError('Unknown cell reached operator: ' + cell_reached_name)

    logger.info('Creating trajectory tracker')
    if trajectory_tracker_name == 'dummy':
        trajectory_tracker = trajectory_trackers.DummyTrajectoryTracker(None)
    elif trajectory_tracker_name == 'reward_only':
        trajectory_tracker = trajectory_trackers.RewardOnlyTrajectoryTracker()
    elif trajectory_tracker_name.startswith('potential'):
        discount = float(trajectory_tracker_name.split(':')[1])
        trajectory_tracker = trajectory_trackers.PotentialRewardTrajectoryTracker(discount, cell_reached)
    elif trajectory_tracker_name == 'sequential':
        trajectory_tracker = trajectory_trackers.SequentialTrajectoryTracker(cell_reached)
    elif trajectory_tracker_name == 'sparse':
        trajectory_tracker = trajectory_trackers.SparseTrajectoryTracker(cell_reached)
    elif trajectory_tracker_name == 'soft':
        trajectory_tracker = trajectory_trackers.SoftTrajectoryTracker(cell_reached)
    elif trajectory_tracker_name == 'sparse_soft':
        trajectory_tracker = trajectory_trackers.SparseSoftTrajectoryTracker(cell_reached, soft_traj_track_window_size)
    elif trajectory_tracker_name == 'delayed_soft':
        trajectory_tracker = trajectory_trackers.DelayedSoftTrajectoryTracker(cell_reached,
                                                                              soft_traj_track_window_size, delay)
    elif trajectory_tracker_name == 'super_cell':
        trajectory_tracker = trajectory_trackers.SuperCellTrajectoryTracker(cell_reached)
    else:
        raise NotImplementedError('Unknown trajectory tracker: ' + trajectory_tracker_name)

    if rep_type == 'default':
        rep_type = trajectory_tracker.get_default_goal()

    # Get goal representation
    logger.info('Obtaining goal representation')
    goal_representation, policy_name = get_goal_rep(goal_representation_name=goal_rep_names,
                                                    cell_representation=cell_representation,
                                                    new_width=new_width,
                                                    new_height=new_height,
                                                    x_res=x_res,
                                                    y_res=y_res,
                                                    goal_value=goal_value,
                                                    rep_type=rep_type,
                                                    rel_final_goal=rel_final_goal,
                                                    rel_sub_goal=rel_sub_goal)

    if cell_trajectories_file is not None and cell_trajectories_file != "":
        cell_trajectory_manager.create_load_ops(cell_trajectories_file, goal_representation)

    logger.info('Creating entropy manager')
    entropy_manager = ge_wrappers.EntropyManager(
        inc_ent_fac=inc_ent_fac,
        ret_inc_ent_thresh=ret_inc_ent_thresh,
        expl_inc_ent_thresh=expl_inc_ent_thresh,
        entropy_strategy=entropy_strategy,
        ent_inc_power=ent_inc_power,
        ret_inc_ent_fac=ret_inc_ent_fac,
        expl_ent_reset=expl_ent_reset,
        legacy_entropy=legacy_entropy)

    # Get environment
    # Note: At this point, the archive and all associated information will copied to the worker processes, meaning that
    # if there is any information that should NOT be copied to the worker processes that information should be set after
    # the get_env function is called.
    env = get_env(game_name=game_name,
                  game_class=game_class,
                  game_args=game_args,
                  clip_rewards=clip_rewards,
                  frame_resize_wrapper=frame_resize_wrapper,
                  scale_rewards=scale_rewards,
                  ignore_negative_rewards=ignore_negative_rewards,
                  sticky=sticky,
                  archive_collection=archive,
                  goal_representation=goal_representation,
                  done_on_return=done_on_return,
                  nb_envs=nb_envs,
                  goal_representation_name=goal_rep_names,
                  make_video=make_video,
                  save_path=base_path,
                  entropy_manager=entropy_manager,
                  plot_goal=plot_goal,
                  plot_archive=plot_archive,
                  plot_return_prob=True,
                  one_vid_per_goal=one_vid_per_goal,
                  skip=skip,
                  goal_explorer=goal_explorer,
                  seed=seed,
                  trajectory_tracker=trajectory_tracker,
                  x_res=x_res,
                  y_res=y_res,
                  max_exploration_steps=max_exploration_steps,
                  max_episode_steps=max_episode_steps,
                  on_done_reward=on_done_reward,
                  no_exploration_gradients=no_exploration_gradients,
                  frame_history=frame_history,
                  pixel_repetition=pixel_repetition,
                  sil=sil,
                  gamma=gamma,
                  noops=noops,
                  game_reward_factor=game_reward_factor,
                  goal_reward_factor=goal_reward_factor,
                  clip_game_reward=clip_game_reward,
                  rew_clip_range=rew_clip_range,
                  max_actions_to_goal=max_actions_to_goal,
                  max_actions_to_new_cell=max_actions_to_new_cell,
                  plot_grid=plot_grid,
                  plot_sub_goal=plot_sub_goal,
                  cell_reached=cell_reached,
                  start_method=start_method,
                  cell_selection_modifier=cell_selection_modifier,
                  traj_modifier=traj_modifier,
                  fail_ent_inc=fail_ent_inc,
                  final_goal_reward=final_goal_reward
                  )

    # Get the policy
    logger.info('Obtaining the policy')
    policy = get_policy(policy_name)

    logger.info('Initializing the model')
    if sil == 'sil' or sil == 'nosil' or sil == 'noframes':
        model = ge_models.GoalConFlexEntSilModel()
        model.init(policy=policy,
                   ob_space=env.observation_space,
                   ac_space=env.action_space,
                   nenv=env.num_envs,
                   nsteps=num_steps + num_steps // 2,
                   ent_coef=entropy_coef,
                   vf_coef=vf_coef,
                   l2_coef=l2_coef,
                   cliprange=clip_range,
                   load_path=model_path,
                   test_mode=test_mode,
                   goal_space=env.goal_space,
                   sil_coef=sil_coef,
                   sil_vf_coef=sil_vf_coef,
                   sil_ent_coef=sil_ent_coef,
                   disable_hvd=bool(disable_hvd))
    elif sil == 'replay' or sil == 'none':
        model = ge_models.GoalConditionedModelFlexEnt()
        model.init(policy=policy,
                   ob_space=env.observation_space,
                   ac_space=env.action_space,
                   nenv=env.num_envs,
                   nsteps=num_steps + num_steps // 2,
                   ent_coef=entropy_coef,
                   vf_coef=vf_coef,
                   l2_coef=l2_coef,
                   cliprange=clip_range,
                   load_path=model_path,
                   test_mode=test_mode,
                   goal_space=env.goal_space,
                   disable_hvd=bool(disable_hvd))
    else:
        raise NotImplementedError('Sil has to be one of: "sil", or "none".')

    logger.info('Creating runner')
    if sil == 'sil' or sil == 'nosil' or sil == 'noframes':
        runner_class = ge_runners.RunnerFlexEntSilProper
    elif sil == 'none':
        runner_class = ppo.Runner
    else:
        raise NotImplementedError('Sil has to be one of: "sil", or "none".')
    norm_adv = True
    subtract_rew_avg = False
    runner = runner_class(env=env,
                          model=model,
                          nsteps=num_steps,
                          gamma=gamma,
                          lam=lam,
                          norm_adv=norm_adv,
                          subtract_rew_avg=subtract_rew_avg
                          )

    logger.info('Creating gatherer')
    trajectory_gatherer = trajectory_gatherers.StochasticGatherer(env=env,
                                                                  nb_of_epochs=nb_of_epochs,
                                                                  learning_rate=learning_rate,
                                                                  model=model,
                                                                  freeze_network=freeze_network,
                                                                  runner=runner
                                                                  )

    if archive_to_load is not None:
        logger.info('Loading archive')
        if isinstance(archive_to_load, str):
            with gzip.open(archive_to_load, 'rb') as file_handle:
                archive_to_load = pickle.load(file_handle)
        archive.set_state(archive_to_load)

    logger.info('Creating explorer')
    expl = Explore(
        trajectory_gatherer=trajectory_gatherer,
        archive=archive,
        sil=sil,
        max_traj_candidates=max_traj_candidates,
        exchange_sil_traj=exchange_sil_traj
    )

    if expl_state is not None:
        logger.info('Loading explorer state')
        if isinstance(expl_state, str):
            with gzip.open(expl_state, 'rb') as file_handle:
                expl_state = pickle.load(file_handle)
        expl.set_state(expl_state)

    if len(expl.archive.archive) != 0:
        logger.info('Archive is initialized: recalculating trajectory id...')
        runner.init_trajectory_id(expl.archive)
    else:
        logger.info('Archive is empty')

    if cell_trajectories_file is not None and cell_trajectories_file != "":
        traj_manager = archive.cell_trajectory_manager
        cell_selector = archive.cell_selector
        traj_manager.traj_prob_dict = cell_selector.get_traj_probabilities_dict(archive.archive)
        local_comm = mpi.get_comm_world().Split_type(mpi.get_comm_type_shared())

        if local_comm.rank == 0:
            cell_trajectory_manager.sess = tf.get_default_session()
            logger.info('Loading full trajectories...')
            cell_trajectory_manager.run_load_op()
            logger.info('Full trajectories loaded.')
        logger.info('Waiting for rank 0 to finish loading checkpoint...')
        local_comm.barrier()

        mpi.get_comm_world().barrier()
        logger.info('Loading trajectories is done!')

    cell_trajectory_manager.keep_new_trajectories = True

    logger.info('Setup finished!')
    return expl, log_par


def safe_set_argument(args, attr, value):
    """
    Intended to prevent accidentally overwritting arguments that were already provided as commandline options.

    @param args:
    @param attr:
    @param value:
    @return: None
    """
    if not hasattr(args, attr):
        setattr(args, attr, value)
    else:
        raise RuntimeError(
            'Argument ' + value + ' has already been provided as an option, and should not be set here.')


class DefaultArg:
    def __init__(self, v):
        self.v = v


def start_logger(kwargs):
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


def del_out_of_setup_args(kwargs):
    # Process the resize shape argument
    if kwargs['resize_shape']:
        x, y, p = kwargs['resize_shape'].split('x')
        kwargs['resize_x'] = int(x)
        kwargs['resize_y'] = int(y)
        kwargs['max_pix_value'] = int(p)

    del kwargs['fail_on_duplicate']
    del kwargs['load_path']
    del kwargs['profile']
    del kwargs['trace_memory']
    del kwargs['resize_shape']
    del kwargs['disable_logging']
    del kwargs['no_option']
    del kwargs['warm_up_cycles']
    del kwargs['continue']
    del kwargs['log_after_warm_up']
    del kwargs['screenshot_merge']
    del kwargs['clear_checkpoints']
    del kwargs['log_info']
    del kwargs['log_files']
    return kwargs


def parse_arguments():
    global PROFILER

    if platform == "darwin":
        # Circumvents the following issue on Mac OS:
        # https://github.com/opencv/opencv/issues/5150
        cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--seed', type=int, default=DefaultArg(0),
                        help='The random seed.')
    parser.add_argument('--profile', dest='profile', action='store_true', default=DefaultArg(False),
                        help='Whether or not to enable a profiler.')
    parser.add_argument('--trace_memory', dest='trace_memory', action='store_true', default=DefaultArg(False),
                        help='Whether or not to enable a memory allocation trace.')

    # Cell representation arguments
    parser.add_argument('--resolution', '--res', type=str, default=DefaultArg('16,16'),
                        help='Length of the side of a grid cell. A doubled atari frame is 320 by 210.')
    parser.add_argument('--score_objects', dest='score_objects', default=DefaultArg(True), action='store_false',
                        help='Use scores in the cell description. Otherwise objects will be used.')
    parser.add_argument('--objects_from_ram', dest='objects_from_pixels', default=DefaultArg(True),
                        action='store_false',
                        help='Get the objects from RAM instead of pixels.')
    parser.add_argument('--all_objects', dest='only_keys', default=DefaultArg(True), action='store_false',
                        help='Use all objects in the state instead of just the keys')
    parser.add_argument('--objects_remember_rooms', dest='objects_remember_rooms', default=DefaultArg(False),
                        action='store_true',
                        help='Remember which room the objects picked up came from. Makes it easier to solve the game '
                             '(because the state encodes the location of the remaining keys anymore), but takes more '
                             'time/memory space, which in practice makes it worse quite often. Using this is better if '
                             'running with --no_optimize_score')
    parser.add_argument('--resize_x', '--rx', type=int, default=DefaultArg(11),
                        help='What to resize the pixels to in the x direction for use as a state.')
    parser.add_argument('--resize_y', '--ry', type=int, default=DefaultArg(8),
                        help='What to resize the pixels to in the y direction for use as a state.')
    parser.add_argument('--state_is_pixels', '--pix', default=DefaultArg(True), dest='use_real_pos',
                        action='store_false',
                        help='If this is on, the state will be resized pixels, not human prior.')
    parser.add_argument('--max_pix_value', '--mpv', type=int, default=DefaultArg(8),
                        help='The range of pixel values when resizing will be rescaled to from 0 to this value. '
                             'Lower means fewer possible states in states_is_pixels.')
    parser.add_argument('--resize_shape', type=str, default=DefaultArg(None),
                        help='Shortcut for passing --resize_x (0), --resize_y (1) and --max_pix_value (2) all at the '
                             'same time: 0x1x2')

    # I/O Arguments
    parser.add_argument('--base_path', '-p', type=str, default=DefaultArg('./results/'),
                        help='Folder in which to store results')
    parser.add_argument('--load_path', type=str, default=DefaultArg(''),
                        help='Folder from which to load a pre-trained model.')
    parser.add_argument('--fail_on_duplicate', default=DefaultArg(False),
                        dest='fail_on_duplicate', action='store_true',
                        help='Fail if the base directory already exists.')

    # Environment arguments
    parser.add_argument('--end_on_death', dest='end_on_death', default=DefaultArg(False), action='store_true',
                        help='End episode on death.')
    parser.add_argument('--game', '-g', type=str, default=DefaultArg('montezuma'),
                        help='Determines the game to which apply goexplore.')
    parser.add_argument('--interval_size', type=float, default=DefaultArg(0.1),
                        help='The interval size for robotics envs.')
    parser.add_argument('--sticky_actions', default=DefaultArg(False), dest='sticky', action='store_true',
                        help='Whether to run with sticky actions.')
    parser.add_argument('--noops', default=DefaultArg(False), dest='noops', action='store_true',
                        help='Whether to run with noops actions.')

    # Stopping criteria arguments
    parser.add_argument('--max_game_steps', type=int, default=DefaultArg(None),
                        help='Maximum number of GAME frames.')
    parser.add_argument('--max_compute_steps', '--mcs', type=int, default=DefaultArg(None),
                        help='Maximum number of COMPUTE frames.')
    parser.add_argument('--max_iterations', type=int, default=DefaultArg(None),
                        help='Maximum number of iterations.')
    parser.add_argument('--max_hours', '--mh', type=float, default=DefaultArg(None),
                        help='Maximum number of hours to run this for.')
    parser.add_argument('--max_cells', type=int, default=DefaultArg(None),
                        help='The maximum number of cells before stopping.')
    parser.add_argument('--max_score', type=float, default=DefaultArg(None),
                        help='Stop when this score (or more) has been reached in the archive.')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_game', type=int, default=DefaultArg(None),
                        help='Save a checkpoint every this many GAME frames (note: recommended to ignore, since this '
                             'grows very fast at the end).')
    parser.add_argument('--checkpoint_compute', type=int, default=DefaultArg(None),
                        help='Save a checkpoint every this many COMPUTE frames.')
    parser.add_argument('--checkpoint_it', type=int, default=DefaultArg(None),
                        help='Save a checkpoint every this many iterations.')
    parser.add_argument('--checkpoint_time', type=int, default=DefaultArg(None),
                        help='Save a checkpoint every this many seconds.')
    parser.add_argument('--clear_checkpoints', dest='clear_checkpoints',
                        type=str, default=DefaultArg(''),
                        help='Colon separated list of the checkpoints that need to be cleared after every iteration. '
                        'Possible options are: archive, model, trajectory, and all.')

    # Picture-dump arguments
    parser.add_argument('--pictures', dest='save_pictures', type=str, default=DefaultArg(''),
                        help='Save pictures of the specified checkpoints')
    parser.add_argument('--clear_pictures', dest='clear_pictures', type=str, default=DefaultArg(''),
                        help='Clear the pictures of the specified checkpoints after every iteration')

    # Go-Explore archive arguments
    parser.add_argument('--archives', dest='archive_names', type=str, default=DefaultArg('DomainKnowledge'),
                        help='What archives to use. '
                             'The first archive in this colon-separated list will be the active archive.')
    parser.add_argument('--no_optimize_score', dest='optimize_score', default=DefaultArg(True), action='store_false',
                        help='Don\'t optimize for score (only speed). Will use fewer "game frames" and come up with '
                             'faster trajectories with lower scores. If not combined with --objects_remember_rooms and '
                             '--objects_from_ram is not enabled, things should run much slower.')
    parser.add_argument('--pre_fill_archive', type=str, default=DefaultArg(None), dest='pre_fill_archive',
                        help='Pre-fill the archive with a predefined set of cells.')

    # Goal representation arguments
    parser.add_argument('--goal_rep', dest='goal_rep_names', type=str, default=DefaultArg('raw'),
                        help='How to represent the goal location to the neural network.')
    parser.add_argument('--goal_value', type=int, default=DefaultArg(255),
                        help='The value of the "goal pixels" when using the "filter" goal representation.')
    parser.add_argument('--final_goal_or_sub_goal', dest='rep_type',
                        type=str, default=DefaultArg('default'),
                        help='Whether to present the network with the final goal, the next sub-goal, or both.'
                             'Options are: default, final_goal, sub_goal, or final_goal_and_sub_goal')
    parser.add_argument('--rel_final_goal', default=DefaultArg(False), dest='rel_final_goal', action='store_true',
                        help='Whether the final goal should be relative to the agents current cell.')
    parser.add_argument('--rel_sub_goal', default=DefaultArg(False), dest='rel_sub_goal', action='store_true',
                        help='Whether the sub goal should be relative to the agents current cell.')

    # State representation arguments
    parser.add_argument('--frame_resize', dest='frame_resize', type=str, default=DefaultArg('RectColorFrame'),
                        help='How to resize the frame, and whether the frame will be color or grey scale.')
    parser.add_argument('--frame_history', dest='frame_history', type=int, default=DefaultArg(4),
                        help='How many old frames to show.')

    # GO-Explore arguments
    parser.add_argument('--done_on_return', default=DefaultArg(False), dest='done_on_return', action='store_true',
                        help='Only return, never explore.')
    parser.add_argument('--trajectory_tracker', dest='trajectory_tracker_name',
                        type=str, default=DefaultArg('dummy'),
                        help='How to track trajectories for returning to a cell, if such functionality is desired.'
                             'Options are: dummy, reward_only, and sequential')
    parser.add_argument('--selector', dest='selector_name',
                        type=str, default=DefaultArg('random'),
                        help='How to select cells to return to. '
                             'Options are: random, mr_curriculum, iterative, and weighted.')
    parser.add_argument('--selector_weights', dest='selector_weights_str',
                        type=str, default=DefaultArg(''),
                        help='How to weight cells during selection.')
    parser.add_argument('--special_attributes', dest='special_attribute_str',
                        type=str, default=DefaultArg(''),
                        help='Special attributes that can be used for weighting.')
    parser.add_argument('--max_exploration_steps', dest='max_exploration_steps', type=int, default=DefaultArg(100),
                        help='For how many steps do we explore before choosing a different exploration goal.')
    parser.add_argument('--max_episode_steps', dest='max_episode_steps', type=int, default=DefaultArg(None),
                        help='The total number of steps we can take each episode.')
    parser.add_argument('--on_done_reward', dest='on_done_reward', type=float, default=DefaultArg(0),
                        help='Reward provided for finishing an episode. Should generally be zero or negative, to '
                             'discourage the agent from ending the episode.')
    parser.add_argument('--no_exploration_gradients', default=DefaultArg(False),
                        dest='no_exploration_gradients', action='store_true',
                        help='Whether we should use exploration steps for updating our policy.')

    # PPO arguments
    parser.add_argument('--nb_envs', type=int, default=DefaultArg(16),
                        help='The number of environments to run in parallel with Horovod.')
    parser.add_argument('--clip_rewards', default=DefaultArg(False), dest='clip_rewards', action='store_true',
                        help='Whether to perform reward clipping.')
    parser.add_argument('--gamma', type=float, default=DefaultArg(0.999),
                        help='The discount factor.')
    parser.add_argument('--freeze_network', default=DefaultArg(False), dest='freeze_network', action='store_true',
                        help='Do not train the network (i.e. the policy and value function).')

    # Entropy related arguments
    parser.add_argument('--entropy_strategy', dest='entropy_strategy', type=str, default=DefaultArg('none'),
                        help='Specifies how to increase entropy (if at all). Currently implemented are the '
                             '"fixed_increase" strategy, which increase the entropy after a threshold is reached and '
                             'the "dynamic_increase" strategy, which slowly increases entropy after a threshold is '
                             'reached based on the number of steps taking beyond this threshold. Also implemented is '
                             'the "none" strategy, which means entropy is never increased.')
    parser.add_argument('--inc_ent_fac', dest='inc_ent_fac', type=float, default=DefaultArg(None),
                        help='The factor by which to increase the entropy, either when the threshold is reached '
                             '("fixed_increase" entropy strategy), or per step beyond the threshold '
                             '("dynamic_increase" entropy strategy).')
    parser.add_argument('--ret_inc_ent_thresh', dest='ret_inc_ent_thresh', type=int, default=DefaultArg(0),
                        help='A grace period on top of the estimated number of steps towards the current goal after '
                             'which entropy will be increased.')
    parser.add_argument('--expl_inc_ent_thresh', dest='expl_inc_ent_thresh', type=int, default=DefaultArg(-1),
                        help='The number of steps of exploration towards a particular goal after which entropy will '
                             'be increased.')
    parser.add_argument('--ret_inc_ent_fac', dest='ret_inc_ent_fac', type=float, default=DefaultArg(1),
                        help='The estimated length of the trajectory is multiplied by this factor to determine the '
                             'threshold after which entropy will be increased. Setting it > 1 means that the policy '
                             'will get a grace period with respect to the expected number of steps that scales with '
                             'the total expected length. Setting it to 0 means the length of the trajectory will be '
                             'ignored.')
    parser.add_argument('--ent_inc_power', dest='ent_inc_power', type=float, default=DefaultArg(1),
                        help='The power by which the scaled number of steps beyond the threshold is raised before it '
                             'is passed as the entropy multiplier to the policy. Is only used in the '
                             '"dynamic_increase" entropy strategy.')
    parser.add_argument('--expl_ent_reset', dest='expl_ent_reset', type=str, default=DefaultArg('on_new_cell'),
                        help='When should we reset the entropy while exploring.')
    parser.add_argument('--sil', dest='sil', type=str, default=DefaultArg('none'),
                        help='Turn self-imitation learning on.')
    parser.add_argument('--sil_coef', dest='sil_coef', type=float, default=DefaultArg(0.1),
                        help='The general coefficient for self-imitation learning.')
    parser.add_argument('--sil_vf_coef', dest='sil_vf_coef', type=float, default=DefaultArg(0.01),
                        help='The self-imitation learning coefficient for the value function.')
    parser.add_argument('--sil_ent_coef', dest='sil_ent_coef', type=float, default=DefaultArg(0),
                        help='The self-imitation learning coefficient for the policy.')
    parser.add_argument('--max_traj_candidates', dest='max_traj_candidates', type=int, default=DefaultArg(1),
                        help='The maximum number of trajectories to keep for the purpose of imitation learning.')
    parser.add_argument('--exchange_sil_traj', dest='exchange_sil_traj', type=str, default=DefaultArg('none'),
                        help='Whether and how to exchange full trajectories with other MPI workers.')
    parser.add_argument('--random_exp_prob', dest='random_exp_prob', type=float, default=DefaultArg(0),
                        help='The probability that we will be taking random actions,'
                             ' rather than sampling from the policy.')
    parser.add_argument('--mean_repeat', dest='mean_repeat', type=int, default=DefaultArg(20),
                        help='The expected number of times that an action will be repeated by the random explorer.')
    parser.add_argument('--checkpoint_first_iteration', dest='checkpoint_first_iteration',
                        type=int, default=DefaultArg(1),
                        help='Whether to write a checkpoint for the first iteration')
    parser.add_argument('--checkpoint_last_iteration', dest='checkpoint_last_iteration',
                        type=int, default=DefaultArg(1),
                        help='Whether to write a checkpoint for the last iteration')
    parser.add_argument('--save_archive', dest='save_archive',
                        type=int, default=DefaultArg(1),
                        help='Whether to save the archive when writing a checkpoint')
    parser.add_argument('--save_model', dest='save_model',
                        type=int, default=DefaultArg(1),
                        help='Whether to save the model when writing a checkpoint')
    parser.add_argument('--disable_hvd', dest='disable_hvd',
                        type=int, default=DefaultArg(0),
                        help='Whether to disable horovod (for debugging purposes)')
    parser.add_argument('--disable_logging', dest='disable_logging',
                        type=int, default=DefaultArg(0),
                        help='Whether to disable logging (for debugging purposes)')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=DefaultArg(1e-4),
                        help='Sets the learning rate')
    parser.add_argument('--cell_representation', dest='cell_representation_name',
                        type=str, default=DefaultArg('level_room_keys_x_y'),
                        help='The cell representation to use.')
    parser.add_argument('--game_reward_factor', dest='game_reward_factor',
                        type=float, default=DefaultArg(0.0),
                        help='The factor by which to multiply the game reward')
    parser.add_argument('--goal_reward_factor', dest='goal_reward_factor',
                        type=float, default=DefaultArg(1.0),
                        help='The factor by which to multiply the goal reward')
    parser.add_argument('--clip_game_reward', dest='clip_game_reward',
                        type=int, default=DefaultArg(1),
                        help='Whether to clip (1) the game reward or not (0)')
    parser.add_argument('--one_vid_per_goal', dest='one_vid_per_goal',
                        type=int, default=DefaultArg(1),
                        help='Whether to create only one video per goal (1) or not (0)')
    parser.add_argument('--rew_clip_range', dest='rew_clip_range',
                        type=str, default=DefaultArg('-1,1'),
                        help='To what range the reward should be clipped')
    parser.add_argument('--no_option', default=DefaultArg(False), dest='no_option', action='store_true',
                        help='Placeholder for providing no option')
    parser.add_argument('--max_actions_to_goal', dest='max_actions_to_goal',
                        type=int, default=DefaultArg(-1),
                        help='The maximum number of actions the agent gets to reach a chosen goal')
    parser.add_argument('--max_actions_to_new_cell', dest='max_actions_to_new_cell',
                        type=int, default=DefaultArg(-1),
                        help='The maximum number of actions the agent gets to reach a new cell')
    parser.add_argument('--cell_reached', dest='cell_reached_name',
                        type=str, default=DefaultArg('equal'),
                        help='How to determine whether a cell is considered reached when following a trajectory.')
    parser.add_argument('--soft_traj_win_size', dest='soft_traj_track_window_size',
                        type=int, default=DefaultArg(10),
                        help='The size of the soft-trajectory window')
    parser.add_argument('--load_archive', dest='archive_to_load',
                        type=str, default=DefaultArg(None),
                        help='The archive to load')
    parser.add_argument('--expl_state', dest='expl_state',
                        type=str, default=DefaultArg(None),
                        help='The go-explore state to load')
    parser.add_argument('--warm_up_cycles', dest='warm_up_cycles',
                        type=int, default=DefaultArg(10),
                        help='How many warmup cycles to perform when continuing from a checkpoint')
    parser.add_argument('--continue', dest='continue',
                        default=DefaultArg(False), action='store_true',
                        help='Whether to automatically continue a run.')
    parser.add_argument('--log_after_warm_up', dest='log_after_warm_up',
                        default=DefaultArg(False), action='store_true',
                        help='Whether to create a log immediately after performing a warm up (for debugging purpose).')
    parser.add_argument('--start_method', dest='start_method',
                        type=str, default=DefaultArg('spawn'),
                        help='Currently, fork causes a deadlock when loading checkpoints.')
    parser.add_argument('--cell_selection_modifier', dest='cell_selection_modifier',
                        type=str, default=DefaultArg('none'),
                        help='Whether to modify the destination cell after we select it. Current option is prev, '
                             'which changes the target cell with a previous cell in the trajectory.')
    parser.add_argument('--traj_modifier', dest='traj_modifier',
                        type=str, default=DefaultArg('none'),
                        help='Whether to modify the trajectory towards our destination cell. Current option is prev, '
                             'which uses the trajectory of a previous cell, and then append selected cell.')
    parser.add_argument('--base_weight', dest='base_weight',
                        type=float, default=DefaultArg(1.0),
                        help='The base weight to which additive and multiplicative weights get applied.')
    parser.add_argument('--delay', dest='delay',
                        type=int, default=DefaultArg(0),
                        help='How long you have to stay in a soft trajectory before you are skipped forward.')
    parser.add_argument('--max_failed', dest='max_failed',
                        type=int, default=DefaultArg(100),
                        help='The maximum when tracking the number of times the agent fails to reach a cell')
    parser.add_argument('--nb_of_epochs', dest='nb_of_epochs',
                        type=int, default=DefaultArg(4),
                        help='The number of PPO epochs')
    parser.add_argument('--legacy_entropy', dest='legacy_entropy',
                        type=int, default=DefaultArg(0),
                        help='Whether to run with a legacy implementation of the entropy increase method')
    parser.add_argument('--fail_ent_inc', dest='fail_ent_inc',
                        type=str, default=DefaultArg('none'),
                        help='Whether an how to increase entropy near high-failure cells.')
    parser.add_argument('--screenshot_merge', dest='screenshot_merge',
                        type=str, default=DefaultArg('mpi'),
                        help='How to merge screenshots.')
    parser.add_argument('--final_goal_reward', dest='final_goal_reward',
                        type=float, default=DefaultArg(1.0),
                        help='The reward obtained for reaching the final goal (as opposed to a sub-goal).')
    parser.add_argument('--low_prob_traj_tresh', dest='low_prob_traj_tresh',
                        type=float, default=DefaultArg(0.00001),
                        help='The probability below which trajectories will be written to disk.')
    parser.add_argument('--reset_on_update', dest='reset_on_update',
                        type=int, default=DefaultArg(0),
                        help='Whether to reset selection information of a cell when it is updated.')
    parser.add_argument('--weight_based_skew', dest='weight_based_skew',
                        type=int, default=DefaultArg(0),
                        help='Whether to skew the cell-selection probability towards higher scoring cells')
    parser.add_argument('--log_info', dest='log_info',
                        type=str, default=DefaultArg(''),
                        help='Whether to enable debug output and at what level. Possible values are: CRITICAL, ERROR'
                             'WARNING, INFO, DEBUG, NOTSET.')
    parser.add_argument('--log_files', dest='log_files',
                        type=str, default=DefaultArg(''),
                        help='From which files we should log information. Example: atari_reset.atari_reset.policies')

    args = parser.parse_args()

    # Hard-coded values
    safe_set_argument(args, 'entropy_coef', DefaultArg(1e-4))
    safe_set_argument(args, 'n_digits', DefaultArg(12))
    safe_set_argument(args, 'scale_rewards', DefaultArg(None))
    safe_set_argument(args, 'ignore_negative_rewards', DefaultArg(False))
    safe_set_argument(args, 'num_steps', DefaultArg(128))

    safe_set_argument(args, 'vf_coef', DefaultArg(0.5))
    safe_set_argument(args, 'l2_coef', DefaultArg(1e-7))
    safe_set_argument(args, 'lam', DefaultArg(.95))
    safe_set_argument(args, 'clip_range', DefaultArg(0.1))
    safe_set_argument(args, 'test_mode', DefaultArg(False))

    safe_set_argument(args, 'seed_low', DefaultArg(None))
    safe_set_argument(args, 'seed_high', DefaultArg(None))
    safe_set_argument(args, 'make_video', DefaultArg(False))
    safe_set_argument(args, 'skip', DefaultArg(4))
    safe_set_argument(args, 'pixel_repetition', DefaultArg(1))
    safe_set_argument(args, 'plot_archive', DefaultArg(True))
    safe_set_argument(args, 'plot_goal', DefaultArg(True))
    safe_set_argument(args, 'plot_grid', DefaultArg(True))
    safe_set_argument(args, 'plot_sub_goal', DefaultArg(True))
    
    return args
