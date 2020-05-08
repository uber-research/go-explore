# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Dict, Tuple, Any, Set, Optional
from .data_classes import dataclass, copyfield
import sys
import random
import copy
import tensorflow as tf
import numpy as np
import goexplore_py.globals as global_const
import pickle
import os


@dataclass
class CellTrajectory:
    cell_ids: List[int] = copyfield([])
    actions_per_cell: List[int] = copyfield([])
    score: int = 0
    total_actions: int = 0
    reference_count: int = 0
    id: int = 0
    fragment_start: int = -1
    finished: bool = False
    tie_breaker: float = 0
    exp_strat: int = 0
    exp_new_cells: int = 0
    ret_new_cells: int = 0
    frame_finished: int = 0


@dataclass
class FullTrajectoryInfo:
    on_disk: bool = False


class CellTrajectoryManager:
    empty_trajectory_id = -1

    def __init__(self, sil, cell_class, sess=None, temp_dir='', low_prob_traj_tresh=1/10000):
        # Cell trajectory information
        self.cell_trajectories: Dict[int, CellTrajectory] = {}
        self.full_trajectories: Dict[int, List[Tuple[Any, np.float32, Tuple[np.ndarray, np.ndarray], np.int64,
                                                     np.float32]]] = {}
        self.full_trajectory_info: Dict[int, FullTrajectoryInfo] = {}
        self.traj_prob_dict: Optional[Dict[int, float]] = None

        self.del_ret_new_cells: int = 0
        self.del_policy_new_cells: int = 0
        self.del_rand_new_cells: int = 0

        # Temporary information that is reset between updates or when switching trajectories
        self.cell_trajectory_id: int = -1
        self.updated_trajectory_fragments: Dict[int, CellTrajectory] = {}
        self.synced_trajectories: List[int] = []
        self.new_trajectories: List[int] = []
        self.cells_seen: Dict[int, Set[Any]] = {}

        # Settings passed as arguments
        self.sil: str = sil
        self.keep_new_trajectories: bool = False
        self.low_prob_traj_tresh: float = low_prob_traj_tresh
        self.temp_dir = temp_dir
        self.read_header_feature = None
        self.read_body_feature = None
        self.get_obs = None
        self.cell_class = cell_class
        self.sess = sess

    def seen(self, cell_key):
        self.cells_seen[self.cell_trajectory_id].add(cell_key)

    def already_seen(self, cell_key):
        return cell_key in self.cells_seen[self.cell_trajectory_id]

    def register_new_trajectory(self, traj_id):
        if self.keep_new_trajectories:
            self.new_trajectories.append(traj_id)
            self.increment_reference(traj_id)

    def clear_new_trajectories(self):
        for traj_id in self.new_trajectories:
            self.decrement_reference(traj_id)
        self.new_trajectories = []

    def create_load_ops(self, filename, goal_representation):
        goal_shape = goal_representation.get_goal_space().shape

        header_feature = {
            'trajectory_id': tf.io.FixedLenFeature([], tf.int64),
            'trajectory_length': tf.io.FixedLenFeature([], tf.int64),
        }

        body_feature = {
            'cell_key': tf.io.FixedLenFeature([self.cell_class.array_length], tf.int64),
            'reward': tf.io.FixedLenFeature([], tf.float32),
            'obs': tf.io.FixedLenFeature([], tf.string),
            'goal': tf.io.FixedLenFeature(goal_shape, tf.float32),
            'action': tf.io.FixedLenFeature([], tf.int64),
            'ge_reward': tf.io.FixedLenFeature([], tf.float32),
        }

        dataset = tf.data.TFRecordDataset(filename)

        iterator = dataset.make_one_shot_iterator()

        get_next = iterator.get_next()

        self.read_header_feature = tf.io.parse_single_example(get_next, header_feature)
        self.read_body_feature = tf.io.parse_single_example(get_next, body_feature)
        self.get_obs = tf.expand_dims(tf.decode_raw(self.read_body_feature['obs'], out_type=tf.int8), axis=1)

    def run_load_op(self):
        nb_traj_tuples = 0
        traj_id = CellTrajectoryManager.empty_trajectory_id
        total_trajectories = len(self.cell_trajectories)
        trajectories_loaded = -1

        while True:
            if nb_traj_tuples == 0:
                trajectories_loaded += 1
                if trajectories_loaded % 100 == 0:
                    print(f'Trajectories loaded: {trajectories_loaded}/{total_trajectories}')
                self.write_low_prob_traj_to_disk(traj_id)
                try:
                    header = self.sess.run(self.read_header_feature)
                except tf.errors.OutOfRangeError:
                    break

                nb_traj_tuples = header['trajectory_length']
                traj_id = int(np.copy(header['trajectory_id']))
                self.set_full_trajectory(traj_id)
            else:
                body, obs = self.sess.run([self.read_body_feature, self.get_obs])
                nb_traj_tuples -= 1
                cell_key_state = body['cell_key']
                reward = body['reward']
                obs = obs
                goal = body['goal']
                action = body['action']
                ge_reward = body['ge_reward']

                cell_key = self.cell_class()
                cell_key.__setstate__(tuple(cell_key_state))

                self.full_trajectories[traj_id].append((cell_key, reward, (obs, goal), action, ge_reward))
        print(f'Trajectories loaded: {trajectories_loaded}/{total_trajectories}')

    def dump(self, filename):
        writer = tf.python_io.TFRecordWriter(filename)
        for key in self.cell_trajectories:
            full_trajectory = self.get_full_trajectory(key)
            trajectory_id = tf.train.Feature(int64_list=tf.train.Int64List(value=[key]))
            trajectory_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(full_trajectory)]))
            feature_map = {
                'trajectory_id': trajectory_id,
                'trajectory_length': trajectory_length
            }
            combined_features = tf.train.Features(feature=feature_map)
            example = tf.train.Example(features=combined_features)
            writer.write(example.SerializeToString())

            for tp in full_trajectory:
                cell_key, reward, obs_and_goal, action, ge_reward = tp
                obs, goal = obs_and_goal
                cell_key_f = tf.train.Feature(int64_list=tf.train.Int64List(value=cell_key.__getstate__()))
                reward_f = tf.train.Feature(float_list=tf.train.FloatList(value=[reward]))
                obs_f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[obs.tostring()]))
                goal_f = tf.train.Feature(float_list=tf.train.FloatList(value=goal))
                action_f = tf.train.Feature(int64_list=tf.train.Int64List(value=[action]))
                ge_reward_f = tf.train.Feature(float_list=tf.train.FloatList(value=[ge_reward]))
                feature = {
                    'cell_key': cell_key_f,
                    'reward': reward_f,
                    'obs': obs_f,
                    'goal': goal_f,
                    'action': action_f,
                    'ge_reward': ge_reward_f,
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())
            self.write_low_prob_traj_to_disk(key)
        writer.close()

    def get_state(self):
        state = {'cell_trajectories': self.cell_trajectories,
                 'del_policy_new_cells': self.del_policy_new_cells,
                 'del_rand_new_cells': self.del_rand_new_cells,
                 'del_ret_new_cells': self.del_ret_new_cells}
        return state

    def set_state(self, state):
        self.cell_trajectories = state['cell_trajectories']
        self.del_policy_new_cells = state['del_policy_new_cells']
        self.del_rand_new_cells = state['del_rand_new_cells']
        self.del_ret_new_cells = state['del_ret_new_cells']

        # Create an updated trajectory fragment for every trajectory so that the information will be updated in
        # sub-processes
        for traj_id in self.cell_trajectories:
            fragment = copy.copy(self.cell_trajectories[traj_id])
            fragment.fragment_start = 0
            fragment.reference_count = 0
            self.updated_trajectory_fragments[traj_id] = fragment

    # Synchronization methods
    def get_info_to_sync(self):
        return self.updated_trajectory_fragments

    def sync_traj_start(self, trajectory_fragments: Dict[int, CellTrajectory]):
        for traj_key in trajectory_fragments:
            trajectory_fragment = trajectory_fragments[traj_key]
            if traj_key not in self.cell_trajectories:
                assert trajectory_fragment.fragment_start == 0, 'error in trajectory: ' + str(trajectory_fragment)
                assert trajectory_fragment.reference_count == 0, 'error in trajectory: ' + str(trajectory_fragment)
                self.cell_trajectories[traj_key] = trajectory_fragment
                self.increment_reference(traj_key)
                if trajectory_fragment.finished:
                    self.synced_trajectories.append(traj_key)
            else:
                assert trajectory_fragments[traj_key].fragment_start == len(self.cell_trajectories[traj_key].cell_ids)
                trajectory = self.cell_trajectories[traj_key]
                trajectory.score = trajectory_fragment.score
                trajectory.exp_strat = trajectory_fragment.exp_strat
                trajectory.exp_new_cells = trajectory_fragment.exp_new_cells
                trajectory.ret_new_cells = trajectory_fragment.ret_new_cells
                trajectory.total_actions = trajectory_fragment.total_actions
                trajectory.actions_per_cell[-1] = trajectory_fragment.actions_per_cell[-1]
                trajectory.cell_ids += trajectory_fragment.cell_ids[1:]
                trajectory.actions_per_cell += trajectory_fragment.actions_per_cell[1:]
                # i.e. If the trajectory is finished, and we didn't already know that the trajectory was finished
                if trajectory_fragment.finished and not trajectory.finished:
                    self.synced_trajectories.append(traj_key)
                    self.register_new_trajectory(traj_key)
                trajectory.finished = trajectory_fragment.finished

    def sync_traj_end(self):
        for traj_key in self.synced_trajectories:
            self.decrement_reference(traj_key)
        self.synced_trajectories = []

    def clear_info_to_sync(self):
        self.updated_trajectory_fragments = {}

    def finish_update(self):
        pass

    def start_update(self):
        self.clear_new_trajectories()

    def switch_trajectory(self, trajectory_id):
        if trajectory_id not in self.cell_trajectories:
            self._start_trajectory(trajectory_id)
        self.cell_trajectory_id = trajectory_id

    def _start_trajectory(self, trajectory_id):
        assert trajectory_id not in self.updated_trajectory_fragments, 'Traj. id ' + str(trajectory_id) + ' reused!'
        trajectory = CellTrajectory()
        trajectory.id = trajectory_id
        trajectory.tie_breaker = random.random()
        self.cell_trajectories[trajectory_id] = trajectory
        self.increment_reference(trajectory_id)
        self.cells_seen[trajectory_id] = set()

    def update_trajectory(self, cell_id: int, reward, obs, goal, action, ge_reward, cell_key, exp_strat, new_cell):
        # Get the current trajectory
        trajectory = self.cell_trajectories[self.cell_trajectory_id]
        assert not trajectory.finished, 'Finished trajectories should not be updated! Trajectory' + str(trajectory)

        # Get the current trajectory fragment
        if self.cell_trajectory_id not in self.updated_trajectory_fragments:
            trajectory_fragment = CellTrajectory()
            trajectory_fragment.id = self.cell_trajectory_id
            trajectory_fragment.fragment_start = 0
            trajectory_fragment.tie_breaker = trajectory.tie_breaker
            if len(trajectory.cell_ids) > 0:
                trajectory_fragment.cell_ids.append(trajectory.cell_ids[-1])
                trajectory_fragment.actions_per_cell.append(trajectory.actions_per_cell[-1])
                trajectory_fragment.fragment_start = len(trajectory.cell_ids)
            self.updated_trajectory_fragments[self.cell_trajectory_id] = trajectory_fragment
        else:
            trajectory_fragment = self.updated_trajectory_fragments[self.cell_trajectory_id]

        if cell_id != self.get_current_cell_id():
            # Update the trajectory
            trajectory.cell_ids.append(cell_id)
            trajectory.actions_per_cell.append(0)

            # Update the trajectory fragment
            trajectory_fragment.cell_ids.append(cell_id)
            trajectory_fragment.actions_per_cell.append(0)

        # Update the trajectory
        trajectory.actions_per_cell[-1] += 1
        trajectory.total_actions += 1
        trajectory.score += reward
        if exp_strat == global_const.EXP_STRAT_NONE:
            if new_cell:
                trajectory.ret_new_cells += 1
        else:
            trajectory.exp_strat = exp_strat
            if new_cell:
                trajectory.exp_new_cells += 1

        # Update the trajectory fragment
        trajectory_fragment.actions_per_cell[-1] = trajectory.actions_per_cell[-1]
        trajectory_fragment.total_actions = trajectory.total_actions
        trajectory_fragment.score = trajectory.score
        trajectory_fragment.exp_strat = trajectory.exp_strat
        trajectory_fragment.exp_new_cells = trajectory.exp_new_cells
        trajectory_fragment.ret_new_cells = trajectory.ret_new_cells

        if self.sil == 'sil' or self.sil == 'replay' or self.sil == 'nosil':
            if self.cell_trajectory_id not in self.full_trajectories:
                self.set_full_trajectory(self.cell_trajectory_id)
            self.full_trajectories[self.cell_trajectory_id].append((cell_key, reward, (obs, goal), action, ge_reward))

    def end_trajectory(self, frame):
        self.cell_trajectories[self.cell_trajectory_id].finished = True
        self.updated_trajectory_fragments[self.cell_trajectory_id].finished = True
        self.cell_trajectories[self.cell_trajectory_id].frame_finished = frame
        self.updated_trajectory_fragments[self.cell_trajectory_id].frame_finished = frame
        self.register_new_trajectory(self.cell_trajectory_id)
        self.decrement_reference(self.cell_trajectory_id)
        del self.cells_seen[self.cell_trajectory_id]

    def get_trajectory_score(self, trajectory_id, archive):
        trajectory = self.cell_trajectories[trajectory_id]
        smallest_reached = sys.maxsize
        nb_zero_reached = 0
        cell_keys = set([archive.cell_id_to_key_dict[cell_id] for cell_id in trajectory.cell_ids])
        for cell_key in cell_keys:
            cell_info = archive.archive[cell_key]
            if cell_info.nb_reached < smallest_reached:
                smallest_reached = cell_info.nb_reached
            if cell_info.nb_reached == 0:
                nb_zero_reached += 1
        return smallest_reached, nb_zero_reached

    # Reference counting methods
    def increment_reference(self, cell_trajectory_id):
        if cell_trajectory_id == self.empty_trajectory_id:
            return
        self.cell_trajectories[cell_trajectory_id].reference_count += 1

    def decrement_reference(self, cell_trajectory_id: int):
        if cell_trajectory_id == self.empty_trajectory_id:
            return
        self.cell_trajectories[cell_trajectory_id].reference_count -= 1
        if self.cell_trajectories[cell_trajectory_id].reference_count <= 0:
            assert self.cell_trajectories[cell_trajectory_id].finished, 'Unfinished trajectories should not be deleted!'
            self.del_ret_new_cells += self.cell_trajectories[cell_trajectory_id].ret_new_cells
            if self.cell_trajectories[cell_trajectory_id].exp_strat == global_const.EXP_STRAT_POLICY:
                self.del_policy_new_cells += self.cell_trajectories[cell_trajectory_id].exp_new_cells
            else:
                self.del_rand_new_cells += self.cell_trajectories[cell_trajectory_id].exp_new_cells
            del self.cell_trajectories[cell_trajectory_id]
            if cell_trajectory_id in self.full_trajectories:
                del self.full_trajectories[cell_trajectory_id]
            if self.has_full_trajectory_on_disk(cell_trajectory_id):
                try:
                    os.remove(self.get_full_trajectory_file_name(cell_trajectory_id))
                except FileNotFoundError:
                    pass
            if cell_trajectory_id in self.full_trajectory_info:
                del self.full_trajectory_info[cell_trajectory_id]

    # Convenience methods for retrieving information about the current trajectory
    def get_current_trajectory(self, cell_key_dict):
        return self.get_trajectory(self.cell_trajectory_id, -1, cell_key_dict)

    def get_current_trajectory_length(self):
        return len(self.cell_trajectories[self.cell_trajectory_id].cell_ids)

    def get_current_cell_id(self):
        if len(self.cell_trajectories[self.cell_trajectory_id].cell_ids) > 0:
            return self.cell_trajectories[self.cell_trajectory_id].cell_ids[-1]
        else:
            return -1

    def current_length(self):
        return self.cell_trajectories[self.cell_trajectory_id].total_actions

    def current_score(self):
        return self.cell_trajectories[self.cell_trajectory_id].score

    # Method for retrieving the cell trajectory
    def get_trajectory(self, cell_trajectory_id, trajectory_end, cell_key_dict):
        result = []
        if cell_trajectory_id == self.empty_trajectory_id:
            return result
        for cell_id, nb_actions in zip(self.cell_trajectories[cell_trajectory_id].cell_ids[0:trajectory_end],
                                       self.cell_trajectories[cell_trajectory_id].actions_per_cell[0:trajectory_end]):
            result.append((cell_key_dict[cell_id], nb_actions))
        return result

    def has_full_trajectory(self, traj_id):
        return traj_id in self.full_trajectory_info

    def has_full_trajectory_on_disk(self, traj_id):
        if traj_id in self.full_trajectory_info:
            return self.full_trajectory_info[traj_id].on_disk
        else:
            return False

    def set_full_trajectory(self, traj_id, traj_data=None):
        if traj_data is None:
            traj_data = []
        self.full_trajectory_info[traj_id] = FullTrajectoryInfo()
        self.full_trajectories[traj_id] = traj_data

    def get_full_trajectory(self, cell_trajectory_id, trajectory_end=-1):
        result = []
        if cell_trajectory_id == self.empty_trajectory_id:
            return result
        if cell_trajectory_id not in self.full_trajectory_info:
            raise KeyError(str(cell_trajectory_id) + ' not in trajectory manager')
        if self.full_trajectory_info[cell_trajectory_id].on_disk:
            self.read_full_trajectory_from_disk(cell_trajectory_id)
        if trajectory_end == -1:
            return self.full_trajectories[cell_trajectory_id]
        nb_of_frames = sum(self.cell_trajectories[cell_trajectory_id].actions_per_cell[0:trajectory_end])
        result = self.full_trajectories[cell_trajectory_id][0:nb_of_frames]
        return result

    def get_full_trajectory_file_name(self, cell_trajectory_id):
        file_name = str(cell_trajectory_id) + '_traj.pkl'
        path_to_file = os.path.join(self.temp_dir, file_name)
        return path_to_file

    def write_low_prob_traj_to_disk(self, traj_id):
        if traj_id == CellTrajectoryManager.empty_trajectory_id:
            return
        has_traj = self.has_full_trajectory(traj_id)
        traj_finished = self.cell_trajectories[traj_id].finished
        not_on_disk = not self.has_full_trajectory_on_disk(traj_id)
        if has_traj and traj_finished and not_on_disk:
            if traj_id in self.traj_prob_dict:
                low_prob = self.traj_prob_dict[traj_id] < self.low_prob_traj_tresh
                if low_prob:
                    self.write_full_trajectory_to_disk(traj_id)

    def write_full_trajectory_to_disk(self, cell_trajectory_id):
        full_trajectory = self.full_trajectories[cell_trajectory_id]
        with open(self.get_full_trajectory_file_name(cell_trajectory_id), 'wb') as fh:
            pickle.dump(len(full_trajectory), fh)
            for item in full_trajectory:
                pickle.dump(item, fh)
        del self.full_trajectories[cell_trajectory_id]
        self.full_trajectory_info[cell_trajectory_id].on_disk = True

    def read_full_trajectory_from_disk(self, cell_trajectory_id):
        with open(self.get_full_trajectory_file_name(cell_trajectory_id), 'rb') as fh:
            unpickle = pickle.Unpickler(fh)
            nb_items = unpickle.load()
            data = []
            for i in range(nb_items):
                data.append(unpickle.load())
        self.set_full_trajectory(cell_trajectory_id, data)
        self.full_trajectory_info[cell_trajectory_id].on_disk = False

    def __getstate__(self):
        return (self.cell_trajectories,
                self.full_trajectories,
                self.full_trajectory_info,
                self.traj_prob_dict,
                self.del_ret_new_cells,
                self.del_policy_new_cells,
                self.del_rand_new_cells,
                self.cell_trajectory_id,
                self.updated_trajectory_fragments,
                self.synced_trajectories,
                self.new_trajectories,
                self.cells_seen,
                self.sil,
                self.keep_new_trajectories,
                self.low_prob_traj_tresh,
                self.temp_dir)

    def __setstate__(self, state):
        (self.cell_trajectories,
         self.full_trajectories,
         self.full_trajectory_info,
         self.traj_prob_dict,
         self.del_ret_new_cells,
         self.del_policy_new_cells,
         self.del_rand_new_cells,
         self.cell_trajectory_id,
         self.updated_trajectory_fragments,
         self.synced_trajectories,
         self.new_trajectories,
         self.cells_seen,
         self.sil,
         self.keep_new_trajectories,
         self.low_prob_traj_tresh,
         self.temp_dir) = state

        # We do not serialize Tensorflow operations or sessions
        self.read_header_feature = None
        self.read_body_feature = None
        self.get_obs = None
        self.cell_class = None
        self.sess = None
