# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import horovod.tensorflow as hvd
from atari_reset.atari_reset.ppo import flatten_lists
import goexplore_py.mpi_support as mpi
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import goexplore_py.globals as global_const
import numpy as np
import time

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if isinstance(obj, np.ndarray) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += obj.nbytes
                need_referents.append(obj)
            elif not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


class Explore:
    def __init__(self, trajectory_gatherer=None, archive=None, sil='none', max_traj_candidates=1,
                 exchange_sil_traj='none'):
        self.archive = archive
        self.frames_compute = 0
        self.start = None
        self.cycles = 0
        self.trajectory_gatherer = trajectory_gatherer
        self.prev_selected_traj = None
        self.prev_selected_traj_len = 0
        self.sil: str = sil
        self.max_traj_candidates: int = max_traj_candidates
        self.exchange_sil_traj: str = exchange_sil_traj
        self.warm_up: bool = False

    def close(self):
        self.trajectory_gatherer.close()

    def init_cycle(self):
        # Add the initial cell to the archive
        if len(self.archive.archive) == 0:
            env_handle = self.trajectory_gatherer.env.get_envs()[0]
            env_handle.reset()
            cell_key = self.archive.get_cell_from_env(env_handle)
            self.archive.add_cell(cell_key)
            self.archive.updated_cells.add(cell_key)
            self.archive.cell_selector.cell_update(cell_key)
            self.archive.archive[cell_key].ret_discovered = global_const.EXP_STRAT_INIT
            self._sync_info()

        self.trajectory_gatherer.init_archive()
        self.start = time.time()

    def _sync_info(self):
        """
        Mostly put this code in its own method such that it will show up as its own category when profiling.

        @return: None
        """
        new_cell_trajectory_info = self.archive.cell_trajectory_manager.get_info_to_sync()
        new_archive_info = self.archive.get_info_to_sync()
        to_send = (new_cell_trajectory_info, new_archive_info)
        if hvd.size() > 1:
            to_send = mpi.COMM_WORLD.allgather(to_send)
        else:
            to_send = [to_send]

        self.trajectory_gatherer.env.recursive_call_method_ignore_return('update_archive', (to_send,))

        # Do not process our own information
        del to_send[hvd.rank()]
        self.archive.sync_info(to_send)

    def get_state(self):
        archive_state = self.archive.get_state()
        gatherer_state = self.trajectory_gatherer.get_state()
        state = {'archive_state': archive_state,
                 'gatherer_state': gatherer_state,
                 'frames_compute': self.frames_compute,
                 'cycles': self.cycles}
        return state

    def set_state(self, state):
        self.archive.set_state(state['archive_state'])
        self.trajectory_gatherer.set_state(state['gatherer_state'])
        self.frames_compute = state['frames_compute']
        self.cycles = state['cycles']

        # Send loaded information to sub-processes
        new_cell_trajectory_info = self.archive.cell_trajectory_manager.get_info_to_sync()
        new_archive_info = self.archive.get_info_to_sync()
        new_archive_info = (self.archive.cell_id_to_key_dict, new_archive_info[1], new_archive_info[2], set())
        to_send = [(new_cell_trajectory_info, new_archive_info)]
        self.trajectory_gatherer.env.recursive_call_method('update_archive', (to_send,))

        # Mark all loaded information as synced
        self.archive.clear_info_to_sync()
        self.archive.cell_trajectory_manager.clear_new_trajectories()

    def start_warm_up(self):
        self.warm_up = True
        self.trajectory_gatherer.warm_up = True

    def end_warm_up(self):
        self.warm_up = False
        self.trajectory_gatherer.warm_up = False

    def get_traj_owners(self, trajectory_ids):
        owned_by_me = []
        for traj_id in trajectory_ids:
            if self.archive.cell_trajectory_manager.has_full_trajectory(traj_id):
                owned_by_me.append(traj_id)
        owned_by_world = mpi.COMM_WORLD.allgather(owned_by_me)
        return owned_by_world

    def get_traj_owner(self, owned_by_world, traj_id):
        owner = None
        for i in range(len(owned_by_world)):
            rank = (i + hvd.rank()) % hvd.size()
            if traj_id in owned_by_world[rank]:
                owner = rank
                break
        assert owner is not None
        return owner

    def process_requests(self, requests, write_to_disk=False):
        ready_message = 'r'
        for traj_requester, traj_id, traj_owner in requests:
            if hvd.rank() == traj_owner and traj_requester is not None:
                # Wait until the receiver is ready to receive the message
                mpi.COMM_WORLD.recv(source=traj_requester, tag=2)
                full_trajectory = self.archive.cell_trajectory_manager.get_full_trajectory(traj_id)
                mpi.COMM_WORLD.send(full_trajectory, dest=traj_requester, tag=1)
            elif hvd.rank() == traj_requester and traj_owner is not None:
                # Signal to sender that we are ready to receive
                mpi.COMM_WORLD.send(ready_message, dest=traj_owner, tag=2)
                data = mpi.COMM_WORLD.recv(source=traj_owner, tag=1)
                self.archive.cell_trajectory_manager.set_full_trajectory(traj_id, data)
                if write_to_disk:
                    self.write_to_disk(traj_id)

    def write_to_disk(self, traj_id):
        self.archive.cell_trajectory_manager.write_low_prob_traj_to_disk(traj_id)

    def sync_before_checkpoint(self):
        if self.sil == 'sil' or self.sil == 'replay':
            # Let everyone in the world know who has which full trajectory
            owned_by_world = self.get_traj_owners(self.archive.cell_trajectory_manager.cell_trajectories)

            requests = []
            if hvd.rank() == 0:
                # Rank 0: figure out which trajectories you are missing
                owned_by_others = []
                for traj_id in self.archive.cell_trajectory_manager.cell_trajectories:
                    if not self.archive.cell_trajectory_manager.has_full_trajectory(traj_id):
                        owned_by_others.append(traj_id)

                # Rank 0: figure out who owns those trajectories
                owners = [self.get_traj_owner(owned_by_world, other_traj_id) for other_traj_id in owned_by_others]

                # Rank 0: construct a set of requests
                requests = [(hvd.rank(), traj_id, owner) for traj_id, owner in zip(owned_by_others, owners)]

            # Exchange requests
            requests = mpi.COMM_WORLD.allgather(requests)
            requests = flatten_lists(requests)
            self.process_requests(requests)

    def run_cycle(self):
        # Warm up cycles are not counted as cycles
        if not self.warm_up:
            self.cycles += 1

        mb_data = self.trajectory_gatherer.gather()

        # While synchronizing the number of frames processed across all environments would be ideal, it makes loading
        # checkpoints harder. Instead, we will provide an estimate based on the maximum number of frames that could
        # potentially be processed
        local_frames = self.trajectory_gatherer.processed_frames
        global_frames = mpi.COMM_WORLD.allgather(local_frames)
        prev_frames = self.frames_compute
        self.frames_compute += sum(global_frames)

        self.archive.frame_skip = self.trajectory_gatherer.frame_skip
        self.archive.frames = prev_frames + sum(global_frames[0:hvd.rank()])
        self.archive.update(mb_data,
                            self.trajectory_gatherer.return_goals_chosen,
                            self.trajectory_gatherer.return_goals_reached,
                            self.trajectory_gatherer.sub_goals,
                            self.trajectory_gatherer.ent_incs)

        self._sync_info()
        cell_selector = self.archive.cell_selector
        trajectory_manager = self.archive.cell_trajectory_manager
        trajectory_manager.traj_prob_dict = cell_selector.get_traj_probabilities_dict(self.archive.archive)

        if self.sil == 'sil' or self.sil == 'replay':
            traj_candidates = []

            # Select a trajectory to imitate based on the current cell-selection procedure
            key = self.archive.cell_selector.choose_cell_key(self.archive.archive)[0]

            cell_traj_id = self.archive.archive[key].cell_traj_id
            cell_traj_end = self.archive.archive[key].cell_traj_end
            score = self.archive.archive[key].score
            trajectory_len = self.archive.archive[key].trajectory_len
            traj_candidates.append((cell_traj_id, cell_traj_end, score, trajectory_len))

            # Let the world know which trajectory we have chosen for self-imitation learning
            selected_trajectories = mpi.COMM_WORLD.allgather(cell_traj_id)

            # Let the world know which of the selected trajectories I have stored locally
            owned_by_world = self.get_traj_owners(selected_trajectories)

            # Determine who has my trajectory
            owner = self.get_traj_owner(owned_by_world, cell_traj_id)

            # If we do not have our own trajectory, and our environment is ready for the next demonstration, make a
            # request for our trajectory
            env_zero = self.trajectory_gatherer.env.get_envs()[0]
            if owner != hvd.rank() and env_zero.recursive_getattr('sil_ready_for_next'):
                request = (hvd.rank(), cell_traj_id, owner)
            else:
                request = (hvd.rank(), None, None)

            requests = mpi.COMM_WORLD.allgather(request)

            # Exchange trajectories
            self.process_requests(requests)

            # If we still do not have a trajectory to imitate, stop imitation learning
            if len(traj_candidates) == 0:
                env_zero.recursive_call_method('set_sil_trajectory', [None, None])
                self.prev_selected_traj = None
                self.prev_selected_traj_len = 0
            else:
                selected_traj, cell_traj_end, selected_traj_score, selected_traj_len = traj_candidates[-1]
                if env_zero.recursive_getattr('sil_ready_for_next') and \
                        (self.prev_selected_traj != selected_traj or self.prev_selected_traj_len < selected_traj_len):
                    cell_trajectory = trajectory_manager.get_trajectory(selected_traj, -1,
                                                                        self.archive.cell_id_to_key_dict)
                    frame_trajectory = trajectory_manager.get_full_trajectory(cell_traj_id, cell_traj_end)
                    env_zero.recursive_call_method('set_sil_trajectory', (frame_trajectory, cell_trajectory))
                    self.prev_selected_traj = selected_traj
                    self.prev_selected_traj_len = selected_traj_len

            for traj_id in self.archive.cell_trajectory_manager.cell_trajectories:
                self.write_to_disk(traj_id)
