# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import deque
import horovod.tensorflow as hvd
from typing import List, Union, Any
import numpy as np

from goexplore_py.explorers import RepeatedRandomExplorer
from goexplore_py.data_classes import CellInfoStochastic
import goexplore_py.mpi_support as mpi
from atari_reset.atari_reset.ppo import flatten_lists, safemean


class StochasticGatherer:
    def __init__(self,
                 env,
                 nb_of_epochs: int,
                 learning_rate: float,
                 log_window_size: int = 100,
                 model=None,
                 freeze_network=False,
                 runner=None):

        # Parameters
        self.nb_of_epochs: int = nb_of_epochs
        self.learning_rate: float = learning_rate
        self.freeze_network: bool = freeze_network
        self.log_window_size: int = log_window_size
        self.warm_up: bool = False

        # Object handles
        self.model = model
        self.runner = runner
        self.env = env
        self.frame_skip = self.env.recursive_getattr('_skip')[0]

        # Persistent data (needs to be restored)
        self.ep_info_window = deque(maxlen=log_window_size)
        self.nb_of_episodes = 0

        # Temporary data (recalculated every iteration)
        self.nb_return_goals_reached: float = 0.0
        self.nb_return_goals_chosen: float = 0.0
        self.nb_exploration_goals_reached: float = 0.0
        self.nb_exploration_goals_chosen: float = 0.0
        self.return_goals_chosen: List[Any] = []
        self.return_goals_reached: List[bool] = []

        self.sub_goals: List[Any] = []
        self.ent_incs: List[int] = []
        self.reward_mean: float = 0.0
        self.length_mean: float = 0.0
        self.loss_values: List = []
        self.ep_infos_to_report: Union[List, deque] = []
        self.processed_frames: int = 0

    def gather(self):
        self.runner.run()

        if hasattr(self.runner, 'trunc_lst_mb_sil_valid'):
            diff = len(self.runner.trunc_lst_mb_sil_valid) - np.sum(self.runner.trunc_lst_mb_sil_valid)
            self.processed_frames = int(diff * self.frame_skip)
            sil_frames = self.runner.trunc_lst_mb_sil_valid
        else:
            self.processed_frames = self.runner.steps_taken * self.runner.nenv * self.frame_skip
            sil_frames = np.zeros_like(self.runner.trunc_lst_mb_dones)

        local_ep_infos = self.runner.epinfos
        if hvd.size() > 1:
            ep_infos = flatten_lists(mpi.COMM_WORLD.allgather(local_ep_infos))
        else:
            ep_infos = local_ep_infos

        self.ep_info_window.extend(ep_infos)
        if len(ep_infos) >= self.log_window_size:
            self.ep_infos_to_report = ep_infos
        else:
            self.ep_infos_to_report = self.ep_info_window
        self.nb_return_goals_reached = sum([ei['reached'] for ei in self.ep_infos_to_report])
        self.nb_return_goals_chosen = len(self.ep_infos_to_report)
        self.nb_exploration_goals_reached = sum([ei['nb_exploration_goals_reached'] for ei in self.ep_infos_to_report])
        self.nb_exploration_goals_chosen = sum([ei['nb_exploration_goals_chosen'] for ei in self.ep_infos_to_report])
        self.reward_mean = safemean([ei['r'] for ei in self.ep_infos_to_report])
        self.length_mean = safemean([ei['l'] for ei in self.ep_infos_to_report])
        self.nb_of_episodes += len(ep_infos)
        self.return_goals_chosen = [ei['goal_chosen'] for ei in local_ep_infos]
        self.return_goals_reached = [ei['reached'] for ei in local_ep_infos]
        self.sub_goals = [ei['sub_goal'] for ei in local_ep_infos]
        self.ent_incs = [ei['inc_ent'] for ei in local_ep_infos]

        # We do not update the network during the warm up period
        if not self.freeze_network and not self.warm_up:
            self._train()

        return (self.runner.ar_mb_cells,
                self.runner.ar_mb_game_reward,
                self.runner.trunc_lst_mb_trajectory_ids,
                self.runner.trunc_lst_mb_dones,
                self.runner.trunc_mb_obs,
                self.runner.trunc_mb_goals,
                self.runner.trunc_mb_actions,
                self.runner.trunc_lst_mb_rewards,
                sil_frames,
                self.runner.ar_mb_ret_strat,
                self.runner.ar_mb_traj_index,
                self.runner.ar_mb_traj_len)

    def broadcast_archive(self, archive):
        self.env.set_archive(archive)

    def broadcast_selector(self, selector):
        self.env.set_selector(selector)

    def update_archive(self, new_archive_information):
        self.env.recursive_call_method('update_archive', new_archive_information)

    def init_archive(self):
        self.runner.init_obs()

    def _train(self):
        self.loss_values = []
        for i in range(self.nb_of_epochs):
            self.loss_values.append(self.model.train_from_runner(self.learning_rate, self.runner))

    def save_model(self, filename):
        self.model.save(filename)

    def get_state(self):
        state = {'ep_info_window': self.ep_info_window,
                 'nb_of_episodes': self.nb_of_episodes}
        return state

    def set_state(self, state):
        self.ep_info_window = state['ep_info_window']
        self.nb_of_episodes = state['nb_of_episodes']

    def close(self):
        if self.env:
            self.env.close()


class DeterministicGatherer:
    def __init__(self,
                 env,
                 log_window_size=100,
                 num_steps=160,
                 explorer=None):
        self.ep_info_window = deque(maxlen=log_window_size)
        self.env = env

        # Data to store
        self.ep_infos_to_report: Union[List, deque] = []
        self.nb_return_goals_reached: float = 0.0
        self.nb_return_goals_chosen: float = 0.0
        self.nb_exploration_goals_reached: float = 0.0
        self.nb_exploration_goals_chosen: float = 0.0
        self.return_goals_chosen: List[Any] = []
        self.return_goals_info_chosen: List[CellInfoStochastic] = []
        self.exploration_goals_chosen: List[Any] = []
        self.return_goals_reached: List[bool] = []
        self.exploration_goals_reached: List[bool] = []
        self.restored: List[bool] = []
        self.reward_mean: float = 0.0
        self.length_mean: float = 0.0
        self.nb_of_episodes: int = 0

        self.num_steps = num_steps
        self.explorer = explorer
        if self.explorer is None:
            self.explorer = RepeatedRandomExplorer(20)

    def gather(self):
        ep_infos = []
        for _ in range(self.num_steps):
            actions = [self.explorer.get_action(env) for env in self.env.get_envs()]
            obs_and_goals, rewards, dones, infos = self.env.step(actions)
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info:
                    ep_infos.append(maybe_ep_info)

        if hvd.size() > 1:
            ep_infos = flatten_lists(mpi.COMM_WORLD.allgather(ep_infos))

        trajectories = [ei['trajectory'] for ei in ep_infos]

        self.ep_info_window.extend(ep_infos)
        if len(ep_infos) >= 100:
            self.ep_infos_to_report = ep_infos
        else:
            self.ep_infos_to_report = self.ep_info_window

        self.nb_return_goals_reached = sum([ei['nb_return_goals_reached'] for ei in self.ep_infos_to_report])
        self.nb_return_goals_chosen = sum([ei['nb_return_goals_chosen'] for ei in self.ep_infos_to_report])
        self.nb_exploration_goals_reached = sum([ei['nb_exploration_goals_reached'] for ei in self.ep_infos_to_report])
        self.nb_exploration_goals_chosen = sum([ei['nb_exploration_goals_chosen'] for ei in self.ep_infos_to_report])
        self.return_goals_chosen = flatten_lists([ei['return_goals_chosen'] for ei in ep_infos])
        self.return_goals_info_chosen = flatten_lists([ei['return_goals_info_chosen'] for ei in ep_infos])
        self.exploration_goals_chosen = flatten_lists([ei['exploration_goals_chosen'] for ei in ep_infos])
        self.return_goals_reached = flatten_lists([ei['return_goals_reached'] for ei in ep_infos])
        self.exploration_goals_reached = flatten_lists([ei['exploration_goals_reached'] for ei in ep_infos])
        self.restored = flatten_lists([ei['restored'] for ei in ep_infos])
        self.reward_mean = safemean([ei['r'] for ei in self.ep_infos_to_report])
        self.length_mean = safemean([ei['l'] for ei in self.ep_infos_to_report])
        self.nb_of_episodes += len(trajectories)

        return trajectories

    def broadcast_archive(self, archive):
        self.env.set_archive(archive)

    def broadcast_selector(self, selector):
        self.env.set_selector(selector)

    def init_archive(self):
        archive = self.env.init_archive()
        self.env.reset()
        return archive

    def save_state(self, filename):
        pass

    def close(self):
        if self.env:
            self.env.close()
