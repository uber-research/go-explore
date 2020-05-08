# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from gym import spaces
import multiprocessing as mp
from typing import Any, List, Tuple, Callable, Dict, Optional
import random
import copy
import cv2
from PIL import Image
from atari_reset.atari_reset.wrappers import MyWrapper, worker, CloudpickleWrapper, VecWrapper, VideoWriter
from goexplore_py.data_classes import CellInfoStochastic
from goexplore_py.goal_representations import AbstractGoalRepresentation
from goexplore_py.trajectory_trackers import TrajectoryTracker
import goexplore_py.utils as utils
import goexplore_py.globals as global_const
import logging
logger = logging.getLogger(__name__)


class GoalConVecFrameStack(VecWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        super(GoalConVecFrameStack, self).__init__(venv)
        self.nstack = nstack
        observation_space = venv.observation_space
        # We are transforming the observation space from unstacked (i.e. the number of RGB channels) to
        # stacked (i.e. the number of RGB channels times the number of stacks).
        ob_low = np.repeat(observation_space.low, self.nstack, axis=-1)
        ob_high = np.repeat(observation_space.high, self.nstack, axis=-1)
        self.stacked_obs = np.zeros((venv.num_envs,) + ob_low.shape, observation_space.dtype)
        self._observation_space = spaces.Box(low=ob_low, high=ob_high)
        self._action_space = venv.action_space
        self._goal_space = venv.goal_space
        self.single_frame_obs_space = observation_space

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs_and_goals, rews, news, infos = self.venv.step(vac)
        obs, goals = obs_and_goals
        self.stacked_obs = np.roll(self.stacked_obs, shift=-obs.shape[-1], axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[..., -obs.shape[-1]:] = obs
        obs_and_goals = (self.stacked_obs, goals)
        return obs_and_goals, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs_and_goals = self.venv.reset()
        obs, goals = obs_and_goals
        self.stacked_obs[...] = 0
        self.stacked_obs[..., -obs.shape[-1]:] = obs
        obs_and_goals = (self.stacked_obs, goals)
        return obs_and_goals

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def goal_space(self):
        return self._goal_space


class GoalConVecGoalStack(VecWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, goal_rep):
        super(GoalConVecGoalStack, self).__init__(venv)
        observation_space = venv.observation_space
        old_shape = observation_space.low.shape
        new_shape = old_shape[0:-1] + (old_shape[-1] + goal_rep.shape[-1], )
        filter_shape = old_shape[0:-1] + (goal_rep.shape[-1],)
        sheet = np.zeros(filter_shape, observation_space.low.dtype)
        ob_low = np.concatenate([observation_space.low, sheet], axis=-1)
        ob_high = np.concatenate([observation_space.high, sheet], axis=-1)
        self.stacked_obs = np.zeros((venv.num_envs,) + new_shape, observation_space.low.dtype)
        self._observation_space = spaces.Box(low=ob_low, high=ob_high)
        self._action_space = venv.action_space
        self._goal_space = venv.goal_space

    def step(self, vac):
        obs_and_goals, rews, news, infos = self.venv.step(vac)
        obs, goals = obs_and_goals
        obs_and_goals = np.concatenate([obs, goals], axis=-1)
        return (obs_and_goals, goals), rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs_and_goals = self.venv.reset()
        obs, goals = obs_and_goals
        obs_and_goals = np.concatenate([obs, goals], axis=-1)
        return obs_and_goals, goals

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def goal_space(self):
        return self._goal_space

    def close(self):
        self.venv.close()

    @property
    def num_envs(self):
        return self.venv.num_envs


def get_neighbor(env, pos, offset, x_range, y_range):
    x = pos.x + offset[0]
    y = pos.y + offset[1]
    room = pos.room
    room_x, room_y = env.recursive_getattr('get_room_xy')(room)
    if x < x_range[0]:
        x = x_range[1]
        room_x -= 1
    elif x > x_range[1]:
        x = x_range[0]
        room_x += 1
    elif y < y_range[0]:
        y = y_range[1]
        room_y -= 1
    elif y > y_range[1]:
        y = y_range[0]
        room_y += 1
    if env.recursive_getattr('get_room_out_of_bounds')(room_x, room_y):
        return None
    room = env.recursive_getattr('get_room_from_xy')(room_x, room_y)
    if room == -1:
        return None
    new_pos = copy.copy(pos)
    new_pos.room = room
    new_pos.x = x
    new_pos.y = y
    return new_pos


class GoalExplorer:
    def __init__(self, random_exp_prob, random_explorer):
        self.exploration_strategy = global_const.EXP_STRAT_NONE
        self.random_exp_prob = random_exp_prob
        self.random_explorer = random_explorer

    def on_reset(self):
        self.exploration_strategy = global_const.EXP_STRAT_NONE

    def on_return(self):
        self.random_explorer.init_seed()
        if random.random() < self.random_exp_prob:
            self.exploration_strategy = global_const.EXP_STRAT_RAND
        else:
            self.exploration_strategy = global_const.EXP_STRAT_POLICY

    def overwrite_action(self, env, policy_action):
        if self.exploration_strategy == global_const.EXP_STRAT_RAND:
            return self.random_explorer.get_action(env)
        else:
            return policy_action

    def choose(self, go_explore_env):
        raise NotImplementedError('GoalExplorers need to implement a choose method.')


class DomKnowNeighborGoalExplorer(GoalExplorer):
    def __init__(self, x_res, y_res, random_exp_prob, random_explorer):
        super(DomKnowNeighborGoalExplorer, self).__init__(random_exp_prob, random_explorer)
        self.x_res = x_res
        self.y_res = y_res

    def choose(self, go_explore_env):
        width = go_explore_env.env.recursive_getattr('screen_width') * go_explore_env.env.recursive_getattr('x_repeat')
        height = go_explore_env.env.recursive_getattr('screen_width')
        max_cell_x = int((width - (self.x_res / 2)) / self.x_res)
        max_cell_y = int((height - (self.y_res / 2)) / self.y_res)
        x_range = (0, max_cell_x)
        y_range = (0, max_cell_y)
        possible_neighbors = []
        unknown_neighbors = []
        for offset in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            possible_neighbor = get_neighbor(go_explore_env.env,
                                             go_explore_env.last_reached_cell,
                                             offset,
                                             x_range,
                                             y_range)
            if possible_neighbor is not None:
                if (possible_neighbor not in go_explore_env.archive.archive and
                        possible_neighbor not in go_explore_env.locally_explored):
                    unknown_neighbors.append(possible_neighbor)
                possible_neighbors.append(possible_neighbor)
        if len(unknown_neighbors) > 0 and random.random() > 0.9:
            target_cell = random.choice(unknown_neighbors)
        elif len(possible_neighbors) > 0 and random.random() > 0.75:
            target_cell = random.choice(possible_neighbors)
        else:
            target_cell = go_explore_env.select_cell_from_archive()
            go_explore_env.last_reached_cell = target_cell
        return target_cell


class EntropyManager:
    def __init__(self,
                 inc_ent_fac: float,
                 ret_inc_ent_thresh: int,
                 expl_inc_ent_thresh: int,
                 entropy_strategy: str,
                 ent_inc_power: float,
                 ret_inc_ent_fac: float,
                 expl_ent_reset: str,
                 legacy_entropy: bool
                 ):
        #: Specifies how to increase entropy (if at all). Currently implemented are the 'fixed_increase' strategy, which
        #: increase the entropy after a threshold is reached and the 'dynamic_increase' strategy, which slowly increases
        #: entropy after a threshold is reached based on the number of steps taking beyond this threshold. Also
        #: implemented is the 'none' strategy, which means entropy is never increased.
        self.entropy_strategy: str = entropy_strategy
        #: The factor by which to increase the entropy, either when the threshold is reached ('fixed_increase' entropy
        #: strategy), or per step after the beyond the threshold ('dynamic_increase' entropy strategy).
        self.inc_ent_fac: float = inc_ent_fac
        #: A grace period on top of the estimated number of steps towards the current goal after which entropy will be
        #: increased.
        self.ret_inc_ent_thresh: int = ret_inc_ent_thresh
        #: The number of steps of exploration towards a particular goal after which entropy will be increased.
        self.expl_inc_ent_thresh: int = expl_inc_ent_thresh
        #: The estimated length of the trajectory is multiplied by this factor to determine the threshold after which
        #: entropy will be increased. Setting it > 1 means that the policy will get a grace period with respect to the
        #: expected number of steps that scales with the total expected length. Setting it to 0 means the length of the
        #: trajectory will be ignored.
        self.ret_inc_ent_fac: float = ret_inc_ent_fac
        #: The power by which the scaled number of steps beyond the threshold is raised before it is passed as the
        #: entropy multiplier to the policy. Is only used in the 'dynamic_increase' entropy strategy.
        self.ent_inc_power: float = ent_inc_power

        self.expl_ent_reset: str = expl_ent_reset

        self.entropy_cells: Dict[Any, Tuple[Any, float]] = {}
        self.ent_end_cell: Any = None
        self.ent_fac: float = 1.0

        self.legacy_entropy: bool = legacy_entropy

    def on_reset(self):
        self.entropy_cells = {}
        self.ent_end_cell: Any = None
        self.ent_fac: float = 1.0

    def get_return_entropy(self, env):
        if env.current_cell in self.entropy_cells:
            self.ent_end_cell, self.ent_fac = self.entropy_cells[env.current_cell]
            del self.entropy_cells[env.current_cell]

        if self.ent_end_cell is not None:
            if isinstance(self.ent_end_cell, int):
                self.ent_end_cell -= 1
                if self.ent_end_cell <= 0:
                    self.ent_end_cell = None
                    self.ent_fac = 1.0
                return self.ent_fac
            elif env.current_cell == self.ent_end_cell:
                self.ent_end_cell = None
                self.ent_fac = 1.0
            else:
                return self.ent_fac

        if not (self.ret_inc_ent_fac == 0 and self.ret_inc_ent_thresh == -1):
            if self.legacy_entropy:
                deadline = env.goal_cell_info.trajectory_len * self.ret_inc_ent_fac + self.ret_inc_ent_thresh
                return self._get_inc_entropy(env.actions_to_goal, deadline)
            else:
                estimate_to_current_cell = env.steps_to_current - env.steps_to_previous
                deadline = estimate_to_current_cell * self.ret_inc_ent_fac + self.ret_inc_ent_thresh
                return self._get_inc_entropy(env.actions_to_sub_goal, deadline)
        return 1

    def get_exploration_entropy(self, env):
        if not self.expl_inc_ent_thresh == -1:
            if self.expl_ent_reset == 'on_new_cell':
                return self._get_inc_entropy(env.expl_steps_since_new_cell, self.expl_inc_ent_thresh)
            elif self.expl_ent_reset == 'on_goal_reached':
                return self._get_inc_entropy(env.actions_to_goal, self.expl_inc_ent_thresh)
        return 1

    def _get_inc_entropy(self, actions_taken, deadline):
        result = 1.0
        if self.entropy_strategy == 'fixed_increase':
            if actions_taken > deadline:
                result = self.inc_ent_fac
        elif self.entropy_strategy == 'none':
            result = 1.0
        elif self.entropy_strategy == 'dynamic_increase':
            if actions_taken > deadline:
                overdue = actions_taken - deadline
                result = 1 + ((overdue * self.inc_ent_fac) ** self.ent_inc_power)
        else:
            raise NotImplementedError('Unknown entropy strategy:', self.entropy_strategy)
        return result


class GoalConGoExploreEnv(MyWrapper):
    """
    Keeps track of the cells and cell representation
    """
    def __init__(self,
                 env: Any,
                 archive: Any,
                 goal_representation: AbstractGoalRepresentation,
                 done_on_return: bool,
                 video_writer: VideoWriter,
                 goal_explorer: GoalExplorer,
                 trajectory_tracker: TrajectoryTracker,
                 entropy_manager: EntropyManager,
                 max_exploration_steps: int,
                 on_done_reward: float,
                 no_exploration_gradients: bool,
                 game_reward_factor: float,
                 goal_reward_factor: float,
                 clip_game_reward: bool,
                 clip_range: Tuple[float, float],
                 max_actions_to_goal: int,
                 max_actions_to_new_cell: int,
                 cell_reached: Callable[[Any, Any], bool],
                 cell_selection_modifier: str,
                 traj_modifier: str,
                 fail_ent_inc: str,
                 final_goal_reward: float):
        super(GoalConGoExploreEnv, self).__init__(env)

        # Classes provided to the environment
        #: The archive from which goals should be chosen.
        #: Also includes a selector, which determines how goals are chosen from the archive.
        self.archive: Any = archive
        #: How to represent the goal to the neural network (e.g. onehot, raw, as a filter, etc.)
        self.goal_representation: AbstractGoalRepresentation = goal_representation
        #: How do we track cell trajectories in order to return to a state
        self.trajectory_tracker: TrajectoryTracker = trajectory_tracker
        #: How to select goals when exploring
        self.goal_explorer: GoalExplorer = goal_explorer
        #: When provided, a pointer to a video writer, which record the agent.
        #: The only reason this pointer is provided is because the video writer needs information about which goals are
        #: currently chosen, and this class holds that information.
        self.video_writer: VideoWriter = video_writer

        self.entropy_manager: EntropyManager = entropy_manager

        # Options provided to the environment
        #: Whether the episode should end as soon as we have returned to the indicated cell.
        #: Only relevant for debugging and experimental purposes, as it means no exploration will be performed.
        self.done_on_return: bool = done_on_return
        #: The maximum number of steps to explore for before choosing a different exploration goal
        self.max_exploration_steps: int = max_exploration_steps

        #: Reward provided for finishing an episode. Should generally be zero or negative, to discourage the agent from
        #: ending the episode.
        self.on_done_reward: float = on_done_reward
        #: Whether we should use exploration steps for updating our policy
        self.no_exploration_gradients: bool = no_exploration_gradients
        #: The factor by which to scale the game reward
        self.game_reward_factor: float = game_reward_factor
        #: The factor by which to scale the goal reward
        self.goal_reward_factor: float = goal_reward_factor
        #: Whether to clip the game reward between -1 and 1
        self.clip_game_reward: bool = clip_game_reward
        #: The range to which to clip if clipping is enabled
        self.clip_range: Tuple[float, float] = clip_range
        #: The maximum number of actions the agent gets to reach a chosen goal
        self.max_actions_to_goal: int = max_actions_to_goal
        #: The maximum number of actions the agent gets to reach a new cell
        self.max_actions_to_new_cell: int = max_actions_to_new_cell
        #: Whether to modify the cell we select
        self.cell_selection_modifier: str = cell_selection_modifier
        #: Whether to modify the trajectory towards the cell we select
        self.traj_modifier: str = traj_modifier
        #: Whether an how to increase entropy near high-failure cells
        self.fail_ent_inc: str = fail_ent_inc
        #: Function which specifies whether a particular cell has been reached or not
        self.cell_reached: Callable[[Any, Any], bool] = cell_reached
        #: Reward obtained for reaching the final goal
        self.final_goal_reward: float = final_goal_reward

        # Data tracked for reporting
        self.nb_return_goals_reached: int = -1
        self.nb_exploration_goals_reached: int = -1
        self.nb_return_goals_chosen: int = -1
        self.nb_exploration_goals_chosen: int = -1
        self.return_goals_reached: List[bool] = []
        self.exploration_goals_reached: List[bool] = []
        self.return_goals_chosen: List[Any] = []
        self.return_goals_info_chosen: List[CellInfoStochastic] = []
        self.exploration_goals_chosen: List[Any] = []
        self.restored: List[bool] = []

        # Data stored only so it can be reported when recursive_getattr is called
        self.goal_space: Any = goal_representation.get_goal_space()

        # Data tracked for the correct functioning of the environment
        #: Total number of actions taken in this episode
        self.action_nr: int = -1
        #: The number of actions taken towards the current final goal (reset whenever a final goal is reached)
        #: Return goals and exploration goals count as final goals, but sub goals do not
        self.actions_to_goal: int = -1
        #: The number of actions taken towards the current sub goal
        self.actions_to_sub_goal: int = -1
        #: Total number of actions taken during the exploration phase of the current episode
        self.exploration_steps: int = -1
        #: Number of actions taken in the current exploration phase since the last time we discovered a new cell
        self.expl_steps_since_new_cell: int = -1

        self.score: int = -1
        self.goal_cell_rep: Any = None
        self.goal_cell_info: Optional[CellInfoStochastic] = None
        self.current_cell: Any = None
        self.last_reached_cell: Any = None
        self.sub_goal_cell_rep: Any = None
        self.returning: bool = True
        self.returned_on_reset: bool = False
        self.locally_explored: set = set()
        self.steps_to_previous: int = 0
        self.steps_to_current: int = 0
        self.info: Dict[str, Any] = {}
        self.total_reward: float = 0

    def set_archive(self, archive):
        assert isinstance(archive, dict)
        self.archive.archive = archive

    def debug_get_traj(self):
        return self.archive.cell_trajectory_manager.cell_trajectories

    def debug_get_archive(self):
        return self.archive.archive

    def update_archive(self, info):
        self.archive.sync_info(info)

    def set_selector(self, selector):
        self.archive.cell_selector = selector

    def _choose_return_goal(self):
        archive = self.archive.archive
        if len(archive) == 0:
            return
        self.actions_to_goal = 0
        goal_cell_rep = self.archive.cell_selector.choose_cell_key(archive)[0]
        prev_goal_cell_rep = None
        if self.cell_selection_modifier == 'prev' or self.traj_modifier == 'prev':
            # Experimental code: instead of always going to the selected cell, sometimes go to cells before the selected
            # cell, in the hopes of finding a better trajectory to the target cell.
            temp_goal_cell_info = archive[goal_cell_rep]
            trajectory = self.archive.cell_trajectory_manager.get_trajectory(temp_goal_cell_info.cell_traj_id,
                                                                             temp_goal_cell_info.cell_traj_end,
                                                                             self.archive.cell_id_to_key_dict)
            if len(trajectory) > 0:
                seen_cells = set()
                unique_trajectory = []
                total_time_spend = 0
                for cell, time_spend in reversed(trajectory):
                    total_time_spend += time_spend
                    if cell not in seen_cells:
                        unique_trajectory.append((cell, total_time_spend))
                        seen_cells.add(cell)

                offset = np.random.geometric(0.5) - 1
                while offset >= len(unique_trajectory):
                    offset = np.random.geometric(0.5) - 1
                prev_goal_cell_rep = unique_trajectory[offset]
            else:
                prev_goal_cell_rep = (goal_cell_rep, 0)

        if self.cell_selection_modifier == 'prev':
            self.goal_cell_rep = prev_goal_cell_rep[0]
        else:
            self.goal_cell_rep = goal_cell_rep
        self.goal_cell_info = archive[self.goal_cell_rep]
        if self.traj_modifier == 'prev' and prev_goal_cell_rep[0] != goal_cell_rep:
            temp_goal_cell_info = archive[prev_goal_cell_rep[0]]
            trajectory = self.archive.cell_trajectory_manager.get_trajectory(temp_goal_cell_info.cell_traj_id,
                                                                             temp_goal_cell_info.cell_traj_end,
                                                                             self.archive.cell_id_to_key_dict)
            final_cell = trajectory[-1]
            trajectory.pop(-1)
            trajectory.append((final_cell[0], prev_goal_cell_rep[1]))
            chosen = self.archive.archive[goal_cell_rep].nb_chosen
            self.entropy_manager.entropy_cells[final_cell[0]] = goal_cell_rep, (chosen * 0.1) ** 2
            trajectory.append((goal_cell_rep, 1))
        else:
            trajectory = self.archive.cell_trajectory_manager.get_trajectory(self.goal_cell_info.cell_traj_id,
                                                                             self.goal_cell_info.cell_traj_end,
                                                                             self.archive.cell_id_to_key_dict)

        if self.fail_ent_inc == 'time' or self.fail_ent_inc == 'cell':
            for i, (cell_key, time_spend) in enumerate(trajectory):
                failed = self.archive.archive[cell_key].nb_sub_goal_failed
                nb_failures_above_thresh = self.archive.archive[cell_key].nb_failures_above_thresh
                if failed > self.archive.max_failed * self.archive.failed_threshold:
                    offset = np.random.geometric(0.5) - 1
                    while i - offset < 0:
                        offset = np.random.geometric(0.5) - 1
                    if offset > 0:
                        if self.fail_ent_inc == 'time':
                            end_con = np.random.randint(1, 20)
                        else:
                            end_con = cell_key
                        ent_cell = trajectory[i - offset][0]
                        self.entropy_manager.entropy_cells[ent_cell] = end_con, 1 + (nb_failures_above_thresh * 0.01)

        self.returning = True
        self.nb_return_goals_chosen += 1
        self.return_goals_chosen.append(self.goal_cell_rep)
        self.return_goals_info_chosen.append(self.goal_cell_info)
        self.return_goals_reached.append(False)
        self.restored.append(False)
        self.sub_goal_cell_rep = self.trajectory_tracker.reset(self.current_cell,
                                                               trajectory,
                                                               self.goal_cell_rep)
        self.steps_to_previous = 0
        self.steps_to_current = 0
        if self.cell_reached(self.current_cell, self.goal_cell_rep):
            self._return_success()
            self._choose_exploration_goal()
            self.returned_on_reset = True

    def _choose_exploration_goal(self):
        self.actions_to_goal = 0
        self.exploration_steps = 0
        self.goal_cell_rep = self.goal_explorer.choose(self)
        if self.goal_cell_rep in self.archive.archive:
            self.goal_cell_info = self.archive.archive[self.goal_cell_rep]
        else:
            self.goal_cell_info = self.archive.get_new_cell_info()
        self.returning = False
        self.nb_exploration_goals_chosen += 1
        self.exploration_goals_chosen.append(self.goal_cell_rep)
        self.exploration_goals_reached.append(False)

    def select_cell_from_archive(self):
        archive = self.archive.archive
        chosen_cell_key = self.archive.cell_selector.choose_cell_key(archive)[0]
        return chosen_cell_key

    def _return_success(self):
        self.nb_return_goals_reached += 1
        self.return_goals_reached[-1] = True
        self.last_reached_cell = self.current_cell
        self.goal_explorer.on_return()

    def _exploration_success(self):
        self.nb_exploration_goals_reached += 1
        self.exploration_goals_reached[-1] = True
        self.last_reached_cell = self.current_cell

    def get_current_cell(self):
        cells = self.archive.get_cells(self.env)
        return cells[self.archive.get_name()]

    def _get_nn_goal_rep(self):
        # Return return information
        if self.goal_cell_rep is None and self.sub_goal_cell_rep is None:
            return None
        goal = self.goal_representation.get(self.current_cell, self.goal_cell_rep, self.sub_goal_cell_rep)
        return goal

    def step(self, action: int):
        # Clear cache
        self.archive.clear_cache()

        # Overwrite action if exploring
        if not self.returning:
            action = self.goal_explorer.overwrite_action(self, action)

        # Take step
        obs, game_reward, done, self.info = self.env.step(action)

        # Update trajectory information
        self.actions_to_goal += 1
        self.action_nr += 1
        self.score += game_reward
        self.current_cell = self.get_current_cell()

        # Check if goal is reached and choose a new goal
        if self.returning:
            self.actions_to_sub_goal += 1
            traj_goals = self.trajectory_tracker.step(self.current_cell, self.goal_cell_rep)
            self.sub_goal_cell_rep, reached_goal_reward, sub_goal_reached = traj_goals
            if sub_goal_reached:
                self.actions_to_sub_goal = 0
            if self.steps_to_current != self.trajectory_tracker.get_steps(-1):
                self.steps_to_previous = self.steps_to_current
                self.steps_to_current = self.trajectory_tracker.get_steps(-1)
            self.info['increase_entropy'] = self.entropy_manager.get_return_entropy(self)
            self.info['exp_strat'] = self.goal_explorer.exploration_strategy
            self.info['traj_index'] = self.trajectory_tracker.trajectory_index
            self.info['traj_len'] = len(self.trajectory_tracker.cell_trajectory)
            steps_difference = self.steps_to_current - self.steps_to_previous
            if self.max_actions_to_goal >= 0 and self.actions_to_sub_goal > steps_difference + self.max_actions_to_goal:
                done = True
        else:
            self.sub_goal_cell_rep = self.goal_cell_rep
            reached_goal_reward = 0
            self.exploration_steps += 1
            self.expl_steps_since_new_cell += 1
            self.info['increase_entropy'] = self.entropy_manager.get_exploration_entropy(self)
            if self.no_exploration_gradients:
                self.info['replay_reset.invalid_transition'] = True
            self.info['overwritten_action'] = action
            self.info['exp_strat'] = self.goal_explorer.exploration_strategy
            if (self.max_actions_to_new_cell >= 0) and (self.expl_steps_since_new_cell > self.max_actions_to_new_cell):
                done = True

        # Logic for when we have reached our current goal
        if self.cell_reached(self.current_cell, self.goal_cell_rep):
            reached_goal_reward = self.final_goal_reward
            if self.returning:
                self._return_success()
            else:
                self._exploration_success()

            if self.done_on_return:
                done = True
            else:
                self._choose_exploration_goal()
        elif self.exploration_steps >= self.max_exploration_steps:
            self._choose_exploration_goal()

        # Update our local archive
        if self.current_cell not in self.archive.archive and self.current_cell not in self.locally_explored:
            self.locally_explored.add(self.current_cell)
            self.expl_steps_since_new_cell = 0

        # There exists a rare case where we successfully returned right after a reset.
        # If done_on_return is True, this means we are done on reset, but because of how the problem is implemented,
        # we have to at least take one step before we can report that we are done. This if statement captures that case.
        if self.done_on_return and self.returned_on_reset:
            done = True

        # Process the end of an episode
        if done:
            reached_goal_reward += self.on_done_reward
            ep_info = {'l': self.action_nr,
                       'r': self.score,
                       'nb_exploration_goals_reached': self.nb_exploration_goals_reached,
                       'nb_exploration_goals_chosen': self.nb_exploration_goals_chosen,
                       'goal_chosen': self.return_goals_chosen[-1],
                       'reached': self.return_goals_reached[-1],
                       'sub_goal': self.sub_goal_cell_rep,
                       'inc_ent': self.entropy_manager.ent_end_cell is not None,
                       }
            self.info['episode'] = ep_info

        # New information that needs to be passed every step
        self.info['cell'] = self.current_cell
        self.info['game_reward'] = game_reward

        goal = self._get_nn_goal_rep()
        obs_and_goal = (obs, goal)

        game_reward = game_reward * self.game_reward_factor
        if self.clip_game_reward:
            game_reward = utils.clip(game_reward, self.clip_range[0], self.clip_range[1])
        self.total_reward = game_reward + reached_goal_reward * self.goal_reward_factor

        if self.video_writer:
            self.video_writer.add_frame()

        return obs_and_goal, self.total_reward, done, self.info

    def reset(self):
        # Clear cache
        self.archive.clear_cache()

        # Reset environment
        obs = self.env.reset()

        # Reset trajectory information
        self.action_nr = 0
        self.score = 0
        self.nb_return_goals_reached = 0
        self.nb_exploration_goals_reached = 0
        self.nb_return_goals_chosen = 0
        self.nb_exploration_goals_chosen = 0
        self.return_goals_chosen = []
        self.return_goals_info_chosen = []
        self.return_goals_reached = []
        self.exploration_goals_chosen = []
        self.exploration_goals_reached = []
        self.restored = []
        self.returned_on_reset = False
        self.expl_steps_since_new_cell = 0
        self.actions_to_sub_goal = 0
        self.info = {}
        self.total_reward = 0

        self.goal_explorer.on_reset()
        self.entropy_manager.on_reset()

        # Record the cell we are currently in
        self.current_cell = self.get_current_cell()

        # Choose a goal
        self._choose_return_goal()

        # Return return information
        goal = self._get_nn_goal_rep()
        obs_and_goal = (obs, goal)
        if self.video_writer:
            self.video_writer.start_video()
            self.video_writer.add_frame()
        return obs_and_goal


class RemoteEnv(object):
    def __init__(self, remote):
        self.remote = remote
        self.waiting = False
        self.remote.send(('get_spaces', None))
        self.action_space, self.observation_space = self._recv(self.remote)
        self.remote.send(('recursive_getattr', 'goal_space'))
        self.goal_space = self._recv(self.remote)

    def step(self, action):
        self.remote.send(('step', action))
        result = self._recv(self.remote)
        return result

    def step_async(self, action):
        self.remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        result = self._recv(self.remote)
        self.waiting = False
        return result

    def reset(self):
        self.remote.send(('reset', None))
        obs = self._recv(self.remote)
        return obs

    def reset_task(self):
        self.remote.send(('reset_task', None))
        obs = self._recv(self.remote)
        return obs

    def get_history(self, nsteps):
        self.remote.send(('get_history', nsteps))
        obs = self._recv(self.remote)
        return obs

    def recursive_getattr(self, name):
        self.remote.send(('recursive_getattr', name))
        obs = self._recv(self.remote)
        return obs

    def recursive_setattr(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            self.remote.send(('recursive_setattr', (name, value)))

    def recursive_call_method(self, name, arguments=()):
        if hasattr(self, name):
            return getattr(self, name)(*arguments)
        else:
            message = (name, arguments)
            self.remote.send(('recursive_call_method', message))
            return self._recv(self.remote)

    def recursive_call_method_ignore_return(self, name, arguments=()):
        if hasattr(self, name):
            getattr(self, name)(*arguments)
        else:
            message = (name, arguments)
            self.remote.send(('recursive_call_method_ignore_return', message))

    def decrement_starting_point(self, n, idx):
        self.remote.send(('decrement_starting_point', (n, idx)))

    def init_archive(self):
        self.remote.send(('init_archive', None))
        obs = self._recv(self.remote)
        return obs

    def set_archive(self, archive):
        self.remote.send(('set_archive', archive))

    def set_selector(self, selector):
        self.remote.send(('set_selector', selector))

    def close(self):
        self.remote.send(('close', None))

    def get_current_cell(self):
        self.remote.send(('get_current_cell', None))
        return self._recv(self.remote)

    def _recv(self, remote):
        result = remote.recv()
        if isinstance(result, Exception):
            raise result
        return result


class GoalConSubprocVecEnv(object):
    def __init__(self, env_fns, start_method):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        mp_context = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[mp_context.Pipe(duplex=True) for _ in range(nenvs)])
        self.ps = [mp_context.Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)), daemon=True)
                   for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()
        # From this moment on, we have to close the environment in order to let the program end
        logger.debug(f'[master] sending command: get_spaces')

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self._recv(self.remotes[0])
        self.remotes[0].send(('recursive_getattr', 'goal_space'))
        self.goal_space = self._recv(self.remotes[0])
        self.waiting = False

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [self._recv(remote) for remote in self.remotes]
        obs_and_goals, rews, dones, infos = zip(*results)

        obs, goals = zip(*obs_and_goals)
        obs_and_goals = np.stack(obs), np.stack(goals)

        return obs_and_goals, np.stack(rews), np.stack(dones), infos

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [self._recv(remote) for remote in self.remotes]
        self.waiting = False
        obs_and_goals, rews, dones, infos = zip(*results)

        obs, goals = obs_and_goals
        obs_and_goals = np.stack(obs), np.stack(goals)

        return obs_and_goals, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_and_goals = [self._recv(remote) for remote in self.remotes]

        obs, goals = zip(*obs_and_goals)
        obs_and_goals = np.stack(obs), np.stack(goals)

        return obs_and_goals

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        obs_and_goals = [self._recv(remote) for remote in self.remotes]

        obs, goals = zip(*obs_and_goals)
        obs_and_goals = np.stack(obs), np.stack(goals)

        return obs_and_goals

    def get_history(self, nsteps):
        for remote in self.remotes:
            remote.send(('get_history', nsteps))
        results = [self._recv(remote) for remote in self.remotes]
        obs_and_goals, acts, dones = zip(*results)
        obs, goals = obs_and_goals
        obs_and_goals = np.stack(obs), np.stack(goals)
        acts = np.stack(acts)
        dones = np.stack(dones)
        return obs_and_goals, acts, dones

    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            for remote in self.remotes:
                remote.send(('recursive_getattr', name))
            return [self._recv(remote) for remote in self.remotes]

    def recursive_setattr(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            for remote in self.remotes:
                remote.send(('recursive_setattr', (name, value)))

    def recursive_call_method(self, name, arguments=()):
        message = (name, arguments)
        for remote in self.remotes:
            remote.send(('recursive_call_method', message))
        response = [self._recv(remote) for remote in self.remotes]
        return response

    def recursive_call_method_ignore_return(self, name, arguments=()):
        message = (name, arguments)
        for remote in self.remotes:
            remote.send(('recursive_call_method_ignore_return', message))

    def decrement_starting_point(self, n, idx):
        for remote in self.remotes:
            remote.send(('decrement_starting_point', (n, idx)))

    def init_archive(self):
        self.remotes[0].send(('init_archive', None))
        archive = self._recv(self.remotes[0])
        self.set_archive(archive.archive)
        self.set_selector(archive.cell_selector)
        return archive

    def set_archive(self, archive):
        for remote in self.remotes:
            remote.send(('set_archive', archive))

    def set_selector(self, selector):
        for remote in self.remotes:
            remote.send(('set_selector', selector))

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)

    def get_envs(self):
        return [RemoteEnv(remote) for remote in self.remotes]

    def _recv(self, remote):
        result = remote.recv()
        if isinstance(result, Exception):
            raise result
        return result


class SquareGreyFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        MyWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1), dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        obs = np.array(Image.fromarray(obs).resize((self.res, self.res),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape((self.res, self.res, 1))

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class RectGreyFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 1)
        self.net_res = (self.res[1], self.res[0], self.res[2])
        self.observation_space = spaces.Box(low=0, high=255, shape=self.net_res, dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return np.expand_dims(obs, -1)

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class RectColorFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        self.net_res = (self.res[1], self.res[0], self.res[2])
        self.observation_space = spaces.Box(low=0, high=255, shape=self.net_res, dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class RectColorFrameWithBug(MyWrapper):
    """
    This is from the OpenAI implementation, but it seems to contain a bug which incorrectly aligns columns and rows,
    causing pixels to get scrambled.
    """
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.res, dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape(self.res)

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class SilEnv(MyWrapper):
    def __init__(self, env, trajectory_tracker, goal_representation, gamma, sil_invalid):
        MyWrapper.__init__(self, env)
        nb_channels = env.observation_space.shape[-1]
        if nb_channels == 1:
            self.image_format = cv2.IMREAD_GRAYSCALE
        elif nb_channels == 3:
            self.image_format = cv2.IMREAD_COLOR
        else:
            raise Exception('Unsupported number of channels:' + str(nb_channels))
        self.sil_on = False
        self.sil_initial_action = None
        self.sil_initial_observation = None
        self.sil_observations = []
        self.sil_rewards = []
        self.sil_actions = []
        self.sil_cells = []
        self.sil_game_rewards = []
        self.sil_steps = 0
        self.sil_action = 0
        self.next_trajectory = None
        self.next_cell_trajectory = None
        self.sil_ready_for_next = True
        self.trajectory_tracker = trajectory_tracker
        self.goal_representation = goal_representation
        self.gamma = gamma
        self.sil_invalid = sil_invalid

    def step(self, *args, **kwargs):
        if not self.sil_on:
            obs_and_goal, reward, done, info = self.env.step(*args, **kwargs)
            return obs_and_goal, reward, done, info
        else:
            current_step = self.sil_steps
            self.sil_steps += 1
            done = self.sil_steps == len(self.sil_actions)
            self.sil_action = self.sil_actions[current_step]

            value = self.get_value(self.sil_rewards[current_step:])

            info = {'cell': self.sil_cells[current_step],
                    'game_reward': self.sil_game_rewards[current_step],
                    'sil_action': self.sil_actions[current_step],
                    'sil_value': value,
                    'replay_reset.invalid_transition': self.sil_invalid}
            return self.decompress(self.sil_observations[current_step]), self.sil_rewards[current_step], done, info

    def decompress(self, obs_and_goal):
        obs, goal = obs_and_goal
        obs = cv2.imdecode(obs, self.image_format)
        if self.image_format == cv2.IMREAD_GRAYSCALE:
            obs = np.expand_dims(obs, axis=-1)
        return obs, goal

    def get_value(self, rewards):
        if rewards is None:
            return 0.0
        cur_discount = 1.0
        total = 0.0
        for r in rewards:
            total += cur_discount * r
            cur_discount *= self.gamma
        return total

    def reset(self):
        self.sil_ready_for_next = True
        if self.next_trajectory is None:
            self.sil_on = False
            return self.env.reset()
        else:
            self.sil_on = True
            self.sil_observations = []
            self.sil_rewards = []
            self.sil_actions = []
            self.sil_cells = []
            self.sil_game_rewards = []
            self.sil_steps = 0

            # Because all trajectories end in death (and death states are not interesting in MR) we imitate the
            # trajectory until the second to last state
            goal_cell_rep = self.next_trajectory[-2][0]

            self.trajectory_tracker.reset(self.next_trajectory[0][0], self.next_cell_trajectory, goal_cell_rep)

            break_next = False
            for i, data in enumerate(self.next_trajectory):
                cell_key, reward, obs_and_goal, action, ge_reward = data
                obs = obs_and_goal[0]

                sub_goal_cell_rep, reached_goal_reward, sub_goal_reached = self.trajectory_tracker.step(cell_key,
                                                                                                        goal_cell_rep)

                goal = self.goal_representation.get(cell_key, goal_cell_rep, sub_goal_cell_rep)

                obs_and_goal = (obs, goal)

                if i == 0:
                    self.sil_initial_observation = self.decompress(obs_and_goal)
                    self.sil_initial_action = action
                else:
                    self.sil_observations.append(obs_and_goal)
                    self.sil_actions.append(action)
                self.sil_rewards.append(reached_goal_reward)
                self.sil_cells.append(cell_key)
                self.sil_game_rewards.append(reward)

                # Since there is no exploration phase when imitation learning, we can terminate the episode as soon as
                # the final goal is reached (even if did this not happen in the original trajectory).
                if break_next:
                    break
                if cell_key == goal_cell_rep:
                    break_next = True

            self.sil_action = self.sil_initial_action
            return self.sil_initial_observation

    def set_sil_trajectory(self, trajectory, cell_trajectory):
        # Do not try to imitate trajectories of length 1, because they will be truncated to length 0, and throw an index
        # out of range exception.
        if trajectory is None or len(trajectory) <= 1:
            self.next_trajectory = None
            self.next_cell_trajectory = None
            self.sil_ready_for_next = True
        else:
            self.next_trajectory = trajectory
            self.next_cell_trajectory = cell_trajectory
            self.sil_ready_for_next = False
