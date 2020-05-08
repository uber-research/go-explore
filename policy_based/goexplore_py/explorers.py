# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import os
import numpy as np
from .globals import get_action_meaning, get_trajectory


class RandomExplorer:
    def init_seed(self):
        pass

    def get_action(self, env):
        return random.randint(0, env.action_space.n - 1)

    def __repr__(self):
        return 'RandomExplorer()'


class RepeatedRandomExplorer:
    def __init__(self, mean_repeat):
        self.mean_repeat = mean_repeat
        self.action = 0
        self.remaining = 0

    def init_seed(self):
        self.remaining = 0

    def get_action(self, env):
        if self.remaining <= 0:
            self.action = random.randint(0, env.action_space.n - 1)
            # Note, this is equivalent to selecting an action and then repeating it
            # with some probability.
            self.remaining = np.random.geometric(1 / self.mean_repeat)
        self.remaining -= 1
        return self.action

    def __repr__(self):
        return f'repeat-{self.mean_repeat}'


class ReplayTrajectoryExplorer:
    def __init__(self, prev_idxs, actions):
        self.prev_idxs = prev_idxs
        self.actions = actions
        self.trajectory = []
        self.action_index = 0
        self.current_goal = None

    def init_seed(self):
        pass

    def get_action(self, env):
        goal_rep = env.recursive_getattr('goal_cell_rep')
        current_cell = env.get_current_cell()
        # We have reached the end of our trajectory, a new goal should have been chosen
        if self.action_index >= len(self.trajectory):
            goal = env.recursive_getattr('goal_cell_info')
            print('Selected goal:', goal_rep)
            print('Previous goal:', self.current_goal)
            if goal_rep == self.current_goal:
                print("ERROR: The same goal was selected twice in a row.")
                raise Exception('The same goal was selected twice in a row.')
            self.current_goal = goal_rep
            self.trajectory = get_trajectory(self.prev_idxs, self.actions, goal.traj_last)
            if goal.real_traj is not None:
                assert goal.real_traj == self.trajectory
            if goal.trajectory_len is not -1:
                assert len(self.trajectory) == goal.trajectory_len
            self.action_index = 0
        elif goal_rep != self.current_goal:
            print("ERROR: New goal selected before trajectory to previous goal was finished.")
            print("Full trajectory was:", self.trajectory)
            print('process id:', os.getpid())
            print("Which is:", [get_action_meaning(a) for a in self.trajectory])
            raise Exception('New goal selected before trajectory to previous goal was finished.')
        if len(self.trajectory) > 0:
            action = self.trajectory[self.action_index]
            self.action_index += 1
        else:
            action = 0
        print('In cell:', current_cell, 'Playing action:', self.action_index-1, action, get_action_meaning(action))
        return action


class RepeatedRandomExplorerRobot:
    def __init__(self, mean_repeat=10):
        self.mean_repeat = mean_repeat
        self.action = 0
        self.remaining = 0

    def init_seed(self):
        self.remaining = 0

    def get_action(self, env):
        if self.remaining <= 0:
            self.action = env.action_space.sample()
            # Note, this is equivalent to selecting an action and then repeating it
            # with some probability.
            self.remaining = np.random.geometric(1 / self.mean_repeat)
        self.remaining -= 1
        return self.action

    def __repr__(self):
        return f'repeat-{self.mean_repeat}'


class DoNothingExplorer:
    def init_seed(self):
        pass

    def get_action(self, *_args):
        return 0
