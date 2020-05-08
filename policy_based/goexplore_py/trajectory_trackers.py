# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from typing import Dict, Any, Callable, Optional


class TrajectoryTracker:
    def __init__(self, cell_reached: Optional[Callable[[Any, Any], bool]]):
        self.trajectory_index = 0
        self.cumulative_actions = []
        self.cell_reached = cell_reached
        self.cell_trajectory = []

    def get_default_goal(self):
        raise NotImplementedError('TrajectoryTracker needs to implement reset.')

    def reset(self, current_cell, cell_trajectory, final_goal):
        total_actions = 0
        self.cumulative_actions = []
        for cell, actions in cell_trajectory:
            total_actions += actions
            self.cumulative_actions.append(total_actions)
        self.trajectory_index = -1

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        raise NotImplementedError('TrajectoryTracker needs to implement step.')

    def get_steps(self, r_index=0):
        c_index = self.trajectory_index + r_index
        if len(self.cumulative_actions) == 0:
            return 0
        elif c_index < 0:
            return 0
        elif c_index < len(self.cumulative_actions):
            return self.cumulative_actions[c_index]
        else:
            return self.cumulative_actions[-1]


class DummyTrajectoryTracker(TrajectoryTracker):
    def get_default_goal(self):
        return 'final_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        super(DummyTrajectoryTracker, self).reset(current_cell, cell_trajectory, final_goal)
        return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        return final_goal, 0, False


class RewardOnlyTrajectoryTracker(TrajectoryTracker):
    def __init__(self):
        super(RewardOnlyTrajectoryTracker, self).__init__(None)
        self.trajectory_cells = set()

    def get_default_goal(self):
        return 'final_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        super(RewardOnlyTrajectoryTracker, self).reset(current_cell, cell_trajectory, final_goal)
        cells_only = [cell for cell, _ in cell_trajectory]
        self.trajectory_cells = set(cells_only)
        return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False
        if current_cell in self.trajectory_cells:
            reward = 1
            sub_goal_reached = True
            self.trajectory_cells.remove(current_cell)
        return final_goal, reward, sub_goal_reached


class PotentialRewardTrajectoryTracker(TrajectoryTracker):
    def __init__(self, discount, cell_reached):
        super(PotentialRewardTrajectoryTracker, self).__init__(cell_reached)
        self.discount: float = discount
        self.potentials: Optional[Dict[Any, float]] = None
        self.previous_potential: Optional[float] = None

    def other_cell_potential(self):
        return 0.0

    def get_default_goal(self):
        return 'final_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        super(PotentialRewardTrajectoryTracker, self).reset(current_cell, cell_trajectory, final_goal)
        super_cell_trajectory = []
        for cell, _ in cell_trajectory:
            for i in range(len(super_cell_trajectory)):
                if cell in super_cell_trajectory[i]:
                    for j in range(i+1, len(super_cell_trajectory)):
                        super_cell_trajectory[i] |= super_cell_trajectory[j]
                    del super_cell_trajectory[i+1:]
                    break
            else:
                super_cell_trajectory.append({cell})

        self.potentials = defaultdict(self.other_cell_potential)
        for i in range(len(super_cell_trajectory)):
            potential = pow(self.discount, (len(super_cell_trajectory)-1)-i)
            for cell in super_cell_trajectory[i]:
                self.potentials[cell] = potential

        self.previous_potential = self.potentials[current_cell]

        return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        new_potential = self.potentials[current_cell]
        reward = new_potential - self.previous_potential
        self.previous_potential = new_potential
        return final_goal, reward, False


class SequentialTrajectoryTracker(TrajectoryTracker):
    def __init__(self, cell_reached):
        super(SequentialTrajectoryTracker, self).__init__(cell_reached)
        self.sub_goal = None
        self.cell_trajectory = None

    def get_default_goal(self):
        return 'sub_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        super(SequentialTrajectoryTracker, self).reset(current_cell, cell_trajectory, final_goal)
        self.trajectory_index = 0
        self.cell_trajectory = cell_trajectory

        if self.trajectory_index < len(self.cell_trajectory):
            self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
            return self.sub_goal
        else:
            return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False
        if not self.trajectory_index < len(self.cell_trajectory):
            return final_goal, 0, False
        if self.cell_reached(current_cell, self.sub_goal) and self.trajectory_index + 1 < len(self.cell_trajectory):
            self.trajectory_index += 1
            self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
            reward = 1
            sub_goal_reached = True
        return self.sub_goal, reward, sub_goal_reached


class SoftTrajectoryTracker(TrajectoryTracker):
    def __init__(self, cell_reached):
        super(SoftTrajectoryTracker, self).__init__(cell_reached)
        self.sub_goal = None
        self.cell_trajectory = None
        self.window_size = 10

    def get_default_goal(self):
        return 'sub_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        super(SoftTrajectoryTracker, self).reset(current_cell, cell_trajectory, final_goal)
        self.trajectory_index = 0
        self.cell_trajectory = cell_trajectory

        if self.trajectory_index < len(self.cell_trajectory):
            self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
            return self.sub_goal
        else:
            return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False
        if not self.trajectory_index < len(self.cell_trajectory):
            return final_goal, 0, False

        steps_remaining = self.window_size
        i = -1
        while steps_remaining > 0 and (self.trajectory_index + i + 1) < len(self.cell_trajectory):
            i += 1
            steps_remaining -= self.cell_trajectory[self.trajectory_index + i][1]

        while i >= 0:
            if self.cell_reached(current_cell, self.cell_trajectory[self.trajectory_index + i][0]):
                self.trajectory_index = self.trajectory_index + i + 1
                reward = 1
                sub_goal_reached = True
                if self.trajectory_index < len(self.cell_trajectory):
                    self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
                else:
                    self.sub_goal = self.cell_trajectory[-1][0]
                break
            i -= 1
        return self.sub_goal, reward, sub_goal_reached


class SparseSoftTrajectoryTracker(TrajectoryTracker):
    def __init__(self, cell_reached, window_size):
        super(SparseSoftTrajectoryTracker, self).__init__(cell_reached)
        self.sub_goal = None
        self.cell_trajectory = None
        self.window_size = window_size

    def get_default_goal(self):
        return 'sub_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        super(SparseSoftTrajectoryTracker, self).reset(current_cell, cell_trajectory, final_goal)
        self.trajectory_index = 0
        self.cell_trajectory = cell_trajectory

        if self.trajectory_index < len(self.cell_trajectory):
            self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
            return self.sub_goal
        else:
            return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False

        if not self.trajectory_index < len(self.cell_trajectory):
            return final_goal, 0, False

        for i in reversed(range(min(self.window_size, len(self.cell_trajectory) - self.trajectory_index))):
            if self.cell_reached(current_cell, self.cell_trajectory[self.trajectory_index + i][0]):
                self.trajectory_index = self.trajectory_index + i + 1
                reward = 1
                sub_goal_reached = True
                if self.trajectory_index < len(self.cell_trajectory):
                    self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
                else:
                    self.sub_goal = self.cell_trajectory[-1][0]
                break
        return self.sub_goal, reward, sub_goal_reached


class DelayedSoftTrajectoryTracker(SparseSoftTrajectoryTracker):
    def __init__(self, cell_reached, window_size, delay):
        super(DelayedSoftTrajectoryTracker, self).__init__(cell_reached, window_size)
        self.time_in_soft = 0
        self.delay = delay

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False

        if not self.trajectory_index < len(self.cell_trajectory):
            return final_goal, 0, False

        # The highlighted cell is reached
        if self.cell_reached(current_cell, self.cell_trajectory[self.trajectory_index][0]):
            self.trajectory_index += 1
            reward = 1
            sub_goal_reached = True
            if self.trajectory_index < len(self.cell_trajectory):
                self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
            else:
                self.sub_goal = self.cell_trajectory[-1][0]
            self.time_in_soft = 0

        # Look through the soft-part of the trajectory
        in_soft_traj = False
        for i in reversed(range(0, min(self.window_size, len(self.cell_trajectory) - self.trajectory_index))):
            if self.cell_reached(current_cell, self.cell_trajectory[self.trajectory_index + i][0]):
                self.time_in_soft += 1
                in_soft_traj = True
                if self.time_in_soft > self.delay:
                    self.trajectory_index = self.trajectory_index + i + 1
                    sub_goal_reached = True
                    if self.trajectory_index < len(self.cell_trajectory):
                        self.sub_goal = self.cell_trajectory[self.trajectory_index][0]
                    else:
                        self.sub_goal = self.cell_trajectory[-1][0]
                    self.time_in_soft = 0
                break

        if not in_soft_traj:
            self.time_in_soft = 0
        return self.sub_goal, reward, sub_goal_reached


def get_super_cell_trajectory(cell_trajectory):
    super_cell_trajectory = []
    super_cell_actions = []
    for cell, actions in cell_trajectory:
        for i in range(len(super_cell_trajectory)):
            if cell in super_cell_trajectory[i]:
                for j in range(i + 1, len(super_cell_trajectory)):
                    super_cell_trajectory[i] += super_cell_trajectory[j]
                    super_cell_actions += super_cell_actions[j]
                del super_cell_trajectory[i + 1:]
                del super_cell_actions[i + 1:]
                break
        else:
            super_cell_trajectory.append([cell])
            super_cell_actions.append(actions)
    return super_cell_trajectory, super_cell_actions


class SparseTrajectoryTracker(TrajectoryTracker):
    """
    Provide the last cell of the super-cell as the target for the neural network, and only move the trajectory
    forward when this goal is reached. This makes the task completely clear for the neural network: at any point it is
    provided with a particular goal, and if it reaches that goal it gets a point, and another goal is revealed (what the
    next goal is will be is stochastic from the perspective of the neural network, as it does not "know" whether it is
    in the return state or the exploration state).
    """
    def __init__(self, cell_reached):
        super(SparseTrajectoryTracker, self).__init__(cell_reached)
        self.cell_trajectory = []
        self.sub_goal = None

    def get_default_goal(self):
        return 'sub_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        self.trajectory_index = 0
        self.cell_trajectory = []

        super_cell_trajectory, super_cell_actions = get_super_cell_trajectory(cell_trajectory)

        total_actions = 0
        self.cumulative_actions = []
        for actions in super_cell_actions:
            total_actions += actions
            self.cumulative_actions.append(total_actions)

        for super_cell in super_cell_trajectory:
            self.cell_trajectory.append(super_cell[-1])

        if self.trajectory_index < len(self.cell_trajectory):
            self.sub_goal = self.cell_trajectory[self.trajectory_index]
            return self.sub_goal
        else:
            return final_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False

        if not self.trajectory_index < len(self.cell_trajectory):
            return final_goal, 0, False
        if self.cell_reached(current_cell, self.sub_goal):
            self.trajectory_index += 1
            reward = 1
            sub_goal_reached = True
            if self.trajectory_index < len(self.cell_trajectory):
                self.sub_goal = self.cell_trajectory[self.trajectory_index]
            else:
                self.sub_goal = final_goal

        return self.sub_goal, reward, sub_goal_reached


class SuperCellTrajectoryTracker(TrajectoryTracker):
    """
    Provide the final cell of a super-cell to the network, but move the trajectory forward if the agent touches
    any of the future super-cells as well. This allows the agent to skip certain goals in the trajectory, but it also
    makes the problem partially observable, as the agent has no way of knowing where the future trajectory is (or even
    know whether there is a future trajectory). This can potentially be resolved by also providing the final goal to
    the network.
    """
    def __init__(self, cell_reached):
        super(SuperCellTrajectoryTracker, self).__init__(cell_reached)
        self.super_cell_trajectory = []
        self.sub_goal = None

    def get_default_goal(self):
        return 'final_goal_and_sub_goal'

    def reset(self, current_cell, cell_trajectory, final_goal):
        self.trajectory_index = 0
        self.super_cell_trajectory, super_cell_actions = get_super_cell_trajectory(cell_trajectory)

        total_actions = 0
        self.cumulative_actions = []
        for actions in super_cell_actions:
            total_actions += actions
            self.cumulative_actions.append(total_actions)

        if self.trajectory_index < len(self.super_cell_trajectory):
            super_cell = self.super_cell_trajectory[self.trajectory_index]
            self.sub_goal = super_cell[-1]
            if current_cell in super_cell:
                self.trajectory_index += 1
                if self.trajectory_index < len(self.super_cell_trajectory):
                    super_cell = self.super_cell_trajectory[self.trajectory_index]
                    self.sub_goal = super_cell[-1]
                else:
                    self.sub_goal = final_goal
        else:
            self.sub_goal = final_goal
        return self.sub_goal

    def step(self, current_cell, final_goal) -> [Any, float, bool]:
        reward = 0
        sub_goal_reached = False

        for i in range(self.trajectory_index, len(self.super_cell_trajectory)):
            if current_cell in self.super_cell_trajectory[i]:
                self.trajectory_index = i+1
                reward = 1
                sub_goal_reached = True
                break

        # We did not reach a new super-cell, continue towards the current sub-goal
        if reward == 0:
            pass
        # We reached a new super-cell, determine the next goal and return a reward
        elif self.trajectory_index < len(self.super_cell_trajectory):
            super_cell = self.super_cell_trajectory[self.trajectory_index]
            self.sub_goal = super_cell[-1]
        # We reached the final super-cell in our trajectory, set the final goal as our sub-goal and return a reward
        else:
            self.sub_goal = final_goal
        return self.sub_goal, reward, sub_goal_reached
