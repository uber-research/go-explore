# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from typing import List, Any
from gym import spaces


class AbstractGoalRepresentation:
    def get_goal_space(self):
        raise NotImplementedError('get_goal_space needs to be implemented.')

    def get(self, current_cell: Any, final_goal: Any, sub_goal: Any):
        raise NotImplementedError('get needs to be implemented.')


class FlatGoalRep(AbstractGoalRepresentation):
    def __init__(self, rep_type: str, rel_final_goal: bool, rel_sub_goal: bool, length_data: Any):
        self.rep_type = rep_type
        self.rel_final_goal = rel_final_goal
        self.rel_sub_goal = rel_sub_goal
        self.length_data = length_data

        if self.rep_type == 'final_goal':
            self.total_length = self._get_length(self.rel_final_goal, length_data)
        elif self.rep_type == 'sub_goal':
            self.total_length = self._get_length(self.rel_sub_goal, length_data)
        elif self.rep_type == 'final_goal_and_sub_goal':
            self.total_length = (self._get_length(self.rel_final_goal, length_data) +
                                 self._get_length(self.rel_sub_goal, length_data))
        else:
            raise NotImplementedError('Unknown representation type: ' + self.rep_type)

    def get_goal_space(self):
        raise NotImplementedError('get_goal_space needs to be implemented.')

    def get(self, current_cell: Any, final_goal: Any, sub_goal: Any):
        if self.rep_type == 'final_goal':
            return self._get_goal_rep(final_goal, current_cell, self.rel_final_goal)
        elif self.rep_type == 'sub_goal':
            return self._get_goal_rep(sub_goal, current_cell, self.rel_sub_goal)
        elif self.rep_type == 'final_goal_and_sub_goal':
            final_rep = self._get_goal_rep(final_goal, current_cell, self.rel_final_goal)
            sub_rep = self._get_goal_rep(sub_goal, current_cell, self.rel_sub_goal)
            return np.concatenate((sub_rep, final_rep))
        else:
            raise NotImplementedError('Unknown representation type: ' + self.rep_type)

    def _get_length(self, relative: bool, length_data: Any):
        raise NotImplementedError('_get_length needs to be implemented.')

    def _get_goal_rep(self, goal: Any, current_cell: Any, relative: bool):
        raise NotImplementedError('_get_goal_rep needs to be implemented.')


class ScaledGoalRep(FlatGoalRep):
    """
    Takes the array from a representation and divides it by normalizing constants.
    """
    def __init__(self, rep_type: str, rel_final_goal: bool, rel_sub_goal: bool, rep_length, norm_const=None,
                 off_const=None):
        super().__init__(rep_type, rel_final_goal, rel_sub_goal, rep_length)

        self.normalizing_constants = np.ones(rep_length)
        self.offset_constants = np.zeros(rep_length)

        if norm_const:
            self.normalizing_constants = norm_const
        if off_const:
            self.offset_constants = off_const

    def get_goal_space(self):
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.total_length,), dtype=np.float32)

    def _get_goal_rep(self, goal: Any, current_cell: Any, relative: bool):
        goal_rep = np.cast[np.float32](goal.as_array())
        goal_rep /= self.normalizing_constants
        goal_rep += self.offset_constants
        if relative:
            current_rep = np.cast[np.float32](current_cell.as_array())
            current_rep /= self.normalizing_constants
            current_rep += self.offset_constants
            goal_rep -= current_rep
        return goal_rep

    def _get_length(self, relative: bool, rep_length):
        return rep_length


class GoalRepData:
    def __init__(self, rep_lengths: List[int], goal: Any, current_loc: Any, relative: bool):
        self.rep_lengths = rep_lengths
        self.goal_array = goal.as_array()
        self.current_array = None
        self.relative = relative
        if self.relative:
            self.current_array = current_loc.as_array()

    def get_index(self, i):
        max_value = self.rep_lengths[i] - 1
        if self.relative:
            feature_index = max_value + int(self.goal_array[i]) - int(self.current_array[i])
            if feature_index < 0:
                feature_index = 0
            elif feature_index > max_value*2 - 1:
                feature_index = max_value*2 - 1
        else:
            feature_index = int(self.goal_array[i])
            if feature_index < 0:
                feature_index = 0
            elif feature_index > max_value:
                feature_index = max_value
        return feature_index


class OneHotGoalRep(FlatGoalRep):
    """
    Takes the array from a representation and discretizes each value into a one-hot vector.
    """
    def __init__(self, rep_type: str, rel_final_goal: bool, rel_sub_goal: bool, rep_lengths: List[int]):
        super().__init__(rep_type, rel_final_goal, rel_sub_goal, rep_lengths)

    def get_goal_space(self):
        return spaces.Box(low=0, high=1, shape=(self.total_length,), dtype=np.float32)

    def _get_goal_rep(self, goal: Any, current_loc: Any, relative: bool):
        cur_index = 0
        length = self._get_length(relative, self.length_data)
        goal_rep = np.zeros(length)
        goal_rep_data = GoalRepData(self.length_data, goal, current_loc, relative)
        for i in range(len(self.length_data)):
            feature_index = goal_rep_data.get_index(i)
            goal_rep[cur_index + feature_index] = 1.0
            cur_index += self.length_data[i]
        return goal_rep

    def _get_length(self, relative: bool, rep_lengths):
        if relative:
            return (sum(rep_lengths) * 2) - 1
        else:
            return sum(rep_lengths)


class PosFilterGoalRep(AbstractGoalRepresentation):
    """
    Takes the x and y attributes from a representation and turns it into an image sheet that can be stacked as a filter.
    """

    def get(self, current_cell: Any, final_goal: Any, sub_goal: Any):
        if self.rep_type == 'final_goal':
            return self._get_goal_rep(final_goal)
        elif self.rep_type == 'sub_goal':
            return self._get_goal_rep(sub_goal)
        elif self.rep_type == 'final_goal_and_sub_goal':
            final_rep = self._get_goal_rep(final_goal)
            sub_rep = self._get_goal_rep(sub_goal)
            return np.concatenate((sub_rep, final_rep))
        else:
            raise NotImplementedError('Unknown representation type: ' + self.rep_type)

    def __init__(self, shape, x_res, y_res, x_offset=0, y_offset=0, goal_value=1, norm_const=None, pos_only=False,
                 rep_type='final_goal'):
        self.shape = shape
        self.x_res = x_res
        self.y_res = y_res
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.goal_value = goal_value
        self.norm_const = norm_const
        self.rep_type = rep_type
        if norm_const is None:
            self.norm_const = np.ones(shape[-1] - 1)
        self.pos_only = pos_only

    def get_goal_space(self):
        return spaces.Box(low=0, high=255, shape=self.shape, dtype=np.float32)

    def _get_goal_rep(self, goal: Any):
        goal_rep = np.zeros(self.shape)
        x = self.x_offset + goal.get_x() * self.x_res
        y = self.y_offset + goal.get_y() * self.y_res
        goal_rep[x:x + self.x_res, y:y + self.y_res, 0] = self.goal_value
        if not self.pos_only:
            non_pos_features = goal.non_pos_as_array()
            for i, feature in enumerate(non_pos_features):
                goal_rep[:, :, i] = (feature / self.norm_const[i]) * 255

        return goal_rep
