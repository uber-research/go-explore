# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings as _warnings
import copy
from typing import List, Any
from collections import deque
import sys

try:
    from dataclasses import dataclass, field as datafield

    def copyfield(data):
        return datafield(default_factory=lambda: copy.deepcopy(data))
except ModuleNotFoundError:
    _warnings.warn('dataclasses not found. To get it, use Python 3.7 or pip install dataclasses')


@dataclass
class GridDimension:
    attr: str
    div: int


@dataclass
class CellInfoDeterministic:
    #: The score of the last accepted trajectory to this cell
    score: int = -float('inf')
    #: Number of trajectories that included this cell
    nb_seen: int = 0
    #: The number of times this cell was chosen as the cell to explore from
    nb_chosen: int = 0
    #: The number of times this cell was chosen since it was last updated
    nb_chosen_since_update: int = 0
    #: The number of times this cell was chosen since it last resulted in discovering a new cell
    nb_chosen_since_to_new: int = 0
    #: The number of times this cell was chosen since it last resulted in updating any cell
    nb_chosen_since_to_update: int = 0
    #: The number of actions that had this cell as the resulting state (i.e. all frames spend in this cell)
    nb_actions: int = 0
    #: The number of times this cell was chosen to explore towards
    nb_chosen_for_exploration: int = 0
    #: The number of times this cell was reached when chosen to explore towards
    nb_reached_for_exploration: int = 0
    #: Length of the trajectory
    trajectory_len: int = float('inf')
    #: Saved restore state. In a purely deterministic environment,
    #: this allows us to fast-forward to the end state instead
    #: of replaying.
    restore: Any = None
    #: Sliding window for calculating our success rate of reaching different cells
    reached: deque = copyfield(deque(maxlen=100))
    #: List of cells that we went through to reach this cell
    cell_traj: List[Any] = copyfield([])
    exact_pos = None
    real_cell = None
    traj_last = None
    real_traj: List[int] = None

    @property
    def nb_reached(self):
        return sum(self.reached)


@dataclass
class CellInfoStochastic:
    # Basic information used by Go-Explore to determine when to update a cell
    #: The score of the last accepted trajectory to this cell
    score: int = -sys.maxsize
    #: Length of the trajectory
    trajectory_len: int = sys.maxsize

    # Together, these determine the trajectory the lead to this cell.
    # Necessary in order to follow cell trajectories, as well as for self-imitation learning.
    #: The identifier of the trajectory leading to this cell
    cell_traj_id: int = -1
    #: The index of the last cell of the cell trajectory
    cell_traj_end: int = -1

    # Optional information for determining whether cells are discovered more quickly over time.
    #: Whether this cell was initially discovered while returning to a cell (rather than while exploring from a cell)
    ret_discovered: int = 0
    #: At which frame was this cell discovered
    frame: int = -1
    #: The trajectory id of the first trajectory to find this cell
    first_cell_traj_id: int = -1
    #: How far along the trajectory was this cell discovered (if it was discovered while returning)
    traj_disc: int = 0
    #: What was the full length (in cells) of the trajectory being followed when this cell was discovered
    #: (if it was discovered while returning)
    total_traj_length: int = 0
    #: Flag to control the update-on-reset process
    should_reset: bool = False

    # Optional information that can be used to take special actions near cells with high failure-to-reach rates.
    #: The number of times the agent has failed to reach this cell when it was presented to the agent as a sub-goal.
    nb_sub_goal_failed: int = 0
    #: Used to track for how long the failure rate has been above a certain threshold.
    nb_failures_above_thresh: int = 0

    # Information used to determine cell-selection probabilities
    #: The number of times this cell was chosen as the cell to explore from
    nb_chosen: int = 0
    #: Number of times this cell has been reached
    nb_reached: int = 0
    #: The number of actions that had this cell as the resulting state (i.e. all frames spend in this cell)
    nb_actions_taken_in_cell: int = 0
    #: The number of times this cell has been part of a trajectory
    nb_seen: int = 0
    #: The number of times the information of this cell was reset when update-on-reset is enabled
    nb_reset: int = 0

    def add(self, other):
        self.nb_chosen += other.nb_chosen
        self.nb_reached += other.nb_reached
        self.nb_actions_taken_in_cell += other.nb_actions_taken_in_cell


@dataclass
class TrajectoryElement:
    __slots__ = ['cells', 'action', 'reward', 'done', 'length', 'score', 'restore']
    cells: {}
    action: int
    reward: float
    done: bool
    length: int
    score: float
    restore: Any


@dataclass
class LogParameters:
    n_digits: int
    checkpoint_game: int
    checkpoint_compute: int
    checkpoint_first_iteration: bool
    checkpoint_last_iteration: bool
    max_game_steps: int
    max_compute_steps: int
    max_time: int
    max_iterations: int
    max_cells: int
    max_score: int
    save_pictures: List[str]
    clear_pictures: List[str]
    base_path: str
    checkpoint_it: int
    save_archive: bool
    save_model: bool
    checkpoint_time: int

    def should_render(self, name):
        return name in self.save_pictures or 'all' in self.save_pictures


@dataclass()
class Weight:
    weight: float = 1.0
    power: float = 1.0

    def __repr__(self):
        return f'w={self.weight:.2f}=p={self.power:.2f}'


@dataclass()
class DirWeights:
    horiz: float = 2.0
    vert: float = 0.3
    score_low: float = 0.0
    score_high: float = 0.0

    def __repr__(self):
        return f'h={self.horiz:.2f}=v={self.vert:.2f}=l={self.score_low:.2f}=h={self.score_high:.2f}'
