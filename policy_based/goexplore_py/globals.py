# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional
import os
import logging
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EXP_STRAT_INIT = -1
EXP_STRAT_NONE = 0
EXP_STRAT_RAND = 1
EXP_STRAT_POLICY = 2


ACTION_MEANINGS: Optional[List] = None
MASTER_PID = None
BASE_PATH = None


def set_action_meanings(meanings=List[str]):
    global ACTION_MEANINGS
    ACTION_MEANINGS = meanings
    logger.debug(f'ACTION_MEANINGS set for process: {os.getpid()}')


def get_action_meaning(i):
    return ACTION_MEANINGS[i]


def get_trajectory(prev_idxs: List[int], actions: List[int], idx: int):
    trajectory = []
    if idx is not None:
        while prev_idxs[idx] is not None:
            action = actions[idx]
            idx = idx - prev_idxs[idx]
            trajectory.append(action)
    trajectory.reverse()
    return trajectory


def set_master_pid(pid):
    global MASTER_PID
    MASTER_PID = pid


def get_master_pid():
    global MASTER_PID
    return MASTER_PID


def set_base_path(base_path):
    global BASE_PATH
    BASE_PATH = base_path


def get_base_path():
    global BASE_PATH
    return BASE_PATH
