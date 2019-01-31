# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

def is_notebook():
    try:
        from IPython import get_ipython as _get_ipython
        if 'IPKernelApp' not in _get_ipython().config:  # pragma: no cover
            raise ImportError("console")
    except:
        return False
    return True
    
if not is_notebook():
    import matplotlib
    matplotlib.use('Agg')
    
from .basics import *

import warnings as _warnings
# Known to be benign: https://github.com/ContinuumIO/anaconda-issues/issues/6678#issuecomment-337279157
_warnings.filterwarnings('ignore', 'numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88')

try:
    import cv2
except ModuleNotFoundError:
    _warnings.warn('cv2 not found')
    
try:
    import gym
except ModuleNotFoundError:
    _warnings.warn('gym not found')

try:
    if not is_notebook():
        from tqdm import tqdm, trange
    else:
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange
except ModuleNotFoundError:
    _warnings.warn('tqdm not found')

import logging
class IgnoreNoHandles(logging.Filter):
    def filter(self, record):
        if record.getMessage() == 'No handles with labels found to put in legend.':
            return 0
        return 1
_plt_logger = logging.getLogger('matplotlib.legend')
_plt_logger.addFilter(IgnoreNoHandles())

    
import matplotlib.pyplot as plt
import numpy as np

class CircularMemory:
    def __init__(self, size):
        self.size = size
        self.mem = []
        self.start_idx = 0

    def add(self, entry):
        if len(self.mem) < self.size:
            self.mem.append(entry)
        else:
            self.mem[self.start_idx] = entry
            self.start_idx = (self.start_idx + 1) % self.size

    def sample(self, n):
        return random.sample(self.mem, n)

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, i):
        assert i < len(self)
        return self.mem[(self.start_idx + i) % self.size]

