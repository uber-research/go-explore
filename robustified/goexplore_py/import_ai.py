
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

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
    if is_notebook():
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange
    elif sys.stderr.isatty() and False:
        from tqdm import tqdm, trange
    else:
        class tqdm:
            def __init__(self, iterator=None, desc=None, smoothing=0, total=None):
                self.iterator = iterator
                self.desc = desc
                self.smoothing = smoothing
                self.total = total
                if self.total is None:
                    try:
                        self.total = len(iterator)
                    except Exception:
                        pass
                self.n = 0
                self.last_printed = 0
                self.start_time = time.time()

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.refresh(force_print=True, done=True)

            def __iter__(self):
                for e in self.iterator:
                    yield e
                    self.update(1)
                    self.refresh()

            def update(self, d):
                if d != 0:
                    self.n += d
                    self.refresh()

            def refresh(self, force_print=False, done=False):
                cur_time = time.time()
                if cur_time - self.last_printed < 10 and not force_print:
                    return
                self.last_printed = cur_time
                self.write(f'{self.get_desc_str():16}[{self.get_prog_str():26}{self.get_speed_str(cur_time):13}]' + (' DONE' if done else ''))

            def get_desc_str(self):
                if self.desc is None:
                    return ''
                return f'{self.desc}: '

            def get_prog_str(self):
                total_str = ''
                if isinstance(self.n, int):
                    if self.total is not None:
                        total_substr = f'{int(self.total)}'
                        total_str = f'{self.n / self.total * 100:2.0f}% {self.n:{len(total_substr)}}it/{total_substr}'
                    else:
                        total_str = str(self.n) + 'it'
                else:
                    if self.total is not None:
                        total_substr = f'{self.total:.1f}'
                        total_str = f'{self.n / self.total * 100:2.0f}% {self.n:{len(total_substr)}.1f}it/{total_substr}'
                    else:
                        total_str = f'{self.n:.1f}it'
                return total_str

            def get_speed_str(self, cur_time):
                if cur_time <= self.start_time:
                    return ''
                speed = self.n / (cur_time - self.start_time)
                if speed > 1:
                    return f' {speed:.1f}it/s'
                if speed < 0.000000000001:
                    return ''
                return f' {1/speed:.1f}s/it'

            @classmethod
            def write(cls, str):
                print(str, file=sys.stderr)
                sys.stderr.flush()

except ModuleNotFoundError:
    _warnings.warn('tqdm not found')

import numpy as np
class RLEArray:
    def __init__(self, array, encoded_array=None, compression=1):
        import cv2
        if array is None:
            self.array = encoded_array
        else:
            assert not isinstance(array, RLEArray)
            # Note: 7 seems to be a good tradeoff between size and speed
            self.array = cv2.imencode('.png', array, [cv2.IMWRITE_PNG_COMPRESSION, compression])[1].flatten().tobytes()

    def to_np(self):
        return cv2.imdecode(np.frombuffer(self.array, np.uint8), 0)

    def tobytes(self):
        return self.array

    @classmethod
    def frombytes(cls, byt, dtype=np.uint8):
        return cls(None, np.frombuffer(byt, dtype=dtype))


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

def show_img(im, figsize=None, ax=None, grid=False):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    if grid:
        ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
        verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)



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

