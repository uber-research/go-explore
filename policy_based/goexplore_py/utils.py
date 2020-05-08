# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import random
import numpy as np
import os
import glob
import hashlib
from contextlib import contextmanager


class TimedPickle:
    def __init__(self, data, name, enabled=True):
        self.data = data
        self.name = name
        self.enabled = enabled

    def __getstate__(self):
        return time.time(), self.data, self.name, self.enabled

    def __setstate__(self, s):
        tstart, self.data, self.name, self.enabled = s
        if self.enabled:
            print(f'pickle time for {self.name} = {time.time() - tstart} seconds')


@contextmanager
def use_seed(seed):
    # Save all the states
    python_state = random.getstate()
    np_state = np.random.get_state()

    # Seed all the rngs (note: adding different values to the seeds
    # in case the same underlying RNG is used by all and in case
    # that could be a problem. Probably not necessary)
    random.seed(seed + 2)
    np.random.seed(seed + 3)

    # Yield control!
    yield

    # Reset the rng states
    random.setstate(python_state)
    np.random.set_state(np_state)


def get_code_hash():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    all_code = ''
    for f in sorted(glob.glob(cur_dir + '**/*.py', recursive=True)):
        # We assume all whitespace is irrelevant, as well as comments
        with open(f) as fh:
            for line in fh:
                line = line.partition('#')[0]
                line = line.rstrip()

                all_code += ''.join(line.split())

    code_hash = hashlib.sha256(all_code.encode('utf8')).hexdigest()

    return code_hash


def clip(value, low, high):
    return max(min(value, high), low)
