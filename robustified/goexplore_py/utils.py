
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *

class TimedPickle:
    def __init__(self, data, name, enabled=True):
        self.data = data
        self.name = name
        self.enabled = enabled

    def __getstate__(self):
        return (time.time(), self.data, self.name, self.enabled)

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
    random.seed(seed)
    np.random.seed(seed + 1)

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
        with open(f) as f:
            for line in f:
                line = line.partition('#')[0]
                line = line.rstrip()

                all_code += ''.join(line.split())

    hash = hashlib.sha256(all_code.encode('utf8')).hexdigest()
    print('HASH', hash)

    return hash


def imdownscale(state, target_shape, max_pix_value):
    if state.shape[::-1] == target_shape:
        resized = state
    else:
        resized = cv2.resize(state, target_shape, interpolation=cv2.INTER_AREA)
    img = ((resized / 255.0) * max_pix_value).astype(np.uint8)
    return RLEArray(img)
