
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .basics import *
from .import_ai import *
from . import montezuma_env
from .utils import imdownscale

def convert_state(state):
    if MyAtari.TARGET_SHAPE is None:
        return None
    import cv2
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    if MyAtari.TARGET_SHAPE == (-1, -1):
        return RLEArray(state)
    return imdownscale(state, MyAtari.TARGET_SHAPE, MyAtari.MAX_PIX_VALUE)

class AtariPosLevel:
    __slots__ = ['level', 'score', 'room', 'x', 'y', 'tuple']

    def __init__(self, level=0, score=0, room=0, x=0, y=0):
        self.level = level
        self.score = score
        self.room = room
        self.x = x
        self.y = y

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.level, self.score, self.room, self.x, self.y)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, AtariPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}'

def clip(a, m, M):
    if a < m:
        return m
    if a > M:
        return M
    return a


class MyAtari:
    def __init__(self, name, x_repeat=2, end_on_death=False):
        self.name = name
        self.env = gym.make(f'{name}Deterministic-v4')
        self.unwrapped.seed(0)
        self.env.reset()
        self.state = []
        self.x_repeat = x_repeat
        self.rooms = []
        self.unprocessed_state = None
        self.end_on_death = end_on_death
        self.prev_lives = 0

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        self.env = gym.make(f'{self.name}Deterministic-v4')
        self.unwrapped.seed(0)
        self.unprocessed_state = self.env.reset()
        self.state = [convert_state(self.unprocessed_state)]
        return copy.copy(self.state)

    def get_restore(self):
        return (
            self.unwrapped.clone_state(),
            copy.copy(self.state),
            self.env._elapsed_steps
        )

    def restore(self, data):
        (
            full_state,
            state,
            elapsed_steps
        ) = data
        self.state = copy.copy(state)
        self.env.reset()
        self.env._elapsed_steps = elapsed_steps
        self.env.unwrapped.restore_state(full_state)
        return copy.copy(self.state)

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        self.unprocessed_state, reward, done, lol = self.env.step(action)
        self.state.append(convert_state(self.unprocessed_state))
        self.state.pop(0)

        cur_lives = self.env.unwrapped.ale.lives()
        if self.end_on_death and cur_lives < self.prev_lives:
            done = True
        self.prev_lives = cur_lives

        return copy.copy(self.state), reward, done, lol

    def get_pos(self):
        # NOTE: this only returns a dummy position
        return AtariPosLevel()

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        pass
