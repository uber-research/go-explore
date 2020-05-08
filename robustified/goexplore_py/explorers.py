
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *

class RandomExplorer:
    def init_seed(self):
        pass

    def get_action(self, env):
        return random.randint(0, env.action_space.n - 1)

    def __repr__(self):
        return 'RandomExplorer()'


class RepeatedRandomExplorer:
    def __init__(self, mean_repeat=10):
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


class RandomDriftExplorerRobot:
    def __init__(self, sd):
        self.sd = sd

    def init_seed(self):
        pass

    def get_action(self, env):
        return env.prev_action + np.random.randn(env.prev_action.size) * self.sd

    def __repr__(self):
        return f'drift-{self.sd}'


def actstr(act):
    return ' '.join([f'{e:01.2f}' for e in act])


class RandomDriftExplorerFetch:
    def __init__(self, sd):
        self.sd = sd

    def init_seed(self):
        pass

    def get_action(self, env):
        action = env.prev_action + np.random.randn(env.prev_action.size) * self.sd
        return action

    def __repr__(self):
        return f'drift-{self.sd}'


class RepeatedRandomExplorerFetch:
    def __init__(self, mean_repeat=10):
        self.mean_repeat = mean_repeat
        self.action = 0
        self.remaining = 0

    def init_seed(self):
        self.remaining = 0

    def get_action(self, env):
        if self.remaining <= 0:
            self.action = env.sample_action(sd=2)
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

    def get_action(self, *args):
        return 0
