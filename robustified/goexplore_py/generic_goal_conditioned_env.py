
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *
import types


def make_robot_env(name, will_render=False):
    env = gym.make(name)

    # Fix rendering to 1/ hide the overlay, 2/ read the proper pixels in rgb_array
    # mode and 3/ prevent rendering if will_render is not announced (necessary because
    # when will_render is announced, we proactively create a viewer as soon as the
    # env is created, because creating it later causes inaccuracies).
    def render(self, mode='human'):
        assert will_render, 'Rendering in an environment with will_render=False'
        self._render_callback()
        self._get_viewer()._hide_overlay = True
        if mode == 'rgb_array':
            self._get_viewer().render()
            import glfw
            width, height = glfw.get_window_size(self.viewer.window)
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def get_full_state(self):
        pass

    def set_full_state(self, state):
        pass

    env.unwrapped.render = types.MethodType(render, env.unwrapped)
    env.unwrapped.get_full_state = types.MethodType(get_full_state, env.unwrapped)
    env.unwrapped.set_full_state = types.MethodType(set_full_state, env.unwrapped)
    if will_render:
        # Pre-cache the viewer because creating it while the environment is running
        # sometimes causes errors
        env.unwrapped._get_viewer()

    if 'Fetch' in name:
        # The way _render_callback is implemented in Fetch environments causes issues.
        # This monkey patch fixes them.
        def _render_callback(self):
            # Visualize target.
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

        env.unwrapped._render_callback = types.MethodType(_render_callback, env.unwrapped)

    return env

class DomainConditionedPosLevel:
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
        if not isinstance(other, DomainConditionedPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}'


class MyRobot:
    TARGET_SHAPE = 0
    MAX_PIX_VALUE = 0

    def __init__(self, env_name, interval_size=0.1, seed_low=0, seed_high=0):
        self.env_name = env_name
        self.env = make_robot_env(env_name)
        self.prev_action = np.zeros_like(self.env.action_space.sample())
        self.interval_size = interval_size
        self.state = None
        self.actual_state = None
        self.rooms = []
        self.trajectory = []

        self.seed_low = seed_low
        self.seed_high = seed_high
        self.seed = None

        self.cur_achieved_goal = None
        self.achieved_has_moved = False
        self.score_so_far = 0

        self.follow_grip_until_moved = ('FetchPickAndPlace' in env_name and False)

        self.reset()

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

    def pos_from_state(self, seed, state):
        if self.follow_grip_until_moved:
            pos = state['achieved_goal'] if self.achieved_has_moved else state['observation'][:3]
            return np.array([seed, self.achieved_has_moved] + list(pos / self.interval_size), dtype=np.int32)
        return np.array([seed, self.score_so_far] + list((state['achieved_goal'] / self.interval_size).astype(np.int32)), dtype=np.int32)

    def reset(self) -> np.ndarray:
        self.seed = None
        self.trajectory = None
        self.actual_state = None
        self.cur_achieved_goal = None
        self.achieved_has_moved = False
        self.score_so_far = 0
        self.state = [self.pos_from_state(-1, {'achieved_goal': np.array([]), 'observation': np.array([])})]
        return copy.copy(self.state)

    def get_restore(self):
        return copy.deepcopy((
            None,
            self.env._elapsed_steps,
            self.interval_size,
            self.cur_achieved_goal,
            self.achieved_has_moved,
            self.score_so_far,
            self.state,
            self.actual_state,
            self.trajectory,
            self.seed,
        ))

    def restore(self, data):
        (
            simstate,
            self.env._elapsed_steps,
            self.interval_size,
            self.cur_achieved_goal,
            self.achieved_has_moved,
            self.score_so_far,
            state,
            actual_state,
            trajectory,
            seed,
        ) = copy.deepcopy(data)
        self.reset()
        self.seed = seed
        for a in trajectory:
            self.step(a)
        assert np.allclose(self.actual_state['achieved_goal'], actual_state['achieved_goal'])
        return copy.copy(self.state)

    def step(self, action):
        self.prev_action = copy.deepcopy(self.prev_action)
        self.prev_action[:] = action
        if self.trajectory is None:
            if self.seed is None:
                self.seed = random.randint(self.seed_low, self.seed_high)
            self.env.unwrapped.sim.reset()
            self.env.seed(self.seed)
            self.actual_state = self.env.reset()
            self.trajectory = []
            self.state = [self.pos_from_state(self.seed, self.actual_state)]

        self.trajectory.append(copy.copy(self.prev_action))
        action = np.tanh(self.prev_action)
        self.actual_state, reward, done, lol = self.env.step(action)
        reward = int(reward) + 1
        self.score_so_far += reward
        self.state = [self.pos_from_state(self.seed, self.actual_state)]

        if not self.achieved_has_moved and self.cur_achieved_goal is not None and not np.allclose(self.cur_achieved_goal, self.actual_state['achieved_goal']):
            self.achieved_has_moved = True
        self.cur_achieved_goal = self.actual_state['achieved_goal']

        return copy.copy(self.state), reward, done, lol

    def get_pos(self):
        return DomainConditionedPosLevel()

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        pass