"""
// Modifications Copyright (c) 2020 Uber Technologies Inc.
"""
import os
import random
import pickle
import gym
from collections import deque
from PIL import Image
from gym import spaces
import imageio
import numpy as np
from multiprocessing import Process, Pipe
import horovod.tensorflow as hvd
import copy
import warnings as _warnings
import cv2
import cloudpickle
import goexplore_py.mpi_support as mpi
import logging
logger = logging.getLogger(__name__)

try:
    from dataclasses import dataclass, field as datafield

    def copyfield(data):
        return datafield(default_factory=lambda: copy.deepcopy(data))
except ModuleNotFoundError:
    _warnings.warn('dataclasses not found. To get it, use Python 3.7 or pip install dataclasses')

reset_for_batch = False


# change for long runs
SCORE_THRESHOLD = 500_000_000_000


class VecWrapper(object):
    def __init__(self, venv):
        self.venv = venv

    def decrement_starting_point(self, nr_steps, idx):
        return self.venv.decrement_starting_point(nr_steps, idx)

    def set_archive(self, archive):
        return self.venv.set_archive(archive)

    def set_selector(self, selector):
        return self.venv.set_selector(selector)

    def init_archive(self):
        archive = self.venv.init_archive()
        return archive

    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.venv.recursive_getattr(name)
            except AttributeError:
                raise Exception(f'Couldn\'t get attr: {name}')

    def recursive_setattr(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            try:
                return self.venv.recursive_setattr(name, value)
            except AttributeError:
                raise Exception(f'Couldn\'t set attr: {name}')

    def recursive_call_method(self, name, arguments=()):
        if hasattr(self, name):
            return getattr(self, name)(*arguments)
        else:
            try:
                return self.venv.recursive_call_method(name, arguments)
            except AttributeError:
                raise Exception(f'Couldn\'t call method: {name}')

    def recursive_call_method_ignore_return(self, name, arguments=()):
        if hasattr(self, name):
            getattr(self, name)(*arguments)
        else:
            try:
                self.venv.recursive_call_method_ignore_return(name, arguments)
            except AttributeError:
                raise Exception(f'Couldn\'t call method: {name}')

    def batch_reset(self):
        global reset_for_batch
        reset_for_batch = True
        obs = self.venv.reset()
        reset_for_batch = False
        return obs

    def reset(self):
        return self.venv.reset()

    def step(self, action):
        return self.venv.step(action)

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset_task(self):
        return self.venv.reset_task()

    @property
    def num_envs(self):
        return self.venv.num_envs

    def close(self):
        return self.venv.close()

    def get_restore(self):
        return self.venv.get_restore()

    def restore(self, state):
        return self.venv.restore(state)

    def get_envs(self):
        return self.venv.get_envs()


class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MyWrapper, self).__init__(env)

    def __getattr__(self, item):
        """
        Un-implements the __getattr__ method from the gym.Wrapper base class, as MyWrapper is build on the assumption
        that __getattr__ is not implemented. The __getattr__ method was implemented around Open AI Gym version 0.12.

        @param item:
        @return:
        """
        raise AttributeError('Could not find ' + item + ' in ' + self.__class__.__name__ + '. ' +
                             '__getattr__ is not (and should not be) implemented in MyWrapper')

    def decrement_starting_point(self, nr_steps, idx):
        return self.env.decrement_starting_point(nr_steps, idx)

    def set_archive(self, archive):
        return self.env.set_archive(archive)

    def set_selector(self, selector):
        return self.env.set_selector(selector)

    def init_archive(self):
        archive = self.env.init_archive()
        return archive

    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.env.recursive_getattr(name)
            except AttributeError:
                raise Exception(f'Couldn\'t get attr: {name}')

    def recursive_setattr(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            try:
                return self.env.recursive_setattr(name, value)
            except AttributeError:
                raise Exception(f'Couldn\'t set attr: {name}')

    def recursive_call_method(self, name, arguments=()):
        if hasattr(self, name):
            return getattr(self, name)(*arguments)
        else:
            try:
                return self.env.recursive_call_method(name, arguments)
            except AttributeError:
                raise Exception(f'Couldn\'t call method: {name}')

    def recursive_call_method_ignore_return(self, name, arguments=()):
        if hasattr(self, name):
            getattr(self, name)(*arguments)
        else:
            try:
                self.env.recursive_call_method_ignore_return(name, arguments)
            except AttributeError:
                raise Exception(f'Couldn\'t call method: {name}')

    def batch_reset(self):
        global reset_for_batch
        reset_for_batch = True
        obs = self.env.reset()
        reset_for_batch = False
        return obs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def reset_task(self):
        return self.env.reset_task()

    @property
    def num_envs(self):
        return self.env.num_envs

    def close(self):
        return self.env.close()

    def get_restore(self):
        return self.env.get_restore()

    def restore(self, state):
        return self.env.restore(state)


class VecFrameStack(VecWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        super(VecFrameStack, self).__init__(venv)
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = venv.action_space
        self._goal_space = venv.goal_space

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def goal_space(self):
        return self._goal_space

    def close(self):
        self.venv.close()

    @property
    def num_envs(self):
        return self.venv.num_envs

    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.venv.recursive_getattr(name)
            except AttributeError:
                raise Exception(f'Couldn\'t get attr: {name}')

    def set_archive(self, archive):
        return self.venv.set_archive(archive)

    def set_selector(self, selector):
        return self.venv.set_selector(selector)


class DemoReplayInfo:
    def __init__(self, demo_file_name, seed, workers_per_sp):
        # Added to allow for the creation of "fake" replay information
        if demo_file_name is None:
            self.actions = None
            self.returns = [0]
            self.checkpoints = None
            self.checkpoint_action_nr = None
            self.starting_point = 0
            self.starting_point_current_ep = None
        else:
            with open(demo_file_name, "rb") as f:
                dat = pickle.load(f)
            self.actions = dat['actions']
            rewards = dat['rewards']
            assert len(rewards) == len(self.actions)
            self.returns = np.cumsum(rewards)
            self.checkpoints = dat['checkpoints']
            self.checkpoint_action_nr = dat['checkpoint_action_nr']
            self.starting_point = len(self.actions) - 1 - seed//workers_per_sp
            self.starting_point_current_ep = None


class ReplayResetEnv(MyWrapper):
    """
    Randomly resets to states from a replay
    """

    def __init__(self,
                 env,
                 demo_file_name,
                 seed,
                 reset_steps_ignored=64,
                 workers_per_sp=4,
                 frac_sample=0.2,
                 game_over_on_life_loss=True,
                 allowed_lag=50,
                 allowed_score_deficit=0,
                 test_from_start=False,
                 from_start_prior=0,
                 demo_selection='uniform',
                 avg_frames_window_size=0):
        super(ReplayResetEnv, self).__init__(env)
        self.rng = np.random.RandomState(seed)
        self.reset_steps_ignored = reset_steps_ignored
        self.actions_to_overwrite = []
        self.frac_sample = frac_sample
        self.game_over_on_life_loss = game_over_on_life_loss
        self.allowed_lag = allowed_lag
        self.allowed_score_deficit = allowed_score_deficit
        self.demo_replay_info = []
        self.test_from_start = test_from_start
        if test_from_start:
            self.demo_replay_info.append(DemoReplayInfo(None, seed, workers_per_sp))
        if os.path.isdir(demo_file_name):
            import glob
            for f in sorted(glob.glob(demo_file_name + '/*.demo')):
                self.demo_replay_info.append(DemoReplayInfo(f, seed, workers_per_sp))
        else:
            self.demo_replay_info.append(DemoReplayInfo(demo_file_name, seed, workers_per_sp))
        self.cur_demo_replay = None
        self.cur_demo_idx = -1
        self.extra_frames_counter = -1
        self.action_nr = -1
        self.score = -1
        self.demo_selection = demo_selection
        self.avg_frames_window_size = avg_frames_window_size
        self.infinite_window_size = False
        if not avg_frames_window_size > 0:
            self.avg_frames_window_size = 1
            self.infinite_window_size = True
        self.times_demos_chosen = np.zeros(len(self.demo_replay_info), dtype=np.int)
        self.steps_taken_per_demo = np.zeros((len(self.demo_replay_info), self.avg_frames_window_size), dtype=np.int)
        for i in range(len(self.demo_replay_info)):
            if from_start_prior > 0 and test_from_start and i == 0:
                self.steps_taken_per_demo[i, :] = from_start_prior
                self.times_demos_chosen[i] = self.avg_frames_window_size
            else:
                self.steps_taken_per_demo[i, 0] = 1

    def recursive_getattr(self, name):
        prefix = 'starting_point_'
        if name[:len(prefix)] == prefix:
            idx = int(name[len(prefix):])
            return self.demo_replay_info[idx].starting_point
        elif name == 'n_demos':
            return len(self.demo_replay_info)
        else:
            return super(ReplayResetEnv, self).recursive_getattr(name)

    def _get_window_index(self):
        window_index = (self.times_demos_chosen[self.cur_demo_idx] - 1) % self.avg_frames_window_size
        assert window_index >= 0
        assert window_index < self.avg_frames_window_size
        return window_index

    def step(self, action):
        if len(self.actions_to_overwrite) > 0:
            action = self.actions_to_overwrite.pop(0)
            valid = False
        else:
            valid = True
        prev_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, info = self.env.step(action)
        info['idx'] = self.cur_demo_idx
        self.steps_taken_per_demo[self.cur_demo_idx, self._get_window_index()] += 1
        self.action_nr += 1
        self.score += reward

        # game over on loss of life, to speed up learning
        if self.game_over_on_life_loss:
            lives = self.env.unwrapped.ale.lives()
            if prev_lives > lives > 0:
                done = True

        if self.test_from_start and self.cur_demo_idx == 0:
            pass
        # kill if we have achieved the final score, or if we're lagging the demo too much
        elif self.score >= self.cur_demo_replay.returns[-1]:
            self.extra_frames_counter -= 1
            if self.extra_frames_counter <= 0:
                done = True
                info['replay_reset.random_reset'] = True  # to distinguish from actual game over
        elif self.action_nr > self.allowed_lag:
            min_index = self.action_nr - self.allowed_lag
            if min_index < 0:
                min_index = 0
            if min_index >= len(self.cur_demo_replay.returns):
                min_index = len(self.cur_demo_replay.returns) - 1
            max_index = self.action_nr + self.allowed_lag
            threshold = min(self.cur_demo_replay.returns[min_index: max_index]) - self.allowed_score_deficit
            if self.score < threshold:
                done = True

        # output flag to increase entropy if near the starting point of this episode
        if self.action_nr < self.cur_demo_replay.starting_point + 100:
            info['increase_entropy'] = True

        if done:
            ep_info = {'l': self.action_nr,
                       'as_good_as_demo': (self.score >=
                                           (self.cur_demo_replay.returns[-1] - self.allowed_score_deficit)),
                       'r': self.score,
                       'starting_point': self.cur_demo_replay.starting_point_current_ep,
                       'idx': self.cur_demo_idx}
            info['episode'] = ep_info

        if not valid:
            info['replay_reset.invalid_transition'] = True

        return obs, reward, done, info

    def decrement_starting_point(self, nr_steps, demo_idx):
        if self.demo_replay_info[demo_idx].starting_point > 0:
            starting_point = self.demo_replay_info[demo_idx].starting_point
            self.demo_replay_info[demo_idx].starting_point = int(np.maximum(starting_point - nr_steps, 0))

    def reset(self):
        obs = self.env.reset()
        # noinspection PyArgumentList
        self.extra_frames_counter = int(np.exp(self.rng.rand()*7))

        # Select demo
        ones = np.ones(len(self.demo_replay_info))
        norm = np.where(self.times_demos_chosen == 0, ones, self.times_demos_chosen)
        if not self.infinite_window_size:
            norm = np.where(norm > self.avg_frames_window_size, self.avg_frames_window_size, norm)
        expected_steps = np.sum(self.steps_taken_per_demo, axis=1) / norm
        inverse_expected = 1 / expected_steps
        if self.demo_selection == 'normalize_from_start':
            logits = inverse_expected
            logits[1:] = np.mean(logits[1:])
        elif self.demo_selection == 'normalize':
            logits = inverse_expected
        elif self.demo_selection == 'uniform':
            logits = ones
        else:
            raise NotImplementedError(f"Unknown operation: {self.demo_selection}")
        logits = logits / logits.sum()
        self.cur_demo_idx = np.random.choice(len(self.demo_replay_info), p=logits)
        self.times_demos_chosen[self.cur_demo_idx] += 1
        if not self.infinite_window_size:
            self.steps_taken_per_demo[self.cur_demo_idx, self._get_window_index()] = 0
        self.cur_demo_replay = self.demo_replay_info[self.cur_demo_idx]

        # Select starting point
        if self.test_from_start and self.cur_demo_idx == 0:
            self.cur_demo_replay.starting_point_current_ep = 0
            self.actions_to_overwrite = []
            self.action_nr = 0
            self.score = 0
            obs = self.env.reset()
            noops = random.randint(0, 30)
            for _ in range(noops):
                obs, _, _, _ = self.env.step(0)
            return obs

        elif reset_for_batch:
            self.cur_demo_replay.starting_point_current_ep = 0
            self.actions_to_overwrite = self.cur_demo_replay.actions[:]
            self.action_nr = 0
            self.score = self.cur_demo_replay.returns[0]
        else:
            # noinspection PyArgumentList
            if self.rng.rand() <= 1.-self.frac_sample:
                self.cur_demo_replay.starting_point_current_ep = self.cur_demo_replay.starting_point
            else:
                self.cur_demo_replay.starting_point_current_ep = self.rng.randint(
                    low=self.cur_demo_replay.starting_point, high=len(self.cur_demo_replay.actions))

            start_action_nr = 0
            start_ckpt = None
            for nr, ckpt in zip(self.cur_demo_replay.checkpoint_action_nr[::-1],
                                self.cur_demo_replay.checkpoints[::-1]):
                if nr <= (self.cur_demo_replay.starting_point_current_ep - self.reset_steps_ignored):
                    start_action_nr = nr
                    start_ckpt = ckpt
                    break
            if start_action_nr > 0:
                self.env.unwrapped.restore_state(start_ckpt)
            nr_to_start_lstm = np.maximum(self.cur_demo_replay.starting_point_current_ep - self.reset_steps_ignored,
                                          start_action_nr)
            if nr_to_start_lstm > start_action_nr:
                for a in self.cur_demo_replay.actions[start_action_nr:nr_to_start_lstm]:
                    # noinspection PyProtectedMember
                    action = self.env.unwrapped._action_set[a]
                    self.env.unwrapped.ale.act(action)
            actions = self.cur_demo_replay.actions
            starting_point = self.cur_demo_replay.starting_point_current_ep
            self.cur_demo_replay.actions_to_overwrite = actions[nr_to_start_lstm:starting_point]
            if nr_to_start_lstm > 0:
                # noinspection PyProtectedMember
                obs = self.env.unwrapped._get_image()
            self.action_nr = nr_to_start_lstm
            self.score = self.cur_demo_replay.returns[nr_to_start_lstm]
            if self.cur_demo_replay.starting_point_current_ep == 0 and self.cur_demo_replay.actions_to_overwrite == []:
                noops = random.randint(0, 30)
                for _ in range(noops):
                    obs, _, _, _ = self.env.step(0)

        return obs


class MaxAndSkipEnv(MyWrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        MyWrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        combined_info = {'skip_env.executed_actions': []}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info['skip_env.executed_actions'].append(info['sticky_env.executed_action'])
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, combined_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ClipRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.sign(reward)
        return obs, reward, done, info


class IgnoreNegativeRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = max(reward, 0)
        return obs, reward, done, info


class ScaledRewardEnv(MyWrapper):
    def __init__(self, env, scale=1):
        MyWrapper.__init__(self, env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward*self.scale
        return obs, reward, done, info


class EpsGreedyEnv(MyWrapper):
    def __init__(self, env, eps=0.01):
        MyWrapper.__init__(self, env)
        self.eps = eps

    def step(self, action):
        if np.random.uniform() < self.eps:
            action = np.random.randint(self.env.action_space.n)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class StickyActionEnv(MyWrapper):
    def __init__(self, env, p=0.25):
        MyWrapper.__init__(self, env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        info['sticky_env.executed_action'] = action
        return obs, reward, done, info

    def reset(self):
        self.last_action = 0
        return self.env.reset()


class FireResetEnv(MyWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        MyWrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class PreventSlugEnv(MyWrapper):
    def __init__(self, env, max_no_rewards=10000):
        """Abort if too much time without getting reward."""
        MyWrapper.__init__(self, env)
        self.last_reward = 0
        self.steps = 0
        self.max_no_rewards = max_no_rewards
        self.got_reward = False

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        self.steps += 1
        if reward > 0:
            self.last_reward = self.steps
        if self.steps - self.last_reward > self.max_no_rewards:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.got_reward = False
        self.steps = 0
        return self.env.reset()


class VideoWriter(MyWrapper):
    def __init__(self, env, file_prefix,
                 plot_goal=False,
                 x_res=16,
                 y_res=16,
                 plot_archive=False,
                 plot_return_prob=True,
                 one_vid_per_goal=False,
                 make_video=False,
                 directory='.',
                 pixel_repetition=1,
                 plot_grid=True,
                 plot_sub_goal=True):
        MyWrapper.__init__(self, env)
        self.file_prefix = file_prefix
        self.video_writer = None
        self.counter = 0
        self.orig_frames = None
        self.cur_step = 0
        self.score = 0
        self.plot_goal = plot_goal
        self.x_res = x_res
        self.y_res = y_res
        self.goal_conditioned_wrapper = None
        self.plot_archive = plot_archive
        self.goals_processed = set()
        self.one_vid_per_goal = one_vid_per_goal
        self.plot_grid = plot_grid
        self.plot_return_prob = plot_return_prob
        self.frames_as_images = False
        self.obs = None
        self.time_in_cell = 0
        self.cell_traj_index = 0
        self.make_video = make_video
        self.directory = directory
        self.plot_cell_traj: bool = False
        self.plot_sub_goal: bool = plot_sub_goal
        self.pixel_repetition: int = pixel_repetition
        self.goal = None
        self.min_video_length = 4
        self.current_file_name = None
        self.num_frames = 0
        self.draw_text = False
        self.local_archive = set()

    def _render_cell(self, canvas, cell, color, overlay=None):
        x_min = cell.x * self.x_res
        y_min = cell.y * self.y_res
        cv2.rectangle(canvas, (x_min, y_min), (x_min + self.x_res, y_min + self.y_res), color, -1)
        if overlay is not None:
            cv2.rectangle(overlay, (x_min, y_min), (x_min + self.x_res, y_min + self.y_res), color, 1)

    def match_attr(self, cell_1, cell_2, attr_name):
        matches = True
        if hasattr(cell_1, attr_name) and hasattr(cell_2, attr_name):
            matches = getattr(cell_1, attr_name) == getattr(cell_2, attr_name)
        return matches

    def process_frame(self, frame):
        f_out = np.zeros((160, 160, 3), dtype=np.uint8)
        f_out[:, :, 0:3] = np.cast[np.uint8](frame)[50:, :, :]
        f_out = f_out.repeat(2, axis=1)
        f_overlay = copy.copy(f_out)

        if self.plot_grid:
            for y in np.arange(self.y_res, f_out.shape[0], self.y_res):
                cv2.line(f_out, (0, 0 + y), (0 + f_out.shape[1], 0 + y), (127, 127, 127), 1)
            for x in np.arange(self.x_res, f_out.shape[1], self.x_res):
                cv2.line(f_out, (0 + x, 0), (0 + x, 0 + f_out.shape[0]), (127, 127, 127), 1)

        current_cell = self.goal_conditioned_wrapper.archive.get_cell_from_env(self.goal_conditioned_wrapper.env)
        if self.plot_archive:
            for cell_key in self.local_archive:
                if self.match_attr(cell_key, current_cell, 'level') and self.match_attr(cell_key, current_cell, 'room'):
                    base_brightness = 50
                    if self.plot_return_prob:
                        reached = self.goal_conditioned_wrapper.archive.cells_reached_dict.get(cell_key, [])
                        if len(reached) > 0:
                            r = base_brightness + (255 - base_brightness) * (sum(reached)/len(reached))
                        else:
                            r = base_brightness
                        color = (0, r, 100)
                    else:
                        color = (0, 0, 200)
                    self._render_cell(f_out, cell_key, color, overlay=f_overlay)
        self.local_archive.add(current_cell)

        if self.plot_goal:
            goal = self.goal_conditioned_wrapper.goal_cell_rep
            if goal is not None:
                if self.match_attr(goal, current_cell, 'level') and self.match_attr(goal, current_cell, 'room'):
                    self._render_cell(f_out, goal, (255, 0, 0))

        if self.plot_cell_traj:
            goal = self.goal_conditioned_wrapper.goal_cell_rep
            if goal is not None:
                traj = self.goal_conditioned_wrapper.goal_cell_info.cell_traj
                if len(traj) > 0:
                    if self.time_in_cell <= 0 and (self.cell_traj_index + 1) < len(traj):
                        self.cell_traj_index += 1
                        self.time_in_cell = traj[self.cell_traj_index][1]
                    traj_cell = traj[self.cell_traj_index][0]
                    self.time_in_cell -= 1

                    self._render_cell(f_out, traj_cell, (255, 255, 0))

        if self.plot_sub_goal:
            goal = self.goal_conditioned_wrapper.sub_goal_cell_rep
            if goal is not None:
                if self.match_attr(goal, current_cell, 'level') and self.match_attr(goal, current_cell, 'room'):
                    self._render_cell(f_out, goal, (255, 255, 0), overlay=f_overlay)
        for cell in self.goal_conditioned_wrapper.entropy_manager.entropy_cells:
            if self.match_attr(cell, current_cell, 'level') and self.match_attr(cell, current_cell, 'room'):
                self._render_cell(f_out, cell, (255, 0, 255))

        cv2.addWeighted(f_overlay, 0.5, f_out, 0.5, 0, f_out)

        f_out = f_out.repeat(self.pixel_repetition, axis=0)
        f_out = f_out.repeat(self.pixel_repetition, axis=1)

        if self.draw_text:
            if 'increase_entropy' in self.goal_conditioned_wrapper.info:
                text = str(self.goal_conditioned_wrapper.info['increase_entropy'])
                f_out = cv2.putText(f_out, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            text = str(self.goal_conditioned_wrapper.total_reward)
            f_out = cv2.putText(f_out, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.frames_as_images:
            filename = self.current_file_name + f'_{self.cur_step:0>3}.png'
            im = Image.fromarray(f_out)
            im.save(filename)
        return f_out

    def add_frame(self):
        if self.video_writer:
            self.video_writer.append_data(self.process_frame(self.obs))
            self.num_frames += 1

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        self.cur_step += 1

        if reward <= -999000:
            reward = 0
        self.score += reward
        return obs, reward, done, info

    def reset(self):
        self._finalize_video()
        self.score = 0
        self.cur_step = 0
        self.time_in_cell = 0
        self.cell_traj_index = -1
        res = self.env.reset()
        self.obs = res
        return res

    def start_video(self):
        self.goal = self.goal_conditioned_wrapper.goal_cell_rep
        self.local_archive = set()
        if self.make_video:
            if self.goal not in self.goals_processed or not self.one_vid_per_goal:
                self.num_frames = 0
                os.makedirs(self.directory, exist_ok=True)
                self.current_file_name = self._get_file_name()
                self.video_writer = imageio.get_writer(self.current_file_name, mode='I', fps=30)
            else:
                self.video_writer = None
        else:
            self.video_writer = None

    def close(self):
        self._finalize_video()
        self.env.close()

    def _finalize_video(self):
        if self.video_writer is not None:
            self.video_writer.close()
            if self.make_video and self.num_frames > self.min_video_length:
                self.goals_processed.add(self.goal)
                print('Score achieved:', self.recursive_getattr('cur_score'))
                print('Video for goal:', self.goal, 'is considered finished.')
            else:
                print('Video for goal:', self.goal, 'considered too short, deleting...')
                os.remove(self.current_file_name)
            self.counter += 1

    def _get_file_name(self):
        goal = self.goal_conditioned_wrapper.goal_cell_rep
        info = self.goal_conditioned_wrapper.goal_cell_info
        print('Starting video for goal:', goal, info)
        rand_val = random.randint(0, 1000000)
        if goal is not None:
            if hasattr(goal, 'level'):
                postfix = f'_{goal.level}_{goal.room}_{goal.objects}_{goal.y:0>2}_{goal.x:0>2}_{self.counter}.mp4'
                name = self.file_prefix + postfix
            elif hasattr(goal, 'treasures'):
                name = self.file_prefix + f'_{goal.treasures}_{goal.room}_{goal.y:0>2}_{goal.x:0>2}_{self.counter}.mp4'
            else:
                name = self.file_prefix + f'_{rand_val}.mp4'
        else:
            name = self.file_prefix + f'_{rand_val}.mp4'
        return name


class NoopEnv(MyWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        MyWrapper.__init__(self, env)
        self.noop_actions_taken = []

    def reset(self):
        obs = self.env.reset()
        noops = random.randint(0, 30)
        done = True
        self.noop_actions_taken = []
        for _ in range(noops):
            obs, _, done, info = self.env.step(0)
            if 'sticky_env.executed_action' in info:
                self.noop_actions_taken.append(info['sticky_env.executed_action'])
            else:
                self.noop_actions_taken.append(0)
            if done:
                self.noop_actions_taken = []
                obs = self.env.reset()
        if done:
            self.noop_actions_taken = []
            obs = self.env.reset()
        return obs


def my_wrapper(env,
               clip_rewards=True,
               frame_resize_wrapper=None,
               scale_rewards=None,
               ignore_negative_rewards=False,
               sticky=True,
               skip=4,
               noops=False):
    assert 'NoFrameskip' in env.spec.id
    assert not (clip_rewards and scale_rewards), "Clipping and scaling rewards makes no sense"
    if scale_rewards is not None:
        env = ScaledRewardEnv(env, scale_rewards)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if sticky:
        env = StickyActionEnv(env)
    if noops:
        env = NoopEnv(env)
    if ignore_negative_rewards:
        env = IgnoreNegativeRewardEnv(env)
    env = MaxAndSkipEnv(env, skip=skip)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = frame_resize_wrapper(env)
    return env


class ResetDemoInfo:
    def __init__(self, env, idx):
        self.env = env
        self.idx = idx
        starting_points = self.env.recursive_getattr(f'starting_point_{idx}')
        all_starting_points = flatten_lists(mpi.COMM_WORLD.allgather(starting_points))
        self.min_starting_point = min(all_starting_points)
        self.max_starting_point = max(all_starting_points)
        self.nrstartsteps = (self.max_starting_point - self.min_starting_point) + 1
        assert (self.nrstartsteps >= 1)
        self.max_max_starting_point = self.max_starting_point
        self.starting_point_success = np.zeros(self.max_starting_point+10000)
        self.infos = []


class ResetManager(MyWrapper):
    def __init__(self, env, move_threshold=0.2, steps_per_demo=1024):
        super(ResetManager, self).__init__(env)
        self.n_demos = self.recursive_getattr('n_demos')[0]
        self.demos = [ResetDemoInfo(self.env, idx) for idx in range(self.n_demos)]
        self.counter = 0
        self.move_threshold = move_threshold
        self.steps_per_demo = steps_per_demo

    def proc_infos(self):
        for idx in range(self.n_demos):
            epinfos = [info['episode'] for info in self.demos[idx].infos if 'episode' in info]

            if hvd.size() > 1:
                epinfos = flatten_lists(mpi.COMM_WORLD.allgather(epinfos))

            new_sp_wins = {}
            new_sp_counts = {}
            for epinfo in epinfos:
                sp = epinfo['starting_point']
                if sp in new_sp_counts:
                    new_sp_counts[sp] += 1
                    if epinfo['as_good_as_demo']:
                        new_sp_wins[sp] += 1
                else:
                    new_sp_counts[sp] = 1
                    if epinfo['as_good_as_demo']:
                        new_sp_wins[sp] = 1
                    else:
                        new_sp_wins[sp] = 0

            for sp, wins in new_sp_wins.items():
                self.demos[idx].starting_point_success[sp] = np.cast[np.float32](wins)/new_sp_counts[sp]

            # move starting point, ensuring at least 20% of workers are able to complete the demo
            csd = np.argwhere(np.cumsum(self.demos[idx].starting_point_success) /
                              self.demos[idx].nrstartsteps >= self.move_threshold)
            if len(csd) > 0:
                new_max_start = csd[0][0]
            else:
                new_max_start = np.minimum(self.demos[idx].max_starting_point + 100,
                                           self.demos[idx].max_max_starting_point)
            n_points_to_shift = self.demos[idx].max_starting_point - new_max_start
            self.decrement_starting_point(n_points_to_shift, idx)
            self.demos[idx].infos = []

    def decrement_starting_point(self, n_points_to_shift, idx):
        self.env.decrement_starting_point(n_points_to_shift, idx)
        starting_points = self.env.recursive_getattr(f'starting_point_{idx}')
        all_starting_points = flatten_lists(mpi.COMM_WORLD.allgather(starting_points))
        self.demos[idx].max_starting_point = max(all_starting_points)

    def set_max_starting_point(self, starting_point, idx):
        n_points_to_shift = self.demos[idx].max_starting_point - starting_point
        self.decrement_starting_point(n_points_to_shift, idx)

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        for info in infos:
            self.demos[info['idx']].infos.append(info)
        self.counter += 1
        if (self.counter > (self.demos[0].max_max_starting_point - self.demos[0].max_starting_point) / 2 and
                self.counter % (self.steps_per_demo * self.n_demos) == 0):
            self.proc_infos()
        return obs, rews, news, infos

    def step_wait(self):
        obs, rews, news, infos = self.env.step_wait()
        for info in infos:
            self.demos[info['idx']].infos.append(info)
        self.counter += 1
        if self.counter > (self.demos[0].max_max_starting_point - self.demos[0].max_starting_point) / 2 and \
                self.counter % (self.steps_per_demo * self.n_demos) == 0:
            self.proc_infos()
        return obs, rews, news, infos


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            logger.debug(f'[{os.getpid()}] received command: {cmd}')
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                    if 'skip_env.executed_actions' in info:
                        info['skip_env.executed_actions'].append(-1)
                        info['skip_env.executed_actions'].append(env.recursive_getattr('noop_actions_taken'))
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.action_space, env.observation_space))
            elif cmd == 'get_goal_space':
                remote.send(env.recursive_getattr('goal_space'))
            elif cmd == 'get_history':
                senv = env
                while not hasattr(senv, 'get_history'):
                    senv = senv.env
                remote.send(senv.get_history(data))
            elif cmd == 'recursive_getattr':
                remote.send(env.recursive_getattr(data))
            elif cmd == 'recursive_setattr':
                env.recursive_setattr(*data)
            elif cmd == 'print_data':
                remote.send('done')
            elif cmd == 'recursive_call_method':
                remote.send(env.recursive_call_method(*data))
            elif cmd == 'recursive_call_method_ignore_return':
                env.recursive_call_method_ignore_return(*data)
            elif cmd == 'decrement_starting_point':
                env.decrement_starting_point(*data)
            elif cmd == 'get_current_cell':
                cell = env.get_current_cell()
                remote.send(cell)
            elif cmd == 'set_archive':
                env.set_archive(data)
            elif cmd == 'set_selector':
                env.set_selector(data)
            elif cmd == 'init_archive':
                archive = env.init_archive()
                remote.send(archive)
            else:
                raise NotImplementedError
    except Exception as e:
        remote.send(e)
        remote.close()
        raise


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


class SubprocVecEnv(object):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.remotes[0].send(('recursive_getattr', 'goal_space'))
        self.goal_space = self.remotes[0].recv()
        self.waiting = False

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, nsteps):
        for remote in self.remotes:
            remote.send(('get_history', nsteps))
        results = [remote.recv() for remote in self.remotes]
        obs, acts, dones = zip(*results)
        obs = np.stack(obs)
        acts = np.stack(acts)
        dones = np.stack(dones)
        return obs, acts, dones

    def recursive_getattr(self, name):
        for remote in self.remotes:
            remote.send(('recursive_getattr', name))
        return [remote.recv() for remote in self.remotes]

    def decrement_starting_point(self, n, idx):
        for remote in self.remotes:
            remote.send(('decrement_starting_point', (n, idx)))

    def set_archive(self, archive):
        for remote in self.remotes:
            remote.send(('set_archive', archive))

    def set_selector(self, selector):
        for remote in self.remotes:
            remote.send(('set_selector', selector))

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, cur_noops=0, n_envs=1, save_path=None):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.override_num_noops = None
        self.noop_action = 0
        self.cur_noops = cur_noops - n_envs
        self.n_envs = n_envs
        self.save_path = save_path
        self.score = 0
        self.levels = 0
        self.in_treasure = False
        self.rewards = []
        self.actions = []
        self.rng = np.random.RandomState(os.getpid())
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def choose_noops(self):
        n_done = []
        for i in range(0, 31):
            json_path = self.save_path + '/' + str(i) + '.json'
            n_data = 0
            try:
                import json
                n_data = len(json.load(open(json_path)))
            except FileNotFoundError:
                pass
            n_done.append(n_data)

        weights = np.array([(max(0.00001, 5 - e)) for e in n_done])
        div = np.sum(weights)
        if div == 0:
            weights = np.array([1 for _ in n_done])
            div = np.sum(weights)
        return np.random.choice(list(range(len(n_done))), p=weights/div)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        noops = self.choose_noops()
        obs = self.env.reset(**kwargs)
        assert noops >= 0
        self.cur_noops = noops
        self.env.cur_noops = noops
        self.score = 0
        self.rewards = []
        self.actions = []
        self.in_treasure = False
        self.levels = 0
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        a, reward, done, c = self.env.step(ac)
        from collections import Counter
        in_treasure = Counter(a[:, :, 2].flatten()).get(136, 0) > 20_000
        if self.in_treasure and not in_treasure:
            self.levels += 1
        self.in_treasure = in_treasure
        if reward <= -999000:
            reward = 0
        self.actions.append(ac)
        self.rewards.append(reward)
        self.score += reward

        if self.save_path and done:
            json_path = self.save_path + '/' + str(self.cur_noops) + '.json'
            if 'episode' not in c:
                c['episode'] = {}
            if 'write_to_json' not in c['episode']:
                c['episode']['write_to_json'] = []
                c['episode']['json_path'] = json_path
            c['episode']['l'] = len(self.actions)
            c['episode']['r'] = self.score
            c['episode']['as_good_as_demo'] = False
            c['episode']['starting_point'] = 0
            c['episode']['idx'] = 0
            c['episode']['write_to_json'].append({'score': self.score, 'levels': self.levels,
                                                  'actions': [int(e) for e in self.actions]})
        return a, reward, done, c
