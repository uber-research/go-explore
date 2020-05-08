import numpy as np
from multiprocessing import Process, Pipe
import gym
from baselines.common.vec_env.subproc_vec_env import CloudpickleWrapper

class ClonedEnv(gym.Wrapper):
    def __init__(self, env, possible_actions_dict, best_action_dict, seed):
        gym.Wrapper.__init__(self, env)
        self.possible_actions_dict = possible_actions_dict
        self.best_action_dict = best_action_dict
        self.state = None
        self.rng = np.random.RandomState(seed)
        self.just_initialized = True
        self.l = 0
        self.r = 0

    def step(self, action=None):
        if self.state in self.possible_actions_dict:
            possible_actions = list(self.possible_actions_dict[self.state])
            action = possible_actions[self.rng.randint(len(possible_actions))]
            obs, reward, done, info = self.env.step(action)
            self.l += 1
            self.r += reward
            self.state = self.env.unwrapped._get_ram().tostring()
            if self.state in self.possible_actions_dict: # still in known territory
                info['possible_actions'] = self.possible_actions_dict[self.state]
                if self.state in self.best_action_dict:
                    info['best_action'] = self.best_action_dict[self.state]
            else:
                done = True
                past_l = self.l
                past_r = self.r
                self.l = 0
                self.r = 0
                if past_l > 0:
                    info['episode'] = {'r': past_r, 'l': past_l}
        else:
            raise Exception('stepping cloned env without resetting')

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs,info = obs
        else:
            info = {}

        self.state = self.env.unwrapped._get_ram().tostring()
        if self.state in self.best_action_dict:
            info['best_action'] = self.best_action_dict[self.state]
        for randop in range(self.rng.randint(30)): # randomize starting point
            obs, reward, done, info = self._step(None)

        if self.just_initialized:
            self.just_initialized = False
            for randops in range(self.rng.randint(50000)):  # randomize starting point further
                obs, reward, done, info = self._step(None)
                if done:
                    obs, info = self._reset()

        return obs, info

def get_best_actions_from_infos(infos):
    k = len(infos)
    best_actions = [0] * k
    action_masks = [1] * k
    for i in range(k):
        if 'best_action' in infos[i]:
            best_actions[i] = infos[i]['best_action']
            action_masks[i] = 0
    return best_actions, action_masks

def get_available_actions_from_infos(infos, n_actions):
    k = len(infos)
    best_actions = np.zeros((k,n_actions), dtype=np.uint8)
    action_masks = [1] * k
    for i in range(k):
        if 'possible_actions' in infos[i]:
            action_masks[i] = 0
            for j in infos[i]['possible_actions']:
                best_actions[i,j] = 1
    return best_actions, action_masks

def worker2(nr, remote, env_fn_wrapper, mode):
    env = env_fn_wrapper.x()
    while True:
        cmd,count = remote.recv()
        if cmd == 'step':
            obs = []
            rews = []
            dones = []
            infos = []
            for step in range(count):
                ob, reward, done, info = env.step(0) # action is ignored in ClonedEnv downstream
                if done:
                    ob = env.reset()
                    if isinstance(ob, tuple):
                        ob, new_info = ob
                        info.update(new_info)
                if 'episode' in info:
                    epinfo = info['episode']
                    print('simulator thread %d completed demo run with total return %d obtained in %d steps' % (nr, epinfo["r"], epinfo["l"]))
                obs.append(ob)
                rews.append(reward)
                dones.append(done)
                infos.append(info)
            if mode == 'best':
                best_actions, action_masks = get_best_actions_from_infos(infos)
            else:
                best_actions, action_masks = get_available_actions_from_infos(infos, env.action_space.n)
            remote.send((obs, rews, dones, best_actions, action_masks))
        elif cmd == 'reset':
            ob = env.reset()
            if isinstance(ob, tuple):
                ob,info = ob
            else:
                info = {}
            remote.send((ob,info))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError(str(cmd) + ' action not implemented in worker')

class ClonedVecEnv(object):
    def __init__(self, env_fns, mode='best'):
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=worker2, args=(nr, work_remote, CloudpickleWrapper(env_fn), mode))
            for (nr, work_remote, env_fn) in zip(range(self.nenvs), self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()
        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.steps_taken = 0

    def step(self, time_steps=128):
        for remote in self.remotes:
            remote.send(('step', time_steps))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, best_actions, action_masks = [np.stack(x) for x in zip(*results)]
        return obs, rews, dones, best_actions, action_masks

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        best_actions, action_masks = [np.stack(x) for x in get_best_actions_from_infos(infos)]
        return np.stack(obs), best_actions, action_masks

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

def make_cloned_vec_env(nenvs, env_id, possible_actions_dict, best_action_dict, wrappers, mode='best'):
    def make_env(rank):
        def env_fn():
            env = gym.make(env_id)
            env = ClonedEnv(env, possible_actions_dict, best_action_dict, rank)
            env = wrappers(env)
            return env
        return env_fn

    return ClonedVecEnv([make_env(i) for i in range(nenvs)], mode)

