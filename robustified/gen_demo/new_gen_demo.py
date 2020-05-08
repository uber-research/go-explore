
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


import imageio
import gzip
import bz2
from PIL import Image, ImageFont, ImageDraw

import gen_demo.atari_demo as atari_demo
import gen_demo.atari_demo.wrappers

from goexplore_py.goexplore import *
import goexplore_py.complex_fetch_env
import cv2

from tqdm import tqdm
import argparse
import multiprocessing
import functools

FETCH_SZ = (752, 912)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FONT = ImageFont.truetype(DIR_PATH + "/../atari_reset/helvetica.ttf", 28)

compress = gzip

class MujocoDemo(gym.Wrapper):
    """
        Records actions taken, creates checkpoints, allows time travel, restoring and saving of states
    """

    def __init__(self, env, save_every_k=100):
        super(MujocoDemo, self).__init__(env)
        self.action_space = env.action_space
        self.save_every_k = save_every_k
        self.max_time_travel_steps = 10000

    def step(self, action):
        if self.steps_in_the_past > 0:
            self.restore_past_state()

        if len(self.done)>0 and self.done[-1]:
            obs = self.obs[-1]
            reward = 0
            done = True
            info = None

        else:
            self.lives.append(1)

            obs, reward, done, info = self.env.step(action)

            self.actions.append(action)
            self.obs.append(obs)
            self.rewards.append(reward)
            self.done.append(done)
            self.info.append(info)

        # periodic checkpoint saving
        if not done:
            if (len(self.checkpoint_action_nr)>0 and len(self.actions) >= self.checkpoint_action_nr[-1] + self.save_every_k) \
                    or (len(self.checkpoint_action_nr)==0 and len(self.actions) >= self.save_every_k):
                self.save_checkpoint()

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.actions = []
        self.lives = []
        self.checkpoints = []
        self.checkpoint_action_nr = []
        self.obs = [obs]
        self.rewards = []
        self.done = [False]
        self.info = [None]
        self.steps_in_the_past = 0
        return obs

    def save_to_file(self, file_name):
        dat = {'actions': self.actions, 'checkpoints': self.checkpoints, 'checkpoint_action_nr': self.checkpoint_action_nr,
               'rewards': self.rewards, 'lives': self.lives, 'obs': self.obs, 'target': self.env.cached_state.object_pos[0]}
        with open(file_name, "wb") as f:
            pickle.dump(dat, f)

    def load_from_file(self, file_name):
        self.reset()
        with open(file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat['actions']
        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.rewards = dat['rewards']
        self.lives = dat['lives']
        self.load_state_and_walk_forward()

    def save_checkpoint(self):
        chk_pnt = self.env.get_inner_state()
        self.checkpoints.append(chk_pnt)
        self.checkpoint_action_nr.append(len(self.actions))

    def restore_past_state(self):
        self.actions = self.actions[:-self.steps_in_the_past]
        while len(self.checkpoints)>0 and self.checkpoint_action_nr[-1]>len(self.actions):
            self.checkpoints.pop()
            self.checkpoint_action_nr.pop()
        self.load_state_and_walk_forward()
        self.steps_in_the_past = 0

    def load_state_and_walk_forward(self):
        if len(self.checkpoints)==0:
            self.env.reset()
            time_step = 0
        else:
            self.env.set_inner_state(self.checkpoints[-1])
            time_step = self.checkpoint_action_nr[-1]

        for a in self.actions[time_step:]:
            action = self.env.unwrapped._action_set[a]
            self.env.unwrapped.ale.act(action)

TREE = None

@functools.lru_cache(maxsize=2)
def get_traj(idx):
    traj = [idx]
    while traj[-1] > 0:
        traj.append(TREE.get_parent(traj[-1]))
    return np.array(list(reversed(traj)))

def cmp_traj(args):
    t1, t2 = args
    t1 = get_traj(t1['idx'])
    t2 = get_traj(t2['idx'])
    l = min(len(t1), len(t2))
    return sum(t1[:l] != t2[:l]) / l

def load_f(f):
    return pickle.load(compress.open(f, 'rb'))[:4]

def num_experience_in_f(f):
    return f, len(load_f(f)[0])

def get_traj_actions(exp_get, next_last, n_steps):
    cur_actions = []
    cur_cells = []
    with tqdm(total=n_steps) as t:
        while next_last is not None:
            if exp_get.action_at(next_last) is not None:
                cur_actions.append(exp_get.action_at(next_last))
                cur_cells.append(exp_get.cell_at(next_last))
            prev_offset = exp_get.prev_at(next_last)
            if prev_offset is None:
                break
            next_last = next_last - prev_offset
            t.update(1)

    return list(reversed(cur_actions)), list(reversed(cur_cells))

def load_f(f):
    return pickle.load(compress.open(f, 'rb'))[:4]

class ExperienceGetter:
    def __init__(self, experience_files):
        self.experience_files = experience_files
        self.cached_data = None
        self.cur_file = None

    def load_file_for_idx(self, idx):
        for file_info in self.experience_files:
            if idx >= file_info[0] and idx < file_info[1]:
                self.cur_file = file_info
                self.cached_data = load_f(file_info[-1])
                if len(self.cached_data) > 3:
                    to_fix = self.cached_data[3]
                    for i in reversed(range(len(to_fix))):
                        if to_fix[i] is None:
                            to_fix[i] = to_fix[i + 1]
                return
        assert False, f'Couldn\'t find file for {idx}'

    def get_at(self, idx, subidx):
        if self.cur_file is None or idx >= self.cur_file[1] or idx < self.cur_file[0]:
            self.load_file_for_idx(idx)
        return self.cached_data[subidx][idx - self.cur_file[0]]

    def action_at(self, idx):
        return self.get_at(idx, 1)

    def prev_at(self, idx):
        return self.get_at(idx, 0)

    def cell_at(self, idx):
        return self.get_at(idx, 3)

    def reward_at(self, idx):
        return self.get_at(idx, 2)

class RefDict:
    '''A dictionary for references. It is always assumed that '''
    def __init__(self):
        self.refs = {}

    def ref(self, from_, to):
        if to - 1 != from_:
            self.refs[to - 1] = []
            tmp = self.get(from_)
            tmp.append(to)
            self.refs[from_] = tmp

    def get(self, i):
        return self.refs.get(i, [i + 1]) or []

    def nodes(self):
        return {k: v for k, v in self.refs.items() if v is not None and len(v) > 1}

def sort_experience(name):
    ret = name.split('/')[-1].split('_')
    typ = 0
    if 'pre' in ret:
        typ = 1
    if 'post' in ret:
        typ = 2
    return ret[0], typ


def load_prev(args):
    (file, return_criteria) = args
    cur = pickle.load(compress.open(file, 'rb'))
    prev = cur[0]
    traj_lens = cur[-1]
    scores = cur[-2]
    cells = cur[-3]
    rewards = cur[2]
    done_info = []
    for i, (traj_len, score, cell, reward) in enumerate(zip(traj_lens, scores, cells, rewards)):
        # Note: meanings of cells: normally a tuple (state_split_rules, cell).
        # If the second element of the tuple is None, then the cell is a done cell
        # If instead the element is 0, then the cell is the same as the next
        # If the element is 1, then the code was run in a "don't save cells" mode, and the cell cannot be inferred. In those cases, the done cells above will still exist.
        should_return_done = True
        if return_criteria.get('done'):
            should_return_done &= (not isinstance(cell, int) and cell is not None and cell[1] is None)
        if return_criteria.get('object_pos') is not None:
            should_return_done &= (cell != 0 and cell is not None and cell[1] is not None and cell[1].object_pos == return_criteria['object_pos'])
        if return_criteria.get('reward') is not None:
            should_return_done &= (reward > 0)
        if should_return_done:
            done_info.append({'idx': i, 'score': score, 'traj_len': traj_len, 'reward': reward, 'cell': cell})
    return prev, done_info, len(prev), file

class RefTree:
    def __init__(self, refdict):
        self.child_tree = refdict
        self.parent_tree = {}
        nodes = refdict.nodes()
        for parent in nodes:
            for child in nodes[parent]:
                self.parent_tree[child] = parent

    def get_children(self, i):
        return self.child_tree.get(i)
    def get_parent(self, i):
        return self.parent_tree.get(i, i - 1)

def maybe_render(game, vid, env, total, reward, n_steps, done):
    if vid is None:
        return

    frame = env.render(mode='rgb_array')
    if 'fetch' not in args.game:
        frame = frame.repeat(4, axis=1).repeat(2, axis=0)
    else:
        frame = cv2.resize(frame, FETCH_SZ, interpolation=cv2.INTER_AREA)

    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)

    info_text = f"Frame: {n_steps}\nReward: {reward}\nTotal: {total}\nDone: {done}"
    if 'fetch' in args.game:
        info_text += f'\nGripped: {env.unwrapped._get_state().gripped_info}'
    draw.text((0, 0), info_text, (255, 255, 255), font=FONT)

    vid.append_data(np.array(frame))


def run(args):
    global TREE
    pool = multiprocessing.Pool(24)
    experience_files = [e for e in sorted(glob.glob(args.source + f'/*_experience.{args.compress}'), key=sort_experience) if 'thisisfake' not in e]

    if len(experience_files) == 0:
        print('No experience files!')
        return
    if int(experience_files[-1].split('/')[-1].split('_')[1]) < args.min_compute_steps:
        print('Not enough compute steps (yet?), exiting.')
        return

    references = RefDict()
    current_idx = 0
    all_dones = []
    exp_files_to_exp_idx_range = []
    done_args = {}
    if args.select_reward:
        done_args['reward'] = True
    if args.select_done:
        done_args['done'] = True
    if args.select_fetch_target:
        done_args['object_pos'] = (args.fetch_target_location,)
    for prev, dones, cur_len, file in tqdm(pool.imap(load_prev, [(e, done_args) for e in experience_files]), total=len(experience_files)):
        exp_files_to_exp_idx_range.append((current_idx, current_idx + cur_len, file))
        for e in dones:
            e['idx'] += current_idx
        all_dones += [e for e in dones if e['idx'] < args.max_frames]
        for e in tqdm(prev, leave=False):
            if e is not None and e != 1:
                references.ref(current_idx - e, current_idx)
            current_idx += 1
            if current_idx >= args.max_frames:
                break
        if current_idx >= args.max_frames:
            break

    reftree = RefTree(references)
    TREE = reftree
    exp_get = ExperienceGetter(exp_files_to_exp_idx_range)
    # Recreate the pool so it has the tree
    pool.close()
    pool = multiprocessing.Pool(24)
    os.makedirs(args.destination, exist_ok=True)

    selected_demos = []
    all_diffs = []
    taken_indices = set()
    for i in range(args.n_demos):
        if all_diffs != []:
            diff_means = np.mean(all_diffs, axis=0)
        else:
            diff_means = np.ones(shape=len(all_dones))

        def chosen_key(i):
            if i in taken_indices:
                return (-1000000, -1000000)
            return ((diff_means[i]) * all_dones[i]['score'], -all_dones[i]['traj_len'])
        chosen = max(range(len(all_dones)), key=chosen_key)
        selected_demos.append(all_dones[chosen])
        taken_indices.add(chosen)

        if i + 1 < args.n_demos:
            diffs = []
            get_traj(selected_demos[-1]['idx'])
            for cmp in tqdm(pool.imap(cmp_traj, [(selected_demos[-1], d) for d in all_dones]), total=len(all_dones)):
                diffs.append(cmp)
            all_diffs.append(diffs)

        list_of_actions, list_of_cells = get_traj_actions(exp_get, selected_demos[-1]['idx'], selected_demos[-1]['traj_len'])

        all_cells = []
        vid_file = f'{args.destination}/{len(selected_demos)}.mp4'
        demo_file = f'{args.destination}/{len(selected_demos)}.demo'

        vid = None
        if args.render:
            if 'fetch' not in args.game:
                fps = 24
            else:
                fps = 1 / args.fetch_nsubsteps / args.fetch_timestep
            vid = imageio.get_writer(vid_file, fps=fps)

        if 'fetch' not in args.game:
            env = gym.make(f'{args.game}NoFrameskip-v4')
            env.unwrapped.seed(0)
            env = atari_demo.wrappers.AtariDemo(env)
        else:
            # TODO: have a way to pass arguments to this
            kwargs = {}
            dargs = vars(args)
            for attr in dargs:
                if attr.startswith('fetch_'):
                    if attr == 'fetch_type':
                        kwargs['model_file'] = f'teleOp_{args.fetch_type}.xml'
                    elif attr != 'fetch_total_timestep':
                        kwargs[attr[len('fetch_'):]] = dargs[attr]

            env = goexplore_py.complex_fetch_env.ComplexFetchEnv(
                **kwargs
            )
            env = MujocoDemo(env, save_every_k=1)

        n_frames = 4
        if 'fetch' in args.game:
            n_frames = 1
        if 'SpaceInvaders' in args.game:
            n_frames = 3

        env.reset()
        total_score = 0
        done = False
        montezuma_level = 1
        in_treasure = False
        maybe_render(args.game, vid, env, total_score, 0, 0, done)
        for cur_frame, action in tqdm(enumerate(tqdm(list_of_actions))):
            if done:
                break

            for _ in range(n_frames):
                state, reward, d_, _ = env.step(action)
                if 'fetch' not in args.game:
                    env.obs = env.obs[:1]

                total_score += reward
                done = done or d_

                if done:
                    break
            maybe_render(args.game, vid, env, total_score, reward, cur_frame + 1, done)

            if args.game == 'MontezumaRevenge' and Counter(list(state[::2,::2, 2].flatten()))[136] > 5_000:
                in_treasure = True
            else:
                if in_treasure:
                    montezuma_level += 1
                in_treasure = False
            if args.game == 'MontezumaRevenge' and args.montezuma_max_level is not None and montezuma_level > args.montezuma_max_level:
                print('Breaking due to montezuma level')
                break

        if vid:
            vid.close()
        print(total_score)

        env.save_to_file(demo_file)
        data = pickle.load(open(demo_file, 'rb'))
        data['cells'] = all_cells
        pickle.dump(data, open(demo_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    current_group = parser

    # TODO: this boolarg logic is copied from goexplore_py/main.py. Extract it into a library!
    def boolarg(arg, *args, default=False, help='', neg=None, dest=None):
        def extract_name(a):
            dashes = ''
            while a[0] == '-':
                dashes += '-'
                a = a[1:]
            return dashes, a

        if dest is None:
            _, dest = extract_name(arg)

        group = current_group.add_mutually_exclusive_group()
        group.add_argument(arg, *args, dest=dest, action='store_true', help=help + (' (DEFAULT)' if default else ''), default=default)
        not_args = []
        for a in [arg] + list(args):
            dashes, name = extract_name(a)
            not_args.append(f'{dashes}no_{name}')
        if isinstance(neg, str):
            not_args[0] = neg
        if isinstance(neg, list):
            not_args = neg
        group.add_argument(*not_args, dest=dest, action='store_false', help=f'Opposite of {arg}' + (' (DEFAULT)' if not default else ''), default=default)

    def add_argument(*args, **kwargs):
        if 'help' in kwargs and kwargs.get('default') is not None:
            kwargs['help'] += f' (default: {kwargs.get("default")})'

        current_group.add_argument(*args, **kwargs)

    current_group = parser.add_argument_group('General')
    add_argument('--game', type=str, default='MontezumaRevenge')
    add_argument('--source', type=str, default=None)
    add_argument('--destination', type=str, default=None)
    add_argument('--n_demos', type=int, default=1)
    add_argument('--max_frames', type=int, default=1_000_000_000_000)
    add_argument('--montezuma_max_level', type=int, default=3)
    add_argument('--min_compute_steps', type=int, default=500_000_000)  # We error out if there are fewer than this many frames
    add_argument('--compress', type=str, default='gz')
    boolarg('--select_reward', default=False)
    boolarg('--select_done', default=False)
    boolarg('--select_fetch_target', default=False)
    boolarg('--render', default=False)

    current_group = parser.add_argument_group('Fetch')
    add_argument('--fetch_nsubsteps', type=int, default=20)
    add_argument('--fetch_timestep', type=float, default=0.002)
    add_argument('--fetch_total_timestep', type=float, default=None)
    add_argument('--fetch_incl_extra_full_state', action='store_true', default=False)
    add_argument('--fetch_state_is_pixels', action='store_true', default=False)
    add_argument('--fetch_force_closed_doors', action='store_true', default=False)
    add_argument('--fetch_include_proprioception', action='store_true', default=False)
    add_argument('--fetch_state_azimuths', type=str, default='145_215')
    add_argument('--fetch_type', type=str, default='boxes')
    add_argument('--fetch_target_location', type=str, default=None)
    add_argument('--fetch_state_wh', type=int, default=96)
    args = parser.parse_args()

    if args.compress == 'bz2':
        compress = bz2

    if '.demo' in args.source:
        sys.exit(0)

    if args.fetch_total_timestep is not None:
        args.fetch_timestep = args.fetch_total_timestep / args.fetch_nsubsteps
    run(args)
