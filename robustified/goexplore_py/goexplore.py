
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .explorers import *
from .montezuma_env import *
from .generic_atari_env import *
from .utils import *
import loky
import gzip
import bz2

compress = bz2
compress_suffix = '.bz2'
compress_kwargs = {}
n_digits = 20
DONE = None

perf_array = None


class MemInfo:
    E_VIRT_USE_MEAN = 0
    E_VIRT_USE_MAX = 1
    E_VIRT_USE_CUR = 2
    E_CPU_MEAN = 3
    E_CPU_MAX = 4
    E_CPU_CUR = 5
    ARRAY_SIZE = 6


class LPool:
    def __init__(self, n_cpus, maxtasksperchild=100):
        self.pool = loky.get_reusable_executor(n_cpus, timeout=100)

    def map(self, f, r):
        return self.pool.map(f, r)


class SyncPool:
    def __init__(self, n_cpus, maxtasksperchild=100):
        pass

    def map(self, f, r):
        res = []
        f_pickle = pickle.dumps(f)
        for e in r:
            e = pickle.loads(pickle.dumps(e))
            f2 = pickle.loads(f_pickle)
            res.append(f2(e))

        return res


def run_f_seeded(args):
    f, seed, args = args
    with use_seed(seed):
        return f(args)


class SeedPoolWrap:
    def __init__(self, pool):
        self.pool = pool

    def map(self, f, r):
        return self.pool.map(run_f_seeded, [(f, random.randint(0, 2**32 - 10), e) for e in r])


def seed_pool_wrapper(pool_class):
    def f(*args, **kwargs):
        return SeedPoolWrap(pool_class(*args, **kwargs))
    return f


class Discretizer:
    def __init__(self, attr, sort=False):
        self.attr = attr
        self.sort = sort
        self.cur_pos = None

    def apply(self, pos):
        self.cur_pos = pos
        obj = getattr(pos, self.attr)
        res = self.apply_rec(obj)
        self.cur_pos = None
        return res

    def apply_rec(self, obj):
        if isinstance(obj, (list, tuple, np.ndarray)):
            res = tuple(self.apply_rec(e) for e in obj)
            if self.sort:
                return tuple(sorted(res))
            else:
                return res
        return self.apply_scalar(obj)


class FetchConditionalObject:
    def __init__(self, object_pos, ifinpos, ifnotinpos):
        self.object_pos = object_pos
        self.ifinpos = ifinpos
        self.ifnotinpos = ifnotinpos

        assert self.ifinpos.attr == self.ifnotinpos.attr
        self.attr = self.ifinpos.attr

    def apply(self, obj):
        if set(obj.object_pos) == set(['0000']) or self.object_pos in obj.object_pos:
            return self.ifinpos.apply(obj)
        else:
            return self.ifnotinpos.apply(obj)


class GridDimension(Discretizer):
    def __init__(self, attr, div, offset=0, sort=False):
        super().__init__(attr, sort=sort)
        self.div = div
        self.offset = offset

    def apply_scalar(self, scalar):
        if scalar is None or isinstance(scalar, str):
            return scalar
        if self.div == 1:
            return scalar
        return int(np.floor((scalar + self.offset) / self.div))

    def __repr__(self):
        return f'GridDimension("{self.attr}", {self.div}, {self.offset})'

class SingleCell(Discretizer):
    def __init__(self, attr, value, sort=False):
        super().__init__(attr, sort=sort)
        self.value = value

    def apply(self, value):
        return self.value


class GridEquality(Discretizer):
    def __init__(self, attr, value, sort=False):
        super().__init__(attr, sort=sort)
        self.value = value

    def apply_scalar(self, scalar):
        return int(scalar == self.value)

    def __repr__(self):
        return f'GridEquality("{self.attr}", {self.value})'


class GridLambda(Discretizer):
    def __init__(self, attr, fn, sort=False):
        super().__init__(attr, sort=sort)
        self.fn = fn

    def apply_scalar(self, scalar):
        return self.fn(scalar)


class Cell:
    def __init__(self, score=-infinity, seen_times=0, chosen_times=0,
                 chosen_since_new=0, action_times=0, trajectory_len=infinity,
                 restore=None, exact_pos=None, real_cell=None, traj_last=None,
                 cell_frame=None):
        self.score = score

        self._seen_times = seen_times
        self._seen_times_diff = 0
        self._chosen_times = chosen_times
        self._chosen_times_diff = 0
        self._chosen_since_new = chosen_since_new
        self._chosen_since_new_diff = 0
        self._action_times = action_times
        self._action_times_diff = 0

        self.trajectory_len = trajectory_len
        self.restore = restore
        self.exact_pos = exact_pos
        self.real_cell = real_cell
        self.traj_last = traj_last
        self.cell_frame = cell_frame

    @property
    def chosen_times(self):
        return self._chosen_times

    @property
    def chosen_times_diff(self):
        return self._chosen_times_diff

    def inc_chosen_times(self, value):
        self._chosen_times += value
        self._chosen_times_diff += value

    def set_chosen_times(self, value):
        self._chosen_times = value
        self._chosen_times_diff = value

    @property
    def seen_times(self):
        return self._seen_times

    @property
    def seen_times_diff(self):
        return self._seen_times_diff

    def inc_seen_times(self, value):
        self._seen_times += value
        self._seen_times_diff += value

    def set_seen_times(self, value):
        self._seen_times = value
        self._seen_times_diff = value

    @property
    def chosen_since_new(self):
        return self._chosen_since_new

    @property
    def chosen_since_new_diff(self):
        return self._chosen_since_new_diff

    def inc_chosen_since_new(self, value):
        self._chosen_since_new += value
        self._chosen_since_new_diff += value

    def set_chosen_since_new(self, value):
        self._chosen_since_new = value
        self._chosen_since_new_diff = value

    @property
    def action_times(self):
        return self._action_times

    @property
    def action_times_diff(self):
        return self._action_times_diff

    def inc_action_times(self, value):
        self._action_times += value
        self._action_times_diff += value

    def set_action_times(self, value):
        self._action_times = value
        self._action_times_diff = value


@dataclass
class PosInfo:
    __slots__ = ['exact', 'cell', 'state', 'restore', 'frame']
    exact: tuple
    cell: tuple
    state: typing.Any
    restore: typing.Any
    frame: typing.Any


@dataclass
class TrajectoryElement:
    __slots__ = ['to', 'action', 'reward', 'done', 'real_pos']
    to: PosInfo
    action: int
    reward: float
    done: bool
    real_pos: MontezumaPosLevel


Experience = tuple

# ### Main

class RotatingSet:
    def __init__(self, M):
        self.max_size = M
        self.clear()

    def clear(self):
        self.set = set()
        self.list = collections.deque(maxlen=self.max_size)

    def add(self, e):
        if e in self.set:
            return
        if len(self.list) == self.max_size:
            self.set.remove(self.list[0])
        self.list.append(e)
        self.set.add(e)
        assert len(self.list) == len(self.set)

    def __iter__(self):
        for e in self.list:
            yield e

    def __len__(self):
        return len(self.list)


POOL = None
ENV = None

def get_env():
    return ENV


def get_downscale(args):
    f, cur_shape, cur_pix_val = args
    return imdownscale(f, cur_shape, cur_pix_val).flatten().tobytes()


@functools.lru_cache(maxsize=1)
def get_saved_grid(file):
    return pickle.load(compress.open(file, 'rb'))

class FormerGrids:
    def __init__(self, args):
        self.args = args
        self.cur_length = 0

    def _getfilename(self, i):
        return f'{self.args.base_path}/__grid_{i}.pickle{compress_suffix}'

    def append(self, elem):
        filename = self._getfilename(self.cur_length)
        assert not os.path.exists(filename)
        fastdump(elem, compress.open(filename, 'wb'))
        self.cur_length += 1

    def pop(self):
        assert self.cur_length >= 1
        filename = self._getfilename(self.cur_length - 1)
        res = get_saved_grid(filename)
        os.remove(filename)
        self.cur_length -= 1
        return res

    def __getitem__(self, item):
        if item < 0:
            item = self.cur_length + item
        return get_saved_grid(self._getfilename(item))

    def __len__(self):
        return self.cur_length



class Explore:
    def __init__(
            self, explorer_policy, cell_selector, env,
            grid_info: tuple,
            pool_class,
            args,
            important_attrs
    ):
        global POOL, ENV
        self.args = args
        self.important_attrs = important_attrs

        self.prev_checkpoint = None
        self.env_info = env
        self.make_env()
        self.pool_class = pool_class
        if self.args.reset_pool:
            POOL = self.pool_class(multiprocessing.cpu_count() * 2)
        else:
            POOL = self.pool_class(multiprocessing.cpu_count() * 2, maxtasksperchild=100)

        self.explorer = explorer_policy
        self.selector = cell_selector
        self.grid_info = grid_info
        self.grid = defaultdict(Cell)
        self.frames_true = 0
        self.frames_compute = 0
        self.start = None
        self.cycles = 0
        self.seen_level_1 = False
        self.dynamic_state_split_rules = (None, None, {})
        self.dynamic_state_frame_sets = defaultdict(set)
        self.random_recent_frames = RotatingSet(self.args.max_recent_frames)
        self.last_recompute_dynamic_state = -self.args.recompute_dynamic_state_every + self.args.first_compute_dynamic_state

        self.max_score = 0
        self.prev_len_grid = 0

        self.state = None
        self.reset()

        self.normal_frame_shape = (160, 210)
        cell_key = self.get_cell()
        self.grid[cell_key] = Cell()
        self.grid[cell_key].trajectory_len = 0
        self.grid[cell_key].score = 0
        self.grid[cell_key].exact_pos = self.get_pos()
        self.grid[cell_key].real_cell = self.get_real_cell()
        self.grid[cell_key].traj_last = 0
        self.grid[cell_key].cell_frame = self.get_frame(True)
        # Create the DONE cell
        self.grid[DONE] = Cell()
        self.selector.cell_update(cell_key, self.grid[cell_key])
        self.selector.cell_update(DONE, self.grid[DONE])
        self.real_grid = set()
        self.pos_cache = None
        self.former_grids = FormerGrids(args)
        self.former_grids.append(copy.deepcopy(self.grid))

        self.cur_experience = 1
        self.experience_prev_ids = [None]
        self.experience_actions = [None]
        self.experience_cells = [None]
        self.experience_rewards = [0]
        self.experience_scores = [0]
        self.experience_lens = [0]

        self.last_added_cell = 0

        self.real_cell = None

        self.gripped_info_count = {}

    def make_env(self):
        global ENV
        if ENV is None:
            ENV = self.env_info[0](**self.env_info[1])
            ENV.reset()

    def reset(self):
        self.real_cell = None
        self.pos_cache = None
        self.make_env()
        return ENV.reset()

    def step(self, action):
        self.real_cell = None
        self.pos_cache = None
        return ENV.step(action)

    def get_dynamic_repr(self, orig_state):
        if isinstance(orig_state, bytes):
            orig_state = RLEArray.frombytes(orig_state, dtype=np.uint8)
        orig_state = orig_state.to_np()
        dynamic_repr = []

        target_size, max_pix_val, cur_split_rules = self.dynamic_state_split_rules

        while True:
            if target_size is None:
                dynamic_repr.append(random.randint(1, self.args.first_compute_archive_size))
            else:
                state = imdownscale(orig_state, target_size, max_pix_val)
                dynamic_repr.append(state.tobytes())
            if dynamic_repr[-1] in cur_split_rules:
                target_size, max_pix_val, cur_split_rules = cur_split_rules[dynamic_repr[-1]]
            else:
                break

        return tuple(dynamic_repr)

    def try_split_frames(self, frames):
        n_processes = multiprocessing.cpu_count()
        tqdm.write('Decoding frames')
        frames = [RLEArray.frombytes(f, dtype=np.uint8) for f in frames]
        tqdm.write('Frames decoded')
        unif_ent_cache = {}
        def get_dist_score(dist):
            if len(dist) == 1:
                return 0.0
            from math import log
            def ent(dist):
                return -sum(log(e) * e for e in dist)
            def unif_ent(l):
                if l not in unif_ent_cache:
                    return ent([1 / l] * l)
                return unif_ent_cache[l]
            def norment(dist):
                return ent(dist) / unif_ent(len(dist))
            target_len = len(frames) * self.args.cell_split_factor
            return norment(dist) / np.sqrt(abs(len(dist) - target_len) / target_len + 1)

        unif_score_cache = {}
        def unif_dist_score(l):
            if l not in unif_score_cache:
                unif_score_cache[l] = get_dist_score([1 / l] * l)
            return unif_score_cache[l]

        best_shape = (random.randint(1, self.normal_frame_shape[0] - 1), random.randint(1, self.normal_frame_shape[1] - 1))
        best_pix_val = random.randint(2, 255)
        best_score = -infinity #get_dist_score([1 / len(frames) for _ in range(len(frames))])
        best_n = 0
        seen = set()

        # Intuition: we want our batch size to be such that it will be processed in two passes
        BATCH_SIZE = len(frames) // (n_processes // 2 + 1) + 1

        def proc_downscale(to_process, returns):
            while True:
                start_batch, cur_shape, cur_pix_val = to_process.get()
                if start_batch == -1:
                    return
                results = []
                for i in range(start_batch, min(len(frames), start_batch + BATCH_SIZE)):
                    results.append(imdownscale(frames[i].to_np(), cur_shape, cur_pix_val).tobytes())
                returns.put(results)

        tqdm.write('Creating processes')
        to_process = multiprocessing.Queue()
        returns = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=proc_downscale, args=(to_process, returns)) for _ in range(n_processes)]
        for p in processes:
            p.start()
        tqdm.write('Processes created')

        for _ in tqdm(range(self.args.split_iterations), desc='New representation'):
            cur_shape = best_shape
            cur_pix_val = best_pix_val
            while (cur_shape, cur_pix_val) in seen:
                cur_shape = list(best_shape)
                for idx in range(2):
                    while True:
                        cur_shape[idx] = np.random.geometric(min(1 / (best_shape[idx] + 1), 20 / self.normal_frame_shape[idx]))
                        if cur_shape[idx] >= 1 and cur_shape[idx] <= self.normal_frame_shape[idx] - 1:
                            break
                cur_shape = tuple(cur_shape)
                while True:
                    cur_pix_val = np.random.geometric(min(1 / best_pix_val, 1 / 12))
                    if cur_pix_val >= 2 and cur_pix_val <= 255:
                        break
            seen.add((cur_shape, cur_pix_val))

            for i in range(0, len(frames), BATCH_SIZE):
                to_process.put((i, cur_shape, cur_pix_val))
            downscaled = []
            for _ in range(0, len(frames), BATCH_SIZE):
                downscaled += returns.get()

            dist = np.array(list(Counter(downscaled).values())) / len(frames)
            cur_score = get_dist_score(dist)

            if cur_score >= best_score:
                if cur_score > best_score:
                    tqdm.write(f'NEW BEST score: {cur_score} n: {len(dist)} shape:{cur_shape} {cur_pix_val}')
                best_score = cur_score
                best_shape = cur_shape
                best_n = len(dist)
                best_pix_val = cur_pix_val

        for i in range(n_processes):
            to_process.put((-1, None, None))
        for p in processes:
            try:
                p.join(1)
            except Exception:
                p.terminate()

        return best_shape, best_pix_val, best_n

    def maybe_split_dynamic_state(self):
        if (self.frames_compute - self.last_recompute_dynamic_state > self.args.recompute_dynamic_state_every or
                len(self.grid) > self.args.max_archive_size):
            if len(self.grid) > self.args.max_archive_size:
                tqdm.write('Recomputing representation because of archive size (should not happen too often)')
            self.save_checkpoint('_pre_recompute')
            self.last_recompute_dynamic_state = self.frames_compute

            tqdm.write('Recomputing state representation')
            best_shape, best_pix_val, best_n = self.try_split_frames(self.random_recent_frames)
            tqdm.write(f'Switching representation to {best_shape} with {best_pix_val} pixels ({best_n} / {len(self.random_recent_frames)})')
            if self.dynamic_state_split_rules[0] is None:
                self.grid = self.former_grids.pop()
            self.dynamic_state_split_rules = (best_shape, best_pix_val, {})

            self.random_recent_frames.clear()

            self.selector.clear_all_cache()
            self.former_grids.append(self.grid)
            self.grid = defaultdict(Cell)

            start = time.time()

            for grid_idx in tqdm(reversed(range(len(self.former_grids))), desc='recompute_grid'):
                tqdm.write('Loading grid')
                old_grid = self.former_grids[grid_idx]
                tqdm.write('Creating queues')
                n_processes = multiprocessing.cpu_count()
                to_process = multiprocessing.Queue()
                returns = multiprocessing.Queue()

                def iter_grid(grid_idx, old_grid):
                    tqdm.write('Iter grid')
                    in_queue = set()
                    has_had_timeout = [False]

                    def queue_process_min(min_size):
                        while len(in_queue) > min_size:
                            if has_had_timeout[0]:
                                cur = in_queue.pop()
                                _, old_key, new_key = get_repr(grid_idx, cur)
                                yield old_key, old_grid[old_key], new_key
                            else:
                                import queue
                                try:
                                    _, old_key, new_key = returns.get(timeout=5*60)
                                    if old_key in in_queue:
                                        in_queue.remove(old_key)
                                        yield old_key, old_grid[old_key], new_key
                                    else:
                                        tqdm.write(f'Warning: saw duplicate key: {old_key}')
                                except queue.Empty:
                                    has_had_timeout[0] = True
                                    tqdm.write('Warning: timeout in receiving from queue. Switching to 100% single threaded')

                    for k in tqdm(old_grid, desc='add_to_grid'):
                        for to_yield in queue_process_min(n_processes):
                            yield to_yield
                        if not has_had_timeout[0]:
                            to_process.put((grid_idx, k), timeout=60)
                        in_queue.add(k)

                    tqdm.write('Clear queue')
                    for to_yield in queue_process_min(0):
                        yield to_yield

                    tqdm.write('Done iter grid')

                def get_repr(i_grid, key):
                    frame = self.former_grids[i_grid][key].cell_frame
                    if frame is None or key is None:
                        return ((i_grid, key, key))
                    else:
                        return ((i_grid, key, self.get_dynamic_repr(frame)))

                def redo_repr():
                    while True:
                        i_grid, key = to_process.get()
                        if i_grid is None:
                            return
                        returns.put(get_repr(i_grid, key))

                tqdm.write('creating processes')
                processes = [multiprocessing.Process(target=redo_repr) for _ in range(n_processes)]
                tqdm.write('starting processes')
                for p in processes:
                    p.start()
                tqdm.write('processes started')
                for cell_key, cell, new_key in iter_grid(grid_idx, old_grid):
                    if new_key not in self.grid or self.should_accept_cell(self.grid[new_key], cell.score, cell.trajectory_len):
                        if new_key not in self.grid:
                            self.grid[new_key] = Cell()
                        self.grid[new_key].score = cell.score
                        self.grid[new_key].trajectory_len = cell.trajectory_len
                        self.grid[new_key].restore = cell.restore
                        self.grid[new_key].exact_pos = cell.exact_pos
                        self.grid[new_key].real_cell = cell.real_cell
                        self.grid[new_key].traj_last = cell.traj_last
                        self.grid[new_key].cell_frame = cell.cell_frame
                        if self.args.reset_cell_on_update:
                            self.grid[new_key].set_seen_times(cell.seen_times)
                            self.grid[new_key].set_chosen_times(cell.chosen_times)
                            self.grid[new_key].set_chosen_since_new(cell.chosen_since_new)
                            self.grid[new_key].set_action_times(cell.action_times)

                    if not self.args.reset_cell_on_update:
                        # NB: this is the ONLY place where accessing these attributes using _ is valid!!!
                        self.grid[new_key]._seen_times += cell.seen_times_diff
                        self.grid[new_key]._chosen_times += cell.chosen_times_diff
                        self.grid[new_key]._chosen_since_new += cell.chosen_since_new_diff
                        self.grid[new_key]._action_times += cell.action_times_diff

                    self.selector.cell_update(new_key, self.grid[new_key])

                tqdm.write('clearing processes')
                for _ in range(n_processes):
                    to_process.put((None, None), block=False)
                for p in tqdm(processes, desc='processes_clear'):
                    try:
                        p.join(timeout=1)
                    except Exception:
                        p.terminate()
                        p.join()
                tqdm.write('processes cleared')

            tqdm.write(f'Recomputing the grid took {time.time() - start} seconds')

            self.prev_len_grid = len(self.grid)
            tqdm.write(f'New size: {len(self.grid)}. Old size: {len(self.former_grids[-1])}')

            self.save_checkpoint('_post_recompute')

    def get_pos(self):
        if self.args.use_real_pos:
            return self.get_real_pos()
        else:
            if not self.pos_cache:
                if self.args.dynamic_state:
                    self.pos_cache = self.get_dynamic_repr(self.get_frame(False))
                else:
                    self.pos_cache = (self.get_frame(True),)
            return self.pos_cache

    def get_frame(self, asbytes):
        if not hasattr(ENV, 'state') or self.args.use_real_pos:
            return None
        frame = ENV.state[-1]
        if asbytes:
            return frame.tobytes()
        return frame

    def get_real_pos(self):
        return ENV.get_pos()

    def get_pos_info(self, include_restore=True):
        return PosInfo(
            self.get_pos() if self.args.use_real_pos else None,
            self.get_cell(),
            None,
            self.get_restore() if include_restore else None,
            self.get_frame(True) if self.args.dynamic_state else None
        )

    def get_restore(self):
        return ENV.get_restore()

    def restore(self, val):
        self.real_cell = None
        self.pos_cache = None
        self.make_env()
        ENV.restore(val)

    def get_real_cell(self):
        if self.real_cell is None:
            pos = self.get_real_pos()
            res = {}
            for dimension in self.grid_info:
                res[dimension.attr] = dimension.apply(pos)
            self.real_cell = pos.__class__(**res)
        return self.real_cell

    def get_cell(self):
        if self.args.use_real_pos:
            return self.get_real_cell()
        else:
            pos = self.get_pos()
            return pos

    def run_explorer(self, explorer, start_cell=None, max_steps=-1):
        trajectory = []
        while True:
            if ((max_steps > 0 and len(trajectory) >= max_steps)):
                break
            action = explorer.get_action(ENV)
            state, reward, done, _ = self.step(action)
            self.frames_true += 1
            self.frames_compute += 1
            trajectory.append(
                TrajectoryElement(
                    # initial_pos_info,
                    self.get_pos_info(),
                    action, reward, done,
                    self.get_real_cell(),
                )
            )
            if done:
                break
        return trajectory

    def run_seed(self, seed, start_cell=None, max_steps=-1):
        with use_seed(seed):
            self.explorer.init_seed()
            return self.run_explorer(self.explorer, start_cell, max_steps)

    def process_cell(self, info):
        # This function runs in a SUBPROCESS, and processes a single cell.
        cell_key, cell, seed, known_rooms, target_shape, max_pix = info.data
        assert cell_key != DONE
        self.env_info[0].TARGET_SHAPE = target_shape
        self.env_info[0].MAX_PIX_VALUE = max_pix
        self.frames_true = 0
        self.frames_compute = 0

        if cell.restore is not None:
            self.restore(cell.restore)
            self.frames_true += cell.trajectory_len
        else:
            assert cell.trajectory_len == 0, 'Cells must have a restore unless they are the initial state'
            self.reset()

        end_trajectory = self.run_seed(seed, max_steps=self.args.explore_steps)

        # # We are not done, check that doing nothing for self.args.ignore_death steps won't kill us.
        if self.args.ignore_death > 0:
            if not end_trajectory[-1].done:
                assert self.args.ignore_death == 1
                end_trajectory += self.run_explorer(DoNothingExplorer(), max_steps=self.args.ignore_death)

        known_room_data = {}
        if len(ENV.rooms) > known_rooms:
            known_room_data = ENV.rooms

        return TimedPickle((cell_key, end_trajectory, self.frames_true, self.frames_compute, known_room_data), 'ret', enabled=info.enabled)

    def sample_only_cycle(self):
        # Choose a bunch of cells, send them to the workers for processing, then combine the results.
        # A lot of what this function does is only aimed at minimizing the amount of data that needs
        # to be pickled to the workers, which is why it sets a lot of variables to None only to restore
        # them later.
        global POOL
        chosen_cells = []
        cell_keys = self.selector.choose_cell(self.grid, size=self.args.batch_size)
        for i, cell_key in enumerate(cell_keys):
            cell_copy = self.grid[cell_key]
            seed = random.randint(0, 2 ** 31)
            chosen_cells.append(TimedPickle((cell_key, cell_copy, seed,
                                             len(ENV.rooms), self.env_info[0].TARGET_SHAPE,
                                             self.env_info[0].MAX_PIX_VALUE), 'args', enabled=(i == 0 and False)))


        # NB: save some of the attrs that won't be necessary but are very large, and set them to none instead,
        #     this way they won't be pickled.
        cache = {}
        to_save = [
            'grid', 'real_grid', 'former_grids', 'experience_prev_ids', 'experience_actions', 'experience_cells', 'experience_rewards', 'experience_scores', 'experience_lens',
            'selector', 'dynamic_state_frame_sets',  'random_recent_frames', 'pool_class'
        ]
        for attr in to_save:
            cache[attr] = getattr(self, attr)
            setattr(self, attr, None)

        trajectories = [e.data for e in POOL.map(self.process_cell, chosen_cells)]
        if self.args.reset_pool and (self.cycles + 1) % 100 == 0:
            POOL.close()
            POOL.join()
            POOL = None
            gc.collect()
            POOL = self.pool_class(self.args.n_cpus)
        chosen_cells = [e.data for e in chosen_cells]

        for attr, v in cache.items():
            setattr(self, attr, v)

        # Note: we do this now because starting here we're going to be concatenating the trajectories
        # of these cells, and they need to remain the same!
        chosen_cells = [(k, copy.copy(c), s, n, shape, pix) for k, c, s, n, shape, pix in chosen_cells]

        for ((cell_key, cell_copy, seed, _, _, _), (_, end_trajectory, ft, fc, known_rooms)) in zip(chosen_cells,
                                                                                                  trajectories):

            for i, elem in enumerate(end_trajectory):
                if i == len(end_trajectory) - self.args.ignore_death:
                    break

                if not elem.done:
                    if elem.to.frame is not None and (random.random() < self.args.recent_frame_add_prob):
                        self.random_recent_frames.add(elem.to.frame)


    def run_cycle(self):
        # Choose a bunch of cells, send them to the workers for processing, then combine the results.
        # A lot of what this function does is only aimed at minimizing the amount of data that needs
        # to be pickled to the workers, which is why it sets a lot of variables to None only to restore
        # them later.
        global POOL
        if self.start is None:
            self.start = time.time()

        if self.args.dynamic_state:
            self.maybe_split_dynamic_state()

        self.cycles += 1
        chosen_cells = []
        cell_keys = self.selector.choose_cell(self.grid, size=self.args.batch_size)
        for i, cell_key in enumerate(cell_keys):
            cell_copy = self.grid[cell_key]
            seed = random.randint(0, 2 ** 31)
            chosen_cells.append(TimedPickle((cell_key, cell_copy, seed,
                                             len(ENV.rooms), self.env_info[0].TARGET_SHAPE,
                                             self.env_info[0].MAX_PIX_VALUE), 'args', enabled=(i == 0 and False)))


        # NB: save some of the attrs that won't be necessary but are very large, and set them to none instead,
        #     this way they won't be pickled.
        cache = {}
        to_save = [
            'grid', 'real_grid', 'former_grids', 'experience_prev_ids', 'experience_actions', 'experience_cells', 'experience_rewards', 'experience_scores', 'experience_lens',
            'selector', 'dynamic_state_frame_sets',  'random_recent_frames', 'pool_class'
        ]
        for attr in to_save:
            cache[attr] = getattr(self, attr)
            setattr(self, attr, None)

        trajectories = [e.data for e in POOL.map(self.process_cell, chosen_cells)]
        if self.args.reset_pool and (self.cycles + 1) % 100 == 0:
            POOL.close()
            POOL.join()
            POOL = None
            gc.collect()
            POOL = self.pool_class(self.args.n_cpus)
        chosen_cells = [e.data for e in chosen_cells]

        for attr, v in cache.items():
            setattr(self, attr, v)

        # Note: we do this now because starting here we're going to be concatenating the trajectories
        # of these cells, and they need to remain the same!
        chosen_cells = [(k, copy.copy(c), s, n, shape, pix) for k, c, s, n, shape, pix in chosen_cells]
        cells_to_reset = set()

        for ((cell_key, cell_copy, seed, _, _, _), (_, end_trajectory, ft, fc, known_rooms)) in zip(chosen_cells,
                                                                                                  trajectories):
            self.frames_true += ft
            self.frames_compute += fc
            seen_cells = set([cell_key])

            for k in known_rooms:
                if k not in ENV.rooms:
                    ENV.rooms[k] = known_rooms[k]

            start_cell = self.grid[cell_key]
            start_cell.inc_chosen_times(1)
            start_cell.inc_chosen_since_new(1)
            start_cell.inc_seen_times(1)
            self.selector.cell_update(cell_key, start_cell)
            cur_score = cell_copy.score
            prev_id = cell_copy.traj_last
            potential_cell = start_cell
            old_potential_cell_key = cell_key
            for i, elem in enumerate(end_trajectory):
                if 'fetch' in self.args.game:
                    self.gripped_info_count[elem.to.exact.gripped_info] = self.gripped_info_count.get(elem.to.exact.gripped_info, 0) + 1
                self.experience_prev_ids.append(self.cur_experience - prev_id)
                self.experience_actions.append(elem.action)
                possible_experience_cell = (self.dynamic_state_split_rules, (DONE if elem.done else elem.to.cell))
                if not self.args.save_cells and not elem.done:
                    possible_experience_cell = 1
                if len(self.experience_cells) > 0 and self.experience_cells[-1] == possible_experience_cell:
                    self.experience_cells[-1] = 0
                self.experience_cells.append(possible_experience_cell)
                self.experience_rewards.append(elem.reward)
                self.experience_scores.append(elem.reward + cur_score)
                self.experience_lens.append(cell_copy.trajectory_len + i + 1)
                prev_id = self.cur_experience
                self.cur_experience += 1

                if i == len(end_trajectory) - self.args.ignore_death:
                    break

                potential_cell_key = elem.to.cell
                if elem.done:
                    potential_cell_key = DONE
                else:
                    if elem.to.frame is not None and (random.random() < self.args.recent_frame_add_prob): #or len(self.random_recent_frames) < self.random_recent_frames.max_size):
                        self.random_recent_frames.add(elem.to.frame)

                if not self.args.use_real_pos:
                    self.real_grid.add(elem.real_pos)

                if not isinstance(potential_cell_key, tuple) and potential_cell_key != DONE and potential_cell_key.level > 0:
                    self.seen_level_1 = True

                was_in_grid = True
                if potential_cell_key != old_potential_cell_key:
                    was_in_grid = potential_cell_key in self.grid
                    potential_cell = self.grid[potential_cell_key]
                    if potential_cell_key not in seen_cells:
                        seen_cells.add(potential_cell_key)
                        potential_cell.inc_seen_times(1)
                        if was_in_grid:
                            self.selector.cell_update(potential_cell_key, potential_cell)
                        else:
                            self.last_added_cell = self.frames_compute

                old_potential_cell_key = potential_cell_key
                full_traj_len = cell_copy.trajectory_len + i + 1
                cur_score += elem.reward

                potential_cell.inc_action_times(1)

                # Note: the DONE element should have a 0% chance of being selected, so OK to add the cell if it is in the DONE state.
                if (elem.to.restore is not None or potential_cell_key == DONE) and self.should_accept_cell(potential_cell, cur_score, full_traj_len):
                    if self.args.use_real_pos:
                        self.real_grid.add(elem.real_pos)

                    start_cell.set_chosen_since_new(0)
                    cells_to_reset.add(potential_cell_key)
                    potential_cell.trajectory_len = full_traj_len
                    potential_cell.restore = elem.to.restore
                    assert potential_cell.restore is not None or potential_cell_key == DONE
                    potential_cell.score = cur_score
                    if cur_score > self.max_score:
                        self.max_score = cur_score
                    potential_cell.real_cell = elem.real_pos
                    if self.args.use_real_pos:
                        potential_cell.exact_pos = elem.to.exact
                    potential_cell.traj_last = self.cur_experience - 1
                    potential_cell.cell_frame = elem.to.frame

                    self.selector.cell_update(potential_cell_key, potential_cell)

        if self.args.reset_cell_on_update:
            for cell_key in cells_to_reset:
                self.grid[cell_key].set_chosen_times(0)
                self.grid[cell_key].set_chosen_since_new(0)
                self.grid[cell_key].set_seen_times(0)
                self.grid[cell_key].set_action_times(0)

        return [(k) for k, c, s, n, shape, pix in chosen_cells], trajectories

    def should_accept_cell(self, potential_cell, cur_score, full_traj_len):
        if self.args.prob_override > 0.0000000001 and random.random() < self.args.prob_override:
            return True
        if self.args.optimize_score:
            return (cur_score > potential_cell.score or
                    (full_traj_len < potential_cell.trajectory_len and
                     cur_score == potential_cell.score))
        return full_traj_len < potential_cell.trajectory_len

    def save_checkpoint(self, suffix=''):
        # Quick bookkeeping, printing update
        seen_level_1 = self.seen_level_1
        filename = f'{self.args.base_path}/{self.frames_true:0{n_digits}}_{self.frames_compute:0{n_digits}}{suffix}'

        def print_sorted_keys(items):
            items = sorted(items.items(), key=lambda x: str(x[0]))
            return f'{{{", ".join(str(k) + ": " + str(v) for k, v in items)}}}'

        for attr in self.important_attrs:
            tqdm.write(f'Cells at {attr}: {print_sorted_keys(Counter(getattr(e, attr) for e in self.real_grid))}')
        tqdm.write(f'Max score: {max(e.score for e in self.grid.values())}')
        tqdm.write(f'Compute cells: {len(self.grid)}')

        # Save pictures
        if self.args.save_pictures or self.args.save_item_pictures or self.args.save_prob_pictures:
            # Show normal grid
            if self.args.save_pictures or self.args.save_item_pictures:
                get_env().render_with_known(
                    list(self.real_grid), self.args.resolution,
                    show=False, filename=filename + '.png',
                    get_val=lambda x: 1,
                    combine_val=lambda x, y: x + y
                )

            if not self.args.use_real_pos:
                object_combinations = sorted(
                    set(e.real_cell.score for e in self.grid.values() if e.real_cell is not None))
                for obj in object_combinations:
                    grid_at_obj = [e.real_cell for e in self.grid.values() if
                                   e.real_cell is not None and e.real_cell.score == obj]
                    get_env().render_with_known(
                        grid_at_obj, self.args.resolution,
                        show=False, filename=filename + f'_object_{obj}.png',
                        get_val=lambda x: 1,
                        combine_val=lambda x, y: x + y
                    )

            # Show probability grid
            if (self.args.use_real_pos and self.args.save_pictures) or self.args.save_prob_pictures:
                self.selector.set_ranges(list(self.grid.keys()))
                possible_scores = sorted(set(e.score for e in self.grid))
                total = np.sum(
                    [self.selector.get_weight(x, self.grid[x], possible_scores, self.grid) for x in self.grid])
                get_env().render_with_known(
                    list(self.grid.keys()), self.args.resolution,
                    show=False, filename=filename + '_prob.PNG',
                    combine_val=lambda x, y: x + y,
                    get_val=lambda x: self.selector.get_weight(x, self.grid[x], possible_scores,
                                                               self.grid) / total,
                )
            if self.prev_checkpoint and self.args.clear_old_checkpoints:
                if not self.args.keep_item_pictures:
                    try:
                        os.remove(self.prev_checkpoint + '.png')
                    except FileNotFoundError:
                        # If it doesn't exists, we don't need to remove it.
                        pass
                if self.args.use_real_pos and not self.args.keep_prob_pictures:
                    try:
                        os.remove(self.prev_checkpoint + '_prob.PNG')
                    except FileNotFoundError:
                        # If it doesn't exists, we don't need to remove it.
                        pass

        # Save checkpoints
        grid_copy = {}
        for k, v in self.grid.items():
            grid_copy[k] = copy.copy(v)
            grid_copy[k].cell_frame = None
            grid_copy[k].restore = None
        fastdump(grid_copy, compress.open(filename + compress_suffix, 'wb', **compress_kwargs))

        # Clean up previous checkpoint.
        if self.prev_checkpoint and self.prev_checkpoint != filename and self.args.clear_old_checkpoints:
            os.remove(self.prev_checkpoint + compress_suffix)
        self.prev_checkpoint = filename

        # A much smaller file that should be sufficient for view folder, but not for restoring
        # the demonstrations. Should make view folder much faster.
        grid_set = {}
        for k, v in self.grid.items():
            grid_set[k] = v.score
        fastdump(grid_set, compress.open(filename + '_set' + compress_suffix, 'wb', **compress_kwargs))
        fastdump(self.real_grid, compress.open(filename + '_set_real' + compress_suffix, 'wb', **compress_kwargs))

        print(self.gripped_info_count)
        perf_info = {'gripped_info': self.gripped_info_count}
        for e in dir(MemInfo):
            if e[:2] == 'E_':
                perf_info[e] = perf_array[getattr(MemInfo, e)]
        fastdump(perf_info, compress.open(filename + '_perf' + compress_suffix, 'wb', **compress_kwargs))

        fastdump(
            (
                self.experience_prev_ids, self.experience_actions, self.experience_rewards, self.experience_cells, self.experience_scores, self.experience_lens
            ),
            compress.open(filename + '_experience' + compress_suffix, 'wb', **compress_kwargs))
        self.experience_prev_ids = []
        self.experience_actions = []
        self.experience_rewards = []
        self.experience_cells = []
        self.experience_scores = []
        self.experience_lens = []
