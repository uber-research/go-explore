# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from .explorers import *
import goexplore_py.policies
from .montezuma_env import *
from .utils import *
import loky
from tensorflow import summary
from diverseExplorer import clipreward, IRonly, smoothReward

class LPool:
    def __init__(self, n_cpus, maxtasksperchild=100):
        self.pool = loky.get_reusable_executor(n_cpus, timeout=100)

    def map(self, f, r):
        return self.pool.map(f, r)


@dataclass
class GridDimension:
    attr: str
    div: int


@dataclass()
class ChainLink:
    __slots__ = ['start_cell', 'end_cell', 'seed']
    start_cell: typing.Any
    end_cell: typing.Any
    seed: int


@dataclass
class Cell:
    # The list of ChainLink that can take us to this place
    chain: typing.List[ChainLink] = copyfield([])
    seen: list = copyfield({})
    score: int = -infinity
    # Number of times this was chosen and seen
    seen_times: int = 0
    chosen_times: int = 0
    chosen_since_new: int = 0
    action_times: int = 0  # This is the number of action that led to this cell
    # Length of the trajectory
    trajectory_len: int = infinity
    # Saved restore state. In a purely deterministic environment,
    # this allows us to fast-forward to the end state instead
    # of replaying.
    restore: typing.Any = None
    # TODO: JH: This should not refer to a Montezuma-only data-structure
    exact_pos: MontezumaPosLevel = None
    trajectory: list = copyfield([])
    real_cell: MontezumaPosLevel = None


@dataclass
class PosInfo:
    __slots__ = ['exact', 'cell', 'state', 'restore']
    exact: tuple
    cell: tuple
    state: typing.Any
    restore: typing.Any


@dataclass
class TrajectoryElement:
    __slots__ = ['from_', 'to', 'action', 'reward', 'done', 'real_pos', 'obs']
    from_: PosInfo
    to: PosInfo
    action: int
    reward: float
    done: bool
    real_pos: MontezumaPosLevel
    obs: typing.Any



# ### Main


POOL = None
ENV = None
# GRID = None

def get_env():
    return ENV


class Explore:
    def __init__(
            self, explorer_policy, cell_selector, env,
            grid_info: tuple,
            explore_steps=50,
            ignore_death: int = 1,
            n_cpus=None,
            optimize_score=True,
            use_real_pos=True,
            prob_override=0.0,
            pool_class=multiprocessing.Pool,
            reset_pool=False,
            batch_size=100,
            reset_cell_on_update=False,
            with_domain = False,
            load_model = None
    ):
        global POOL, ENV
        self.env_info = env
        self.make_env()



        self.pool_class = pool_class
        self.reset_pool = reset_pool
        if self.reset_pool:
            POOL = self.pool_class(multiprocessing.cpu_count() * 2)
        else:
            POOL = self.pool_class(multiprocessing.cpu_count() * 2, maxtasksperchild=100)
        self.use_real_pos = use_real_pos

        self.n_cpus = n_cpus
        self.batch_size = batch_size
        self.explore_steps = explore_steps
        self.explorer = explorer_policy
        self.with_domain = with_domain

        if self.explorer.__repr__() == 'mlsh':
            if isinstance(get_env().observation_space, gym.spaces.Box):
                if with_domain:
                    y, x = get_env().env.unwrapped.observation_space.shape[0:2]
                    x = x * self.env_info[1]['x_repeat'] // grid_info[-1].div
                    y = y // grid_info[-1].div
                    domain_shape = (y, x, 1)
                    self.explorer.init_model(get_env(), domain_shape, masterPolicy=goexplore_py.policies.CnnPolicy_withDomain,
                                             subPolicies=goexplore_py.policies.CnnPolicy)
                    if load_model is not None:
                        self.explorer.master.load(f'{load_model}/master')
                        for sub in self.explorer.subs:
                            sub.model.load(f'{load_model}/{sub}')
                else:
                    self.explorer.init_model(get_env(), masterPolicy=goexplore_py.policies.CnnPolicy, subPolicies=goexplore_py.policies.CnnPolicy)
            else:
                raise Exception("Unkown observation space")
        elif self.explorer.__repr__() == 'ppo':
            if isinstance(get_env().observation_space, gym.spaces.Box):
                self.explorer.init_model(get_env(), policy=goexplore_py.policies.CnnPolicy)
            elif isinstance(get_env().observation_space, gym.spaces.MultiBinary):
                self.explorer.init_model(get_env(), policy=goexplore_py.policies.MlpPolicy)
            else:
                raise Exception("Unkown observation space")

        self.selector = cell_selector
        self.grid_info = grid_info
        self.grid = defaultdict(Cell)
        self.ignore_death = ignore_death
        self.frames_true = 0
        self.frames_compute = 0
        self.start = None
        self.cycles = 0
        self.seen_level_1 = False
        self.optimize_score = optimize_score
        self.prob_override = prob_override

        self.state = None
        self.reset()

        self.grid[self.get_cell()].trajectory_len = 0
        self.grid[self.get_cell()].score = 0
        self.grid[self.get_cell()].exact_pos = self.get_pos()
        self.grid[self.get_cell()].real_cell = self.get_real_cell()
        self.grid[self.get_cell()].seen_times = 1
        self.real_grid = set()
        self.pos_cache = None
        self.reset_cell_on_update = reset_cell_on_update
        self.summary = []
        self.IR = 0
        self.dones = 0


        def empty_room():
            y, x = get_env().env.unwrapped.observation_space.shape[0:2]
            x = x * self.env_info[1]['x_repeat'] // grid_info[-1].div
            y = y // grid_info[-1].div
            domain_shape = (y, x,1)
            return np.zeros(domain_shape, dtype=np.uint8)

        self.domain_knowledge = defaultdict(empty_room)
        if with_domain:

            cell = self.grid[self.get_cell()].real_cell
            self.domain_knowledge[0][cell.y][cell.x] += 1



    def make_env(self):
        global ENV
        if ENV is None:
            ENV = self.env_info[0](**self.env_info[1])

    def reset(self):
        self.pos_cache = None
        self.make_env()
        return ENV.reset()

    def step(self, action):
        self.pos_cache = None
        return ENV.step(action)

    def get_pos(self):
        if self.use_real_pos:
            return self.get_real_pos()
        else:
            if not self.pos_cache:
                self.pos_cache = (ENV.state[-1].reshape((ENV.state[-1].size,)).tobytes(),)
            return self.pos_cache

    def get_real_pos(self):
        return ENV.get_pos()

    def get_pos_info(self, include_restore=True):
        return PosInfo(self.get_pos() if self.use_real_pos else None, self.get_cell(), None, self.get_restore() if include_restore else None)

    def get_restore(self):
        return ENV.get_restore()

    def restore(self, val):
        self.make_env()
        self.state = ENV.restore(val)

    def get_real_cell(self):
        pos = self.get_real_pos()
        res = {}
        for dimension in self.grid_info:
            value = getattr(pos, dimension.attr)

            if dimension.div == 1:
                res[dimension.attr] = value
            else:
                res[dimension.attr] = (int(value / dimension.div))
        return pos.__class__(**res)

    def get_cell(self):
        if self.use_real_pos:
            return self.get_real_cell()
        else:
            pos = self.get_pos()
            return pos

    def run_explorer(self, explorer, start_cell=None, max_steps=-1):
        # global GRID
        explorer.init_trajectory(self.state, self.domain_knowledge[self.get_real_cell().room])
        trajectory = []
        seen_cells = set()
        while True:
            initial_pos_info = self.get_pos_info(include_restore=False)
            if ((max_steps > 0 and len(trajectory) >= max_steps) or
                    initial_pos_info.cell == start_cell):
                break

            action = explorer.get_action(self.state, ENV) # doesn't use the state, state is instead given in .seen_state()
            if not isinstance(action, int):
                action = action.squeeze()
            self.state, reward, done, _ = self.step(action)

            self.frames_true += 1
            self.frames_compute += 1
            trajectory.append(
                TrajectoryElement(
                    initial_pos_info,
                    self.get_pos_info(),
                    action, reward, done,
                    self.get_real_cell(), None
                )
            )

            #assert trajectory[-1].to.restore is not None, "Failed to assign restore in trajectory"
            if explorer.__repr__() == "mlsh" or explorer.__repr__() == "ppo":

                e = {'done': done, 'observation': self.state, 'domain': self.domain_knowledge[trajectory[-1].real_pos.room]}
                # if (max_steps > 0 and len(trajectory) >= max_steps):
                #     e['done'] = 1

                # if trajectory[-1].to.cell not in GRID and trajectory[-1].to.cell not in seen_cells:
                #     e['reward'] = 1 + np.clip(reward, -1, 1)
                #     explorer.seen_state(e)
                #     self.IR += 1
                #     seen_cells.add(trajectory[-1].to.cell)
                # else:
                #     e['reward'] = 0 + np.clip(reward, -1, 1)
                #     explorer.seen_state(e)
                reward = smoothReward(trajectory[-1].to.cell, reward, self.grid, seen_cells)
                self.IR += reward
                e['reward'] = reward
                seen_cells.add(trajectory[-1].to.cell)
                explorer.seen_state(e)
            else:
                explorer.seen_state(trajectory[-1])
            if done:
                self.dones += 1
                break
        return trajectory

    def run_seed(self, seed, start_cell=None, max_steps=-1):
        with use_seed(seed):
            self.explorer.init_seed()
            return self.run_explorer(self.explorer, start_cell, max_steps)

    def process_cell(self, info):
        # This function runs in a SUBPROCESS, and processes a single cell.
        cell_key, cell, seed, known_rooms, target_shape, max_pix = info.data

        self.env_info[0].TARGET_SHAPE = target_shape
        self.env_info[0].MAX_PIX_VALUE = max_pix
        self.frames_true = 0
        self.frames_compute = 0

        if cell.restore is not None:
            self.restore(cell.restore)
            self.frames_true += cell.trajectory_len
        else:
            # TODO: implement recovering the restore from, say, the trajectory on the cell, so that this
            # isn't a problem anymore when recovering from a checkpoint.
            assert cell.trajectory_len == 0, 'Cells must have a restore unless they are the initial state'
            self.state = self.reset()

        start_cell = self.get_cell()
        end_trajectory = self.run_seed(seed, start_cell=cell, max_steps=self.explore_steps)

        # We are not done, check that doing nothing for self.ignore_death steps won't kill us.
        if self.ignore_death > 0:
            if not end_trajectory[-1].done:
                end_trajectory += self.run_explorer(DoNothingExplorer(), max_steps=self.ignore_death)
            end_trajectory = end_trajectory[:-self.ignore_death]


        seen_to = set()
        for e in end_trajectory:
            e.from_.restore = None
            e.from_.state = None
            if e.to.cell in seen_to:
                e.to.restore = None
                e.to.state = None
            seen_to.add(e.to.cell)

        known_room_data = {}
        if len(ENV.rooms) > known_rooms:
            known_room_data = ENV.rooms

        return TimedPickle((start_cell, end_trajectory, self.frames_true, self.frames_compute, known_room_data), 'ret', enabled=info.enabled)

    def run_cycle(self):
        # Choose a bunch of cells, send them to the workers for processing, then combine the results.
        # A lot of what this function does is only aimed at minimizing the amount of data that needs
        # to be pickled to the workers, which is why it sets a lot of variables to None only to restore
        # them later.
        # for key in self.grid.keys():
        #     if self.grid[key].trajectory_len > 0:
        #         assert self.grid[key].restore is not None, f'No cell restore in iter: {self.cycles}'

        global POOL
        if self.start is None:
            self.start = time.time()

        self.cycles += 1
        chosen_cells = []
        cell_keys = self.selector.choose_cell(self.grid, size=self.batch_size)
        old_trajectories = []
        for i, cell_key in enumerate(cell_keys):
            cell = self.grid[cell_key]
            old_trajectories.append((cell.trajectory, cell.seen, cell.chain))
            cell.trajectory = None
            cell.seen = None
            cell.chain = None
            seed = random.randint(0, 2 ** 31)
            # if cell.trajectory_len > 0:
            #     assert cell.restore is not None, 'Choosen cell without restore'
            chosen_cells.append(TimedPickle((cell_key, cell, seed,
                                             len(ENV.rooms), self.env_info[0].TARGET_SHAPE,
                                             self.env_info[0].MAX_PIX_VALUE), 'args', enabled=(i == 0 and False)))

        # NB: self.grid is uncessecary for process_cell, and might be
        # VERY large. We temporarily replace it with None so it doesn't
        # need to be serialized by the pool.

        # old_grid = self.grid
        # global GRID
        # GRID = old_grid
        # self.grid = None

        self.IR = 0
        self.dones = 0

        #trajectories = [e.data for e in POOL.map(self.process_cell, chosen_cells)]
        trajectories = [e.data for e in map(self.process_cell, chosen_cells)]

        self.summary.append(summary.Summary.Value(tag='Intrisinc Reward', simple_value=self.IR))
        self.summary.append(summary.Summary.Value(tag='Deaths', simple_value=self.dones))

        if self.reset_pool and (self.cycles + 1) % 100 == 0:
            POOL.close()
            POOL.join()
            POOL = None
            gc.collect()
            POOL = self.pool_class(self.n_cpus)
        chosen_cells = [e.data for e in chosen_cells]

        # self.grid = old_grid

        for ((_, cell, _, _, _, _), (old_traj, old_seen, old_chain)) in zip(chosen_cells, old_trajectories):
            if old_traj is not None:
                cell.trajectory = old_traj
            if old_seen is not None:
                cell.seen = old_seen
            if old_chain is not None:
                cell.chain = old_chain

        # Note: we do this now because starting here we're going to be concatenating the trajectories
        # of these cells, and they need to remain the same!
        chosen_cells = [(k, copy.copy(c), s, n, shape, pix) for k, c, s, n, shape, pix in chosen_cells]
        cells_to_reset = set()

        for ((cell_key, cell, seed, _, _, _), (start_cell, end_trajectory, ft, fc, known_rooms)) in zip(chosen_cells,
                                                                                                  trajectories):
            self.frames_true += ft
            self.frames_compute += fc
            if cell.seen is None:
                continue
            seen_cells = {}
            # Note(adrien): this changes the behavior of seen_times and action_times,
            # but it makes the whole code slower and it isn't clear that the behavior
            # implied by these next few lines is better anyway.

            # for e in cell.seen:
            #     if e not in seen_cells:
            #         seen_cells[e] = cell.seen[e]
            #         self.grid[e].seen_times += 1
            #         self.grid[e].action_times += cell.seen[e]

            for k in known_rooms:
                if k not in ENV.rooms:
                    ENV.rooms[k] = known_rooms[k]

            self.grid[cell_key].chosen_times += 1
            self.grid[cell_key].chosen_since_new += 1
            cur_score = cell.score
            for i, elem in enumerate(end_trajectory):
                potential_cell_key = elem.to.cell
                self.selector.reached_state(elem)
                if self.with_domain and elem.real_pos not in self.real_grid:
                    self.domain_knowledge[elem.real_pos.room][elem.real_pos.y][elem.real_pos.x] += 1
                self.real_grid.add(elem.real_pos)

                if not isinstance(potential_cell_key, tuple) and potential_cell_key.level > 0:
                    self.seen_level_1 = True

                potential_cell = self.grid[potential_cell_key]
                full_traj_len = cell.trajectory_len + i + 1
                cur_score += elem.reward
                for p in [potential_cell_key, elem.from_.cell]:
                    if p not in seen_cells:
                        seen_cells[p] = 0
                        self.grid[p].seen_times += 1

                self.grid[potential_cell_key].action_times += 1
                seen_cells[potential_cell_key] += 1

                if elem.to.restore is not None and self.should_accept_cell(potential_cell, cur_score, full_traj_len):
                    self.grid[cell_key].chosen_since_new = 0
                    cells_to_reset.add(potential_cell_key)
                    potential_cell.chain = cell.chain + [ChainLink(start_cell, potential_cell_key, seed)]
                    potential_cell.trajectory = cell.trajectory + end_trajectory[:i + 1]
                    potential_cell.trajectory_len = full_traj_len
                    assert len(potential_cell.trajectory) == potential_cell.trajectory_len
                    potential_cell.restore = elem.to.restore
                    assert potential_cell.restore is not None
                    potential_cell.seen = copy.copy(seen_cells)
                    potential_cell.score = cur_score
                    potential_cell.real_cell = elem.real_pos
                    if self.use_real_pos:
                        potential_cell.exact_pos = elem.to.exact

                elem.from_.restore = None
                elem.to.restore = None
            self.selector.update()
        if self.reset_cell_on_update:
            for cell_key in cells_to_reset:
                self.grid[cell_key].chosen_times = 0
                self.grid[cell_key].chosen_since_new = 0



        return [(k) for k, c, s, n, shape, pix in chosen_cells], trajectories

    def should_accept_cell(self, potential_cell, cur_score, full_traj_len):
        if random.random() < self.prob_override:
            return True
        if self.optimize_score:
            return (cur_score > potential_cell.score or
                    (full_traj_len < potential_cell.trajectory_len and
                     cur_score == potential_cell.score))
        return full_traj_len < potential_cell.trajectory_len
