
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *

import types
os.environ["PATH"] = os.environ["PATH"].replace('/usr/local/nvidia/bin', '')
try:
    import mujoco_py
    import gym.envs.robotics.utils
    import gym.envs.robotics.rotations
except Exception:
    print('WARNING: could not import mujoco_py. This means robotics environments will not work')
import gym.spaces
from scipy.spatial.transform import Rotation
from collections import defaultdict, namedtuple
import os

DOOR_NAMES = ['door', 'door1', 'latch1', 'latch']

FetchState = namedtuple('FetchState', ('door_dists', 'door1_dists', 'gripped_info', 'gripped_pos', 'object_pos', 'gripper_pos'))

class FakeAle:
    def __init__(self, env):
        self.env = env

    def lives(self):
        return 1

    def act(self, action):
        self.env.step(action, need_return=False)

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

class FakeActionSet:
    def __getitem__(self, item):
        return item

class FakeUnwrapped:
    def __init__(self, env):
        self.env = env
        self.ale = FakeAle(env)
        self._action_set = FakeActionSet()

    def restore_state(self, state):
        self.env.set_inner_state(state)

    def clone_state(self):
        return self.env.get_inner_state()

    def _get_image(self):
        if self.env.state_is_pixels:
            return self.env._get_pixel_state()
        return self.env._get_full_state()

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

class ComplexSpec:
    def __init__(self, id_):
        self.id = id_

class ComplexFetchEnv:
    MJ_INIT = False

    def __init__(self, model_file='teleOp_boxes.xml', nsubsteps=20, min_grip_score=0,
                 max_grip_score=0, ret_full_state=True, incl_extra_full_state=False, max_steps=4000, target_single_shelf=False,
                 combine_table_shelf_box=False, ordered_grip=False,
                 do_tanh=False, target_location=None, timestep=0.002, state_is_pixels=False, include_proprioception=False,
                 state_wh=192, state_azimuths='145_215', force_closed_doors=False):
        self.force_closed_doors = force_closed_doors
        self.state_is_pixels = state_is_pixels
        self.include_proprioception = include_proprioception
        self.state_wh = state_wh
        self.state_azimuths = [int(e) for e in state_azimuths.split('_')]
        self.do_tanh = do_tanh
        model_file = os.path.dirname(os.path.realpath(__file__)) + '/fetch_xml/' + model_file
        self.model = mujoco_py.load_model_from_path(model_file)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=nsubsteps)
        self.viewer = None
        self.target_location = target_location
        self.ordered_grip = ordered_grip
        self.cached_state = None
        self.cached_done = None
        self.cached_info = None
        self.cached_full_state = None
        self.render_cache = defaultdict(dict)
        self.cached_contacts = None
        self.has_had_contact = set()
        self.first_shelf_reached = None
        self.target_single_shelf = target_single_shelf
        self.min_grip_score = min_grip_score
        self.max_grip_score = max_grip_score
        self.ret_full_state = ret_full_state and not state_is_pixels
        self.incl_extra_full_state = (incl_extra_full_state and not include_proprioception)
        self.unwrapped = FakeUnwrapped(self)
        self.max_steps = max_steps
        self.spec = ComplexSpec('fetch')

        self.filtered_idxs_for_full_state = None

        self.contact_bodies = sorted([
            'world',
            'gripper_link',
            'r_gripper_finger_link',
            'l_gripper_finger_link',
            'Table',
            'DoorLR',
            'frameR1',
            'door1',
            'latch1',
            'frameL1',
            'DoorUR',
            'frameR',
            'door',
            'latch',
            'frameL',
            'Shelf',
            'obj0',
            'obj1',
            'obj2',
            'obj3',
            'obj4'
        ])
        self.contact_body_idx = [(self.sim.model.body_name2id(c) if c in self.sim.model.body_names else None) for c in self.contact_bodies]
        self.contact_indexes = {}
        self.contact_names = []
        contact_idx = 0
        for i in range(0, len(self.contact_bodies)):
            for j in range(i + 1, len(self.contact_bodies)):
                pair = (self.contact_bodies[i], self.contact_bodies[j])
                self.contact_indexes[pair] = contact_idx
                self.contact_names.append(pair)
                contact_idx += 1

        assert self.sim.model.nmocap == 1, 'Only supports model with a single mocap (for now)'
        self.sim.data.mocap_pos[0, :] = [10, 10, 10]
        self.n_actions = len(self.sim.model.actuator_ctrlrange)
        self.sim.model.eq_active[0] = 0

        self.excluded_bodies = [
            # Elements in the world with a fixed position
            'world', 'Table', 'Shelf',
            # These elements could move, but don't in the current implementation
            'base_link', 'torso_lift_link', 'estop_link', 'laser_link'
        ]
        self.excluded_bodies.append('mocap0')

        self.action_space = gym.spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.prev_action = np.zeros(self.n_actions)
        self.observation_space = gym.spaces.Box(-3., 3., shape=(268 + 336 * incl_extra_full_state,), dtype='float32')
        if state_is_pixels:
            self.observation_space = gym.spaces.Box(0, 255, shape=(self.state_wh, self.state_wh, len(self.state_azimuths) * 3), dtype='uint8')
            if include_proprioception:
                old_obs_space = self.observation_space
                self.observation_space = (gym.spaces.Box(-3., 255., shape=(np.product(self.observation_space.shape) + 156,), dtype='float32'))
                self.observation_space.pixel_space = old_obs_space
                self.excluded_bodies = [
                    # Elements in the world, regardless of position
                    'world', 'Table', 'Shelf', 'mocap0', 'DoorLR', 'frameR1', 'door1',
                    'latch1', 'frameL1', 'DoorUR', 'frameR', 'door', 'latch', 'frameL',
                    'Shelf', 'obj0', 'obj1', 'obj2', 'obj3', 'obj4', 'obj5',
                    # These elements could move, but don't in the current implementation
                    'base_link', 'torso_lift_link', 'estop_link', 'laser_link'
                ]

        # Actually the range is -1 to 1, but let's ignore this for now.
        self.reward_range = None
        self.metadata = {}

        self.sim.forward()

        self.door_ids = [self.sim.model.body_name2id(name) for name in DOOR_NAMES]
        self.door_init_pos = [np.copy(self.sim.data.body_xpos[i]) for i in self.door_ids]

        self.object_names = sorted([name for name in self.sim.model.body_names if 'obj' in name])
        self.object_ids = [self.sim.model.body_name2id(name) for name in self.object_names]
        self.grip_id = self.sim.model.body_name2id('gripper_link')

        self.sim.model.opt.timestep = timestep

        self.boxes = {}

        def get_geom_box(e, adj_range=1.0):
            assert np.allclose(self.sim.data.geom_xmat[e], np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))
            # Note: the geom_size is actually a HALF-size, so no need to divide by 2.
            xymin = self.sim.data.geom_xpos[e] - self.sim.model.geom_size[e] * adj_range
            xymax = self.sim.data.geom_xpos[e] + self.sim.model.geom_size[e] * adj_range
            return xymin, xymax

        for e in range(self.sim.model.ngeom):
            body_name = self.sim.model.body_id2name(self.sim.model.geom_bodyid[e])
            if body_name == 'Table' and self.sim.model.geom_contype[e]:
                table_range = get_geom_box(e, 1.2)
                table_range[1][-1] = 100
                self.boxes['table'] = table_range
            elif body_name == 'annotation:outer_bound':
                self.boxes['shelf'] = get_geom_box(e, 1.2)
            elif 'annotation:inside' in str(body_name):
                self.boxes[str(body_name)[len('annotation:inside_'):]] = get_geom_box(e, 0.9)

        if combine_table_shelf_box:
            combined_box = np.min([self.boxes['table'][0], self.boxes['shelf'][0]], axis=0), np.max([self.boxes['table'][1], self.boxes['shelf'][1]], axis=0)
            self.boxes['table'] = combined_box
            self.boxes['shelf'] = combined_box

        self.box_names = sorted(self.boxes.keys())
        assert self.box_names[-2:] == ['shelf', 'table']

        self.box_mins = np.array([self.boxes[name][0] for name in self.box_names])
        self.box_maxs = np.array([self.boxes[name][1] for name in self.box_names])

        self.n_steps = 0
        self.start_state = self.get_inner_state()

    def body_in_box(self, body, box):
        if isinstance(body, str):
            try:
                body = self.sim.model.body_name2id(body)
            except ValueError:
                # The body does not exist, and therefore is not in the box
                return False

        pos = self.sim.data.body_xpos[body]
        xymin, xymax = self.boxes[box]
        res = np.all(xymin <= pos) and np.all(pos <= xymax)
        return res

    def body_in_boxes(self, body):
        if isinstance(body, str):
            try:
                body = self.sim.model.body_name2id(body)
            except ValueError:
                # The body does not exist, and therefore is not in the box
                return False

        pos = self.sim.data.body_xpos[body]
        return np.all((self.box_mins <= pos) & (pos <= self.box_maxs), axis=1).astype(np.int32).tolist()

    def bodies_in_boxes(self, bodies):
        body_xpos = self.sim.data.body_xpos

        pos = []
        for body in bodies:
            if isinstance(body, str):
                try:
                    body = self.sim.model.body_name2id(body)
                except ValueError:
                    # The body does not exist, and therefore is not in the box
                    return False

            if body is None:
                pos.append([0, 0, 0])
            else:
                pos.append(body_xpos[body])


        pos = np.array(pos)
        return np.all((self.box_mins <= pos[:, None, :]) & (pos[:, None, :] <= self.box_maxs), axis=2).astype(np.int32)

    def render(self, mode='new', width=752, height=912, distance=3, azimuth=170, elevation=-30, cache_key='current'):
        key = (distance, azimuth, elevation)
        target = 'latch1'

        if key not in self.render_cache[cache_key]:
            # The mujoco renderer is stupid and changes the inner state in minor
            # ways, which can ruin some long trajectories. Because of this, we
            # save the inner state before rendering and restore it afterwards.
            inner_state = copy.deepcopy(self.get_inner_state())

            if self.viewer is None:
                if 'CUSTOM_DOCKER_IMAGE' not in os.environ:
                    # We detected that we are running on desktop, in which case we should
                    # use glfw as the mode.
                    mode = 'glfw'
                if not self.__class__.MJ_INIT and mode == 'glfw':
                    print("WTF")
                    try:
                        mujoco_py.MjViewer(self.sim)
                        print("WOW")
                    except Exception:
                        print('Failed to initialize GLFW, rendering may or may not work.')
                    self.__class__.MJ_INIT = True

                # Note: opus machines typically have 4 GPUs, but the docker is given access to just 1. The problem
                # is that OpenGL "sees" all 4 GPUs, but errors out if it tries to use one that it doesn't have
                # access to, so we simply try all 4 GPUs in turn until it works. We start at -1 (autodetect GPU)
                # for cases where we are running this outside of opus, in which case the default most likely works.
                for device in range(-1, 4):
                    try:
                        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=device)
                        break
                    except RuntimeError:
                        print('Device', device, 'didn\'t work.')
                self.viewer.scn.flags[2] = 0 # Disable reflections (~25% speedup)
                body_id = self.sim.model.body_name2id(target)
                lookat = self.sim.data.body_xpos[body_id]
                for idx, value in enumerate(lookat):
                    self.viewer.cam.lookat[idx] = value

            self.viewer.cam.distance = distance
            self.viewer.cam.azimuth = azimuth
            self.viewer.cam.elevation = elevation
            self.viewer.render(width, height)
            img = self.viewer.read_pixels(width, height, depth=False)
            img = img[::-1, :, :]

            self.set_inner_state(inner_state)

            self.render_cache[cache_key][key] = img

        return self.render_cache[cache_key][key]

    def sample_action(self, sd=-1, angle=None):
        # If sd < 0, we use the uniform distribution from -1 to 1, otherwise we use a normal distribution
        # centered at 0 with the specified sd, and clip between -1 and 1
        if sd < 0:
            action = np.random.random(self.n_actions) * 2 - 1
        else:
            action = np.tanh(np.random.randn(self.n_actions) * sd)

        return action

    def _mocap_set_action(self, action):
        if self.sim.model.nmocap > 0:
            action, _ = np.split(action, (self.sim.model.nmocap * 6,))
            action = action.reshape(self.sim.model.nmocap, 6)

            pos_delta = action[:, :3]
            rot_delta = action[:, 3:]

            gym.envs.robotics.utils.reset_mocap2body_xpos(self.sim)

            # Note: we use np.roll here because Rotation assumes an (x, y, z, w) representation while
            # mujoco assumes a (w, x, y, z) representation.
            orig_rot = Rotation.from_quat(np.roll(self.sim.data.mocap_quat, 3))
            rot_delta = Rotation.from_rotvec(rot_delta)

            self.sim.data.mocap_pos[:] = self.sim.data.mocap_pos + pos_delta
            self.sim.data.mocap_quat[:] = np.roll((rot_delta * orig_rot).as_quat(), 1)

    def _iter_contact(self):
        if self.cached_contacts is None:
            contact = self.sim.data.contact
            geom_bodyid = self.sim.model.geom_bodyid
            res = []
            seen = []
            for i in range(self.sim.data.ncon):
                c = contact[i]
                id1 = geom_bodyid[c.geom1]
                id2 = geom_bodyid[c.geom2]
                if (id1, id2) not in seen:
                    seen.append((id1, id2))
                    name1 = self.sim.model.body_id2name(id1)
                    name2 = self.sim.model.body_id2name(id2)
                    res.append((name1, name2))
            self.cached_contacts = res

        return self.cached_contacts

    def _get_done(self):
        if self.cached_done is None:
            self._get_state()

        return self.cached_done

    def _get_state(self):
        if self.cached_state is None:
            done_reasons = []
            self.cached_info = None
            self.cached_done = False
            nopos = np.array([-1000.0, -1000.0, -1000.0])

            GRIPPERS = ['l_gripper_finger_link', 'r_gripper_finger_link']
            # Note: code for the 'fo' reason
            touching_table = set()
            gripped = defaultdict(set)
            def handle_names_contact(name1, name2):
                if 'obj' in str(name1):
                    # Note: code for the 'fo' reason
                    if 'able' in str(name2):
                        touching_table.add(name1)
                    elif 'orld' in str(name2):
                        self.cached_done = True
                    elif name2 in GRIPPERS:
                        gripped[name1].add(name2)

            for name1, name2 in self._iter_contact():
                handle_names_contact(name1, name2)
                handle_names_contact(name2, name1)

            gripped_info = None
            gripped_pos = nopos
            for g in gripped:
                grip_score = len(gripped[g])
                if grip_score == 2 and g not in touching_table:
                    grip_score += 1
                if grip_score > 0:
                    gripped_info = (g, grip_score)
                    if grip_score > 1:
                        gripped_pos = self.sim.data.body_xpos[self.object_ids[self.object_names.index(g)]].copy()

            if gripped_info is not None:
                if gripped_info[-1] < self.min_grip_score:
                    gripped_info = None
                    gripped_pos = nopos
                elif gripped_info[-1] > self.max_grip_score:
                    gripped_info = (gripped_info[0], min(self.max_grip_score, gripped_info[-1]))

            door_dists = []
            door1_dists = []
            for i in range(len(DOOR_NAMES)):
                idx = self.door_ids[i]
                init_pos = self.door_init_pos[i]
                dist = np.linalg.norm(self.sim.data.body_xpos[idx] - init_pos)
                if 'latch' in DOOR_NAMES[i]:
                    dist /= 2
                if '1' in DOOR_NAMES[i]:
                    door1_dists.append(dist)
                else:
                    door_dists.append(dist)

            grip_pos = self.sim.data.body_xpos[self.grip_id].copy()

            object_pos = []
            body_pos = self.bodies_in_boxes(self.object_ids)
            for cur_pos, idx in zip(body_pos, self.object_ids):
                # Note: remove table and shelf from the position
                object_pos.append(''.join(map(str, cur_pos[:-2])))
                if self.first_shelf_reached is None and object_pos[-1] != '0' * len(object_pos[-1]):
                    self.first_shelf_reached = object_pos[-1]
                elif self.target_single_shelf and object_pos[-1] != self.first_shelf_reached:
                    object_pos[-1] = '0' * len(object_pos[-1])

            if self.ordered_grip and gripped_info is not None:
                min_grip_id = 0
                while min_grip_id < len(object_pos):
                    if object_pos[min_grip_id] == 0 or object_pos[min_grip_id] == '0000':
                        break
                    min_grip_id += 1

                if gripped_info[0] != f'obj{min_grip_id}' and gripped_info[0] != f'obj{min_grip_id-1}':
                    gripped_info = None
                    gripped_pos = nopos

            self.cached_state = FetchState(
                door_dists=door_dists,
                door1_dists=door1_dists,
                gripped_info=gripped_info,
                gripped_pos=gripped_pos,
                object_pos=object_pos,
                gripper_pos=grip_pos
            )
            self.cached_info = {'done_reasons': done_reasons}

        return copy.deepcopy(self.cached_state)

    def _get_reward(self, prev_state, state):
        if self.target_location is None:
            obj_prev = np.array(list(map(int, ''.join(prev_state.object_pos))))
            obj_now = np.array(list(map(int, ''.join(state.object_pos))))
        else:
            obj_prev = np.array([int(e == self.target_location) for e in prev_state.object_pos])
            obj_now = np.array([int(e == self.target_location) for e in state.object_pos])
        door_factor = 1
        prev_door_factor = 1
        if self.force_closed_doors:
            door_factor = int(self._doors_are_closed(state))
            prev_door_factor = int(self._doors_are_closed(prev_state))
        return int(np.sum(obj_now * door_factor - obj_prev * prev_door_factor))

    def _doors_are_closed(self, state):
        return np.all(np.array(state.door_dists + state.door1_dists) < 0.01)

    def seed(self, seed):
        pass

    def _get_extra_full_state(self, include_names=False):
        extra_names = []
        # Step 1: contacts
        contacts = np.zeros(len(self.contact_names))
        for contact in self._iter_contact():
            if None in contact:
                continue
            contact = tuple(sorted(contact))
            if contact in self.contact_indexes:
                contacts[self.contact_indexes[contact]] = 1

        if include_names:
            extra_names += self.contact_names

        # Step 2: target
        target = [int(e) for e in self.target_location] if self.target_location else [0] * 4
        if include_names:
            extra_names += ['target_location'] * 4

        # Step 3: boxes
        if include_names:
            extra_state = list(contacts) + list(target)
            for body in self.contact_body_idx:
                for box in self.box_names:
                    extra_state.append(int(self.body_in_box(body, box)))
                    if include_names:
                        extra_names.append(('in_box', self.sim.model.body_id2name(body), box))
        else:
            extra_state = np.concatenate([contacts, target, self.bodies_in_boxes(self.contact_body_idx).flatten()])

        return extra_state, extra_names

    def _get_full_state(self, include_names=False):
        if self.cached_full_state is None or include_names != isinstance(self.cached_full_state, tuple):
            if self.filtered_idxs_for_full_state is None:
                self.filtered_idxs_for_full_state = []
                self.filtered_names_for_full_state = []
                for e in self.sim.model.body_names:
                    # All "annotation" elements are virtual, while all "Door" and "frame" elements are static.
                    if e in self.excluded_bodies or 'annotation' in e or 'Door' in e or 'frame' in e:
                        continue
                    self.filtered_names_for_full_state.append(e)
                    self.filtered_idxs_for_full_state.append(self.sim.model.body_name2id(e))
                self.filtered_idxs_for_full_state = np.array(self.filtered_idxs_for_full_state, dtype=np.int32)

            dt = self.sim.nsubsteps * self.sim.model.opt.timestep

            dim_size = len(self.filtered_idxs_for_full_state) * 3
            n_dims = 4
            if self.incl_extra_full_state:
                extra_state, extra_names = self._get_extra_full_state(include_names)
                all_state = np.empty(len(extra_state) + dim_size * n_dims)
                all_state[dim_size * n_dims:] = extra_state
            else:
                all_state = np.empty(dim_size * n_dims)

            eulers = gym.envs.robotics.rotations.mat2euler(
                self.sim.data.body_xmat[self.filtered_idxs_for_full_state].reshape((len(self.filtered_idxs_for_full_state), 3, 3))
            )

            all_state[:dim_size] = eulers.flatten()
            all_state[dim_size:2*dim_size] = self.sim.data.body_xpos[self.filtered_idxs_for_full_state].flatten()
            all_state[dim_size*2:3*dim_size] = (self.sim.data.body_xvelp[self.filtered_idxs_for_full_state]).flatten()
            all_state[dim_size*3:4*dim_size] = (self.sim.data.body_xvelr[self.filtered_idxs_for_full_state]).flatten()
            all_state[dim_size*2:4*dim_size] *= dt

            if include_names:
                all_names = []
                for type in ['rot', 'pos', 'velp', 'velr']:
                    for name in self.filtered_names_for_full_state:
                        for sub in ['x', 'y', 'z']:
                            all_names.append(f'{name}_{type}_{sub}')
                all_names += extra_names
                assert len(all_names) == len(all_state)
                self.cached_full_state = (all_state, all_names)
            else:
                self.cached_full_state = all_state

        if include_names:
            return copy.deepcopy(self.cached_full_state)
        res = self.cached_full_state.copy()
        if res.shape != self.observation_space.shape and not self.include_proprioception:
            old_res = res
            res = np.zeros(self.observation_space.shape, dtype=res.dtype)
            res[:old_res.size] = old_res
        return res

    def step(self, action_small, need_return=True):
        if self.do_tanh:
            action_small = np.tanh(action_small)
        had_exception = False
        prev_state = self._get_state()
        try:
            self.sim.forward()
        except mujoco_py.builder.MujocoException:
            had_exception = True

        assert not isinstance(action_small, (int, float))
        action = np.zeros(self.n_actions)
        action[:] = action_small
        action = np.clip(action, -1, 1)
        self.prev_action = action
        ctrl = (action + 1) / 2
        ctrl = self.sim.model.actuator_ctrlrange[:, 0] + \
                ctrl * (self.sim.model.actuator_ctrlrange[:, 1] - self.sim.model.actuator_ctrlrange[:, 0])

        self.sim.data.ctrl[:] = ctrl

        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException:
            had_exception = True
        self.cached_state = None
        self.cached_contacts = None
        self.render_cache['current'] = {}
        self.cached_done = None
        self.cached_info = None
        self.cached_full_state = None

        state = self._get_state()
        reward = self._get_reward(prev_state, state)
        if self.ret_full_state:
            state = self._get_full_state()
        done = self._get_done() or had_exception
        info = copy.copy(self.cached_info)

        self.n_steps += 1
        if self.n_steps > self.max_steps:
            done = True
            info['done_reasons'] = info.get('done_reasons', []) + ['ms']
        # Note that we don't give a negative rewards if done comes from reaching the max steps
        if not isinstance(state, FetchState):
            assert np.all(~np.isnan(state))
        if self.state_is_pixels and need_return:
            state = self._get_pixel_state()
        return copy.deepcopy((state, reward, (done or self.n_steps > self.max_steps), info))

    DATA_TO_SAVE = [
        'qpos',
        'qvel',
        'act',
        'mocap_pos',
        'mocap_quat',
        'userdata',
        'qacc_warmstart',
        'ctrl',
    ]

    def get_inner_state(self):
        return copy.deepcopy((
            tuple(
                getattr(self.sim.data, attr) for attr in self.DATA_TO_SAVE
            ),
            self._get_state(),
            self._get_done(),
            (self._get_full_state() if self.ret_full_state else None),
            self.n_steps,
            self.has_had_contact,
            self.first_shelf_reached,
            self.prev_action,
            self.cached_info
        ))

    def set_inner_state(self, data):
        for attr, val in zip(self.DATA_TO_SAVE, data[0]):
            current_value = getattr(self.sim.data, attr)
            if current_value is None:
                assert val is None
            else:
                current_value[:] = val
        self.cached_state = data[1]
        self.cached_done = data[2]
        self.cached_full_state = data[3]
        self.n_steps = data[4]
        self.has_had_contact = data[5]
        self.first_shelf_reached = data[6]
        if len(data) > 7:
            self.prev_action = data[7]
        if len(data) > 8:
            self.cached_info = data[8]
        else:
            self.cached_info = None
        self.render_cache['current'] = {}

        self.cached_contacts = None

    def reset(self):
        self.set_inner_state(self.start_state)
        if self.state_is_pixels:
            return self._get_pixel_state(cache_key='reset')
        if self.ret_full_state:
            res = self._get_full_state()
        else:
            res = self._get_state()
        return res

    def _get_pixel_state(self, cache_key='current'):
        states = []
        for azimuth in self.state_azimuths:
            wh = self.state_wh
            if wh == 96:
                wh = 256
            img = self.render(width=wh, height=wh, azimuth=azimuth, cache_key=cache_key)
            if self.state_wh != wh:
                img = cv2.resize(img, (self.state_wh, self.state_wh), interpolation=cv2.INTER_AREA)
            states.append(img)
        res = np.concatenate(states, axis=2)
        if self.include_proprioception:
            if self.cached_full_state is not None and self.cached_full_state.size != 156:
                self.cached_full_state = None
            full_state = self._get_full_state()
            assert full_state.size == 156
            res = np.concatenate([res.flatten().astype(np.float32), full_state])
            return res
        else:
            return res


class MyComplexFetchEnv:
    TARGET_SHAPE = 0
    MAX_PIX_VALUE = 0

    def __init__(self, model_file='teleOp_objects.xml', nsubsteps=20, min_grip_score=0, max_grip_score=0,
                 target_single_shelf=False, combine_table_shelf_box=False, ordered_grip=False,
                 target_location=None, timestep=0.002, force_closed_doors=False):
        self.env = ComplexFetchEnv(
            model_file=model_file, nsubsteps=nsubsteps,
            min_grip_score=min_grip_score, max_grip_score=max_grip_score,
            ret_full_state=False, target_single_shelf=target_single_shelf,
            combine_table_shelf_box=combine_table_shelf_box, ordered_grip=ordered_grip,
            target_location=target_location, timestep=timestep,
            force_closed_doors=force_closed_doors
        )
        self.rooms = []

        self.reset()

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.env._get_state()

    def get_restore(self):
        return (
            self.env.get_inner_state(),
        )

    def restore(self, data):
        self.env.set_inner_state(data[0])
        return self.env._get_state()

    def step(self, action):
        return self.env.step(action)

    def get_pos(self):
        return self.env._get_state()

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        pass
