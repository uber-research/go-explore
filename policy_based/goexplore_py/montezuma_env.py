# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Any
from collections import defaultdict
from atari_reset.atari_reset.wrappers import MyWrapper


def convert_state(state):
    if MyMontezuma.TARGET_SHAPE is None:
        return None
    import cv2
    resized_state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY),
                               MyMontezuma.TARGET_SHAPE,
                               interpolation=cv2.INTER_AREA)
    return ((resized_state / 255.0) * MyMontezuma.MAX_PIX_VALUE).astype(np.uint8)


PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23]
]

NB_KEY_PIXELS = 40

OBJECT_PIXELS = [
    50,  # Hammer/mallet
    NB_KEY_PIXELS,  # Key 1
    NB_KEY_PIXELS,  # Key 2
    NB_KEY_PIXELS,  # Key 3
    37,  # Sword 1
    37,  # Sword 2
    42   # Torch
]

KNOWN_XY: List[Any] = [None] * 24

KEY_BITS = 0x8 | 0x4 | 0x2
ROOM_INDEX = 3


def get_room_xy(room):
    if KNOWN_XY[room] is None:
        for y, l in enumerate(PYRAMID):
            if room in l:
                KNOWN_XY[room] = (l.index(room), y)
                break
    return KNOWN_XY[room]


def clip(a, min_v, max_v):
    if a < min_v:
        return min_v
    if a > max_v:
        return max_v
    return a


def sum_two(a, b):
    return a + b


class MyMontezuma(MyWrapper):
    TARGET_SHAPE = None
    MAX_PIX_VALUE = None
    screen_width = 160
    screen_height = 210 - 50
    x_repeat = 2
    attr_max = {'level': 3,
                'room': 24,
                'objects': 4}

    @staticmethod
    def get_attr_max(name):
        if name == 'x':
            return MyMontezuma.screen_width * MyMontezuma.x_repeat
        elif name == 'y':
            return MyMontezuma.screen_height
        else:
            return MyMontezuma.attr_max[name]

    def __init__(self,
                 env,
                 check_death: bool = True,
                 score_objects: bool = False,
                 objects_from_pixels=True,
                 objects_remember_rooms=False,
                 only_keys=False,
                 only_nb_keys=True,
                 cell_representation=None):
        super(MyMontezuma, self).__init__(env)
        self.env.reset()
        self.score_objects = score_objects
        self.ram = None
        self.check_death = check_death
        self.cur_steps = 0
        self.total_steps = 0
        self.cur_score = 0
        self.rooms = {}
        self.room_time = None
        self.room_threshold = 40
        self.env.unwrapped.seed(0)
        self.state = []
        self.ram_death_state = -1
        self.cur_lives = 5
        self.ignore_ram_death = False
        self.objects_from_pixels = objects_from_pixels
        self.death_from_pixels = True
        self.objects_remember_rooms = objects_remember_rooms
        self.only_keys = only_keys
        self.pos = None
        self.level = None
        self.objects = None
        self.room = None
        self.x = None
        self.y = None
        self.only_nb_keys = only_nb_keys
        self.cell_representation = cell_representation
        self.death_at_prev_step = False
        self.transition_at_prev_step = False
        self.level_transition = False
        self.prev_ram_room = None
        self.prev_ram_transition = False
        self.done = 0

    def __getattr__(self, e):
        return getattr(self.env, e)

    def get_my_montezuma(self):
        return self

    def reset(self) -> np.ndarray:
        unprocessed_state = self.env.reset()
        self.cur_lives = 5
        self.state = [convert_state(unprocessed_state)]
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = 0
        self.cur_steps = 0
        self.ram_death_state = -1
        self.pos = None
        self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state), unprocessed_state, False)
        self.pos = self.cell_representation(self)
        if self.room not in self.rooms and self.level >= 0:
            self.rooms[self.room] = (False, unprocessed_state[50:].repeat(self.x_repeat, axis=1))
        self.room_time = (self.room, 0)
        self.death_at_prev_step = False
        self.transition_at_prev_step = False

        self.prev_ram_room = self.ram[ROOM_INDEX]
        self.prev_ram_transition = False
        self.level_transition = False
        self.done = 0
        return unprocessed_state

    def pos_from_unprocessed_state(self, face_pixels, unprocessed_state, _death):
        face_pixels = [(y, x * self.x_repeat) for y, x in face_pixels]
        # While we are dead or if we are on a transition screen, we assume that our position does not change
        if len(face_pixels) == 0:
            assert self.pos is not None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        room = 1
        level = 0
        old_objects = tuple()
        self.level_transition = False

        # Check transition from previous position
        if self.pos is not None:
            room = self.room
            level = self.level
            old_objects = self.objects
            if self.room == 15 and self.y > 100 and 30 > y > 23:
                # self.y > 100: means the agent was somewhere near the bottom of the screen in the previous time step
                # 30 > y > 23: means the agent has currently respawned at the default height of 29.0
                room = 1
                level += 1
                self.level_transition = True
            elif self.transition_at_prev_step:
                room_x, room_y = get_room_xy(self.room)
                if x > 309:
                    direction_x = -1
                elif x < 7:
                    direction_x = 1
                else:
                    direction_x = 0
                if y > 130:
                    direction_y = -1
                elif y < 12:
                    direction_y = 1
                else:
                    direction_y = 0

                assert direction_x == 0 or direction_y == 0, f'Room change in more than two directions : ' \
                                                             f'({direction_y}, {direction_x}) room={self.room} ' \
                                                             f'prev={self.x},{self.y} new={x},{y}'
                room = PYRAMID[room_y + direction_y][room_x + direction_x]
                assert room != -1, f'Impossible room change: ({direction_y}, {direction_x})'

        score = self.cur_score
        if self.score_objects:
            if not self.objects_from_pixels:
                score = self.ram[65]
                if self.only_keys:
                    # These are the key bytes
                    score &= KEY_BITS
            else:
                score = self.get_objects_from_pixels(unprocessed_state, room, old_objects)

        self.level = level
        self.objects = score
        self.room = room
        self.x = x
        self.y = y

    def get_objects_from_pixels(self, unprocessed_state, room, old_objects):
        object_part = (unprocessed_state[25:45, 55:110, 0] != 0).astype(np.uint8) * 255
        connected_components = cv2.connectedComponentsWithStats(object_part)
        pixel_areas = list(e[-1] for e in connected_components[2])[1:]

        if self.objects_remember_rooms:
            cur_object = []
            old_objects = list(old_objects)
            for i, n_pixels in enumerate(OBJECT_PIXELS):
                if n_pixels != 40 and self.only_keys:
                    continue
                if n_pixels in pixel_areas:

                    pixel_areas.remove(n_pixels)
                    orig_types = [e[0] for e in old_objects]
                    if n_pixels in orig_types:
                        idx = orig_types.index(n_pixels)
                        cur_object.append((n_pixels, old_objects[idx][1]))
                        old_objects.pop(idx)
                    else:
                        cur_object.append((n_pixels, room))
            return tuple(cur_object)

        elif self.only_nb_keys:
            cur_object = 0
            for n_pixels in pixel_areas:
                if n_pixels == NB_KEY_PIXELS:
                    cur_object += 1
            return cur_object
        else:
            cur_object = 0
            for i, n_pixels in enumerate(OBJECT_PIXELS):
                if n_pixels in pixel_areas:
                    pixel_areas.remove(n_pixels)
                    cur_object |= 1 << i

            if self.only_keys:
                # These are the key bytes
                cur_object &= KEY_BITS
            return cur_object

    def get_restore(self):
        return (
            self.unwrapped.clone_full_state(),
            copy.copy(self.state),
            self.cur_score,
            self.cur_steps,
            self.pos,
            self.room_time,
            self.ram_death_state,
            self.score_objects,
            self.cur_lives
        )

    def restore(self, data):
        (full_state, state, score, steps, pos, room_time, ram_death_state, self.score_objects, self.cur_lives) = data
        self.state = copy.copy(state)
        self.env._elapsed_steps = 0
        self.env._episode_started_at = time.time()

        self.unwrapped.restore_full_state(full_state)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = score
        self.cur_steps = steps
        self.pos = pos
        self.room_time = room_time
        self.ram_death_state = ram_death_state
        self.done = 0
        return copy.copy(self.state)

    def is_transition_screen(self, unprocessed_state):
        unprocessed_state = unprocessed_state[50:, :, :]
        # The screen is a transition screen if it is all black or if its color is made up only of black and
        # (0, 28, 136), which is a color seen in the transition screens between two levels.
        return (
                       np.sum(unprocessed_state[:, :, 0] == 0) +
                       np.sum((unprocessed_state[:, :, 1] == 0) | (unprocessed_state[:, :, 1] == 28)) +
                       np.sum((unprocessed_state[:, :, 2] == 0) | (unprocessed_state[:, :, 2] == 136))
               ) == unprocessed_state.size

    def get_face_pixels(self, unprocessed_state):
        return set(zip(*np.where(unprocessed_state[50:, :, 0] == 228)))

    def is_pixel_death(self, face_pixels, transition_screen):
        # There are no face pixels and yet we are not in a transition screen. We
        # must be dead!
        if len(face_pixels) == 0:
            # All of the screen except the bottom is black: this is not a death but a
            # room transition. Ignore.
            if transition_screen:
                return False
            return True

        # We already checked for the presence of no face pixels, however,
        # sometimes we can die and still have face pixels. In those cases,
        # the face pixels will be DISCONNECTED.
        for pixel in face_pixels:
            for neighbor in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (pixel[0] + neighbor[0], pixel[1] + neighbor[1]) in face_pixels:
                    return False

        return True

    def is_ram_death(self):
        if self.ram[58] > self.cur_lives:
            self.cur_lives = self.ram[58]
        return self.ram[55] != 0 or self.ram[58] < self.cur_lives

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        unprocessed_state, reward, done, lol = self.env.step(action)
        self.state.append(convert_state(unprocessed_state))
        self.state.pop(0)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_steps += 1
        self.total_steps += 1

        face_pixels = self.get_face_pixels(unprocessed_state)
        transition_screen = self.is_transition_screen(unprocessed_state)
        ram_transition = self.prev_ram_room != self.ram[ROOM_INDEX]
        if self.death_from_pixels:
            death = self.is_pixel_death(face_pixels, transition_screen)
        else:
            death = self.is_ram_death()

        # To make sure that death is detected at the same time by all wrappers, we look at ram-death, rather than pixel
        # death.
        if self.check_death and death:
            done = True

        self.pos_from_unprocessed_state(face_pixels, unprocessed_state, death)

        if not transition_screen and not ram_transition:
            ram_room = self.ram[ROOM_INDEX]
            assert self.room == ram_room, f'Incorrect room: room={self.room} ram_room={ram_room} pos={self.x},{self.y}'

        self.cur_score += reward
        if (not done and not transition_screen and not ram_transition and not self.level_transition and
                self.room not in self.rooms):
            screen_shot = unprocessed_state[50:].repeat(self.x_repeat, axis=1)
            self.rooms[self.room] = (True, screen_shot)

        self.death_at_prev_step = death
        self.transition_at_prev_step = transition_screen
        self.prev_ram_room = self.ram[ROOM_INDEX]
        self.prev_ram_transition = ram_transition

        if done:
            self.level = 0
            self.objects = 0
            self.room = 0
            self.x = 0
            self.y = 0
            self.done = 1
        self.pos = self.cell_representation(self)

        return unprocessed_state, reward, done, lol

    def get_pos(self):
        """
        Returns a domain-knowledge based state representation including the position of Panama Joe on the screen, the
        room he is in, the level he is in, and the keys he is holding.

        This is not quite the same as the domain-knowledge based cell representation, as this state representation does
        not conflate nearby positions, but it is also not a full state representation, as it does not include
        information about items other than keys or information about the state or position of enemies and hazards.

        @return: An instance of MontezumaPosLevel, which specifies the position of Panama Joe, the room he is in, the
                 level he is in, and the keys he is holding.
        """
        assert self.pos is not None
        return self.pos

    def render_with_known(self, known_positions, x_res, y_res, show=False, filename=None, combine_val=sum_two,
                          get_val=lambda x: x.score, minmax=None, log_scale=False):
        height, width = list(self.rooms.values())[0][1].shape[:2]

        final_image = np.zeros((height * 4, width * 9, 3), dtype=np.uint8) + 255

        positions = PYRAMID

        def room_pos(room_):
            for height_, l in enumerate(positions):
                for width_, r in enumerate(l):
                    if r == room_:
                        return height_, width_
            return None

        points = defaultdict(int)

        for room in range(24):
            if room in self.rooms:
                img = self.rooms[room][1]
            else:
                img = np.zeros((height, width, 3)) + 127
            y_room, x_room = room_pos(room)
            y_room *= height
            x_room *= width
            final_image[y_room:y_room + height, x_room:x_room + width, :] = img

        img = np.zeros((height, width, 3)) + 127
        plt.figure(figsize=(final_image.shape[1] // 30, final_image.shape[0] // 30))

        for room in range(24):
            y_room, x_room = room_pos(room)
            y_room *= height
            x_room *= width

            for y in np.arange(y_res, img.shape[0], y_res):
                cv2.line(final_image, (x_room, y_room + y), (x_room + img.shape[1], y_room + y), (127, 127, 127), 1)
            for x in np.arange(x_res, img.shape[1], x_res):
                cv2.line(final_image, (x_room + x, y_room), (x_room + x, y_room + img.shape[0]), (127, 127, 127), 1)

            cv2.line(final_image, (x_room, y_room), (x_room, y_room + img.shape[0]), (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room), (x_room + img.shape[1], y_room), (255, 255, 255), 1)
            cv2.line(final_image, (x_room + img.shape[1], y_room), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room + img.shape[0]), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)

            for k in known_positions:
                if k.level == -1:
                    assert k.x == 0
                    assert k.y == 0
                    assert k.room == 0
                    continue
                if k.room != room:
                    continue
                x = x_room + (k.x * x_res + x_res / 2) - 5.5 + k.objects * 5.575
                y = y_room + (k.y * y_res + y_res / 2) - 5.7
                points[(x, y)] = combine_val(points[(x, y)], get_val(k))

        plt.imshow(final_image)
        if minmax:
            points[(0, 0)] = minmax[0]
            points[(0, 1)] = minmax[1]

        vals = list(points.values())
        points = list(points.items())
        plt.scatter([p[0][0] for p in points], [p[0][1] for p in points], c=[p[1] for p in points], cmap='bwr',
                    s=(min(x_res, y_res)**2) * 0.15, marker='s')
        plt.legend()

        import matplotlib.cm
        import matplotlib.colors
        from matplotlib.ticker import ScalarFormatter
        if log_scale:

            v = np.geomspace(np.min(vals), np.max(vals), 11, endpoint=True)
            mappable = matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.LogNorm(vmin=np.min(vals), vmax=np.max(vals)),
                cmap='bwr')
            mappable.set_array(np.array(vals))
            matplotlib.rcParams.update({'font.size': 22})
            formatter = ScalarFormatter()
            plt.colorbar(mappable, ticks=v, format=formatter)
        else:
            v = np.linspace(np.min(vals), np.max(vals), 11, endpoint=True)
            mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.min(vals),
                                                                                     vmax=np.max(vals)),
                                                    cmap='bwr')
            mappable.set_array(np.array(vals))
            matplotlib.rcParams.update({'font.size': 22})
            plt.colorbar(mappable, ticks=v)

        plt.axis('off')
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def get_room_xy(room):
        if KNOWN_XY[room] is None:
            for y, l in enumerate(PYRAMID):
                if room in l:
                    KNOWN_XY[room] = (l.index(room), y)
                    break
        return KNOWN_XY[room]

    @staticmethod
    def get_room_out_of_bounds(room_x, room_y):
        return room_y < 0 or room_x < 0 or room_y >= len(PYRAMID) or room_x >= len(PYRAMID[0])

    @staticmethod
    def get_room_from_xy(room_x, room_y):
        return PYRAMID[room_y][room_x]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, ob):
        self.__dict__ = ob
