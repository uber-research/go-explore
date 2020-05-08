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
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any
from collections import defaultdict
from atari_reset.atari_reset.wrappers import MyWrapper


class PitfallPosLevel:
    __slots__ = ['level', 'score', 'room', 'x', 'y', 'tuple']

    # noinspection PyUnusedLocal
    def __init__(self, level, score, room, x, y):
        self.level = 0
        self.score = 0
        self.room = room
        self.x = x
        self.y = y
        self.tuple = None

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.level, self.score, self.room, self.x, self.y)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, PitfallPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d


TARGET_SHAPE = None
MAX_PIX_VALUE = None


def clip(a, min_v, max_v):
    if a < min_v:
        return min_v
    if a > max_v:
        return max_v
    return a


class MyPitfall(MyWrapper):
    TARGET_SHAPE = None
    MAX_PIX_VALUE = None
    #: The original width of the screen
    screen_width = 160
    #: The original height of the screen
    screen_height = 210
    #: A factor by which to multiply the width of the screen to account for the fact that pixels where assumed to be
    #: wider than they were tall when displayed on a television.
    x_repeat = 2
    #: The space, in pixels, on the top of the screen that displays information and that can not be reached by the
    #: player
    gui_size = 50
    #: A rough estimate of the y position of the ground on screen. Used to determine whether the player is above or
    #: below ground.
    ground_y = 70
    #: If the player moves more than this distance along the x-axis in a single frame, this is considered a room
    #: transition. Otherwise, large jumps in player position indicate player death and respawn.
    x_jump_threshold = 270
    #: If a score increase exceeds this value, we know that a treasure has been collected.
    treasure_collected_threshold = 100
    game_screen_height = screen_height - gui_size
    nb_rooms = 255
    attr_max = {'treasures': 32,
                'room': nb_rooms}

    @staticmethod
    def get_attr_max(name):
        if name == 'x':
            return MyPitfall.screen_width * MyPitfall.x_repeat
        elif name == 'y':
            return MyPitfall.game_screen_height
        else:
            return MyPitfall.attr_max[name]

    def __init__(self,
                 env: Any,
                 cell_representation: Any = None):
        super(MyPitfall, self).__init__(env)
        self.env.reset()
        self.ram = None
        self.cur_steps = 0
        self.cur_score = 0
        self.rooms = {}
        self.room_time = None
        self.room_threshold = 40
        self.unwrapped.seed(0)
        self.state = []

        self.cell_representation = cell_representation

        self.pos = None
        self.treasures = None
        self.room = None
        self.x = None
        self.y = None
        self.done = None

    def __getattr__(self, e):
        return getattr(self.env, e)

    def get_my_montezuma(self):
        return self

    def reset(self) -> np.ndarray:
        unprocessed_state = self.env.reset()
        self.state = [self.convert_state(unprocessed_state)]
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = 0
        self.cur_steps = 0
        self.pos = None
        self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state))
        self.pos = self.cell_representation(self)
        self.treasures = 0
        self.done = 0
        if self.room not in self.rooms:
            self.rooms[self.room] = (True, unprocessed_state[MyPitfall.gui_size:].repeat(self.x_repeat, axis=1))
        self.room_time = (self.room, 0)
        return unprocessed_state

    def pos_from_unprocessed_state(self, face_pixels):
        face_pixels = [(y, x * self.x_repeat) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos is not None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        room = 0
        # level = 0
        if self.pos is not None:
            direction_x = clip(int((self.x - x) / MyPitfall.x_jump_threshold), -1, 1)
            if y < MyPitfall.ground_y:
                room = (self.room + direction_x) % MyPitfall.nb_rooms
            else:
                room = (self.room + direction_x*3) % MyPitfall.nb_rooms

        self.room = room
        self.x = x
        self.y = y

        return room, x, y

    def get_restore(self):
        return (
            self.unwrapped.clone_full_state(),
            copy.copy(self.state),
            self.cur_score,
            self.cur_steps,
            self.pos,
            self.room_time,
        )

    def restore(self, data):
        (full_state, state, score, steps, pos, room_time) = data
        self.state = copy.copy(state)
        self.env.reset()
        self.unwrapped.restore_full_state(full_state)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = score
        self.cur_steps = steps
        self.pos = pos
        self.room_time = room_time
        return copy.copy(self.state)

    def get_face_pixels(self, unprocessed_state):
        result = set(zip(*np.where(unprocessed_state[MyPitfall.gui_size:, :, 0] == 228)))
        return result

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        unprocessed_state, reward, done, lol = self.env.step(action)
        self.state.append(self.convert_state(unprocessed_state))
        self.state.pop(0)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_steps += 1

        face_pixels = self.get_face_pixels(unprocessed_state)

        self.cur_score += reward
        self.pos_from_unprocessed_state(face_pixels)

        if self.room != self.room_time[0]:
            self.room_time = (self.room, 0)
        self.room_time = (self.room, self.room_time[1] + 1)
        if self.room not in self.rooms:
            self.rooms[self.room] = (True, unprocessed_state[MyPitfall.gui_size:].repeat(self.x_repeat, axis=1))

        if reward >= MyPitfall.treasure_collected_threshold:
            self.treasures += 1

        if done:
            self.treasures = 0
            self.room = 0
            self.x = 0
            self.y = 0
            self.done = 1

        self.pos = self.cell_representation(self)

        return unprocessed_state, reward, done, lol

    def get_pos(self):
        assert self.pos is not None
        return self.pos

    # noinspection PyUnusedLocal
    def render_with_known(self, known_positions, x_res, y_res, show=False, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None, log_scale=False):
        height, width = list(self.rooms.values())[0][1].shape[:2]

        final_image = np.zeros((height * 22, width * 12, 3), dtype=np.uint8) + MyPitfall.nb_rooms

        def room_pos(room_):
            y_ = room_ % 12
            x_ = int(room_ / 12)
            return x_, y_

        points = defaultdict(int)

        img = np.zeros((height, width, 3)) + 127
        for room in range(MyPitfall.nb_rooms):
            if room in self.rooms:
                img = self.rooms[room][1]
            else:
                img = np.zeros((height, width, 3)) + 127
            y_room, x_room = room_pos(room)
            y_room *= height
            x_room *= width
            final_image[y_room:y_room + height, x_room:x_room + width, :] = img

        plt.figure(figsize=(final_image.shape[1] // 40, final_image.shape[0] // 40))

        for room in range(255):
            y_room, x_room = room_pos(room)
            y_room *= height
            x_room *= width

            for i in np.arange(int(y_res), img.shape[0], int(y_res)):
                cv2.line(final_image, (x_room, y_room + i), (x_room + img.shape[1], y_room + i), (127, 127, 127), 1)
            for i in np.arange(int(x_res), img.shape[1], int(x_res)):
                cv2.line(final_image, (x_room + i, y_room), (x_room + i, y_room + img.shape[0]), (127, 127, 127), 1)

            cv2.line(final_image, (x_room, y_room), (x_room, y_room + img.shape[0]), (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room), (x_room + img.shape[1], y_room), (255, 255, 255), 1)
            cv2.line(final_image, (x_room + img.shape[1], y_room), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room + img.shape[0]), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)

            for k in known_positions:
                if k.room != room:
                    continue
                x = x_room + (k.x * x_res + x_res / 2)
                y = y_room + (k.y * y_res + y_res / 2)
                points[(x, y)] = combine_val(points[(x, y)], get_val(k))

        plt.imshow(final_image)
        if minmax:
            points[(0, 0)] = minmax[0]
            points[(0, 1)] = minmax[1]

        vals = list(points.values())
        points = list(points.items())
        plt.scatter([p[0][0] for p in points], [p[0][1] for p in points], c=[p[1] for p in points], cmap='bwr',
                    s=x_res ** 2, marker='*')
        plt.legend()

        import matplotlib.cm
        import matplotlib.colors
        mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.min(vals), vmax=np.max(vals)),
                                                cmap='bwr')
        mappable.set_array(np.array(vals))
        matplotlib.rcParams.update({'font.size': 50})
        plt.colorbar(mappable, fraction=0.043, pad=0.01)

        plt.axis('off')
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def get_room_xy(room):
        return room, 0

    @staticmethod
    def get_room_out_of_bounds(room_x, room_y):
        if room_y != 0:
            return True
        elif room_x < 0 or room_x >= MyPitfall.nb_rooms:
            return True

    @staticmethod
    def get_room_from_xy(room_x, _):
        return room_x

    @staticmethod
    def convert_state(state):
        if MyPitfall.TARGET_SHAPE is None:
            return None
        import cv2
        return ((cv2.resize(
            cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), MyPitfall.TARGET_SHAPE, interpolation=cv2.INTER_AREA) / 255.0) *
                MyPitfall.MAX_PIX_VALUE).astype(np.uint8)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, ob):
        self.__dict__ = ob
