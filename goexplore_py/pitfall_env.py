# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from .import_ai import *


class PitfallPosLevel:
    __slots__ = ['level', 'score', 'room', 'x', 'y', 'tuple']

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


def clip(a, m, M):
    if a < m:
        return m
    if a > M:
        return M
    return a


class MyPitfall:
    TARGET_SHAPE = None
    MAX_PIX_VALUE = None

    def __init__(self, check_death: bool = True, unprocessed_state: bool = False, score_objects: bool = False,
                 x_repeat=2):
        self.env = gym.make('PitfallDeterministic-v4')
        self.env.reset()
        self.score_objects = score_objects
        self.ram = None
        self.check_death = check_death
        self.cur_steps = 0
        self.cur_score = 0
        self.rooms = {}
        self.room_time = None
        self.room_threshold = 40
        self.unwrapped.seed(0)
        self.unprocessed_state = unprocessed_state
        self.state = []
        self.ram_death_state = -1
        self.x_repeat = x_repeat
        self.cur_lives = 5
        self.ignore_ram_death = False

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        unprocessed_state = self.env.reset()
        self.cur_lives = 5
        self.state = [self.convert_state(unprocessed_state)]
        for _ in range(3):
            unprocessed_state = self.env.step(0)[0]
            self.state.append(self.convert_state(unprocessed_state))
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = 0
        self.cur_steps = 0
        self.ram_death_state = -1
        self.pos = None
        self.pos = self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state))
        if self.get_pos().room not in self.rooms:
            self.rooms[self.get_pos().room] = (False, unprocessed_state[50:].repeat(self.x_repeat, axis=1))
        self.room_time = (self.get_pos().room, 0)
        if self.unprocessed_state:
            return unprocessed_state
        return copy.copy(self.state)

    def pos_from_unprocessed_state(self, face_pixels):
        face_pixels = [(y, x * self.x_repeat) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos != None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        room = 0
        if self.pos is not None:
            direction_x = clip(int((self.pos.x - x) / 270), -1, 1)
            if y < 70:
                room = (self.pos.room + direction_x) % 255
            else:
                room = (self.pos.room + direction_x*3) % 255

        score = self.cur_score
        if self.score_objects:  # TODO: detect objects from the frame!
            score = self.ram[65]
        return PitfallPosLevel(0, score, room, x, y)

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
        self.env.reset()
        self.unwrapped.restore_full_state(full_state)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = score
        self.cur_steps = steps
        self.pos = pos
        self.room_time = room_time
        self.ram_death_state = ram_death_state
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
        # TODO: double check that this color does not re-occur somewhere else
        # in the environment.
        result = set(zip(*np.where(unprocessed_state[50:, :, 0] == 228)))
        return result

    def is_pixel_death(self, unprocessed_state, face_pixels):
        # There are no face pixels and yet we are not in a transition screen. We
        # must be dead!
        if len(face_pixels) == 0:
            # All of the screen except the bottom is black: this is not a death but a
            # room transition. Ignore.
            if self.is_transition_screen(unprocessed_state):
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

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        unprocessed_state, reward, done, lol = self.env.step(action)
        self.state.append(self.convert_state(unprocessed_state))
        self.state.pop(0)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_steps += 1

        face_pixels = self.get_face_pixels(unprocessed_state)

        self.cur_score += reward
        self.pos = self.pos_from_unprocessed_state(face_pixels)
        if self.pos.room != self.room_time[0]:
            self.room_time = (self.pos.room, 0)
        self.room_time = (self.pos.room, self.room_time[1] + 1)
        if (self.pos.room not in self.rooms or
                (self.room_time[1] == self.room_threshold and
                 not self.rooms[self.pos.room][0])):
            self.rooms[self.pos.room] = (
                self.room_time[1] == self.room_threshold,
                unprocessed_state[50:].repeat(self.x_repeat, axis=1)
            )
        if self.unprocessed_state:
            return unprocessed_state, reward, done, lol
        return copy.copy(self.state), reward, done, lol

    def get_pos(self):
        assert self.pos is not None
        return self.pos

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        height, width = list(self.rooms.values())[0][1].shape[:2]
        final_image = np.zeros((height * 22, width * 12, 3), dtype=np.uint8) + 255

        def room_pos(room):
            y = room % 12
            x = int(room / 12)
            return x, y

        points = defaultdict(int)

        for room in range(255):
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

            for i in np.arange(int(resolution), img.shape[0], int(resolution)):
                cv2.line(final_image, (x_room, y_room + i), (x_room + img.shape[1], y_room + i), (127, 127, 127), 1)
            for i in np.arange(int(resolution), img.shape[1], int(resolution)):
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
                # TODO: check offsets
                x = x_room + (k.x * resolution + resolution / 2)
                y = y_room + (k.y * resolution + resolution / 2)
                points[(x, y)] = combine_val(points[(x, y)], get_val(k))

        plt.imshow(final_image)
        if minmax:
            points[(0, 0)] = minmax[0]
            points[(0, 1)] = minmax[1]

        vals = list(points.values())
        points = list(points.items())
        plt.scatter([p[0][0] for p in points], [p[0][1] for p in points], c=[p[1] for p in points], cmap='bwr',
                    s=(resolution) ** 2, marker='*')
        plt.legend()

        import matplotlib.cm
        import matplotlib.colors
        mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.min(vals), vmax=np.max(vals)),
                                                cmap='bwr')
        mappable.set_array(vals)
        matplotlib.rcParams.update({'font.size': 50})
        plt.colorbar(mappable, fraction=0.043, pad=0.01)

        plt.axis('off')
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            # plt.savefig(filename, bbox_inches=0.001)
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
        elif room_x < 0 or room_x > 254:
            return True

    @staticmethod
    def get_room_from_xy(room_x, _):
        return room_x

    @staticmethod
    def make_pos(score, pos):
        return PitfallPosLevel(pos.level, score, pos.room, pos.x, pos.y)

    @staticmethod
    def convert_state(state):
        import cv2
        return ((cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), MyPitfall.TARGET_SHAPE, interpolation=cv2.INTER_AREA) / 255.0) * MyPitfall.MAX_PIX_VALUE).astype(np.uint8)
