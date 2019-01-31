# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from .import_ai import *


class MontezumaPosLevel:
    __slots__ = ['level', 'score', 'room', 'x', 'y', 'tuple']

    def __init__(self, level, score, room, x, y):
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
        if not isinstance(other, MontezumaPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}'


def convert_state(state):
    import cv2
    return ((cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), MyMontezuma.TARGET_SHAPE, interpolation=cv2.INTER_AREA) / 255.0) * MyMontezuma.MAX_PIX_VALUE).astype(np.uint8)


PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23]
]

OBJECT_PIXELS = [
    50,  # Hammer/mallet
    40,  # Key 1
    40,  # Key 2
    40,  # Key 3
    37,  # Sword 1
    37,  # Sword 2
    42   # Torch
]

KNOWN_XY = [None] * 24

KEY_BITS = 0x8 | 0x4 | 0x2


def get_room_xy(room):
    if KNOWN_XY[room] is None:
        for y, l in enumerate(PYRAMID):
            if room in l:
                KNOWN_XY[room] = (l.index(room), y)
                break
    return KNOWN_XY[room]


def clip(a, m, M):
    if a < m:
        return m
    if a > M:
        return M
    return a


class MyMontezuma:
    def __init__(self, check_death: bool = True, unprocessed_state: bool = False, score_objects: bool = False,
                 x_repeat=2, objects_from_pixels=True, objects_remember_rooms=False, only_keys=False):  # TODO: version that also considers the room objects were found in
        self.env = gym.make('MontezumaRevengeDeterministic-v4')
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
        self.objects_from_pixels = objects_from_pixels
        self.objects_remember_rooms = objects_remember_rooms
        self.only_keys = only_keys

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        unprocessed_state = self.env.reset()
        self.cur_lives = 5
        self.state = [convert_state(unprocessed_state)]
        for _ in range(3):
            unprocessed_state = self.env.step(0)[0]
            self.state.append(convert_state(unprocessed_state))
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = 0
        self.cur_steps = 0
        self.ram_death_state = -1
        self.pos = None
        self.pos = self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state), unprocessed_state)
        if self.get_pos().room not in self.rooms:
            self.rooms[self.get_pos().room] = (False, unprocessed_state[50:].repeat(self.x_repeat, axis=1))
        self.room_time = (self.get_pos().room, 0)
        if self.unprocessed_state:
            return unprocessed_state
        return copy.copy(self.state)

    def pos_from_unprocessed_state(self, face_pixels, unprocessed_state):
        face_pixels = [(y, x * self.x_repeat) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos != None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        room = 1
        level = 0
        old_objects = tuple()
        if self.pos is not None:
            room = self.pos.room
            level = self.pos.level
            old_objects = self.pos.score
            direction_x = clip(int((self.pos.x - x) / 50), -1, 1)
            direction_y = clip(int((self.pos.y - y) / 50), -1, 1)
            if direction_x != 0 or direction_y != 0:
                # TODO(AE): shoudln't this call the static method?
                room_x, room_y = get_room_xy(self.pos.room)
                if room == 15 and room_y + direction_y >= len(PYRAMID):
                    room = 1
                    level += 1
                else:
                    assert direction_x == 0 or direction_y == 0, f'Room change in more than two directions : ({direction_y}, {direction_x})'
                    room = PYRAMID[room_y + direction_y][room_x + direction_x]
                    assert room != -1, f'Impossible room change: ({direction_y}, {direction_x})'

        score = self.cur_score
        if self.score_objects:  # TODO: detect objects from the frame!
            if not self.objects_from_pixels:
                score = self.ram[65]
                if self.only_keys:
                    # These are the key bytes
                    score &= KEY_BITS
            else:
                score = self.get_objects_from_pixels(unprocessed_state, room, old_objects)
        return MontezumaPosLevel(level, score, room, x, y)

    def get_objects_from_pixels(self, unprocessed_state, room, old_objects):
        object_part = (unprocessed_state[25:45, 55:110, 0] != 0).astype(np.uint8) * 255
        connected_components = cv2.connectedComponentsWithStats(object_part)
        pixel_areas = list(e[-1] for e in connected_components[2])[1:]

        # Note: this "ground truth" logic is excluded because the RAM and the frames are
        # not updated at the same time, so that the asserts would fail even in normal cases.
        # Commented all related code and not just the asserts so that we don't waste time making
        # checks that are impossible.

        # expected = self.ram[65]
        # if self.only_keys:
        #     expected &= KEY_BITS
        #
        # ground_truth = defaultdict(int)
        # for i, n_pixels in enumerate(OBJECT_PIXELS):
        #     if (1 << i) & expected:
        #         ground_truth[n_pixels] += 1

        if self.objects_remember_rooms:
            cur_object = []
            old_objects = list(old_objects)
            for i, n_pixels in enumerate(OBJECT_PIXELS):
                if n_pixels != 40 and self.only_keys:
                    continue
                if n_pixels in pixel_areas:
                    # ground_truth[n_pixels] -= 1

                    pixel_areas.remove(n_pixels)
                    orig_types = [e[0] for e in old_objects]
                    if n_pixels in orig_types:
                        idx = orig_types.index(n_pixels)
                        cur_object.append((n_pixels, old_objects[idx][1]))
                        old_objects.pop(idx)
                    else:
                        cur_object.append((n_pixels, room))

            # TODO: bring back these asserts. Unfortunately the ram and the frames aren't updated at
            # the same time so these would normally fail :(
            #assert all(e == 0 for e in ground_truth.values())
            return tuple(cur_object)

        else:
            cur_object = 0
            for i, n_pixels in enumerate(OBJECT_PIXELS):
                if n_pixels in pixel_areas:
                    # ground_truth[n_pixels] -= 1

                    pixel_areas.remove(n_pixels)
                    cur_object |= 1 << i

            if self.only_keys:
                # These are the key bytes
                cur_object &= KEY_BITS

            # TODO: bring back these asserts. Unfortunately the ram and the frames aren't updated at
            # the same time so these would normally fail :(
            #assert all(e == 0 for e in ground_truth.values())
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
        return set(zip(*np.where(unprocessed_state[50:, :, 0] == 228)))

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
        self.state.append(convert_state(unprocessed_state))
        self.state.pop(0)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_steps += 1

        face_pixels = self.get_face_pixels(unprocessed_state)
        pixel_death = self.is_pixel_death(unprocessed_state, face_pixels)
        ram_death = self.is_ram_death()
        # TODO: remove all this stuff
        if self.check_death and pixel_death:
            if not ram_death:
                print('Image-detected death. Check it!', self.ram[55], self.ram[58])
                plt.imshow(unprocessed_state[50:, :, :])
                print(np.sum(unprocessed_state[51:, :, :] == 0), unprocessed_state[50:].size)
                plt.show()
                print(np.where(unprocessed_state[:, :, 0] == 228))
                plt.imshow(unprocessed_state[:, :, 0] == 228)
                plt.show()
                print(Counter(list(zip(unprocessed_state[51:, :, 0].flatten(), unprocessed_state[51:, :, 1].flatten(),
                                       unprocessed_state[51:, :, 2].flatten()))))
            done = True
        elif self.check_death and not pixel_death and ram_death:
            if self.ram_death_state == -1:
                self.ram_death_state = self.cur_steps
            if self.cur_steps - self.ram_death_state > 0 and not self.ignore_ram_death:
                print('OMG, undetected death!!!')
                self.ignore_ram_death = True
                plt.imshow(unprocessed_state)
                plt.show()
                plt.imshow(unprocessed_state[:, :, 0] == 228)
                plt.show()
                print(np.where(unprocessed_state[:, :, 0] == 228))
                print(self.ram)
                print(Counter(list(zip(unprocessed_state[:, :, 0].flatten(), unprocessed_state[:, :, 1].flatten(),
                                       unprocessed_state[:, :, 2].flatten()))))
                # done = True

        self.cur_score += reward
        self.pos = self.pos_from_unprocessed_state(face_pixels, unprocessed_state)
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

        final_image = np.zeros((height * 4, width * 9, 3), dtype=np.uint8) + 255

        positions = PYRAMID

        def room_pos(room):
            for height, l in enumerate(positions):
                for width, r in enumerate(l):
                    if r == room:
                        return (height, width)
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

        plt.figure(figsize=(final_image.shape[1] // 30, final_image.shape[0] // 30))

        for room in range(24):
            y_room, x_room = room_pos(room)
            y_room *= height
            x_room *= width

            for i in np.arange(resolution, img.shape[0], resolution):
                cv2.line(final_image, (x_room, y_room + i), (x_room + img.shape[1], y_room + i), (127, 127, 127), 1)
                # plt.plot([x_room, x_room + img.shape[1]], [y_room + i, y_room + i], '--', linewidth=1, color='gray')
            for i in np.arange(resolution, img.shape[1], resolution):
                cv2.line(final_image, (x_room + i, y_room), (x_room + i, y_room + img.shape[0]), (127, 127, 127), 1)
                # plt.plot([x_room + i, x_room + i], [y_room, y_room + img.shape[0]], '--', linewidth=1, color='gray')

            cv2.line(final_image, (x_room, y_room), (x_room, y_room + img.shape[0]), (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room), (x_room + img.shape[1], y_room), (255, 255, 255), 1)
            cv2.line(final_image, (x_room + img.shape[1], y_room), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room + img.shape[0]), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)

            for k in known_positions:
                if k.room != room:
                    continue
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
        matplotlib.rcParams.update({'font.size': 22})
        plt.colorbar(mappable)

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

    @staticmethod
    def make_pos(score, pos):
        return MontezumaPosLevel(pos.level, score, pos.room, pos.x, pos.y)
