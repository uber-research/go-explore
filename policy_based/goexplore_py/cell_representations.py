# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from typing import List, Any, Type
from goexplore_py.montezuma_env import MyMontezuma
from goexplore_py.pitfall_env import MyPitfall


class CellRepresentationBase:
    __slots__ = []
    supported_games = ()

    @staticmethod
    def make(env=None) -> Any:
        raise NotImplementedError('Cell representation needs to implement make')

    @staticmethod
    def get_array_length() -> int:
        raise NotImplementedError('Cell representation needs to implement get_array_length')

    @staticmethod
    def get_attributes() -> List[str]:
        raise NotImplementedError('Cell representation needs to implement get_attributes')

    @staticmethod
    def get_attr_max(name) -> int:
        raise NotImplementedError('Cell representation needs to implement get_attr_max')

    def as_array(self) -> np.ndarray:
        raise NotImplementedError('Cell representation needs to implement as_array')


class CellRepresentationFactory:
    def __init__(self, cell_rep_class: Type[CellRepresentationBase]):
        self.cell_rep_class: Type[CellRepresentationBase] = cell_rep_class
        self.array_length: int = self.cell_rep_class.get_array_length()
        self.grid_resolution = None
        self.grid_res_dict = None
        self.max_values = None

    def __call__(self, env=None):
        cell_representation = self.cell_rep_class.make(env)

        if env is not None:
            for dimension in self.grid_resolution:
                if dimension.div != 1:
                    value = getattr(cell_representation, dimension.attr)
                    value = (int(value / dimension.div))
                    setattr(cell_representation, dimension.attr, value)

        return cell_representation

    def set_grid_resolution(self, grid_resolution):
        self.grid_resolution = grid_resolution
        self.grid_res_dict = {}
        self.max_values = []

        for dimension in self.grid_resolution:
            self.grid_res_dict[dimension.attr] = dimension.div

        for attr_name in self.cell_rep_class.get_attributes():
            max_val = self.cell_rep_class.get_attr_max(attr_name)
            if attr_name in self.grid_res_dict:
                max_val, remainder = divmod(max_val, self.grid_res_dict[attr_name])
                if remainder > 0:
                    max_val += 1
            self.max_values.append(max_val)

    def get_max_values(self):
        return self.max_values

    def supported(self, game_name):
        return game_name in self.cell_rep_class.supported_games


class RoomXY(CellRepresentationBase):
    __slots__ = ['_room', '_x', '_y', '_done', 'tuple']
    attributes = ('room', 'x', 'y', 'done')
    array_length = 4
    supported_games = ('pitfall', 'montezuma')

    @staticmethod
    def get_attr_max(name):
        if name == 'done':
            return 2
        return MyPitfall.get_attr_max(name)

    @staticmethod
    def get_array_length():
        return RoomXY.array_length

    @staticmethod
    def get_attributes():
        return RoomXY.attributes

    def __init__(self, pitfall_env=None):
        self._room = None
        self._x = None
        self._y = None
        self._done = None
        self.tuple = None

        if pitfall_env is not None:
            self._room = pitfall_env.room
            self._x = pitfall_env.x
            self._y = pitfall_env.y
            self._done = pitfall_env.done
            self.set_tuple()

    @staticmethod
    def make(env=None):
        return RoomXY(env)

    def set_tuple(self):
        self.tuple = (self._room, self._x, self._y, self._done)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.set_tuple()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.set_tuple()

    @property
    def room(self):
        return self._room

    @room.setter
    def room(self, value):
        self._room = value
        self.set_tuple()

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        self._done = value
        self.set_tuple()

    def as_array(self):
        return np.array(self.tuple)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, RoomXY):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self._room, self._x, self._y, self._done = d
        self.tuple = d

    def __repr__(self):
        return f'room={self._room} x={self._x} y={self._y} done={self._done}'


class PitfallPosLevel(CellRepresentationBase):
    __slots__ = ['_treasures', '_room', '_x', '_y', '_done', 'tuple']
    attributes = ('treasures', 'room', 'x', 'y', 'done')
    array_length = 5
    supported_games = ('pitfall',)

    @staticmethod
    def get_attr_max(name):
        if name == 'done':
            return 2
        return MyPitfall.get_attr_max(name)

    @staticmethod
    def get_array_length():
        return PitfallPosLevel.array_length

    @staticmethod
    def get_attributes():
        return PitfallPosLevel.attributes

    def __init__(self, pitfall_env=None):
        self._treasures = 0
        self._room = None
        self._x = None
        self._y = None
        self._done = None
        self.tuple = None

        if pitfall_env is not None:
            self._treasures = pitfall_env.treasures
            self._room = pitfall_env.room
            self._x = pitfall_env.x
            self._y = pitfall_env.y
            self._done = pitfall_env.done
            self.set_tuple()

    @staticmethod
    def make(env=None):
        return PitfallPosLevel(env)

    def set_tuple(self):
        self.tuple = (self._treasures, self._room, self._x, self._y, self._done)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.set_tuple()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.set_tuple()

    @property
    def treasures(self):
        return self._treasures

    @treasures.setter
    def treasures(self, value):
        self._treasures = value
        self.set_tuple()

    @property
    def room(self):
        return self._room

    @room.setter
    def room(self, value):
        self._room = value
        self.set_tuple()

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        self._done = value
        self.set_tuple()

    def as_array(self):
        return np.array(self.tuple)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, PitfallPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self._treasures, self._room, self._x, self._y, self._done = d
        self.tuple = d

    def __repr__(self):
        return f'room={self._room} treasures={self._treasures} x={self._x} y={self._y} done={self._done}'


class MontezumaPosLevel(CellRepresentationBase):
    __slots__ = ['_level', '_objects', '_room', '_x', '_y', '_done', 'tuple']
    attributes = ('level', 'objects', 'room', 'x', 'y', 'done')
    array_length = 6
    supported_games = ('montezuma', 'montezumarevenge')

    @staticmethod
    def get_attr_max(name):
        if name == 'done':
            return 2
        return MyMontezuma.get_attr_max(name)

    @staticmethod
    def get_array_length():
        return MontezumaPosLevel.array_length

    @staticmethod
    def get_attributes():
        return MontezumaPosLevel.attributes

    @staticmethod
    def make(env=None):
        return MontezumaPosLevel(env)

    def __init__(self, montezuma_env=None):
        self.tuple = None
        self._level = None
        self._objects = None
        self._room = None
        self._x = None
        self._y = None
        self._done = None
        if montezuma_env is not None:
            self._level = montezuma_env.level
            self._objects = montezuma_env.objects
            self._room = montezuma_env.room
            self._x = montezuma_env.x
            self._y = montezuma_env.y
            self._done = montezuma_env.done
            self.set_tuple()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.set_tuple()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.set_tuple()

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value
        self.set_tuple()

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value
        self.set_tuple()

    @property
    def room(self):
        return self._room

    @room.setter
    def room(self, value):
        self._room = value
        self.set_tuple()

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        self._done = value
        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self._level, self._objects, self._room, self._x, self._y, self._done)

    def as_array(self):
        self.set_tuple()
        return np.array(self.tuple)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def non_pos_as_array(self):
        return np.array((self._level, self._objects, self._room))

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, MontezumaPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        # Backwards compatibility
        if len(d) == 5:
            self._level, self._objects, self._room, self._x, self._y = d
            self._done = 0
        else:
            self._level, self._objects, self._room, self._x, self._y, self._done = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self._level} Room={self._room} Objects={self._objects} ' \
            f'x={self._x} y={self._y} done={self._done}'


class LevelKeysRoomXYScore(CellRepresentationBase):
    __slots__ = ['_level', '_objects', '_room', '_x', '_y', '_score', '_done', 'tuple', 'no_score_tuple']
    attributes = ('level', 'objects', 'room', 'x', 'y', 'score')
    array_length = 7
    supported_games = ('montezuma', 'montezumarevenge')

    @staticmethod
    def get_attr_max(name):
        if name == 'done':
            return 2
        if name == 'score':
            return 0
        return MyMontezuma.get_attr_max(name)

    @staticmethod
    def get_array_length():
        return LevelKeysRoomXYScore.array_length

    @staticmethod
    def get_attributes():
        return LevelKeysRoomXYScore.attributes

    @staticmethod
    def make(env=None):
        return LevelKeysRoomXYScore(env)

    def __init__(self, montezuma_env=None):
        self.tuple = None
        self._level = None
        self._objects = None
        self._room = None
        self._x = None
        self._y = None
        self._score = None
        self._done = None
        if montezuma_env is not None:
            self._level = montezuma_env.level
            self._objects = montezuma_env.objects
            self._room = montezuma_env.room
            self._x = montezuma_env.x
            self._y = montezuma_env.y
            self._score = int(montezuma_env.cur_score)
            self._done = montezuma_env.done
            self.set_tuple()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.set_tuple()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.set_tuple()

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, value):
        self._objects = value
        self.set_tuple()

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value
        self.set_tuple()

    @property
    def room(self):
        return self._room

    @room.setter
    def room(self, value):
        self._room = value
        self.set_tuple()

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = int(value)
        self.set_tuple()

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        self._done = value
        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self._level, self._objects, self._room, self._x, self._y, self._score, self._done)

    def as_array(self):
        return np.array(self.tuple)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def non_pos_as_array(self):
        return np.array((self._level, self._objects, self._room))

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, LevelKeysRoomXYScore):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self._level, self._objects, self._room, self._x, self._y, self._score, self._done = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self._level} Room={self._room} Objects={self._objects} x={self._x} ' \
            f'y={self._y} score={self._score} done={self.done}'
