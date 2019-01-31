# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from .import_ai import *

# from montezuma_env import *


@dataclass()
class Weight:
    weight: float = 1.0
    power: float = 1.0

    def __repr__(self):
        return f'w={self.weight:.2f}=p={self.power:.2f}'


@dataclass()
class DirWeights:
    horiz: float = 2.0
    vert: float = 0.3
    score_low: float = 0.0
    score_high: float = 0.0

    def __repr__(self):
        return f'h={self.horiz:.2f}=v={self.vert:.2f}=l={self.score_low:.2f}=h={self.score_high:.2f}'


def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def convert_score(e):
    # TODO: this doesn't work when actual score is used!! Fix?
    if isinstance(e, tuple):
        return len(e)
    return numberOfSetBits(e)


class WeightedSelector:
    def __init__(self, game, seen=Weight(0.1), chosen=Weight(), action=Weight(0.1, power=0.5),
                 room_cells=Weight(0.0, power=0.5), dir_weights=DirWeights(), low_level_weight=0.0,
                 chosen_since_new_weight=Weight()):
        self.seen: Weight = seen
        self.chosen: Weight = chosen
        self.chosen_since_new_weight: Weight = chosen_since_new_weight
        self.room_cells: Weight = room_cells
        self.dir_weights: DirWeights = dir_weights
        self.action: Weight = action
        self.low_level_weight: float = low_level_weight
        self.game = game

    def reached_state(self, elem):
        pass

    def update(self):
        pass

    def compute_weight(self, value, weight):
        return weight.weight * 1 / (value + 0.001) ** weight.power + 0.00001

    def get_seen_weight(self, cell):
        return self.compute_weight(cell.seen_times, self.seen)

    def get_chosen_weight(self, cell):
        return self.compute_weight(cell.chosen_times, self.chosen)

    def get_chosen_since_new_weight(self, cell):
        return self.compute_weight(cell.chosen_since_new, self.chosen_since_new_weight)

    def get_action_weight(self, cell):
        return self.compute_weight(cell.action_times, self.action)

    def no_neighbor(self, pos, offset, known_cells):
        x = pos.x + offset[0]
        y = pos.y + offset[1]
        room = pos.room
        room_x, room_y = self.game.get_room_xy(room)
        if x < self.xrange[0]:
            x = self.xrange[1]
            room_x -= 1
        elif x > self.xrange[1]:
            x = self.xrange[0]
            room_x += 1
        elif y < self.yrange[0]:
            y = self.yrange[1]
            room_y -= 1
        elif y > self.yrange[1]:
            y = self.yrange[0]
            room_y += 1
        if self.game.get_room_out_of_bounds(room_x, room_y):
            return True
        room = self.game.get_room_from_xy(room_x, room_y)
        if room == -1:
            return True
        new_pos = copy.copy(pos)
        new_pos.room = room,
        new_pos.x = x
        new_pos.y = y
        res = self.game.make_pos(pos.score, new_pos) not in known_cells
        return res

    def get_pos_weight(self, pos, cell, known_cells, possible_scores):
        if isinstance(pos, tuple):
            # Logic for the score stuff: the highest score will get a weight of 1, second highest a weight of sqrt(1/2), third sqrt(1/3) etc.
            return 1 + self.dir_weights.score_high * 1 / np.sqrt(len(possible_scores) - possible_scores.index(cell.score))
        no_low = True
        if convert_score(pos.score) == convert_score(possible_scores[0]):
            pass
        else:
            for score in possible_scores:
                if convert_score(score) >= convert_score(pos.score):
                    break
                if self.game.make_pos(score, pos) in known_cells:
                    no_low = False
                    break

        no_high = True
        if convert_score(pos.score) == convert_score(possible_scores[-1]):
            pass
        else:
            for score in reversed(possible_scores):
                if convert_score(score) <= convert_score(pos.score):
                    break
                if self.game.make_pos(score, pos) in known_cells:
                    no_high = False
                    break

        neigh_horiz = 0.0
        if self.dir_weights.horiz:
            neigh_horiz = (self.no_neighbor(pos, (-1, 0), known_cells) + self.no_neighbor(pos, (1, 0), known_cells))
        neigh_vert = 0.0
        if self.dir_weights.vert:
            neigh_vert = (self.no_neighbor(pos, (0, -1), known_cells) + self.no_neighbor(pos, (0, 1), known_cells))

        res = self.dir_weights.horiz * neigh_horiz + self.dir_weights.vert * neigh_vert + self.dir_weights.score_low * no_low + self.dir_weights.score_high * no_high + 1
        return res

    def get_weight(self, cell_key, cell, possible_scores, known_cells):
        level_weight = 1.0
        if not isinstance(cell_key, tuple) and cell_key.level < self.max_level:
            level_weight = self.low_level_weight ** (self.max_level - cell_key.level)
        if level_weight == 0.0:
            return 0.0
        res = (self.get_pos_weight(cell_key, cell, known_cells, possible_scores) +
               self.get_seen_weight(cell) +
               self.get_chosen_weight(cell) +
               self.get_action_weight(cell) +
               self.get_chosen_since_new_weight(cell)
               ) * level_weight
        return res

    def set_ranges(self, to_choose):
        if isinstance(to_choose[0], tuple):
            return
        self.xrange = (min(e.x for e in to_choose), max(e.x for e in to_choose))
        self.yrange = (min(e.y for e in to_choose), max(e.y for e in to_choose))
        self.max_level = max(e.level for e in to_choose)

    def choose_cell(self, known_cells, size=1):
        to_choose = list(known_cells.keys())
        self.set_ranges(to_choose)
        if not isinstance(to_choose[0], tuple):
            possible_scores = sorted(set(e.score for e in to_choose), key=convert_score)
        else:
            possible_scores = sorted(set(e.score for e in known_cells.values()))
        if len(to_choose) == 1:
            return [to_choose[0]] * size
        weights = [
            self.get_weight(
                k, known_cells[k], possible_scores, known_cells)
            for k in to_choose
        ]
        total = np.sum(weights)
        idxs = np.random.choice(
            list(range(len(to_choose))),
            size=size,
            p=[w / total for w in weights]
        )
        return [to_choose[i] for i in idxs]

    def __repr__(self):
        return f'weight-seen-{self.seen}-chosen-{self.chosen}-chosen-since-new-{self.chosen_since_new_weight}-action-{self.action}-room-{self.room_cells}-dir-{self.dir_weights}'
