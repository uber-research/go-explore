
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


from .import_ai import *

# from montezuma_env import *
from goexplore_py.complex_fetch_env import *
from goexplore_py.goexplore import DONE


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
                 chosen_since_new_weight=Weight(), door_weight=0.0, grip_weight=0.0):
        self.seen: Weight = seen
        self.chosen: Weight = chosen
        self.chosen_since_new_weight: Weight = chosen_since_new_weight
        self.room_cells: Weight = room_cells
        self.dir_weights: DirWeights = dir_weights
        self.action: Weight = action
        self.low_level_weight: float = low_level_weight
        self.door_weight = door_weight
        self.grip_weight = grip_weight
        self.game = game

        self.clear_all_cache()

    def clear_all_cache(self):
        self.all_weights = []
        self.to_choose_idxs = []
        self.cells = []
        self.all_weights_nparray = None
        self.cell_pos = {}
        self.cell_score = {}
        self.cached_pos_weights = {}
        self.possible_scores = defaultdict(int)
        self.to_update = set()
        self.update_all = False

        self.xrange = (10000000, -10000000)
        self.yrange = (10000000, -10000000)
        self.max_level = -10000000
        self.known_object_pos = set()

    def get_score(self, cell_key, cell):
        if cell_key == DONE:
            return 0.0
        elif not isinstance(cell_key, tuple):
            return cell_key.score
        else:
            return cell.score

    def cell_update(self, cell_key, cell):
        prev_possible_scores = len(self.possible_scores)
        is_new = cell_key not in self.cell_pos
        if is_new:
            self.cell_pos[cell_key] = len(self.all_weights)
            self.all_weights.append(0.0)
            self.cells.append(cell_key)
            self.to_choose_idxs.append(len(self.to_choose_idxs))
            self.all_weights_nparray = None

            if cell_key != DONE:
                self.cell_score[cell_key] = self.get_score(cell_key, cell)

                self.possible_scores[self.get_score(cell_key, cell)] += 1

                if isinstance(cell_key, FetchState):
                    if cell_key.object_pos not in self.known_object_pos:
                        self.known_object_pos.add(cell_key.object_pos)
                        self.update_all = True
                if not isinstance(cell_key, tuple):
                    if cell_key.x < self.xrange[0] or cell_key.x > self.xrange[1]:
                        self.xrange  = (min(cell_key.x, self.xrange[0]), max(cell_key.x, self.xrange[1]))
                        self.update_all = True
                    if cell_key.y < self.yrange[0] or cell_key.y > self.yrange[1]:
                        self.yrange  = (min(cell_key.y, self.yrange[0]), max(cell_key.y, self.yrange[1]))
                        self.update_all = True
                    if cell_key.level > self.max_level:
                        self.max_level = cell_key.level
                        self.update_all = True

                    if not self.update_all:
                        # print(self.possible_scores)
                        for score in self.possible_scores:
                            possible_neighbor = self.game.make_pos(score, cell_key)
                            if possible_neighbor in self.cached_pos_weights:
                                del self.cached_pos_weights[possible_neighbor]
                            if possible_neighbor in self.cell_score:
                                self.to_update.add(possible_neighbor)

                        for offset in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                            possible_neighbor = self.get_neighbor(cell_key, offset)
                            if possible_neighbor in self.cached_pos_weights:
                                del self.cached_pos_weights[possible_neighbor]
                            if possible_neighbor in self.cell_score:
                                self.to_update.add(possible_neighbor)
        elif cell_key != DONE:
            score = self.get_score(cell_key, cell)
            old_score = self.cell_score[cell_key]
            self.possible_scores[score] += 1
            self.possible_scores[old_score] -= 1
            self.cell_score[cell_key] = score
            if self.possible_scores[old_score] == 0:
                del self.possible_scores[old_score]

        self.to_update.add(cell_key)

        if isinstance(cell_key, tuple) and self.dir_weights.score_high > 0.0000001 and prev_possible_scores != len(self.possible_scores):
            self.update_all = True

    def compute_weight(self, value, weight):
        return weight.weight * 1 / (value + 1) ** weight.power

    def get_seen_weight(self, cell):
        return self.compute_weight(cell.seen_times, self.seen)

    def get_chosen_weight(self, cell):
        return self.compute_weight(cell.chosen_times, self.chosen)

    def get_chosen_since_new_weight(self, cell):
        return self.compute_weight(cell.chosen_since_new, self.chosen_since_new_weight)

    def get_action_weight(self, cell):
        return self.compute_weight(cell.action_times, self.action)

    def get_neighbor(self, pos, offset):
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
        new_pos.room = room
        new_pos.x = x
        new_pos.y = y
        res = self.game.make_pos(pos.score, new_pos)
        return res

    def no_neighbor(self, pos, offset, known_cells):
        return self.get_neighbor(pos, offset) not in known_cells

    def get_pos_weight(self, pos, cell, known_cells, possible_scores):
        if isinstance(pos, FetchState):
            return self.door_weight * sum(pos.door_dists) + self.grip_weight * (pos.gripped_info[1] if pos.gripped_info is not None else 0)
        elif isinstance(pos, tuple):
            # Logic for the score stuff: the highest score will get a weight of 1, second highest a weight of sqrt(1/2), third sqrt(1/3) etc.
            return self.dir_weights.score_high * 1 / np.sqrt(len(possible_scores) - possible_scores.index(cell.score))
        elif pos not in self.cached_pos_weights:
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

            res = self.dir_weights.horiz * neigh_horiz + self.dir_weights.vert * neigh_vert + self.dir_weights.score_low * no_low + self.dir_weights.score_high * no_high
            self.cached_pos_weights[pos] = res
        return self.cached_pos_weights[pos]

    def get_weight(self, cell_key, cell, possible_scores, known_cells):
        if cell_key == DONE:
            return 0.0

        level_weight = 1.0

        if not isinstance(cell_key, tuple) and cell_key.level < self.max_level:
            level_weight = self.low_level_weight ** (self.max_level - cell_key.level)
        elif isinstance(cell_key, FetchState):
            if False:  # TODO: make this accessible somehow
                level_weight = (1/self.low_level_weight)**sum([(e if isinstance(e, int) else sum(map(int, e))) for e in cell_key.object_pos])
            else:
                def generate_next_levels(obj_pos):
                    next_levels = []
                    if obj_pos[-1] == '0000':
                        for new_elem in ['0001', '0010', '0100']:  # TODO: bring back 1000 if/when you find why it's never reached
                            cur = list(obj_pos)
                            cur[-1] = new_elem
                            next_levels.append(tuple(cur))
                    elif obj_pos[0] == '0000':
                        new_obj_pos = list(obj_pos)
                        for i in range(1, len(new_obj_pos)):
                            if new_obj_pos[i] != '0000':
                                new_obj_pos[i - 1] = new_obj_pos[-1]
                                break
                        next_levels.append(tuple(new_obj_pos))

                    return next_levels

                level_weight = 1.0
        if level_weight == 0.0:
            return 0.0
        res = (self.get_pos_weight(cell_key, cell, known_cells, possible_scores) +
               self.get_seen_weight(cell) +
               self.get_chosen_weight(cell) +
               self.get_action_weight(cell) +
               self.get_chosen_since_new_weight(cell)
               ) * level_weight
        return res

    def update_weights(self, known_cells):
        if len(known_cells) == 0:
            return

        if self.update_all:
            self.cached_pos_weights = {}
            to_update = self.cells
        else:
            to_update = self.to_update

        for example_key in known_cells:
            if example_key is not None:
                break

        possible_scores = sorted(self.possible_scores, key=((lambda x: x) if isinstance(example_key, tuple) else convert_score))
        for cell in to_update:
            idx = self.cell_pos[cell]
            self.all_weights[idx] = self.get_weight(cell, known_cells[cell], possible_scores, known_cells)
            if self.all_weights_nparray is not None:
                self.all_weights_nparray[idx] = self.all_weights[idx]

        self.update_all = False
        self.to_update = set()

        # NB: this is debugging code for checking that the weight computation is correct.
        # if you want it to run, set the probability to something between 0 and 1. Note
        # that this will slow things down (though not by much if you set the probability
        # to a low number like 0.1 or 0.01)
        if random.random() < 0.0 and len(self.cells) > 1:
            tqdm.write('Testing weights')
            old_pos_weights = self.cached_pos_weights
            self.cached_pos_weights = {}
            to_choose = list(known_cells.keys())
            if not isinstance(to_choose[0], tuple):
                assert self.xrange == (min(e.x for e in to_choose), max(e.x for e in to_choose))
                assert self.yrange == (min(e.y for e in to_choose), max(e.y for e in to_choose))
                assert self.max_level == max(e.level for e in to_choose)
            if not isinstance(to_choose[0], tuple):
                possible_scores = sorted(set(e.score for e in to_choose), key=convert_score)
            else:
                possible_scores = sorted(set(e.score for e in known_cells.values()))
            weights = [
                self.get_weight(
                    k, known_cells[k], possible_scores, known_cells)
                for k in to_choose
            ]

            expected = dict(zip(to_choose, weights))
            actual = dict(zip(self.cells, self.all_weights))
            for k in expected:
                EPS = 0.0000000001
                if not (expected[k] - EPS < actual[k] < expected[k] + EPS):
                    tqdm.write(f'Weight for {k}: {actual[k]} != {expected[k]}. Cached pos weight: {old_pos_weights.get(k)} vs {self.cached_pos_weights.get(k)}')
                    assert False


    def choose_cell(self, known_cells, size=1):
        self.update_weights(known_cells)
        if len(known_cells) != len(self.all_weights):
            print('ERROR, known_cells has a different number of cells than all_weights')
            print(f'Cell numbers: known_cells {len(known_cells)}, all_weights {len(self.all_weights)}, to_choose_idx {len(self.to_choose_idxs)}, cell_pos {len(self.cell_pos)}')
            for c in known_cells:
                if c not in self.cell_pos:
                    print(f'Tracked but unknown cell: {c}')
            for c in self.cell_pos:
                if c not in known_cells:
                    print(f'Untracked cell: {c}')
            assert False, 'Incorrect length stuff'

        if len(self.cells) == 1:
            return [self.cells[0]] * size
        if self.all_weights_nparray is None:
            self.all_weights_nparray = np.array(self.all_weights)
        weights = self.all_weights_nparray
        to_choose = self.cells
        total = np.sum(weights)
        idxs = np.random.choice(
            self.to_choose_idxs,
            size=size,
            p=weights / total
        )
        # TODO: in extremely rare cases, we do select the DONE cell. Not sure why. We filter it out here but should
        # try to fix the underlying bug eventually.
        return [to_choose[i] for i in idxs if to_choose[i] != DONE]

    def __repr__(self):
        return f'weight-seen-{self.seen}-chosen-{self.chosen}-chosen-since-new-{self.chosen_since_new_weight}-action-{self.action}-room-{self.room_cells}-dir-{self.dir_weights}'
