# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import logging
import copy
import numpy as np
from collections import defaultdict
from typing import List, Any, Dict
from .data_classes import CellInfoStochastic
from .cell_representations import MontezumaPosLevel
logger = logging.getLogger(__name__)


def number_of_set_bits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


class Selector(object):
    def cell_update(self, cell_key):
        pass

    def choose_cell_key(self, archive: Dict[Any, CellInfoStochastic], size=1):
        raise NotImplementedError('Selectors need to implement a choose_cell_key method')

    def choose_cell(self, archive: Dict[Any, CellInfoStochastic], size=1):
        chosen_keys = self.choose_cell_key(archive, size)
        return [archive[key] for key in chosen_keys]


class RandomSelector(Selector):
    def choose_cell_key(self, archive: Dict[Any, CellInfoStochastic], size=1):
        to_choose = list(archive.keys())
        chosen = np.random.choice(to_choose, size=size)
        return chosen

    def choose_cell(self, archive: Dict[Any, CellInfoStochastic], size=1):
        to_choose = list(archive.values())
        chosen = np.random.choice(to_choose, size=size)
        return chosen


class IterativeSelector(Selector):
    def __init__(self):
        self.i = 0
        self.full_cycles = 0

    def _choose(self, to_choose: List[Any], size):
        chosen = []
        self.selected_all = False
        for _ in range(size):
            if self.i >= len(to_choose):
                self.i = 0
                self.full_cycles += 1
            i = self.i
            self.i += 1
            chosen.append(to_choose[i])
        return chosen

    def choose_cell_key(self, archive: Dict[Any, CellInfoStochastic], size=1):
        to_choose = list(archive.keys())
        return self._choose(to_choose, size)

    def choose_cell(self, archive: Dict[Any, CellInfoStochastic], size=1):
        to_choose = list(archive.values())
        return self._choose(to_choose, size)


def compute_weight(value, weight):
    return weight.weight * 1 / (value + 0.001) ** weight.power + 0.00001


class AbstractWeight:
    def additive_weight(self, cell_key, cell, known_cells, special_attributes):
        return 0

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        return 1

    def cell_update(self, cell_key, is_new, to_update, update_all, archive):
        return to_update, update_all

    def update_weights(self, known_cells, update_all):
        pass


class MaxScoreCell(AbstractWeight):
    def __init__(self):
        self.max_score: float = -float('inf')

    def update_weights(self, archive, update_all):
        max_cell = None
        for cell, info in archive.items():
            if info.score > self.max_score:
                self.max_score = info.score
                max_cell = cell
        logger.debug(f'max score: {self.max_score} cell: {max_cell}')

    def additive_weight(self, cell_key, cell, known_cells, special_attributes):
        assert self.max_score != -float('inf'), 'Max score was not initialized!'
        if cell.score == self.max_score:
            logger.debug(f'max cell found: {self.max_score} cell: {cell_key}')
            return 1
        else:
            return 0

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        assert self.max_score != -float('inf'), 'Max score was not initialized!'
        if cell.score == self.max_score:
            logger.debug(f'max cell found: {self.max_score} cell: {cell_key}')
            return 1
        else:
            return 0

    def __repr__(self):
        return f'MaxScoreCell()'


class TargetCell(AbstractWeight):
    def __init__(self):
        self.desired_attr: Dict[str, int] = {}

    def match(self, cell_key):
        for attribute, value in self.desired_attr.items():
            if getattr(cell_key, attribute) != value:
                return False
        return True

    def additive_weight(self, cell_key, cell, known_cells, special_attributes):
        if self.match(cell_key):
            return 1
        else:
            return 0

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        if self.match(cell_key):
            return 1
        else:
            return 0

    def __repr__(self):
        return f'TargetCell()'


class MaxScoreAndDone(AbstractWeight):
    def __init__(self):
        self.max_score: float = -float('inf')
        self.check_done: bool = True

    def update_weights(self, archive, update_all):
        max_cell = None
        alt_max_score = -float('inf')
        done_cell_exists = False
        self.check_done = True
        for cell, info in archive.items():
            if cell.done:
                done_cell_exists = True
                if info.score > self.max_score:
                    self.max_score = info.score
                    max_cell = cell
            if info.score > alt_max_score:
                alt_max_score = info.score
        if not done_cell_exists:
            self.max_score = alt_max_score
            self.check_done = False
            logger.warning(f'Done cell does not exist in checkpoint, using regular max-score instead.')
        logger.debug(f'max score: {self.max_score} cell: {max_cell}')

    def additive_weight(self, cell_key, cell, known_cells, special_attributes):
        assert self.max_score != -float('inf'), 'Max score was not initialized!'
        if cell.score == self.max_score and (cell_key.done or not self.check_done):
            logger.debug(f'max cell found: {self.max_score} cell: {cell_key}')
            return 1
        else:
            return 0

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        assert self.max_score != -float('inf'), 'Max score was not initialized!'
        if cell.score == self.max_score and (cell_key.done or not self.check_done):
            logger.debug(f'max cell found: {self.max_score} cell: {cell_key}')
            return 1
        else:
            return 0

    def __repr__(self):
        return f'MaxScoreAndDone()'


class ScoreBasedFilter(AbstractWeight):
    def __init__(self):
        self.min_score: float = -float('inf')
        self.roll_threshold: float = 0.5

    def update_weights(self, archive, update_all):
        scores = [info.score for cell, info in archive.items()]
        roll = random.random()
        if roll < self.roll_threshold:
            self.min_score = -float('inf')
        else:
            # More advanced versions could skew the distribution to select higher scoring cells with higher probability,
            # but the uniform random method is cheaper, and may work about as well.
            self.min_score = random.choice(scores)

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        if cell.score < self.min_score:
            return 0
        else:
            return 1

    def __repr__(self):
        return f'ScoreBasedFilter()'


class MaxScoreOnly(AbstractWeight):
    """
    When there are multiple cells that only differ in their score, only select cells that have the highest score.
    """
    def __init__(self, attr: str):
        self.max_score_dict = {}
        self.aggregated_cell_info = {}
        self.attr = attr

    def __repr__(self):
        return f'MaxScoreOnly(attr={self.attr})'

    def _get_no_score_key(self, key):
        no_score_key = copy.copy(key)
        no_score_key.score = 0
        return no_score_key

    def cell_update(self, cell_key, is_new, to_update, update_all, archive):
        assert hasattr(cell_key, 'score')
        score = cell_key.score
        no_score_key = self._get_no_score_key(cell_key)
        if no_score_key not in self.max_score_dict:
            self.max_score_dict[no_score_key] = score
        elif score > self.max_score_dict[no_score_key]:
            self.max_score_dict[no_score_key] = score
        update_all = True
        return to_update, update_all

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        assert hasattr(cell_key, 'score')
        no_score_key = self._get_no_score_key(cell_key)
        if cell_key.score < self.max_score_dict[no_score_key]:
            return 0
        else:
            return 1

    def update_weights(self, known_cells, update_all):
        self.aggregated_cell_info = {}
        for cell in known_cells:
            no_score_key = self._get_no_score_key(cell)
            if no_score_key not in self.aggregated_cell_info:
                self.aggregated_cell_info[no_score_key] = known_cells[cell].__class__()
            self.aggregated_cell_info[no_score_key].add(known_cells[cell])

    # noinspection PyUnusedLocal
    def calculate_value(self, cell_key, cell, known_cells):
        no_score_key = self._get_no_score_key(cell_key)
        attr_value = getattr(self.aggregated_cell_info[no_score_key], self.attr)
        return attr_value

    @staticmethod
    def get_name():
        return 'aggregate'


class MaxScoreReset(AbstractWeight):
    """
    When there are multiple cells that only differ in their score, only select cells that have the highest score.
    In addition, do not aggregate the information from the other cells, calculate the probability of the highest scoring
    cell as if it was a brand-new cell.
    """
    def __init__(self):
        self.max_score_dict = {}

    def _get_no_score_key(self, key):
        no_score_key = copy.copy(key)
        no_score_key.score = 0
        return no_score_key

    def cell_update(self, cell_key, is_new, to_update, update_all, archive):
        assert hasattr(cell_key, 'score')
        score = cell_key.score
        no_score_key = self._get_no_score_key(cell_key)
        if no_score_key not in self.max_score_dict:
            self.max_score_dict[no_score_key] = score
        elif score > self.max_score_dict[no_score_key]:
            self.max_score_dict[no_score_key] = score
        update_all = True
        return to_update, update_all

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        assert hasattr(cell_key, 'score')
        no_score_key = self._get_no_score_key(cell_key)
        if cell_key.score < self.max_score_dict[no_score_key]:
            return 0
        else:
            return 1

    def __repr__(self):
        return f'MaxScoreReset()'


class MaxScoreOnlyNoScore(AbstractWeight):
    """
    When there are multiple cells that only differ in their score, only select cells that have the highest score.
    """
    def __init__(self, attr: str):
        self.attr = attr

    # noinspection PyUnusedLocal
    def calculate_value(self, cell_key, cell, known_cells):
        assert not hasattr(cell_key, 'score')
        attr_value = getattr(cell, self.attr)
        return attr_value

    @staticmethod
    def get_name():
        return 'aggregate'

    def __repr__(self):
        return f'MaxScoreOnlyNoScore(attr={self.attr})'


class NeighborWeights(AbstractWeight):
    def __init__(self, game, horiz, vert, score_low, score_high):
        self.game = game
        self.horiz: float = horiz
        self.vert: float = vert
        self.score_low: float = score_low
        self.score_high: float = score_high
        self.xrange = (10000000, -10000000)
        self.yrange = (10000000, -10000000)
        self.cached_pos_weights = {}
        #: A dictionary keeping track of the "possible scores". In most cases, the possible scores will be items held,
        #: meaning this dictionary keeps track of how many cells there exist carrying various items.
        #: Without domain knowledge, this would do the same thing with the actual score.
        self.possible_scores = defaultdict(int)
        #: This is a dictionary from cell key to the "score" of the cell,
        #: where score can be the objects held if domain knowledge is used,
        #: but it will be the actual game score otherwise.
        self.cell_score: Dict[Any, int] = {}

        self.sorted_scores = None
        self.count_object_bits = False

    def __repr__(self):
        return f'NeighborWeights(horiz={self.horiz}, vert={self.vert}, score_low={self.score_low}, ' \
               f'score_high={self.score_high})'

    def get_neighbor(self, pos: MontezumaPosLevel, offset):
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
        res = self.game.make_pos(pos.objects, new_pos)
        return res

    def no_neighbor(self, pos: MontezumaPosLevel, offset, known_cells):
        return self.get_neighbor(pos, offset) not in known_cells

    def convert_score(self, e):
        if not self.count_object_bits:
            return e
        if isinstance(e, tuple):
            return len(e)
        return number_of_set_bits(e)

    def additive_weight(self,
                        pos: MontezumaPosLevel,
                        cell: CellInfoStochastic,
                        known_cells: Dict[MontezumaPosLevel, CellInfoStochastic],
                        special_attributes):
        possible_scores = self.sorted_scores

        if pos not in self.cached_pos_weights:
            no_low = True
            if self.convert_score(pos.objects) == self.convert_score(possible_scores[0]):
                pass
            else:
                for score in possible_scores:
                    if self.convert_score(score) >= self.convert_score(pos.objects):
                        break
                    if self.game.make_pos(score, pos) in known_cells:
                        no_low = False
                        break

            no_high = True
            if self.convert_score(pos.objects) == self.convert_score(possible_scores[-1]):
                pass
            else:
                for score in reversed(possible_scores):
                    if self.convert_score(score) <= self.convert_score(pos.objects):
                        break
                    if self.game.make_pos(score, pos) in known_cells:
                        no_high = False
                        break

            neigh_horiz = 0.0
            if self.horiz:
                neigh_horiz = (self.no_neighbor(pos, (-1, 0), known_cells) + self.no_neighbor(pos, (1, 0), known_cells))
            neigh_vert = 0.0
            if self.vert:
                neigh_vert = (self.no_neighbor(pos, (0, -1), known_cells) + self.no_neighbor(pos, (0, 1), known_cells))

            res = (self.horiz * neigh_horiz +
                   self.vert * neigh_vert +
                   self.score_low * no_low +
                   self.score_high * no_high)
            self.cached_pos_weights[pos] = res
        return self.cached_pos_weights[pos]

    def cell_update(self, cell_key, is_new, to_update, update_all, archive):
        if is_new:
            self.cell_score[cell_key] = cell_key.objects

            self.possible_scores[cell_key.objects] += 1

            # Currently, this assumes that every cell_key that is not a tuple has an x, y, level, room, and score
            # attribute
            if cell_key.x < self.xrange[0] or cell_key.x > self.xrange[1]:
                self.xrange = (min(cell_key.x, self.xrange[0]), max(cell_key.x, self.xrange[1]))
                update_all = True
            if cell_key.y < self.yrange[0] or cell_key.y > self.yrange[1]:
                self.yrange = (min(cell_key.y, self.yrange[0]), max(cell_key.y, self.yrange[1]))
                update_all = True

            if update_all:
                for score in self.possible_scores:
                    possible_neighbor = self.game.make_pos(score, cell_key)
                    if possible_neighbor in self.cached_pos_weights:
                        del self.cached_pos_weights[possible_neighbor]
                    if possible_neighbor in archive:
                        to_update.add(possible_neighbor)

                for offset in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    possible_neighbor = self.get_neighbor(cell_key, offset)
                    if possible_neighbor in self.cached_pos_weights:
                        del self.cached_pos_weights[possible_neighbor]
                    if possible_neighbor in archive:
                        to_update.add(possible_neighbor)
        else:
            score = cell_key.objects
            old_score = self.cell_score[cell_key]
            self.possible_scores[score] += 1
            self.possible_scores[old_score] -= 1
            self.cell_score[cell_key] = score
            if self.possible_scores[old_score] == 0:
                del self.possible_scores[old_score]

        return to_update, update_all

    def update_weights(self, known_cells, update_all):
        if len(known_cells) == 0:
            return

        if update_all:
            self.cached_pos_weights = {}

        self.sorted_scores = sorted(self.possible_scores, key=self.convert_score)


class LevelWeights(AbstractWeight):
    def __init__(self, low_level_weight=0.0):
        self.max_level = -10000000
        self.low_level_weight: float = low_level_weight

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        level_weight = 1.0
        if cell_key.level < self.max_level:
            level_weight = self.low_level_weight ** (self.max_level - cell_key.level)
        return level_weight

    def cell_update(self, cell_key, is_new, to_update, update_all, archive):
        if is_new:
            if cell_key.level > self.max_level:
                self.max_level = cell_key.level
                update_all = True
        return to_update, update_all

    def update_weights(self, known_cells, update_all):
        pass

    def __repr__(self):
        return f'LevelWeights(low_level_weight={self.low_level_weight})'


class AttrWeight(AbstractWeight):
    def __init__(self, attr, weight, power, scalar):
        self.attr: str = attr
        self.weight: float = weight
        self.power: float = power
        self.scalar: float = scalar

    def additive_weight(self, cell_key, cell, known_cells, special_attributes):
        if self.attr in special_attributes[cell_key]:
            value = special_attributes[cell_key][self.attr]
        else:
            value = getattr(cell, self.attr)
        return self.weight * ((1 / (1 + self.scalar*value)) ** self.power)

    def __repr__(self):
        return f'AttrWeight(attr={self.attr}, weight={self.weight}, power={self.power}, scalar={self.scalar})'


class MultWeight(AbstractWeight):
    def __init__(self, attr, scalar, power):
        self.attr: str = attr
        self.scalar: float = scalar
        self.power: float = power

    def multiplicative_weight(self, cell_key, cell, known_cells, special_attributes):
        if self.attr in special_attributes[cell_key]:
            value = special_attributes[cell_key][self.attr]
        else:
            value = getattr(cell, self.attr)
        return (1 / (1 + self.scalar*value)) ** self.power

    def __repr__(self):
        return f'SubGoalFailWeight(attr={self.attr}, scalar={self.scalar}, power={self.power})'


class SubGoalFailWeight(AbstractWeight):
    def additive_weight(self, cell_key, cell, known_cells, special_attributes):
        fail_dif = max(cell.nb_sub_goal_failed - (cell.nb_chosen + cell.nb_seen), 0)
        w = min(fail_dif, 100) / 100.0
        return w

    def __repr__(self):
        return f'SubGoalFailWeight()'


# Special attributes

class SpecialAttribute(object):
    def calculate_value(self, cell_key, cell, known_cells):
        raise NotImplementedError('Special attributes need to implement value calculation')

    @staticmethod
    def get_name():
        raise NotImplementedError('Special attributes need to have a name')


class WeightedSumAttribute(SpecialAttribute):
    def __init__(self, attr1: str, weight1: str, attr2: str, weight2: str):
        self.attr1 = attr1
        self.weight1 = float(weight1)
        self.attr2 = attr2
        self.weight2 = float(weight2)

    def calculate_value(self, cell_key, cell, known_cells):
        attr1_value = getattr(cell, self.attr1)
        attr2_value = getattr(cell, self.attr2)
        return self.weight1 * attr1_value + self.weight2 * attr2_value

    @staticmethod
    def get_name():
        return 'weighted_sum'


class SubGoalFailAttribute(SpecialAttribute):
    def calculate_value(self, cell_key, cell, known_cells):
        return max(cell.nb_sub_goal_failed - cell.nb_reached, 0)

    @staticmethod
    def get_name():
        return 'sub_goal_fail'


class WeightedSelector(Selector):
    def __init__(self, selector_weights: List[AbstractWeight],
                 special_attributes: List[SpecialAttribute],
                 base_weight: float,
                 weight_based_skew):
        #: List of the weights for each cell
        self.all_weights: List[float] = []
        #: List for the cell keys at the same index of the weights above
        self.cells: List[Any] = []
        #: This is a dictionary from cell key to its index in the cells and all_weights vectors
        self.cell_pos: Dict[Any, int] = {}
        #: This is a dictionary from cell key to any special attributes
        self.special_attribute_dict: Dict[Any, Dict[str, float]] = {}

        self.to_update: set = set()
        self.update_all: bool = False

        self.special_attributes: List[SpecialAttribute] = special_attributes
        self.selector_weights: List[AbstractWeight] = selector_weights
        self.base_weight = base_weight

        self.weight_based_skew = weight_based_skew

    def cell_update(self, cell_key):
        """
        Informs the selector that some information associated with the provided cell-key has changed, and that the cell
        needs to be updated.

        @param cell_key: The cell-key that needs to be updated.
        @return: None.
        """
        is_new = cell_key not in self.cell_pos

        for weight in self.selector_weights:
            self.to_update, self.update_all = weight.cell_update(cell_key,
                                                                 is_new,
                                                                 self.to_update,
                                                                 self.update_all,
                                                                 self.cell_pos)

        if is_new:
            self.cell_pos[cell_key] = len(self.all_weights)
            self.all_weights.append(0.0)
            self.cells.append(cell_key)

        self.to_update.add(cell_key)

    def update_weights(self, known_cells):
        """
        Actually updates the weights of all cells in need of being updated.

        @param known_cells: The archive containing all necessary cell information.
        @return: None.
        """
        if len(known_cells) == 0:
            return

        if self.weight_based_skew:
            self.update_all = True

        for weight in self.selector_weights:
            weight.update_weights(known_cells, self.update_all)

        if self.update_all:
            to_update = self.cells
        else:
            to_update = self.to_update

        for cell in to_update:
            self.all_weights[self.cell_pos[cell]] = self.get_weight(cell, known_cells[cell], known_cells)

        if self.weight_based_skew:
            weight_sorted_cells = sorted(self.cells, key=lambda x: known_cells[x].score, reverse=True)
            weight_sum = 0
            new_weights = [0] * len(self.all_weights)
            for i, cell in enumerate(weight_sorted_cells):
                cell_index = self.cell_pos[cell]
                weight_sum += self.all_weights[cell_index]
                for j in range(i+1):
                    j_cell = weight_sorted_cells[j]
                    j_cell_index = self.cell_pos[j_cell]
                    new_weights[j] += (self.all_weights[j_cell_index] / weight_sum) * (0.5 / len(self.all_weights))

            for i, cell in enumerate(weight_sorted_cells):
                cell_index = self.cell_pos[cell]
                new_weights[i] += (self.all_weights[cell_index] / weight_sum) * 0.5
                self.all_weights[cell_index] = new_weights[i]

        self.update_all = False
        self.to_update = set()

    def get_probabilities(self, archive: Dict[Any, CellInfoStochastic]):
        self.update_weights(archive)

        assert len(archive) == len(self.all_weights)
        total = np.sum(self.all_weights)
        probabilities = [w / total for w in self.all_weights]
        return probabilities

    def get_probabilities_dict(self, archive: Dict[Any, CellInfoStochastic]):
        probabilities = self.get_probabilities(archive)
        probabilities_dict = {}
        for cell, prob in zip(self.cells, probabilities):
            probabilities_dict[cell] = prob
        return probabilities_dict

    def get_traj_probabilities_dict(self, archive: Dict[Any, CellInfoStochastic]):
        probabilities = self.get_probabilities(archive)
        traj_probabilities_dict = {}
        for cell, prob in zip(self.cells, probabilities):
            traj_id = archive[cell].cell_traj_id
            if traj_id not in traj_probabilities_dict:
                traj_probabilities_dict[traj_id] = 0
            traj_probabilities_dict[traj_id] += prob
        return traj_probabilities_dict

    def choose_cell_key(self, archive: Dict[Any, CellInfoStochastic], size=1):
        """
        Chooses a cell from the archive.

        Will recalculate the cell weights if there are any cells scheduled to be updated, or if update_all is set.

        @param archive: The archive dictionary from which to select the cell.
        @param size: How many cells to select.
        @return: A list of selected cells.
        """
        if len(self.cells) == 1:
            return [self.cells[0]] * size
        probabilities = self.get_probabilities(archive)
        logger.debug(f'probabilities: {probabilities}')
        selected = np.random.choice(self.cells, size=size, p=probabilities)
        logger.debug(f'selected cell: {selected}')
        return selected

    def get_weight(self, cell_key, cell, known_cells):
        if cell_key not in self.special_attribute_dict:
            self.special_attribute_dict[cell_key] = dict()
        for special_attribute in self.special_attributes:
            value = special_attribute.calculate_value(cell_key, cell, known_cells)
            self.special_attribute_dict[cell_key][special_attribute.get_name()] = value

        weight_f = self.base_weight
        for selector_weight in self.selector_weights:
            weight_f += selector_weight.additive_weight(cell_key, cell, known_cells, self.special_attribute_dict)

        for selector_weight in self.selector_weights:
            weight_f *= selector_weight.multiplicative_weight(cell_key, cell, known_cells, self.special_attribute_dict)

        logger.debug(f'{cell_key} weight: {weight_f} score: {cell.score}')

        return weight_f
