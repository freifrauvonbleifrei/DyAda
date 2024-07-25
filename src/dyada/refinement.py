import bitarray as ba
from collections import deque, Counter
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import operator
from typing import Generator

from dyada.linearization import Linearization
from dyada.coordinates import (
    get_coordinates_from_level_index,
    LevelIndex,
)


# generalized (2^d-ary) ruler function, e.g. https://oeis.org/A115362
def generalized_ruler(num_dimensions: int, level: int) -> np.ndarray:
    assert level >= 0 and level < 256
    current_list = np.array([], dtype=np.uint8)
    current_list = np.array([1], dtype=np.uint8)
    for i in range(0, level):
        current_list = np.tile(current_list, 2**num_dimensions)
        # actually, a reversed version, change the first element
        current_list[0] += 1
    return current_list


def get_level_from_branch(branch: deque) -> np.ndarray:
    num_dimensions = len(branch[0].level_increment)
    found_level = np.array([0] * num_dimensions, dtype=np.int8)
    for level_count in range(1, len(branch)):
        found_level += np.asarray(
            list(branch[level_count].level_increment), dtype=np.int8
        )
    return found_level


class RefinementDescriptor:
    """A RefinementDescriptor holds a bitarray that describes a refinement tree. The bitarray is a depth-first linearized 2^n tree, with the parents having the refined dimensions set to 1 and the leaves containing all 0s."""

    def __init__(self, num_dimensions: int, base_resolution_level=0):
        self._num_dimensions = num_dimensions
        if isinstance(base_resolution_level, int):
            base_resolution_level = [base_resolution_level] * self._num_dimensions
        assert len(base_resolution_level) == self._num_dimensions

        # establish the base resolution level
        self._data = self.get_d_zeros()
        # iterate in reverse from max(level) to 0
        for l in reversed(range(max(base_resolution_level))):
            at_least_l = ba.bitarray([i > l for i in base_resolution_level])
            factor = 2 ** at_least_l.count()
            self._data = at_least_l + self._data * factor

    def __len__(self):
        """
        return the number of refinement descriptions, will be somewhere between get_num_boxes() and 2*get_num_boxes()
        """
        return len(self._data) // self._num_dimensions

    def get_num_dimensions(self):
        return self._num_dimensions

    def get_d_zeros(self):
        return ba.frozenbitarray(self._num_dimensions)

    def get_num_boxes(self):
        # count number of d*(0) bit blocks
        dZeros = self.get_d_zeros()
        # todo check back if there will be such a function in bitarray
        count = sum(
            1
            for i in range(0, len(self._data), self._num_dimensions)
            if self._data[i : i + self._num_dimensions] == dZeros
        )
        return count

    def get_data(self):
        return self._data

    def is_box(self, index: int):
        return self[index] == self.get_d_zeros()

    def __iter__(self):
        for i in range(len(self)):
            yield ba.frozenbitarray(self[i])

    def __getitem__(self, index_or_slice):
        nd = self._num_dimensions
        if isinstance(index_or_slice, slice):
            assert index_or_slice.step == 1 or index_or_slice.step is None
            start = index_or_slice.start
            stop = index_or_slice.stop
            start = 0 if start is None else start
            stop = len(self) if stop is None else stop
            return self.get_data()[start * nd : stop * nd]
        else:  # it should be an index
            index_or_slice = operator.index(index_or_slice)
            return self.get_data()[index_or_slice * nd : (index_or_slice + 1) * nd]

    def is_pow2tree(self):
        """Is this a quadtree / octree / general power-of-2 tree?"""
        c = Counter(self)
        return c.keys() == {
            ba.frozenbitarray(self._num_dimensions),
            ba.frozenbitarray("1" * self._num_dimensions),
        }

    @dataclass
    class LevelCounter:
        level_increment: ba.frozenbitarray
        count_to_go_up: int

    def get_branch(self, index: int, is_box_index: bool = True) -> deque[LevelCounter]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # traverse tree
        # store/stack how many boxes on this level are left to go up again
        current_branch: deque = deque()
        dZeros = self.get_d_zeros()
        current_branch.append(self.LevelCounter(dZeros, 1))
        box_counter = 0
        i = 0
        while is_box_index or i < index:
            current_refinement = self[i]
            if current_refinement == dZeros:
                box_counter += 1
                if is_box_index and box_counter > index:
                    break
                current_branch[-1].count_to_go_up -= 1
                assert current_branch[-1].count_to_go_up >= 0
                while current_branch[-1].count_to_go_up == 0:
                    current_branch.pop()
                    current_branch[-1].count_to_go_up -= 1
                    assert current_branch[-1].count_to_go_up >= 0
            else:
                current_branch.append(
                    self.LevelCounter(
                        current_refinement.copy(), 2 ** current_refinement.count()
                    )
                )
            i += 1
        return current_branch

    def get_level(self, index: int, is_box_index: bool = True) -> npt.NDArray[np.int8]:
        current_branch = self.get_branch(index, is_box_index)
        found_level = get_level_from_branch(current_branch)
        return found_level


def validate_descriptor(descriptor: RefinementDescriptor):
    assert len(descriptor._data) % descriptor._num_dimensions == 0
    branch = descriptor.get_branch(len(descriptor) - 1, False)
    assert len(branch) > 0
    for twig in branch:
        assert twig.count_to_go_up == 1


def get_level_index(
    linearization: Linearization,
    descriptor: RefinementDescriptor,
    index: int,
    is_box_index: bool = True,
) -> LevelIndex:
    num_dimensions = descriptor.get_num_dimensions()
    current_branch = descriptor.get_branch(index, is_box_index)
    found_level = get_level_from_branch(current_branch)

    # once the branch is found, we can infer the vector index from the branch stack
    current_index: np.ndarray = np.array([0] * num_dimensions, dtype=int)
    decreasing_level_difference = found_level.copy()
    history_of_indices: list[int] = []
    history_of_level_increments: list[ba.bitarray] = []
    for level_count in range(1, len(current_branch)):
        current_refinement = current_branch[level_count].level_increment
        linear_index_at_level = (
            2 ** current_refinement.count() - current_branch[level_count].count_to_go_up
        )
        history_of_level_increments.append(current_refinement)
        history_of_indices.append(linear_index_at_level)
        bit_index = linearization.get_binary_position_from_index(
            history_of_indices,
            history_of_level_increments,
        )
        array_index = np.asarray(list(bit_index))
        assert len(array_index) == num_dimensions
        decreasing_level_difference -= np.asarray(
            list(current_refinement), dtype=np.uint8
        )
        current_index += array_index * 2**decreasing_level_difference

    return LevelIndex(found_level, current_index)


class Refinement:
    def __init__(self, linearization: Linearization, descriptor: RefinementDescriptor):
        self._linearization = linearization
        self._descriptor = descriptor

    def get_level_index(self, index: int, is_box_index: bool = True) -> LevelIndex:
        return get_level_index(
            self._linearization,
            self._descriptor,
            index,
            is_box_index,
        )

    def get_all_boxes_level_indices(self) -> Generator:
        for i, _ in enumerate(self._descriptor):
            if self._descriptor.is_box(i):
                yield self.get_level_index(i, False)
