import bitarray as ba
from collections import deque, Counter
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import operator
from typing import Generator, Optional

from dyada.linearization import Linearization
from dyada.coordinates import (
    get_coordinates_from_level_index,
    LevelIndex,
    Coordinate,
)


# generalized (2^d-ary) ruler function, e.g. https://oeis.org/A115362
def generalized_ruler(num_dimensions: int, level: int) -> np.ndarray:
    assert level >= 0 and level < 256
    current_list = np.array([1], dtype=np.uint8)
    for i in range(0, level):
        current_list = np.tile(current_list, 2**num_dimensions)
        # actually, a reversed version, change the first element
        current_list[0] += 1
    return current_list


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
            start = 0 if start is None else operator.index(start)
            stop = len(self) if stop is None else operator.index(stop)
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
        current_branch: deque = get_empty_branch(self._num_dimensions)
        dZeros = self.get_d_zeros()
        box_counter = 0
        i = 0
        while is_box_index or i < index:
            current_refinement = self[i]
            if current_refinement == dZeros:
                box_counter += 1
                if is_box_index and box_counter > index:
                    break
                advance_branch(current_branch)
            else:
                grow_branch(current_branch, current_refinement)
            i += 1
        return current_branch

    def get_level(self, index: int, is_box_index: bool = True) -> npt.NDArray[np.int8]:
        current_branch = self.get_branch(index, is_box_index)
        found_level = get_level_from_branch(current_branch)
        return found_level

    def to_box_index(self, index: int) -> int:
        assert self.is_box(index)
        # count zeros up to index, zero-indexed
        count = -1
        for i in self:
            if i == self.get_d_zeros():
                count += 1
            if index == 0:
                break
            index -= 1
        return count


def validate_descriptor(descriptor: RefinementDescriptor):
    assert len(descriptor._data) % descriptor._num_dimensions == 0
    branch = descriptor.get_branch(len(descriptor) - 1, False)
    assert len(branch) > 0
    for twig in branch:
        assert twig.count_to_go_up == 1


def get_empty_branch(num_dimensions: int) -> deque:
    dZeros = ba.frozenbitarray([0] * num_dimensions)
    current_branch: deque = deque()
    current_branch.append(RefinementDescriptor.LevelCounter(dZeros, 1))
    return current_branch


def grow_branch(branch: deque, level_increment: ba.frozenbitarray) -> None:
    branch.append(
        RefinementDescriptor.LevelCounter(level_increment, 2 ** level_increment.count())
    )


def get_level_from_branch(branch: deque) -> np.ndarray:
    num_dimensions = len(branch[0].level_increment)
    found_level = np.array([0] * num_dimensions, dtype=np.int8)
    for level_count in range(1, len(branch)):
        found_level += np.asarray(
            list(branch[level_count].level_increment), dtype=np.int8
        )
    return found_level


def advance_branch(branch: deque) -> None:
    """Advance the branch to the next sibling, in-place"""
    branch[-1].count_to_go_up -= 1
    assert branch[-1].count_to_go_up >= 0
    while branch[-1].count_to_go_up == 0:
        branch.pop()
        branch[-1].count_to_go_up -= 1
        assert branch[-1].count_to_go_up >= 0


def get_level_index_from_branch(
    linearization: Linearization, branch: deque
) -> LevelIndex:
    num_dimensions = len(branch[0].level_increment)
    found_level = get_level_from_branch(branch)

    # once the branch is found, we can infer the vector index from the branch stack
    current_index: np.ndarray = np.array([0] * num_dimensions, dtype=int)
    decreasing_level_difference = found_level.copy()
    history_of_indices: list[int] = []
    history_of_level_increments: list[ba.bitarray] = []
    for level_count in range(1, len(branch)):
        current_refinement = branch[level_count].level_increment
        linear_index_at_level = (
            2 ** current_refinement.count() - branch[level_count].count_to_go_up
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


def get_level_index_from_linear_index(
    linearization: Linearization,
    descriptor: RefinementDescriptor,
    linear_index: int,
    is_box_index: bool = True,
) -> LevelIndex:
    current_branch = descriptor.get_branch(linear_index, is_box_index)
    return get_level_index_from_branch(linearization, current_branch)


class Refinement:
    def __init__(self, linearization: Linearization, descriptor: RefinementDescriptor):
        self._linearization = linearization
        self._descriptor = descriptor

    def get_level_index_from_branch(self, branch: deque) -> LevelIndex:
        return get_level_index_from_branch(self._linearization, branch)

    def get_level_index(self, index: int, is_box_index: bool = True) -> LevelIndex:
        return get_level_index_from_linear_index(
            self._linearization, self._descriptor, index, is_box_index
        )

    def get_all_boxes_level_indices(self) -> Generator:
        for i, _ in enumerate(self._descriptor):
            if self._descriptor.is_box(i):
                yield self.get_level_index(i, False)

    def get_containing_box(self, coordinate: Coordinate):
        # traverse the tree
        # start at the root, coordinate has to be in the patch
        current_branch: deque = get_empty_branch(self._descriptor.get_num_dimensions())
        level_index = self.get_level_index_from_branch(current_branch)
        current_patch_bounds = get_coordinates_from_level_index(level_index)
        if not current_patch_bounds.contains(coordinate):
            raise ValueError("Coordinate is not in the domain [0., 1.]^d]")

        dZeros = self._descriptor.get_d_zeros()
        box_index = -1
        descriptor_iterator = iter(self._descriptor)

        while True:
            current_refinement = next(descriptor_iterator)
            level_index = self.get_level_index_from_branch(current_branch)
            current_patch_bounds = get_coordinates_from_level_index(level_index)

            # is the coordinate in this patch?
            if current_patch_bounds.contains(coordinate):
                if current_refinement == dZeros:
                    # found!
                    box_index += 1
                    break
                else:
                    # go deeper in this branch
                    grow_branch(current_branch, current_refinement)
            else:
                # sweep to the next patch on the same level
                # = count ones and balance them against found boxes
                if current_refinement != dZeros:
                    sub_count_boxes_to_close = 2 ** current_refinement.count()
                    while sub_count_boxes_to_close > 0:
                        # this fast-forwards the descriptor iterator
                        current_refinement = next(descriptor_iterator)
                        sub_count_boxes_to_close -= 1
                        if current_refinement != dZeros:
                            sub_count_boxes_to_close += 2 ** current_refinement.count()
                        else:
                            box_index += 1
                    assert sub_count_boxes_to_close == 0
                else:
                    box_index += 1
                advance_branch(current_branch)

        return box_index
