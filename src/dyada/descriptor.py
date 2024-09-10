import bitarray as ba
from collections import deque, Counter
from dataclasses import dataclass
from functools import cached_property
import numpy as np
import numpy.typing as npt
import operator
from reprlib import repr
from typing import Iterator, Optional, Sequence, Union


# generalized (2^d-ary) ruler function, e.g. https://oeis.org/A115362
def generalized_ruler(num_dimensions: int, level: int) -> np.ndarray:
    assert level >= 0 and level < 256
    current_list = np.array([1], dtype=np.uint8)
    for i in range(0, level):
        current_list = np.tile(current_list, 2**num_dimensions)
        # actually, a reversed version, change the first element
        current_list[0] += 1
    return current_list


def get_regular_refined(added_level: Sequence[int]) -> ba.bitarray:
    num_dimensions = len(added_level)
    data = ba.bitarray(num_dimensions)

    # iterate in reverse from max(level) to 0...
    for l in reversed(range(max(added_level))):
        at_least_l = ba.bitarray([1 if i > l else 0 for i in added_level])
        # power of two by bitshift
        factor = 1 << at_least_l.count()
        # ...while duplicating the current data as new children
        data = at_least_l + data * factor

    return data


def get_num_children_from_refinement(refinement: ba.bitarray) -> int:
    num_ones = refinement.count()
    if num_ones == 0:
        return 0
    else:
        return 1 << num_ones


@dataclass
class LevelCounter:
    level_increment: ba.frozenbitarray
    count_to_go_up: int


class Branch(deque[LevelCounter]):
    """A branch points us to a location in a tree: it is a stack (implemented
    as deque) of LevelCounter "twigs" of level increments and a count that goes
    down as we progress in the number of siblings (this count is always
    initialized to the number of children of the node with the respective
    level_increment/refinement).
    Note that the refinement of the considered node itself is not part of
    the branch, there is only the parent refinement.
    """

    def __init__(self, num_dimensions_or_other_branch: Union[int, "Branch"]):
        if isinstance(num_dimensions_or_other_branch, int):
            super().__init__()
            num_dimensions = num_dimensions_or_other_branch
            dZeros = ba.frozenbitarray([0] * num_dimensions)
            self.append(LevelCounter(dZeros, 1))
        else:
            super().__init__(num_dimensions_or_other_branch)

    def advance_branch(self, check_depth: Optional[int] = None) -> None:
        """Advance the branch to the next sibling, in-place"""
        self[-1].count_to_go_up -= 1
        assert self[-1].count_to_go_up >= 0
        while self[-1].count_to_go_up == 0:
            self.pop()
            if len(self) == check_depth:
                raise IndexError
            self[-1].count_to_go_up -= 1
            assert self[-1].count_to_go_up >= 0

    def grow_branch(self, level_increment: ba.frozenbitarray) -> None:
        """Go deeper in the branch hierarchy / add a twig"""
        # power of two by bitshift
        self.append(LevelCounter(level_increment, 1 << level_increment.count()))

    def __repr__(self) -> str:
        contents = ""
        for twig in self:
            contents += f"{twig.level_increment}({twig.count_to_go_up})-"
        return f"Branch({contents})"

    def to_history(self) -> tuple[list[int], list[ba.frozenbitarray]]:
        history_of_indices: list[int] = []
        history_of_level_increments: list[ba.frozenbitarray] = []
        for level_count in range(1, len(self)):
            current_refinement = self[level_count].level_increment
            # power of two by bitshift
            linear_index_at_level = (1 << current_refinement.count()) - self[
                level_count
            ].count_to_go_up
            history_of_level_increments.append(current_refinement)
            history_of_indices.append(linear_index_at_level)
        return history_of_indices, history_of_level_increments


class RefinementDescriptor:
    """A RefinementDescriptor holds a bitarray that describes a refinement tree. The bitarray is a depth-first linearized 2^n tree, with the parents having the refined dimensions set to 1 and the leaves containing all 0s."""

    def __init__(self, num_dimensions: int, base_resolution_level=0):
        self._num_dimensions = num_dimensions
        if isinstance(base_resolution_level, int):
            base_resolution_level = [base_resolution_level] * self._num_dimensions
        assert len(base_resolution_level) == self._num_dimensions
        _ = self.d_zeros

        # establish the base resolution level
        self._data = get_regular_refined(base_resolution_level)

    @staticmethod
    def from_binary(num_dimensions: int, binary: ba.bitarray) -> "RefinementDescriptor":
        assert len(binary) % num_dimensions == 0
        descriptor = RefinementDescriptor(num_dimensions)
        descriptor._data = binary
        validate_descriptor(descriptor)
        return descriptor

    def __len__(self):
        """
        return the number of refinement descriptions, will be somewhere between get_num_boxes() and 2*get_num_boxes()
        """
        return len(self._data) // self._num_dimensions

    def get_num_dimensions(self):
        return self._num_dimensions

    @cached_property
    def d_zeros(self):
        return ba.frozenbitarray(self._num_dimensions)

    def get_num_boxes(self):
        # count number of d*(0) bit blocks
        dZeros = self.d_zeros
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
        return self[index] == self.d_zeros

    def __repr__(self) -> str:
        return f"RefinementDescriptor({repr(' '.join([b.to01() for b in self]))})"

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
            self.d_zeros,
            ba.frozenbitarray("1" * self._num_dimensions),
        }

    def num_boxes_up_to(self, index: int) -> int:
        count = -1
        for i in self:
            if i == self.d_zeros:
                count += 1
            if index == 0:
                break
            index -= 1
        return count

    def to_box_index(self, index: int) -> int:
        assert self.is_box(index)
        # count zeros up to index, zero-indexed
        return self.num_boxes_up_to(index)

    def to_hierarchical_index(self, box_index: int) -> int:
        linear_index = 0
        # count down box index
        for i in self:
            if i == self.d_zeros:
                box_index -= 1
            if box_index < 0:
                break
            linear_index += 1
        assert self.is_box(linear_index)
        return linear_index

    def get_branch(
        self,
        index: int,
        is_box_index: bool = True,
        hint_previous_branch: tuple[int, Branch] | None = None,
    ) -> tuple[Branch, Iterator]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        if hint_previous_branch is None:
            current_branch = Branch(self._num_dimensions)
            box_counter = 0
            i = 0
            current_iterator = iter(self)
        else:
            i, current_branch = hint_previous_branch
            current_branch = current_branch.copy()
            box_counter = self.num_boxes_up_to(i)
            current_iterator = iter(self)
            for _ in range(i):
                next(current_iterator)
        # traverse tree
        # store/stack how many boxes on this level are left to go up again
        while is_box_index or i < index:
            current_refinement = next(current_iterator)
            if current_refinement == self.d_zeros:
                box_counter += 1
                if is_box_index and box_counter > index:
                    break
                current_branch.advance_branch()
            else:
                current_branch.grow_branch(current_refinement)
            i += 1
        return current_branch, current_iterator

    # now a collection of branch-based family-finding functions
    def get_parent(self, child_branch: Branch) -> tuple[int, Iterator]:
        """Find the parent index of a lost child node by advancing and
        growing a new branch until it matches the child's branch sufficiently.

        Args:
            child_branch (Branch): the branch belonging to the child index

        Returns:
            tuple[int, Iterator]: the index of the parent and
                the iterator to one-past-the-parent
        """
        current_iterator = iter(self)
        if len(child_branch) < 2:
            return -1, current_iterator
        # have to start from beginning, note when we go down matching twigs
        current_branch = Branch(self._num_dimensions)
        twig_index = 1
        for i in range(len(self)):
            current_refinement = next(current_iterator)
            if current_refinement == self.d_zeros:
                current_branch.advance_branch()
            else:
                current_branch.grow_branch(current_refinement)
            # if this matches the child at the twig index, count up the twig index
            if (
                twig_index < len(child_branch) - 1
                and current_branch[twig_index] == child_branch[twig_index]
            ):
                twig_index += 1
            if twig_index == len(child_branch) - 1 and len(current_branch) == len(
                child_branch
            ):
                # technically the iterator and branch will be too far advanced now,
                # if we needed to return the branch we could prune the last entry
                break
        return i, current_iterator

    def get_ancestry(self, child_branch: Branch) -> list[int]:
        """like get_parent, but returns a sorted list of all ancestors"""
        if len(child_branch) < 2:
            return list()
        # have to start from beginning, note when we go down matching twigs
        ancestry: set[int] = set()
        current_branch = Branch(self._num_dimensions)
        twig_index = 0
        for i, current_refinement in enumerate(self):
            # if this matches the child at the twig index, count up the twig index
            while (
                twig_index < len(current_branch)
                and current_branch[twig_index] == child_branch[twig_index]
            ):
                twig_index += 1
                ancestry.add(i)
                if twig_index == len(child_branch) - 1:
                    return sorted(ancestry)
            if current_refinement == self.d_zeros:
                current_branch.advance_branch()
            else:
                current_branch.grow_branch(current_refinement)
        raise IndexError("Child branch not found in descriptor")

    def get_oldest_sibling(self, younger_branch: Branch) -> tuple[int, Iterator]:
        parent_index, parent_iterator = self.get_parent(younger_branch)
        return parent_index + 1, parent_iterator

    def get_siblings(
        self, hierarchical_index: int, branch_to_index: Optional[Branch] = None
    ) -> list[int]:
        siblings: set[int] = {hierarchical_index}
        if branch_to_index is None:
            branch, descriptor_iterator = self.get_branch(hierarchical_index, False)
        else:
            branch = branch_to_index.copy()
            descriptor_iterator = iter(self)
            for _ in range(hierarchical_index):
                next(descriptor_iterator)
        if len(branch) < 2:
            # we are at the root
            return list(siblings)
        total_num_siblings = 1 << branch[-1].level_increment.count()
        num_older_siblings = total_num_siblings - branch[-1].count_to_go_up
        if num_older_siblings > 0:
            assert not branch_to_index
            running_index, descriptor_iterator = self.get_oldest_sibling(branch)
            siblings.add(running_index)
        else:
            running_index = hierarchical_index

        for _ in range(total_num_siblings - 1):
            current_refinement = next(descriptor_iterator)
            assert current_refinement == self[running_index]
            _, num_patches_skipped = self.skip_to_next_neighbor(
                descriptor_iterator, current_refinement
            )
            running_index += num_patches_skipped
            siblings.add(running_index)

        assert len(siblings) == total_num_siblings
        return sorted(list(siblings))

    def get_children(
        self, parent_index: int, branch_to_parent: Optional[Branch] = None
    ):
        if self.is_box(parent_index):
            return []
        first_child_index = parent_index + 1
        if branch_to_parent is not None:
            branch_to_first_child = branch_to_parent.copy()
            branch_to_first_child.grow_branch(self[parent_index])
        else:
            branch_to_first_child = None
        return self.get_siblings(first_child_index, branch_to_first_child)

    def get_level(self, index: int, is_box_index: bool = True) -> npt.NDArray[np.int8]:
        current_branch, _ = self.get_branch(index, is_box_index)
        found_level = get_level_from_branch(current_branch)
        return found_level

    def skip_to_next_neighbor(
        self, descriptor_iterator: Iterator, current_refinement: ba.frozenbitarray
    ) -> tuple[int, int]:
        """Advances the iterator until it points to the end of the current patch and all its children,
        returns the number of boxes or patches it skipped -- including the current one.
        """
        # sweep to the next patch on the same level
        # = count ones and balance them against found boxes
        added_box_index = 0
        added_hierarchical_index = 1
        boxes_to_close = get_num_children_from_refinement(current_refinement)
        if boxes_to_close == 0:
            added_box_index = 1
        else:
            while boxes_to_close > 0:
                next_refinement = next(descriptor_iterator)
                added_hierarchical_index += 1
                boxes_to_close -= 1
                if next_refinement.count() == 0:
                    added_box_index += 1
                else:
                    boxes_to_close += 1 << next_refinement.count()

        return added_box_index, added_hierarchical_index


def branch_generator(descriptor: RefinementDescriptor):
    current_branch = Branch(descriptor.get_num_dimensions())
    i = 0
    for refinement in descriptor:
        yield current_branch, refinement

        i += 1
        if i == len(descriptor):
            return
        if refinement == descriptor.d_zeros:
            current_branch.advance_branch()
        else:
            current_branch.grow_branch(refinement)


def validate_descriptor(descriptor: RefinementDescriptor):
    assert len(descriptor._data) % descriptor._num_dimensions == 0
    branch, _ = descriptor.get_branch(len(descriptor) - 1, False)
    assert len(branch) > 0
    for twig in branch:
        assert twig.count_to_go_up == 1
    return True


def get_level_from_branch(branch: Branch) -> np.ndarray:
    num_dimensions = len(branch[0].level_increment)
    found_level = np.array([0] * num_dimensions, dtype=np.int8)
    for level_count in range(1, len(branch)):
        found_level += np.fromiter(
            branch[level_count].level_increment, dtype=np.int8, count=num_dimensions
        )
    return found_level
