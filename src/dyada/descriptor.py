import bitarray as ba
from collections import deque, Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from re import findall
import numpy as np
import numpy.typing as npt
import operator
import reprlib
from typing import Iterator, Optional, Sequence, Union


# related to generalized (2^d-ary) ruler function, e.g. https://oeis.org/A115362
def get_regular_refined(added_level: Sequence[int]) -> ba.bitarray:
    num_dimensions = len(added_level)
    data = ba.bitarray(num_dimensions)

    # iterate in reverse from max(level) to 0...
    for level in reversed(range(max(added_level))):
        at_least_l = ba.bitarray([1 if i > level else 0 for i in added_level])
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

    def advance_branch(self, check_depth: int = 0) -> None:
        """Advance the branch to the next sibling, in-place"""
        self[-1].count_to_go_up -= 1
        assert self[-1].count_to_go_up >= 0
        while self[-1].count_to_go_up == 0:
            self.pop()
            if len(self) <= check_depth:
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


class DyadaInvalidDescriptorError(Exception):
    pass


class RefinementDescriptor:
    """
    A RefinementDescriptor holds a bitarray that describes a refinement tree.
    The bitarray is a depth-first linearized 2^n tree, with the parents having
    the refined dimensions set to 1 and the leaves containing all 0s.
    It is the (preorder depth-first) linearized binary representation of omnitrees
    described in https://arxiv.org/abs/2508.06316 .
    """

    def __init__(self, num_dimensions: int, base_resolution_level=0):
        self._num_dimensions = num_dimensions
        if isinstance(base_resolution_level, int):
            base_resolution_level = [base_resolution_level] * self._num_dimensions
        assert len(base_resolution_level) == self._num_dimensions

        # establish the base resolution level
        self._data = get_regular_refined(base_resolution_level)

    def to_file(self, filename: str):
        filename_full = filename
        filename_full += "_{dim}d.bin".format(dim=self.get_num_dimensions())
        with open(filename_full, "wb") as file:
            # invert so that trailing zeroes are easier to remove
            inverted_data = ~self._data
            inverted_data.tofile(file)

    @staticmethod
    def from_file(filename: str):
        # extract dimensionality from filename
        num_dimensions_list = findall(r"_(\d+)d.bin", filename)
        num_dimensions = int(num_dimensions_list[0])
        inverted_data = ba.bitarray()
        with open(filename, "rb") as file:
            inverted_data.fromfile(file)
        # remove trailing zeroes
        num_trailing_zeros = 0
        while inverted_data[-num_trailing_zeros - 1] == 0:
            num_trailing_zeros += 1
        inverted_data = inverted_data[:-num_trailing_zeros]
        return RefinementDescriptor.from_binary(num_dimensions, ~inverted_data)

    @staticmethod
    def from_binary(num_dimensions: int, binary: ba.bitarray) -> "RefinementDescriptor":
        if len(binary) % num_dimensions != 0:
            raise ValueError("Invalid binary input length")
        descriptor = RefinementDescriptor(num_dimensions)
        descriptor._data = binary
        try:
            validate_descriptor(descriptor)
        except DyadaInvalidDescriptorError as e:
            raise ValueError("Invalid binary input") from e
        return descriptor

    def __eq__(self, other):
        return (
            (self._num_dimensions == other._num_dimensions)
            and (len(self) == len(other))
            and (self._data == other._data)
        )

    def __hash__(self):
        return hash((self._num_dimensions, self._data.tobytes()))

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
        return sum(x.count() == 0 for x in self)

    def get_data(self):
        return self._data

    def is_box(self, index: int):
        return self[index] == self.d_zeros

    def __repr__(self) -> str:
        return (
            f"RefinementDescriptor({reprlib.repr(' '.join([b.to01() for b in self]))})"
        )

    def __iter__(self, start: int = 0):
        if __debug__:  # slow and safe mode
            for i in range(start, len(self)):
                # same as yield ba.frozenbitarray(self[i])
                yield ba.frozenbitarray(
                    self.get_data()[
                        i * self._num_dimensions : (i + 1) * self._num_dimensions
                    ]
                )
            return
        # fast mode, won't work with the functions that use Counter directly
        j = self._num_dimensions * start
        for _ in range(start, len(self)):
            next_j = j + self._num_dimensions
            yield self.get_data()[j:next_j]
            j = next_j

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

    def is_pow2tree(self) -> bool:
        """Is this a quadtree / octree / general power-of-2 tree?"""
        c = Counter(self)
        return c.keys() == {
            self.d_zeros,
            ba.frozenbitarray("1" * self._num_dimensions),
        }

    def num_boxes_up_to(self, index: int) -> int:
        # count zeros up to index, zero-indexed
        # (Counter is also possible, but keeps us from performance opt. in __iter__)
        count = sum((x == self.d_zeros) for x in islice(self, index))
        return count

    def to_box_index(self, index: int) -> int:
        assert self.is_box(index)
        # count zeros up to index, zero-indexed
        return self.num_boxes_up_to(index)

    def _to_box_index_recursive(self, index: int) -> int:
        assert self.is_box(index)
        try:
            # iterate down until we find the prior box
            for i in range(index - 1, 0, -1):
                if self[i] == self.d_zeros:
                    return self._to_box_index_recursive(i) + 1
            # if we are at the first box, return 0
            return 0
        except RecursionError:
            return self.num_boxes_up_to(index)

    def get_maximum_level(self) -> npt.NDArray[np.int8]:
        max_level = np.max(list(self.level_iterator()), axis=0)
        return max_level

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

        box_counter = 0
        if hint_previous_branch is None:
            current_branch = Branch(self._num_dimensions)
            i = 0
            current_iterator = iter(self)
        else:
            i, current_branch = hint_previous_branch
            current_branch = current_branch.copy()
            if is_box_index:
                box_counter = self.num_boxes_up_to(i)
            current_iterator = self.__iter__(start=i)  # type: ignore
        # traverse tree
        # store/stack how many boxes on this level are left to go up again
        while is_box_index or i < index:
            current_refinement = next(current_iterator)
            if current_refinement == self.d_zeros:
                if is_box_index:
                    box_counter += 1
                    if box_counter > index:
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
            descriptor_iterator = self.__iter__(start=hierarchical_index)  # type: ignore
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
        return sorted(siblings)

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

    def level_iterator(self):
        for current_branch, _ in branch_generator(self):
            yield get_level_from_branch(current_branch)


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
    if len(descriptor._data) % descriptor._num_dimensions != 0:
        raise DyadaInvalidDescriptorError("Uneven number of bits in descriptor")
    try:
        branch, _ = descriptor.get_branch(len(descriptor) - 1, False)
    except IndexError as e:
        raise DyadaInvalidDescriptorError(
            "Descriptor does not form a valid omnitree"
        ) from e
    if not (len(branch) > 0) or any(twig.count_to_go_up != 1 for twig in branch):
        raise DyadaInvalidDescriptorError("Descriptor not fully traversed")

    return True


def int8_ndarray_from_iterable(iterable) -> npt.NDArray[np.int8]:
    return np.fromiter(iterable, dtype=np.int8, count=len(iterable))


def get_level_from_branch(branch: Branch) -> np.ndarray:
    num_dimensions = len(branch[0].level_increment)
    found_level = np.array([0] * num_dimensions, dtype=np.int8)
    for level_count in range(1, len(branch)):
        found_level += int8_ndarray_from_iterable(branch[level_count].level_increment)
    return found_level


def hierarchical_to_box_index_mapping(
    hierarchical_mapping: list[set[int]],
    key_descriptor: RefinementDescriptor,
    value_descriptor: RefinementDescriptor,
) -> list[set[int]]:
    box_mapping = [
        set(
            value_descriptor.to_box_index(new_index)
            for new_index in new_indices
            if value_descriptor.is_box(new_index)
        )
        for old_index, new_indices in enumerate(hierarchical_mapping)
        if key_descriptor.is_box(old_index)
    ]
    return box_mapping


def find_uniqueness_violations(
    descriptor: RefinementDescriptor,
) -> list[set[int]]:
    """
    Find the tuples of indices where the uniqueness condition is violated.
    This is the case when all children of a box have a refinement (==1) where
    the parent box has no refinement (==0).
    """
    violations: list[set[int]] = []
    children_refinement_stack: list[tuple[int, ba.bitarray]] = []
    # iterate the descriptor from the end to the beginning
    for i in range(len(descriptor) - 1, -1, -1):
        if descriptor[i].count() > 0:
            num_children = 2 ** descriptor[i].count()
            # pop the children from the stack
            children = []
            for _ in range(num_children):
                children.append(children_refinement_stack.pop())

            all_children_have_refinement = children[0][1]
            for child in children[1:]:
                all_children_have_refinement &= child[1]
            if (all_children_have_refinement & ~descriptor[i]).count() > 0:
                # uniqueness condition violated, add to violations
                violations.append({i, *[child[0] for child in children]})

        # put on the stack
        children_refinement_stack.append((i, descriptor[i]))
    return violations
