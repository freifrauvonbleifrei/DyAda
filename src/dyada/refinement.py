import bitarray as ba
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import operator
from queue import PriorityQueue
from typing import Generator, Iterator, Sequence, Union

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


def get_regular_refined(added_level: Sequence[int]) -> ba.bitarray:
    num_dimensions = len(added_level)
    data = ba.bitarray(num_dimensions)

    # iterate in reverse from max(level) to 0...
    for l in reversed(range(max(added_level))):
        at_least_l = ba.bitarray([i > l for i in added_level])
        # power of two by bitshift
        factor = 1 << at_least_l.count()
        # ...while duplicating the current data as new children
        data = at_least_l + data * factor

    return data


class RefinementDescriptor:
    """A RefinementDescriptor holds a bitarray that describes a refinement tree. The bitarray is a depth-first linearized 2^n tree, with the parents having the refined dimensions set to 1 and the leaves containing all 0s."""

    def __init__(self, num_dimensions: int, base_resolution_level=0):
        self._num_dimensions = num_dimensions
        if isinstance(base_resolution_level, int):
            base_resolution_level = [base_resolution_level] * self._num_dimensions
        assert len(base_resolution_level) == self._num_dimensions

        # establish the base resolution level
        self._data = get_regular_refined(base_resolution_level)

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

    Branch = deque[LevelCounter]

    def get_branch(
        self, index: int, is_box_index: bool = True
    ) -> tuple[Branch, Iterator]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        # traverse tree
        # store/stack how many boxes on this level are left to go up again
        current_branch: Branch = get_empty_branch(self._num_dimensions)
        dZeros = self.get_d_zeros()
        box_counter = 0
        i = 0
        current_iterator = iter(self)
        while is_box_index or i < index:
            current_refinement = next(current_iterator)
            if current_refinement == dZeros:
                box_counter += 1
                if is_box_index and box_counter > index:
                    break
                advance_branch(current_branch)
            else:
                grow_branch(current_branch, current_refinement)
            i += 1
        return current_branch, current_iterator

    def get_level(self, index: int, is_box_index: bool = True) -> npt.NDArray[np.int8]:
        current_branch, _ = self.get_branch(index, is_box_index)
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

    def to_hierarchical_index(self, box_index: int) -> int:
        linear_index = 0
        # count down box index
        for i in self:
            if i == self.get_d_zeros():
                box_index -= 1
            if box_index < 0:
                break
            linear_index += 1
        assert self.is_box(linear_index)
        return linear_index

    def get_siblings(self, hierarchical_index: int) -> list[int]:
        siblings: list[int] = []
        branch, branch_iterator = self.get_branch(hierarchical_index, False)
        if len(branch) < 2:
            # we are at the root
            return siblings
        # assumes we get called on the first index of a sibling group
        total_num_siblings = 1 << branch[-1].level_increment.count()
        if branch[-1].count_to_go_up != total_num_siblings:
            raise ValueError(
                "This is not the first sibling: " + str(hierarchical_index)
            )

        running_index = hierarchical_index
        for i in range(total_num_siblings - 1):
            next(branch_iterator)
            _, added_hierarchical_index = self.skip_to_next_neighbor(
                branch_iterator, self[hierarchical_index]
            )
            running_index += added_hierarchical_index + 1
            siblings.append(running_index)

        return siblings

    def get_children(self, parent_index: int) -> list[int]:
        first_child_index = parent_index + 1
        child_indices = [first_child_index] + self.get_siblings(first_child_index)
        return child_indices

    def skip_to_next_neighbor(
        self, descriptor_iterator: Iterator, current_refinement: ba.frozenbitarray
    ) -> tuple[int, int]:
        """Advances the iterator until it points to the end of the current patch and all its children,
        returns the number of boxes (plus one) or patches it skipped."""
        # sweep to the next patch on the same level
        # = count ones and balance them against found boxes
        added_box_index = 0
        added_hierarchical_index = 0
        dZeros = self.get_d_zeros()
        if current_refinement != dZeros:
            # power of two by bitshift
            sub_count_boxes_to_close = 1 << current_refinement.count()
            while sub_count_boxes_to_close > 0:
                # this fast-forwards the descriptor iterator
                current_refinement = next(descriptor_iterator)
                added_hierarchical_index += 1
                sub_count_boxes_to_close -= 1
                if current_refinement != dZeros:
                    # power of two by bitshift
                    sub_count_boxes_to_close += 1 << current_refinement.count()
                else:
                    added_box_index += 1
            assert sub_count_boxes_to_close == 0
        else:
            added_box_index += 1
        return added_box_index, added_hierarchical_index


def validate_descriptor(descriptor: RefinementDescriptor):
    assert len(descriptor._data) % descriptor._num_dimensions == 0
    branch, _ = descriptor.get_branch(len(descriptor) - 1, False)
    assert len(branch) > 0
    for twig in branch:
        assert twig.count_to_go_up == 1
    return True


Branch = RefinementDescriptor.Branch


def get_empty_branch(num_dimensions: int) -> Branch:
    dZeros = ba.frozenbitarray([0] * num_dimensions)
    current_branch: Branch = deque()
    current_branch.append(RefinementDescriptor.LevelCounter(dZeros, 1))
    return current_branch


def grow_branch(branch: Branch, level_increment: ba.frozenbitarray) -> None:
    # power of two by bitshift
    branch.append(
        RefinementDescriptor.LevelCounter(level_increment, 1 << level_increment.count())
    )


def get_level_from_branch(branch: Branch) -> np.ndarray:
    num_dimensions = len(branch[0].level_increment)
    found_level = np.array([0] * num_dimensions, dtype=np.int8)
    for level_count in range(1, len(branch)):
        found_level += np.fromiter(
            branch[level_count].level_increment, dtype=np.int8, count=num_dimensions
        )
    return found_level


def advance_branch(branch: Branch) -> None:
    """Advance the branch to the next sibling, in-place"""
    branch[-1].count_to_go_up -= 1
    assert branch[-1].count_to_go_up >= 0
    while branch[-1].count_to_go_up == 0:
        branch.pop()
        branch[-1].count_to_go_up -= 1
        assert branch[-1].count_to_go_up >= 0


def get_level_index_from_branch(
    linearization: Linearization, branch: Branch
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
        # power of two by bitshift
        linear_index_at_level = (1 << current_refinement.count()) - branch[
            level_count
        ].count_to_go_up
        history_of_level_increments.append(current_refinement)
        history_of_indices.append(linear_index_at_level)
        bit_index = linearization.get_binary_position_from_index(
            history_of_indices,
            history_of_level_increments,
        )
        array_index = np.fromiter(bit_index, dtype=np.int8, count=num_dimensions)
        assert len(array_index) == num_dimensions
        decreasing_level_difference -= np.fromiter(
            current_refinement, dtype=np.uint8, count=num_dimensions
        )
        # power of two by bitshift
        current_index += array_index * 1 << decreasing_level_difference

    return LevelIndex(found_level, current_index)


def get_level_index_from_linear_index(
    linearization: Linearization,
    descriptor: RefinementDescriptor,
    linear_index: int,
    is_box_index: bool = True,
) -> LevelIndex:
    current_branch, _ = descriptor.get_branch(linear_index, is_box_index)
    return get_level_index_from_branch(linearization, current_branch)


class Discretization:
    def __init__(self, linearization: Linearization, descriptor: RefinementDescriptor):
        self._linearization = linearization
        self._descriptor = descriptor

    @property
    def descriptor(self):
        return self._descriptor

    def get_level_index_from_branch(self, branch: Branch) -> LevelIndex:
        return get_level_index_from_branch(self._linearization, branch)

    def get_level_index(self, index: int, is_box_index: bool = True) -> LevelIndex:
        return get_level_index_from_linear_index(
            self._linearization, self._descriptor, index, is_box_index
        )

    def get_all_boxes_level_indices(self) -> Generator:
        for i, _ in enumerate(self._descriptor):
            if self._descriptor.is_box(i):
                yield self.get_level_index(i, False)

    def get_containing_box(self, coordinate: Coordinate) -> Union[int, tuple[int, ...]]:
        # traverse the tree
        # start at the root, coordinate has to be in the patch
        current_branch: Branch = get_empty_branch(self._descriptor.get_num_dimensions())
        level_index = self.get_level_index_from_branch(current_branch)
        first_patch_bounds = get_coordinates_from_level_index(level_index)
        if not first_patch_bounds.contains(coordinate):
            raise ValueError("Coordinate is not in the domain [0., 1.]^d]")

        dZeros = self._descriptor.get_d_zeros()
        found_box_indices = []
        box_index = -1
        descriptor_iterator = iter(self._descriptor)

        while True:
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
                    box_index += self._descriptor.skip_to_next_neighbor(
                        descriptor_iterator, current_refinement
                    )[0]
                    advance_branch(current_branch)

            found_box_indices.append(box_index)
            if np.any(
                current_patch_bounds.upper_bound == coordinate,
                where=current_patch_bounds.upper_bound < 1.0,
            ):
                # if any of the "upper" faces of the patch touches the coordinate,
                # there is still more to find
                advance_branch(current_branch)
            else:
                # all found!
                break

        return (
            tuple(found_box_indices)
            if len(found_box_indices) > 1
            else found_box_indices[0]
        )


class PlannedAdaptiveRefinement:
    def __init__(self, discretization: Discretization):
        self._discretization = discretization
        # initialize planned refinement list and data structures used later
        self._planned_refinements: PriorityQueue = PriorityQueue()

        def get_d_zeros_as_array():
            return np.zeros(
                self._discretization.descriptor.get_num_dimensions(), dtype=np.int8
            )

        self._markers: defaultdict[int, npt.NDArray[np.int8]] = defaultdict(
            get_d_zeros_as_array
        )
        self._upward_queue: PriorityQueue[tuple[int, int]] = PriorityQueue()

    def plan_refinement(self, box_index: int, dimensions_to_refine) -> None:
        dimensions_to_refine = ba.frozenbitarray(dimensions_to_refine)
        # get hierarchical index
        linear_index = self._discretization.descriptor.to_hierarchical_index(box_index)
        # store by linear index
        self._planned_refinements.put((linear_index, dimensions_to_refine))

    def populate_queue(self) -> None:
        assert len(self._markers) == 0 and self._upward_queue.empty()

        # put initial markers
        for linear_index, dimensions_to_refine in self._planned_refinements.queue:
            self._markers[linear_index] += np.fromiter(
                dimensions_to_refine,
                dtype=np.int8,
                count=self._discretization.descriptor.get_num_dimensions(),
            )

        # put into the upward queue
        for linear_index in self._markers.keys():
            # obtain level sum to know the priority, highest level should come first
            level_sum = sum(
                self._discretization.descriptor.get_level(linear_index, False)
            )
            self._upward_queue.put((-level_sum, linear_index))

        self._planned_refinements = PriorityQueue()

    def move_marker_to_parent(
        self, marker: npt.NDArray[np.int8], sibling_indices
    ) -> None:
        assert len(sibling_indices) > 1 and sibling_indices.bit_count() == 1
        sibling_indices = sorted(sibling_indices)
        # subtract from the current sibling markers
        for sibling in sibling_indices:
            self._markers[sibling] -= marker
            if np.all(self._markers[sibling] == np.zeros(marker.shape, dtype=np.int8)):
                self._markers.pop(sibling)

        # and add it to the parent's marker
        parent = sibling_indices[0] - 1
        self._markers[parent] += marker

    def move_marker_to_descendants(
        self, ancestor_index, marker: npt.NDArray[np.int8], descendants_indices=None
    ):
        if descendants_indices is None:
            # assume we want the direct children
            descendants_indices = self._discretization.descriptor.get_children(
                ancestor_index
            )
        assert len(descendants_indices) > 1 and descendants_indices.bit_count() == 1
        assert ancestor_index < min(descendants_indices)
        # subtract from the ancestor
        self._markers[ancestor_index] -= marker
        if np.all(
            self._markers[ancestor_index] == np.zeros(marker.shape, dtype=np.int8)
        ):
            self._markers.pop(ancestor_index)

        # and add to the descendants' markers
        for descendant in descendants_indices:
            self._markers[descendant] += marker

    def upwards_sweep(self) -> None:
        num_dimensions = self._discretization.descriptor.get_num_dimensions()
        # traverse the tree from down (high level sums) to the coarser levels
        while not self._upward_queue.empty():
            level_sum, linear_index = self._upward_queue.get()

            # check if refinement can be moved up the branch (even partially);
            # this requires that all siblings are or would be refined
            siblings = [linear_index] + self._discretization.descriptor.get_siblings(
                linear_index
            )

            all_siblings_refinements = [
                np.fromiter(
                    self._discretization.descriptor[sibling],
                    dtype=np.int8,
                    count=num_dimensions,
                )
                for sibling in siblings
            ]

            for i, sibling in enumerate(siblings):
                if sibling in self._markers:
                    all_siblings_refinements[i] += self._markers[sibling]

            # check where the siblings are all refined
            possible_to_move_up = np.min(all_siblings_refinements, axis=0)
            assert possible_to_move_up.shape == (num_dimensions,)
            assert np.all(possible_to_move_up >= 0)

            if np.any(possible_to_move_up > 0):
                self.move_marker_to_parent(possible_to_move_up, siblings)

                # put the parent into the upward queue
                parent_level_sum = level_sum + (
                    len(all_siblings_refinements).bit_length() - 1
                )
                assert parent_level_sum > level_sum and parent_level_sum <= 0
                parent = siblings[0] - 1
                self._upward_queue.put((parent_level_sum, parent))

            # if any siblings were in the upward queue, remove them too
            for sibling in siblings:
                if (level_sum, sibling) in self._upward_queue.queue:
                    self._upward_queue.queue.remove((level_sum, sibling))

            self._upward_queue.task_done()

    def add_refined_data(
        self, new_descriptor: RefinementDescriptor, data_interval: tuple[int, int]
    ):
        minimum_marked = min(self._markers.keys(), default=-1)
        
        if data_interval[0] <= minimum_marked and minimum_marked < data_interval[1]:
            # copy up to marked
            new_descriptor._data.extend(
                self._discretization.descriptor[data_interval[0] : minimum_marked]
            )
            
            # deal with refinement
            if self._discretization.descriptor.is_box(minimum_marked):
                # if the marked item is a box, refine directly
                new_descriptor._data.extend(
                    get_regular_refined(self._markers[minimum_marked])  # type: ignore
                )
                last_processed = minimum_marked

            else:
                raise NotImplementedError("Not yet!")
            self._markers.pop(minimum_marked)
            # call again with remaining interval
            self.add_refined_data(
                new_descriptor, (last_processed + 1, data_interval[1])
            )

        else:
            # copy all and return
            new_descriptor._data.extend(
                self._discretization.descriptor._data[
                    data_interval[0] : data_interval[1]
                ]
            )
            return

    def create_new_descriptor(self) -> RefinementDescriptor:
        new_descriptor = RefinementDescriptor(
            self._discretization.descriptor.get_num_dimensions()
        )
        new_descriptor._data = ba.bitarray()
        self.add_refined_data(new_descriptor, (0, len(self._discretization.descriptor)))

        assert len(new_descriptor._data) >= len(self._discretization.descriptor)
        return new_descriptor

    def apply_refinements(self) -> RefinementDescriptor:
        # todo: magic
        self.populate_queue()
        self.upwards_sweep()

        new_descriptor = self.create_new_descriptor()
        return new_descriptor
        self.apply_refinements.__code__ = (
            lambda: None
        ).__code__  # disable the function after running once
