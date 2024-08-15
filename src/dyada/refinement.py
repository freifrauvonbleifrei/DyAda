import bitarray as ba
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import numpy as np
import numpy.typing as npt
from queue import PriorityQueue
from typing import Generator, Union

from dyada.coordinates import (
    get_coordinates_from_level_index,
    LevelIndex,
    Coordinate,
)

from dyada.descriptor import (
    Branch,
    RefinementDescriptor,
    get_level_from_branch,
    get_regular_refined,
)
from dyada.linearization import Linearization


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
        current_branch = Branch(self._descriptor.get_num_dimensions())
        level_index = self.get_level_index_from_branch(current_branch)
        first_patch_bounds = get_coordinates_from_level_index(level_index)
        if not first_patch_bounds.contains(coordinate):
            raise ValueError("Coordinate is not in the domain [0., 1.]^d]")

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
                    if current_refinement == self._descriptor.d_zeros:
                        # found!
                        box_index += 1
                        break
                    else:
                        # go deeper in this branch
                        current_branch.grow_branch(current_refinement)
                else:
                    box_index += self._descriptor.skip_to_next_neighbor(
                        descriptor_iterator, current_refinement
                    )[0]
                    current_branch.advance_branch()

            found_box_indices.append(box_index)
            if np.any(
                current_patch_bounds.upper_bound == coordinate,
                where=current_patch_bounds.upper_bound < 1.0,
            ):
                # if any of the "upper" faces of the patch touches the coordinate,
                # there is still more to find
                current_branch.advance_branch()
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
        assert len(sibling_indices) > 1 and len(sibling_indices).bit_count() == 1
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
            # only continue upwards if not yet at the root
            if linear_index != 0:
                # check if refinement can be moved up the branch (even partially);
                # this requires that all siblings are or would be refined
                siblings = self._discretization.descriptor.get_siblings(linear_index)

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
                self._discretization.descriptor[
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
