import bitarray as ba
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
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
    get_num_children_from_refinement,
    get_level_from_branch,
    get_regular_refined,
)
from dyada.linearization import (
    binary_position_gen_from_mask,
    interleave_binary_positions,
    Linearization,
)


def get_level_index_from_branch(
    linearization: Linearization, branch: Branch
) -> LevelIndex:
    num_dimensions = len(branch[0].level_increment)
    found_level = get_level_from_branch(branch)

    # once the branch is found, we can infer the vector index from the branch stack
    current_index: np.ndarray = np.array([0] * num_dimensions, dtype=int)
    decreasing_level_difference = found_level.copy()
    history_of_indices, history_of_level_increments = branch.to_history()
    for level_count in range(1, len(branch)):
        bit_index = linearization.get_binary_position_from_index(
            history_of_indices[:level_count],
            history_of_level_increments[:level_count],
        )
        array_index = np.fromiter(bit_index, dtype=np.int8, count=num_dimensions)
        assert len(array_index) == num_dimensions
        decreasing_level_difference -= np.fromiter(
            branch[level_count].level_increment, dtype=np.uint8, count=num_dimensions
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
        if np.all(marker == np.zeros(marker.shape, dtype=np.int8)):
            return
        if descendants_indices is None:
            # assume we want the direct children
            descendants_indices = self._discretization.descriptor.get_children(
                ancestor_index, and_after=False
            )
        assert (
            len(descendants_indices) > 1 and len(descendants_indices).bit_count() == 1
        )
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
                siblings = self._discretization.descriptor.get_siblings(
                    linear_index, and_after=False
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

    @dataclass
    class RefinementCommission:
        lower: int
        upper: int
        refine_from_index: int = -1
        split_dimensions: ba.bitarray = field(default_factory=ba.bitarray)
        split_binary_position: ba.bitarray = field(default_factory=ba.bitarray)

    def get_added_refinement_bits(
        self, current_index: int, current_refinement: ba.bitarray
    ) -> ba.frozenbitarray:
        added_refinement = self._markers[current_index].copy()
        # cap at 1
        added_refinement[added_refinement > 0] = 1
        # set to 0 where the parent is already refined
        added_refinement[
            np.fromiter(current_refinement, dtype=bool, count=len(current_refinement))
        ] = 0
        assert all(added_refinement >= 0) and all(added_refinement <= 1)
        return ba.frozenbitarray(list(added_refinement))

    def refine_parent_and_reorder_children(
        self,
        new_descriptor: RefinementDescriptor,
        parent_index: int,
        parent_current_refinement: ba.frozenbitarray,
        children_intervals: list[tuple[int, int]],
    ) -> list[RefinementCommission]:

        linearization = self._discretization._linearization
        old_descriptor = self._discretization.descriptor

        assert parent_current_refinement.count() > 0
        assert len(children_intervals) == 1 << parent_current_refinement.count()
        num_dimensions = len(parent_current_refinement)
        parent_added_refinement_bits = self.get_added_refinement_bits(
            parent_index, parent_current_refinement
        )
        # any marker that can't be used here is split off and pushed down to children's markers
        remaining_marker = self._markers[parent_index].copy()
        remaining_marker[
            np.fromiter(parent_added_refinement_bits, dtype=bool, count=num_dimensions)
        ] -= 1
        self.move_marker_to_descendants(parent_index, remaining_marker)

        parent_final_refinement = ba.frozenbitarray(
            parent_current_refinement ^ parent_added_refinement_bits
        )
        assert parent_final_refinement.count() > 0
        new_descriptor._data.extend(parent_final_refinement)

        expected_num_children = get_num_children_from_refinement(
            parent_final_refinement
        )
        # data structure to put the reordered children's commissions
        reordered_new_children = {}

        # here's some helper functions w/ captured variables to get the right order of future children
        parent_new_index = len(new_descriptor) - 1
        future_branch, _ = new_descriptor.get_branch(parent_new_index, False)
        (
            future_history_of_indices,
            future_history_of_level_increments,
        ) = future_branch.to_history()
        future_history_of_level_increments.append(parent_final_refinement)

        # to determine the future index of the (grand)children
        def get_index_in_future_patch(binary_position: ba.bitarray) -> int:
            return linearization.get_index_from_binary_position(
                binary_position,
                future_history_of_indices,
                future_history_of_level_increments,
            )

        new_child_factor = 1 << parent_added_refinement_bits.count()

        for child_interval in children_intervals:
            child = child_interval[0]
            child_current_refinement = old_descriptor[child]
            # if any upward refinement bits are newly set, there needs to be some
            # mixed refinement / reordering of the grandchildren
            upward_refinement_bits = (
                ~parent_current_refinement
                & child_current_refinement
                & parent_final_refinement
            )
            child_history = old_descriptor.get_branch(child, False)[0].to_history()
            child_current_binary_position = (
                linearization.get_binary_position_from_index(*child_history)
            )

            def get_future_index(
                inner_refinement: ba.bitarray, inner_position: ba.bitarray
            ) -> int:
                interleaved_binary_position = interleave_binary_positions(
                    parent_current_refinement,
                    child_current_binary_position,
                    inner_refinement,
                    inner_position,
                )
                return get_index_in_future_patch(interleaved_binary_position)

            # get grandchildren's current intervals
            grandchildren_indices, index_after_grandchildren = (
                old_descriptor.get_children(child, and_after=True)
            )
            grandchildren_indices.append(index_after_grandchildren)
            grandchildren = [
                (begin, after_end)
                for begin, after_end in pairwise(grandchildren_indices)
            ]

            if new_child_factor == 1:
                # just keep the child
                # find child's future position
                child_index_in_new_box = get_index_in_future_patch(
                    child_current_binary_position
                )
                reordered_new_children[child_index_in_new_box] = (
                    self.RefinementCommission(*child_interval)
                )

            elif (
                len(grandchildren) == new_child_factor
                and upward_refinement_bits == child_current_refinement
            ):
                # discard the children and adopt the grandchildren
                # get branch of oldest grandchild...
                branch, _ = old_descriptor.get_branch(grandchildren[0][0], False)
                history_of_indices, history_of_level_increments = branch.to_history()

                # to find the respective indices in the inner and outer boxes
                for j, grandchild in enumerate(grandchildren):
                    history_of_indices[-1] = j
                    grandchild_binary_position = (
                        linearization.get_binary_position_from_index(
                            history_of_indices, history_of_level_increments
                        )
                    )

                    # take the child bits where the parent refinement is set
                    # and che grandchild bits where the child refinement is set
                    grandchild_index_in_new_box = get_future_index(
                        child_current_refinement, grandchild_binary_position
                    )

                    reordered_new_children[grandchild_index_in_new_box] = (
                        self.RefinementCommission(*grandchild)
                    )

            elif len(grandchildren) > 0:
                # upward move of child refinement -> split this child into multiple children
                # needs to consider custody of grandchildren in the next recurision

                # find the future binary positions of the split children and add info to commission
                for child_new_binary in binary_position_gen_from_mask(
                    parent_added_refinement_bits
                ):
                    # get new binary position by interleaving the current binary position
                    # with the added refinement
                    child_new_binary_position = interleave_binary_positions(
                        parent_current_refinement,
                        child_current_binary_position,
                        parent_added_refinement_bits,
                        child_new_binary,
                    )

                    child_index_in_new_box = get_index_in_future_patch(
                        child_new_binary_position
                    )

                    reordered_new_children[child_index_in_new_box] = (
                        self.RefinementCommission(
                            *child_interval,
                            parent_index,
                            parent_added_refinement_bits,
                            parent_added_refinement_bits & child_new_binary_position,
                        )
                    )

            else:  # there are currently no grandchildren, but there will be
                # and we already adopt them
                # find grandchildren's future positions
                for grandchild_bin_position in binary_position_gen_from_mask(
                    parent_added_refinement_bits
                ):
                    grandchild_index_in_new_box = get_future_index(
                        parent_added_refinement_bits,
                        grandchild_bin_position,
                    )

                    # the reordered grandchildren need their parent's info,
                    # because they don't yet exist
                    reordered_new_children[grandchild_index_in_new_box] = (
                        self.RefinementCommission(-1, -1, child)
                    )

        assert len(reordered_new_children) == expected_num_children
        return [reordered_new_children[i] for i in range(expected_num_children)]

    def add_refined_data(
        self, new_descriptor: RefinementDescriptor, data_interval: RefinementCommission
    ):
        linearization = self._discretization._linearization
        old_descriptor = self._discretization.descriptor

        if data_interval.lower == -1:  # here, a former leaf/box is growing children
            assert old_descriptor.is_box(data_interval.refine_from_index)
            # refine based on the refine_from_index = former parent
            new_descriptor._data.extend(
                get_regular_refined(self._markers[data_interval.refine_from_index])  # type: ignore
            )

        elif len(data_interval.split_dimensions) == 0:  # "normal" refinement
            # filter the markers to the current interval
            filtered_markers = {
                k: v
                for k, v in self._markers.items()
                if data_interval.lower <= k and k < data_interval.upper
            }
            index_to_refine = min(filtered_markers.keys(), default=-2)
            while (
                index_to_refine != -2
                and sum(np.abs(filtered_markers[index_to_refine])) == 0
            ):  # remove zero markers
                self._markers.pop(index_to_refine)
                filtered_markers.pop(index_to_refine)
                index_to_refine = min(filtered_markers.keys(), default=-2)

            if index_to_refine != -2:
                # if a marker was found in the commission interval
                # copy up to marked
                new_descriptor._data.extend(
                    old_descriptor[data_interval.lower : index_to_refine]
                )

                # deal with refinement
                if old_descriptor.is_box(index_to_refine):
                    # if the marked item is a box, refine directly
                    new_descriptor._data.extend(
                        get_regular_refined(self._markers[index_to_refine])  # type: ignore
                    )
                    last_processed = index_to_refine

                else:
                    # if the marked item is a patch, recursively call on the descendant intervals in the sub-tree
                    # but re-sorted/interleaved according to linearization
                    children, index_after_children = old_descriptor.get_children(
                        index_to_refine, and_after=True
                    )
                    children_and_end = [*children, index_after_children]
                    children_intervals = [
                        (begin, after_end)
                        for begin, after_end in pairwise(children_and_end)
                    ]
                    reordered_new_children = self.refine_parent_and_reorder_children(
                        new_descriptor,
                        index_to_refine,
                        old_descriptor[index_to_refine],
                        children_intervals,
                    )
                    for child_interval in reordered_new_children:
                        # recurse
                        self.add_refined_data(new_descriptor, child_interval)
                    last_processed = max(
                        max(child_interval.upper - 1, child_interval.refine_from_index)
                        for child_interval in reordered_new_children
                    )

                if (last_processed + 1) != data_interval.upper:
                    assert (last_processed + 1) < data_interval.upper
                    # recurse with remaining interval
                    self.add_refined_data(
                        new_descriptor,
                        self.RefinementCommission(
                            last_processed + 1, data_interval.upper
                        ),
                    )

            else:
                # copy all and return
                new_descriptor._data.extend(
                    old_descriptor[data_interval.lower : data_interval.upper]
                )

        else:
            # this is a parent separating itself
            # -> interval is the former parent's full interval
            current_index = data_interval.lower
            current_refinement = old_descriptor[current_index]
            remaining_refinement_bits = (
                current_refinement & ~data_interval.split_dimensions
            )
            children, index_after_children = old_descriptor.get_children(
                current_index, and_after=True
            )
            children_and_end = [*children, index_after_children]
            history_of_indices, history_of_level_increments = old_descriptor.get_branch(
                children[0], False
            )[0].to_history()
            history_of_indices = history_of_indices[:-1]

            # find the current binary positions of the split node's children
            children_in_part_intervals: list[tuple[int, int]] = []
            for child_binary_position in binary_position_gen_from_mask(
                remaining_refinement_bits
            ):
                child_index_in_previous_box = (
                    linearization.get_index_from_binary_position(
                        child_binary_position,
                        history_of_indices,
                        history_of_level_increments,
                    )
                )
                children_in_part_intervals.append(
                    (
                        children[child_index_in_previous_box],
                        children_and_end[child_index_in_previous_box + 1],
                    )
                )

            subtracted_marker = np.fromiter(
                current_refinement ^ remaining_refinement_bits,
                dtype=np.int8,
                count=old_descriptor.get_num_dimensions(),
            )
            # to temporarily denote that the (negative) marker is already applied to the remaining_refinement_bits
            self._markers[current_index] += subtracted_marker
            reordered_new_children = self.refine_parent_and_reorder_children(
                new_descriptor,
                current_index,
                remaining_refinement_bits,
                children_in_part_intervals,
            )
            # needs to be reapplied for the next instance of this split child
            self._markers[current_index] -= subtracted_marker
            # recurse
            for child_interval in reordered_new_children:
                self.add_refined_data(new_descriptor, child_interval)

    def create_new_descriptor(self) -> RefinementDescriptor:
        new_descriptor = RefinementDescriptor(
            self._discretization.descriptor.get_num_dimensions()
        )
        new_descriptor._data = ba.bitarray()
        self.add_refined_data(
            new_descriptor,
            self.RefinementCommission(0, len(self._discretization.descriptor)),
        )

        assert len(new_descriptor._data) >= len(self._discretization.descriptor)
        return new_descriptor

    def apply_refinements(self) -> RefinementDescriptor:
        assert self._upward_queue.empty()
        assert self._markers == {}
        self.populate_queue()
        self.upwards_sweep()

        new_descriptor = self.create_new_descriptor()
        return new_descriptor
