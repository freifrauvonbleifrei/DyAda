import bitarray as ba
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from queue import PriorityQueue
from typing import Generator, Optional, Union

from dyada.coordinates import (
    get_coordinates_from_level_index,
    LevelIndex,
    Coordinate,
    CoordinateInterval,
)

from dyada.descriptor import (
    Branch,
    RefinementDescriptor,
    get_level_from_branch,
    get_regular_refined,
)
from dyada.linearization import (
    get_dimensionwise_positions,
    Linearization,
)


class RefinementError(Exception):
    def __init__(
        self,
        message: str,
        descriptor: RefinementDescriptor,
        markers: Optional[dict] = None,
        error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.descriptor = descriptor
        self.markers = markers
        self.error = error


def get_level_index_from_branch(
    linearization: Linearization, branch: Branch
) -> LevelIndex:
    num_dimensions = len(branch[0].level_increment)
    found_level = get_level_from_branch(branch)

    # once the branch is found, we can infer the vector index from the branch stack
    current_index: np.ndarray = np.array([0] * num_dimensions, dtype=np.int64)
    decreasing_level_difference = found_level.copy()
    history_of_indices, history_of_level_increments = branch.to_history()
    for level_count in range(1, len(branch)):
        bit_index = linearization.get_binary_position_from_index(
            history_of_indices[:level_count],
            history_of_level_increments[:level_count],
        )
        array_index = np.fromiter(bit_index, dtype=np.int64, count=num_dimensions)
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

    def __len__(self):
        return self._descriptor.get_num_boxes()

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


def coordinates_from_box_index(
    discretiztion: Discretization,
    index: int,
    full_domain: Optional[CoordinateInterval] = None,
) -> CoordinateInterval:
    level_index = get_level_index_from_linear_index(
        discretiztion._linearization, discretiztion._descriptor, index
    )
    coordinates = get_coordinates_from_level_index(level_index)
    if full_domain is not None:
        scaling_factor = full_domain.upper_bound - full_domain.lower_bound
        offset = full_domain.lower_bound
        coordinates = CoordinateInterval(
            lower_bound=coordinates.lower_bound * scaling_factor + offset,
            upper_bound=coordinates.upper_bound * scaling_factor + offset,
        )
    return coordinates


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
        self._remembered_splits: dict[int, tuple[ba.bitarray, int]] = {}

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
            # TODO accept optional branch argument to reduce lookup time
            descendants_indices = self._discretization.descriptor.get_children(
                ancestor_index
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

    def downwards_sweep(self) -> None:
        descriptor = self._discretization.descriptor
        if len(self._markers) == 0:
            return
        current_index = min(self._markers.keys())
        while True:
            if descriptor.is_box(current_index):
                # if it's a leaf node, continue
                assert np.all(self._markers[current_index] > -1)

            else:
                # check if (parts of) the marker need pushing down
                marker_to_push_down = self._markers[current_index].copy()

                # 1st case: negative but cannot be used to coarsen here
                marker_negative = marker_to_push_down < 0
                can_be_coarsened = np.fromiter(
                    descriptor[current_index],
                    dtype=np.int8,
                    count=descriptor.get_num_dimensions(),
                )
                marker_to_push_down[marker_negative] += can_be_coarsened[
                    marker_negative
                ]

                # 2nd case: positive but cannot be used to refine here
                marker_positive = marker_to_push_down > 0
                can_be_refined = np.fromiter(
                    ~descriptor[current_index],
                    dtype=np.int8,
                    count=descriptor.get_num_dimensions(),
                )
                marker_to_push_down[marker_positive] -= can_be_refined[marker_positive]

                self.move_marker_to_descendants(current_index, marker_to_push_down)

            filtered_markers = {
                k: v for k, v in self._markers.items() if current_index < k
            }
            current_index = min(filtered_markers.keys(), default=-2)
            while current_index != -2 and not np.any(
                filtered_markers[current_index]
            ):  # remove zero markers
                self._markers.pop(current_index)
                filtered_markers.pop(current_index)
                current_index = min(filtered_markers.keys(), default=-2)
            if current_index == -2:
                # no more indices left in markers
                break

    def refinement_with_marker_applied(
        self, linear_index
    ) -> tuple[ba.frozenbitarray, npt.NDArray[np.int8]]:
        refinement = self._discretization.descriptor[linear_index]
        marker = self._markers[linear_index]
        if refinement.count() == 0:
            assert np.all(marker > -1)
            return refinement, marker

        positive = ba.bitarray([1 if m > 0 else 0 for m in marker])
        assert (refinement & positive).count() == 0
        refinement |= positive

        negative = ba.bitarray([1 if m < 0 else 0 for m in marker])
        assert (~refinement & negative).count() == 0
        refinement ^= negative

        return refinement, marker

    def modified_branch_generator(self, starting_index: int):
        descriptor = self._discretization.descriptor
        # iterates a modified version of the descriptor, incorporating the markers knowledge
        # and keeping track of the ancestry
        current_branch, _ = descriptor.get_branch(starting_index, is_box_index=False)
        initial_branch_depth = len(current_branch)
        current_branch_depth = initial_branch_depth
        history_of_indices, history_of_level_increments = current_branch.to_history()
        history_of_binary_positions = []
        for i in range(initial_branch_depth - 1):
            history_of_binary_positions.append(
                self._discretization._linearization.get_binary_position_from_index(
                    history_of_indices[: i + 1], history_of_level_increments[: i + 1]
                )
            )

        ancestry = descriptor.get_ancestry(current_branch)
        assert len(ancestry) == current_branch_depth - 1
        ancestry.append(starting_index)
        current_old_index = starting_index
        current_refinement = descriptor[current_old_index]
        next_refinement, next_marker = self.refinement_with_marker_applied(
            current_old_index
        )
        # currently_accumulated_markers = {starting_index: next_marker}
        while True:
            if next_refinement == descriptor.d_zeros:
                assert (current_refinement).count() == 0
                # on leaves, add end-refinement info
                yield current_old_index, next_refinement, next_marker
                try:
                    current_branch.advance_branch(initial_branch_depth)
                except IndexError:
                    # done!
                    return
                # prune all other data to current length
                current_branch_depth = len(current_branch)
                history_of_binary_positions = history_of_binary_positions[
                    : current_branch_depth - 2
                ]
                history_of_indices = history_of_indices[: current_branch_depth - 1]
                history_of_indices[-1] += 1
                history_of_level_increments = history_of_level_increments[
                    : current_branch_depth - 1
                ]
                ancestry = ancestry[: current_branch_depth - 1]
            else:
                yield current_old_index, next_refinement
                current_branch.grow_branch(next_refinement)
                history_of_level_increments.append(next_refinement)
                history_of_indices.append(0)

            # with which binary position do we get the current history_of_indices?
            latest_binary_position = (
                self._discretization._linearization.get_binary_position_from_index(
                    history_of_indices, history_of_level_increments
                )
            )
            history_of_binary_positions.append(latest_binary_position)
            # the associated location info
            modified_dimensionwise_positions = get_dimensionwise_positions(
                history_of_binary_positions, history_of_level_increments
            )

            parent_of_next_refinement = ancestry[-1]
            parent_branch, _ = descriptor.get_branch(
                parent_of_next_refinement, is_box_index=False
            )
            children = set(
                descriptor.get_children(parent_of_next_refinement, parent_branch)
            )
            children_to_consider = children.copy()
            updating_children = True
            while updating_children:
                updating_children = False
                for child in children:
                    child_future_refinement, child_marker = (
                        self.refinement_with_marker_applied(child)
                    )
                    # if a child has negative markers and will be coarsened away,
                    #  we'll need to look at its descendants instead
                    if (
                        np.min(child_marker) < 0
                        and child_future_refinement == descriptor.d_zeros
                    ):
                        children_of_coarsened = descriptor.get_children(child)
                        children_to_consider.update(children_of_coarsened)
                        children_to_consider.remove(child)
                        updating_children = True
                children = children_to_consider.copy()

            # determine which child twig to go down next
            for child in children:
                # find the child whose branch puts it at the same level/index as
                # the modified branch we're looking at
                child_old_branch, _ = descriptor.get_branch(
                    child,
                    is_box_index=False,
                    hint_previous_branch=(parent_of_next_refinement, parent_branch),
                )
                child_history_of_indices, child_history_of_level_increments = (
                    child_old_branch.to_history()
                )
                child_history_of_binary_positions = []
                for i in range(len(child_history_of_indices)):
                    child_history_of_binary_positions.append(
                        self._discretization._linearization.get_binary_position_from_index(
                            child_history_of_indices[: i + 1],
                            child_history_of_level_increments[: i + 1],
                        )
                    )
                child_ancestors = descriptor.get_ancestry(child_old_branch)
                child_accumulated_markers = np.sum(
                    [self._markers[ancestor] for ancestor in child_ancestors], axis=0
                )
                assert len(child_accumulated_markers) == len(next_marker)
                assert np.all(child_accumulated_markers >= 0)
                child_old_dimensionwise_positions = get_dimensionwise_positions(
                    child_history_of_binary_positions, child_history_of_level_increments
                )

                history_matches = True
                for d in range(len(modified_dimensionwise_positions)):
                    no_compare_at_end = child_accumulated_markers[d]
                    child_compare_this_dim = child_old_dimensionwise_positions[d][:]
                    modified_compare_this_dim = modified_dimensionwise_positions[d][
                        : len(modified_dimensionwise_positions[d]) - no_compare_at_end
                    ]

                    if child_compare_this_dim != modified_compare_this_dim:
                        history_matches = False
                        break
                if history_matches:
                    current_old_index = child
                    break

            ancestry.append(current_old_index)
            current_refinement = descriptor[current_old_index]
            next_refinement, next_marker = self.refinement_with_marker_applied(
                current_old_index
            )

    def extend_descriptor_and_track_boxes(
        self,
        new_descriptor: RefinementDescriptor,
        range_to_extend: Union[int, tuple[int, int]],
        extension: ba.bitarray,
    ) -> None:
        previous_length = len(new_descriptor)
        new_descriptor._data.extend(extension)
        if self._box_index_mapping is None:
            return

        old_descriptor = self._discretization.descriptor
        if isinstance(range_to_extend, int):
            self._box_index_mapping[
                old_descriptor.to_box_index(range_to_extend)
            ] += list(
                new_descriptor.to_box_index(i)
                for i in range(previous_length, len(new_descriptor))
                if new_descriptor.is_box(i)
            )
        else:
            for old_index, new_index in zip(
                range(range_to_extend[0], range_to_extend[1]),
                range(previous_length, len(new_descriptor)),
            ):
                if old_descriptor.is_box(old_index):
                    assert new_descriptor.is_box(new_index)
                    self._box_index_mapping[old_descriptor.to_box_index(old_index)] += [
                        new_descriptor.to_box_index(new_index)
                    ]

    def add_refined_data(
        self, new_descriptor: RefinementDescriptor
    ) -> RefinementDescriptor:
        old_descriptor = self._discretization.descriptor
        one_after_last_extended_index = 0

        while one_after_last_extended_index < len(old_descriptor):
            # filter the markers to the current interval
            filtered_markers = {
                k: v
                for k, v in self._markers.items()
                if k >= one_after_last_extended_index
            }
            index_to_refine = min(filtered_markers.keys(), default=-1)
            if index_to_refine == -1:
                break

            # copy up to marked
            self.extend_descriptor_and_track_boxes(
                new_descriptor,
                (one_after_last_extended_index, index_to_refine),
                old_descriptor[one_after_last_extended_index:index_to_refine],
            )

            modified_branches = self.modified_branch_generator(index_to_refine)
            for old_index, new_refinement, *marker in modified_branches:
                if marker != []:
                    assert self._discretization.descriptor[old_index].count() == 0
                    assert np.min(marker) >= 0
                    self.extend_descriptor_and_track_boxes(
                        new_descriptor,
                        old_index,
                        get_regular_refined(self._markers[old_index]),  # type: ignore
                    )
                else:
                    new_descriptor._data.extend(new_refinement)
            one_after_last_extended_index = old_index + 1

        # copy rest and return
        self.extend_descriptor_and_track_boxes(
            new_descriptor,
            (one_after_last_extended_index, len(old_descriptor)),
            old_descriptor[one_after_last_extended_index : len(old_descriptor)],
        )
        return new_descriptor

    def create_new_descriptor(
        self, track_mapping: bool
    ) -> Union[RefinementDescriptor, tuple[RefinementDescriptor, dict]]:
        self._box_index_mapping: Optional[dict[int, list[int]]] = None
        if track_mapping:
            self._box_index_mapping = defaultdict(list)

        # start generating the new descriptor
        new_descriptor = RefinementDescriptor(
            self._discretization.descriptor.get_num_dimensions()
        )
        new_descriptor._data = ba.bitarray()
        try:
            new_descriptor = self.add_refined_data(new_descriptor)
        except Exception as e:
            raise RefinementError(
                "Error during refinement", new_descriptor, self._markers, e
            )

        assert len(new_descriptor._data) >= len(self._discretization.descriptor)
        if track_mapping:
            assert self._box_index_mapping is not None
            return new_descriptor, self._box_index_mapping
        return new_descriptor

    def apply_refinements(
        self, track_mapping: bool = False
    ) -> Union[RefinementDescriptor, tuple[RefinementDescriptor, dict]]:
        assert self._upward_queue.empty()
        assert self._markers == {}
        self.populate_queue()
        self.upwards_sweep()
        self.downwards_sweep()
        assert self._planned_refinements.empty()

        return self.create_new_descriptor(track_mapping)


def apply_single_refinement(
    discretization: Discretization, box_index: int, dimensions_to_refine: ba.bitarray
) -> tuple[Discretization, dict]:
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(box_index, dimensions_to_refine)
    new_descriptor, mapping = p.apply_refinements(track_mapping=True)
    return Discretization(discretization._linearization, new_descriptor), mapping
