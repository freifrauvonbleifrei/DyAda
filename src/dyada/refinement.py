import bitarray as ba
import bitarray.util
from collections import defaultdict
from functools import lru_cache
import numpy as np
import numpy.typing as npt
from queue import PriorityQueue
from typing import Optional, Sequence, Union

from dyada.coordinates import bitarray_startswith
from dyada.descriptor import (
    RefinementDescriptor,
    Branch,
    get_regular_refined,
    hierarchical_to_box_index_mapping,
    find_uniqueness_violations,
    branch_generator,
)
from dyada.discretization import Discretization
from dyada.linearization import (
    get_dimensionwise_positions,
    get_dimensionwise_positions_from_branch,
    MortonOrderLinearization,
)
from dyada.locationcodemap import LocationCodeMap


def is_lru_cached(func):
    while hasattr(func, "__wrapped__"):
        if hasattr(func, "cache_info"):
            return True
        func = func.__wrapped__
    return hasattr(func, "cache_info")


class PlannedAdaptiveRefinement:
    def __init__(self, discretization: Discretization):
        self._discretization = discretization
        # initialize planned refinement list and data structures used later
        self._planned_refinements: list[tuple[int, npt.NDArray[np.int8]]] = []

        def get_d_zeros_as_array():
            return np.zeros(
                self._discretization.descriptor.get_num_dimensions(), dtype=np.int8
            )

        self._markers: defaultdict[int, npt.NDArray[np.int8]] = defaultdict(
            get_d_zeros_as_array
        )
        self._upward_queue: PriorityQueue[tuple[int, int]] = PriorityQueue()
        self._sub_location_code_map = LocationCodeMap()

    def plan_refinement(self, box_index: int, dimensions_to_refine=None) -> None:
        if dimensions_to_refine is None:
            dimensions_to_refine = (
                "1" * self._discretization.descriptor.get_num_dimensions()
            )
        # must be iterable, convert to np.array
        dimensions_to_refine = np.fromiter(
            dimensions_to_refine,
            dtype=np.int8,
            count=self._discretization.descriptor.get_num_dimensions(),
        )
        # get hierarchical index
        linear_index = self._discretization.descriptor.to_hierarchical_index(box_index)
        # store by linear index
        self._planned_refinements.append(
            (
                linear_index,
                dimensions_to_refine,
            )
        )

    def plan_coarsening(
        self,
        index: int,
        dimensions_to_coarsen: ba.bitarray,
    ) -> None:
        """plan to coarsen a node in the given directions (which merges its children);
        will raise error if this is not possible

        Args:
            index (int): hierarchical index of the node to coarsen
            dimensions_to_coarsen (ba.bitarray): 1 denotes if the node should be coarsened
        """
        num_dimensions = self._discretization.descriptor.get_num_dimensions()
        if not len(dimensions_to_coarsen) == num_dimensions:
            raise ValueError(
                f"Expected {num_dimensions} dimensions, got {len(dimensions_to_coarsen)}"
            )
        dimensions_not_to_coarsen = ~dimensions_to_coarsen
        parent = index

        # assert that the parent refinement has a 1 where coarsening is requested
        parent_refinement = self._discretization.descriptor[parent]
        if (
            not (parent_refinement | dimensions_not_to_coarsen).count()
            == num_dimensions
        ):
            raise ValueError("Current refinement does not match coarsening dimensions")

        children = self._discretization.descriptor.get_siblings(index + 1)
        for child in children:
            # make sure that the children are not refined in the dimensions to be coarsened
            child_refinement = self._discretization.descriptor[child]
            if (child_refinement & dimensions_to_coarsen).count() > 0:
                raise ValueError(
                    f"Child {child} is refined in a dimension to be coarsened"
                )

        np_dimensions_to_coarsen = np.fromiter(
            dimensions_to_coarsen,
            dtype=np.int8,
            count=num_dimensions,
        )
        # then, store planned refinements, but the inverted one for the parent node
        self._planned_refinements.append((parent, -np_dimensions_to_coarsen))

    def populate_queue(self) -> None:
        assert len(self._markers) == 0 and self._upward_queue.empty()

        # put initial markers
        for linear_index, dimensions_to_refine in self._planned_refinements:
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

        self._planned_refinements = []  # clear the planned refinements

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
        current_modified_branch, _ = descriptor.get_branch(
            starting_index, is_box_index=False
        )
        initial_branch_depth = len(current_modified_branch)
        history_of_indices, history_of_level_increments = (
            current_modified_branch.to_history()
        )
        history_of_binary_positions = [
            self._discretization._linearization.get_binary_position_from_index(
                history_of_indices[: i + 1], history_of_level_increments[: i + 1]
            )
            for i in range(initial_branch_depth - 1)
        ]
        # the ancestry, in old indices but new relatonships
        ancestry = descriptor.get_ancestry(current_modified_branch)
        assert len(ancestry) == initial_branch_depth - 1

        self._sub_location_code_map = LocationCodeMap()
        intermediate_generation: list[tuple[int, list[ba.bitarray]]] = []
        while True:
            # get the currently desired location info
            modified_dimensionwise_positions = get_dimensionwise_positions(
                history_of_binary_positions, history_of_level_increments
            )
            current_old_index, intermediate_generation = self.find_next_twig(
                modified_dimensionwise_positions,
                ancestry[-1] if len(ancestry) > 0 else 0,
            )
            if len(modified_dimensionwise_positions) == 0:
                modified_dimensionwise_positions = [
                    ba.bitarray() for _ in range(descriptor.get_num_dimensions())
                ]
            ancestry.append(current_old_index)
            next_refinement, next_marker = self.refinement_with_marker_applied(
                current_old_index
            )
            if next_refinement == descriptor.d_zeros:
                yield current_old_index, next_refinement, modified_dimensionwise_positions, next_marker
                for p in intermediate_generation:
                    # find out which is the new ancestor with the tightest match
                    # for the old location code
                    yield p[0], ba.bitarray(None), None, self._sub_location_code_map[
                        p[1]
                    ]
                # only on leaves, we can advance the branch
                try:
                    current_modified_branch.advance_branch(initial_branch_depth)
                except IndexError:
                    # done!
                    self._sub_location_code_map = LocationCodeMap()
                    return
                # prune all other data to current length,
                # so we don't have to recompute from new branch
                current_modified_branch_depth = len(current_modified_branch)
                history_of_binary_positions = history_of_binary_positions[
                    : current_modified_branch_depth - 2
                ]
                history_of_indices = history_of_indices[
                    : current_modified_branch_depth - 1
                ]
                history_of_indices[-1] += 1
                history_of_level_increments = history_of_level_increments[
                    : current_modified_branch_depth - 1
                ]
                ancestry = ancestry[: current_modified_branch_depth - 1]
            else:
                # on non-leaves, we need to grow the branch
                yield current_old_index, next_refinement, modified_dimensionwise_positions
                current_modified_branch.grow_branch(next_refinement)
                history_of_level_increments.append(next_refinement)
                history_of_indices.append(0)

            # update history of binary positions
            latest_binary_position = (
                self._discretization._linearization.get_binary_position_from_index(
                    history_of_indices, history_of_level_increments
                )
            )
            history_of_binary_positions.append(latest_binary_position)

    def is_old_index_now_at_or_containing_location_code(
        self,
        desired_dimensionwise_positions: list[ba.bitarray],
        parent_of_next_refinement: int,
        parent_branch: Branch,
        old_index: int,
    ) -> tuple[bool, list[ba.bitarray]]:
        descriptor = self._discretization.descriptor
        num_dimensions = descriptor.get_num_dimensions()
        old_index_branch, _ = descriptor.get_branch(
            old_index,
            is_box_index=False,
            hint_previous_branch=(parent_of_next_refinement, parent_branch),
        )
        old_index_dimensionwise_positions = get_dimensionwise_positions_from_branch(
            old_index_branch, self._discretization._linearization
        )

        old_index_ancestors = descriptor.get_ancestry(old_index_branch)
        old_index_ancestry_accumulated_markers = np.sum(
            [self._markers[ancestor] for ancestor in old_index_ancestors],
            axis=0,
        )
        shortened_parent_positions = [
            desired_dimensionwise_positions[d][
                : len(desired_dimensionwise_positions[d])
                - old_index_ancestry_accumulated_markers[d]
            ]
            for d in range(num_dimensions)
        ]
        part_of_history = all(
            bitarray_startswith(
                old_index_dimensionwise_positions[d],
                shortened_parent_positions[d],
            )
            for d in range(num_dimensions)
        )
        return part_of_history, old_index_dimensionwise_positions

    def find_next_twig(
        self,
        desired_dimensionwise_positions: list[ba.bitarray],
        parent_of_next_refinement: int,
    ) -> tuple[int, list[tuple[int, list[ba.bitarray]]]]:
        """get the (old) tree node corresponding to the location code, and any nodes encountered on the way

        Args:
            desired_dimensionwise_positions (list[ba.bitarray]): the location code we're looking for next
            parent_of_next_refinement (int): parent or other ancestor of the index we're looking for

        Returns:
            tuple[int, list[tuple[int, list[ba.bitarray]]]]:
                tuple consisting of the (old) node index
                and a list of (otherwise forgotten) intermediate nodes and their location codes
                    these intermediate nodes are descendants of parent_of_next_refinement,
                    and either culled children of the returned node
                    (in case it is now a leaf),
                    or culled ancestors of the returned node
        """
        # with which next (old) index do we get the currently desired position?
        if len(desired_dimensionwise_positions) == 0:
            return 0, []  # root node
        descriptor = self._discretization.descriptor
        num_dimensions = descriptor.get_num_dimensions()
        parent_branch, _ = descriptor.get_branch(
            parent_of_next_refinement, is_box_index=False
        )
        children = descriptor.get_children(parent_of_next_refinement, parent_branch)

        # determine which child twig to go down next,
        # keeping track of predecessors that are going to disappear
        twig_found = False
        intermediate_generation: list[tuple[int, list[ba.bitarray]]] = (
            []
        )  # with each second entry the location codes

        while not twig_found:
            # find the child whose branch puts it at the same level/index as
            # the modified branch we're looking at
            for child in children:
                part_of_history, child_dimensionwise_positions = (
                    self.is_old_index_now_at_or_containing_location_code(
                        desired_dimensionwise_positions,
                        parent_of_next_refinement,
                        parent_branch,
                        child,
                    )
                )

                if not part_of_history:
                    continue

                child_future_refinement, child_marker = (
                    self.refinement_with_marker_applied(child)
                )
                if (
                    np.min(child_marker) >= 0
                    or child_future_refinement != descriptor.d_zeros
                ):
                    return child, intermediate_generation

                child_refinement = self._discretization.descriptor[child]
                child_refined_dimensions = {
                    d for d in range(num_dimensions) if child_refinement[d] == 1
                }
                children_of_coarsened = descriptor.get_children(child)
                history_matches = all(
                    child_dimensionwise_positions[d]
                    == desired_dimensionwise_positions[d]
                    for d in range(num_dimensions)
                )
                # if it's a perfect match, it's a coarsened node
                if history_matches:
                    twig_found = True
                    # this means that its former children are now gone
                    # and need to be mapped to this child's index
                    for grandchild_index, child_of_coarsened in enumerate(
                        children_of_coarsened
                    ):
                        # need to append the binarized index of the child, broadcast to split dimensions
                        # this needs linearization (if not morton order)
                        if not (
                            self._discretization._linearization
                            == MortonOrderLinearization()
                        ):
                            raise NotImplementedError(
                                "Refinement tracking not implemented for non-Morton order linearizations"
                            )
                        grandchild_location_code = child_dimensionwise_positions.copy()
                        binarized_index = bitarray.util.int2ba(
                            grandchild_index, length=child_refinement.count()
                        )
                        for d_i, d in enumerate(child_refined_dimensions):
                            grandchild_location_code[d].append(binarized_index[d_i])
                        intermediate_generation.append(
                            (child_of_coarsened, grandchild_location_code)
                        )
                    return child, intermediate_generation
                else:
                    # else, it's a node that's going to disappear
                    # -> restart loop with new children and remember this one
                    intermediate_generation.append(
                        (child, child_dimensionwise_positions.copy())
                    )
                    children = children_of_coarsened
        assert False

    def track_indices(self, old_index: int, new_index: int) -> None:
        assert new_index > -1
        if new_index not in self._index_mapping[old_index]:
            self._index_mapping[old_index] += [new_index]

    def extend_descriptor_and_track_indices(
        self,
        new_descriptor: RefinementDescriptor,
        range_to_extend: Union[int, tuple[int, int]],
        extension: ba.bitarray,
        list_of_new_dimensionwise_positions: Sequence[ba.bitarray] | None = None,
    ) -> None:
        previous_length = len(new_descriptor)
        new_descriptor._data.extend(extension)
        if isinstance(range_to_extend, int):
            for new_index in range(previous_length, len(new_descriptor)):
                self.track_indices(range_to_extend, new_index)

            if list_of_new_dimensionwise_positions is not None:
                assert (
                    len(list_of_new_dimensionwise_positions)
                    == self._discretization.descriptor.get_num_dimensions()
                )
                if len(new_descriptor) - previous_length > 1:
                    mini_descriptor = RefinementDescriptor.from_binary(
                        self._discretization.descriptor.get_num_dimensions(), extension
                    )
                    for branch, _ in branch_generator(mini_descriptor):
                        mini_dimensionwise_position = (
                            get_dimensionwise_positions_from_branch(
                                branch, self._discretization._linearization
                            )
                        )
                        if len(mini_dimensionwise_position) == 0:
                            new_dimensionwise_position = (
                                list_of_new_dimensionwise_positions
                            )
                        else:
                            assert len(mini_dimensionwise_position) == len(
                                list_of_new_dimensionwise_positions
                            )
                            new_dimensionwise_position = [
                                list_of_new_dimensionwise_positions[d]
                                + mini_dimensionwise_position[d]
                                for d in range(len(mini_dimensionwise_position))
                            ]
                        self._sub_location_code_map.add(
                            new_dimensionwise_position,
                            new_index,
                        )
                else:
                    self._sub_location_code_map.add(
                        list_of_new_dimensionwise_positions,
                        previous_length,
                    )
        else:
            for old_index, new_index in zip(
                range(range_to_extend[0], range_to_extend[1]),
                range(previous_length, len(new_descriptor)),
            ):
                self.track_indices(old_index, new_index)

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

            # linearly copy up to marked
            self.extend_descriptor_and_track_indices(
                new_descriptor,
                (one_after_last_extended_index, index_to_refine),
                old_descriptor[one_after_last_extended_index:index_to_refine],
            )

            # only refine where there are markers
            last_extended_index = -1
            for (
                old_index,
                new_refinement,
                list_of_new_dimensionwise_positions,
                *marker_or_ancestor,
            ) in self.modified_branch_generator(index_to_refine):
                if len(marker_or_ancestor) > 0:
                    assert (
                        new_refinement == old_descriptor.d_zeros
                        or new_refinement == ba.bitarray(None)
                    )
                    if np.min(marker_or_ancestor) < 0:
                        # a node was coarsened, but is still there
                        assert self._discretization.descriptor[old_index].count() > 0
                        self.extend_descriptor_and_track_indices(
                            new_descriptor,
                            old_index,
                            new_refinement,
                            list_of_new_dimensionwise_positions,
                        )
                    else:
                        if new_refinement == ba.bitarray(None):
                            # node is vanishing
                            assert len(marker_or_ancestor) == 1
                            # track only the index,
                            # namely the one of the previous (grand...)parent
                            new_index = marker_or_ancestor[0]
                            self.track_indices(old_index, new_index)
                        else:
                            # case of expanding a leaf
                            assert (
                                self._discretization.descriptor[old_index].count() == 0
                            )
                            self.extend_descriptor_and_track_indices(
                                new_descriptor,
                                old_index,
                                get_regular_refined(self._markers[old_index]),  # type: ignore #marker?
                                list_of_new_dimensionwise_positions,
                            )
                else:
                    # a stable parent node
                    self.extend_descriptor_and_track_indices(
                        new_descriptor,
                        old_index,
                        new_refinement,
                        list_of_new_dimensionwise_positions,
                    )
                last_extended_index = max(last_extended_index, old_index)
            one_after_last_extended_index = last_extended_index + 1

        # copy rest and return
        self.extend_descriptor_and_track_indices(
            new_descriptor,
            (one_after_last_extended_index, len(old_descriptor)),
            old_descriptor[one_after_last_extended_index : len(old_descriptor)],
        )
        return new_descriptor

    def create_new_descriptor(
        self, track_mapping: str = "boxes"
    ) -> Union[RefinementDescriptor, tuple[RefinementDescriptor, dict]]:
        self._index_mapping: dict[int, list[int]] = defaultdict(list)

        # start generating the new descriptor
        new_descriptor = RefinementDescriptor(
            self._discretization.descriptor.get_num_dimensions()
        )
        new_descriptor._data = ba.bitarray()

        # we are not changing the old descriptor, and greedily build the new one
        # so we can cache the box indices of both
        if not is_lru_cached(self._discretization.descriptor.to_box_index):
            self._discretization.descriptor.to_box_index = lru_cache(maxsize=None)(
                self._discretization.descriptor._to_box_index_recursive
            )
            self._discretization.descriptor._to_box_index_recursive = lru_cache(
                maxsize=None
            )(self._discretization.descriptor._to_box_index_recursive)
        new_descriptor.to_box_index = lru_cache(maxsize=None)(  # type: ignore
            new_descriptor._to_box_index_recursive
        )
        new_descriptor._to_box_index_recursive = lru_cache(maxsize=None)(  # type: ignore
            new_descriptor._to_box_index_recursive
        )

        new_descriptor = self.add_refined_data(new_descriptor)

        if track_mapping == "boxes":
            # transform the mapping to box indices
            self._index_mapping = hierarchical_to_box_index_mapping(
                self._index_mapping,
                self._discretization.descriptor,
                new_descriptor,
            )
        elif track_mapping == "patches":
            pass
        else:
            raise ValueError(
                "track_mapping must be either 'boxes' or 'patches', got "
                + str(track_mapping)
            )
        return new_descriptor, self._index_mapping

    def apply_refinements(
        self, track_mapping: str = "boxes"
    ) -> Union[RefinementDescriptor, tuple[RefinementDescriptor, dict]]:
        assert self._upward_queue.empty()
        assert self._markers == {}
        self.populate_queue()
        self.upwards_sweep()
        self.downwards_sweep()
        assert len(self._planned_refinements) == 0

        return self.create_new_descriptor(track_mapping)


def apply_single_refinement(
    discretization: Discretization,
    box_index: int,
    dimensions_to_refine: Optional[ba.bitarray] = None,
    track_mapping: str = "boxes",
) -> tuple[Discretization, dict]:
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(box_index, dimensions_to_refine)
    new_descriptor, mapping = p.apply_refinements(track_mapping=track_mapping)
    return Discretization(discretization._linearization, new_descriptor), mapping


def merge_mappings(
    first_mapping: dict[int, list[int]],
    second_mapping: dict[int, list[int]],
) -> dict[int, list[int]]:
    # if either mapping is empty, return the other one
    if not first_mapping:
        return second_mapping
    if not second_mapping:
        return first_mapping
    for k, v in first_mapping.items():
        assert isinstance(k, int) and isinstance(v, list)
        for v_i in v:
            assert isinstance(v_i, int)
    # merge the mappings
    merged_mapping: dict[int, list[int]] = defaultdict(list)
    for k, v in first_mapping.items():
        for v_i in v:
            merged_mapping[k] += second_mapping[v_i]
    return merged_mapping


def normalize_discretization(
    discretization: Discretization,
    track_mapping: str = "patches",
    max_normalization_rounds: int = 2**31 - 1,
) -> tuple[RefinementDescriptor, dict[int, list[int]], int]:
    """
    Normalize the discretization so that it fulfills the uniqueness condition
    and we get a normalized omnitree.
    """
    descriptor = discretization.descriptor
    normalization_rounds = 0
    # find the tuples of indices where the uniqueness condition is violated
    violations = find_uniqueness_violations(descriptor)
    mapping: dict[int, list[int]] = {}
    while len(violations) > 0 and normalization_rounds < max_normalization_rounds:
        normalization_rounds += 1
        p = PlannedAdaptiveRefinement(
            Discretization(discretization._linearization, descriptor)
        )
        # remove these violations, by putting markers and executing create_new_descriptor
        for violation in violations:
            # find the dimension(s) of the violation
            sorted_violation = sorted(violation)
            dimensions_to_shift = ~ba.bitarray(descriptor[sorted_violation[0]])
            for i in sorted_violation[1:]:
                dimensions_to_shift &= descriptor[i]
            assert dimensions_to_shift.count() > 0
            dimensions_to_shift_array = np.fromiter(
                dimensions_to_shift,
                dtype=np.int8,
                count=descriptor.get_num_dimensions(),
            )
            p._markers[sorted_violation[0]] += dimensions_to_shift_array
            for i in sorted_violation[1:]:
                p._markers[i] -= dimensions_to_shift_array
        # apply the refinements
        new_descriptor, new_mapping = p.create_new_descriptor(
            track_mapping=track_mapping
        )
        mapping = merge_mappings(mapping, new_mapping)

        descriptor = new_descriptor
        violations = find_uniqueness_violations(descriptor)

    return descriptor, mapping, normalization_rounds
