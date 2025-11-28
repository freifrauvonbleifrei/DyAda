import bitarray as ba
import numpy as np
import numpy.typing as npt
from types import MappingProxyType

from dyada.coordinates import bitarray_startswith
from dyada.descriptor import (
    RefinementDescriptor,
    Branch,
)
from dyada.discretization import Discretization
from dyada.linearization import (
    MortonOrderLinearization,
    get_dimensionwise_positions,
    get_dimensionwise_positions_from_branch,
)


class AncestryBranch:
    """An AncestryBranch couples a Branch together with a history of binary positions
     and node indices (as in the descriptors get_ancestry function).

    It can be used to iterate a new descriptor before it's explicitly constructed, by
    using the markers to determine to which location / position in the old discretization to grow next.

    Invariants:
    - branch, history_of_indices, history_of_level_increments, history_of_binary_positions and ancestry
      are always consistent with each other
    - ancestry corresponds to the old indices on the current_modified_branch, but with potentially
      new relationships (not determined by old descriptor)
    """

    def __init__(self, discretization: Discretization, starting_index: int):
        self._discretization = discretization
        descriptor = self._discretization.descriptor
        # iterates a modified version of the descriptor, incorporating the markers knowledge
        # and keeping track of the ancestry
        self._current_modified_branch: Branch
        self._current_modified_branch, _ = descriptor.get_branch(
            starting_index, is_box_index=False
        )
        self._initial_branch_depth = len(self._current_modified_branch)
        self._history_of_indices, self._history_of_level_increments = (
            self._current_modified_branch.to_history()
        )
        self._history_of_binary_positions = []
        for i in range(self._initial_branch_depth - 1):
            self._history_of_binary_positions.append(
                self._discretization._linearization.get_binary_position_from_index(
                    self._history_of_indices[: i + 1],
                    self._history_of_level_increments[: i + 1],
                )
            )
        # the ancestry, in old indices but new relationships
        self.ancestry = descriptor.get_ancestry(self._current_modified_branch)
        assert len(self.ancestry) == self._initial_branch_depth - 1

    def get_current_location_info(
        self, markers: MappingProxyType[int, npt.NDArray[np.int8]]
    ) -> tuple[int, set[int], ba.frozenbitarray, npt.NDArray[np.int8]]:
        # get the currently desired location info
        current_old_index = 0
        intermediate_generation: set[int] = set()
        if len(self._history_of_binary_positions) > 0:  # if not at root
            modified_dimensionwise_positions = get_dimensionwise_positions(
                self._history_of_binary_positions, self._history_of_level_increments
            )
            current_old_index, intermediate_generation = find_next_twig(
                self._discretization,
                markers,
                modified_dimensionwise_positions,
                self.ancestry[-1],
            )
        self.ancestry.append(current_old_index)
        next_refinement, next_marker = refinement_with_marker_applied(
            self._discretization.descriptor, current_old_index, markers
        )

        return current_old_index, intermediate_generation, next_refinement, next_marker

    def advance(self) -> None:
        self._current_modified_branch.advance_branch(self._initial_branch_depth)
        # prune all other data to current length,
        # so we don't have to recompute from new branch
        current_modified_branch_depth = len(self._current_modified_branch)
        self._history_of_binary_positions = self._history_of_binary_positions[
            : current_modified_branch_depth - 2
        ]
        self._history_of_indices = self._history_of_indices[
            : current_modified_branch_depth - 1
        ]
        self._history_of_indices[-1] += 1
        self._history_of_level_increments = self._history_of_level_increments[
            : current_modified_branch_depth - 1
        ]
        self.ancestry = self.ancestry[: current_modified_branch_depth - 1]

        # update history of binary positions
        latest_binary_position = (
            self._discretization._linearization.get_binary_position_from_index(
                self._history_of_indices, self._history_of_level_increments
            )
        )
        self._history_of_binary_positions.append(latest_binary_position)

    def grow(self, next_refinement: ba.frozenbitarray) -> None:
        self._current_modified_branch.grow_branch(next_refinement)
        self._history_of_level_increments.append(next_refinement)
        self._history_of_indices.append(0)
        # update history of binary positions
        latest_binary_position = (
            self._discretization._linearization.get_binary_position_from_index(
                self._history_of_indices, self._history_of_level_increments
            )
        )
        self._history_of_binary_positions.append(latest_binary_position)


def refinement_with_marker_applied(
    descriptor: RefinementDescriptor,
    linear_index: int,
    markers: MappingProxyType[int, np.ndarray],
) -> tuple[ba.frozenbitarray, npt.NDArray[np.int8]]:
    refinement = descriptor[linear_index]
    marker = markers[linear_index]
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


def is_old_index_now_at_or_containing_location_code(
    discretization: Discretization,
    markers: MappingProxyType[int, npt.NDArray[np.int8]],
    desired_dimensionwise_positions: list[ba.bitarray],
    parent_of_next_refinement: int,
    parent_branch: Branch,
    old_index: int,
) -> tuple[bool, list[ba.bitarray]]:
    """Check whether old_index is now at or containing the next hyperrectangular location code.
    Args:
        discretization (Discretization): the old discretization we're referring to
        markers (MappingProxyType[int, npt.NDArray[np.int8]]): markers that should be applied to the old discretization
        desired_dimensionwise_positions (list[ba.bitarray]): the location code we're looking for next
        parent_of_next_refinement (int): the current parent or other ancestor of the index we're looking for
        parent_branch (Branch): the branch corresponding to the parent_of_next_refinement
        old_index (int): the old index we're checking

    Returns:
        bool            : whether old_index is now at or containing the location code
        list[ba.bitarray]: the location code of the old_index
    """
    descriptor = discretization.descriptor
    old_index_branch, _ = descriptor.get_branch(
        old_index,
        is_box_index=False,
        hint_previous_branch=(parent_of_next_refinement, parent_branch),
    )
    old_index_dimensionwise_positions = get_dimensionwise_positions_from_branch(
        old_index_branch, discretization._linearization
    )

    old_index_ancestors = descriptor.get_ancestry(old_index_branch)
    old_index_ancestry_accumulated_markers = np.sum(
        [markers[ancestor] for ancestor in old_index_ancestors],
        axis=0,
    )
    shortened_parent_positions = [
        desired_dimensionwise_positions[d][
            : len(desired_dimensionwise_positions[d])
            - old_index_ancestry_accumulated_markers[d]
        ]
        for d in range(descriptor.get_num_dimensions())
    ]
    part_of_history = all(
        bitarray_startswith(
            old_index_dimensionwise_positions[d],
            shortened_parent_positions[d],
        )
        for d in range(descriptor.get_num_dimensions())
    )
    return part_of_history, old_index_dimensionwise_positions


def old_node_will_be_contained_in_new_descriptor(
    descriptor: RefinementDescriptor,
    old_index: int,
    markers: MappingProxyType[int, npt.NDArray[np.int8]],
) -> bool:
    """Check whether the node at old_index will be contained in the new descriptor after applying markers.

    Returns:
        bool: True if the node will be contained, False if it will be coarsened away.
    """
    future_refinement, marker = refinement_with_marker_applied(
        descriptor, old_index, markers
    )
    return np.min(marker) >= 0 or future_refinement != descriptor.d_zeros  # type: ignore


def find_next_twig(
    discretization: Discretization,
    markers: MappingProxyType[int, npt.NDArray[np.int8]],
    desired_dimensionwise_positions: list[ba.bitarray],
    parent_of_next_refinement: int,
) -> tuple[int, set[int]]:
    """Get the (old) tree node corresponding to the location code, and any nodes encountered on the way.
    Args:
        discretization (Discretization): the old discretization we're referring to
        markers (MappingProxyType[int, npt.NDArray[np.int8]]): refinement markers that should be applied to the discretization
        desired_dimensionwise_positions (list[ba.bitarray]): the location code we're looking for next
        parent_of_next_refinement (int): parent or other ancestor of the index we're looking for
    Returns:
        tuple[int, set[int]]:
            tuple consisting of the (old) node index
            and a set of (otherwise forgotten) intermediate nodes
    """
    descriptor = discretization.descriptor
    parent_branch, _ = descriptor.get_branch(
        parent_of_next_refinement, is_box_index=False
    )
    children = descriptor.get_children(parent_of_next_refinement, parent_branch)
    intermediate_generation: set[int] = set()
    while True:
        for child in children:
            part_of_history, child_dimensionwise_positions = (
                is_old_index_now_at_or_containing_location_code(
                    discretization,
                    markers,
                    desired_dimensionwise_positions,
                    parent_of_next_refinement,
                    parent_branch,
                    child,
                )
            )
            if not part_of_history:
                continue

            if old_node_will_be_contained_in_new_descriptor(descriptor, child, markers):
                return child, intermediate_generation  # we found the next twig

            # else it's a coarsened node and we can see if there is a matching child
            children_of_coarsened = descriptor.get_children(child)
            history_matches = (
                child_dimensionwise_positions == desired_dimensionwise_positions
            )
            if history_matches:
                # this means that its former children are now gone and need to be mapped to this child's index
                for child_of_coarsened in children_of_coarsened:
                    # need to append the binarized index of the child, broadcast to split dimensions
                    if not (
                        discretization._linearization == MortonOrderLinearization()
                    ):  # this would need proper linearization
                        raise NotImplementedError("Refinement tracking")
                    intermediate_generation.add(child_of_coarsened)
                return child, intermediate_generation
            else:
                # else, it's an ancestor node that's going to disappear
                # -> restart loop with new children and remember this one
                intermediate_generation.add(child)
                children = children_of_coarsened
