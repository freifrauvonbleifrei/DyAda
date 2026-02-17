# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from typing import TypeAlias, Union

from dyada.coordinates import bitarray_startswith
from dyada.descriptor import (
    RefinementDescriptor,
    Branch,
)
from dyada.discretization import Discretization, branch_to_location_code
from dyada.linearization import (
    CoarseningStack,
    TrackToken,
    LocationCode,
    binary_or_none_generator,
    bitmask_to_indices,
    get_initial_coarsen_refine_stack,
    location_code_from_history,
    location_code_from_branch,
    inform_same_remaining_position_about_index,
)
from dyada.markers import MarkerType, MarkersMapProxyType


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

    TrackInfo: TypeAlias = CoarseningStack

    @dataclass
    class _AncestryBranchState:
        discretization: Discretization
        current_modified_branch: Branch
        initial_branch_depth: int
        history_of_indices: list[int]
        history_of_level_increments: list[ba.frozenbitarray]
        history_of_binary_positions: list[ba.bitarray]

        def __init__(self, discretization: Discretization, starting_index: int):
            self.discretization = discretization
            descriptor = discretization.descriptor
            self.current_modified_branch, _ = descriptor.get_branch(
                starting_index, is_box_index=False
            )
            self.initial_branch_depth = len(self.current_modified_branch)

            self.history_of_indices, self.history_of_level_increments = (
                self.current_modified_branch.to_history()
            )
            self.history_of_binary_positions = [
                discretization._linearization.get_binary_position_from_index(
                    self.history_of_indices[: i + 1],
                    self.history_of_level_increments[: i + 1],
                )
                for i in range(len(self.current_modified_branch) - 1)
            ]

        def __len__(self) -> int:
            return len(self.history_of_binary_positions)

        def advance(self) -> int:
            self.current_modified_branch.advance_branch(self.initial_branch_depth)

            # prune all other data to current length,
            # so we don't have to recompute from new branch
            current_modified_branch_depth = len(self.current_modified_branch)
            self.history_of_binary_positions = self.history_of_binary_positions[
                : current_modified_branch_depth - 2
            ]
            self.history_of_indices = self.history_of_indices[
                : current_modified_branch_depth - 1
            ]
            self.history_of_indices[-1] += 1
            self.history_of_level_increments = self.history_of_level_increments[
                : current_modified_branch_depth - 1
            ]

            # update history of binary positions
            latest_binary_position = (
                self.discretization._linearization.get_binary_position_from_index(
                    self.history_of_indices, self.history_of_level_increments
                )
            )
            self.history_of_binary_positions.append(latest_binary_position)
            return current_modified_branch_depth

        def grow(self, next_refinement: ba.frozenbitarray) -> None:
            self.current_modified_branch.grow_branch(next_refinement)
            self.history_of_level_increments.append(next_refinement)
            self.history_of_indices.append(0)
            # update history of binary positions
            latest_binary_position = (
                self.discretization._linearization.get_binary_position_from_index(
                    self.history_of_indices, self.history_of_level_increments
                )
            )
            self.history_of_binary_positions.append(latest_binary_position)

    @dataclass
    class _AncestryMappingState:
        # the ancestry, in old indices but new relationships
        ancestry: list[int]
        old_indices_map_track_tokens: defaultdict[int, list[TrackToken]]
        current_track_token: TrackToken
        missed_mappings: defaultdict[int, set[TrackToken]]
        track_info_mapping: dict[int, "AncestryBranch.TrackInfo"]

        def __init__(
            self,
            discretization: Discretization,
            ancestry_branch_state: "AncestryBranch._AncestryBranchState",
        ):
            self.ancestry = discretization.descriptor.get_ancestry(
                ancestry_branch_state.current_modified_branch
            )
            self.old_indices_map_track_tokens = defaultdict(list)
            self.current_track_token = TrackToken(-1)
            self.missed_mappings = defaultdict(set)
            self.track_info_mapping = {}

        def update(
            self,
            discretization,
            current_old_index: int,
            markers: MarkersMapProxyType,
            next_marker: MarkerType,
            intermediate_generation: set[int],
            exact: bool,
        ) -> TrackToken:
            self.current_track_token = TrackToken(self.current_track_token.t + 1)

            self.store_missed_mappings(
                current_old_index,
                intermediate_generation,
                exact,
            )
            # process new track info
            if current_old_index not in self.track_info_mapping:
                relevant_markers = [next_marker]
                for ancestor_index in self.ancestry:
                    relevant_markers.append(markers[ancestor_index])
                most_recent_track_info = self._get_initial_track_info(
                    discretization.descriptor[current_old_index],
                    relevant_markers,
                )
                if most_recent_track_info is not None:
                    self.track_info_mapping[current_old_index] = most_recent_track_info
            self.ancestry.append(current_old_index)
            self.old_indices_map_track_tokens[current_old_index].append(
                self.current_track_token
            )

            return self.current_track_token

        def _get_initial_track_info(
            self,
            current_refinement: ba.frozenbitarray,
            markers: list[MarkerType],
        ) -> Union["AncestryBranch.TrackInfo", None]:
            if current_refinement.count() == 0:
                return None
            marker = np.sum(markers, axis=0)
            if marker.min() < 0:
                negative = ba.bitarray(
                    1 if marker[i] < 0 else 0 for i in range(len(marker))
                )

                dimensions_to_coarsen = ba.frozenbitarray(negative & current_refinement)
                dimensions_cannot_coarsen = ba.frozenbitarray(
                    negative & ~current_refinement
                )
                dimensions_to_refine = ba.frozenbitarray(
                    ba.frozenbitarray(
                        1 if marker[i] > 0 else 0 for i in range(len(marker))
                    )
                )
                coarsen_refine_stack = get_initial_coarsen_refine_stack(
                    ba.frozenbitarray(current_refinement),
                    dimensions_to_coarsen,
                    dimensions_to_refine,
                    (
                        dimensions_cannot_coarsen
                        if dimensions_cannot_coarsen.count() > 0
                        else None
                    ),
                )
                return coarsen_refine_stack
            return None

        def store_missed_mappings(
            self, current_old_index: int, intermediate_generation: set[int], exact: bool
        ) -> None:
            if not self.ancestry:
                return  # at root, nothing to map
            parent_index = self.ancestry[-1]
            parent_tokens = self.old_indices_map_track_tokens[parent_index]

            for skipped_index in intermediate_generation:
                if skipped_index < current_old_index:  # is ancestor
                    # map upward to parent
                    self.missed_mappings[skipped_index].add(parent_tokens[-1])
                # either descendant -> map upward to current, or ancestor -> map downward to current
                self.missed_mappings[skipped_index].add(self.current_track_token)

            if not exact:
                # no exact match, map current_old_index upward
                self.missed_mappings[current_old_index].add(parent_tokens[-1])
                for ancestor_index, ancestor in reversed(
                    list(enumerate(self.ancestry))
                ):
                    ancestor_exact = len(self.missed_mappings.get(ancestor, [])) == 0
                    if ancestor_exact:
                        break
                    older_ancestor = self.ancestry[ancestor_index - 1]
                    older_ancestor_token = min(
                        self.old_indices_map_track_tokens[older_ancestor]
                    )
                    self.missed_mappings[current_old_index].add(older_ancestor_token)
            # update existing track info
            ancestor_track_info = self.track_info_mapping.get(parent_index)
            if ancestor_track_info is not None:
                this_item = ancestor_track_info.pop()
                if this_item.same_index_as is None:
                    inform_about = {self.current_track_token}
                    inform_same_remaining_position_about_index(
                        ancestor_track_info, this_item, inform_about
                    )

    def __init__(
        self,
        discretization: Discretization,
        starting_index: int,
        markers: MarkersMapProxyType,
    ):
        self.markers = markers
        self._discretization = discretization
        # iterates a modified version of the descriptor, incorporating the markers knowledge
        # and keeping track of the ancestry
        self._branch_state = AncestryBranch._AncestryBranchState(
            discretization, starting_index
        )
        self._mapping_state = AncestryBranch._AncestryMappingState(
            discretization, self._branch_state
        )
        assert (
            len(self._mapping_state.ancestry)
            == self._branch_state.initial_branch_depth - 1
        )

    def get_current_location_info(
        self,
    ) -> tuple[int, TrackToken, ba.frozenbitarray, MarkerType]:
        # get the currently desired location info
        current_old_index = 0
        intermediate_generation: set[int] = set()
        exact = True
        if len(self._branch_state) > 0:  # if not at root
            modified_dimensionwise_positions = location_code_from_history(
                self._branch_state.history_of_binary_positions,
                self._branch_state.history_of_level_increments,
            )
            current_old_index, intermediate_generation, exact = _find_next_twig(
                self._discretization,
                self.markers,
                modified_dimensionwise_positions,
                self._mapping_state.ancestry[-1],
            )
        next_refinement = _refinement_with_marker_applied(
            self._discretization.descriptor[current_old_index],
            next_marker := self.markers[current_old_index],
        )

        current_track_token = self._mapping_state.update(
            self._discretization,
            current_old_index,
            self.markers,
            next_marker,
            intermediate_generation,
            exact,
        )
        return (
            current_old_index,
            current_track_token,
            next_refinement,
            next_marker,
        )

    class WeAreDoneAndHereAreTheMissingRelationships(Exception):
        def __init__(
            self,
            mapping: dict[int, set[TrackToken]],
        ):
            self.missing_mapping = mapping
            super().__init__()

    def advance(self) -> None:
        try:
            current_modified_branch_depth = self._branch_state.advance()
        except IndexError as e:
            mapping = self._gather_remaining_mappings()
            raise AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships(
                mapping
            ) from e

        self._mapping_state.ancestry = self._mapping_state.ancestry[
            : current_modified_branch_depth - 1
        ]

    def grow(self, next_refinement: ba.frozenbitarray) -> None:
        self._branch_state.grow(next_refinement)

    def _gather_remaining_mappings(self) -> dict[int, set[TrackToken]]:
        # check if all relationships from coarsening tracking are exhausted
        mapping: defaultdict[int, set[TrackToken]] = self._mapping_state.missed_mappings
        hint_previous_branch = None
        dimensionality = self._discretization.descriptor.get_num_dimensions()
        for key, track_info in sorted(self._mapping_state.track_info_mapping.items()):
            ancestor_branch = self._discretization.descriptor.get_branch(
                key, is_box_index=False, hint_previous_branch=hint_previous_branch
            )[0]
            hint_previous_branch = (key, ancestor_branch)
            ancestor_location_code = branch_to_location_code(
                ancestor_branch, self._discretization._linearization
            )
            unresolved_coarsen_mask = track_info[0].unresolved_coarsen_mask
            unresolved_iterable = None
            if unresolved_coarsen_mask is not None:
                # shorten the ancestor location code for each unresolved coarsen dimension
                for d, b in enumerate(unresolved_coarsen_mask):
                    if b:
                        ancestor_location_code[d].pop()
                unresolved_iterable = binary_or_none_generator(
                    bitmask_to_indices(unresolved_coarsen_mask),
                    dimensionality,
                )
                # todo get analytically
                num_to_drop = 2**dimensionality - len(track_info)
                for _ in range(num_to_drop):
                    next(unresolved_iterable)
            current_refinement_dimensions = bitmask_to_indices(
                self._discretization.descriptor[key]
            )
            while len(track_info) > 0:
                index = track_info.pop()
                # get their indices in the old discretization by their location code
                missed_descendant_location_code = [
                    a.copy() for a in ancestor_location_code
                ]
                for d in current_refinement_dimensions:
                    if unresolved_iterable is not None:
                        unresolved_to_append = next(unresolved_iterable)
                        for ud in range(len(unresolved_to_append)):
                            if unresolved_to_append[ud] is None:
                                continue
                            missed_descendant_location_code[ud].append(
                                unresolved_to_append[ud]
                            )
                    missed_descendant_location_code[d].append(index.local_position[d])

                missed_descendant_index = self._discretization.get_index_from_location_code(
                    missed_descendant_location_code, get_box=False  # type: ignore
                )
                if index.same_index_as is None:
                    map_to = set(self._mapping_state.old_indices_map_track_tokens[key])
                else:
                    map_to = index.same_index_as

                assert isinstance(missed_descendant_index, int)
                mapping[missed_descendant_index].update(map_to)
            assert (  # assert exhausted
                unresolved_iterable is None or next(unresolved_iterable, None) is None
            )

        return mapping


def _refinement_with_marker_applied(
    refinement: ba.frozenbitarray,
    marker: MarkerType,
) -> ba.frozenbitarray:
    if refinement.count() == 0:
        assert np.all(marker > -1)
        return refinement

    positive = ba.bitarray([1 if m > 0 else 0 for m in marker])
    assert (refinement & positive).count() == 0
    refinement_modified = refinement | positive

    negative = ba.bitarray([1 if m < 0 else 0 for m in marker])
    assert (~refinement & negative).count() == 0
    refinement_modified ^= negative
    return ba.frozenbitarray(refinement_modified)


def _is_old_index_now_at_or_containing_location_code(
    discretization: Discretization,
    markers: MarkersMapProxyType,
    desired_dimensionwise_positions: LocationCode,
    parent_of_next_refinement: int,
    parent_branch: Branch,
    old_index: int,
) -> tuple[bool, LocationCode]:
    """Check whether old_index is now at or containing the next hyperrectangular location code.
    Args:
        discretization (Discretization): the old discretization we're referring to
        markers (MarkersMapProxyType): markers that should be applied to the old discretization
        desired_dimensionwise_positions (LocationCode): the location code we're looking for next
        parent_of_next_refinement (int): the current parent or other ancestor of the index we're looking for
        parent_branch (Branch): the branch corresponding to the parent_of_next_refinement
        old_index (int): the old index we're checking

    Returns:
        bool         : whether old_index is now at or containing the location code
        LocationCode : the location code of the old_index
    """
    descriptor = discretization.descriptor
    old_index_branch, _ = descriptor.get_branch(
        old_index,
        is_box_index=False,
        hint_previous_branch=(parent_of_next_refinement, parent_branch),
    )
    old_index_dimensionwise_positions = location_code_from_branch(
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


def _old_node_will_be_in_new_descriptor(
    descriptor: RefinementDescriptor,
    old_index: int,
    markers: MarkersMapProxyType,
) -> bool:
    """Check whether the node at old_index will be contained in the new descriptor after applying markers.

    Returns:
        bool: True if the node will be contained, False if it will be coarsened away.
    """
    future_refinement = _refinement_with_marker_applied(
        descriptor[old_index], marker := markers[old_index]
    )
    return np.min(marker) >= 0 or future_refinement != descriptor.d_zeros  # type: ignore


def _find_next_twig(
    discretization: Discretization,
    markers: MarkersMapProxyType,
    desired_dimensionwise_positions: LocationCode,
    parent_of_next_refinement: int,
) -> tuple[int, set[int], bool]:
    """Get the (old) tree node corresponding to the location code, and any nodes encountered on the way.
    Args:
        discretization (Discretization): the old discretization we're referring to
        markers (MarkersMapProxyType): refinement markers that should be applied to the discretization
        desired_dimensionwise_positions (list[ba.bitarray]): the location code we're looking for next
        parent_of_next_refinement (int): parent or other ancestor of the index we're looking for
    Returns:
        tuple[int, set[int], bool]:
            tuple consisting of the (old) node index
            and a set of (otherwise forgotten) intermediate nodes
            and a bool indicating whether the node is exactly at the location code
    """
    descriptor = discretization.descriptor
    parent_branch, _ = descriptor.get_branch(parent_of_next_refinement, False)
    children = descriptor.get_children(parent_of_next_refinement, parent_branch)
    intermediate_generation: set[int] = set()
    while True:
        for child in children:
            part_of_history, child_dimensionwise_positions = (
                _is_old_index_now_at_or_containing_location_code(
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

            location_code_matches = (
                child_dimensionwise_positions == desired_dimensionwise_positions
            )
            if _old_node_will_be_in_new_descriptor(descriptor, child, markers):
                # we found the next twig
                return child, intermediate_generation, location_code_matches

            # else it's a coarsened node and we can see if there is a matching child
            children_of_coarsened = descriptor.get_children(child)
            if location_code_matches:
                # this means that its former children are now gone and need to be mapped to this child's index
                for child_of_coarsened in children_of_coarsened:
                    # need to append the binarized index of the child, broadcast to split dimensions
                    intermediate_generation.add(child_of_coarsened)
                return child, intermediate_generation, True
            else:
                # else, it's an ancestor node that's going to disappear
                # -> restart loop with new children and remember this one
                intermediate_generation.add(child)
                children = children_of_coarsened
