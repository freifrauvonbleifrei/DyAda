# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Literal

import bitarray as ba

from dyada.descriptor import (
    RefinementDescriptor,
    get_num_children_from_refinement,
    hierarchical_to_box_index_mapping,
)
from dyada.discretization import Discretization
from dyada.linearization import (
    get_initial_child_grouping,
)


def _merged_planned_downsplits(
    descriptor: RefinementDescriptor,
    planned_downsplits: list[tuple[int, ba.bitarray]],
) -> dict[int, ba.bitarray]:
    num_dimensions = descriptor.get_num_dimensions()
    merged: dict[int, ba.bitarray] = {}
    for parent_index, dimensions_to_downsplit in planned_downsplits:
        if parent_index < 0 or parent_index >= len(descriptor):
            raise IndexError("parent_index out of range")
        if len(dimensions_to_downsplit) != num_dimensions:
            raise ValueError(
                "dimensions_to_downsplit length does not match discretization dimensionality"
            )
        existing = merged.get(parent_index)
        if existing is None:
            merged[parent_index] = dimensions_to_downsplit.copy()
        else:
            merged[parent_index] |= dimensions_to_downsplit
    return merged


def _subtree_size(descriptor: RefinementDescriptor, index: int) -> int:
    """Return the number of nodes in the subtree rooted at index (inclusive)."""
    ref = descriptor[index]
    it = descriptor.__iter__(start=index + 1)  # type: ignore
    _, size = descriptor.skip_to_next_neighbor(it, ref)
    return size


def apply_planned_downsplits(
    discretization: Discretization,
    planned_downsplits: list[tuple[int, ba.bitarray]],
    track_mapping: Literal["boxes", "patches"],
) -> tuple[Discretization, list[set[int]]]:
    """Apply all planned downsplits in a single forward pass over the descriptor.

    Uses the ChildGroupingTracker to group children of each downsplit node, then
    emits intermediate nodes and re-ordered children into the target descriptor.
    """
    descriptor = discretization.descriptor
    nd = descriptor.get_num_dimensions()
    linearization = discretization._linearization
    n = len(descriptor)

    merged_downsplits = _merged_planned_downsplits(descriptor, planned_downsplits)
    if not merged_downsplits:
        return discretization, [{i} for i in range(n)]

    # Single forward DFS pass: walk old descriptor, emit new descriptor.
    new_data = ba.bitarray()
    mapping: list[set[int]] = [set() for _ in range(n)]

    def _track(old_idx: int, new_idx: int) -> None:
        mapping[old_idx].add(new_idx)

    def _emit(ref_bits: ba.bitarray | ba.frozenbitarray) -> int:
        """Emit one node into the new descriptor, return its new index."""
        new_idx = len(new_data) // nd
        new_data.extend(ref_bits)
        return new_idx

    def _copy_range(old_start: int, old_end: int) -> None:
        """Copy a contiguous range of nodes from old to new descriptor."""
        new_start = len(new_data) // nd
        new_data.extend(descriptor._data[old_start * nd : old_end * nd])
        for offset in range(old_end - old_start):
            mapping[old_start + offset].add(new_start + offset)

    def _process_downsplit(old_idx: int) -> int:
        """Process a downsplit node, returning the old index after its subtree."""
        dims_to_split = merged_downsplits[old_idx]
        parent_ref = ba.bitarray(descriptor[old_idx])

        remaining_ref = parent_ref & ~dims_to_split

        if remaining_ref.count() == 0:
            raise ValueError(
                f"Downsplit at node {old_idx}: cannot push down all refined dims"
            )

        # Emit the parent with reduced ref
        _track(old_idx, _emit(remaining_ref))

        # Collect old children: (local_position, old_start, subtree_size)
        num_old_children = get_num_children_from_refinement(parent_ref)
        children_info: list[tuple[ba.frozenbitarray, int, int]] = []
        child_old = old_idx + 1
        for child_local_idx in range(num_old_children):
            pos = ba.frozenbitarray(
                linearization.get_binary_position_from_index(
                    (child_local_idx,), (parent_ref,)
                )
            )
            size = _subtree_size(descriptor, child_old)
            children_info.append((pos, child_old, size))
            child_old += size
        subtree_end = child_old

        # Group children by remaining-position bits (dims NOT being pushed down).
        # Pass remaining_ref as separated_mask so the tracker sorts by remaining
        # bits first — making same-group members adjacent in pop order.
        tracker = get_initial_child_grouping(
            ba.frozenbitarray(parent_ref),
            ba.frozenbitarray(remaining_ref),
            linearization=linearization,
        )

        pos_to_info = {info[0]: (info[1], info[2]) for info in children_info}

        # Process groups: same_as is None for the first member of each group
        current_members: list[tuple[ba.frozenbitarray, int, int]] = []

        def _flush_group():
            if not current_members:
                return
            members = current_members
            # Check if children can be merged (absorbed into one node)
            old_refs = [descriptor[m[1]] for m in members]
            can_merge = (
                len(set(ba.frozenbitarray(r) for r in old_refs)) == 1
                and not (old_refs[0] & dims_to_split).any()
                and old_refs[0].count() > 0
            )

            if can_merge:
                # Absorb: one merged node replaces intermediate + children
                merged_ref = ba.bitarray(old_refs[0]) | dims_to_split
                new_merged_idx = _emit(merged_ref)
                _track(old_idx, new_merged_idx)
                for _, old_start, _ in members:
                    _track(old_start, new_merged_idx)

                # Interleave grandchildren in merged Morton order
                gc_entries: dict[ba.frozenbitarray, tuple[int, int]] = {}
                for local_pos, old_start, _ in members:
                    old_child_ref = descriptor[old_start]
                    num_gc = get_num_children_from_refinement(old_child_ref)
                    gc_old = old_start + 1
                    for gc_idx in range(num_gc):
                        gc_pos = linearization.get_binary_position_from_index(
                            (gc_idx,), (old_child_ref,)
                        )
                        full_pos = ba.frozenbitarray(
                            (local_pos & dims_to_split) | (gc_pos & old_child_ref)
                        )
                        gc_size = _subtree_size(descriptor, gc_old)
                        gc_entries[full_pos] = (gc_old, gc_size)
                        gc_old += gc_size

                num_merged_gc = get_num_children_from_refinement(merged_ref)
                for m in range(num_merged_gc):
                    m_pos = ba.frozenbitarray(
                        linearization.get_binary_position_from_index(
                            (m,), (merged_ref,)
                        )
                    )
                    gc_old, gc_size = gc_entries[m_pos]
                    _copy_range(gc_old, gc_old + gc_size)

            else:
                # Cannot merge: emit intermediate, then each child subtree
                _track(old_idx, _emit(ba.bitarray(dims_to_split)))
                # Members are already in down-dim Morton order from the tracker
                for _, old_start, sz in members:
                    _copy_range(old_start, old_start + sz)

        current_key: ba.frozenbitarray | None = None
        while len(tracker) > 0:
            local_pos, _ = tracker.pop()
            old_start, size = pos_to_info[local_pos]
            remaining_key = ba.frozenbitarray(local_pos & remaining_ref)
            if remaining_key != current_key:
                _flush_group()
                current_members = []
                current_key = remaining_key
            current_members.append((local_pos, old_start, size))
        _flush_group()

        return subtree_end

    # Walk the full descriptor: copy contiguous non-downsplit regions in bulk,
    # process downsplit nodes via _process_downsplit.
    downsplit_indices = sorted(merged_downsplits.keys())
    old_idx = 0
    for ds_idx in downsplit_indices:
        if old_idx < ds_idx:
            _copy_range(old_idx, ds_idx)
        old_idx = _process_downsplit(ds_idx)
    if old_idx < n:
        _copy_range(old_idx, n)

    new_descriptor = RefinementDescriptor(nd)
    new_descriptor._data = new_data
    new_discretization = Discretization(linearization, new_descriptor)

    if track_mapping == "patches":
        return new_discretization, mapping
    elif track_mapping == "boxes":
        return new_discretization, hierarchical_to_box_index_mapping(
            mapping, descriptor, new_descriptor
        )
    else:
        raise ValueError(
            "track_mapping must be either 'boxes' or 'patches', got "
            + str(track_mapping)
        )
