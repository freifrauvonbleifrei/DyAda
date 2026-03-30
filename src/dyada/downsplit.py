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
    planned_downsplits: list[tuple[int, ba.bitarray]],
) -> dict[int, ba.bitarray]:
    merged: dict[int, ba.bitarray] = {}
    for parent_index, dimensions_to_downsplit in planned_downsplits:
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


def _can_absorb(
    members: list[tuple[ba.frozenbitarray, int, int]],
    dims_to_split: ba.bitarray,
    descriptor: RefinementDescriptor,
) -> bool:
    """Check if grouped children can be absorbed into a single merged node."""
    old_refs = [descriptor[m[1]] for m in members]
    return (
        len(set(ba.frozenbitarray(r) for r in old_refs)) == 1
        and not (old_refs[0] & dims_to_split).any()
        and old_refs[0].count() > 0
    )


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

    merged_downsplits = _merged_planned_downsplits(planned_downsplits)
    if not merged_downsplits:
        return discretization, [{i} for i in range(len(descriptor))]

    # Single forward DFS pass: walk old descriptor, emit new descriptor.
    new_data = ba.bitarray()
    mapping: list[set[int]] = [set() for _ in range(len(descriptor))]

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

    def _walk(old_start: int, size: int) -> None:
        """Copy or process a subtree, handling nested downsplits."""
        old_end = old_start + size
        i = old_start
        while i < old_end:
            if i in merged_downsplits:
                i = _process_downsplit(i)
            else:
                run_end = i + 1
                while run_end < old_end and run_end not in merged_downsplits:
                    run_end += 1
                _copy_range(i, run_end)
                i = run_end

    def _emit_absorbed(
        members: list[tuple[ba.frozenbitarray, int, int]],
        parent_old_idx: int,
        dims_to_split: ba.bitarray,
    ) -> None:
        """Emit a merged node absorbing the intermediate and its children.

        The children's refinement is combined with the pushed-down dims.
        Grandchildren are interleaved in merged Morton order.
        """
        child_ref = descriptor[members[0][1]]
        merged_ref = ba.bitarray(child_ref) | dims_to_split
        new_merged_idx = _emit(merged_ref)
        _track(parent_old_idx, new_merged_idx)
        for _, old_start, _ in members:
            _track(old_start, new_merged_idx)

        # Collect grandchildren keyed by full position under merged_ref
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

        # Emit in merged order
        num_merged_gc = get_num_children_from_refinement(merged_ref)
        for m in range(num_merged_gc):
            m_pos = ba.frozenbitarray(
                linearization.get_binary_position_from_index((m,), (merged_ref,))
            )
            gc_old, gc_size = gc_entries[m_pos]
            _walk(gc_old, gc_size)

    def _emit_intermediate(
        members: list[tuple[ba.frozenbitarray, int, int]],
        parent_old_idx: int,
        dims_to_split: ba.bitarray,
    ) -> None:
        """Emit an intermediate node followed by each child subtree."""
        _track(parent_old_idx, _emit(ba.bitarray(dims_to_split)))
        for _, old_start, sz in members:
            _walk(old_start, sz)

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

        # Map each child's local position to (old_start, subtree_size)
        child_ranges = descriptor.get_child_ranges(old_idx)
        pos_to_info: dict[ba.frozenbitarray, tuple[int, int]] = {}
        for child_local_idx, (start, end) in enumerate(child_ranges):
            pos = ba.frozenbitarray(
                linearization.get_binary_position_from_index(
                    (child_local_idx,), (parent_ref,)
                )
            )
            pos_to_info[pos] = (start, end - start)
        subtree_end = child_ranges[-1][1]

        # Group children by remaining-position bits (dims NOT being pushed down).
        # Pass remaining_ref as sort_dimensions so the tracker sorts by remaining
        # bits first — making same-group members adjacent in pop order.
        tracker = get_initial_child_grouping(
            ba.frozenbitarray(parent_ref),
            ba.frozenbitarray(remaining_ref),
            linearization=linearization,
        )

        current_members: list[tuple[ba.frozenbitarray, int, int]] = []

        def _flush_group() -> None:
            if not current_members:
                return
            if _can_absorb(current_members, dims_to_split, descriptor):
                _emit_absorbed(current_members, old_idx, dims_to_split)
            else:
                _emit_intermediate(current_members, old_idx, dims_to_split)

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

    # Walk the full descriptor
    _walk(0, len(descriptor))

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
