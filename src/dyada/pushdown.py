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
from dyada.mappings import merge_mappings


def _group_children_for_pushdown(
    linearization,
    parent_ref: ba.bitarray,
    pushed_dim: int,
    child_ranges: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Group child ranges into (zero_d_range, one_d_range) pairs for each new child slot."""
    remaining_ref = ba.bitarray(parent_ref)
    remaining_ref[pushed_dim] = 0
    num_new_children = 2 ** remaining_ref.count() if remaining_ref.count() > 0 else 1
    groups: list[list[tuple[int, int] | None]] = [[None, None] for _ in range(num_new_children)]
    for child_pos, child_range in enumerate(child_ranges):
        child_bits = linearization.get_binary_position_from_index([child_pos], [parent_ref])
        pushed_bit = int(child_bits[pushed_dim])
        remaining_bits = child_bits.copy()
        remaining_bits[pushed_dim] = 0
        group_pos = (
            linearization.get_index_from_binary_position(remaining_bits, [], [remaining_ref])
            if remaining_ref.count() > 0
            else 0
        )
        groups[group_pos][pushed_bit] = child_range
    return [(g[0], g[1]) for g in groups]  # type: ignore[return-value]


def _apply_single_dim_pushdown(
    descriptor: RefinementDescriptor,
    linearization,
    parent_index: int,
    pushed_dim: int,
) -> tuple[RefinementDescriptor, list[set[int]]]:
    """Rewrite the descriptor applying a one-dimension pushdown at parent_index.

    Returns the new descriptor and a step mapping (current patch index →
    set of new patch indices) for composing into the cumulative patch mapping.
    """
    nd = descriptor.get_num_dimensions()
    n = len(descriptor)

    parent_ref = ba.bitarray(descriptor[parent_index])
    remaining_ref = parent_ref.copy()
    remaining_ref[pushed_dim] = 0
    if remaining_ref.count() == 0:
        raise ValueError(
            "Pushdown of a node refined only in pushed dimensions is not possible"
        )

    child_ranges = descriptor.get_child_ranges(parent_index)
    subtree_end = child_ranges[-1][1]
    groups = _group_children_for_pushdown(linearization, parent_ref, pushed_dim, child_ranges)

    # Build the replacement section starting at parent_index.
    # rel_to_old maps (relative offset in new section) → set of old absolute indices.
    new_section = ba.bitarray()
    rel_to_old: list[tuple[int, set[int]]] = []

    new_section.extend(remaining_ref)
    rel_to_old.append((0, {parent_index}))
    new_rel = 1

    for (r0, r1), (r2, r3) in groups:
        ref_zero = ba.bitarray(descriptor[r0])
        ref_one = ba.bitarray(descriptor[r2])
        can_merge = ref_zero == ref_one and not ref_zero[pushed_dim]

        if not can_merge:
            # Insert an intermediate node that splits on pushed_dim only.
            interm_ref = ba.bitarray(descriptor.d_zeros)
            interm_ref[pushed_dim] = 1
            new_section.extend(interm_ref)
            rel_to_old.append((new_rel, {parent_index}))
            new_rel += 1
            # Copy the zero_d and one_d subtrees verbatim.
            for offset, old in enumerate(range(r0, r1)):
                rel_to_old.append((new_rel + offset, {old}))
            new_section.extend(descriptor._data[r0 * nd : r1 * nd])
            new_rel += r1 - r0
            for offset, old in enumerate(range(r2, r3)):
                rel_to_old.append((new_rel + offset, {old}))
            new_section.extend(descriptor._data[r2 * nd : r3 * nd])
            new_rel += r3 - r2
        else:
            # Merge the two children into one node that also refines pushed_dim.
            merged_ref = ref_zero.copy()
            merged_ref[pushed_dim] = 1
            new_section.extend(merged_ref)
            merged_old: set[int] = {parent_index}
            if not descriptor.is_box(r0):
                # Non-leaf children are absorbed into the merged node.
                merged_old |= {r0, r2}
            rel_to_old.append((new_rel, merged_old))
            new_rel += 1

            if descriptor.is_box(r0):
                # Leaf children stay as explicit leaves under the merged node.
                new_section.extend(descriptor._data[r0 * nd : r1 * nd])
                rel_to_old.append((new_rel, {r0}))
                new_rel += 1
                new_section.extend(descriptor._data[r2 * nd : r3 * nd])
                rel_to_old.append((new_rel, {r2}))
                new_rel += 1
            else:
                # Non-leaf children: interleave grandchildren in merged Morton order.
                gc_zero = descriptor.get_child_ranges(r0)
                gc_one = descriptor.get_child_ranges(r2)
                num_merged_gc = get_num_children_from_refinement(merged_ref)
                for m in range(num_merged_gc):
                    m_bits = linearization.get_binary_position_from_index([m], [merged_ref])
                    m_pushed_bit = int(m_bits[pushed_dim])
                    m_remaining = m_bits.copy()
                    m_remaining[pushed_dim] = 0
                    old_gc_pos = linearization.get_index_from_binary_position(
                        m_remaining, [], [ref_zero]
                    )
                    gc_start, gc_end = (gc_zero if m_pushed_bit == 0 else gc_one)[old_gc_pos]
                    for offset, old in enumerate(range(gc_start, gc_end)):
                        rel_to_old.append((new_rel + offset, {old}))
                    new_section.extend(descriptor._data[gc_start * nd : gc_end * nd])
                    new_rel += gc_end - gc_start

    # Assemble the new descriptor.
    new_data = descriptor._data[: parent_index * nd].copy()
    new_data.extend(new_section)
    new_data.extend(descriptor._data[subtree_end * nd :])
    new_desc = RefinementDescriptor(nd)
    new_desc._data = new_data

    # Build the step mapping (current desc → new desc).
    delta = new_rel - (subtree_end - parent_index)
    step_mapping: list[set[int]] = [set() for _ in range(n)]
    for i in range(parent_index):
        step_mapping[i].add(i)
    for rel, old_set in rel_to_old:
        abs_new = parent_index + rel
        for old in old_set:
            step_mapping[old].add(abs_new)
    for i in range(subtree_end, n):
        step_mapping[i].add(i + delta)

    return new_desc, step_mapping


def _merged_planned_pushdowns(
    descriptor: RefinementDescriptor,
    planned_pushdowns: list[tuple[int, ba.bitarray]],
) -> dict[int, ba.bitarray]:
    num_dimensions = descriptor.get_num_dimensions()
    merged: dict[int, ba.bitarray] = {}
    for parent_index, dimensions_to_push_down in planned_pushdowns:
        if parent_index < 0 or parent_index >= len(descriptor):
            raise IndexError("parent_index out of range")
        if len(dimensions_to_push_down) != num_dimensions:
            raise ValueError(
                "dimensions_to_push_down length does not match discretization dimensionality"
            )
        existing = merged.get(parent_index)
        if existing is None:
            merged[parent_index] = dimensions_to_push_down.copy()
        else:
            merged[parent_index] |= dimensions_to_push_down
    return merged


def apply_planned_pushdowns(
    discretization: Discretization,
    planned_pushdowns: list[tuple[int, ba.bitarray]],
    track_mapping: Literal["boxes", "patches"],
) -> tuple[Discretization, list[set[int]]]:
    descriptor = discretization.descriptor
    nd = descriptor.get_num_dimensions()
    linearization = discretization._linearization

    merged_pushdowns = _merged_planned_pushdowns(descriptor, planned_pushdowns)

    current_descriptor = descriptor
    cumulative_mapping: list[set[int]] = [{i} for i in range(len(descriptor))]

    # Process pushdowns deepest-first (reverse preorder index) so that child
    # pushdowns are applied before their ancestor's pushdown restructures the tree.
    for parent_index in sorted(merged_pushdowns.keys(), reverse=True):
        dims_to_push = merged_pushdowns[parent_index]
        for pushed_dim in range(nd):
            if not dims_to_push[pushed_dim]:
                continue
            current_descriptor, step_mapping = _apply_single_dim_pushdown(
                current_descriptor, linearization, parent_index, pushed_dim
            )
            cumulative_mapping = merge_mappings(cumulative_mapping, step_mapping)

    new_descriptor = current_descriptor
    new_discretization = Discretization(discretization._linearization, new_descriptor)

    if track_mapping == "patches":
        mapping = cumulative_mapping
    elif track_mapping == "boxes":
        mapping = hierarchical_to_box_index_mapping(
            cumulative_mapping, descriptor, new_descriptor
        )
    else:
        raise ValueError(
            "track_mapping must be either 'boxes' or 'patches', got " + str(track_mapping)
        )
    return new_discretization, mapping
