# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later


import bitarray as ba
from typing import Sequence

from dyada.descriptor import RefinementDescriptor, validate_descriptor
from dyada.linearization import grid_coord_to_z_index, flat_to_coord
from dyada.mappings import IndexMapping


class DescriptorBuilder:
    """Incrementally builds a new RefinementDescriptor with index tracking"""

    def __init__(self, source: RefinementDescriptor):
        self._nd = source.get_num_dimensions()
        self._source = source
        self.data = ba.bitarray()
        self.mapping: IndexMapping = [set() for _ in range(len(source))]

    def __len__(self) -> int:
        return len(self.data) // self._nd

    def emit(self, ref_bits: ba.bitarray | ba.frozenbitarray) -> int:
        """Emit one node, return its new index."""
        new_idx = len(self)
        self.data.extend(ref_bits)
        return new_idx

    def track(self, old_idx: int, new_idx: int) -> None:
        """Record that old_idx maps to new_idx."""
        self.mapping[old_idx].add(new_idx)

    def emit_and_track(
        self, old_idx: int, ref_bits: ba.bitarray | ba.frozenbitarray
    ) -> int:
        """Emit one node and track the mapping. Returns new index."""
        new_idx = self.emit(ref_bits)
        self.track(old_idx, new_idx)
        return new_idx

    def emit_for(self, old_idx: int, extension: ba.bitarray) -> None:
        """Emit multiple nodes, all mapping back to one old index."""
        for new_idx in range(len(self), len(self) + len(extension) // self._nd):
            self.mapping[old_idx].add(new_idx)
        self.data.extend(extension)

    def copy_range(self, old_start: int, old_end: int) -> None:
        """Copy a contiguous range of nodes from source, tracking 1:1."""
        new_start = len(self)
        self.data.extend(self._source._data[old_start * self._nd : old_end * self._nd])
        for offset in range(old_end - old_start):
            self.mapping[old_start + offset].add(new_start + offset)

    def build(self) -> RefinementDescriptor:
        """Construct the final RefinementDescriptor."""
        desc = RefinementDescriptor(self._nd)
        desc._data = self.data
        return desc


def compose_descriptors(
    base: RefinementDescriptor,
    sub_descriptors: dict[int, RefinementDescriptor],
) -> tuple[RefinementDescriptor, IndexMapping, dict[int, IndexMapping]]:
    """Build a descriptor by grafting sub-descriptors into leaves of a base.

    Returns:
        A tuple of ``(combined_descriptor, base_mapping, sub_mappings)``
    """
    for box_idx, sub in sub_descriptors.items():
        if not 0 <= box_idx < base.get_num_boxes():
            raise ValueError(
                f"Box index {box_idx} out of range [0, {base.get_num_boxes()})"
            )
        if sub.get_num_dimensions() != base.get_num_dimensions():
            raise ValueError(
                f"Sub-descriptor at box {box_idx} has "
                f"{sub.get_num_dimensions()} dimensions, expected {base.get_num_dimensions()}"
            )

    # box-index keys to hierarchical-index keys
    hier_subs = {
        base.to_hierarchical_index(base_box_idx): (base_box_idx, sub)
        for base_box_idx, sub in sub_descriptors.items()
    }

    builder = DescriptorBuilder(base)
    sub_mappings: dict[int, IndexMapping] = {}

    for base_node, ref in enumerate(base):
        if base_node in hier_subs:
            base_box_idx, sub = hier_subs[base_node]
            new_start = len(builder)
            builder.emit_for(base_node, sub._data)
            sub_mappings[base_box_idx] = [
                {i} for i in range(new_start, new_start + len(sub))
            ]
        else:
            builder.emit_and_track(base_node, ref)

    desc = builder.build()
    if __debug__:
        validate_descriptor(desc)
    return desc, builder.mapping, sub_mappings


def decompose_descriptor(
    source: RefinementDescriptor,
    cut_indices: Sequence[int],
) -> tuple[
    RefinementDescriptor,
    list[RefinementDescriptor],
    IndexMapping,
    list[dict[int, set[int]]],
]:
    """Inverse of :func:`compose_descriptors` — extract sub-descriptors rooted
    at the given hierarchical indices.

    Each ``cut_indices[k]`` must be a valid hierarchical index of ``source``.
    Its full subtree is removed and returned as ``sub_descriptors[k]``;
    in ``parent_descriptor`` the cut node becomes a leaf.
    Cuts must not overlap (no cut index may be inside another cut's subtree)
    and must be unique.

    Returns:
        ``(parent_descriptor, sub_descriptors, parent_mapping, sub_mappings)``
    """
    n = len(source)
    d_zeros = source.d_zeros

    if len(set(cut_indices)) != len(cut_indices):
        raise ValueError("cut_indices contains duplicates")
    for ci in cut_indices:
        if not 0 <= ci < n:
            raise ValueError(f"cut index {ci} out of range [0, {n})")
        if source[ci] == d_zeros:
            raise ValueError(
                f"cut index {ci} is a leaf; cannot extract an empty subtree"
            )

    # Walk cuts in ascending order, compute each subtree's end, reject overlaps.
    cut_spans: dict[int, int] = {}  # insertion-ordered ↔ ascending
    prev_end = 0
    for ci in sorted(cut_indices):
        if ci < prev_end:
            raise ValueError(
                f"cut index {ci} lies inside another cut's subtree "
                f"(ending at {prev_end})"
            )
        prev_end = cut_spans[ci] = source.get_child_ranges(ci)[-1][1]

    parent_builder = DescriptorBuilder(source)
    sub_builders = {ci: DescriptorBuilder(source) for ci in cut_spans}

    prev = 0
    for ci, ce in cut_spans.items():
        if prev < ci:
            parent_builder.copy_range(prev, ci)
        leaf_idx = parent_builder.emit_and_track(ci, d_zeros)
        # Every index inside the cut subtree collapses into this new leaf.
        for s in parent_builder.mapping[ci + 1 : ce]:
            s.add(leaf_idx)
        sub_builders[ci].copy_range(ci, ce)
        prev = ce
    if prev < n:
        parent_builder.copy_range(prev, n)

    parent_desc = parent_builder.build()
    sub_descriptors = [sub_builders[ci].build() for ci in cut_indices]
    if __debug__:
        validate_descriptor(parent_desc)
        for sd in sub_descriptors:
            validate_descriptor(sd)
    # Source indices ci..ce-1 map 1:1 to sub indices 0..ce-ci-1.
    sub_mappings = [
        {j: {j - ci} for j in range(ci, cut_spans[ci])} for ci in cut_indices
    ]
    return parent_desc, sub_descriptors, parent_builder.mapping, sub_mappings


def compose_grid(
    grid_levels: Sequence[int],
    sub_descriptors: Sequence[RefinementDescriptor | None],
) -> tuple[RefinementDescriptor, dict[int, IndexMapping]]:
    """Build a descriptor from sub-descriptors arranged on a regular grid.

    The base grid has level=grid_levels.
    Sub-descriptors are given as a flat sequence in Fortran order.

    Returns:
        A tuple of ``(combined_descriptor, sub_mappings)`` where
        ``sub_mappings[flat_idx]`` maps each sub-descriptor node index
        to its set of combined-descriptor node indices.
    """
    nd = len(grid_levels)
    grid_shape = tuple(1 << lv for lv in grid_levels)
    total_cells = 1
    for s in grid_shape:
        total_cells *= s

    if len(sub_descriptors) != total_cells:
        raise ValueError(
            f"Expected {total_cells} sub-descriptors for grid "
            f"{'x'.join(str(s) for s in grid_shape)}, got {len(sub_descriptors)}"
        )

    # Map Fortran-order flat index to grid coordinate, then to Z order box index.
    z_subs: dict[int, RefinementDescriptor] = {}
    flat_to_z: dict[int, int] = {}
    for flat_idx, sub in enumerate(sub_descriptors):
        if sub is None:
            continue
        if sub.get_num_dimensions() != nd:
            raise ValueError(
                f"Sub-descriptor at flat index {flat_idx} has "
                f"{sub.get_num_dimensions()} dimensions, expected {nd}"
            )
        coord = flat_to_coord(flat_idx, grid_shape)
        z_idx = grid_coord_to_z_index(coord, grid_levels)
        z_subs[z_idx] = sub
        flat_to_z[flat_idx] = z_idx

    base = RefinementDescriptor(nd, list(grid_levels))
    desc, _, z_mappings = compose_descriptors(base, z_subs)

    # Re-key mappings from Z-order box index to flat index.
    flat_mappings = {
        flat_idx: z_mappings[z_idx] for flat_idx, z_idx in flat_to_z.items()
    }
    return desc, flat_mappings
