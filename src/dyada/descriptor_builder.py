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
    nd = base.get_num_dimensions()
    num_boxes = base.get_num_boxes()

    for base_box_idx, sub in sub_descriptors.items():
        if not 0 <= base_box_idx < num_boxes:
            raise ValueError(f"Box index {base_box_idx} out of range [0, {num_boxes})")
        if sub.get_num_dimensions() != nd:
            raise ValueError(
                f"Sub-descriptor at box {base_box_idx} has {sub.get_num_dimensions()} "
                f"dimensions, expected {nd}"
            )

    # box-index keys to hierarchical-index keys
    hier_subs = {
        base.to_hierarchical_index(base_box_idx): (base_box_idx, sub)
        for base_box_idx, sub in sub_descriptors.items()
    }

    builder = DescriptorBuilder(base)
    node_mappings: dict[int, IndexMapping] = {}

    for base_node, ref in enumerate(base):
        if base_node in hier_subs:
            base_box_idx, sub = hier_subs[base_node]
            new_start = len(builder)
            builder.emit_for(base_node, sub._data)
            node_mappings[base_box_idx] = [
                {new_start + sub_node} for sub_node in range(len(sub))
            ]
        else:
            builder.emit_and_track(base_node, ref)

    desc = builder.build()
    if __debug__:
        validate_descriptor(desc)
    return desc, builder.mapping, node_mappings


def compose_grid(
    grid_levels: Sequence[int],
    sub_descriptors: Sequence[RefinementDescriptor | None],
) -> tuple[RefinementDescriptor, dict[int, IndexMapping]]:
    """Build a descriptor from sub-descriptors arranged on a regular grid.

    The base grid has level=grid_levels.
    Sub-descriptors are given as a flat sequence in Fortran order.

    Returns:
        A tuple of ``(combined_descriptor, node_mappings)`` where
        ``node_mappings[flat_idx]`` maps each sub-descriptor node index
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
