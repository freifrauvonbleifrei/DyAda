# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba

from dyada.descriptor import RefinementDescriptor
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
