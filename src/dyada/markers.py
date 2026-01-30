# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import defaultdict
import numpy as np
import numpy.typing as npt
from types import MappingProxyType
from typing import TypeAlias

MarkerType: TypeAlias = npt.NDArray[np.int8]
MarkersType: TypeAlias = defaultdict[int, npt.NDArray[np.int8]]
MarkersMapProxyType: TypeAlias = MappingProxyType[int, npt.NDArray[np.int8]]


def filter_markers_by_min_index(
    markers: MarkersType | MarkersMapProxyType, min_index: int
) -> MarkersMapProxyType:
    # filter the markers to the current interval
    filtered_markers = {k: v for k, v in markers.items() if k >= min_index}
    return MarkersMapProxyType(filtered_markers)


def get_next_largest_markered_index(
    markers: MarkersType | MarkersMapProxyType, min_index: int
) -> int:
    return min(
        filter_markers_by_min_index(markers, min_index).keys(),
        default=-1,
    )


def get_defaultdict_for_markers(
    num_dimensions: int,
) -> MarkersType:
    def get_d_zeros_as_array():
        return np.zeros(num_dimensions, dtype=np.int8)

    return MarkersType(get_d_zeros_as_array)
