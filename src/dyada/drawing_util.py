# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

try:
    from cmap import Colormap
except ImportError:
    warnings.warn("cmap not found, some plotting functions will not work")
from itertools import product
from string import ascii_uppercase
from typing import Sequence, Union, Mapping, Optional

from dyada.coordinates import (
    Coordinate,
    CoordinateInterval,
    get_coordinates_from_level_index,
)
from dyada.discretization import (
    Discretization,
)
from dyada.structure import depends_on_optional


def labels_from_discretization(
    discretization: Discretization, labels: Union[None, str, Sequence[str]]
):
    if labels == "patches":
        labels = []
        for i in range(len(discretization._descriptor)):
            if discretization._descriptor.is_box(i):
                labels.append(str(i))
    elif labels == "boxes":
        labels = [str(i) for i in range(len(discretization))]

    assert labels is None or len(labels) == discretization._descriptor.get_num_boxes()
    return labels


def cuboid_from_interval(
    interval: CoordinateInterval, projection: Sequence[int] = [0, 1, 2]
) -> tuple:
    lower = interval[0][projection]
    upper = interval[1][projection]
    # create vertices from lower and upper bounds
    vertices = list(product(*zip(lower, upper)))
    # define rectangular faces
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
    )
    edges = (
        *((0, 1), (2, 3), (4, 5), (6, 7)),
        *((0, 2), (1, 3), (4, 6), (5, 7)),
        *((0, 4), (1, 5), (2, 6), (3, 7)),
    )
    return vertices, faces, edges


def side_corners_generator(
    lower_cube_corner: Coordinate, upper_cube_corner: Coordinate
):
    # iterate the six sides of the cuboid
    # by always selecting four corners that have one coordinate in common
    corners = list(product(*zip(lower_cube_corner, upper_cube_corner)))
    for bound in [lower_cube_corner, upper_cube_corner]:
        for i, b in enumerate(bound):
            side_corners = list(filter(lambda c: c[i] == b, corners))
            assert len(side_corners) == 4
            # correct the order
            yield side_corners[0], side_corners[1], side_corners[3], side_corners[2]


def boxes_to_2d_ascii(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    resolution=(16, 8),
    projection: Sequence[int] = [0, 1],
    **kwargs,
) -> str:
    """
    returns an ASCII string visualization of a 2-D omnitree discretization
    using | and _ characters for cell boundaries.

    cells: list of ((x0, x1), (y0, y1)) dyadic rectangles
    resolution: grid resolution in (width, height) (must be power-of-two compatible)
    """

    # unit coordinates -> grid indices
    def to_idx_vertical(x):
        return int(round(x * resolution[1]))

    def to_idx_horizontal(x):
        return int(round(x * resolution[0]))

    # canvas with boundary slots
    W = resolution[0] + 1
    H = resolution[1] + 1
    canvas = [[" " for _ in range(W)] for _ in range(H)]
    for x in range(W):
        canvas[0][x] = "_"
        canvas[H - 1][x] = "_"
    for y in range(H):
        canvas[y][0] = "|"
        canvas[y][W - 1] = "|"

    # draw cell boundaries
    for interval in intervals:
        lower = interval[0][projection]
        upper = interval[1][projection]
        ix0 = to_idx_horizontal(lower[0])
        ix1 = to_idx_horizontal(upper[0])
        iy0 = to_idx_vertical(lower[1])
        iy1 = to_idx_vertical(upper[1])

        # Horizontal edges
        for x in range(ix0, ix1 + 1):
            canvas[H - iy0 - 1][x] = "_"
            canvas[H - iy1 - 1][x] = "_"

        # Vertical edges
        for y in range(iy0, iy1):
            canvas[H - y - 1][ix0] = "|"
            canvas[H - y - 1][ix1] = "|"

    return "\n".join("".join(row) for row in canvas)


@depends_on_optional("cmap")
def get_colors(num_colors: int, colormap_name="CET_R3"):
    cm = Colormap(colormap_name)
    for leaf in range(num_colors):
        colormapped = cm(leaf * 1.0 / num_colors)
        yield colormapped  # RGB values in [0, 1] range


def get_colors_byte(num_colors: int, colormap_name="CET_R3"):
    for color in get_colors(num_colors, colormap_name):
        color = [int(255 * c) for c in color]  # convert to [0, 255] range
        yield color


def letter_counter(length):
    """Generate up to `length` Excel-style letter labels (A, B, ..., Z, AA, AB, ...)."""
    count = 0
    size = 1
    while count < length:
        for combo in product(ascii_uppercase, repeat=size):
            yield "".join(combo)
            count += 1
            if count >= length:
                break
        size += 1
