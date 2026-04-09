# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
import pytest

from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization
from dyada.descriptor_builder import (
    compose_descriptors,
    compose_grid,
)


def _disc(desc):
    return Discretization(MortonOrderLinearization(), desc)


def test_multiple_leaves_2d():
    base = RefinementDescriptor(2, [1, 1])
    sub_a = RefinementDescriptor(2, [1, 1])
    sub_b = RefinementDescriptor(2, [1, 1])
    combined = compose_descriptors(base, {0: sub_a, 2: sub_b})
    assert combined.get_num_boxes() == 10
    assert (
        str(_disc(combined))
        == """\
_________
|_|_|   |
|_|_|___|
|_|_|   |
|_|_|___|"""
    )


def test_all_leaves_2d():
    base = RefinementDescriptor(2, [1, 1])
    subs = {i: RefinementDescriptor(2, [1, 1]) for i in range(4)}
    combined = compose_descriptors(base, subs)
    # Every leaf replaced: same as 2-level uniform
    assert combined.get_num_boxes() == 16
    expected = RefinementDescriptor(2, [2, 2])
    assert combined._data == expected._data


def test_empty_sub_descriptors():
    base = RefinementDescriptor(2, [1, 1])
    combined = compose_descriptors(base, {})
    assert combined._data == base._data


def test_anisotropic_sub_2d():
    base = RefinementDescriptor(2, [1, 1])
    sub = RefinementDescriptor.from_binary(2, ba.bitarray("10 00 00"))
    combined = compose_descriptors(base, {1: sub})
    assert combined.get_num_boxes() == 5
    assert (
        str(_disc(combined))
        == """\
_________
|___|___|
|___|_|_|"""
    )


def test_3d_anisotropic_sub():
    base = RefinementDescriptor(3, [1, 1, 1])
    sub = RefinementDescriptor.from_binary(3, ba.bitarray("100 000 000"))
    combined = compose_descriptors(base, {3: sub})
    assert combined.get_num_boxes() == 8 - 1 + 2  # 9


def test_nested_compose():
    """Composing twice with anisotropic subs."""
    base = RefinementDescriptor(2, [1, 1])
    sub_x = RefinementDescriptor(2, [2, 1])  # 4x2 = 8 boxes
    desc2 = compose_descriptors(base, {0: sub_x, 3: sub_x})
    assert desc2.get_num_boxes() == 4 - 2 + 8 + 8  # 18
    assert str(_disc(desc2)) == (
        "_________________\n"
        "|       |_|_|_|_|\n"
        "|_______|_|_|_|_|\n"
        "|_|_|_|_|       |\n"
        "|_|_|_|_|_______|"
    )
    sub_y = RefinementDescriptor(2, [1, 2])
    desc3 = compose_descriptors(desc2, {0: sub_y})
    assert str(_disc(desc3)) == (
        "_________________________________\n"
        "|               |   |   |   |   |\n"
        "|               |   |   |   |   |\n"
        "|               |   |   |   |   |\n"
        "|               |___|___|___|___|\n"
        "|               |   |   |   |   |\n"
        "|               |   |   |   |   |\n"
        "|               |   |   |   |   |\n"
        "|_______________|___|___|___|___|\n"
        "|   |   |   |   |               |\n"
        "|   |   |   |   |               |\n"
        "|   |   |   |   |               |\n"
        "|___|___|___|___|               |\n"
        "|_|_|   |   |   |               |\n"
        "|_|_|   |   |   |               |\n"
        "|_|_|   |   |   |               |\n"
        "|_|_|___|___|___|_______________|"
    )


def test_2d_single_cell():
    sub = RefinementDescriptor(2, [1, 1])
    combined = compose_grid([1, 1], [None, sub, None, None])
    assert (
        str(_disc(combined))
        == """\
_________
|   |   |
|___|___|
|   |_|_|
|___|_|_|"""
    )


def test_2d_grid_coordinates_match_spatial():
    sub = RefinementDescriptor(2, [1, 1])
    subs = [None] * 8
    subs[7] = sub
    combined = compose_grid([2, 1], subs)
    assert str(_disc(combined)) == (
        "_________________\n"
        "|   |   |   |_|_|\n"
        "|___|___|___|_|_|\n"
        "|   |   |   |   |\n"
        "|___|___|___|___|"
    )


def test_2d_ascii_multiple_cells():
    sub_x = RefinementDescriptor(2, [2, 1])
    sub_y = RefinementDescriptor(2, [1, 2])
    combined = compose_grid([1, 1], [sub_x, None, None, sub_y])
    assert str(_disc(combined)) == (
        "_________________\n"
        "|       |___|___|\n"
        "|       |___|___|\n"
        "|       |___|___|\n"
        "|_______|___|___|\n"
        "| | | | |       |\n"
        "|_|_|_|_|       |\n"
        "| | | | |       |\n"
        "|_|_|_|_|_______|"
    )


def test_2d_anisotropic_grid():
    sub = RefinementDescriptor(2, [1, 2])
    subs = [None] * 8
    subs[0] = sub
    combined = compose_grid([2, 1], subs)
    assert str(_disc(combined)) == (
        "_________________\n"
        "|   |   |   |   |\n"
        "|   |   |   |   |\n"
        "|   |   |   |   |\n"
        "|___|___|___|___|\n"
        "|_|_|   |   |   |\n"
        "|_|_|   |   |   |\n"
        "|_|_|   |   |   |\n"
        "|_|_|___|___|___|"
    )


@pytest.mark.parametrize("nd", [3, 4, 5])
def test_high_dim_grid(nd):
    """Compose an anisotropic sub-descriptor into an anisotropic base grid."""
    import random

    rng = random.Random(42 + nd)

    # Base: random permutation of [1, 2, ..., nd] as per-dim levels
    base_levels = list(range(1, nd + 1))
    rng.shuffle(base_levels)
    num_base_cells = 1
    for lv in base_levels:
        num_base_cells *= 1 << lv

    # Sub: anisotropic refinement (different level per dimension)
    sub_levels = list(range(1, nd + 1))
    sub = RefinementDescriptor(nd, sub_levels)
    sub_boxes = sub.get_num_boxes()

    # Place into a random cell
    cell_idx = rng.randrange(num_base_cells)
    subs = [None] * num_base_cells
    subs[cell_idx] = sub
    combined = compose_grid(base_levels, subs)
    assert combined.get_num_boxes() == num_base_cells - 1 + sub_boxes


def test_all_none():
    """All-None sequence produces the base grid."""
    combined = compose_grid([1, 1], [None, None, None, None])
    base = RefinementDescriptor(2, [1, 1])
    assert combined._data == base._data


def test_all_refined():
    sub = RefinementDescriptor(2, [1, 1])
    combined = compose_grid([1, 1], [sub, sub, sub, sub])
    expected = RefinementDescriptor(2, [2, 2])
    assert combined._data == expected._data


def test_wrong_length():
    sub = RefinementDescriptor(2, [1, 1])
    with pytest.raises(ValueError, match="Expected 4"):
        compose_grid([1, 1], [sub, None])


def test_dimension_mismatch():
    sub_3d = RefinementDescriptor(3, [1, 1, 1])
    with pytest.raises(ValueError, match="dimensions"):
        compose_grid([1, 1], [sub_3d, None, None, None])


def test_box_index_out_of_range():
    base = RefinementDescriptor(2, [1, 1])
    sub = RefinementDescriptor(2, [1, 1])
    with pytest.raises(ValueError, match="out of range"):
        compose_descriptors(base, {99: sub})


def test_negative_box_index():
    base = RefinementDescriptor(2, [1, 1])
    sub = RefinementDescriptor(2, [1, 1])
    with pytest.raises(ValueError, match="out of range"):
        compose_descriptors(base, {-1: sub})
