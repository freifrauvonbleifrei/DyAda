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
    combined, _, mappings = compose_descriptors(base, {0: sub_a, 2: sub_b})
    assert combined.get_num_boxes() == 10
    assert sorted(mappings.keys()) == [0, 2]
    assert len(mappings[0]) == len(sub_a)
    assert len(mappings[2]) == len(sub_b)
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
    combined, _, _ = compose_descriptors(base, subs)
    assert combined.get_num_boxes() == 16
    expected = RefinementDescriptor(2, [2, 2])
    assert combined._data == expected._data


def test_empty_sub_descriptors():
    base = RefinementDescriptor(2, [1, 1])
    combined, _, mappings = compose_descriptors(base, {})
    assert combined._data == base._data
    assert mappings == {}


def test_anisotropic_sub_2d():
    base = RefinementDescriptor(2, [1, 1])
    sub = RefinementDescriptor.from_binary(2, ba.bitarray("10 00 00"))
    combined, _, mappings = compose_descriptors(base, {1: sub})
    assert combined.get_num_boxes() == 5
    assert list(mappings.keys()) == [1]
    # Sub has 3 nodes (10, 00, 00) which land at combined nodes 2, 3, 4
    assert mappings[1] == [{2}, {3}, {4}]
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
    combined, _, _ = compose_descriptors(base, {3: sub})
    assert combined.get_num_boxes() == 8 - 1 + 2  # 9


def test_nested_compose():
    """Composing twice with anisotropic subs."""
    base = RefinementDescriptor(2, [1, 1])
    sub_x = RefinementDescriptor(2, [2, 1])
    desc2, _, _ = compose_descriptors(base, {0: sub_x, 3: sub_x})
    assert str(_disc(desc2)) == (
        "_________________\n"
        "|       |_|_|_|_|\n"
        "|_______|_|_|_|_|\n"
        "|_|_|_|_|       |\n"
        "|_|_|_|_|_______|"
    )
    sub_y = RefinementDescriptor(2, [1, 2])
    desc3, _, _ = compose_descriptors(desc2, {0: sub_y})
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
    combined, mappings = compose_grid([1, 1], [None, sub, None, None])
    assert list(mappings.keys()) == [1]
    assert len(mappings[1]) == len(sub)
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
    combined, mappings = compose_grid([2, 1], subs)
    assert list(mappings.keys()) == [7]
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
    combined, mappings = compose_grid([1, 1], [sub_x, None, None, sub_y])
    assert sorted(mappings.keys()) == [0, 3]
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
    combined, _ = compose_grid([2, 1], subs)
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

    base_levels = list(range(1, nd + 1))
    rng.shuffle(base_levels)
    num_base_cells = 1
    for lv in base_levels:
        num_base_cells *= 1 << lv

    sub_levels = list(range(1, nd + 1))
    sub = RefinementDescriptor(nd, sub_levels)
    sub_boxes = sub.get_num_boxes()

    cell_idx = rng.randrange(num_base_cells)
    subs = [None] * num_base_cells
    subs[cell_idx] = sub
    combined, mappings = compose_grid(base_levels, subs)
    assert combined.get_num_boxes() == num_base_cells - 1 + sub_boxes
    assert list(mappings.keys()) == [cell_idx]
    assert len(mappings[cell_idx]) == len(sub)


def test_all_none():
    """All-None sequence produces the base grid."""
    combined, mappings = compose_grid([1, 1], [None, None, None, None])
    base = RefinementDescriptor(2, [1, 1])
    assert combined._data == base._data
    assert mappings == {}


def test_all_refined():
    sub = RefinementDescriptor(2, [1, 1])
    combined, _ = compose_grid([1, 1], [sub, sub, sub, sub])
    expected = RefinementDescriptor(2, [2, 2])
    assert combined._data == expected._data


def test_node_mapping_values():
    base = RefinementDescriptor(2, [1, 1])
    sub = RefinementDescriptor(2, [1, 1])

    # Splice sub into base box 0 (base node 1).
    combined, base_mapping, sub_mappings = compose_descriptors(base, {0: sub})
    assert (
        str(_disc(combined))
        == """\
_________
|   |   |
|___|___|
|_|_|   |
|_|_|___|"""
    )

    # Sub mapping: sub's 5 nodes land at combined 1..5
    assert sub_mappings[0] == [{1}, {2}, {3}, {4}, {5}]
    for sub_node, combined_nodes in enumerate(sub_mappings[0]):
        for cn in combined_nodes:
            assert ba.bitarray(combined[cn]) == ba.bitarray(sub[sub_node])

    assert len(base_mapping) == len(base)
    assert base_mapping[0] == {0}  # root → combined 0
    assert base_mapping[1] == {1, 2, 3, 4, 5}  # replaced leaf → spliced nodes
    assert base_mapping[2] == {6}  # remaining leaves
    assert base_mapping[3] == {7}
    assert base_mapping[4] == {8}


def test_node_mapping_grid():
    sub = RefinementDescriptor(2, [1, 1])
    combined, mappings = compose_grid([1, 1], [None, None, None, sub])
    assert (
        str(_disc(combined))
        == """\
_________
|   |_|_|
|___|_|_|
|   |   |
|___|___|"""
    )
    assert list(mappings.keys()) == [3]
    for sub_node, combined_nodes in enumerate(mappings[3]):
        for cn in combined_nodes:
            assert ba.bitarray(combined[cn]) == ba.bitarray(sub[sub_node])


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
