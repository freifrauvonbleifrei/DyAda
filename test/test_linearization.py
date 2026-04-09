# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
import pytest

from dyada.linearization import (
    MortonOrderLinearization,
    TrackToken,
    binary_or_none_generator,
    flat_to_coord,
    get_initial_coarsening_stack,
    get_initial_coarsen_refine_stack,
    grid_coord_to_z_index,
    indices_to_bitmask,
)
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization, coordinates_from_box_index


def test_get_position_morton_order():
    lin = MortonOrderLinearization()

    level_increment = ba.bitarray("1")
    position = lin.get_binary_position_from_index([0], [level_increment])
    assert position == ba.bitarray("0")
    position = lin.get_binary_position_from_index([1], [level_increment])
    assert position == ba.bitarray("1")
    with pytest.raises(IndexError):
        lin.get_binary_position_from_index([2], [level_increment])

    level_increment = ba.bitarray("11")
    position = lin.get_binary_position_from_index([0], [level_increment])
    assert position == ba.bitarray("00")
    position = lin.get_binary_position_from_index([1], [level_increment])
    assert position == ba.bitarray("10")
    position = lin.get_binary_position_from_index([2], [level_increment])
    assert position == ba.bitarray("01")
    position = lin.get_binary_position_from_index([3], [level_increment])
    assert position == ba.bitarray("11")
    with pytest.raises(IndexError):
        lin.get_binary_position_from_index([4], [level_increment])

    level_increment = ba.bitarray("01")
    position = lin.get_binary_position_from_index([0], [level_increment])
    assert position == ba.bitarray("00")
    position = lin.get_binary_position_from_index([1], [level_increment])
    assert position == ba.bitarray("01")
    with pytest.raises(IndexError):
        lin.get_binary_position_from_index([2], [level_increment])

    level_increment = ba.bitarray("10")
    position = lin.get_binary_position_from_index([0], [level_increment])
    assert position == ba.bitarray("00")
    position = lin.get_binary_position_from_index([1], [level_increment])
    assert position == ba.bitarray("10")
    with pytest.raises(IndexError):
        lin.get_binary_position_from_index([2], [level_increment])

    level_increment = ba.bitarray("111")
    position = lin.get_binary_position_from_index([0], [level_increment])
    assert position == ba.bitarray("000")
    position = lin.get_binary_position_from_index([1], [level_increment])
    assert position == ba.bitarray("100")
    position = lin.get_binary_position_from_index([2], [level_increment])
    assert position == ba.bitarray("010")
    position = lin.get_binary_position_from_index([3], [level_increment])
    assert position == ba.bitarray("110")
    position = lin.get_binary_position_from_index([7], [level_increment])
    assert position == ba.bitarray("111")
    with pytest.raises(IndexError):
        lin.get_binary_position_from_index([8], [level_increment])

    level_increment = ba.bitarray("110")
    position = lin.get_binary_position_from_index([0], [level_increment])
    assert position == ba.bitarray("000")
    position = lin.get_binary_position_from_index([1], [level_increment])
    assert position == ba.bitarray("100")
    position = lin.get_binary_position_from_index([2], [level_increment])
    assert position == ba.bitarray("010")
    position = lin.get_binary_position_from_index([3], [level_increment])
    assert position == ba.bitarray("110")
    with pytest.raises(IndexError):
        lin.get_binary_position_from_index([4], [level_increment])


def test_get_index_morton_order():
    lin = MortonOrderLinearization()

    level_increment = ba.bitarray("1")
    index = lin.get_index_from_binary_position(ba.bitarray("0"), [], [level_increment])
    assert index == 0
    index = lin.get_index_from_binary_position(ba.bitarray("1"), [], [level_increment])
    assert index == 1
    with pytest.raises(AssertionError):
        lin.get_index_from_binary_position(ba.bitarray("10"), [], [level_increment])

    level_increment = ba.bitarray("11")
    position = lin.get_index_from_binary_position(
        ba.bitarray("00"), [], [level_increment]
    )
    assert position == 0
    position = lin.get_index_from_binary_position(
        ba.bitarray("10"), [], [level_increment]
    )
    assert position == 1
    position = lin.get_index_from_binary_position(
        ba.bitarray("01"), [], [level_increment]
    )
    assert position == 2
    position = lin.get_index_from_binary_position(
        ba.bitarray("11"), [], [level_increment]
    )
    assert position == 3

    level_increment = ba.bitarray("01")
    position = lin.get_index_from_binary_position(
        ba.bitarray("00"), [], [level_increment]
    )
    assert position == 0
    position = lin.get_index_from_binary_position(
        ba.bitarray("01"), [], [level_increment]
    )
    assert position == 1
    with pytest.raises(AssertionError):
        lin.get_index_from_binary_position(ba.bitarray("10"), [], [level_increment])
    with pytest.raises(AssertionError):
        lin.get_index_from_binary_position(ba.bitarray("11"), [], [level_increment])

    level_increment = ba.bitarray("10")
    position = lin.get_index_from_binary_position(
        ba.bitarray("00"), [], [level_increment]
    )
    assert position == 0
    position = lin.get_index_from_binary_position(
        ba.bitarray("10"), [], [level_increment]
    )
    assert position == 1

    level_increment = ba.bitarray("111")
    position = lin.get_index_from_binary_position(
        ba.bitarray("000"), [], [level_increment]
    )
    assert position == 0
    position = lin.get_index_from_binary_position(
        ba.bitarray("100"), [], [level_increment]
    )
    assert position == 1
    position = lin.get_index_from_binary_position(
        ba.bitarray("010"), [], [level_increment]
    )
    assert position == 2
    position = lin.get_index_from_binary_position(
        ba.bitarray("110"), [], [level_increment]
    )
    assert position == 3
    position = lin.get_index_from_binary_position(
        ba.bitarray("111"), [], [level_increment]
    )
    assert position == 7


def test_binary_or_none_generator():
    indices = {0, 2}
    N = 4
    generated = list(binary_or_none_generator(indices, N))
    expected = [
        (0, None, 0, None),
        (0, None, 0, None),
        (0, None, 0, None),
        (0, None, 0, None),
        (1, None, 0, None),
        (1, None, 0, None),
        (1, None, 0, None),
        (1, None, 0, None),
        (0, None, 1, None),
        (0, None, 1, None),
        (0, None, 1, None),
        (0, None, 1, None),
        (1, None, 1, None),
        (1, None, 1, None),
        (1, None, 1, None),
        (1, None, 1, None),
    ]
    assert generated == expected


def _pop_all_positions(tracker):
    """Pop all positions from a tracker, returning them in pop order."""
    positions = []
    while len(tracker) > 0:
        pos, _ = tracker.pop()
        positions.append(pos)
    return positions


def test_empty_coarsening_stack_initialization():
    fba = ba.frozenbitarray
    tracker = get_initial_coarsening_stack(
        current_parent_refinement=fba("111"),
        dimensions_to_coarsen=fba("000"),
    )
    assert tracker.separated_mask == fba("000")
    assert tracker.unresolved_mask is None
    expected_pop_order = [
        fba("000"),
        fba("100"),
        fba("010"),
        fba("110"),
        fba("001"),
        fba("101"),
        fba("011"),
        fba("111"),
    ]
    assert _pop_all_positions(tracker) == expected_pop_order


def test_all_coarsening_stack_initialization():
    fba = ba.frozenbitarray
    tracker = get_initial_coarsening_stack(
        current_parent_refinement=fba("111"),
        dimensions_to_coarsen=fba("111"),
    )
    assert tracker.separated_mask == fba("111")
    expected_pop_order = [
        fba("000"),
        fba("100"),
        fba("010"),
        fba("110"),
        fba("001"),
        fba("101"),
        fba("011"),
        fba("111"),
    ]
    assert _pop_all_positions(tracker) == expected_pop_order


def test_coarsening_stack_2d():
    tracker = get_initial_coarsening_stack(
        current_parent_refinement=ba.frozenbitarray("11"),
        dimensions_to_coarsen=indices_to_bitmask((0,), 2),
    )
    assert tracker.separated_mask == ba.frozenbitarray("10")
    expected_pop_order = [
        ba.frozenbitarray("00"),
        ba.frozenbitarray("01"),
        ba.frozenbitarray("10"),
        ba.frozenbitarray("11"),
    ]
    assert _pop_all_positions(tracker) == expected_pop_order


def test_coarsening_stack_3d():
    fba = ba.frozenbitarray
    tracker = get_initial_coarsening_stack(
        current_parent_refinement=fba("111"),
        dimensions_to_coarsen=indices_to_bitmask((0, 1), 3),
    )
    assert tracker.separated_mask == fba("110")

    # Pop first item and register its group
    first_pos, first_same = tracker.pop()
    assert first_pos == fba("000")
    assert first_same is None

    tracker.register_group(first_pos, TrackToken(42))

    # Remaining pops should resolve groups via the dict
    expected = [
        (fba("001"), None),
        (fba("100"), TrackToken(42)),
        (fba("101"), None),
        (fba("010"), TrackToken(42)),
        (fba("011"), None),
        (fba("110"), TrackToken(42)),
        (fba("111"), None),
    ]
    for expected_pos, expected_same in expected:
        pos, same = tracker.pop()
        assert pos == expected_pos
        assert same == expected_same


def test_coarsen_refine_stack_cannot_coarsen_2d():
    fba = ba.frozenbitarray
    tracker = get_initial_coarsen_refine_stack(
        current_parent_refinement=fba("01"),
        dimensions_to_coarsen=fba("00"),
        dimensions_to_refine=fba("00"),
        dimensions_cannot_coarsen=fba("10"),
    )
    assert tracker.separated_mask == fba("00")
    assert tracker.unresolved_mask == fba("10")
    expected_pop_order = [fba("00"), fba("01"), fba("10"), fba("11")]
    assert _pop_all_positions(tracker) == expected_pop_order


def test_coarsen_refine_stack_3d():
    fba = ba.frozenbitarray
    tracker = get_initial_coarsen_refine_stack(
        current_parent_refinement=ba.frozenbitarray("011"),
        dimensions_to_coarsen=indices_to_bitmask((2,), 3),
        dimensions_to_refine=indices_to_bitmask((0,), 3),
    )
    assert tracker.separated_mask == fba("001")

    # Pop first and register group
    first_pos, first_same = tracker.pop()
    assert first_pos == fba("000")
    assert first_same is None
    tracker.register_group(first_pos, TrackToken(99))

    # Only pos "001" shares remaining_positions "00" with first_pos "000"
    expected = [
        (fba("100"), None),
        (fba("010"), None),
        (fba("110"), None),
        (fba("001"), TrackToken(99)),
        (fba("101"), None),
        (fba("011"), None),
        (fba("111"), None),
    ]
    for expected_pos, expected_same in expected:
        pos, same = tracker.pop()
        assert pos == expected_pos
        assert same == expected_same


def test_flat_to_coord_1d():
    assert flat_to_coord(0, (4,)) == (0,)
    assert flat_to_coord(3, (4,)) == (3,)


def test_flat_to_coord_3d():
    assert flat_to_coord(0, (2, 2, 2)) == (0, 0, 0)
    assert flat_to_coord(1, (2, 2, 2)) == (1, 0, 0)
    assert flat_to_coord(2, (2, 2, 2)) == (0, 1, 0)
    assert flat_to_coord(5, (2, 2, 2)) == (1, 0, 1)
    assert flat_to_coord(7, (2, 2, 2)) == (1, 1, 1)


def test_flat_to_coord_anisotropic():
    assert flat_to_coord(0, (4, 2)) == (0, 0)
    assert flat_to_coord(3, (4, 2)) == (3, 0)
    assert flat_to_coord(4, (4, 2)) == (0, 1)
    assert flat_to_coord(7, (4, 2)) == (3, 1)


def test_z_index_2d_anisotropic():
    assert grid_coord_to_z_index((0, 0), [2, 1]) == 0
    assert grid_coord_to_z_index((1, 0), [2, 1]) == 1
    assert grid_coord_to_z_index((2, 0), [2, 1]) == 2
    assert grid_coord_to_z_index((3, 0), [2, 1]) == 3
    assert grid_coord_to_z_index((0, 1), [2, 1]) == 4
    assert grid_coord_to_z_index((1, 1), [2, 1]) == 5
    assert grid_coord_to_z_index((2, 1), [2, 1]) == 6
    assert grid_coord_to_z_index((3, 1), [2, 1]) == 7


def test_z_index_3d():
    assert grid_coord_to_z_index((0, 0, 0), [1, 1, 1]) == 0
    assert grid_coord_to_z_index((1, 0, 0), [1, 1, 1]) == 1
    assert grid_coord_to_z_index((0, 1, 0), [1, 1, 1]) == 2
    assert grid_coord_to_z_index((1, 1, 0), [1, 1, 1]) == 3
    assert grid_coord_to_z_index((0, 0, 1), [1, 1, 1]) == 4
    assert grid_coord_to_z_index((1, 1, 1), [1, 1, 1]) == 7


def test_z_index_round_trip_with_flat():
    """Z-index of flat_to_coord should produce a permutation of [0..total)."""
    for levels in ([1, 1], [2, 1], [1, 2], [2, 2], [1, 1, 1], [2, 1, 1]):
        shape = tuple(1 << lv for lv in levels)
        total = 1
        for s in shape:
            total *= s
        z_indices = [
            grid_coord_to_z_index(flat_to_coord(i, shape), levels) for i in range(total)
        ]
        assert sorted(z_indices) == list(range(total))


@pytest.mark.parametrize(
    "grid_levels",
    [
        # 2D
        [1, 1],
        [2, 1],
        [2, 3],
        # 3D
        [1, 1, 1],
        [1, 1, 2],
        [2, 2, 1],
        [2, 1, 2],
        # 4D
        [1, 1, 1, 1],
        [1, 2, 1, 2],
        # 5D
        [1, 2, 1, 1, 1],
    ],
)
def test_z_index_matches_descriptor(grid_levels):
    """Validate grid_coord_to_z_index against dyada's actual descriptor."""
    nd = len(grid_levels)
    desc = RefinementDescriptor(nd, list(grid_levels))
    disc = Discretization(MortonOrderLinearization(), desc)
    num_boxes = desc.get_num_boxes()
    grid_shape = tuple(1 << lv for lv in grid_levels)

    for box_idx in range(num_boxes):
        interval = coordinates_from_box_index(disc, box_idx)
        coord = tuple(int(interval.lower_bound[d] * grid_shape[d]) for d in range(nd))
        z = grid_coord_to_z_index(coord, grid_levels)
        assert z == box_idx, (
            f"grid_levels={grid_levels}, coord={coord}: "
            f"expected z={box_idx}, got z={z}"
        )
