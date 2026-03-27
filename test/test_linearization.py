# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
import pytest

from dyada.linearization import (
    MortonOrderLinearization,
    binary_or_none_generator,
    get_initial_coarsening_stack,
    get_initial_coarsen_refine_stack,
    indices_to_bitmask,
)


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

    tracker.register_group(first_pos, 42)

    # Remaining pops should resolve groups via the dict
    expected = [
        (fba("001"), None),
        (fba("100"), 42),
        (fba("101"), None),
        (fba("010"), 42),
        (fba("011"), None),
        (fba("110"), 42),
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
    tracker.register_group(first_pos, 99)

    # Only pos "001" shares remaining_positions "00" with first_pos "000"
    expected = [
        (fba("100"), None),
        (fba("010"), None),
        (fba("110"), None),
        (fba("001"), 99),
        (fba("101"), None),
        (fba("011"), None),
        (fba("111"), None),
    ]
    for expected_pos, expected_same in expected:
        pos, same = tracker.pop()
        assert pos == expected_pos
        assert same == expected_same
