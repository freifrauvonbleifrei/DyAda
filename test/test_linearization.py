import bitarray as ba
import pytest

from dyada.linearization import (
    MortonOrderLinearization,
    DimensionSeparatedLocalPosition,
    get_initial_coarsening_stack,
    replace_same_remaining_position_by_mapped_to_index,
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


def test_empty_coarsening_stack_initialization():
    initial_coarsening_stack = get_initial_coarsening_stack(
        current_parent_refinement=ba.frozenbitarray("111"),
        dimensions_to_coarsen=tuple(),
    )
    expected_coarsening_stack = [
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("000")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("100")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("010")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("110")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("001")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("101")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("011")),
        DimensionSeparatedLocalPosition(ba.frozenbitarray(), ba.frozenbitarray("111")),
    ]
    expected_coarsening_stack.reverse()
    assert initial_coarsening_stack == expected_coarsening_stack


def test_all_coarsening_stack_initialization():
    initial_coarsening_stack = get_initial_coarsening_stack(
        current_parent_refinement=ba.frozenbitarray("111"),
        dimensions_to_coarsen=(0, 1, 2),
    )

    expected_coarsening_stack = [
        DimensionSeparatedLocalPosition(ba.frozenbitarray("000"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("100"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("010"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("110"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("001"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("101"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("011"), ba.frozenbitarray()),
        DimensionSeparatedLocalPosition(ba.frozenbitarray("111"), ba.frozenbitarray()),
    ]
    expected_coarsening_stack.reverse()
    assert initial_coarsening_stack == expected_coarsening_stack


def test_coarsening_stack_3d():
    current_coarsening_stack = get_initial_coarsening_stack(
        current_parent_refinement=ba.frozenbitarray("111"),
        dimensions_to_coarsen=(0, 1),
    )
    expected_coarsening_stack = [
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("00"), ba.frozenbitarray("0")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("10"), ba.frozenbitarray("0")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("01"), ba.frozenbitarray("0")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("11"), ba.frozenbitarray("0")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("00"), ba.frozenbitarray("1")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("10"), ba.frozenbitarray("1")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("01"), ba.frozenbitarray("1")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("11"), ba.frozenbitarray("1")
        ),
    ]
    expected_coarsening_stack.reverse()
    assert current_coarsening_stack == expected_coarsening_stack

    first_item = current_coarsening_stack.pop()
    first_item_replace_index = 42
    replace_same_remaining_position_by_mapped_to_index(
        coarsening_stack=current_coarsening_stack,
        position_to_replace=first_item,
        mapped_to_index=first_item_replace_index,
    )
    expected_coarsening_stack_after_replacement = [
        42,
        42,
        42,
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("00"), ba.frozenbitarray("1")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("10"), ba.frozenbitarray("1")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("01"), ba.frozenbitarray("1")
        ),
        DimensionSeparatedLocalPosition(
            ba.frozenbitarray("11"), ba.frozenbitarray("1")
        ),
    ]
    expected_coarsening_stack_after_replacement.reverse()
    assert current_coarsening_stack == expected_coarsening_stack_after_replacement
