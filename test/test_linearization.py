import bitarray as ba
import pytest

from dyada.linearization import (
    MortonOrderLinearization,
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
