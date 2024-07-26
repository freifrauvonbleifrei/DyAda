import bitarray as ba
from functools import cmp_to_key
from itertools import product
import numpy as np
import pytest

# import matplotlib.pyplot as plt

from dyada.linearization import (
    MortonOrderLinearization,
    compare_morton_order,
    float_to_bits,
    most_significant_different_bit,
    modified_frexp,
    get_most_significant_dimension,
    xormsb,
)
from dyada.coordinates import Coordinate, coordinate_from_sequence


def test_float_to_bits():
    f = 0.0
    assert f"{float_to_bits(f):064b}" == "0" * 64
    f = 0.5
    # cf. https://binaryconvert.com/result_double.html?decimal=048046053
    assert (
        f"{float_to_bits(f):064b}"
        == "00111111 11100000 00000000 00000000 00000000 00000000 00000000 00000000".replace(
            " ", ""
        )
    )
    f = 0.25
    # cf. https://binaryconvert.com/result_double.html?decimal=048046050053
    assert (
        f"{float_to_bits(f):064b}"
        == "00111111 11010000 00000000 00000000 00000000 00000000 00000000 00000000".replace(
            " ", ""
        )
    )
    f = 0.75
    # cf. https://binaryconvert.com/result_double.html?decimal=048046055053
    assert (
        f"{float_to_bits(f):064b}"
        == "00111111 11101000 00000000 00000000 00000000 00000000 00000000 00000000".replace(
            " ", ""
        )
    )
    # and a more random one
    f = 0.179387174350486983120944728398
    # cf. https://binaryconvert.com/result_double.html?decimal=048046049055057051056055049055052051053048052056054057056051049050048057052052055050056051057056
    assert (
        f"{float_to_bits(f):064b}"
        == "00111111 11000110 11110110 00101000 10101111 10010100 00011110 11011011".replace(
            " ", ""
        )
    )


def test_most_significant_different_bit():
    assert most_significant_different_bit(0.75, 0.5) == -1
    assert most_significant_different_bit(0.625, 0.875) == -1


def test_xormsb():
    a = 0.5
    b = 0.5
    assert xormsb(a, b) == np.finfo(np.float32).minexp - 1
    a = 0.0
    b = 0.5
    assert xormsb(a, b) == 0
    a = 0.5
    b = 0.25
    assert xormsb(a, b) == -1
    a = 0.5
    b = 0.75
    assert xormsb(a, b) == -2

    a = 0.125
    # 0|0111111 1100|0000 00000000 00000000 00000000 00000000 00000000 00000000
    # -3 | 0
    assert np.signbit(a) == 0
    assert np.frexp(a) == (0.5, -2)
    assert modified_frexp(a) == (1.0, -3)
    b = 0.875
    # 0|0111111 1110|1100 00000000 00000000 00000000 00000000 00000000 00000000
    # -1 | 3
    c = 0.375
    # 0|0111111 1101|1000 00000000 00000000 00000000 00000000 00000000 00000000
    # -2 | 1
    d = 0.625
    # 0|0111111 1110|0100 00000000 00000000 00000000 00000000 00000000 00000000
    # -1 | 2
    assert xormsb(a, b) == -1
    assert xormsb(a, c) == -2
    assert xormsb(a, d) == -1
    assert xormsb(b, c) == -1
    assert xormsb(b, d) == -2
    assert xormsb(c, d) == -1

    e = 0.8125
    f = 0.9375
    g = 0.6875
    assert xormsb(e, f) == -3
    assert xormsb(e, g) == -2
    assert xormsb(f, g) == -2


def test_get_most_significant_dimension():
    a = coordinate_from_sequence([0.8125, 0.8125])
    b = coordinate_from_sequence([0.9375, 0.6875])
    assert get_most_significant_dimension(a, b) == 1


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


def test_morton_comparison():
    p = np.array([0.0, 0.0])
    q = np.array([0.0, 1.0])
    assert compare_morton_order(p, q) == -1
    p = np.array([0.06849298, 0.41116066])
    q = np.array([0.03211638, 0.91644133])
    assert compare_morton_order(p, q) == -1

    coordinates = [
        coordinate_from_sequence([x, y]) for x, y in product([0.25, 0.75], [0.25, 0.75])
    ]
    sorted_coords = np.asarray(
        sorted(coordinates, key=cmp_to_key(compare_morton_order))
    )
    assert np.array_equal(
        sorted_coords,
        np.asarray([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]),
    )
    coordinates = [
        coordinate_from_sequence([x, y])
        for x, y in product([0.125, 0.375, 0.625, 0.875], [0.125, 0.375, 0.625, 0.875])
    ]
    sorted_coords = np.asarray(
        sorted(coordinates, key=cmp_to_key(compare_morton_order))
    )
    assert np.array_equal(
        sorted_coords,
        np.asarray(
            [
                [0.125, 0.125],
                [0.125, 0.375],
                [0.375, 0.125],
                [0.375, 0.375],
                [0.125, 0.625],
                [0.125, 0.875],
                [0.375, 0.625],
                [0.375, 0.875],
                [0.625, 0.125],
                [0.625, 0.375],
                [0.875, 0.125],
                [0.875, 0.375],
                [0.625, 0.625],
                [0.625, 0.875],
                [0.875, 0.625],
                [0.875, 0.875],
            ]
        ),
    )

    # create regular grid with spacing 0.125
    grid = np.meshgrid(
        np.linspace(0.0625, 1 - 0.0625, 8), np.linspace(0.0625, 1 - 0.0625, 8)
    )
    coordinates = [
        np.array([x, y]) for x, y in zip(grid[0].flatten(), grid[1].flatten())
    ]
    sorted_coords = np.asarray(
        sorted(coordinates, key=cmp_to_key(compare_morton_order))
    )
    # validated through plotting, see below!
    assert np.array_equal(
        sorted_coords,
        np.asarray(
            [
                [0.0625, 0.0625],
                [0.0625, 0.1875],
                [0.1875, 0.0625],
                [0.1875, 0.1875],
                [0.0625, 0.3125],
                [0.0625, 0.4375],
                [0.1875, 0.3125],
                [0.1875, 0.4375],
                [0.3125, 0.0625],
                [0.3125, 0.1875],
                [0.4375, 0.0625],
                [0.4375, 0.1875],
                [0.3125, 0.3125],
                [0.3125, 0.4375],
                [0.4375, 0.3125],
                [0.4375, 0.4375],
                [0.0625, 0.5625],
                [0.0625, 0.6875],
                [0.1875, 0.5625],
                [0.1875, 0.6875],
                [0.0625, 0.8125],
                [0.0625, 0.9375],
                [0.1875, 0.8125],
                [0.1875, 0.9375],
                [0.3125, 0.5625],
                [0.3125, 0.6875],
                [0.4375, 0.5625],
                [0.4375, 0.6875],
                [0.3125, 0.8125],
                [0.3125, 0.9375],
                [0.4375, 0.8125],
                [0.4375, 0.9375],
                [0.5625, 0.0625],
                [0.5625, 0.1875],
                [0.6875, 0.0625],
                [0.6875, 0.1875],
                [0.5625, 0.3125],
                [0.5625, 0.4375],
                [0.6875, 0.3125],
                [0.6875, 0.4375],
                [0.8125, 0.0625],
                [0.8125, 0.1875],
                [0.9375, 0.0625],
                [0.9375, 0.1875],
                [0.8125, 0.3125],
                [0.8125, 0.4375],
                [0.9375, 0.3125],
                [0.9375, 0.4375],
                [0.5625, 0.5625],
                [0.5625, 0.6875],
                [0.6875, 0.5625],
                [0.6875, 0.6875],
                [0.5625, 0.8125],
                [0.5625, 0.9375],
                [0.6875, 0.8125],
                [0.6875, 0.9375],
                [0.8125, 0.5625],
                [0.8125, 0.6875],
                [0.9375, 0.5625],
                [0.9375, 0.6875],
                [0.8125, 0.8125],
                [0.8125, 0.9375],
                [0.9375, 0.8125],
                [0.9375, 0.9375],
            ]
        ),
    )
    # _ , ax1 = plt.subplots(1, 1)
    # ax1.plot(sorted_coords[:, 1], sorted_coords[:, 0], "C3", label="sorted")
    # plt.show()
