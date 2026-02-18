# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
import numpy as np
import pytest
from random import randint

from dyada.coordinates import (
    DyadaTooFineError,
    LevelIndex,
    bitarray_startswith,
    level_index_from_sequence,
    interval_from_sequences,
    get_coordinates_from_level_index,
    float_parts_bitarray,
    location_code_from_float,
    location_code_from_coordinate,
    location_code_from_level_index,
    level_index_from_location_code,
)


def test_get_coordinates_from_level_index():
    level_index = level_index_from_sequence([0, 0], [0, 0])
    assert get_coordinates_from_level_index(level_index) == interval_from_sequences(
        [0.0, 0.0], [1.0, 1.0]
    )
    with pytest.raises(ValueError):
        get_coordinates_from_level_index(level_index_from_sequence([0, 0], [1, 0]))
    level_index = level_index_from_sequence([0, 1, 0], [0, 1, 0])
    assert get_coordinates_from_level_index(level_index) == interval_from_sequences(
        [0.0, 0.5, 0.0], [1.0, 1.0, 1.0]
    )
    level_index = level_index_from_sequence([5, 4, 3, 2, 1, 0], [31, 15, 7, 3, 1, 0])
    expected_lower = [0.96875, 0.9375, 0.875, 0.75, 0.5, 0.0]
    assert get_coordinates_from_level_index(level_index) == interval_from_sequences(
        expected_lower, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    level_index = level_index_from_sequence([0, 1, 2, 3], [0, 0, 0, 0])
    assert get_coordinates_from_level_index(level_index) == interval_from_sequences(
        [0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.25, 0.125]
    )
    # make sure that if too large, we get a DyadaTooFineError
    with pytest.raises(DyadaTooFineError):
        level_index_from_sequence([0, 63, 2, 3], [0, 0, 0, 0])
    barely_ok_level_index = level_index_from_sequence([0, 62, 2, 3], [0, 0, 0, 0])
    barely_ok_level_index.d_level[1] = 63  # this should be too fine
    with pytest.raises(DyadaTooFineError):
        get_coordinates_from_level_index(barely_ok_level_index)

    # some randomization
    for _ in range(100):
        num_dimensions = randint(1, 10)
        level = [randint(0, 12) for _ in range(num_dimensions)]
        index = [randint(0, 2**l_d - 1) for l_d in level]
        level_index = LevelIndex(
            np.asarray(level, dtype=np.uint8), np.asarray(index, dtype=np.uint16)
        )  # use of uint is intentional!

        interval = get_coordinates_from_level_index(level_index)
        assert np.all(interval.lower_bound < interval.upper_bound)
        assert np.all(interval.lower_bound >= 0.0)
        assert np.all(interval.upper_bound <= 1.0)


def test_float_parts_bitarray():
    # test the float_parts_bitarray function
    # 1.0 should be 0b
    _, _, mantissa_bits = float_parts_bitarray(1.0)
    assert mantissa_bits.count() == 0
    # but putting 1.0 in the location code should give us 1111...
    mantissa_bits = location_code_from_float(1.0)
    assert mantissa_bits.count() == len(mantissa_bits)

    with pytest.raises(ValueError):
        float_parts_bitarray(np.longdouble(1.0))
    with pytest.raises(ValueError):
        float_parts_bitarray(np.half(1.0))


def test_location_code_from_coordinate():
    coordinate = np.array([0.0, 0.25, 0.5, 0.75])
    location_code = location_code_from_coordinate(coordinate)
    expected_codes = [
        ba.bitarray(""),  # 0.0
        ba.bitarray("01"),  # 0.25
        ba.bitarray("1"),  # 0.5
        ba.bitarray("11"),  # 0.75
    ]
    for code, expected in zip(location_code, expected_codes):
        assert bitarray_startswith(code, expected)


def test_conversions_random_data():
    assert len(ba.bitarray("0")) == 1
    for _ in range(100):
        num_dimensions = randint(1, 6)
        level = [randint(0, 10) for _ in range(num_dimensions)]
        index = [randint(0, 2**l_d - 1) for l_d in level]
        level_index = level_index_from_sequence(level, index)
        interval = get_coordinates_from_level_index(level_index)
        coordinate_from_li = interval.lower_bound

        location_code_from_c = location_code_from_coordinate(coordinate_from_li)
        location_code_from_li = location_code_from_level_index(level_index)
        for from_c, from_li in zip(location_code_from_c, location_code_from_li):
            assert bitarray_startswith(from_c, from_li)
        # coordinate_from_location_code = ...#TODO
        # assert np.allclose(coordinate_from_li, coordinate_from_location_code)

        level_index_from_lc = level_index_from_location_code(location_code_from_li)
        assert level_index == level_index_from_lc
