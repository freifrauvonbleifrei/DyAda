import numpy as np
import pytest
from random import randint

from dyada.coordinates import (
    DyadaTooFineError,
    LevelIndex,
    level_index_from_sequence,
    interval_from_sequences,
    get_coordinates_from_level_index,
    float_parts_bitarray,
    deciding_bitarray_from_float,
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
    # but putting 1.0 in the deciding_bitarray should give us 1111...
    mantissa_bits = deciding_bitarray_from_float(1.0)
    assert mantissa_bits.count() == len(mantissa_bits)

    with pytest.raises(ValueError):
        float_parts_bitarray(np.float128(1.0))
    with pytest.raises(ValueError):
        float_parts_bitarray(np.float16(1.0))
