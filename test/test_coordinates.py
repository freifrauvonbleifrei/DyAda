import numpy as np
import pytest
from random import randint

from dyada.coordinates import (
    LevelIndex,
    level_index_from_sequence,
    interval_from_sequences,
    get_coordinates_from_level_index,
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
    # some randomization
    for _ in range(100):
        num_dimensions = randint(1, 10)
        level = [randint(0, 12) for _ in range(num_dimensions)]
        index = [randint(0, 2**l - 1) for l in level]
        level_index = LevelIndex(
            np.asarray(level, dtype=np.uint8), np.asarray(index, dtype=np.uint16)
        )  # use of uint is intentional!

        interval = get_coordinates_from_level_index(level_index)
        assert np.all(interval.lower_bound < interval.upper_bound)
        assert np.all(interval.lower_bound >= 0.0)
        assert np.all(interval.upper_bound <= 1.0)
