from dyada.coordinates import (
    level_index_from_sequence,
    get_interval_from_sequences,
    get_coordinates_from_level_index,
)


def test_get_coordinates_from_level_index():
    level_index = level_index_from_sequence([0, 0], [0, 0])
    assert get_coordinates_from_level_index(level_index) == get_interval_from_sequences(
        [0.0, 0.0], [1.0, 1.0]
    )
    level_index = level_index_from_sequence([0, 1, 0], [0, 1, 0])
    assert get_coordinates_from_level_index(level_index) == get_interval_from_sequences(
        [0.0, 0.5, 0.0], [1.0, 1.0, 1.0]
    )
    level_index = level_index_from_sequence([5, 4, 3, 2, 1, 0], [31, 15, 7, 3, 1, 0])
    expected_lower = [0.96875, 0.9375, 0.875, 0.75, 0.5, 0.0]
    assert get_coordinates_from_level_index(level_index) == get_interval_from_sequences(
        expected_lower, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
