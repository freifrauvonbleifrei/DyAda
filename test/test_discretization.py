import pytest
import bitarray as ba
import numpy as np

from dyada.descriptor import (
    RefinementDescriptor,
    validate_descriptor,
)

from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization


def test_get_level_index():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))

    # with indices that consider the parents
    level, index = r.get_level_index(0, False)
    assert np.array_equal(level, [0, 0]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(1, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(2, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(3, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(4, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(5, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(6, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(7, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(8, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 2])
    level, index = r.get_level_index(9, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 3])
    level, index = r.get_level_index(10, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(11, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 2])
    level, index = r.get_level_index(12, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 3])
    with pytest.raises(IndexError):
        r.get_level_index(13, False)

    # with indices that consider only leaves / boxes
    level, index = r.get_level_index(0, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(1, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(2, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(3, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(4, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 2])
    level, index = r.get_level_index(5, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 3])
    level, index = r.get_level_index(6, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 2])
    level, index = r.get_level_index(7, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 3])
    with pytest.raises(IndexError):
        r.get_level_index(8, True)


def test_get_all_boxes_level_indices():
    # same example as test_get_level_index
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    expected_indices = ([0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3])
    for i, level_index in enumerate(r.get_all_boxes_level_indices()):
        assert np.array_equal(level_index.d_level, np.asarray([1, 2]))
        assert np.array_equal(level_index.d_index, np.asarray(expected_indices[i]))


def test_get_box_from_coordinate():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    # check the midpoints of each box
    assert r.get_containing_box(np.array([0.25, 0.125])) == 0
    assert r.get_containing_box(np.array([0.25, 0.375])) == 1
    assert r.get_containing_box(np.array([0.75, 0.125])) == 2
    assert r.get_containing_box(np.array([0.75, 0.375])) == 3
    assert r.get_containing_box(np.array([0.25, 0.625])) == 4
    assert r.get_containing_box(np.array([0.25, 0.875])) == 5
    assert r.get_containing_box(np.array([0.75, 0.625])) == 6
    assert r.get_containing_box(np.array([0.75, 0.875])) == 7

    # now along the edges
    assert r.get_containing_box(np.array([0.0, 0.0])) == 0
    assert r.get_containing_box(np.array([0.0, 0.25])) == (0, 1)
    assert r.get_containing_box(np.array([0.0, 0.5])) == (1, 4)
    assert r.get_containing_box(np.array([0.0, 0.75])) == (4, 5)
    assert r.get_containing_box(np.array([0.0, 1.0])) == 5
    assert r.get_containing_box(np.array([0.25, 0.0])) == 0
    assert r.get_containing_box(np.array([0.25, 0.25])) == (0, 1)
    assert r.get_containing_box(np.array([0.25, 0.5])) == (1, 4)
    assert r.get_containing_box(np.array([0.25, 0.75])) == (4, 5)
    assert r.get_containing_box(np.array([0.25, 1.0])) == 5
    assert r.get_containing_box(np.array([0.5, 0.0])) == (0, 2)
    assert r.get_containing_box(np.array([0.5, 0.25])) == (0, 1, 2, 3)
    assert r.get_containing_box(np.array([0.5, 0.5])) == (1, 3, 4, 6)
    assert r.get_containing_box(np.array([0.5, 0.75])) == (4, 5, 6, 7)
    assert r.get_containing_box(np.array([0.5, 1.0])) == (5, 7)
    assert r.get_containing_box(np.array([0.75, 0.0])) == 2
    assert r.get_containing_box(np.array([0.75, 0.25])) == (2, 3)
    assert r.get_containing_box(np.array([0.75, 0.5])) == (3, 6)
    assert r.get_containing_box(np.array([0.75, 0.75])) == (6, 7)
    assert r.get_containing_box(np.array([0.75, 1.0])) == 7
    assert r.get_containing_box(np.array([1.0, 0.0])) == 2
    assert r.get_containing_box(np.array([1.0, 0.25])) == (2, 3)
    assert r.get_containing_box(np.array([1.0, 0.5])) == (3, 6)
    assert r.get_containing_box(np.array([1.0, 0.75])) == (6, 7)
    assert r.get_containing_box(np.array([1.0, 1.0])) == 7

    # outside the domain
    with pytest.raises(ValueError):
        r.get_containing_box(np.array([0.0, 1.5]))
    with pytest.raises(ValueError):
        r.get_containing_box(np.array([1.5, 0.0]))
    with pytest.raises(ValueError):
        r.get_containing_box(np.array([1.5, 1.5]))


def test_slice_discretization_3d():
    # same 3d discretization as in test_plot_boxes_3d_from_descriptor
    descriptor = RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "101 000 001 000 000 010 100 000 000 000"
            "101 000 000 010 000 101 000 000 000 000 000"
        ),
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)

    for z_i in range(0, 20):
        z = z_i / 19
        discretization_xy, mapping_xy = discretization.slice([None, None, z])
        assert validate_descriptor(discretization_xy.descriptor)

        for x_i in range(0, 20):
            x = x_i / 19
            discretization_y, mapping_y = discretization_xy.slice([x, None])
            assert validate_descriptor(discretization_y.descriptor)
            mapping_z_to_x_to_y = {}
            for old_y, new_y in mapping_y.items():
                # invert the xy mapping, find old_y
                for old_xy, new_xy in mapping_xy.items():
                    if new_xy == old_y:
                        mapping_z_to_x_to_y[old_xy] = new_y
                        break
            # check if slicing in two steps is the same as slicing in one step
            discretization_y_at_once, mapping_y_at_once = discretization.slice(
                [x, None, z]
            )
            # assert that everything in mapping_z_to_x_to_y is in mapping_y
            assert all(
                [
                    mapping_y_at_once[old_y] == new_y
                    for old_y, new_y in mapping_z_to_x_to_y.items()
                ]
            )
            assert validate_descriptor(discretization_y_at_once.descriptor)
            assert discretization_y == discretization_y_at_once
