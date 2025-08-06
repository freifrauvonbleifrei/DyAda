import pytest
import bitarray as ba
import numpy as np

from dyada.descriptor import (
    RefinementDescriptor,
    validate_descriptor,
)
from dyada.discretization import (
    Discretization,
    SliceDictInDimension,
    discretization_to_location_stack_strings,
)
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


def helper_mapping_as_box_mapping(
    old_descriptor: RefinementDescriptor, new_descriptor: RefinementDescriptor, mapping
):
    """
    Helper function to assert that a mapping can be converted to a box mapping.
    """
    box_keys = list()
    box_values = list()
    for key, value in mapping.items():
        try:
            box_keys.append(old_descriptor.to_box_index(key))
        except AssertionError:
            # if the key is not a box index, we skip it
            continue
        box_values.append(new_descriptor.to_box_index(value))

    assert len(set(box_keys)) == len(box_keys)
    assert len(set(box_values)) == len(box_values)

    box_mapping = {
        old_descriptor.to_box_index(k): new_descriptor.to_box_index(v)
        for k, v in mapping.items()
        if old_descriptor.is_box(k)
    }
    assert set(box_mapping.keys()) == set(box_keys)


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

    # # plot
    # plot_all_boxes_3d(
    #     discretization,
    #     filename="test_slice_discretization_3d",
    #     labels="patches",
    #     title="test_slice_discretization_3d",
    #     backend="tikz",
    # )

    for z_i in range(0, 20):
        z = z_i / 19
        discretization_xy, mapping_xy = discretization.slice([None, None, z])
        assert validate_descriptor(discretization_xy.descriptor)
        helper_mapping_as_box_mapping(
            descriptor, discretization_xy.descriptor, mapping_xy
        )
        # plot_all_boxes_2d(
        #     discretization_xy,
        #     filename="test_slice_discretization_3d_z" + str(z_i),
        #     labels="patches",
        #     title="test_slice_discretization_3d_z" + str(z_i),
        #     backend="tikz",
        # )

        for x_i in range(0, 20):
            x = x_i / 19
            discretization_y, mapping_y = discretization_xy.slice([x, None])
            assert validate_descriptor(discretization_y.descriptor)
            helper_mapping_as_box_mapping(
                discretization_xy.descriptor,
                discretization_y.descriptor,
                mapping_y,
            )
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
            helper_mapping_as_box_mapping(
                discretization.descriptor,
                discretization_y_at_once.descriptor,
                mapping_y_at_once,
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


def test_all_slices_3d():
    # same 3d discretization as in test_plot_boxes_3d_from_descriptor
    descriptor = RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "101 000 001 000 000 010 100 000 000 000"
            "101 000 000 010 000 101 000 000 000 000 000"
        ),
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)

    z_start = 0.0
    x_start = 0.0
    z_used = set()
    x_used = set()
    while z_start < 1.0 and x_start < 1.0:
        # slice at z_start and x_start
        discretization_xz, mapping_xz, levels_xz = discretization.slice(
            [x_start, None, z_start], get_level=True
        )
        z_used.add(z_start)
        x_used.add(x_start)
        assert validate_descriptor(discretization_xz.descriptor)
        x_start += 2.0 ** -levels_xz[0]
        z_start += 2.0 ** -levels_xz[2]

        helper_mapping_as_box_mapping(
            descriptor, discretization_xz.descriptor, mapping_xz
        )

    assert len(z_used) == 3
    assert len(x_used) == 3


def test_hash_discretization():
    descriptor = RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "101 000 001 000 000 010 100 000 000 000"
            "101 000 000 010 000 101 000 000 000 000 000"
        ),
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    other_discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert hash(discretization) == hash(other_discretization)
    different_descriptor = RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "101 000 001 000 000 010 100 000 000 000 101 000 000 010 000 000 000"
        ),
    )
    different_discretization = Discretization(
        MortonOrderLinearization(), different_descriptor
    )
    assert hash(discretization) != hash(different_discretization)


def test_slice_dict_3d():
    descriptor = RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "101 000 001 000 000 010 100 000 000 000"
            "101 000 000 010 000 101 000 000 000 000 000"
        ),
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    slice_dict_z = SliceDictInDimension(discretization, 2, True)
    assert slice_dict_z.keys() == {0.0, 0.25, 0.5, 0.75, 0.875}
    for key, item in slice_dict_z.items():
        assert key in slice_dict_z.keys()
        assert isinstance(item[0], Discretization)
        assert item[0].descriptor.get_num_dimensions() == 2
        assert isinstance(item[1], dict)
    assert slice_dict_z[0.1] is slice_dict_z[0.0]
    assert slice_dict_z[0.4999] is slice_dict_z[0.25]
    assert slice_dict_z[1.0] is slice_dict_z[0.875]
    with pytest.raises(KeyError):
        slice_dict_z[1.00001]
    # try to store other data types as values
    for key in slice_dict_z.keys():
        slice_dict_z[key] = len(slice_dict_z[key][0])
    assert slice_dict_z[0.0] == 2
    assert slice_dict_z[0.25] == 2
    assert slice_dict_z[0.5] == 5
    assert slice_dict_z[0.75] == 7
    assert slice_dict_z[0.875] == 7


def test_stack():
    first_descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("10 01 00 00 10 01 00 00 00")
    )
    assert discretization_to_location_stack_strings(
        Discretization(MortonOrderLinearization(), first_descriptor)
    ) == [
        ("λ", ""),
        ("0", "λ"),
        ("0", "0"),
        ("0", "1"),
        ("1λ", ""),
        ("10", "λ"),
        ("10", "0"),
        ("10", "1"),
        ("11", ""),
    ]
    second_descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 00 10 01 00 00 01 00 00 00 10 00 01 00 00")
    )
    assert discretization_to_location_stack_strings(
        Discretization(MortonOrderLinearization(), second_descriptor)
    ) == [
        ("λ", "λ"),
        ("0", "0"),
        ("1λ", "0"),
        ("10", "0λ"),
        ("10", "00"),
        ("10", "01"),
        ("11", "0λ"),
        ("11", "00"),
        ("11", "01"),
        ("0", "1"),
        ("1λ", "1"),
        ("10", "1"),
        ("11", "1λ"),
        ("11", "10"),
        ("11", "11"),
    ]
    third_descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 00 11 00 00 00 00 00 10 00 01 00 00")
    )
    assert discretization_to_location_stack_strings(
        Discretization(MortonOrderLinearization(), third_descriptor)
    ) == [
        ("λ", "λ"),
        ("0", "0"),
        ("1λ", "0λ"),
        ("10", "00"),
        ("11", "00"),
        ("10", "01"),
        ("11", "01"),
        ("0", "1"),
        ("1λ", "1"),
        ("10", "1"),
        ("11", "1λ"),
        ("11", "10"),
        ("11", "11"),
    ]
    fourth_descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("10 01 00 00 10 00 00")
    )
    assert discretization_to_location_stack_strings(
        Discretization(MortonOrderLinearization(), fourth_descriptor)
    ) == [
        ("λ", ""),
        ("0", "λ"),
        ("0", "0"),
        ("0", "1"),
        ("1λ", ""),
        ("10", ""),
        ("11", ""),
    ]
