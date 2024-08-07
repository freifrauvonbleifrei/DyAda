import pytest
import bitarray as ba
from collections import deque
import numpy as np
from os.path import abspath

from dyada.refinement import (
    generalized_ruler,
    RefinementDescriptor,
    Discretization,
    validate_descriptor,
    PlannedAdaptiveRefinement,
)
from dyada.linearization import MortonOrderLinearization


def test_ruler():
    one = generalized_ruler(2, 0)
    assert np.array_equal(one, [1])
    two = generalized_ruler(2, 1)
    assert np.array_equal(two, [2, 1, 1, 1])
    three = generalized_ruler(2, 2)
    assert np.array_equal(three, [3, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1])
    other = generalized_ruler(1, 4)
    assert np.array_equal(other, [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1])


def test_construct():
    for i in range(1, 128):
        r = RefinementDescriptor(i)
        assert r
        validate_descriptor(r)


def test_zero_level():
    for i in range(1, 10):
        r = RefinementDescriptor(i, 0)
        assert len(r) == 1
        assert r.get_data().any() == False
        print(r.get_data())
        assert r.get_num_boxes() == 1
        validate_descriptor(r)


def test_one_level():
    for i in range(1, 2):
        r = RefinementDescriptor(i, 1)
        d = r.get_num_dimensions()
        assert d == i
        assert len(r) == 2**i + 1
        assert r.get_num_boxes() == 2**i
        assert r.get_data().count() == i
        assert r.get_data()[0] == 1
        assert r.is_pow2tree() == True
        validate_descriptor(r)


def test_six_d():
    for l in range(1, 4):
        r = RefinementDescriptor(6, l)
        assert r.get_num_dimensions() == 6
        acc = 1
        for i in range(l):
            acc = acc * 2**6 + 1
        assert len(r) == acc
        assert r.get_num_boxes() == 2 ** (l * 6)
        # count length of one-blocks
        lengths = []
        current_length = 0
        for i in range(0, len(r.get_data()), 6):
            c = r.get_data()[i : i + 6].count()
            # either all-zero or all-one
            assert c in [0, 6]
            if c == 6:
                current_length += 1
            elif current_length != 0:
                lengths.append(current_length)
                current_length = 0
        assert lengths == generalized_ruler(6, l - 1).tolist()
        print(r.is_pow2tree())
        assert r.is_pow2tree() == True
        validate_descriptor(r)


def test_construct_anisotropic():
    r = RefinementDescriptor(4, [0, 1, 2, 3])
    assert r.get_num_dimensions() == 4
    assert len(r) == ((2 + 1) * 4 + 1) * 8 + 1
    assert r.get_num_boxes() == 2**6
    assert r.get_data()[: 4 * 5] == ba.bitarray("01110011000100000000")
    assert r.get_data()[4 * -6 :] == ba.bitarray("000100000000000100000000")
    assert r[:5] == r.get_data()[: 4 * 5]
    assert r[5] == r.get_data()[4 * 5 : 4 * 6]
    assert r[-6:] == r.get_data()[4 * -6 :]
    assert r.is_pow2tree() == False
    validate_descriptor(r)


def test_get_level_isotropic():
    r = RefinementDescriptor(4, 3)
    with pytest.raises(IndexError):
        r.get_level(len(r), False)
    with pytest.raises(IndexError):
        r.get_level(-1, False)
    assert np.array_equal(r.get_level(0, False), [0, 0, 0, 0])
    assert np.array_equal(r.get_level(1, False), [1, 1, 1, 1])
    assert np.array_equal(r.get_level(2, False), [2, 2, 2, 2])
    for i in range(0, 16):
        assert np.array_equal(r.get_level(3 + i, False), [3, 3, 3, 3])
        assert np.array_equal(r.get_level(len(r) - i - 1, False), [3, 3, 3, 3])
    assert np.array_equal(r.get_level(len(r) - 16 - 1, False), [2, 2, 2, 2])


def test_get_branch():
    r = RefinementDescriptor(2, [1, 2])
    assert r.get_branch(2, False)[0] == r.get_branch(0, True)[0]
    assert r.get_branch(3, False)[0] == r.get_branch(1, True)[0]
    assert r.get_branch(5, False)[0] == r.get_branch(2, True)[0]
    assert r.get_branch(6, False)[0] == r.get_branch(3, True)[0]
    assert r.get_branch(8, False)[0] == r.get_branch(4, True)[0]
    assert r.get_branch(9, False)[0] == r.get_branch(5, True)[0]
    assert r.get_branch(9, False)[0] == deque(
        [
            RefinementDescriptor.LevelCounter(
                level_increment=ba.frozenbitarray("00"),
                count_to_go_up=1,
            ),
            RefinementDescriptor.LevelCounter(
                level_increment=ba.frozenbitarray("11"),
                count_to_go_up=2,
            ),
            RefinementDescriptor.LevelCounter(
                level_increment=ba.frozenbitarray("01"),
                count_to_go_up=1,
            ),
        ]
    )


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


def test_to_box_index():
    # according to the same mapping as in test_get_level_index
    r = RefinementDescriptor(2, [1, 2])
    for non_box_index in (0, 1, 4, 7, 10, 13):
        with pytest.raises(AssertionError):
            r.to_box_index(non_box_index)
    assert r.to_box_index(2) == 0
    assert r.to_box_index(3) == 1
    assert r.to_box_index(5) == 2
    assert r.to_box_index(6) == 3
    assert r.to_box_index(8) == 4
    assert r.to_box_index(9) == 5
    assert r.to_box_index(11) == 6
    assert r.to_box_index(12) == 7


def test_to_hierarchical_index():
    # according to the same mapping as in test_get_level_index
    r = RefinementDescriptor(2, [1, 2])
    assert r.to_hierarchical_index(0) == 2
    assert r.to_hierarchical_index(1) == 3
    assert r.to_hierarchical_index(2) == 5
    assert r.to_hierarchical_index(3) == 6
    assert r.to_hierarchical_index(4) == 8
    assert r.to_hierarchical_index(5) == 9
    assert r.to_hierarchical_index(6) == 11
    assert r.to_hierarchical_index(7) == 12
    with pytest.raises(AssertionError):
        r.to_hierarchical_index(8)


def test_get_siblings():
    r = RefinementDescriptor(2, [1, 2])
    assert r.get_siblings(0) == []
    assert r.get_siblings(1) == [4, 7, 10]
    assert r.get_siblings(2) == [3]
    assert r.get_siblings(5) == [6]
    assert r.get_siblings(8) == [9]
    assert r.get_siblings(11) == [12]

    not_first_sibling = [3, 4, 6, 7, 9, 10, 12]
    for s in not_first_sibling:
        with pytest.raises(ValueError):
            r.get_siblings(s)

    with pytest.raises(IndexError):
        r.get_siblings(13)


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


def test_refine():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("01"))
    p.plan_refinement(1, ba.bitarray("10"))
    p.plan_refinement(2, ba.bitarray("11"))
    p.plan_refinement(4, ba.bitarray("11"))
    p.plan_refinement(6, ba.bitarray("01"))

    # p.apply_refinements() #TODO
    assert validate_descriptor(r.descriptor)


def test_refine_simplest():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 0]))
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(1, ba.bitarray("10"))
    p.plan_refinement(0, ba.bitarray("01"))

    assert p._planned_refinements.queue == [
        (1, ba.bitarray("01")),
        (2, ba.bitarray("10")),
    ]

    # don't do this at home -- call p.apply_refinements() directly
    p.populate_queue()
    assert p._planned_refinements.empty()
    assert (
        len(p._markers) == 2
        and all(p._markers[1] == [0, 1])
        and all(p._markers[2] == [1, 0])
    )
    assert p._upward_queue.queue == [(-1, 1), (-1, 2)]

    p.upwards_sweep()
    assert (
        len(p._markers) == 2
        and all(p._markers[1] == [0, 1])
        and all(p._markers[2] == [1, 0])
    )
    assert p._upward_queue.empty()

    new_descriptor = p.create_new_descriptor()
    assert new_descriptor._data == ba.bitarray("10010000100000")
    assert validate_descriptor(new_descriptor)


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
