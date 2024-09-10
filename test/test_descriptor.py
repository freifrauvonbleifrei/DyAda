import pytest
import bitarray as ba
from collections import deque
import numpy as np
from os.path import abspath

from dyada.descriptor import (
    generalized_ruler,
    LevelCounter,
    RefinementDescriptor,
    validate_descriptor,
)


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
        assert repr(r).startswith("RefinementDescriptor")


def test_zero_level():
    for i in range(1, 10):
        r = RefinementDescriptor(i, 0)
        assert len(r) == 1
        assert r.get_data().any() == False
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
            LevelCounter(
                level_increment=ba.frozenbitarray("00"),
                count_to_go_up=1,
            ),
            LevelCounter(
                level_increment=ba.frozenbitarray("11"),
                count_to_go_up=2,
            ),
            LevelCounter(
                level_increment=ba.frozenbitarray("01"),
                count_to_go_up=1,
            ),
        ]
    )
    # test the branch's repr
    assert (repr(r.get_branch(9, False)[0])).startswith("Branch")


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


def test_family_relations():
    r = RefinementDescriptor(2, [1, 2])

    families = {
        -1: [0],
        0: [1, 4, 7, 10],
        1: [2, 3],
        4: [5, 6],
        7: [8, 9],
        10: [11, 12],
    }
    for parent, siblings in families.items():
        for sibling in siblings:
            sibling_branch = r.get_branch(sibling, False)[0]
            assert r.get_parent(sibling_branch)[0] == parent
            assert r.get_siblings(sibling) == siblings
        assert r.get_children(parent) == siblings
    for leaf in (2, 3, 5, 6, 8, 9, 11, 12):
        assert r.get_children(leaf) == []

    with pytest.raises(IndexError):
        r.get_siblings(13)


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
