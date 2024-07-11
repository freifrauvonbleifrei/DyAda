import pytest
import bitarray as ba
import numpy as np
from os.path import abspath

from dyada.refinement import (
    generalized_ruler,
    RefinementDescriptor,
    Refinement,
    validate_descriptor,
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
        r.get_level(len(r))
    with pytest.raises(IndexError):
        r.get_level(-1)
    assert np.array_equal(r.get_level(0), [0, 0, 0, 0])
    assert np.array_equal(r.get_level(1), [1, 1, 1, 1])
    assert np.array_equal(r.get_level(2), [2, 2, 2, 2])
    for i in range(0, 16):
        assert np.array_equal(r.get_level(3 + i), [3, 3, 3, 3])
        assert np.array_equal(r.get_level(len(r) - i - 1), [3, 3, 3, 3])
    assert np.array_equal(r.get_level(len(r) - 16 - 1), [2, 2, 2, 2])


def test_get_level_index():
    r = Refinement(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    level, index = r.get_level_index(0)
    assert np.array_equal(level, [0, 0]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(1)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(2)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(3)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(4)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(5)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(6)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(7)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(8)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 2])
    level, index = r.get_level_index(9)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 3])
    level, index = r.get_level_index(10)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(11)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 2])
    level, index = r.get_level_index(12)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 3])
    with pytest.raises(IndexError):
        r.get_level_index(13)


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
