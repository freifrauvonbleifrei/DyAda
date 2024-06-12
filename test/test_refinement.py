import pytest
import numpy as np
from os.path import abspath
from dyada.refinement import RefinementDescriptor, generalized_ruler


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


def test_zero_level():
    for i in range(1, 10):
        r = RefinementDescriptor(i, 0)
        assert len(r) == 1
        assert r.get_data().any() == False
        print(r.get_data())
        assert r.get_num_boxes() == 1


def test_one_level():
    for i in range(1, 2):
        r = RefinementDescriptor(i, 1)
        d = r.get_num_dimensions()
        assert d == i
        assert len(r) == 2**i + 1
        assert r.get_num_boxes() == 2**i
        assert r.get_data().count() == i
        assert r.get_data()[0] == 1
        # every d-block in data is either 0 or 1
        for i in range(0, len(r.get_data()), d):
            assert r.get_data()[i : i + r.get_num_dimensions()].count() in [0, d]


def test_six_d():
    for l in range(0, 4):
        r = RefinementDescriptor(6, l)
        assert r.get_num_dimensions() == 6
        # assert len(r) == 2**6 + 1
        assert r.get_num_boxes() == 2**(l*6)
        for i in range(0, len(r.get_data()), 6):
            assert r.get_data()[i : i + 6].count() in [0, 6]



if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
