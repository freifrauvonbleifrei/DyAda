# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import bitarray as ba
from os.path import abspath

from dyada.descriptor import RefinementDescriptor, validate_descriptor
from dyada.refinement import (
    Discretization,
    PlannedAdaptiveRefinement,
    apply_single_refinement,
)
from dyada.linearization import MortonOrderLinearization

from test_refinement import helper_check_mapping


def test_refine_3d_1():
    prependable_string = "110001000000001000000001000000"
    for round_number in range(4):
        descriptor = RefinementDescriptor.from_binary(
            3,
            ba.bitarray(
                prependable_string * round_number
                + "110001000000001010000000000001010000000000000"
            ),
        )
        validate_descriptor(descriptor)
        num_boxes_before = descriptor.get_num_boxes()
        discretization = Discretization(MortonOrderLinearization(), descriptor)

        new_discretization, index_mapping = apply_single_refinement(
            discretization, len(discretization) - 1, ba.bitarray("001")
        )

        new_descriptor = new_discretization.descriptor
        validate_descriptor(new_descriptor)
        assert new_descriptor.get_num_boxes() == num_boxes_before + 1
        helper_check_mapping(index_mapping, discretization, new_discretization)
        # make sure only the last box maps to two new boxes
        for b in range(descriptor.get_num_boxes()):
            assert len(index_mapping[b]) == 1 or (
                b == (len(discretization) - 1) and len(index_mapping[b]) == 2
            )


def test_refine_3d_2():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(3, ba.bitarray("001 000 100 000 000")),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "110")
    p.plan_refinement(1, "011")
    p.plan_refinement(2, "011")
    new_discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2, 3, 4},
        2: {0, 5, 8, 11, 14},
        3: {0, 5, 6, 7, 11, 12, 13},
        4: {0, 8, 9, 10, 14, 15, 16},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]
    helper_check_mapping(patch_mapping, discretization, new_discretization, False)


def test_refine_3d_3():
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("010 000 101 100 000 000 100 000 000 100 000 000 100 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "101")
    p.plan_refinement(1, "100")
    p.plan_refinement(2, "100")
    new_discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2, 13, 14},
        2: {0, 3, 10, 15, 18},
        3: {3, 4, 7},
        4: {4, 5, 6},
        5: {7, 8, 9},
        6: {10},
        7: {11},
        8: {12},
        9: {15},
        10: {16},
        11: {17},
        12: {18},
        13: {19},
        14: {20},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]
    helper_check_mapping(patch_mapping, discretization, new_discretization, False)


def test_refine_3d_4():
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("010 000 001 000 100 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "101")
    p.plan_refinement(1, "100")
    p.plan_refinement(2, "111")
    p.plan_refinement(3, "111")
    new_discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2, 5, 6},
        2: {0, 3, 4, 7, 16},
        3: {0, 3, 4},
        4: {0, 7, 16},
        5: {7, 8, 9, 10, 11, 12, 13, 14, 15},
        6: {16, 17, 18, 19, 20, 21, 22, 23, 24},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]
    helper_check_mapping(patch_mapping, discretization, new_discretization, False)


def test_refine_3d_5():
    # almost same beginning as 3d_4
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("001 010 100 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(2, "100")
    p.plan_refinement(3, "110")
    new_discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    expected_patch_mapping = {
        0: {0},
        1: {0, 3, 4},
        2: {0},
        3: {1},
        4: {2},
        5: {0, 3, 4},
        6: {0, 5, 6, 7, 8},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]
    helper_check_mapping(patch_mapping, discretization, new_discretization, False)


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
