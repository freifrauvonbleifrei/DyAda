# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
import pytest
from os.path import abspath
from dyada.drawing import discretization_to_2d_ascii

from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.linearization import MortonOrderLinearization
from dyada.refinement import (
    PlannedAdaptiveRefinement,
)


def test_coarsen_simplest_2d():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(2, ba.bitarray("10 00 00")),
    )
    p = PlannedAdaptiveRefinement(discretization)
    assert (
        discretization_to_2d_ascii(discretization, resolution=(4, 2))
        == """\
_____
| | |
|_|_|"""
    )
    p.plan_coarsening(0, ba.bitarray("10"))
    new_discretization, _ = p.apply_refinements()
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("00")
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(4, 2))
        == """\
_____
|   |
|___|"""
    )

    # aaand transposed
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(2, ba.bitarray("01 00 00")),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_coarsening(0, ba.bitarray("01"))
    new_discretization, _ = p.apply_refinements()
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("00")


def test_coarsen_partly_2d():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(2, ba.bitarray("11 00 00 00 00")),
    )
    assert (
        discretization_to_2d_ascii(discretization, resolution=(4, 2))
        == """\
_____
|_|_|
|_|_|"""
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_coarsening(0, ba.bitarray("10"))
    new_discretization, index_mapping = p.apply_refinements(track_mapping="patches")
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(4, 2))
        == """\
_____
|___|
|___|"""
    )
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("01 00 00")
    expected_index_mapping = {0: {0}, 1: {0, 1}, 2: {0, 1}, 3: {0, 2}, 4: {0, 2}}
    assert index_mapping == [
        expected_index_mapping[i] for i in range(len(expected_index_mapping))
    ]

    p = PlannedAdaptiveRefinement(discretization)
    p.plan_coarsening(0, ba.bitarray("01"))
    new_discretization, index_mapping = p.apply_refinements(track_mapping="patches")
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(4, 2))
        == """\
_____
| | |
|_|_|"""
    )
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("10 00 00")

    expected_index_mapping = {0: {0}, 1: {0, 1}, 2: {0, 2}, 3: {0, 1}, 4: {0, 2}}
    assert index_mapping == [
        expected_index_mapping[i] for i in range(len(expected_index_mapping))
    ]


def test_coarsen_octree_first():
    for dimensionality in range(1, 5):
        desc_initial = RefinementDescriptor(dimensionality, [2] * dimensionality)
        discretization_initial = Discretization(
            MortonOrderLinearization(), desc_initial
        )
        # coarsen first parent
        all_coarsening = ba.bitarray("1" * dimensionality)
        coarsen_first_oct_plan = PlannedAdaptiveRefinement(discretization_initial)
        coarsen_first_oct_plan.plan_coarsening(1, all_coarsening)
        first_coarsened_discretization, coarsen_first_oct_mapping = (
            coarsen_first_oct_plan.apply_refinements(track_mapping="patches")
        )
        first_coarsened_descriptor = first_coarsened_discretization.descriptor
        assert first_coarsened_descriptor[1].count() == 0
        remaining_length = len(first_coarsened_descriptor) - 2
        assert (
            first_coarsened_descriptor[-remaining_length:]
            == desc_initial[-remaining_length:]
        )
        assert coarsen_first_oct_mapping[0] == {0}
        for i in range(1, 2**dimensionality + 2):
            assert coarsen_first_oct_mapping[i] == {1}
        for i in range(2**dimensionality + 2, len(desc_initial)):
            assert coarsen_first_oct_mapping[i] == {i - 2**dimensionality}


def test_coarsen_octree_all_2d():
    dimensionality = 2
    desc_initial = RefinementDescriptor(dimensionality, [2] * dimensionality)
    discretization_initial = Discretization(MortonOrderLinearization(), desc_initial)
    assert (
        discretization_to_2d_ascii(discretization_initial, resolution=(8, 4))
        == """\
_________
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|"""
    )

    # coarsen all parents
    all_coarsening = ba.bitarray("1" * dimensionality)
    coarsen_plan = PlannedAdaptiveRefinement(discretization_initial)
    coarsen_plan.plan_coarsening(1, all_coarsening)
    coarsen_plan.plan_coarsening(6, all_coarsening)
    coarsen_plan.plan_coarsening(11, all_coarsening)
    coarsen_plan.plan_coarsening(16, all_coarsening)
    coarsened_discretization, coarsen_mapping = coarsen_plan.apply_refinements(
        track_mapping="patches"
    )
    coarsened_descriptor = coarsened_discretization.descriptor
    assert coarsened_descriptor == RefinementDescriptor(
        dimensionality, [1] * dimensionality
    )
    assert (
        discretization_to_2d_ascii(coarsened_discretization, resolution=(4, 2))
        == """\
_____
|_|_|
|_|_|"""
    )
    expected_coarsen_mapping = {
        0: {0},
        1: {1},
        2: {1},
        3: {1},
        4: {1},
        5: {1},
        6: {2},
        7: {2},
        8: {2},
        9: {2},
        10: {2},
        11: {3},
        12: {3},
        13: {3},
        14: {3},
        15: {3},
        16: {4},
        17: {4},
        18: {4},
        19: {4},
        20: {4},
    }
    assert coarsen_mapping == [
        expected_coarsen_mapping[i] for i in range(len(expected_coarsen_mapping))
    ]


def test_flip_2d():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(2, ba.bitarray("01 00 00")),
    )
    assert (
        discretization_to_2d_ascii(discretization, resolution=(4, 2))
        == """\
_____
|___|
|___|"""
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "10")
    p.plan_refinement(1, "10")
    p.plan_coarsening(0, ba.bitarray("01"))
    new_discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("10 00 00")
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(4, 2))
        == """\
_____
| | |
|_|_|"""
    )

    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2},
        2: {0, 1, 2},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]


def test_flip_3d():
    # needs the info from test_coarsen_refine_stack_3d to be correct
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(3, ba.bitarray("011 000 000 000 000")),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "100")
    p.plan_refinement(1, "100")
    p.plan_refinement(2, "100")
    p.plan_refinement(3, "100")
    p.plan_coarsening(0, ba.bitarray("001"))
    new_discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("110 000 000 000 000")
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2},
        2: {0, 3, 4},
        3: {0, 1, 2},
        4: {0, 3, 4},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]


def test_coarsen_right_half_2d():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(2, ba.bitarray("11 00 00 00 00")),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_coarsening(0, ba.bitarray("01"))  # <- hierarchical index
    p.plan_refinement(0, "01")  # <- box/leaf index
    # p.plan_refinement(2, "01") # necessary?
    assert (
        discretization_to_2d_ascii(discretization, resolution=(4, 2))
        == """\
_____
|_|_|
|_|_|"""
    )
    new_discretization, index_mapping = p.apply_refinements(track_mapping="patches")
    new_descriptor = new_discretization.descriptor
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(4, 2))
        == """\
_____
|_| |
|_|_|"""
    )
    assert new_descriptor._data == ba.bitarray("10 01 00 00 00")
    expected_index_mapping = {
        0: {0},
        1: {1, 2},
        2: {0, 4},
        3: {1, 3},
        4: {0, 4},
    }
    assert index_mapping == [
        expected_index_mapping[i] for i in range(len(expected_index_mapping))
    ]


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
