# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import bitarray as ba
from dyada.descriptor import (
    RefinementDescriptor,
    validate_descriptor,
    find_uniqueness_violations,
)
from dyada.refinement import (
    Discretization,
    PlannedAdaptiveRefinement,
)
from dyada.drawing import discretization_to_2d_ascii
from dyada.linearization import MortonOrderLinearization
from test_refinement import helper_check_mapping


def test_downsplit_uncanonicalize_2d():
    descriptor = RefinementDescriptor.from_binary(2, ba.bitarray("11 00 00 00 00"))
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    ascii_before_and_after = """\
_____
|_|_|
|_|_|"""
    assert str(discretization) == ascii_before_and_after
    # Keep downsplit markers exactly as requested
    down_plan = PlannedAdaptiveRefinement(discretization)
    down_plan.plan_downsplit(0, ba.bitarray("10"))
    down_discretization, _ = down_plan.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert str(down_discretization) == ascii_before_and_after
    down_descriptor = down_discretization.descriptor
    assert down_descriptor == RefinementDescriptor.from_binary(
        2, ba.bitarray("01 10 00 00 10 00 00")
    )
    assert find_uniqueness_violations(down_descriptor) == [{0, 1, 4}]


def test_downsplit_child():
    descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 10 00 00 00 00 00")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    ascii_before_and_after = """\
_________
|___|___|
|_|_|___|"""
    assert str(discretization) == ascii_before_and_after

    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("10"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    # descriptor must have changed, e.g. starting with 01
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        2, ba.bitarray("01 10 10 00 00 00 10 00 00")
    )
    assert str(down_discretization) == ascii_before_and_after
    assert patch_mapping == [
        {0, 1, 6},
        {2},
        {3},
        {4},
        {5},
        {7},
        {8},
    ]
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )

    # again in the other direction
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("01"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        2, ba.bitarray("10 01 10 00 00 00 01 00 00")
    )
    assert str(down_discretization) == ascii_before_and_after
    assert patch_mapping == [
        {0, 1, 6},
        {2},
        {3},
        {4},
        {7},
        {5},
        {8},
    ]
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )


def test_downsplit_child_mixed_root_children_2d():
    descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 01 00 00 10 00 00 00 00")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    ascii_before_and_after = """\
_________
|   |   |
|___|___|
|___| | |
|___|_|_|"""
    assert str(discretization) == ascii_before_and_after

    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("10"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        2, ba.bitarray("01 10 01 00 00 10 00 00 10 00 00")
    )
    assert str(down_discretization) == ascii_before_and_after
    assert patch_mapping == [
        {0, 1, 8},
        {2},
        {3},
        {4},
        {5},
        {6},
        {7},
        {9},
        {10},
    ]
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )


def test_downsplit_root_3d():
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("111 000 000 000 000 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("100"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        3, ba.bitarray("011 100 000 000 100 000 000 100 000 000 100 000 000")
    )
    assert patch_mapping == [
        {0, 1, 4, 7, 10},
        {2},
        {3},
        {5},
        {6},
        {8},
        {9},
        {11},
        {12},
    ]
    assert descriptor.get_num_boxes() == down_discretization.descriptor.get_num_boxes()
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )


def test_downsplit_nonroot_3d_mixed_children():
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("111 110 010 000 000 000 000 000 000 000 000 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(1, ba.bitarray("100"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "111 010 100 010 000 000 000 100 000 000 000 000 000 000 000 000 000"
        ),
    )
    assert patch_mapping == [
        {0},
        {1, 2, 7},
        {3},
        {4},
        {5},
        {6},
        {8},
        {9},
        {10},
        {11},
        {12},
        {13},
        {14},
        {15},
        {16},
    ]
    assert descriptor.get_num_boxes() == down_discretization.descriptor.get_num_boxes()
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )


def test_downsplit_2dims_simultaneously_3d():
    # Split two dimensions at once from a fully-refined 3D root with leaf children.
    # When both grouped children can absorb the split, we merge directly
    # into those children instead of inserting intermediate nodes.
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("111 000 000 000 000 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("110"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        3, ba.bitarray("001 110 000 000 000 000 110 000 000 000 000")
    )
    assert patch_mapping == [
        {0, 1, 6},
        {2},
        {3},
        {4},
        {5},
        {7},
        {8},
        {9},
        {10},
    ]
    assert descriptor.get_num_boxes() == down_discretization.descriptor.get_num_boxes()
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )


def test_downsplit_merged_plans_3d():
    # Two separate plan_downsplit calls on the same node are OR-merged before
    # application and must produce the same result as a single combined call.
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("111 000 000 000 000 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("100"))
    p.plan_downsplit(0, ba.bitarray("010"))
    down_discretization, patch_mapping = p.apply_refinements(
        track_mapping="patches",
        sweep_mode="as_planned",
    )
    assert down_discretization.descriptor == RefinementDescriptor.from_binary(
        3, ba.bitarray("001 110 000 000 000 000 110 000 000 000 000")
    )
    assert patch_mapping == [
        {0, 1, 6},
        {2},
        {3},
        {4},
        {5},
        {7},
        {8},
        {9},
        {10},
    ]
    validate_descriptor(down_discretization.descriptor)
    helper_check_mapping(
        patch_mapping,
        discretization,
        down_discretization,
        mapping_indices_are_boxes=False,
    )


def test_downsplit_boxes_mapping_3d():
    # Downsplit only restructures intermediate nodes; it never splits or merges
    # leaf boxes, so the boxes mapping must be the identity.
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("111 000 000 000 000 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("100"))
    down_discretization, boxes_mapping = p.apply_refinements(
        track_mapping="boxes",
        sweep_mode="as_planned",
    )
    assert boxes_mapping == [{i} for i in range(8)]


def test_downsplit_only_dim_not_implemented():
    # Downsplitting the sole remaining refined dimension would leave the node with no
    # refinement at all; this path is not yet supported.
    descriptor = RefinementDescriptor.from_binary(2, ba.bitarray("10 00 00"))
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("10"))
    with pytest.raises(ValueError):
        p.apply_refinements(sweep_mode="as_planned")


def test_downsplit_nested_2d():
    """Downsplit both root and its child in the same pass."""
    descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 11 00 00 00 00 00 00 00")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert descriptor.get_num_boxes() == 7
    ascii_before_and_after = """\
_________
|   |   |
|___|___|
|_|_|   |
|_|_|___|"""
    assert (
        discretization_to_2d_ascii(discretization, resolution=(8, 4))
        == ascii_before_and_after
    )

    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("10"))
    p.plan_downsplit(1, ba.bitarray("10"))
    new_disc, mapping = p.apply_refinements(
        track_mapping="patches", sweep_mode="as_planned"
    )
    new_desc = new_disc.descriptor

    # Leaf count and spatial discretization must be preserved
    assert new_desc.get_num_boxes() == 7
    assert validate_descriptor(new_desc)
    assert (
        discretization_to_2d_ascii(new_disc, resolution=(8, 4))
        == ascii_before_and_after
    )
    assert new_desc == RefinementDescriptor.from_binary(
        2, ba.bitarray("01 10 01 10 00 00 10 00 00 00 10 00 00")
    )


def test_downsplit_nested_3d():
    descriptor = RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "111 111 000 000 000 000 000 000 000 000 000 000 000 000 000 000 000"
        ),
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    num_boxes = descriptor.get_num_boxes()

    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("100"))
    p.plan_downsplit(1, ba.bitarray("100"))
    new_disc, _ = p.apply_refinements(track_mapping="patches", sweep_mode="as_planned")
    new_desc = new_disc.descriptor

    assert new_desc.get_num_boxes() == num_boxes
    assert validate_descriptor(new_desc)
    # Root becomes 011, each child gets 100 intermediate with nested downsplit
    assert new_desc == RefinementDescriptor.from_binary(
        3,
        ba.bitarray(
            "011"
            " 100 011 100 000 000 100 000 000 100 000 000 100 000 000 000"
            " 100 000 000"
            " 100 000 000"
            " 100 000 000"
        ),
    )


def test_downsplit_invalid_track_mapping():
    descriptor = RefinementDescriptor.from_binary(2, ba.bitarray("11 00 00 00 00"))
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_downsplit(0, ba.bitarray("10"))
    with pytest.raises(ValueError, match="track_mapping"):
        p.apply_refinements(track_mapping="invalid", sweep_mode="as_planned")
