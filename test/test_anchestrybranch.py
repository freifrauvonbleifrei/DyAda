# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import bitarray as ba
import pytest
import numpy as np
from os.path import abspath

from dyada.ancestrybranch import AncestryBranch
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import Discretization
from dyada.drawing import discretization_to_2d_ascii
from dyada.linearization import MortonOrderLinearization, TrackToken
from dyada.refinement import PlannedAdaptiveRefinement, get_defaultdict_for_markers


def advance_or_grow(
    ancestrybranch: AncestryBranch, next_refinement: ba.frozenbitarray
) -> AncestryBranch:
    is_leaf = next_refinement.count() == 0
    if is_leaf:
        ancestrybranch.advance()
    else:
        # on non-leaves, we grow the branch
        ancestrybranch.grow(next_refinement)
    return ancestrybranch


def test_ancestry_branch_2d_6():
    # inspired by test_refine_2d_6
    descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("01 00 10 10 00 00 10 00 00")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert (
        discretization_to_2d_ascii(discretization, resolution=(8, 4))
        == """\
_________
| | | | |
|_|_|_|_|
|       |
|_______|"""
    )
    markers = get_defaultdict_for_markers(
        discretization.descriptor.get_num_dimensions()
    )
    markers[0] = np.array([1, 0], dtype=np.int8)
    markers[6] = np.array([-1, 0], dtype=np.int8)

    ancestrybranch = AncestryBranch(discretization, starting_index=0, markers=markers)
    # step through ancestry branch
    expected_return_values = [
        (0, TrackToken(0), ba.frozenbitarray("11"), np.array([1, 0], dtype=np.int8)),
        (1, TrackToken(1), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (1, TrackToken(2), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (2, TrackToken(3), ba.frozenbitarray("10"), np.array([0, 0], dtype=np.int8)),
        (3, TrackToken(4), ba.frozenbitarray("10"), np.array([0, 0], dtype=np.int8)),
        (4, TrackToken(5), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (4, TrackToken(6), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (3, TrackToken(7), ba.frozenbitarray("10"), np.array([0, 0], dtype=np.int8)),
        (5, TrackToken(8), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (5, TrackToken(9), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (2, TrackToken(10), ba.frozenbitarray("10"), np.array([0, 0], dtype=np.int8)),
        (7, TrackToken(11), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (8, TrackToken(12), ba.frozenbitarray("00"), np.array([0, 0], dtype=np.int8)),
        (None),
    ]
    for expected in expected_return_values:
        # new index given by track_token
        current_old_index, track_token, next_refinement, next_marker = (
            ancestrybranch.get_current_location_info()
        )
        assert (current_old_index, track_token, next_refinement) == expected[0:-1]
        assert np.array_equal(next_marker, expected[-1])
        try:
            ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)
        except AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships as e:
            assert track_token == TrackToken(12)  # make sure it happens at end
            assert e.missing_mapping == {
                1: {TrackToken(0)},
                2: {TrackToken(0)},
                3: {TrackToken(0), TrackToken(3)},
                4: {TrackToken(0), TrackToken(3), TrackToken(4)},
                5: {TrackToken(0), TrackToken(3), TrackToken(7)},
                6: {TrackToken(10), TrackToken(11), TrackToken(12)},
            }
            break


def test_ancestrybranch_3d_5():
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("001 010 100 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)

    markers = get_defaultdict_for_markers(
        discretization.descriptor.get_num_dimensions()
    )
    markers[0] = np.array([1, 1, 0], dtype=np.int8)
    markers[1] = np.array([0, -1, 0], dtype=np.int8)
    markers[2] = np.array([-1, 0, 0], dtype=np.int8)
    ancestrybranch = AncestryBranch(discretization, starting_index=0, markers=markers)

    TT = TrackToken  # type alias for brevity
    leaf_no_marker = (ba.frozenbitarray("000"), np.array([0, 0, 0], dtype=np.int8))
    # step through ancestry branch
    expected_return_values = [
        (
            0,
            TT(0),
            ba.frozenbitarray("111"),
            np.array([1, 1, 0], dtype=np.int8),
        ),
        (3, TT(1), *leaf_no_marker),
        (4, TT(2), *leaf_no_marker),
        (5, TT(3), *leaf_no_marker),
        (5, TT(4), *leaf_no_marker),
        (6, TT(5), *leaf_no_marker),
        (6, TT(6), *leaf_no_marker),
        (6, TT(7), *leaf_no_marker),
        (6, TT(8), *leaf_no_marker),
        (None),
    ]
    for expected in expected_return_values:
        current_old_index, track_token, next_refinement, next_marker = (
            ancestrybranch.get_current_location_info()
        )
        assert (current_old_index, track_token, next_refinement) == expected[0:-1]
        assert np.array_equal(next_marker, expected[-1])
        try:
            ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)
        except AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships as e:
            assert track_token == TT(8)
            assert e.missing_mapping == {
                1: {TT(0), TT(1), TT(2), TT(3), TT(4)},
                2: {TT(0), TT(1), TT(2)},
                5: {TT(0)},
                6: {TT(0)},
            }
            break


def test_ancestrybranch_coarsen_nested_2d():
    descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 00 00 01 00 00 01 00 00")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)

    markers = get_defaultdict_for_markers(
        discretization.descriptor.get_num_dimensions()
    )
    markers[0] = np.array([-1, 0], dtype=np.int8)
    ancestrybranch = AncestryBranch(discretization, starting_index=0, markers=markers)

    TT = TrackToken
    bba = ba.bitarray
    fba = ba.frozenbitarray
    no_marker = np.array([0, 0], dtype=np.int8)
    leaf = fba("00")
    # step through ancestry branch
    expected_return_values = [
        (0, TT(0), fba("01"), np.array([-1, 0], dtype=np.int8)),
        (1, TT(1), leaf, no_marker),
        (3, TT(2), fba("01"), no_marker),
        (4, TT(3), bba("00"), no_marker),
        (5, TT(4), leaf, no_marker),
        (None),
    ]
    for expected in expected_return_values:
        current_old_index, track_token, next_refinement, next_marker = (
            ancestrybranch.get_current_location_info()
        )
        assert (current_old_index, track_token, next_refinement) == expected[0:-1]
        assert np.array_equal(next_marker, expected[-1])
        try:
            ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)
        except AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships as e:
            assert track_token == TT(4)
            assert e.missing_mapping == {
                1: {TT(0)},
                2: {TT(1)},
                3: {TT(0)},
                4: {TT(0), TT(2)},
                5: {TT(0), TT(2)},
                6: {TT(2)},
                7: {TT(3)},
                8: {TT(4)},
            }
            break


def test_modified_branch_generator_2d_6():
    descriptor = RefinementDescriptor.from_binary(
        2,
        ba.bitarray("01 00 10 10 00 00 10 00 00"),
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert (
        discretization_to_2d_ascii(discretization, resolution=(8, 4))
        == """\
_________
| | | | |
|_|_|_|_|
|       |
|_______|"""
    )
    p = PlannedAdaptiveRefinement(discretization)
    p._markers[0] = np.array([1, 0], dtype=np.int8)
    p._markers[6] = np.array([-1, 0], dtype=np.int8)
    new_descriptor = RefinementDescriptor(2)
    new_descriptor._data = ba.bitarray()
    p._index_mapping = [set() for _ in range(len(discretization.descriptor))]
    # call like in add_refined_data
    generator = p.modified_branch_generator(
        starting_index=0, new_descriptor=new_descriptor
    )
    zeroth_refinement = next(generator)
    assert zeroth_refinement == p.Refinement(
        p.Refinement.Type.CopyOver, 0, ba.frozenbitarray("11"), None
    )
    p.extend_descriptor_and_track_indices(
        new_descriptor, zeroth_refinement.old_index, zeroth_refinement.new_refinement
    )
    assert p._index_mapping == [{0}, *[set()] * 8]
    first_refinement = next(generator)
    assert first_refinement.type == p.Refinement.Type.ExpandLeaf
    assert first_refinement.old_index == 1
    assert first_refinement.new_refinement == ba.bitarray("00")
    assert np.array_equal(
        first_refinement.marker_or_ancestor, np.array([0, 0], dtype=np.int8)
    )
    p.extend_descriptor_and_track_indices(
        new_descriptor,
        first_refinement.old_index,
        ba.bitarray("00"),
    )
    assert p._index_mapping == [{0}, {1}, *[set()] * 7]

    second_refinement = next(generator)
    assert second_refinement.type == p.Refinement.Type.ExpandLeaf
    assert second_refinement.old_index == 1
    assert second_refinement.new_refinement == ba.bitarray("00")
    assert np.array_equal(
        second_refinement.marker_or_ancestor, np.array([0, 0], dtype=np.int8)
    )
    p.extend_descriptor_and_track_indices(
        new_descriptor,
        first_refinement.old_index,
        ba.bitarray("00"),
    )

    third_refinement = next(generator)
    assert third_refinement == p.Refinement(
        p.Refinement.Type.CopyOver, 2, ba.bitarray("10"), None
    )
    p.extend_descriptor_and_track_indices(
        new_descriptor,
        third_refinement.old_index,
        third_refinement.new_refinement,
    )

    fourth_refinement = next(generator)
    assert fourth_refinement == p.Refinement(
        p.Refinement.Type.CopyOver, 3, ba.bitarray("10"), None
    )
    p.extend_descriptor_and_track_indices(
        new_descriptor,
        fourth_refinement.old_index,
        fourth_refinement.new_refinement,
    )
    assert new_descriptor._data == ba.bitarray("11 00 00 10 10")


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
