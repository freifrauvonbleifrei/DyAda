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
from dyada.linearization import MortonOrderLinearization, SameIndexAs
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
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(
            2,
            ba.bitarray("01 00 10 10 00 00 10 00 00"),
        ),
    )
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

    ancestrybranch = AncestryBranch(
        discretization=discretization, starting_index=0, markers=markers
    )
    # step through ancestry branch
    # new index 0
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 0
    assert track_token == 0
    assert next_refinement == ba.frozenbitarray("11")
    assert np.array_equal(next_marker, np.array([1, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 1
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 1
    assert track_token == 1
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 2
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 1
    assert track_token == 2
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 3
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 2
    assert track_token == 3
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 4
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 3
    assert track_token == 4
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 5
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 4
    assert track_token == 5
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 6
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 4
    assert track_token == 6
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 7
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 3
    assert track_token == 7
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 8
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 5
    assert track_token == 8
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 9
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 5
    assert track_token == 9
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 10
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 2
    assert track_token == 10
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 11
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 7
    assert track_token == 11
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 12
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 8
    assert track_token == 12
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    try:
        ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)
    except AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships as e:
        assert e.missing_mapping == {
            1: {0},
            2: {0},
            3: {0, 3},
            4: {0, 3, 4},
            5: {0, 3, 7},
            6: {10, 11, 12},
        }


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
    ancestrybranch = AncestryBranch(
        discretization=discretization, starting_index=0, markers=markers
    )

    # step through ancestry branch
    # new index 0
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 0
    assert track_token == 0
    assert next_refinement == ba.frozenbitarray("111")
    assert np.array_equal(next_marker, np.array([1, 1, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)
    # new index 1
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 3
    assert track_token == 1
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 2
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 4
    assert track_token == 2
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 3
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 5
    assert track_token == 3
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 4
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 5
    assert track_token == 4
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 5
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 6
    assert track_token == 5
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 6
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 6
    assert track_token == 6
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 7
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 6
    assert track_token == 7
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 8
    current_old_index, track_token, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 6
    assert track_token == 8
    assert next_refinement == ba.frozenbitarray("000")
    assert np.array_equal(next_marker, np.array([0, 0, 0], dtype=np.int8))

    try:
        ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)
    except AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships as e:
        assert e.missing_mapping == {
            1: {0, 1, 2, 3, 4},
            2: {0, 1, 2},
            5: {0},
            6: {0},
        }


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
    assert p._index_mapping == [
        {0},
        *(set(), set(), set(), set(), set(), set(), set(), set()),
    ]
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
    assert p._index_mapping == [
        {0},
        {1},
        *(set(), set(), set(), set(), set(), set(), set()),
    ]

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
