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
from dyada.linearization import MortonOrderLinearization
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


def test_ancestry_branch():
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
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 0
    assert intermediate_generation == {0}
    assert next_refinement == ba.frozenbitarray("11")
    assert np.array_equal(next_marker, np.array([1, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 1
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 1
    assert intermediate_generation == {1}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 2
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 1
    assert intermediate_generation == {1}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 3
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 2
    assert intermediate_generation == {2, 3}
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 4
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 3
    assert intermediate_generation == {3, 4}
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 5
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 4
    assert intermediate_generation == {4}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 6
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 4
    assert intermediate_generation == {4}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 7
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 3
    assert intermediate_generation == {3, 5}
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 8
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 5
    assert intermediate_generation == {5}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 9
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 5
    assert intermediate_generation == {5}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 10
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 2
    assert intermediate_generation == {2, 6}
    assert next_refinement == ba.frozenbitarray("10")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 11
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 7
    assert intermediate_generation == {7}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)

    # new index 12
    current_old_index, intermediate_generation, next_refinement, next_marker = (
        ancestrybranch.get_current_location_info()
    )
    assert current_old_index == 8
    assert intermediate_generation == {8}
    assert next_refinement == ba.frozenbitarray("00")
    assert np.array_equal(next_marker, np.array([0, 0], dtype=np.int8))
    with pytest.raises(AncestryBranch.WeAreDoneAndHereAreTheMissingRelationships):
        ancestrybranch = advance_or_grow(ancestrybranch, next_refinement)


def test_modified_branch_generator():
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
    p = PlannedAdaptiveRefinement(discretization)
    p._markers[0] = np.array([1, 0], dtype=np.int8)
    p._markers[6] = np.array([-1, 0], dtype=np.int8)
    new_descriptor = RefinementDescriptor(2)
    new_descriptor._data = ba.bitarray()
    p._index_mapping = [set() for _ in range(len(discretization.descriptor))]
    # call like in add_refined_data
    generator = p.modified_branch_generator(starting_index=0)
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

    tracking_refinement = next(generator)
    assert tracking_refinement == p.Refinement(p.Refinement.Type.TrackOnly, 0, None, 0)
    p.track_indices(
        tracking_refinement.old_index, tracking_refinement.marker_or_ancestor
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


    tracking_refinement = next(generator)
    assert tracking_refinement == p.Refinement(p.Refinement.Type.TrackOnly, 1, None, 1)
    p.track_indices(
        tracking_refinement.old_index, tracking_refinement.marker_or_ancestor
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

    tracking_refinement = next(generator)
    assert tracking_refinement == p.Refinement(p.Refinement.Type.TrackOnly, 1, None, 1)
    p.track_indices(
        tracking_refinement.old_index, tracking_refinement.marker_or_ancestor
    )
    assert p._index_mapping == [
        {0},
        {1, 2},
        *(set(), set(), set(), set(), set(), set(), set()),
    ]

    third_refinement = next(generator)
    assert third_refinement == p.Refinement(
        p.Refinement.Type.CopyOver, 2, ba.bitarray("10"), None
    )
    p.extend_descriptor_and_track_indices(
        new_descriptor,
        third_refinement.old_index,
        third_refinement.new_refinement,
    )
    tracking_refinement = next(generator)
    assert tracking_refinement == p.Refinement(p.Refinement.Type.TrackOnly, 2, None, 3)
    p.track_indices(
        tracking_refinement.old_index, tracking_refinement.marker_or_ancestor
    )
    assert p._index_mapping == [
        {0},
        {1, 2},
        {3},
        *(set(), set(), set(), set(), set(), set()),
    ]

    tracking_refinement = next(generator)
    assert tracking_refinement == p.Refinement(p.Refinement.Type.TrackOnly, 3, None, 3)
    p.track_indices(
        tracking_refinement.old_index, tracking_refinement.marker_or_ancestor
    )
    assert p._index_mapping == [
        {0},
        {1, 2},
        {3},
        {3},
        *(set(), set(), set(), set(), set()),
    ]

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

    tracking_refinement = next(generator)
    assert tracking_refinement == p.Refinement(p.Refinement.Type.TrackOnly, 3, None, 3)
    p.track_indices(
        tracking_refinement.old_index, tracking_refinement.marker_or_ancestor
    )
    assert p._index_mapping == [
        {0},
        {1, 2},
        {3},
        {3, 4},
        *(set(), set(), set(), set(), set()),
    ]


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
