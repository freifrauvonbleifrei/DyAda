# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import Counter
import pytest
import bitarray as ba
import numpy as np
from os.path import abspath

from dyada.descriptor import (
    RefinementDescriptor,
    validate_descriptor,
    hierarchical_to_box_index_mapping,
    find_uniqueness_violations,
)
from dyada.discretization import (
    coordinates_from_box_index,
    coordinates_from_index,
)
from dyada.drawing import discretization_to_2d_ascii
from dyada.refinement import (
    Discretization,
    PlannedAdaptiveRefinement,
    apply_single_refinement,
    get_defaultdict_for_markers,
    normalize_discretization,
)
from dyada.linearization import MortonOrderLinearization, single_bit_set_gen


def test_refine_2d_only_leaves():
    desc_initial = RefinementDescriptor(2, [1, 2])
    discretization = Discretization(MortonOrderLinearization(), desc_initial)
    p = PlannedAdaptiveRefinement(discretization)
    refinements_to_apply = [
        (0, np.array([0, 1])),
        (0, np.array([0, 1])),
        (1, np.array([1, 0])),
        (3, np.array([3, 2])),
        (2, np.array([1, 1])),
        (4, np.array([1, 1])),
        (6, np.array([0, 1])),
    ]

    for box_index, refinement in refinements_to_apply:
        p.plan_refinement(box_index, refinement)

    new_discretization, _ = p.apply_refinements()
    assert validate_descriptor(new_discretization.descriptor)


def test_refine_3d_only_leaves():
    descriptor = RefinementDescriptor(3, [1, 0, 1])
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "101")
    p.plan_refinement(1, "001")
    p.plan_refinement(2, "010")
    new_discretization, _ = p.apply_refinements()
    new_descriptor = new_discretization.descriptor
    assert new_descriptor.get_num_boxes() == 9
    assert validate_descriptor(new_descriptor)


def helper_check_mapping(
    index_mapping: list[set[int]],
    old_discretization: Discretization,
    new_discretization: Discretization,
    mapping_indices_are_boxes=True,
    tested_refinement=None,
):
    old_descriptor = old_discretization.descriptor
    new_descriptor = new_discretization.descriptor
    count_new_indices: Counter = Counter()
    for value in index_mapping:
        count_new_indices.update(value)
    if mapping_indices_are_boxes:
        assert len(index_mapping) == old_descriptor.get_num_boxes()
        assert set() not in index_mapping
        assert sorted(count_new_indices.elements()) == list(
            range(new_descriptor.get_num_boxes())
        )
    else:
        assert len(index_mapping) == len(old_descriptor)
        assert set() not in index_mapping
        assert set(count_new_indices.elements()) == set(range(len(new_descriptor)))
    for b, mapped_to_indices in enumerate(index_mapping):
        if len(mapped_to_indices) == 1:
            # make sure the coordinates are correct
            (mapped_to_index,) = mapped_to_indices
            if not (
                coordinates_from_index(
                    new_discretization, mapped_to_index, mapping_indices_are_boxes
                )
                == coordinates_from_index(
                    old_discretization, b, mapping_indices_are_boxes
                )
            ):
                print(
                    f"mapping failed for box {b} with index {mapped_to_index}"
                    " and coordinates {coordinates_from_box_index(old_discretization, b)} "
                    "-> {coordinates_from_box_index(new_discretization, mapped_to_index)}"
                )
                print(f"old descriptor: {old_descriptor}")
                print(f"new descriptor: {new_descriptor}")
                if tested_refinement is not None:
                    print(f"tested refinement: {tested_refinement}")
            if mapping_indices_are_boxes:
                # otherwise, there may be smaller, now-deleted patches as well
                assert coordinates_from_index(
                    new_discretization, mapped_to_index, mapping_indices_are_boxes
                ) == coordinates_from_index(
                    old_discretization, b, mapping_indices_are_boxes
                )
        else:
            pass
            # TODO: what are the correct assumptions?
            # old_interval = coordinates_from_index(
            #     old_discretization, b, mapping_indices_are_boxes
            # )
            # for new_index in mapped_to_indices:
            #     new_interval = coordinates_from_index(
            #         new_discretization, new_index, mapping_indices_are_boxes
            #     )
            #     assert np.all(
            #         old_interval.lower_bound <= new_interval.lower_bound
            #     ) and np.all(
            #         new_interval.lower_bound <= old_interval.upper_bound
            #     )  # type: ignore
            #     assert np.all(
            #         old_interval.lower_bound <= new_interval.upper_bound
            #     ) and np.all(
            #         new_interval.upper_bound <= old_interval.upper_bound
            #     )  # type: ignore


def test_refine_simplest_not_only_leaves():
    discretization = Discretization(
        MortonOrderLinearization(), RefinementDescriptor(2, [1, 0])
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "01")

    assert len(p._planned_refinements) == 1
    assert p._planned_refinements[0][0] == 1
    assert all(p._planned_refinements[0][-1] == [0, 1])

    # don't do this at home -- call p.apply_refinements() directly
    p.populate_queue()
    assert len(p._planned_refinements) == 0
    assert len(p._markers) == 1 and all(p._markers[1] == [0, 1])
    assert p._upward_queue.queue == [(-1, 1)]

    p.upwards_sweep()
    assert len(p._markers) == 1 and all(p._markers[1] == [0, 1])
    assert p._upward_queue.empty()

    new_discretization, _ = p.create_new_discretization()
    assert new_discretization.descriptor._data == ba.bitarray("1001000000")
    assert validate_descriptor(new_discretization.descriptor)

    p = PlannedAdaptiveRefinement(new_discretization)
    p.plan_refinement(2, "01")

    new_discretization_2, _ = p.apply_refinements()
    assert new_discretization_2.descriptor == RefinementDescriptor(2, [1, 1])
    assert validate_descriptor(new_discretization_2.descriptor)


def test_refine_simplest_grandchild_split():
    discretization = Discretization(
        MortonOrderLinearization(), RefinementDescriptor(2, [1, 0])
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "01")
    p.plan_refinement(1, "10")

    assert len(p._planned_refinements) == 2
    assert p._planned_refinements[0][0] == 1
    assert all(p._planned_refinements[0][-1] == [0, 1])
    assert p._planned_refinements[1][0] == 2
    assert all(p._planned_refinements[1][-1] == [1, 0])

    # don't do this at home -- call p.apply_refinements() directly
    p.populate_queue()
    assert len(p._planned_refinements) == 0
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

    new_discretization, index_mapping = p.create_new_discretization(
        track_mapping="boxes"
    )
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("10010000100000")
    assert validate_descriptor(new_descriptor)
    helper_check_mapping(index_mapping, discretization, new_discretization)

    # plot_tree_tikz(new_descriptor, filename="simplest_grandchild_split_before")
    # plot_all_boxes_2d(Discretization(MortonOrderLinearization(), new_descriptor), labels="patches")
    p = PlannedAdaptiveRefinement(new_discretization)
    p.plan_refinement(2, ba.bitarray("01"))
    p.plan_refinement(3, ba.bitarray("01"))
    assert len(p._planned_refinements) == 2
    assert p._planned_refinements[0][0] == 5
    assert all(p._planned_refinements[0][-1] == [0, 1])
    assert p._planned_refinements[1][0] == 6
    assert all(p._planned_refinements[1][-1] == [0, 1])

    p.populate_queue()
    p.upwards_sweep()
    assert (
        len(p._markers) == 2
        and all(p._markers[0] == [0, 1])
        and all(p._markers[1] == [0, -1])
    )
    assert p._upward_queue.empty()
    new_discretization_2, index_mapping_2 = p.create_new_discretization(
        track_mapping="patches"
    )
    new_descriptor_2 = new_discretization_2.descriptor
    assert new_descriptor_2._data == ba.bitarray("110010000000100000")
    assert validate_descriptor(new_descriptor_2)
    helper_check_mapping(
        index_mapping_2,
        new_discretization,
        new_discretization_2,
        mapping_indices_are_boxes=False,
    )


def test_refine_grandchild_split():
    p = PlannedAdaptiveRefinement(
        Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 1]))
    )
    # prepare the initial state
    p.plan_refinement(0, ba.bitarray("01"))
    p.plan_refinement(2, ba.bitarray("10"))
    discretization, _ = p.apply_refinements()
    new_discretization, _ = apply_single_refinement(discretization, 1)
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(16, 8))
        == """\
_________________
|   |   |       |
|   |   |       |
|   |   |       |
|___|___|_______|
|___|___|       |
|___|___|       |
|       |       |
|_______|_______|"""
    )
    new_descriptor = new_discretization.descriptor
    four_branch = new_descriptor.get_branch(4, False)[0]
    assert new_descriptor.get_ancestry(four_branch) == [0, 1, 3]
    p = PlannedAdaptiveRefinement(new_discretization)

    # the actual test
    p.plan_refinement(0, ba.bitarray("10"))
    new_discretization, box_mapping = p.apply_refinements(track_mapping="boxes")
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(16, 8))
        == """\
_________________
|   |   |       |
|   |   |       |
|   |   |       |
|___|___|_______|
|___|___|       |
|___|___|       |
|   |   |       |
|___|___|_______|"""
    )
    new_descriptor = new_discretization.descriptor
    assert validate_descriptor(new_descriptor)
    assert new_descriptor._data == ba.bitarray("111100000100000100000010000000")

    # test the mapping of the boxes
    former_to_now = [
        {0, 1},
        {2},
        {4},
        {3},
        {5},
        {6},
        {7},
        {8},
        {9},
    ]
    assert box_mapping == former_to_now


def test_refine_multi_grandchild_split():
    discretization = Discretization(
        MortonOrderLinearization(), RefinementDescriptor(2, [2, 0])
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(2, ba.bitarray("10"))
    p.plan_refinement(3, ba.bitarray("01"))
    discretization, _ = p.apply_refinements()

    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(2, ba.bitarray("01"))
    p.plan_refinement(3, ba.bitarray("11"))
    p.plan_refinement(3, ba.bitarray("10"))
    p.plan_refinement(3, ba.bitarray("10"))

    new_discretization, index_mapping = p.apply_refinements(track_mapping="boxes")
    assert new_discretization.descriptor._data == ba.bitarray(
        "10 10 00 00 11 10 00 10 10 10 00 00"
        "10 00 00 10 10 00 00 10 00 00 00 10"
        "00 10 10 10 00 00 10 00 00 10 10 00 00 10 00 00 00"
    )
    helper_check_mapping(index_mapping, discretization, new_discretization)


def test_refine_fully():
    for d in range(1, 6):
        for level in range(1, 2):
            descriptor = RefinementDescriptor(d, level)
            discretization = Discretization(MortonOrderLinearization(), descriptor)
            p = PlannedAdaptiveRefinement(discretization)
            for i in range(1, descriptor.get_num_boxes() - 1):
                # do octree refinement
                p.plan_refinement(i, ba.bitarray("1" * d))
            new_discretization, _ = p.apply_refinements()
            assert validate_descriptor(new_discretization.descriptor)
            # now also refine the first and last box
            p = PlannedAdaptiveRefinement(new_discretization)
            p.plan_refinement(0, ba.bitarray("1" * d))
            p.plan_refinement(
                new_discretization.descriptor.get_num_boxes() - 1, ba.bitarray("1" * d)
            )

            new_discretization, _ = p.apply_refinements()
            assert validate_descriptor(new_discretization.descriptor)
            regular_descriptor = RefinementDescriptor(d, level + 1)
            assert new_discretization.descriptor == regular_descriptor


def test_refine_2d_1():
    prependable_string = "10 01 00 00"
    descriptor = RefinementDescriptor.from_binary(
        2,
        ba.bitarray(prependable_string + "10 11 00 00 00 00 00"),
    )
    validate_descriptor(descriptor)
    correct_descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 00 10 10 00 00 00 00 10 10 00 00 00")
    )
    validate_descriptor(correct_descriptor)

    num_boxes_before = descriptor.get_num_boxes()
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert (
        discretization_to_2d_ascii(discretization, resolution=(16, 8))
        == """\
_________________
|       | | |   |
|       | | |   |
|       | | |   |
|_______|_|_|   |
|       | | |   |
|       | | |   |
|       | | |   |
|_______|_|_|___|"""
    )

    new_discretization, index_mapping = apply_single_refinement(
        discretization, len(discretization) - 1, ba.bitarray("01"), "patches"
    )
    new_discretization_, box_mapping = apply_single_refinement(
        discretization, len(discretization) - 1, ba.bitarray("01")
    )
    assert new_discretization == new_discretization_
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(16, 8))
        == """\
_________________
|       | | |   |
|       | | |   |
|       | | |   |
|_______|_|_|___|
|       | | |   |
|       | | |   |
|       | | |   |
|_______|_|_|___|"""
    )

    new_descriptor = new_discretization.descriptor
    validate_descriptor(new_descriptor)
    assert new_descriptor.get_num_boxes() == num_boxes_before + 1
    assert new_descriptor == correct_descriptor
    expected_box_mapping = {
        0: {0},
        1: {4},
        2: {1},
        3: {2},
        4: {5},
        5: {6},
        6: {3, 7},
    }
    assert box_mapping == list(expected_box_mapping.values())
    expected_index_mapping = {
        0: {0},
        1: {0, 1, 7},
        2: {1},
        3: {7},
        4: {0, 2, 8},
        5: {0, 2, 3, 8, 9},
        6: {4},
        7: {5},
        8: {10},
        9: {11},
        10: {0, 2, 8, 6, 12},
    }
    assert index_mapping == list(expected_index_mapping.values())


def test_refine_2d_2():
    descriptor = RefinementDescriptor.from_binary(
        2,
        ba.bitarray("11 00 00 10 00 00 10 00 11 00 00 00 00"),
    )
    validate_descriptor(descriptor)
    num_boxes_before = descriptor.get_num_boxes()
    assert num_boxes_before == 9
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert (
        discretization_to_2d_ascii(discretization, resolution=(16, 8))
        == """\
_________________
|   |   |   | | |
|   |   |   |_|_|
|   |   |   | | |
|___|___|___|_|_|
|       |       |
|       |       |
|       |       |
|_______|_______|"""
    )

    p = PlannedAdaptiveRefinement(discretization)
    refinements = [
        (3, ba.bitarray("11")),
        (4, ba.bitarray("01")),
    ]
    for box, refinement in refinements:
        p.plan_refinement(box, refinement)
    assert len(p._planned_refinements) == 2
    assert p._planned_refinements[0][0] == 5
    assert all(p._planned_refinements[0][-1] == [1, 1])
    assert p._planned_refinements[1][0] == 7
    assert all(p._planned_refinements[1][-1] == [0, 1])
    p.populate_queue()
    p.upwards_sweep()
    assert (
        len(p._markers) == 3
        and all(p._markers[5] == [1, 1])
        and all(p._markers[6] == [0, 1])
        and all(p._markers[8] == [0, -1])
    )
    assert p._upward_queue.empty()
    new_discretization, index_mapping = p.create_new_discretization(
        track_mapping="boxes"
    )
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(16, 8))
        == """\
_________________
|   | | |   | | |
|   |_|_|___|_|_|
|   | | |   | | |
|___|_|_|___|_|_|
|       |       |
|       |       |
|       |       |
|_______|_______|"""
    )
    validate_descriptor(new_discretization.descriptor)

    descriptor_expected = RefinementDescriptor.from_binary(
        2,
        ba.bitarray("11 00 00 10 00 11 00 00 00 00 11 00 10 00 00 00 10 00 00"),
    )
    assert new_discretization.descriptor == descriptor_expected

    helper_check_mapping(index_mapping, discretization, new_discretization)


def test_refine_2d_3():
    """Masado's example"""
    descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("10 01 00 00 10 01 00 00 00")
    )
    assert find_uniqueness_violations(descriptor) == []

    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(2, "01")
    p.plan_refinement(4, "02")
    non_normalized_discretization, _ = p.apply_refinements()
    ascii_before_and_after = """\
_________________
|       |   |   |
|       |   |___|
|       |   |   |
|_______|___|___|
|       |   |   |
|       |___|___|
|       |   |   |
|_______|___|___|"""
    assert (
        discretization_to_2d_ascii(non_normalized_discretization, resolution=(16, 8))
        == ascii_before_and_after
    )
    non_normalized_descriptor = non_normalized_discretization.descriptor
    assert non_normalized_descriptor._data == ba.bitarray(
        "11 00 10 01 00 00 01 00 00 00 10 00 01 00 00"
    )
    assert find_uniqueness_violations(non_normalized_descriptor) == [{2, 3, 6}]
    p = PlannedAdaptiveRefinement(non_normalized_discretization)

    p._markers = get_defaultdict_for_markers(descriptor.get_num_dimensions())
    p._markers[2] = np.array([0, 1], dtype=np.int8)
    p._markers[3] = np.array([0, -1], dtype=np.int8)
    p._markers[6] = np.array([0, -1], dtype=np.int8)

    new_discretization, _ = p.create_new_discretization(track_mapping="patches")
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(16, 8))
        == ascii_before_and_after
    )
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("11 00 11 00 00 00 00 00 10 00 01 00 00")
    assert find_uniqueness_violations(new_descriptor) == []
    normalized_discretization, mapping, num_rounds = normalize_discretization(
        non_normalized_discretization
    )
    assert num_rounds == 1
    assert normalized_discretization.descriptor == new_descriptor
    assert mapping == [
        {0},
        {1},
        {2},
        {2, 3, 5},
        {3},
        {5},
        {2, 4, 6},
        {4},
        {6},
        {7},
        {8},
        {9},
        {10},
        {11},
        {12},
    ]


def test_refine_2d_4():
    "Masado's more complex example"
    non_normalized_descriptor = RefinementDescriptor.from_binary(
        2,
        ba.bitarray(
            "10 10 01 01 11 00 00 00 00 00 00 \
            10 01 01 01 00 00 00 00 01 01 01 00 00 00 00 01 00 00"
        ),
    )
    assert find_uniqueness_violations(non_normalized_descriptor) == [{11, 12, 19}]
    ascii_before_and_after = """\
_________________________________
|       |   |   |               |
|       |   |   |               |
|       |   |   |               |
|       |   |   |               |
|       |   |   |               |
|       |   |   |               |
|       |   |   |               |
|_______|___|___|_______________|
|       |   |   |               |
|       |   |   |               |
|       |   |   |               |
|_______|___|___|               |
|   |   |   |   |               |
|___|___|___|___|               |
|   |   |   |   |               |
|___|___|___|___|_______________|"""

    non_normalized_discretization = Discretization(
        MortonOrderLinearization(), non_normalized_descriptor
    )
    new_discretization, mapping, num_rounds = normalize_discretization(
        non_normalized_discretization,
        track_mapping="patches",
        max_normalization_rounds=1,
    )
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(32, 16))
        == ascii_before_and_after
    )
    assert num_rounds == 1
    assert new_discretization.descriptor == RefinementDescriptor.from_binary(
        2,
        ba.bitarray(
            "10 10 01 01 11 00 00 00 00 00 00 11 01 01 00 00 00 01 01 00 00 00 00 00 01 00 00"
        ),
    )
    assert find_uniqueness_violations(new_discretization.descriptor) == [{1, 2, 11}]
    new_discretization, mapping, num_rounds = normalize_discretization(
        non_normalized_discretization,
        track_mapping="boxes",
    )
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(32, 16))
        == ascii_before_and_after
    )
    assert num_rounds == 5
    assert new_discretization.descriptor == RefinementDescriptor.from_binary(
        2,
        ba.bitarray(
            "11 11 11 00 00 00 00 11 00 00 00 00 00 10 00 00 00 10 00 10 00 00 00"
        ),
    )
    assert find_uniqueness_violations(new_discretization.descriptor) == []
    assert mapping == [
        {0},
        {1},
        {2},
        {3},
        {8},
        {12},
        {4},
        {6},
        {9},
        {13},
        {5},
        {7},
        {10},
        {14},
        {11},
        {15},
    ]
    _, patch_mapping, num_rounds = normalize_discretization(
        non_normalized_discretization,
        track_mapping="patches",
    )
    assert num_rounds == 5
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 17},
        2: {0, 1, 2, 12, 17, 18},
        3: {1, 2, 12},
        4: {2},
        5: {3},
        6: {4},
        7: {5},
        8: {6},
        9: {12},
        10: {18},
        11: {0, 1, 7, 13, 17, 19},
        12: {0, 1, 7, 8, 10, 13, 14, 17, 19, 20},
        13: {1, 7, 8, 10, 13, 14},
        14: {7, 8, 10},
        15: {8},
        16: {10},
        17: {14},
        18: {20},
        19: {0, 1, 7, 9, 11, 13, 15, 17, 19, 21},
        20: {1, 7, 9, 11, 13, 15},
        21: {7, 9, 11},
        22: {9},
        23: {11},
        24: {15},
        25: {21},
        26: {0, 16, 22},
        27: {16},
        28: {22},
    }
    assert patch_mapping == list(expected_patch_mapping.values())


def test_refine_2d_5():
    """Test found through randomized testing, it used to return incomplete mapping"""
    descriptor = RefinementDescriptor.from_binary(
        2,
        ba.bitarray("10 00 01 01 00 00 01 00 00"),
    )
    assert validate_descriptor(descriptor)
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    assert (
        discretization_to_2d_ascii(discretization, resolution=(8, 4))
        == """\
_________
|   |___|
|   |___|
|   |___|
|___|___|"""
    )

    refinement = ba.bitarray("01")
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, refinement)
    new_discretization, index_mapping = p.apply_refinements(track_mapping="patches")

    assert validate_descriptor(new_discretization.descriptor)
    helper_check_mapping(
        index_mapping,
        discretization,
        new_discretization,
        mapping_indices_are_boxes=False,
        tested_refinement=[refinement],
    )

    assert new_discretization.descriptor._data == ba.bitarray(
        "11 00 01 00 00 00 01 00 00"
    )
    assert (
        discretization_to_2d_ascii(new_discretization, resolution=(8, 4))
        == """\
_________
|   |___|
|___|___|
|   |___|
|___|___|"""
    )
    expected_mapping = {
        0: {0},
        1: {0, 1, 5},
        2: {0, 2, 6},
        3: {2},
        4: {3},
        5: {4},
        6: {6},
        7: {7},
        8: {8},
    }
    assert index_mapping == [expected_mapping[i] for i in range(len(expected_mapping))]


def test_refine_2d_6():
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
    p.plan_refinement(0, "10")
    p.plan_refinement(1, "10")
    p.plan_refinement(2, "10")
    discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    assert (
        discretization_to_2d_ascii(discretization, resolution=(16, 8))
        == """\
_________________
| | | | |   |   |
| | | | |   |   |
| | | | |   |   |
|_|_|_|_|___|___|
|       |       |
|       |       |
|       |       |
|_______|_______|"""
    )
    validate_descriptor(discretization.descriptor)
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2},
        2: {0, 3, 10},
        3: {3, 4, 7},
        4: {4, 5, 6},
        5: {7, 8, 9},
        6: {10},
        7: {11},
        8: {12},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]


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
        for b in range(descriptor.get_num_boxes()):
            assert len(index_mapping[b]) == 1 or (
                b == (len(discretization) - 1) and len(index_mapping[b]) == 2
            )
            if len(index_mapping[b]) == 1:
                # make sure the coordinates are correct
                (mapped_to_index,) = index_mapping[b]
                assert coordinates_from_box_index(
                    new_discretization, mapped_to_index
                ) == coordinates_from_box_index(discretization, b)


def test_refine_3d_2():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(3, ba.bitarray("001 000 100 000 000")),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "110")
    p.plan_refinement(1, "011")
    p.plan_refinement(2, "011")
    discretization, _ = p.apply_refinements()
    # TODO check mapping


def test_refine_3d_3():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(
            3,
            ba.bitarray("010 000 101 100 000 000 100 000 000 100 000 000 100 000 000"),
        ),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "101")
    p.plan_refinement(1, "100")
    p.plan_refinement(2, "100")
    discretization, _ = p.apply_refinements()
    # TODO check mapping


def test_refine_3d_4():
    discretization = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor.from_binary(
            3,
            ba.bitarray("010 000 001 000 100 000 000"),
        ),
    )
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, "101")
    p.plan_refinement(1, "100")
    p.plan_refinement(2, "111")
    p.plan_refinement(3, "111")
    discretization, _ = p.apply_refinements()
    # TODO check mapping


def test_refine_3d_5():
    # almost same beginning as 3d_4
    descriptor = RefinementDescriptor.from_binary(
        3, ba.bitarray("001 010 100 000 000 000 000")
    )
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(2, "100")
    p.plan_refinement(3, "110")
    discretization, patch_mapping = p.apply_refinements(track_mapping="patches")
    expected_patch_mapping = {
        0: {0},
        1: {0, 1, 2, 3, 4},
        2: {0, 1, 2},
        3: {1},
        4: {2},
        5: {0, 3, 4},
        6: {0, 5, 6, 7, 8},
    }
    assert patch_mapping == [
        expected_patch_mapping[i] for i in range(len(expected_patch_mapping))
    ]


def test_refine_4d():
    descriptor = RefinementDescriptor(4, 0)
    discretization = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, ba.bitarray("0010"))
    discretization, _ = p.apply_refinements()
    p = PlannedAdaptiveRefinement(discretization)
    p.plan_refinement(0, ba.bitarray("1101"))
    discretization, _ = p.apply_refinements()
    p = PlannedAdaptiveRefinement(discretization)
    last_box_index = discretization.descriptor.get_num_boxes() - 1
    p.plan_refinement(last_box_index, ba.bitarray("0001"))

    new_discretization, index_mapping = p.apply_refinements(track_mapping="boxes")

    assert new_discretization.descriptor._data == ba.bitarray(
        "0011110000000000000000000000110000000000000000000000"
    )
    helper_check_mapping(index_mapping, discretization, new_discretization)


def test_refine_random():
    for d in range(1, 5):
        discretization = Discretization(
            MortonOrderLinearization(), RefinementDescriptor(d, 0)
        )
        for round_number in range(3):
            descriptor = discretization.descriptor
            p = PlannedAdaptiveRefinement(discretization)
            round_refinements = []
            for _ in range(2 ** (round_number + d // 2)):
                random_box = np.random.randint(0, descriptor.get_num_boxes())
                random_refinement = ba.bitarray(
                    (np.random.randint(0, 2) for _ in range(d))
                )
                p.plan_refinement(random_box, random_refinement)
                round_refinements.append((random_box, random_refinement))

            new_discretization, index_mapping = p.apply_refinements(
                track_mapping="patches"
            )
            box_mapping = hierarchical_to_box_index_mapping(
                index_mapping, descriptor, new_discretization.descriptor
            )

            # the "=" may happen if only zeros are chosen
            assert len(new_discretization.descriptor) >= len(descriptor)
            assert validate_descriptor(new_discretization.descriptor)
            helper_check_mapping(
                index_mapping,
                discretization,
                new_discretization,
                mapping_indices_are_boxes=False,
                tested_refinement=round_refinements,
            )
            helper_check_mapping(
                box_mapping,
                discretization,
                new_discretization,
                mapping_indices_are_boxes=True,
                tested_refinement=round_refinements,
            )
            discretization = new_discretization


def test_refine_random_increments():
    for d in range(1, 5):
        descriptor = RefinementDescriptor(d, 0)
        assert descriptor.get_num_boxes() == 1
        possible_increments = list(single_bit_set_gen(d))
        for round_number in range(10 + 4 ^ d):
            random_box = np.random.randint(0, descriptor.get_num_boxes())
            random_refinement = possible_increments[np.random.randint(0, d)]

            discretization = Discretization(MortonOrderLinearization(), descriptor)
            new_discretization, index_mapping = apply_single_refinement(
                discretization,
                random_box,
                random_refinement,
            )
            new_descriptor = new_discretization.descriptor
            assert new_descriptor.get_num_boxes() == descriptor.get_num_boxes() + 1
            assert validate_descriptor(new_descriptor)
            try:
                helper_check_mapping(index_mapping, discretization, new_discretization)
                for b in range(descriptor.get_num_boxes()):
                    assert len(index_mapping[b]) == 1 or (
                        b == random_box and len(index_mapping[b]) == 2
                    )
            except Exception as e:
                print(
                    f"failed for round {round_number} with box {random_box} and refinement {random_refinement}"
                )
                print(f"old descriptor: {descriptor}")
                print(f"new descriptor: {new_descriptor}")
                raise e
            descriptor = new_descriptor


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
