from collections import Counter
import pytest
import bitarray as ba
import numpy as np
from os.path import abspath

from dyada.descriptor import (
    RefinementDescriptor,
    validate_descriptor,
)

from dyada.refinement import (
    Discretization,
    PlannedAdaptiveRefinement,
    apply_single_refinement,
    RefinementError,
    coordinates_from_box_index,
)
from dyada.linearization import MortonOrderLinearization, single_bit_set_gen


def test_get_level_index():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))

    # with indices that consider the parents
    level, index = r.get_level_index(0, False)
    assert np.array_equal(level, [0, 0]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(1, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(2, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(3, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(4, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(5, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(6, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(7, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(8, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 2])
    level, index = r.get_level_index(9, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 3])
    level, index = r.get_level_index(10, False)
    assert np.array_equal(level, [1, 1]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(11, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 2])
    level, index = r.get_level_index(12, False)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 3])
    with pytest.raises(IndexError):
        r.get_level_index(13, False)

    # with indices that consider only leaves / boxes
    level, index = r.get_level_index(0, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 0])
    level, index = r.get_level_index(1, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 1])
    level, index = r.get_level_index(2, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 0])
    level, index = r.get_level_index(3, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 1])
    level, index = r.get_level_index(4, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 2])
    level, index = r.get_level_index(5, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [0, 3])
    level, index = r.get_level_index(6, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 2])
    level, index = r.get_level_index(7, True)
    assert np.array_equal(level, [1, 2]) and np.array_equal(index, [1, 3])
    with pytest.raises(IndexError):
        r.get_level_index(8, True)


def test_get_all_boxes_level_indices():
    # same example as test_get_level_index
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    expected_indices = ([0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3])
    for i, level_index in enumerate(r.get_all_boxes_level_indices()):
        assert np.array_equal(level_index.d_level, np.asarray([1, 2]))
        assert np.array_equal(level_index.d_index, np.asarray(expected_indices[i]))


def test_get_box_from_coordinate():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    # check the midpoints of each box
    assert r.get_containing_box(np.array([0.25, 0.125])) == 0
    assert r.get_containing_box(np.array([0.25, 0.375])) == 1
    assert r.get_containing_box(np.array([0.75, 0.125])) == 2
    assert r.get_containing_box(np.array([0.75, 0.375])) == 3
    assert r.get_containing_box(np.array([0.25, 0.625])) == 4
    assert r.get_containing_box(np.array([0.25, 0.875])) == 5
    assert r.get_containing_box(np.array([0.75, 0.625])) == 6
    assert r.get_containing_box(np.array([0.75, 0.875])) == 7

    # now along the edges
    assert r.get_containing_box(np.array([0.0, 0.0])) == 0
    assert r.get_containing_box(np.array([0.0, 0.25])) == (0, 1)
    assert r.get_containing_box(np.array([0.0, 0.5])) == (1, 4)
    assert r.get_containing_box(np.array([0.0, 0.75])) == (4, 5)
    assert r.get_containing_box(np.array([0.0, 1.0])) == 5
    assert r.get_containing_box(np.array([0.25, 0.0])) == 0
    assert r.get_containing_box(np.array([0.25, 0.25])) == (0, 1)
    assert r.get_containing_box(np.array([0.25, 0.5])) == (1, 4)
    assert r.get_containing_box(np.array([0.25, 0.75])) == (4, 5)
    assert r.get_containing_box(np.array([0.25, 1.0])) == 5
    assert r.get_containing_box(np.array([0.5, 0.0])) == (0, 2)
    assert r.get_containing_box(np.array([0.5, 0.25])) == (0, 1, 2, 3)
    assert r.get_containing_box(np.array([0.5, 0.5])) == (1, 3, 4, 6)
    assert r.get_containing_box(np.array([0.5, 0.75])) == (4, 5, 6, 7)
    assert r.get_containing_box(np.array([0.5, 1.0])) == (5, 7)
    assert r.get_containing_box(np.array([0.75, 0.0])) == 2
    assert r.get_containing_box(np.array([0.75, 0.25])) == (2, 3)
    assert r.get_containing_box(np.array([0.75, 0.5])) == (3, 6)
    assert r.get_containing_box(np.array([0.75, 0.75])) == (6, 7)
    assert r.get_containing_box(np.array([0.75, 1.0])) == 7
    assert r.get_containing_box(np.array([1.0, 0.0])) == 2
    assert r.get_containing_box(np.array([1.0, 0.25])) == (2, 3)
    assert r.get_containing_box(np.array([1.0, 0.5])) == (3, 6)
    assert r.get_containing_box(np.array([1.0, 0.75])) == (6, 7)
    assert r.get_containing_box(np.array([1.0, 1.0])) == 7

    # outside the domain
    with pytest.raises(ValueError):
        r.get_containing_box(np.array([0.0, 1.5]))
    with pytest.raises(ValueError):
        r.get_containing_box(np.array([1.5, 0.0]))
    with pytest.raises(ValueError):
        r.get_containing_box(np.array([1.5, 1.5]))


def test_refine_2d_only_leaves():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 2]))
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("01"))
    p.plan_refinement(0, ba.bitarray("01"))
    p.plan_refinement(1, ba.bitarray("10"))
    p.plan_refinement(2, ba.bitarray("11"))
    p.plan_refinement(4, ba.bitarray("11"))
    p.plan_refinement(6, ba.bitarray("01"))

    p.apply_refinements()
    assert validate_descriptor(r.descriptor)


def test_refine_3d_only_leaves():
    descriptor = RefinementDescriptor(3, [1, 0, 1])
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("101"))
    p.plan_refinement(1, ba.bitarray("001"))
    p.plan_refinement(2, ba.bitarray("010"))
    new_descriptor = p.apply_refinements()
    assert new_descriptor.get_num_boxes() == 9
    assert validate_descriptor(new_descriptor)


def helper_check_mapping(index_mapping, old_discretization, new_discretization):
    old_descriptor = old_discretization.descriptor
    new_descriptor = new_discretization.descriptor
    assert index_mapping.keys() == set(range(old_descriptor.get_num_boxes()))
    count_new_indices = Counter()
    for value in index_mapping.values():
        count_new_indices.update(value)
    assert sorted(count_new_indices.elements()) == list(
        range(new_descriptor.get_num_boxes())
    )
    for b in range(old_descriptor.get_num_boxes()):
        if len(index_mapping[b]) == 1:
            # make sure the coordinates are correct
            assert coordinates_from_box_index(
                new_discretization, index_mapping[b][0]
            ) == coordinates_from_box_index(old_discretization, b)
        else:
            old_interval = coordinates_from_box_index(old_discretization, b)
            for new_index in index_mapping[b]:
                new_interval = coordinates_from_box_index(new_discretization, new_index)
                assert old_interval.contains(new_interval.lower_bound)
                assert old_interval.contains(new_interval.upper_bound)


def test_refine_simplest_not_only_leaves():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 0]))
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("01"))

    assert p._planned_refinements.queue == [
        (1, ba.bitarray("01")),
    ]

    # don't do this at home -- call p.apply_refinements() directly
    p.populate_queue()
    assert p._planned_refinements.empty()
    assert len(p._markers) == 1 and all(p._markers[1] == [0, 1])
    assert p._upward_queue.queue == [(-1, 1)]

    p.upwards_sweep()
    assert len(p._markers) == 1 and all(p._markers[1] == [0, 1])
    assert p._upward_queue.empty()

    new_descriptor = p.create_new_descriptor(track_mapping=False)
    assert new_descriptor._data == ba.bitarray("1001000000")
    assert validate_descriptor(new_descriptor)

    r = Discretization(MortonOrderLinearization(), new_descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(2, ba.bitarray("01"))

    new_descriptor_2 = p.apply_refinements()
    assert new_descriptor_2._data == RefinementDescriptor(2, [1, 1])._data
    assert validate_descriptor(new_descriptor_2)


def test_refine_simplest_grandchild_split():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 0]))
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("01"))
    p.plan_refinement(1, ba.bitarray("10"))

    assert p._planned_refinements.queue == [
        (1, ba.bitarray("01")),
        (2, ba.bitarray("10")),
    ]

    # don't do this at home -- call p.apply_refinements() directly
    p.populate_queue()
    assert p._planned_refinements.empty()
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

    new_descriptor, index_mapping = p.create_new_descriptor(track_mapping=True)
    assert new_descriptor._data == ba.bitarray("10010000100000")
    assert validate_descriptor(new_descriptor)
    r_2 = Discretization(MortonOrderLinearization(), new_descriptor)
    helper_check_mapping(index_mapping, r, r_2)

    # plot_tree_tikz(new_descriptor, filename="simplest_grandchild_split_before")
    # plot_all_boxes_2d(Discretization(MortonOrderLinearization(), new_descriptor), labels="patches")
    p = PlannedAdaptiveRefinement(r_2)
    p.plan_refinement(2, ba.bitarray("01"))
    p.plan_refinement(3, ba.bitarray("01"))
    assert p._planned_refinements.queue == [
        (5, ba.bitarray("01")),
        (6, ba.bitarray("01")),
    ]
    p.populate_queue()
    p.upwards_sweep()
    assert (
        len(p._markers) == 2
        and all(p._markers[0] == [0, 1])
        and all(p._markers[1] == [0, -1])
    )
    assert p._upward_queue.empty()
    new_descriptor_2, index_mapping_2 = p.create_new_descriptor(track_mapping=True)
    assert new_descriptor_2._data == ba.bitarray("110010000000100000")
    assert validate_descriptor(new_descriptor_2)
    helper_check_mapping(
        index_mapping_2,
        r_2,
        Discretization(MortonOrderLinearization(), new_descriptor_2),
    )


def test_refine_grandchild_split():
    p = PlannedAdaptiveRefinement(
        Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [1, 1]))
    )
    # prepare the "initial" state
    p.plan_refinement(0, ba.bitarray("01"))
    p.plan_refinement(2, ba.bitarray("10"))
    new_descriptor = p.apply_refinements()
    p = PlannedAdaptiveRefinement(
        Discretization(MortonOrderLinearization(), new_descriptor)
    )
    p.plan_refinement(1, ba.bitarray("11"))
    new_descriptor = p.apply_refinements()
    four_branch = new_descriptor.get_branch(4, False)[0]
    assert new_descriptor.get_ancestry(four_branch) == [0, 1, 3]

    p = PlannedAdaptiveRefinement(
        Discretization(MortonOrderLinearization(), new_descriptor)
    )

    # the actual test
    p.plan_refinement(0, ba.bitarray("10"))
    new_descriptor, box_mapping = p.apply_refinements(track_mapping=True)
    assert validate_descriptor(new_descriptor)
    assert new_descriptor._data == ba.bitarray("111100000100000100000010000000")

    # test the mapping of the boxes
    former_to_now = {
        0: [0, 1],
        1: [2],
        2: [4],
        3: [3],
        4: [5],
        5: [6],
        6: [7],
        7: [8],
        8: [9],
    }
    for former, now in former_to_now.items():
        assert box_mapping[former] == now


def test_refine_multi_grandchild_split():
    r = Discretization(MortonOrderLinearization(), RefinementDescriptor(2, [2, 0]))
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(2, ba.bitarray("10"))
    p.plan_refinement(3, ba.bitarray("01"))
    descriptor = p.apply_refinements(track_mapping=False)

    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(2, ba.bitarray("01"))
    p.plan_refinement(3, ba.bitarray("11"))
    p.plan_refinement(3, ba.bitarray("10"))
    p.plan_refinement(3, ba.bitarray("10"))

    new_descriptor, index_mapping = p.apply_refinements(track_mapping=True)
    helper_check_mapping(
        index_mapping, r, Discretization(MortonOrderLinearization(), new_descriptor)
    )


def test_refine_fully():
    for d in range(1, 6):
        for l in range(1, 2):
            descriptor = RefinementDescriptor(d, l)
            r = Discretization(MortonOrderLinearization(), descriptor)
            p = PlannedAdaptiveRefinement(r)
            for i in range(1, descriptor.get_num_boxes() - 1):
                # do octree refinement
                p.plan_refinement(i, ba.bitarray("1" * d))
            new_descriptor = p.apply_refinements()
            assert validate_descriptor(new_descriptor)
            # now also refine the first and last box
            r = Discretization(MortonOrderLinearization(), new_descriptor)
            p = PlannedAdaptiveRefinement(r)
            p.plan_refinement(0, ba.bitarray("1" * d))
            p.plan_refinement(new_descriptor.get_num_boxes() - 1, ba.bitarray("1" * d))

            new_descriptor = p.apply_refinements()
            assert validate_descriptor(new_descriptor)

            regular_descriptor = RefinementDescriptor(d, l + 1)
            assert new_descriptor._data == regular_descriptor._data


def test_refine_2d():
    prependable_string = "10010000"
    descriptor = RefinementDescriptor.from_binary(
        2,
        ba.bitarray(prependable_string + "10110000000000"),
    )
    validate_descriptor(descriptor)
    correct_descriptor = RefinementDescriptor.from_binary(
        2, ba.bitarray("11001010000000001010000000")
    )
    validate_descriptor(correct_descriptor)

    num_boxes_before = descriptor.get_num_boxes()
    discretization = Discretization(MortonOrderLinearization(), descriptor)

    new_discretization, index_mapping = apply_single_refinement(
        discretization, len(discretization) - 1, ba.bitarray("01")
    )

    new_descriptor = new_discretization.descriptor
    validate_descriptor(new_descriptor)
    assert new_descriptor.get_num_boxes() == num_boxes_before + 1
    assert new_descriptor._data == correct_descriptor._data
    helper_check_mapping(index_mapping, discretization, new_discretization)
    for b in range(descriptor.get_num_boxes()):
        assert len(index_mapping[b]) == 1 or (
            b == (len(discretization) - 1) and len(index_mapping[b]) == 2
        )
        if len(index_mapping[b]) == 1:
            # make sure the coordinates are correct
            assert coordinates_from_box_index(
                new_discretization, index_mapping[b][0]
            ) == coordinates_from_box_index(discretization, b)


def test_refine_3d():
    prependable_string = "110001000000001000000001000000"
    for round in range(4):
        descriptor = RefinementDescriptor.from_binary(
            3,
            ba.bitarray(
                prependable_string * round
                + "110001000000001010000000000001010000000000000"
            ),
        )
        validate_descriptor(descriptor)
        num_boxes_before = descriptor.get_num_boxes()
        discretization = Discretization(MortonOrderLinearization(), descriptor)

        try:
            new_discretization, index_mapping = apply_single_refinement(
                discretization, len(discretization) - 1, ba.bitarray("001")
            )
        except RefinementError as e:
            # example of how RefinementErrors can be useful:
            print(e.markers)
            # plot_tree_tikz(e.descriptor, filename="error")
            labels = [""] * len(descriptor)
            for key, value in e.markers.items():
                labels[key] = ",".join(str(v) for v in value)
            # plot_tree_tikz(descriptor, labels=labels, filename="error_labels")
            raise e.error

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
                assert coordinates_from_box_index(
                    new_discretization, index_mapping[b][0]
                ) == coordinates_from_box_index(discretization, b)


def test_refine_4d():
    descriptor = RefinementDescriptor(4, 0)
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("0010"))
    descriptor = p.apply_refinements()
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(0, ba.bitarray("1101"))
    descriptor = p.apply_refinements()
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    last_box_index = descriptor.get_num_boxes() - 1
    p.plan_refinement(last_box_index, ba.bitarray("0001"))

    new_descriptor, index_mapping = p.apply_refinements(track_mapping=True)

    assert new_descriptor._data == ba.bitarray(
        "0011110000000000000000000000110000000000000000000000"
    )
    helper_check_mapping(
        index_mapping, r, Discretization(MortonOrderLinearization(), new_descriptor)
    )


def test_refine_random():
    for d in range(1, 5):
        descriptor = RefinementDescriptor(d, 0)
        for round in range(3):
            r = Discretization(MortonOrderLinearization(), descriptor)
            p = PlannedAdaptiveRefinement(r)
            for _ in range(2 ** (round + d // 2)):
                random_box = np.random.randint(0, descriptor.get_num_boxes())
                random_refinement = ba.bitarray(
                    (np.random.randint(0, 2) for _ in range(d))
                )
                p.plan_refinement(random_box, random_refinement)

            new_descriptor, index_mapping = p.apply_refinements(track_mapping=True)

            # the "=" may happen if only zeros are chosen
            assert len(new_descriptor) >= len(descriptor)
            assert validate_descriptor(new_descriptor)
            helper_check_mapping(
                index_mapping,
                r,
                Discretization(MortonOrderLinearization(), new_descriptor),
            )
            descriptor = new_descriptor


def test_refine_random_increments():
    for d in range(1, 5):
        descriptor = RefinementDescriptor(d, 0)
        assert descriptor.get_num_boxes() == 1
        possible_increments = list(single_bit_set_gen(d))
        for round in range(10 + 4 ^ d):
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
            helper_check_mapping(index_mapping, discretization, new_discretization)
            for b in range(descriptor.get_num_boxes()):
                assert len(index_mapping[b]) == 1 or (
                    b == random_box and len(index_mapping[b]) == 2
                )
            descriptor = new_descriptor


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
