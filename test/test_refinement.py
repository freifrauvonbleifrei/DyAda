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
)
from dyada.discretization import (
    coordinates_from_box_index,
    coordinates_from_index,
)
from dyada.refinement import (
    Discretization,
    PlannedAdaptiveRefinement,
    apply_single_refinement,
)
from dyada.linearization import MortonOrderLinearization, single_bit_set_gen


def helper_check_box_mapping(
    index_mapping: list[set[int]],
    old_discretization: Discretization,
    new_discretization: Discretization,
    tested_refinement=None,
):
    old_descriptor = old_discretization.descriptor
    new_descriptor = new_discretization.descriptor
    assert len(index_mapping) == old_descriptor.get_num_boxes()
    assert set() not in index_mapping
    count_new_indices: Counter = Counter()
    for value in index_mapping:
        count_new_indices.update(value)
    # check that each is there once
    for e in count_new_indices.most_common():
        assert e[1] == 1
    assert sorted(count_new_indices.elements()) == list(
        range(new_descriptor.get_num_boxes())
    )
    for mapped_from_index, mapped_to_indices in enumerate(index_mapping):
        old_coordinates = coordinates_from_box_index(
            old_discretization, mapped_from_index
        )
        if len(mapped_to_indices) == 1:
            (mapped_to_index,) = mapped_to_indices
            new_coordinates = coordinates_from_box_index(
                new_discretization, mapped_to_index
            )
            # make sure the coordinates are correct
            assert old_coordinates == new_coordinates
        else:
            # assert that the new boxes cover the old box
            # new_coordinates_list = [
            #     coordinates_from_index(new_discretization, new_index, is_box_index=True)
            #     for new_index in mapped_to_indices
            # ]
            # # needs recursive line-sweep #TODO
            pass


def helper_check_mapping(
    index_mapping: list[set[int]],
    old_discretization: Discretization,
    new_discretization: Discretization,
    mapping_indices_are_boxes=True,
    tested_refinement=None,
) -> None:
    if mapping_indices_are_boxes:
        return helper_check_box_mapping(
            index_mapping, old_discretization, new_discretization, tested_refinement
        )
    else:
        # get box mapping from index mapping
        box_mapping = hierarchical_to_box_index_mapping(
            index_mapping,
            old_discretization.descriptor,
            new_discretization.descriptor,
        )
        helper_check_box_mapping(box_mapping, old_discretization, new_discretization)

    old_descriptor = old_discretization.descriptor
    new_descriptor = new_discretization.descriptor
    count_new_indices: Counter = Counter()
    for value in index_mapping:
        count_new_indices.update(value)
    assert len(index_mapping) == len(old_descriptor)
    assert set() not in index_mapping
    assert set(count_new_indices.elements()) == set(range(len(new_descriptor)))
    for b, mapped_to_indices in enumerate(index_mapping):
        old_coordinates = coordinates_from_index(
            old_discretization, b, mapping_indices_are_boxes
        )
        # the smallest of the mapped_to_indices' interval should cover the old interval
        most_senior_mapped_to = min(mapped_to_indices)
        new_coordinates = coordinates_from_index(
            new_discretization,
            most_senior_mapped_to,
            mapping_indices_are_boxes,
        )
        assert np.all(new_coordinates.lower_bound <= old_coordinates.lower_bound)
        assert np.all(old_coordinates.upper_bound <= new_coordinates.upper_bound)
        if len(mapped_to_indices) > 1:
            # and this should not be true for the second
            second_most_senior_mapped_to = sorted(mapped_to_indices)[1]
            new_coordinates = coordinates_from_index(
                new_discretization,
                second_most_senior_mapped_to,
                mapping_indices_are_boxes,
            )
            assert not (
                np.all(new_coordinates.lower_bound <= old_coordinates.lower_bound)
                and np.all(old_coordinates.upper_bound <= new_coordinates.upper_bound)
            )


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
