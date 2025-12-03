import bitarray as ba
import pytest
from os.path import abspath

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
    p.plan_coarsening(0, ba.bitarray("10"))
    new_discretization, _ = p.apply_refinements()
    new_descriptor = new_discretization.descriptor
    assert new_descriptor._data == ba.bitarray("00")

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


def test_coarsen_octree():
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


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
