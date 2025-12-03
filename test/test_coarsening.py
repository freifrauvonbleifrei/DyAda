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


if __name__ == "__main__":
    here = abspath(__file__)
    pytest.main([here, "-s"])
