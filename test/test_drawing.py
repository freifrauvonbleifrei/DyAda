import bitarray as ba
from itertools import permutations
import matplotlib.pyplot as plt
import pytest

from dyada.coordinates import (
    get_coordinates_from_level_index,
    level_index_from_sequence,
)
from dyada.descriptor import RefinementDescriptor, validate_descriptor
from dyada.drawing import (
    plot_boxes_2d,
    plot_all_boxes_2d,
    plot_all_boxes_3d,
    plot_tree_tikz,
    plot_descriptor_tikz,
)
from dyada.linearization import MortonOrderLinearization
from dyada.refinement import (
    Discretization,
    PlannedAdaptiveRefinement,
)
from dyada.structure import depends_on_optional


def test_depends_on_optional():
    @depends_on_optional("nonexistent_module")
    def should_not_run():
        pass

    with pytest.raises(ImportError):
        should_not_run()


# todo consider comparing images: https://github.com/matplotlib/pytest-mpl
def test_plot_boxes_2d_matplotlib():
    with plt.ion():  # turns off blocking figures for test
        level_index = level_index_from_sequence([0, 0], [0, 0])
        coordinates = get_coordinates_from_level_index(level_index)
        plot_boxes_2d([coordinates], labels=[str(level_index)])
        level_index = level_index_from_sequence([0, 1, 0], [0, 0, 0])
        level_index2 = level_index_from_sequence([0, 1, 0], [0, 1, 0])
        coordinates = get_coordinates_from_level_index(level_index)
        coordinates2 = get_coordinates_from_level_index(level_index2)
        plot_boxes_2d([coordinates, coordinates2], labels=["0", "1"], alpha=0.2)
        level_index = level_index_from_sequence(
            [5, 4, 3, 2, 1, 0], [31, 15, 7, 3, 1, 0]
        )
        coordinates = get_coordinates_from_level_index(level_index)
        plot_boxes_2d([coordinates], projection=[3, 4])

    with pytest.raises(AssertionError):
        plot_boxes_2d([coordinates], projection=[2, 3, 4])

    with pytest.raises(ValueError):
        plot_boxes_2d([coordinates], backend="unknown")


def test_plot_boxes_2d_from_descriptor():
    descriptor = RefinementDescriptor(4, [0, 1, 2, 3])
    r = Discretization(MortonOrderLinearization(), descriptor)
    with plt.ion():  # turns off blocking figures for test
        # try all combinations of projections
        for projection in permutations(range(4), 2):
            # the transparency should give murky colors for the lower projections
            # (because many boxes will be stacked on top of each other)
            # and nicer colors for the higher ones
            plot_all_boxes_2d(r, projection=list(projection), alpha=0.3)


def test_plot_boxes_3d_from_descriptor():
    descriptor = RefinementDescriptor(3, [1, 0, 1])
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(3, ba.bitarray("101"))
    p.plan_refinement(1, ba.bitarray("001"))
    p.plan_refinement(2, ba.bitarray("010"))
    new_descriptor = p.apply_refinements()
    validate_descriptor(new_descriptor)
    r = Discretization(MortonOrderLinearization(), new_descriptor)
    plot_all_boxes_3d(r, wireframe=False, draw_options="fill opacity=0.1")
    plot_all_boxes_3d(r, wireframe=True, filename="test_filename")
    plot_tree_tikz(new_descriptor)
    plot_descriptor_tikz(new_descriptor)


def test_plot_octree_3d_from_descriptor():
    descriptor = RefinementDescriptor(3, [1, 1, 1])
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(4, ba.bitarray("111"))
    p.plan_refinement(7, ba.bitarray("111"))
    new_descriptor = p.apply_refinements()
    validate_descriptor(new_descriptor)
    r = Discretization(MortonOrderLinearization(), new_descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(new_descriptor.get_num_boxes() - 2, ba.bitarray("111"))
    new_descriptor = p.apply_refinements()
    r = Discretization(MortonOrderLinearization(), new_descriptor)
    plot_all_boxes_3d(
        r, wireframe=False, filename="octree", draw_options="fill opacity=0.1"
    )
    plot_all_boxes_3d(r, wireframe=True, filename="octree")
    plot_tree_tikz(new_descriptor, filename="octree_tree")
