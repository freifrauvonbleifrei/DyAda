from itertools import permutations
import matplotlib.pyplot as plt
import pytest

from dyada.coordinates import (
    get_coordinates_from_level_index,
    level_index_from_sequence,
)
from dyada.drawing import plot_boxes_2d, plot_all_boxes_2d
from dyada.refinement import Refinement, RefinementDescriptor
from dyada.linearization import MortonOrderLinearization


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
    r = Refinement(MortonOrderLinearization(), descriptor)
    with plt.ion():  # turns off blocking figures for test
        # try all combinations of projections
        for projection in permutations(range(4), 2):
            # the transparency should give murky colors for the lower projections
            # (because many boxes will be stacked on top of each other)
            # and nicer colors for the higher ones
            plot_all_boxes_2d(r, projection=list(projection), alpha=0.3)
