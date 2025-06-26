import bitarray as ba
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

from dyada.coordinates import (
    get_coordinates_from_level_index,
    level_index_from_sequence,
)
from dyada.descriptor import RefinementDescriptor, validate_descriptor
from dyada.drawing import (
    latex_write_and_compile,
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
from dyada.structure import depends_on_optional, module_is_available


def test_depends_on_optional():
    @depends_on_optional("nonexistent_module")
    def should_not_run():
        pass

    with pytest.raises(ImportError):
        should_not_run()


def test_no_latex_error():
    with pytest.warns(UserWarning):
        latex_write_and_compile("test", "test_latex_no_error.tex")
    os.remove("test_latex_no_error.tex")
    # check that there are no files with "test_latex_no_error" in the current directory
    files = [f for f in os.listdir(".") if os.path.isfile(f)]
    for f in files:
        assert "test_latex_no_error" not in f


# todo consider comparing images: https://github.com/matplotlib/pytest-mpl
def test_plot_boxes_2d():
    with plt.ion():  # turns off blocking figures for test
        for backend in ["matplotlib", "tikz"]:
            level_index = level_index_from_sequence([0, 0], [0, 0])
            coordinates = get_coordinates_from_level_index(level_index)
            plot_boxes_2d([coordinates], labels=[str(level_index)], backend=backend)
            level_index = level_index_from_sequence([0, 1, 0], [0, 0, 0])
            level_index2 = level_index_from_sequence([0, 1, 0], [0, 1, 0])
            coordinates = get_coordinates_from_level_index(level_index)
            coordinates2 = get_coordinates_from_level_index(level_index2)
            plot_boxes_2d(
                [coordinates, coordinates2],
                labels=["0", "1"],
                alpha=0.2,
                backend=backend,
            )
            level_index = level_index_from_sequence(
                [5, 4, 3, 2, 1, 0], [31, 15, 7, 3, 1, 0]
            )
            coordinates = get_coordinates_from_level_index(level_index)
            plot_boxes_2d([coordinates], projection=[3, 4], backend=backend)

    with pytest.raises(AssertionError):
        plot_boxes_2d([coordinates], projection=[2, 3, 4])

    with pytest.raises(ValueError):
        plot_boxes_2d([coordinates], backend="unknown")


def test_plot_boxes_2d_from_descriptor():
    descriptor = RefinementDescriptor(4, [0, 1, 2, 3])
    r = Discretization(MortonOrderLinearization(), descriptor)
    with plt.ion():  # turns off blocking figures for test
        for backend in ["matplotlib", "tikz"]:
            # try all combinations of projections
            for projection in permutations(range(4), 2):
                # the transparency should give murky colors for the lower projections
                # (because many boxes will be stacked on top of each other)
                # and nicer colors for the higher ones
                plot_all_boxes_2d(
                    r,
                    projection=list(projection),
                    alpha=0.3,
                    filename="2d_transparent",
                    backend=backend,
                    colors="orange",
                )
                plot_all_boxes_2d(
                    r, projection=list(projection), labels="boxes", backend=backend
                )


def test_draw_simplest_grandchild_split_tikz():
    # cf. test_refine_simplest_grandchild_split
    descriptor_1 = RefinementDescriptor.from_binary(
        2, ba.bitarray("10 01 00 00 10 00 00")
    )
    descriptor_2 = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 00 10 00 00 00 10 00 00")
    )
    # transpose refinements of the two above
    descriptor_3 = RefinementDescriptor.from_binary(
        2, ba.bitarray("01 10 00 00 01 00 00")
    )
    descriptor_4 = RefinementDescriptor.from_binary(
        2, ba.bitarray("11 00 00 01 00 00 01 00 00")
    )

    plot_tree_tikz(descriptor_1, filename="grandchild_split_before")
    plot_all_boxes_2d(
        Discretization(MortonOrderLinearization(), descriptor_1),
        backend="tikz",
        filename="grandchild_split_before_2d",
        labels="patches",
        connect_centers=True,
    )
    plot_all_boxes_2d(
        Discretization(MortonOrderLinearization(), descriptor_2),
        backend="tikz",
        filename="grandchild_split_after_2d",
        labels=None,
        connect_centers=True,
    )
    plot_all_boxes_2d(
        Discretization(MortonOrderLinearization(), descriptor_3),
        backend="tikz",
        filename="grandchild_split_before_2d_transpose",
        labels="patches",
        projection=[1, 0],
        connect_centers=True,
    )  # should be the same as the first
    plot_all_boxes_2d(
        Discretization(MortonOrderLinearization(), descriptor_4),
        backend="tikz",
        filename="grandchild_split_after_2d_transpose",
        labels=None,
        projection=[1, 0],
        connect_centers=True,
    )  # should be different from the second


def test_plot_boxes_3d_from_descriptor():
    descriptor = RefinementDescriptor(3, [1, 0, 1])
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(3, ba.bitarray("101"))
    p.plan_refinement(1, ba.bitarray("001"))
    p.plan_refinement(2, ba.bitarray("010"))
    descriptor, _ = p.apply_refinements()
    validate_descriptor(descriptor)
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(3, ba.bitarray("100"))
    p.plan_refinement(7, ba.bitarray("010"))
    descriptor, _ = p.apply_refinements()
    validate_descriptor(descriptor)
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(9, ba.bitarray("101"))
    new_descriptor, _ = p.apply_refinements()
    validate_descriptor(new_descriptor)
    r = Discretization(MortonOrderLinearization(), new_descriptor)
    backends = ["tikz", "obj"]
    backends.append("matplotlib") if module_is_available("matplotlib") else None
    backends.append("opengl") if module_is_available("OpenGL") else None
    (
        backends.append("aaaaargh") if module_is_available("aaaaargh") else None
    )  # should not raise
    for backend in backends:
        if backend == "tikz":
            plot_all_boxes_3d(
                r, labels="boxes", draw_options="fill opacity=0.1", backend=backend
            )
            plot_tree_tikz(
                new_descriptor,
                labels=["ö" + str(a) for a in np.arange(len(new_descriptor))],
            )
            plot_descriptor_tikz(new_descriptor)
        else:
            with plt.ion():  # turns off blocking figures for test
                plot_all_boxes_3d(r, labels="boxes", alpha=0.1, backend=backend)
        plot_all_boxes_3d(
            r, wireframe=True, filename="test_filename_wireframe", backend=backend
        )


def test_plot_octree_3d_from_descriptor():
    descriptor = RefinementDescriptor(3, [1, 1, 1])
    r = Discretization(MortonOrderLinearization(), descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(4, ba.bitarray("111"))
    p.plan_refinement(7, ba.bitarray("111"))
    new_descriptor, _ = p.apply_refinements()
    validate_descriptor(new_descriptor)
    r = Discretization(MortonOrderLinearization(), new_descriptor)
    p = PlannedAdaptiveRefinement(r)
    p.plan_refinement(new_descriptor.get_num_boxes() - 2, ba.bitarray("111"))
    new_descriptor, _ = p.apply_refinements()
    r = Discretization(MortonOrderLinearization(), new_descriptor)
    backends = ["tikz", "obj"]
    backends.append("matplotlib") if module_is_available("matplotlib") else None
    backends.append("opengl") if module_is_available("OpenGL") else None
    for backend in backends:
        if backend == "matplotlib" or backend == "opengl":
            plot_all_boxes_3d(
                r,
                labels="boxes",
                filename="octree",
                alpha=0.1,
                backend=backend,
                colors="red",
                linewidth=3.0,
            )
        else:
            plot_all_boxes_3d(
                r,
                labels="boxes",
                filename="octree",
                draw_options="fill opacity=0.1",
                backend=backend,
            )
    plot_all_boxes_3d(r, wireframe=True, filename="octree", labels=None)
    plot_tree_tikz(new_descriptor, filename="octree_tree")
    with pytest.raises(ValueError):
        plot_all_boxes_3d(r, filename="octree_tree", backend="unknown")
