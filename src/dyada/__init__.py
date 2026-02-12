# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from dyada.coordinates import (
    CoordinateInterval,
    get_coordinates_from_level_index,
    interval_from_sequences,
    DyadaTooFineError,
)
from dyada.descriptor import RefinementDescriptor
from dyada.discretization import (
    Discretization,
    coordinates_from_box_index,
    SliceDictInDimension,
    get_level_index_from_linear_index,
)
from dyada.drawing import (  # TODO move to drawing once ruff allows finer control
    discretization_to_2d_ascii,
    plot_all_boxes_2d,
    plot_all_boxes_3d,
    get_figure_2d_matplotlib,
    get_figure_3d_matplotlib,
)
from dyada.drawing_opengl_obj import (
    plot_boxes_3d_pyopengl,
    export_boxes_3d_to_obj,
)
from dyada.drawing_tikz import (
    plot_tree_tikz,
    plot_descriptor_tikz,
    plot_boxes_2d_tikz,
    plot_boxes_3d_tikz,
    plot_location_stack_tikz,
)
from dyada.linearization import MortonOrderLinearization
from dyada.refinement import PlannedAdaptiveRefinement, apply_single_refinement

assert sys.version_info >= (3, 10), "Use Python 3.10 or newer"
