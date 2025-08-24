import warnings

try:
    from cmap import Colormap
except ImportError:
    warnings.warn("cmap not found, some plotting functions will not work")
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
    from matplotlib.colors import to_rgba, to_rgb
except ImportError:
    warnings.warn("matplotlib not found, some plotting functions will not work")
try:
    import OpenGL.GL as gl  # type: ignore
    import OpenGL.GLU as glu  # type: ignore
    import OpenGL.GLUT as glut  # type: ignore
    from PIL import Image, ImageOps
except ImportError:
    warnings.warn("pyopengl not found, some plotting functions will not work")

from io import StringIO
from itertools import pairwise, product
from pathlib import Path
from string import ascii_uppercase
from typing import Sequence, Union, Mapping, Optional
import subprocess

from dyada.coordinates import (
    Coordinate,
    CoordinateInterval,
    get_coordinates_from_level_index,
)
from dyada.descriptor import branch_generator, RefinementDescriptor
from dyada.discretization import (
    Discretization,
    discretization_to_location_stack_strings,
)
from dyada.structure import depends_on_optional


def labels_from_discretization(
    discretization: Discretization, labels: Union[None, str, Sequence[str]]
):
    if labels == "patches":
        labels = []
        for i in range(len(discretization._descriptor)):
            if discretization._descriptor.is_box(i):
                labels.append(str(i))
    elif labels == "boxes":
        labels = [str(i) for i in range(len(discretization))]

    assert labels is None or len(labels) == discretization._descriptor.get_num_boxes()
    return labels


def plot_boxes_2d(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]] = None,
    projection: Sequence[int] = [0, 1],
    backend="matplotlib",
    **kwargs,
) -> None:
    assert len(projection) == 2
    if backend == "matplotlib":
        return plot_boxes_2d_matplotlib(intervals, labels, projection, **kwargs)
    elif backend == "tikz":
        return plot_boxes_2d_tikz(intervals, labels, projection, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_all_boxes_2d(
    discretization: Discretization,
    projection: Sequence[int] = [0, 1],
    labels: Union[None, str, Sequence[str]] = "patches",
    **kwargs,
) -> None:
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    labels = labels_from_discretization(discretization, labels)
    plot_boxes_2d(coordinates, projection=projection, labels=labels, **kwargs)


def plot_boxes_3d(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Union[None, str, Sequence[str]] = None,
    projection: Sequence[int] = [0, 1, 2],
    backend: str = "tikz",
    **kwargs,
) -> None:
    assert len(projection) == 3
    if backend == "matplotlib":
        return plot_boxes_3d_matplotlib(intervals, labels, projection, **kwargs)
    elif backend == "obj":
        return export_boxes_3d_to_obj(intervals, projection, **kwargs)
    elif backend == "tikz":
        return plot_boxes_3d_tikz(intervals, labels, projection, **kwargs)
    elif backend == "opengl":
        return plot_boxes_3d_pyopengl(intervals, labels, projection, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_all_boxes_3d(
    discretization: Discretization,
    projection: Sequence[int] = [0, 1, 2],
    labels: Union[None, str, Sequence[str]] = "patches",
    **kwargs,
) -> None:
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    labels = labels_from_discretization(discretization, labels)
    plot_boxes_3d(coordinates, projection=projection, labels=labels, **kwargs)


def cuboid_from_interval(
    interval: CoordinateInterval, projection: Sequence[int] = [0, 1, 2]
) -> tuple:
    lower = interval[0][projection]
    upper = interval[1][projection]
    # create vertices from lower and upper bounds
    vertices = list(product(*zip(lower, upper)))
    # define rectangular faces
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
    )
    edges = (
        *((0, 1), (2, 3), (4, 5), (6, 7)),
        *((0, 2), (1, 3), (4, 6), (5, 7)),
        *((0, 4), (1, 5), (2, 6), (3, 7)),
    )
    return vertices, faces, edges


def side_corners_generator(
    lower_cube_corner: Coordinate, upper_cube_corner: Coordinate
):
    # iterate the six sides of the cuboid
    # by always selecting four corners that have one coordinate in common
    corners = list(product(*zip(lower_cube_corner, upper_cube_corner)))
    for bound in [lower_cube_corner, upper_cube_corner]:
        for i, b in enumerate(bound):
            side_corners = list(filter(lambda c: c[i] == b, corners))
            assert len(side_corners) == 4
            # correct the order
            yield side_corners[0], side_corners[1], side_corners[3], side_corners[2]


@depends_on_optional("matplotlib.pyplot")
def get_figure_2d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int] = [0, 1],
    **kwargs,
) -> tuple:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = kwargs.pop("colors", prop_cycle.by_key()["color"])
    if isinstance(colors, str):
        colors = [colors] * len(intervals)

    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    for i, interval in enumerate(intervals):
        anchor_point = interval.lower_bound[projection]
        extent = interval.upper_bound[projection] - anchor_point
        rectangle = plt.Rectangle(
            tuple(anchor_point),  # type: ignore
            extent[0],
            extent[1],
            fill=True,
            figure=fig,
            color=colors[i % len(colors)],
            **kwargs,
        )
        ax1.add_artist(rectangle)
        ax1.set_aspect("equal")

        if labels is not None:
            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0
            ax1.annotate(
                labels[i],
                (cx, cy),
                fontsize=6,
                ha="center",
                va="center",
            )
    # add title with projection
    ax1.set_title(f"Dimensions {projection[0]} and {projection[1]}")
    return fig, ax1


@depends_on_optional("matplotlib.pyplot")
def plot_boxes_2d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    **kwargs,
) -> None:
    filename = kwargs.pop("filename", None)
    fig, ax1 = get_figure_2d_matplotlib(intervals, labels, projection, **kwargs)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


@depends_on_optional("matplotlib.pyplot")
def draw_cuboid_on_axis(
    ax: plt.Axes,
    interval: CoordinateInterval,
    projection: Sequence[int] = [0, 1, 2],
    color="skyblue",
    wireframe: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Draw a cuboid on the given axis.
    :param ax: The axis to draw on.
    :param interval: The interval to draw.
    :param projection: The projection to use.
    :param color: The color of the cuboid.
    :param wireframe: Whether to draw the cuboid as a wireframe.
    :param kwargs: Additional arguments to pass to the Poly3DCollection.
    :return: The axis with the cuboid drawn on it.
    """
    lower = interval[0][projection]
    upper = interval[1][projection]
    faces = []
    for side_corners in side_corners_generator(lower, upper):
        face = [*side_corners]
        faces.append(face)
    alpha = kwargs.pop("alpha", 0.5)
    if wireframe:
        color_rgba = to_rgba(color, alpha=alpha)
        cuboid = Poly3DCollection(
            faces,
            facecolors=(0, 0, 0, 0),  # fully transparent faces
            edgecolors=color_rgba,
            **kwargs,
        )
    else:
        edgecolors = kwargs.pop("edgecolors", "gray")
        cuboid = Poly3DCollection(
            faces,
            facecolors=color,
            alpha=alpha,
            **kwargs,
        )
        cuboid.set_edgecolor(edgecolors)
    ax.add_collection(cuboid)
    return ax


@depends_on_optional("matplotlib.pyplot")
def get_figure_3d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int] = [0, 1, 2],
    wireframe: bool = False,
    **kwargs,
) -> tuple:
    if labels is not None:
        warnings.warn("Labels are currently not used in 3D plots w/ matplotlib")

    # plt.ion()
    # plt.show() # using this and the pause below gives a neat animation
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = kwargs.pop("colors", prop_cycle.by_key()["color"])
    if isinstance(colors, str):
        colors = [colors] * len(intervals)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    for i, interval in enumerate(intervals):
        # draw each as cuboid
        draw_cuboid_on_axis(
            ax1,
            interval,
            projection=projection,
            color=colors[i % len(colors)],
            wireframe=wireframe,
            **kwargs,
        )
        # plt.pause(0.01)

    # add title with projection
    ax1.set_title(f"Dimensions {projection[0]}, {projection[1]}, {projection[2]}")
    return fig, ax1


@depends_on_optional("matplotlib.pyplot")
def plot_boxes_3d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    **kwargs,
) -> None:
    filename = kwargs.pop("filename", None)
    fig, ax1 = get_figure_3d_matplotlib(intervals, labels, projection, **kwargs)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def latex_write_and_compile(latex_string: str, filename: str) -> None:
    dirname = Path.cwd()
    with open(dirname / filename, "w") as f:
        f.write(latex_string)

    # if in environment, run pdflatex to generate pdf
    # this needs `latexmk`, `pdflatex`, and `tikz` to be installed, e.g.
    # through `apt install texlive-latex-extra latexmk`
    try:
        subprocess.check_output(
            ["latexmk", "-interaction=nonstopmode", "-pdf", filename],
            cwd=dirname,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"Error while running latexmk on {filename}: {e}")
    try:  # try to clean up regardless of success
        subprocess.run(
            ["latexmk", "-c"],
            check=False,
            cwd=dirname,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


@depends_on_optional("cmap")
def get_colors(num_colors: int, colormap_name="CET_R3"):
    cm = Colormap(colormap_name)
    for leaf in range(num_colors):
        colormapped = cm(leaf * 1.0 / num_colors)
        yield colormapped  # RGB values in [0, 1] range


def get_colors_byte(num_colors: int, colormap_name="CET_R3"):
    for color in get_colors(num_colors, colormap_name):
        color = [int(255 * c) for c in color]  # convert to [0, 255] range
        yield color


def latex_add_color_defs(
    tikz_string: str, num_colors: int, colormap_name="CET_R3"
) -> str:
    for leaf, color in enumerate(get_colors_byte(num_colors, colormap_name)):
        color_str = "color_%d" % leaf
        tikz_string += "\\definecolor{%s}{RGB}{%d,%d,%d}\n" % (
            color_str,
            color[0],
            color[1],
            color[2],
        )
    return tikz_string


def letter_counter(length):
    """Generate up to `length` Excel-style letter labels (A, B, ..., Z, AA, AB, ...)."""
    count = 0
    size = 1
    while count < length:
        for combo in product(ascii_uppercase, repeat=size):
            yield "".join(combo)
            count += 1
            if count >= length:
                break
        size += 1


def plot_boxes_2d_tikz(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    filename: Optional[str] = None,
    connect_centers=False,
    **kwargs,
) -> None:
    tikz_string = R"""\documentclass{standalone}
% autogenerated with dyada : https://github.com/freifrauvonbleifrei/DyAda
\usepackage{xcolor}
\usepackage{relsize}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
"""
    # add one white rectangle to the background
    tikz_string += "\\draw[very thin, white] (0,0) rectangle (1,1);\n"
    letter_counter_first = letter_counter(len(intervals))

    def tikz_rectangle(interval: CoordinateInterval, option_string="", label_string=""):
        lower = interval[0][projection]
        upper = interval[1][projection]
        rectangle_string = "\\draw[%s] (%f,%f) rectangle (%f,%f);\n" % (
            option_string,
            *lower,
            *upper,
        )
        middle = (lower + upper) / 2.0
        min_extent = min(upper - lower)  # type: ignore
        if label_string != "":
            if min_extent <= 0.125:
                label_string = "\\tiny \\relsize{-1} " + label_string
            elif min_extent <= 0.25:
                label_string = "\\tiny \\relsize{-0.5}" + label_string
            elif min_extent <= 0.5:
                label_string = "\\footnotesize " + label_string
        label_tex = f"\\node[inner sep=0] ({next(letter_counter_first)}) at ({middle[0]},{middle[1]}) {{{label_string}}};\n"

        return rectangle_string + label_tex

    if labels is None:

        def none_iter():
            while True:
                yield ""

        label_iter = none_iter()
    else:
        label_iter = iter(labels)
    tikz_string = latex_add_color_defs(tikz_string, len(intervals))
    for grid_idx, interval in enumerate(intervals):
        color_str = "color_%d" % grid_idx
        option_string = "very thin, gray, fill=%s,fill opacity=0.3" % (color_str)
        tikz_string += tikz_rectangle(interval, option_string, next(label_iter))

    if connect_centers:
        # connect the named nodes with lines
        letter_counter_again = letter_counter(len(intervals))
        connect_string = "\\draw[thick, densely dotted] "
        for node_name in letter_counter_again:
            connect_string += f"({node_name}) -- "
        tikz_string += connect_string[:-4] + ";\n"

    tikz_string += "\\end{tikzpicture}\n"
    tikz_string += "\\end{document}\n"
    if filename is None:
        filename = "tikz_rectangles"
    filename += ".tex"
    latex_write_and_compile(tikz_string, filename)


# inspired by @griegler : https://github.com/griegler/octnet/issues/28
def plot_boxes_3d_tikz(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    wireframe: bool = False,
    filename: Optional[str] = None,
    **kwargs,
) -> None:

    def tikz_cuboid(
        interval: CoordinateInterval, option_string="", label_string=""
    ) -> str:
        tikz_string = ""
        line_string = "\\draw[%s] (%f,%f,%f) -- (%f,%f,%f) -- (%f,%f,%f) -- (%f,%f,%f) -- cycle;\n"
        lower = interval[0][projection]
        upper = interval[1][projection]
        for side_corners in side_corners_generator(lower, upper):
            tikz_string += line_string % (
                option_string,
                *side_corners[0],
                *side_corners[1],
                *side_corners[2],
                *side_corners[3],
            )
        middle = (lower + upper) / 2.0
        min_extent = min(upper - lower)  # type: ignore
        if min_extent < 0.125:
            label_string = "\\tiny \\relsize{-1} " + label_string
        elif min_extent < 0.25:
            label_string = "\\tiny \\relsize{-0.5}" + label_string
        elif min_extent < 0.5:
            label_string = "\\footnotesize " + label_string
        tikz_string += (
            f"\\node at ({middle[0]},{middle[1]},{middle[2]}) {{{label_string}}};\n"
        )
        return tikz_string

    def tikz_grid(intervals, wireframe: bool):
        tikz_string = R"""\documentclass{standalone}
% autogenerated with dyada : https://github.com/freifrauvonbleifrei/DyAda
% inspired by @griegler : https://github.com/griegler/octnet/issues/28
\usepackage{xcolor}
\usepackage{relsize}
\usepackage{tikz,tikz-3dplot}
\begin{document}
\tdplotsetmaincoords{50}{130}
\begin{tikzpicture}[scale=1.0, tdplot_main_coords]"""
        if labels is None:

            def none_iter():
                while True:
                    yield ""

            label_iter = none_iter()
        else:
            label_iter = iter(labels)
        tikz_string = latex_add_color_defs(tikz_string, len(intervals))
        for grid_idx, interval in enumerate(intervals):
            color_str = "color_%d" % grid_idx
            if wireframe:
                option_string = "very thin, %s" % (color_str)
            else:
                option_string = "very thin, gray, fill=%s,fill opacity=0.3" % (
                    color_str
                )
            tikz_string += tikz_cuboid(interval, option_string, next(label_iter))

        tikz_string += "\\end{tikzpicture}\n"
        tikz_string += "\\end{document}\n"
        return tikz_string

    latex_string = tikz_grid(intervals, wireframe)
    if filename is None:
        filename = "tikz_cuboids"
    if wireframe:
        filename += "_wireframe.tex"
    else:
        filename += "_solid.tex"
    latex_write_and_compile(latex_string, filename)


def plot_tree_tikz(
    refinement_descriptor: RefinementDescriptor, labels=None, filename="omnitree"
):
    tikz_string = R"""\documentclass{standalone}
% autogenerated with dyada : https://github.com/freifrauvonbleifrei/DyAda
\usepackage{forest}
\begin{document}
% cf. https://tex.stackexchange.com/questions/332300/draw-lines-on-top-of-tikz-forest
\forestset{%
    declare keylist register={through},
    through={},
    tracing tree/.style={%
    delay={%
        for #1={%
        if phantom={}{through+/.option=name},
        }
    },
    before drawing tree={%
        tikz+/.wrap pgfmath arg={%
        \foreach \i [count=\j, remember=\i as \k] in {##1} \ifnum\j>1 \draw [densely dashed, ->] (\k.west) -- (\i.west)\fi;
        }{(through)}
    },
    }
}
\tikzset{every label/.style={xshift=-2ex,yshift=-2ex , font=\footnotesize, text=red}}
"""
    n_leaves = refinement_descriptor.get_num_boxes()
    tikz_string = latex_add_color_defs(tikz_string, n_leaves)

    assert labels is None or len(labels) == len(refinement_descriptor)

    tikz_string += "\\begin{forest}\n     tracing tree=tree,\n"
    d_zeros = "0" * refinement_descriptor.get_num_dimensions()
    leaf_string = (
        "[" + d_zeros + ",circle,draw,fill=color_%d,fill opacity=0.4,label={%s}]\n"
    )
    last_indent = 0
    tab = "    "
    num_box = 0
    num_patch = 0
    try:
        for current_branch, refinement in branch_generator(refinement_descriptor):
            current_indent = len(current_branch)
            while current_indent < last_indent:
                tikz_string += tab * current_indent + "]\n"
                last_indent -= 1
            if labels is not None:
                label_string = labels[num_patch]
            else:
                label_string = str(num_patch)
            if refinement.count() == 0:
                tikz_string += tab * current_indent + leaf_string % (
                    num_box,
                    label_string,
                )
                num_box += 1
            else:
                tikz_string += (
                    tab * current_indent
                    + "["
                    + refinement.to01()
                    + ",label={"
                    + label_string
                    + "}\n"
                )
            last_indent = current_indent
            num_patch += 1
    except IndexError:
        # if this happens, the descriptor is not valid but we can plot part of it
        warnings.warn("Descriptor is not valid, plotting as much as possible.")
        pass

    tikz_string += "]" * (last_indent)
    tikz_string += "\n\\end{forest}\n\\end{document}"

    if not filename.endswith(".tex"):
        filename += ".tex"
    latex_write_and_compile(tikz_string, filename)


def plot_descriptor_tikz(
    refinement_descriptor: RefinementDescriptor, filename="descriptor"
):
    tikz_string = R"""\documentclass{standalone}
% autogenerated with dyada : https://github.com/freifrauvonbleifrei/DyAda
\usepackage{tikz}
\usetikzlibrary{matrix}
\begin{document}"""
    num_colors = refinement_descriptor.get_num_boxes()
    tikz_string = latex_add_color_defs(tikz_string, num_colors)
    tikz_string += R"""\begin{tikzpicture}[every node/.style={draw,align=center,text height=2ex,minimum width=2ex, inner sep=0.2ex, fill opacity=0.4, text opacity=1}]
    \matrix [draw=none, matrix of nodes,nodes in empty cells]
    {"""
    tab = "    "
    box_counter = 0
    for refinement in refinement_descriptor:
        tikz_string += tab
        if refinement.count() == 0:
            tikz_string += f"|[fill=color_{box_counter}]| "
            box_counter += 1
        tikz_string += refinement.to01() + "&\n"

    # remove last newline and ampersand
    tikz_string = tikz_string[:-2]
    tikz_string += R"""\\
    };   
\end{tikzpicture}
\end{document}
"""
    if not filename.endswith(".tex"):
        filename += ".tex"
    latex_write_and_compile(tikz_string, filename)


def plot_location_stack_tikz(discretization: Discretization, filename="location_stack"):
    descriptor = discretization.descriptor
    location_stack = discretization_to_location_stack_strings(discretization)

    tikz_string = R"""\documentclass[tikz]{standalone}
% autogenerated with dyada : https://github.com/freifrauvonbleifrei/DyAda
\usepackage{tikz}
\usetikzlibrary{matrix}
\begin{document}"""
    num_colors = descriptor.get_num_boxes()
    tikz_string = latex_add_color_defs(tikz_string, num_colors)
    tikz_string += R"""\begin{tikzpicture}[
       every node/.style={align=left,anchor=west,
       text height=2ex,minimum width=2ex,
       inner sep=0.2ex,
       fill opacity=0.4, text opacity=1}]
    \matrix [
       draw=none, matrix of nodes] (n)
    {"""
    tab = "    "
    box_counter = 0
    for location_code_tuple, refinement in zip(location_stack, descriptor):
        tikz_string += tab
        for dimension in range(descriptor.get_num_dimensions()):
            location_code_string = "\\texttt{" + location_code_tuple[dimension] + "}"
            if dimension < descriptor.get_num_dimensions() - 1:
                location_code_string = location_code_string + "&"
            location_code_string = location_code_string.replace(
                "∩", "\\scalebox{0.79}{$\\cap$}"
            )
            location_code_string = location_code_string.replace(
                "λ", "\\scalebox{0.85}{$\\lambda$}"
            )

            if refinement.count() == 0 and len(location_code_tuple[dimension]) > 0:
                tikz_string += f"|[fill=color_{box_counter}]| "
            tikz_string += location_code_string
        tikz_string += "\\\\ \n"
        if refinement.count() == 0:
            box_counter += 1
    tikz_string = tikz_string[:-2]
    tikz_string += "\n};\n"
    tikz_string += "\\draw "
    for column in range(descriptor.get_num_dimensions()):
        tikz_string += f"(n-1-{column+1}.north west) -- (n-{len(location_stack)}-{column+1}.south west)"
    tikz_string += ";"
    tikz_string += R"""
\end{tikzpicture}
\end{document}
"""
    if not filename.endswith(".tex"):
        filename += ".tex"
    latex_write_and_compile(tikz_string, filename)


@depends_on_optional("OpenGL")
def draw_cuboid_opengl(
    interval: CoordinateInterval,
    projection: Sequence[int] = [0, 1, 2],
    wireframe: bool = False,
    alpha: float = 0.1,
    linewidth: float = 1.0,
    color=(0.5, 0.5, 0.5),
):
    color = to_rgb(color)

    if wireframe:
        alpha_faces = 0.0
        alpha_lines = alpha
        color_faces = (0.0, 0.0, 0.0)
        color_lines = color
    else:
        alpha_faces = alpha
        alpha_lines = 1.0
        color_faces = color
        color_lines = (0.0, 0.0, 0.0)

    lower = interval[0][projection]
    upper = interval[1][projection]

    gl.glLineWidth(linewidth)
    gl.glBegin(gl.GL_LINES)
    gl.glColor4fv((*color_lines, alpha_lines))
    for side_corners in side_corners_generator(lower, upper):
        for side in pairwise(side_corners):
            gl.glVertex3fv(side[0])
            gl.glVertex3fv(side[1])
    gl.glEnd()

    gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)  # try to avoid overdrawing
    gl.glPolygonOffset(1.0, 1.0)
    gl.glBegin(gl.GL_QUADS)
    gl.glColor4fv((*color_faces, alpha_faces))
    for side_corners in side_corners_generator(lower, upper):
        for corner in side_corners:
            gl.glVertex3fv(corner)
    gl.glEnd()
    gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)


@depends_on_optional("OpenGL")
def gl_save_file(filename: str, width=1024, height=1024) -> None:
    gl.glFlush()
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), data)
    image = ImageOps.flip(image)
    if not filename.endswith(".png"):
        filename += ".png"
    image.save(filename, "PNG")


@depends_on_optional("OpenGL")
def plot_boxes_3d_pyopengl(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]] = None,
    projection: Sequence[int] = [0, 1, 2],
    wireframe: bool = False,
    filename: str = "omnitree",
    width: int = 1024,
    height: int = 1024,
    alpha: float = 0.1,
    linewidth: float = 1.0,
    **kwargs,
) -> None:
    if labels is not None:
        warnings.warn("Labels are currently not used in 3D plots w/ pyopengl")

    colors = kwargs.pop("colors", get_colors(len(intervals)))
    if isinstance(colors, str):
        colors = [colors] * len(intervals)

    def init_glu():
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
        glut.glutInitWindowSize(width, height)
        glut.glutCreateWindow(b"Dyada OpenGL")
        glut.glutHideWindow()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(
            35.0,
            float(width) / height,
            0.1,
            10.0,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        # move camera back to see the unit cube
        glu.gluLookAt(
            *(-1.5, 1.5, 3.0),
            *(0.5, 0.5, 0.5),
            *(0, 1, 0),
        )
        # for opacity
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # draw background
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    init_glu()

    for interval, color in zip(intervals, colors):
        draw_cuboid_opengl(
            interval,
            projection,
            wireframe=wireframe,
            alpha=alpha,
            linewidth=linewidth,
            color=color,
        )

    if filename is not None:
        gl_save_file(filename, width, height)


def add_cuboid_to_buffer(
    buffer: StringIO,
    vertex_offset: int,
    interval: CoordinateInterval,
    projection: Sequence[int] = [0, 1, 2],
    wireframe: bool = False,
):
    verts, faces, edges = cuboid_from_interval(interval, projection)
    # Write vertices
    for v in verts:
        buffer.write(f"v {v[0]} {v[1]} {v[2]}\n")

    if wireframe:
        # Write edges
        for edge in edges:
            a, b = (vertex_offset + i for i in edge)
            buffer.write(f"l {a} {b}\n")
    else:
        # Write faces
        for quad in faces:
            a, b, c, d = (vertex_offset + i for i in quad)
            buffer.write(f"f {a} {b} {c} {d}\n")

    vertex_offset += len(verts)  # Move offset for next cuboid
    return buffer, vertex_offset


def write_obj_file(buffer: StringIO, filename: str):
    if not filename.endswith(".obj"):
        filename += ".obj"
    with open(filename, "w") as f:
        return f.write(buffer.getvalue())


def export_boxes_3d_to_obj(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    projection: Sequence[int] = [0, 1, 2],
    wireframe: bool = False,
    filename: str = "omnitree",
    **kwargs,
):
    buffer = StringIO()
    # .obj is 1-indexed
    vertex_offset = 1
    for interval in intervals:
        buffer, vertex_offset = add_cuboid_to_buffer(
            buffer, vertex_offset, interval, projection, wireframe
        )

    if filename is None:
        return buffer, vertex_offset
    return write_obj_file(buffer, filename)
