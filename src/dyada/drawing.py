import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib not found, plotting functions will not work")

from cmap import Colormap
from itertools import product
from pathlib import Path
from typing import Sequence, Union, Mapping, Optional
import subprocess

from dyada.coordinates import CoordinateInterval, get_coordinates_from_level_index
from dyada.descriptor import branch_generator, RefinementDescriptor
from dyada.refinement import Discretization
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
    if backend == "tikz":
        return plot_boxes_3d_tikz(intervals, labels, projection, **kwargs)
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


@depends_on_optional("matplotlib.pyplot")
def plot_boxes_2d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    **kwargs,
) -> None:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    filename = None
    if "filename" in kwargs:
        filename = kwargs.pop("filename")

    fig, ax1 = plt.subplots(1, 1)
    for i, interval in enumerate(intervals):
        anchor_point = interval.lower_bound[projection]
        extent = interval.upper_bound[projection] - anchor_point
        rectangle = plt.Rectangle(
            (anchor_point[0], anchor_point[1]),
            extent[0],
            extent[1],
            fill=True,
            figure=fig,
            color=colors[i % len(colors)],
            **kwargs,
        )
        ax1.add_artist(rectangle)

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
        subprocess.run(
            ["latexmk", "-interaction=nonstopmode", "-pdf", filename],
            check=True,
            cwd=dirname,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
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
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pass


def latex_add_color_defs(
    tikz_string: str, num_colors: int, colormap_name="CET_R3"
) -> str:
    cm = Colormap(colormap_name)
    for leaf in range(num_colors):
        colormap = cm(leaf * 1.0 / num_colors)
        color = [int(255 * c) for c in colormap]

        color_str = "color_%d" % leaf
        tikz_string += "\\definecolor{%s}{RGB}{%d,%d,%d}\n" % (
            color_str,
            color[0],
            color[1],
            color[2],
        )
    return tikz_string


# inspired by @griegler : https://github.com/griegler/octnet/issues/28
def plot_boxes_3d_tikz(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    wireframe: bool = False,
    filename: Optional[str] = None,
    **kwargs,
) -> None:

    def tikz_cube(
        interval: CoordinateInterval, option_string="", label_string=""
    ) -> str:
        tikz_string = ""
        line_string = "\\draw[%s] (%f,%f,%f) -- (%f,%f,%f) -- (%f,%f,%f) -- (%f,%f,%f) -- cycle;\n"
        lower = interval[0][projection]
        upper = interval[1][projection]
        # iterate the six sides of the cube
        # by always selecting four corners that have one coordinate in common
        corners = list(product(*zip(lower, upper)))
        for bound in [lower, upper]:
            for i, b in enumerate(bound):
                side_corners = list(filter(lambda c: c[i] == b, corners))
                assert len(side_corners) == 4
                tikz_string += line_string % (
                    option_string,
                    *side_corners[0],
                    *side_corners[1],
                    *side_corners[3],
                    *side_corners[2],
                )
        middle = (lower + upper) / 2.0
        extent = upper - lower
        if min(extent) < 0.125:
            label_string = "\\tiny \\relsize{-1} " + label_string
        elif min(extent) < 0.25:
            label_string = "\\tiny \\relsize{-0.5}" + label_string
        elif min(extent) < 0.5:
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
            tikz_string += tikz_cube(interval, option_string, next(label_iter))

        tikz_string += "\\end{tikzpicture}\n"
        tikz_string += "\\end{document}\n"
        return tikz_string

    latex_string = tikz_grid(intervals, wireframe)
    if filename is None:
        filename = "tikz_cubes"
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
            tikz_string += tab * current_indent + leaf_string % (num_box, label_string)
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
