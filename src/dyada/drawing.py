import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib not found, plotting functions will not work")

from pathlib import Path
from typing import Sequence, Union, Mapping, Optional
import subprocess

from dyada.coordinates import CoordinateInterval, get_coordinates_from_level_index
from dyada.descriptor import branch_generator
from dyada.structure import depends_on_optional


def plot_boxes_2d(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]] = None,
    projection: Sequence[int] = [0, 1],
    backend="matplotlib",
    **kwargs,
) -> None:
    assert len(projection) == 2
    if labels is None:
        labels = [str(i) for i in range(len(intervals))]
    if backend == "matplotlib":
        return plot_boxes_2d_matplotlib(intervals, labels, projection, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_all_boxes_2d(
    refinement, projection: Sequence[int] = [0, 1], labels="patches", **kwargs
) -> None:
    level_indices = list(refinement.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    if labels == "patches":
        labels = []
        for i in range(len(refinement._descriptor)):
            if refinement._descriptor.is_box(i):
                labels.append(str(i))
    if labels == "boxes":
        labels = None
    plot_boxes_2d(coordinates, projection=projection, labels=labels, **kwargs)


def plot_boxes_3d(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]] = None,
    projection: Sequence[int] = [0, 1, 2],
    backend: str = "tikz",
    **kwargs,
) -> None:
    assert len(projection) == 3
    if labels is None:
        labels = [str(i) for i in range(len(intervals))]
    if backend == "tikz":
        return plot_boxes_3d_tikz(intervals, labels, projection, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_all_boxes_3d(
    refinement, projection: Sequence[int] = [0, 1, 2], **kwargs
) -> None:
    level_indices = list(refinement.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    plot_boxes_3d(coordinates, projection=projection, **kwargs)


@depends_on_optional("matplotlib.pyplot")
def plot_boxes_2d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
    **kwargs,
) -> None:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

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
        subprocess.run(
            ["latexmk", "-c"],
            check=False,
            cwd=dirname,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"Error while running latexmk on {filename}: {e}")


def latex_add_color_defs(
    tikz_string: str, num_colors: int, colormap_name="viridis"
) -> str:
    cm = plt.get_cmap(colormap_name)
    for leaf in range(num_colors):
        colormap = cm(leaf * 1.0 / num_colors)
        color = [int(255 * c) for c in colormap[:3]]

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
    projection: Sequence[int],  # TODO use
    wireframe: bool = False,
    filename: Optional[str] = None,
    **kwargs,
) -> None:

    def tikz_cube(interval: CoordinateInterval, option_string="") -> str:
        tikz_string = ""
        line_string = "\\draw[%s] (%f,%f,%f) -- (%f,%f,%f) -- (%f,%f,%f) -- (%f,%f,%f) -- cycle;\n"
        lower = interval[0]
        upper = interval[1]
        # iterate the six sides of the cube
        # by always selecting four corners that have one coordinate in common
        # TODO use iterator
        tikz_string += line_string % (
            option_string,
            *lower,
            lower[0],
            upper[1],
            lower[2],
            upper[0],
            upper[1],
            lower[2],
            upper[0],
            lower[1],
            lower[2],
        )

        tikz_string += line_string % (
            option_string,
            *lower,
            lower[0],
            lower[1],
            upper[2],
            lower[0],
            upper[1],
            upper[2],
            lower[0],
            upper[1],
            lower[2],
        )

        tikz_string += line_string % (
            option_string,
            lower[0],
            upper[1],
            lower[2],
            upper[0],
            upper[1],
            lower[2],
            *upper,
            lower[0],
            upper[1],
            upper[2],
        )
        tikz_string += line_string % (
            option_string,
            *lower,
            upper[0],
            lower[1],
            lower[2],
            upper[0],
            lower[1],
            upper[2],
            lower[0],
            lower[1],
            upper[2],
        )
        tikz_string += line_string % (
            option_string,
            lower[0],
            lower[1],
            upper[2],
            lower[0],
            upper[1],
            upper[2],
            *upper,
            upper[0],
            lower[1],
            upper[2],
        )
        tikz_string += line_string % (
            option_string,
            upper[0],
            lower[1],
            lower[2],
            upper[0],
            lower[1],
            upper[2],
            *upper,
            upper[0],
            upper[1],
            lower[2],
        )
        return tikz_string

    def tikz_grid(intervals, wireframe: bool):
        tikz_string = R"""\documentclass{standalone}
% inspired by @griegler : https://github.com/griegler/octnet/issues/28
\usepackage{xcolor}
\usepackage{tikz,tikz-3dplot}
\begin{document}
\tdplotsetmaincoords{50}{130}
\begin{tikzpicture}[scale=1.0, tdplot_main_coords]"""
        tikz_string = latex_add_color_defs(tikz_string, len(intervals))
        for grid_idx, interval in enumerate(intervals):
            color_str = "color_%d" % grid_idx
            if wireframe:
                option_string = "very thin, %s" % (color_str)
            else:
                option_string = "very thin, gray, fill=%s,fill opacity=0.3" % (
                    color_str
                )
            tikz_string += tikz_cube(interval, option_string)

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


def plot_tree_tikz(refinement_descriptor, filename="pow2tree"):
    tikz_string = R"""\documentclass{standalone}
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

    tikz_string += "\\begin{forest}\n     tracing tree=tree,\n"
    d_zeros = "0" * refinement_descriptor.get_num_dimensions()
    leaf_string = (
        "[" + d_zeros + ",circle,draw,fill=color_%d,fill opacity=0.4,label=%d]\n"
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
        if refinement.count() == 0:
            tikz_string += tab * current_indent + leaf_string % (num_box, num_patch)
            num_box += 1
        else:
            tikz_string += (
                tab * current_indent
                + "["
                + refinement.to01()
                + ",label="
                + str(num_patch)
                + "\n"
            )
        last_indent = current_indent
        num_patch += 1

    tikz_string += "]" * (last_indent)
    tikz_string += "\n\\end{forest}\n\\end{document}"

    if not filename.endswith(".tex"):
        filename += ".tex"
    latex_write_and_compile(tikz_string, filename)


def plot_descriptor_tikz(refinement_descriptor, filename="descriptor"):
    tikz_string = R"""\documentclass{standalone}
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
