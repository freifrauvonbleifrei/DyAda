# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
    from matplotlib.colors import to_rgba
except ImportError:
    warnings.warn("matplotlib not found, some plotting functions will not work")

from typing import Sequence, Union, Mapping, Optional

from dyada.coordinates import (
    CoordinateInterval,
    get_coordinates_from_level_index,
)
from dyada.discretization import (
    Discretization,
)
from dyada.drawing_opengl_obj import plot_boxes_3d_pyopengl, export_boxes_3d_to_obj
from dyada.drawing_tikz import plot_boxes_3d_tikz, plot_boxes_2d_tikz
from dyada.drawing_util import (
    labels_from_discretization,
    side_corners_generator,
    boxes_to_2d_ascii,
)
from dyada.structure import depends_on_optional


def plot_boxes_2d(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]] = None,
    projection: Sequence[int] | None = None,
    backend="matplotlib",
    **kwargs,
) -> None:
    if projection is None:
        projection = [0, 1]
    assert len(projection) == 2
    if backend == "matplotlib":
        return plot_boxes_2d_matplotlib(intervals, labels, projection, **kwargs)
    elif backend == "tikz":
        return plot_boxes_2d_tikz(intervals, labels, projection, **kwargs)
    elif backend == "ascii":
        print(boxes_to_2d_ascii(intervals, projection=projection, **kwargs))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_all_boxes_2d(
    discretization: Discretization,
    projection: Sequence[int] | None = None,
    labels: Union[None, str, Sequence[str]] = "patches",
    **kwargs,
) -> None:
    if projection is None:
        projection = [0, 1]
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    labels = labels_from_discretization(discretization, labels)
    plot_boxes_2d(coordinates, projection=projection, labels=labels, **kwargs)


def plot_boxes_3d(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Union[None, str, Sequence[str]] = None,
    projection: Sequence[int] | None = None,
    backend: str = "tikz",
    **kwargs,
) -> None:
    if projection is None:
        projection = [0, 1, 2]
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
    projection: Sequence[int] | None = None,
    labels: Union[None, str, Sequence[str]] = "patches",
    **kwargs,
) -> None:
    if projection is None:
        projection = [0, 1, 2]
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    labels = labels_from_discretization(discretization, labels)
    plot_boxes_3d(coordinates, projection=projection, labels=labels, **kwargs)


def discretization_to_2d_ascii(
    discretization: Discretization,
    resolution=None,
    projection: Sequence[int] | None = None,
    **kwargs,
) -> str:
    if projection is None:
        projection = [0, 1]
    if resolution is None:
        max_level = discretization._descriptor.get_maximum_level()
        resolution = (2 ** (max_level[0] + 1), 2 ** max_level[1])
        return discretization_to_2d_ascii(discretization, resolution)
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    return boxes_to_2d_ascii(
        coordinates, projection=projection, resolution=resolution, **kwargs
    )


def discretization_str(discretization: Discretization):
    dimensionality = discretization._descriptor.get_num_dimensions()
    if dimensionality == 2:
        return discretization_to_2d_ascii(discretization)
    else:
        return repr(discretization)


# Monkey-patch Discretization.__str__ to use this function
Discretization.__str__ = wraps(Discretization.__str__)(discretization_str)  # type: ignore


@depends_on_optional("matplotlib.pyplot")
def get_figure_2d_matplotlib(
    intervals: Union[Sequence[CoordinateInterval], Mapping[CoordinateInterval, str]],
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
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
    projection: Sequence[int],
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
    projection: Sequence[int],
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
