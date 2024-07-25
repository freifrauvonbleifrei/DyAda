import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib not found, plotting functions will not work")

from typing import Sequence, Union, Mapping, Optional

from dyada.coordinates import CoordinateInterval, get_coordinates_from_level_index
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


def plot_all_boxes_2d(refinement, projection: Sequence[int] = [0, 1], **kwargs) -> None:
    level_indices = list(refinement.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    plot_boxes_2d(coordinates, projection=projection, **kwargs)
