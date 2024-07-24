import matplotlib.pyplot as plt
from typing import Sequence, Union, Mapping, Optional

from dyada.coordinates import CoordinateInterval


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
            transform=fig.transFigure,
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
    plt.show()
