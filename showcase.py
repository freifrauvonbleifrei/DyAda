from queue import PriorityQueue
from typing import Callable
import bitarray as ba
import numpy as np
import numpy.typing as npt
from dyada import (
    get_coordinates_from_level_index,
    get_level_index_from_linear_index,
    Discretization,
    CoordinateInterval,
    PlannedAdaptiveRefinement,
    MortonOrderLinearization,
    RefinementDescriptor
)
from dyada.drawing_util import (
    side_corners_generator,
)
from matplotlib import pyplot as plt, animation
from matplotlib.axes import Axes
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def refine(
        discretization: Discretization,
        max_num_boxes: int,
        check_inside: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool]],
        cutoff_percentage: float,
        progress: bool = True
) -> tuple[Discretization, PriorityQueue]:
    """
    Refine "check_inside" using a Dyada discretization.
    :param discretization: The baseline discretization.
    :param max_num_boxes: The maximum number of boxes, the discretization is allowed to have.
    :param check_inside: The function to approximate.
    :param cutoff_percentage: The Percentage of importance drop of between refinements that the refinements should be applied after.
    :param progress: If True, print current progress.
    :return: The finished Discretization and the remaining refinement queue.
    """
    priority_queue: PriorityQueue[tuple[float, int, int]] = PriorityQueue()
    for box_index in range(len(discretization)):
        level_index = get_level_index_from_linear_index(
            discretization.linearization, discretization.descriptor, box_index
        )
        interval = get_coordinates_from_level_index(level_index)
        for importance, dim in calc_importance(interval, check_inside):
            priority_queue.put((-importance, dim, box_index))

    while len(discretization) < max_num_boxes and not priority_queue.empty():
        if progress:
            current = len(discretization)
            pct = current / max_num_boxes
            prog = int(20 * pct)
            bar = f"{'█' * prog + '░' * (20 - prog)}"
            print(f"refining... [{bar}] len:{current:5} progress: ({pct:6.1%})")
        p = PlannedAdaptiveRefinement(discretization)
        first_priority, first_dim, first_box_index = priority_queue.get()

        p.plan_refinement(first_box_index, dim_to_refinement(first_dim))
        indices_to_refine: set[int] = {first_box_index}

        while len(discretization) + len(
                indices_to_refine) * 4 < max_num_boxes and not priority_queue.empty():
            second_priority, second_dim, second_box_index = priority_queue.get()
            if second_priority > cutoff_percentage * first_priority:
                priority_queue.put((second_priority, second_dim, second_box_index))
                break
            indices_to_refine.add(second_box_index)
            p.plan_refinement(second_box_index, dim_to_refinement(second_dim))
        discretization, index_mapping = p.apply_refinements(track_mapping="boxes")
        new_priority_queue: PriorityQueue = PriorityQueue()

        while not priority_queue.empty():
            second_priority, second_dim, second_box_index = priority_queue.get()
            if len(index_mapping[second_box_index]) != 1:
                continue
            new_index = index_mapping[second_box_index].pop()
            new_priority_queue.put((second_priority, second_dim, new_index))

        for next_refinement_index in indices_to_refine:
            for box_index in index_mapping[next_refinement_index]:
                level_index = get_level_index_from_linear_index(
                    discretization.linearization, discretization.descriptor, box_index
                )
                interval = get_coordinates_from_level_index(level_index)
                for importance, dim in calc_importance(interval, check_inside):
                    new_priority_queue.put((-importance, dim, box_index))
        priority_queue = new_priority_queue
    return discretization, priority_queue


def dim_to_refinement(index: int) -> ba.bitarray:
    """
    Compute the refinement for a given axis.
    :param index: The dimension index (0 indexed) in which to refine.
    :return: The refinement bitarray.
    """
    out = ba.bitarray(4)
    out[index] = 1
    return out


def check_inside_rotating_cube(
        points: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool]:
    """
    Check which points are contained inside the area.
    :param points: The points for which to check. The points are being mutated. The points should have the shape: [number_of_points,4]
    :return: which points are contained in the area. This array has the shape: [number_of_points]
    """

    sin_w = np.sin(points[:, 3] * np.pi * 4)
    cos_w = np.cos(points[:, 3] * np.pi * 4)
    temp_x = points[:, 0] - 0.5
    temp_y = points[:, 1] - 0.5
    points[:, 0] = temp_x * cos_w + temp_y * sin_w + 0.5
    points[:, 1] = - temp_x * sin_w + temp_y * cos_w + 0.5
    points[:, 2] = points[:, 2] - 0.25 + sin_w * 1 / 4
    return np.all((points[:, :3] <= 1 / 2) & (points[:, :3] >= 0.2), axis=1)


def check_inside_rotating_tetraeder(
        points: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Check which points are contained inside the area.
    :param points: The points for which to check. The points are being mutated. The points should have the shape: [number_of_points,4]
    :return: which points are contained in the area. This array has the shape: [number_of_points]
    """

    sin_w = np.sin(points[:, 3] * np.pi * 2)
    cos_w = np.cos(points[:, 3] * np.pi * 2)
    temp_x = points[:, 0] - 0.5
    temp_y = points[:, 1] - 0.5
    temp_z = points[:, 2]
    points[:, 0] = temp_x * cos_w + temp_y * sin_w + 0.25
    points[:, 1] = - temp_x * sin_w + temp_y * cos_w + 0.25
    points[:, 2] = -1 / 2 - sin_w * 1 / 8 + temp_z
    return (
            (points[:, 0] >= 0) &
            (points[:, 1] >= 0) &
            (2 * points[:, 0] + points[:, 1] - points[:, 2] <= 0) &
            (points[:, 2] + points[:, 1] - 1 / 2 <= 0)
    )


def calc_importance(interval: CoordinateInterval,
                    check_inside: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool]],
                    points_per_axis: float | int = 10.0) -> list[
    tuple[float, int]]:
    """
    Calculate the importance of refinement along each axis.
    :param interval: The interval to test for refinement
    :param check_inside: the function used to determine which points are "inside" and "outside"
    :param points_per_axis: In how many points the interval should be split.
    :return: A List of Tuples containing: (importance, dimension (0 indexed))
    """
    points = np.mgrid[*[slice(0, 1, 1 / points_per_axis)] * 4]
    size = interval[1] - interval[0]
    for i in range(4):
        points[i, ...] *= size[i]
        points[i, ...] += interval[0][i]
    vals = check_inside(points.reshape([4, -1]).T).reshape(points.shape[1:])
    out = [(np.var(vals, axis=d).mean() * size[d] ** 2, d) for d in range(4)]
    return out


def plot_all_boxes_3d(
        discretization: Discretization,
        check_inside: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool]],
        wireframe: bool = False,
        **kwargs,
):
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [get_coordinates_from_level_index(box_li) for box_li in level_indices]
    lower_bounds = np.array([interval.lower_bound for interval in coordinates])
    upper_bounds = np.array([interval.upper_bound for interval in coordinates])
    mid = (lower_bounds + upper_bounds) / 2
    inside = check_inside(mid)
    fig = plt.figure()
    # noinspection PyTypeChecker
    ax1: Axes3D = fig.add_axes((0, 0.1, 1, 0.9),
                               projection="3d")  # because matplotlib uses dynamic typing, the typechecker breaks here

    class Bpause:
        def __init__(self, ax: Axes):
            self.pause = False
            self.button = Button(ax, "pause")
            self.button.on_clicked(self.toggle)

        def toggle(self, _):
            self.pause = not self.pause
            self.button.label.set_text("play" if self.pause else "pause")

    b = Bpause(fig.add_axes((0.8, 0.025, 0.1, 0.05)))

    class Stime:
        def __init__(self, ax: Axes):
            self.time: float = 0.0
            self.slider = Slider(ax, "time", 0, 1)
            self.slider.on_changed(self.change)

        def change(self, val: float):
            self.time = val

        def inc(self, val: float):
            self.time += val
            while self.time > 1:
                self.time -= 1
            self.slider.set_val(self.time)

    s = Stime(fig.add_axes((0.1, 0.0, 0.6, 0.1)))

    def update(_):
        # noinspection PyProtectedMember
        view = ax1._get_view()  # the use of the protected function is just for convenience.
        ax1.clear()
        # noinspection PyProtectedMember
        ax1._set_view(view)
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')
        if not b.pause:
            s.inc(0.01)

        for i, interval in enumerate(coordinates):
            draw_cuboid_on_axis(
                ax1,
                interval,
                s.time,
                bool(inside[i]),
                wireframe,
                **kwargs,
            )
        return []

    return animation.FuncAnimation(fig=fig, func=update, cache_frame_data=False), b, s


def draw_cuboid_on_axis(
        ax: plt.Axes,
        interval: CoordinateInterval,
        time: float,
        inside: bool,
        wireframe: bool,
        **kwargs,
) -> plt.Axes:
    """
    Plot a cuboid of given position and size on a given axis.
    """
    if not (interval[0][3] <= time <= interval[1][3]):
        return ax
    if wireframe or inside:
        faces = list(side_corners_generator(interval.lower_bound[:-1], interval.upper_bound[:-1]))
        cuboid = Poly3DCollection(
            faces,
            facecolors=(0.58, 0.4, 0.6, 1.) if inside else (0, 0, 0, 0),
            edgecolors=(0.0, 0.1, 0.1, 0.04) if wireframe else (0, 0, 0, 0),
            **kwargs,
        )
        ax.add_collection(cuboid)
    return ax


if __name__ == "__main__":
    disc = Discretization(
        MortonOrderLinearization(),
        RefinementDescriptor(4, 0),
    )
    disc, _ = refine(disc, 4096, check_inside_rotating_cube, 0.5)
    anim, button, slider = plot_all_boxes_3d(disc, check_inside_rotating_cube, wireframe=True)
    plt.show()
