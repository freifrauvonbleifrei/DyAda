# SPDX-FileCopyrightText: 2025 Theresa Pollinger
#
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

try:
    from matplotlib.colors import to_rgb
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
from itertools import pairwise
from typing import Sequence, Union, Mapping, Optional

from dyada.coordinates import (
    CoordinateInterval,
)
from dyada.drawing_util import (
    cuboid_from_interval,
    side_corners_generator,
    get_colors,
)
from dyada.structure import depends_on_optional


@depends_on_optional("OpenGL")
def draw_cuboid_opengl(
    interval: CoordinateInterval,
    projection: Sequence[int],
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
    labels: Optional[Sequence[str]],
    projection: Sequence[int],
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
    projection: Sequence[int],
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
    projection: Sequence[int],
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
