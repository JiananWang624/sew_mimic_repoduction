"""Headless visualization primitives in the authoritative Gen3 base frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common import HumanArmTarget, gen3_end_effector_pose
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, StereoSewSingularityError


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Color = tuple[float, float, float, float]
LineKind = Literal["line", "arrow"]

HUMAN_MARKER_RADIUS_M = 0.020
TASK_MARKER_RADIUS_M = 0.016
PINCH_MARKER_RADIUS_M = 0.018
SEW_MARKER_RADIUS_M = 0.012
ARM_LINE_WIDTH_M = 0.004
OVERLAY_LINE_WIDTH_M = 0.003
HAND_AXIS_LENGTH_M = 0.10
SEW_NORMAL_LENGTH_M = 0.10

HUMAN_COLORS: tuple[Color, Color, Color] = (
    (1.0, 0.25, 0.10, 1.0),
    (1.0, 0.60, 0.10, 1.0),
    (1.0, 0.90, 0.10, 1.0),
)
TARGET_COLOR: Color = (0.10, 1.0, 0.20, 1.0)
ROBOT_COLOR: Color = (0.10, 0.75, 1.0, 1.0)
ERROR_COLOR: Color = (1.0, 0.10, 1.0, 1.0)
HUMAN_SEW_COLOR: Color = (1.0, 0.45, 0.10, 1.0)
ROBOT_SEW_COLOR: Color = (0.25, 0.55, 1.0, 1.0)
HUMAN_TRAIL_COLOR: Color = (1.0, 0.65, 0.20, 0.75)
ROBOT_TRAIL_COLOR: Color = (0.20, 0.90, 1.0, 0.75)


def _readonly_vector(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape (3,)")
    return np.frombuffer(vector.tobytes(), dtype=np.float64)


def _readonly_rotation(value: ArrayLike, name: str) -> Matrix:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be finite with shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return np.frombuffer(rotation.tobytes(), dtype=np.float64).reshape(3, 3)


def _color(value: Color) -> Color:
    rgba = tuple(float(component) for component in value)
    if len(rgba) != 4 or not all(np.isfinite(rgba)):
        raise ValueError("color must contain four finite values")
    return rgba  # type: ignore[return-value]


@dataclass(frozen=True)
class SpherePrimitive:
    name: str
    position: Vector
    radius_m: float
    color: Color

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _readonly_vector(self.position, "position"))
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        object.__setattr__(self, "color", _color(self.color))


@dataclass(frozen=True)
class LinePrimitive:
    name: str
    start: Vector
    end: Vector
    width_m: float
    color: Color
    kind: LineKind = "line"

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _readonly_vector(self.start, "start"))
        object.__setattr__(self, "end", _readonly_vector(self.end, "end"))
        if not np.isfinite(self.width_m) or self.width_m <= 0.0:
            raise ValueError("width_m must be positive and finite")
        if self.kind not in ("line", "arrow"):
            raise ValueError("kind must be 'line' or 'arrow'")
        object.__setattr__(self, "color", _color(self.color))


@dataclass(frozen=True)
class AxisPrimitive:
    name: str
    origin: Vector
    rotation: Matrix
    length_m: float = HAND_AXIS_LENGTH_M

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _readonly_vector(self.origin, "origin"))
        object.__setattr__(
            self, "rotation", _readonly_rotation(self.rotation, "rotation")
        )
        if not np.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("length_m must be positive and finite")


@dataclass(frozen=True)
class OverlayFrame:
    """Viewer-independent primitives, all expressed in one documented frame."""

    spheres: tuple[SpherePrimitive, ...] = ()
    lines: tuple[LinePrimitive, ...] = ()
    axes: tuple[AxisPrimitive, ...] = ()

    def sphere(self, name: str) -> SpherePrimitive:
        return next(primitive for primitive in self.spheres if primitive.name == name)

    def line(self, name: str) -> LinePrimitive:
        return next(primitive for primitive in self.lines if primitive.name == name)

    def axis(self, name: str) -> AxisPrimitive:
        return next(primitive for primitive in self.axes if primitive.name == name)


def overlay_base_to_world(
    overlay: OverlayFrame,
    rotation_world_from_base: ArrayLike,
    position_world_of_base: ArrayLike,
) -> OverlayFrame:
    """Convert primitives at the display boundary; numerical targets stay fixed."""
    rotation = _readonly_rotation(rotation_world_from_base, "rotation_world_from_base")
    position = _readonly_vector(position_world_of_base, "position_world_of_base")

    def point(value: Vector) -> Vector:
        return rotation @ value + position

    return OverlayFrame(
        spheres=tuple(
            SpherePrimitive(item.name, point(item.position), item.radius_m, item.color)
            for item in overlay.spheres
        ),
        lines=tuple(
            LinePrimitive(
                item.name,
                point(item.start),
                point(item.end),
                item.width_m,
                item.color,
                item.kind,
            )
            for item in overlay.lines
        ),
        axes=tuple(
            AxisPrimitive(
                item.name,
                point(item.origin),
                rotation @ item.rotation,
                item.length_m,
            )
            for item in overlay.axes
        ),
    )


def _sew_normal(
    shoulder: Vector,
    elbow: Vector,
    wrist: Vector,
    stereo: StereoSew,
) -> Vector:
    # StereoSew.forward supplies the authoritative singularity checks and
    # establishes the same oriented S/E/W convention used by the solvers.
    stereo.forward(shoulder, elbow, wrist)
    normal = np.cross(wrist - shoulder, elbow - shoulder)
    return normal / np.linalg.norm(normal)


def _trail_lines(
    name: str,
    points: Iterable[ArrayLike],
    color: Color,
    *,
    offset: Vector | None = None,
) -> list[LinePrimitive]:
    translated = [
        _readonly_vector(point, f"{name} point")
        + (np.zeros(3) if offset is None else offset)
        for point in points
    ]
    return [
        LinePrimitive(
            f"{name}_{index}",
            start,
            end,
            OVERLAY_LINE_WIDTH_M,
            color,
        )
        for index, (start, end) in enumerate(zip(translated, translated[1:]))
    ]


def _append_sew_geometry(
    prefix: str,
    shoulder: Vector,
    elbow: Vector,
    wrist: Vector,
    normal: Vector,
    color: Color,
    spheres: list[SpherePrimitive],
    lines: list[LinePrimitive],
) -> None:
    for name, point in (("shoulder", shoulder), ("elbow", elbow), ("wrist", wrist)):
        spheres.append(
            SpherePrimitive(f"{prefix}_{name}", point, SEW_MARKER_RADIUS_M, color)
        )
    for name, start, end in (
        ("se", shoulder, elbow),
        ("ew", elbow, wrist),
        ("sw", shoulder, wrist),
    ):
        lines.append(
            LinePrimitive(
                f"{prefix}_plane_{name}",
                start,
                end,
                OVERLAY_LINE_WIDTH_M,
                color,
            )
        )
    center = (shoulder + elbow + wrist) / 3.0
    lines.append(
        LinePrimitive(
            f"{prefix}_normal",
            center,
            center + SEW_NORMAL_LENGTH_M * normal,
            OVERLAY_LINE_WIDTH_M,
            color,
            "arrow",
        )
    )


def build_overlay(
    target: HumanArmTarget,
    robot: Gen3Kinematics,
    geometry: Gen3StereoSewGeometry,
    stereo: StereoSew,
    display_q: ArrayLike | None = None,
    *,
    current_q_available: bool = True,
    human_display_offset_m: ArrayLike = (0.0, 0.0, 0.0),
    show_human: bool = True,
    show_target: bool = True,
    show_sew: bool = False,
    show_error: bool = True,
    human_task_trail: Iterable[ArrayLike] = (),
    robot_pinch_trail: Iterable[ArrayLike] = (),
) -> OverlayFrame:
    """Build one display frame without changing targets, configurations, or metrics."""
    if not isinstance(target, HumanArmTarget):
        raise ValueError("target must be a HumanArmTarget")
    offset = _readonly_vector(human_display_offset_m, "human_display_offset_m")
    human_shoulder = target.shoulder + offset
    human_elbow = target.elbow + offset
    human_wrist = target.wrist + offset
    human_task = target.task_point + offset

    spheres: list[SpherePrimitive] = []
    lines: list[LinePrimitive] = []
    axes: list[AxisPrimitive] = []

    if show_human:
        spheres.extend(
            (
                SpherePrimitive(
                    "human_shoulder", human_shoulder, HUMAN_MARKER_RADIUS_M, HUMAN_COLORS[0]
                ),
                SpherePrimitive(
                    "human_elbow", human_elbow, HUMAN_MARKER_RADIUS_M, HUMAN_COLORS[1]
                ),
                SpherePrimitive(
                    "human_wrist", human_wrist, HUMAN_MARKER_RADIUS_M, HUMAN_COLORS[2]
                ),
                SpherePrimitive(
                    "human_task", human_task, TASK_MARKER_RADIUS_M, HUMAN_COLORS[2]
                ),
            )
        )
        lines.extend(
            (
                LinePrimitive(
                    "human_upper_arm",
                    human_shoulder,
                    human_elbow,
                    ARM_LINE_WIDTH_M,
                    HUMAN_COLORS[0],
                ),
                LinePrimitive(
                    "human_forearm",
                    human_elbow,
                    human_wrist,
                    ARM_LINE_WIDTH_M,
                    HUMAN_COLORS[1],
                ),
            )
        )
        axes.append(AxisPrimitive("human_hand_frame", human_wrist, target.hand_rotation))

    if show_target:
        spheres.append(
            SpherePrimitive(
                "target_task", target.task_point, TASK_MARKER_RADIUS_M, TARGET_COLOR
            )
        )
        axes.append(
            AxisPrimitive("target_hand_frame", target.task_point, target.hand_rotation)
        )

    robot_points = None
    actual_pinch = None
    if display_q is not None:
        actual_pinch, actual_rotation = gen3_end_effector_pose(display_q, robot)
        robot_points = geometry.sew_points(display_q)
        spheres.append(
            SpherePrimitive(
                "actual_pinch", actual_pinch, PINCH_MARKER_RADIUS_M, ROBOT_COLOR
            )
        )
        axes.append(
            AxisPrimitive("actual_aligned_pinch_frame", actual_pinch, actual_rotation)
        )
        if show_error and current_q_available:
            lines.append(
                LinePrimitive(
                    "ee_position_error",
                    actual_pinch,
                    target.task_point,
                    OVERLAY_LINE_WIDTH_M,
                    ERROR_COLOR,
                    "arrow",
                )
            )

    if show_sew:
        try:
            human_normal = _sew_normal(
                target.shoulder, target.elbow, target.wrist, stereo
            )
        except StereoSewSingularityError:
            pass
        else:
            _append_sew_geometry(
                "human_sew",
                human_shoulder,
                human_elbow,
                human_wrist,
                human_normal,
                HUMAN_SEW_COLOR,
                spheres,
                lines,
            )
        if robot_points is not None:
            try:
                robot_normal = _sew_normal(
                    robot_points.shoulder,
                    robot_points.elbow,
                    robot_points.wrist,
                    stereo,
                )
            except StereoSewSingularityError:
                pass
            else:
                _append_sew_geometry(
                    "robot_sew",
                    robot_points.shoulder,
                    robot_points.elbow,
                    robot_points.wrist,
                    robot_normal,
                    ROBOT_SEW_COLOR,
                    spheres,
                    lines,
                )

    if show_human:
        lines.extend(
            _trail_lines(
                "human_task_trail",
                human_task_trail,
                HUMAN_TRAIL_COLOR,
                offset=offset,
            )
        )
    lines.extend(
        _trail_lines("robot_pinch_trail", robot_pinch_trail, ROBOT_TRAIL_COLOR)
    )
    return OverlayFrame(tuple(spheres), tuple(lines), tuple(axes))
