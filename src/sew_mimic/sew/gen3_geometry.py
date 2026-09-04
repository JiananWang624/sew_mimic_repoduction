"""Independent POE geometry for the fixed-base MuJoCo Kinova Gen3.

The representation follows the ``H, P, R_7T`` convention used by IK-Geo:
start with ``p=P[:,0], R=I``; for joint ``i`` rotate ``R`` about ``H[:,i]``
and then add ``R @ P[:,i+1]``.  It is deliberately forward-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import CONFIG
from ..geometry import rot
from ..kinematics import Gen3Kinematics
from .stereo import StereoSewReference


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
_GEOMETRY_TOL = 1e-12
_AXIS_PARALLEL_TOL = 1e-12
# The MuJoCo model is exact here; this admits only extraction roundoff.
FAMILY_INTERSECTION_TOL_M = 1e-10


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _vector3(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return vector.copy()


def _unit(vector: ArrayLike, name: str) -> Vector:
    value = _vector3(vector, name)
    norm = float(np.linalg.norm(value))
    if norm <= _GEOMETRY_TOL:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def _closest_line_intersection(
    point_a: Vector, direction_a: Vector, point_b: Vector, direction_b: Vector
) -> tuple[Vector, float]:
    """Return the midpoint of closest axis-line points and their separation."""
    a, b = _unit(direction_a, "axis direction"), _unit(direction_b, "axis direction")
    cross = np.cross(a, b)
    denominator = float(cross @ cross)
    if denominator <= _AXIS_PARALLEL_TOL**2:
        raise ValueError("intersecting joint axes are parallel")
    offset = point_b - point_a
    t_a = float(np.linalg.det(np.stack((offset, b, cross))) / denominator)
    t_b = float(np.linalg.det(np.stack((offset, a, cross))) / denominator)
    closest_a, closest_b = point_a + t_a * a, point_b + t_b * b
    return 0.5 * (closest_a + closest_b), float(np.linalg.norm(closest_a - closest_b))


def _rotation_angle(rotation: Matrix) -> float:
    sine = 0.5 * np.linalg.norm((rotation - rotation.T)[[2, 0, 1], [1, 2, 0]])
    cosine = 0.5 * (float(np.trace(rotation)) - 1.0)
    return float(np.arctan2(sine, cosine))


def _parallel_angle(direction_a: Vector, direction_b: Vector) -> float:
    """Return unsigned line-angle error, treating anti-parallel as parallel."""
    return float(
        np.arctan2(
            np.linalg.norm(np.cross(direction_a, direction_b)),
            abs(float(direction_a @ direction_b)),
        )
    )


@dataclass(frozen=True)
class Gen3SewPoints:
    """Official R-2R-2R-2R SEW points: joint 1, (4,5), and (6,7)."""

    shoulder: Vector
    elbow: Vector
    wrist: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "shoulder", _readonly(_vector3(self.shoulder, "shoulder")))
        object.__setattr__(self, "elbow", _readonly(_vector3(self.elbow, "elbow")))
        object.__setattr__(self, "wrist", _readonly(_vector3(self.wrist, "wrist")))


@dataclass(frozen=True)
class Gen3StructuralResiduals:
    """Residuals proving the three required intersecting axis pairs."""

    pair_intersection_m: Vector
    axis_unit_error: float
    odd_axis_parallel_error_rad: float
    even_axis_parallel_error_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_intersection_m", _readonly(_vector3(self.pair_intersection_m, "pair_intersection_m")))


@dataclass(frozen=True)
class MarginStatistics:
    samples: int
    minimum_rad: float
    p1_rad: float
    p5_rad: float
    median_rad: float
    exact_singular: int
    near_singular: int


@dataclass(frozen=True)
class ReferenceSearchResult:
    candidates: tuple[tuple[str, MarginStatistics], ...]
    reference: StereoSewReference


@dataclass(frozen=True)
class Gen3StereoSewGeometry:
    """Immutable Gen3 parameters in native ``base_link`` coordinates."""

    H: Matrix
    P: Matrix
    R_7T: Matrix
    structural_residuals: Gen3StructuralResiduals

    def __post_init__(self) -> None:
        h = np.asarray(self.H, dtype=float)
        p = np.asarray(self.P, dtype=float)
        tool = np.asarray(self.R_7T, dtype=float)
        if h.shape != (3, 7) or p.shape != (3, 8) or tool.shape != (3, 3):
            raise ValueError("H, P, R_7T shapes must be (3,7), (3,8), (3,3)")
        if not (np.all(np.isfinite(h)) and np.all(np.isfinite(p)) and np.all(np.isfinite(tool))):
            raise ValueError("Gen3 geometry must be finite")
        if not np.allclose(np.linalg.norm(h, axis=0), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("H columns must be unit joint axes")
        if not np.allclose(tool.T @ tool, np.eye(3), atol=1e-12, rtol=0.0):
            raise ValueError("R_7T must be orthonormal")
        if not np.isclose(np.linalg.det(tool), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("R_7T must have determinant +1")
        object.__setattr__(self, "H", _readonly(h))
        object.__setattr__(self, "P", _readonly(p))
        object.__setattr__(self, "R_7T", _readonly(tool))

    @classmethod
    def from_robot(cls, robot: Gen3Kinematics) -> "Gen3StereoSewGeometry":
        """Extract q=0 axes, virtual intersections, and native pinch tool pose."""
        data = mujoco.MjData(robot.model)
        mujoco.mj_forward(robot.model, data)
        base = int(robot.frame_body_ids[0])
        r_world_base = data.xmat[base].reshape(3, 3)
        p_world_base = data.xpos[base]
        anchors = (r_world_base.T @ (data.xanchor[robot.joint_ids] - p_world_base).T).T
        axes = (r_world_base.T @ data.xaxis[robot.joint_ids].T).T
        axes = axes / np.linalg.norm(axes, axis=1)[:, None]
        virtual_23, residual_23 = _closest_line_intersection(
            anchors[1], axes[1], anchors[2], axes[2]
        )
        virtual_45, residual_45 = _closest_line_intersection(
            anchors[3], axes[3], anchors[4], axes[4]
        )
        virtual_67, residual_67 = _closest_line_intersection(
            anchors[5], axes[5], anchors[6], axes[6]
        )
        pair_residuals = np.array([residual_23, residual_45, residual_67])
        if np.any(pair_residuals > FAMILY_INTERSECTION_TOL_M):
            raise ValueError(
                "Gen3 does not satisfy the R-2R-2R-2R intersecting-axis "
                f"tolerance {FAMILY_INTERSECTION_TOL_M:.1e} m: "
                f"residuals={pair_residuals.tolist()}"
            )
        site_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        site_position = r_world_base.T @ (data.site_xpos[site_id] - p_world_base)
        site_rotation = r_world_base.T @ data.site_xmat[site_id].reshape(3, 3)
        points = (
            anchors[0],
            virtual_23,
            virtual_23,
            virtual_45,
            virtual_45,
            virtual_67,
            virtual_67,
            site_position,
        )
        displacements = (
            points[index + 1] - points[index] for index in range(7)
        )
        p = np.column_stack((points[0], *displacements))
        residuals = Gen3StructuralResiduals(
            pair_residuals,
            float(np.max(np.abs(np.linalg.norm(axes, axis=1) - 1.0))),
            float(max(_parallel_angle(axes[0], axes[index]) for index in (2, 4, 6))),
            float(max(_parallel_angle(axes[1], axes[index]) for index in (3, 5))),
        )
        return cls(axes.T, p, site_rotation, residuals)

    @staticmethod
    def _configuration(q: ArrayLike) -> Vector:
        value = np.asarray(q, dtype=float)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("q must be finite with shape (7,)")
        return value

    def joint_axis_lines(self, q: ArrayLike) -> tuple[Matrix, Matrix]:
        """Return the seven current axis points and directions from only H/P."""
        configuration = self._configuration(q)
        p, r = self.P[:, 0].copy(), np.eye(3)
        points, directions = np.empty((7, 3)), np.empty((7, 3))
        for index in range(7):
            points[index], directions[index] = p, r @ self.H[:, index]
            r = r @ rot(self.H[:, index], configuration[index])
            p = p + r @ self.P[:, index + 1]
        return points, directions

    def forward(self, q: ArrayLike) -> NDArray[np.float64]:
        """Return native-base homogeneous transform from ``pinch_site`` to base."""
        configuration = self._configuration(q)
        p, r = self.P[:, 0].copy(), np.eye(3)
        for index in range(7):
            r = r @ rot(self.H[:, index], configuration[index])
            p = p + r @ self.P[:, index + 1]
        transform = np.eye(4)
        transform[:3, :3] = r @ self.R_7T
        transform[:3, 3] = p
        return transform

    def sew_points(self, q: ArrayLike) -> Gen3SewPoints:
        """Return S=joint1, E=axes(4,5), W=axes(6,7) virtual intersections."""
        points, _ = self.joint_axis_lines(q)
        return Gen3SewPoints(points[0], points[3], points[5])


def sample_gen3_configurations(robot: Gen3Kinematics, count: int, seed: int) -> Matrix:
    """Sample unlimited joints in [-pi,pi] and limited joints mechanically."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    lower = np.where(robot.joint_limited, robot.joint_limits[:, 0], -math.pi)
    upper = np.where(robot.joint_limited, robot.joint_limits[:, 1], math.pi)
    return rng.uniform(lower, upper, size=(count, 7))


def angular_margins(directions: ArrayLike, e_t: ArrayLike) -> MarginStatistics:
    values = np.asarray(directions, dtype=float)
    target = _unit(e_t, "e_t")
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("directions must be finite with shape (N,3)")
    if len(values) == 0:
        raise ValueError("directions must contain at least one row")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= _GEOMETRY_TOL):
        raise ValueError("directions must not contain zero rows")
    normalized = values / norms[:, None]
    angles = np.arctan2(
        np.linalg.norm(np.cross(normalized, target), axis=1), normalized @ target
    )
    exact_tolerance = 64 * np.finfo(float).eps
    exact = angles <= exact_tolerance
    near = (~exact) & (angles <= math.radians(5.0))
    return MarginStatistics(
        samples=len(angles),
        minimum_rad=float(np.min(angles)),
        p1_rad=float(np.percentile(angles, 1)),
        p5_rad=float(np.percentile(angles, 5)),
        median_rad=float(np.median(angles)),
        exact_singular=int(np.count_nonzero(exact)),
        near_singular=int(np.count_nonzero(near)),
    )


def select_project_reference(
    robot_directions: ArrayLike, human_directions: ArrayLike
) -> ReferenceSearchResult:
    """Maximize combined minimum margin; ties follow +x,-x,+y,-y,+z,-z."""
    candidates = (
        ("+x", np.array([1.0, 0.0, 0.0])),
        ("-x", np.array([-1.0, 0.0, 0.0])),
        ("+y", np.array([0.0, 1.0, 0.0])),
        ("-y", np.array([0.0, -1.0, 0.0])),
        ("+z", np.array([0.0, 0.0, 1.0])),
        ("-z", np.array([0.0, 0.0, -1.0])),
    )
    robot_values = np.asarray(robot_directions, dtype=float)
    human_values = np.asarray(human_directions, dtype=float)
    # Validate the two evidence sources independently so a missing source
    # cannot silently become a combined robot-only or human-only selection.
    angular_margins(robot_values, candidates[0][1])
    angular_margins(human_values, candidates[0][1])
    combined = np.vstack((robot_values, human_values))
    reports = tuple(
        (name, angular_margins(combined, direction))
        for name, direction in candidates
    )
    selected_index = max(
        range(len(reports)), key=lambda index: reports[index][1].minimum_rad
    )
    e_t = candidates[selected_index][1]
    canonical = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    perpendicular = [axis - float(axis @ e_t) * e_t for axis in canonical]
    selected_r = max(
        range(3), key=lambda index: float(np.linalg.norm(perpendicular[index]))
    )
    e_r = perpendicular[selected_r] / np.linalg.norm(perpendicular[selected_r])
    return ReferenceSearchResult(reports, StereoSewReference(e_t, e_r))


def project_stereo_sew_reference() -> StereoSewReference:
    """Load the validated explicit project pair; StereoSew itself has no default."""
    section = CONFIG.get("stereo_sew")
    if not isinstance(section, dict) or "e_t" not in section or "e_r" not in section:
        raise ValueError("config.yaml must define stereo_sew.e_t and stereo_sew.e_r")
    return StereoSewReference(section["e_t"], section["e_r"])


def rotation_geodesic_error(actual: Matrix, expected: Matrix) -> float:
    return _rotation_angle(actual.T @ expected)
