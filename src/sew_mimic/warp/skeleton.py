"""Exact generic WARP corrected-skeleton construction; no robot IK is here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..angles import angular_difference
from ..common.types import HumanArmTarget
from ..geometry import rot, sp3
from ..sew.stereo import StereoSew, StereoSewSingularityError
from .geometry import WarpArmGeometry

Vector = NDArray[np.float64]
PALM_TOLERANCE_M = 1e-12
LENGTH_TOLERANCE_M = 1e-12
SEW_TOLERANCE_RAD = 1e-10
POSITIVE_THETA_TOLERANCE_RAD = 1e-12


class WarpSkeletonStatus(str, Enum):
    SUCCESS_EXACT = "SUCCESS_EXACT"
    INVALID_INPUT = "INVALID_INPUT"
    SEW_SINGULAR = "SEW_SINGULAR"
    NO_EXACT_SP3_ROOT = "NO_EXACT_SP3_ROOT"
    NO_POSITIVE_EXACT_ROOT = "NO_POSITIVE_EXACT_ROOT"
    POSTVALIDATION_FAILURE = "POSTVALIDATION_FAILURE"


def _readonly(value: ArrayLike) -> Vector:
    value = np.asarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


@dataclass(frozen=True)
class WarpSkeletonResult:
    status: WarpSkeletonStatus
    reason: str
    shoulder: Vector | None = None
    elbow: Vector | None = None
    wrist: Vector | None = None
    hand_rotation: NDArray[np.float64] | None = None
    psi_human: float | None = None
    psi_robot: float | None = None
    theta_sew: float | None = None
    palm_error_m: float | None = None
    upper_length_error_m: float | None = None
    forearm_length_error_m: float | None = None
    sew_error_rad: float | None = None
    exact: bool = False
    sp3_exact_root_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.status, WarpSkeletonStatus):
            raise ValueError("status must be a WarpSkeletonStatus")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a nonempty string")
        if not isinstance(self.sp3_exact_root_count, int) or self.sp3_exact_root_count < 0:
            raise ValueError("sp3_exact_root_count must be a nonnegative integer")
        for name in ("shoulder", "elbow", "wrist"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (3,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{name} must be finite shape (3,)")
                object.__setattr__(self, name, _readonly(value))
        if self.hand_rotation is not None:
            rotation = np.asarray(self.hand_rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("hand_rotation must be finite shape (3,3)")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
                raise ValueError("hand_rotation must be orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
                raise ValueError("hand_rotation must have determinant +1")
            object.__setattr__(self, "hand_rotation", _readonly(rotation))
        scalar_names = (
            "psi_human",
            "psi_robot",
            "theta_sew",
            "palm_error_m",
            "upper_length_error_m",
            "forearm_length_error_m",
            "sew_error_rad",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present")
        success = self.status is WarpSkeletonStatus.SUCCESS_EXACT
        if self.exact is not success:
            raise ValueError("exact must be true exactly for SUCCESS_EXACT")
        if success:
            required = (
                self.shoulder,
                self.elbow,
                self.wrist,
                self.hand_rotation,
                self.psi_human,
                self.psi_robot,
                self.theta_sew,
                self.palm_error_m,
                self.upper_length_error_m,
                self.forearm_length_error_m,
                self.sew_error_rad,
            )
            if any(value is None for value in required):
                raise ValueError("SUCCESS_EXACT requires complete skeleton diagnostics")
            if (
                self.palm_error_m > PALM_TOLERANCE_M
                or self.upper_length_error_m > LENGTH_TOLERANCE_M
                or self.forearm_length_error_m > LENGTH_TOLERANCE_M
                or self.sew_error_rad > SEW_TOLERANCE_RAD
            ):
                raise ValueError("SUCCESS_EXACT diagnostics exceed WARP tolerances")


def _failure(status: WarpSkeletonStatus, reason: str, exact_count: int = 0) -> WarpSkeletonResult:
    return WarpSkeletonResult(status=status, reason=reason, sp3_exact_root_count=exact_count)


def construct_warp_skeleton(
    human: HumanArmTarget,
    target_task_robot: ArrayLike,
    geometry: WarpArmGeometry,
    stereo: StereoSew,
) -> WarpSkeletonResult:
    """Construct WARP's positive-root fixed-link skeleton in an explicit frame.

    All human points, ``human.hand_rotation``, ``geometry.shoulder``, and
    ``target_task_robot`` must already share one orientation-aligned frame.
    The function cannot infer or apply a frame transform.
    """
    try:
        if not isinstance(human, HumanArmTarget) or not isinstance(geometry, WarpArmGeometry) or not isinstance(stereo, StereoSew):
            raise ValueError("human, geometry, and stereo have invalid types")
        target = np.asarray(target_task_robot, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_task_robot must be finite shape (3,)")
        hand = human.hand_rotation
        shoulder = geometry.shoulder
        wrist = target - hand @ geometry.wrist_to_task
        psi_human = stereo.forward(human.shoulder, human.elbow, human.wrist)
        inverse = stereo.inverse(shoulder, wrist, psi_human)
    except StereoSewSingularityError as error:
        return _failure(WarpSkeletonStatus.SEW_SINGULAR, str(error))
    except (TypeError, ValueError) as error:
        return _failure(WarpSkeletonStatus.INVALID_INPUT, str(error))
    try:
        shoulder_wrist = wrist - shoulder
        sw_norm = float(np.linalg.norm(shoulder_wrist))
        if sw_norm <= 1e-12:
            return _failure(WarpSkeletonStatus.INVALID_INPUT, "robot shoulder-to-wrist vector is zero")
        result = sp3(geometry.upper_arm_length * shoulder_wrist / sw_norm, shoulder_wrist, inverse.plane_normal, geometry.forearm_length)
    except ValueError as error:
        return _failure(WarpSkeletonStatus.INVALID_INPUT, str(error))
    exact = [(theta, residual) for theta, is_exact, residual in zip(result.angles, result.is_exact, result.residuals, strict=True) if is_exact]
    if not exact:
        return _failure(WarpSkeletonStatus.NO_EXACT_SP3_ROOT, result.message or "SP3 returned no exact root")
    eligible = [(theta, residual) for theta, residual in exact if theta > POSITIVE_THETA_TOLERANCE_RAD]
    if not eligible:
        return _failure(WarpSkeletonStatus.NO_POSITIVE_EXACT_ROOT, "SP3 has no positive exact root", len(exact))
    theta, _ = min(eligible, key=lambda item: (item[0], item[1]))
    elbow = shoulder + rot(inverse.plane_normal, theta) @ (geometry.upper_arm_length * shoulder_wrist / sw_norm)
    try:
        psi_robot = stereo.forward(shoulder, elbow, wrist)
    except StereoSewSingularityError as error:
        return _failure(WarpSkeletonStatus.POSTVALIDATION_FAILURE, str(error), len(exact))
    palm_error = float(np.linalg.norm(wrist + hand @ geometry.wrist_to_task - target))
    upper_error = abs(float(np.linalg.norm(elbow - shoulder)) - geometry.upper_arm_length)
    forearm_error = abs(float(np.linalg.norm(wrist - elbow)) - geometry.forearm_length)
    sew_error = abs(angular_difference(psi_robot, psi_human))
    if palm_error > PALM_TOLERANCE_M or upper_error > LENGTH_TOLERANCE_M or forearm_error > LENGTH_TOLERANCE_M or sew_error > SEW_TOLERANCE_RAD:
        return _failure(WarpSkeletonStatus.POSTVALIDATION_FAILURE, "exact construction failed postvalidation", len(exact))
    return WarpSkeletonResult(WarpSkeletonStatus.SUCCESS_EXACT, "exact positive SP3 root", shoulder, elbow, wrist, hand, psi_human, psi_robot, theta, palm_error, upper_error, forearm_error, sew_error, True, len(exact))
