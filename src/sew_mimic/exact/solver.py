"""Method-2 exact pinch-pose plus Stereo-SEW solver."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..common import ExactSewTarget, HumanArmTarget, SolverDiagnostics, SolverResult, SolverStatus
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, StereoSewSingularityError
from .branch_selection import BranchPolicy, select_exact_sew_branch
from .stereo_backend import R2R2R2RSearchConfig, enumerate_exact_sew_candidates


METHOD_NAME = "exact_sew"
CONSTRAINT_SET = "pinch_pose_plus_stereo_sew"


def human_arm_to_exact_sew_target(human_target: HumanArmTarget, stereo: StereoSew) -> ExactSewTarget:
    """Use the established task point, hand frame, and human Stereo-SEW angle."""
    if not isinstance(human_target, HumanArmTarget):
        raise ValueError("human_target must be a HumanArmTarget")
    return ExactSewTarget(
        human_target.task_point,
        human_target.hand_rotation,
        stereo.forward(human_target.shoulder, human_target.elbow, human_target.wrist),
    )


def _failure(status: SolverStatus, started: float, message: str, metadata: dict[str, Any]) -> SolverResult:
    return SolverResult(
        method=METHOD_NAME,
        status=status,
        q=None,
        diagnostics=SolverDiagnostics(
            solve_time_ms=1000.0 * (time.perf_counter() - started), metadata=metadata
        ),
        message=message,
    )


def solve_exact_sew(
    target: ExactSewTarget,
    robot: Gen3Kinematics,
    geometry: Gen3StereoSewGeometry,
    stereo: StereoSew,
    *,
    branch_policy: BranchPolicy = "canonical",
    q_previous: ArrayLike | None = None,
    search_config: R2R2R2RSearchConfig = R2R2R2RSearchConfig(),
) -> SolverResult:
    """Enumerate once, then choose one strict Method-2 branch only."""
    started = time.perf_counter()
    metadata: dict[str, Any] = {"constraint_set": CONSTRAINT_SET, "branch_policy": branch_policy}
    try:
        if not isinstance(target, ExactSewTarget):
            raise ValueError("target must be an ExactSewTarget")
        # Validate selection inputs before costly history-independent enumeration.
        if branch_policy not in ("canonical", "continuous"):
            raise ValueError("branch_policy must be 'canonical' or 'continuous'")
        if q_previous is not None:
            previous = np.asarray(q_previous, dtype=float)
            if previous.shape != (7,) or not np.all(np.isfinite(previous)):
                raise ValueError("q_previous must be finite with shape (7,)")
    except StereoSewSingularityError as error:
        return _failure(SolverStatus.SEW_SINGULAR, started, str(error), metadata)
    except (ValueError, TypeError) as error:
        return _failure(SolverStatus.INVALID_INPUT, started, str(error), metadata)

    try:
        candidate_set = enumerate_exact_sew_candidates(
            target, robot, geometry, stereo, search_config=search_config
        )
        outcome = select_exact_sew_branch(
            candidate_set, branch_policy=branch_policy, q_previous=q_previous
        )
    except StereoSewSingularityError as error:
        return _failure(SolverStatus.SEW_SINGULAR, started, str(error), metadata)
    except Exception as error:  # Backend exceptions cannot be safely reclassified.
        metadata["exception_type"] = type(error).__name__
        return _failure(SolverStatus.NUMERICAL_FAILURE, started, str(error), metadata)

    metadata.update(
        candidate_count=len(candidate_set.candidates),
        joint_limit_valid_candidate_count=sum(c.joint_limit_valid for c in candidate_set.candidates),
        search_mode=search_config.mode,
        backend_elapsed_ms=candidate_set.elapsed_ms,
    )
    if outcome.status is not SolverStatus.SUCCESS_EXACT:
        return _failure(outcome.status, started, "no selectable exact branch", metadata)
    assert outcome.candidate is not None and outcome.branch_id is not None
    candidate = outcome.candidate
    metadata.update(
        backend_branch_identity=outcome.branch_id,
        wrist_search_angle=candidate.wrist_search_angle,
        search_branch=candidate.search_branch,
        backend_metadata=dict(candidate.metadata),
    )
    return SolverResult(
        method=METHOD_NAME,
        status=SolverStatus.SUCCESS_EXACT,
        q=candidate.q,
        diagnostics=SolverDiagnostics(
            position_error_m=candidate.position_error_m,
            orientation_error_rad=candidate.orientation_error_rad,
            sew_error_rad=candidate.sew_error_rad,
            joint_limit_margin_rad=candidate.joint_limit_margin_rad,
            solve_time_ms=1000.0 * (time.perf_counter() - started),
            branch_id=outcome.branch_id,
            metadata=metadata,
        ),
    )


def retarget_exact_sew(
    human_target: HumanArmTarget,
    robot: Gen3Kinematics,
    geometry: Gen3StereoSewGeometry,
    stereo: StereoSew,
    *,
    branch_policy: BranchPolicy = "canonical",
    q_previous: ArrayLike | None = None,
    search_config: R2R2R2RSearchConfig = R2R2R2RSearchConfig(),
) -> SolverResult:
    """Human-facing Method-2 wrapper sharing the exact solver path."""
    started = time.perf_counter()
    try:
        target = human_arm_to_exact_sew_target(human_target, stereo)
    except StereoSewSingularityError as error:
        return _failure(SolverStatus.SEW_SINGULAR, started, str(error), {"constraint_set": CONSTRAINT_SET})
    except ValueError as error:
        return _failure(SolverStatus.INVALID_INPUT, started, str(error), {"constraint_set": CONSTRAINT_SET})
    return solve_exact_sew(
        target,
        robot,
        geometry,
        stereo,
        branch_policy=branch_policy,
        q_previous=q_previous,
        search_config=search_config,
    )
