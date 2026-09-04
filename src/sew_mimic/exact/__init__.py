"""Exact-SEW production APIs and the lazily loaded Method-3 oracle."""

from typing import TYPE_CHECKING, Any

from .residuals import ExactSewResiduals, robot_exact_sew_residuals, so3_log
from .stereo_backend import (
    ExactSewCandidate,
    ExactSewCandidateSet,
    NativeStereoSewTarget,
    R2R2R2RSearchConfig,
    enumerate_exact_sew_candidates,
    to_native_stereo_sew_target,
)
from .branch_selection import (
    BranchSelectionOutcome,
    candidate_branch_id,
    normalized_authoritative_residual,
    select_exact_sew_branch,
)
from .solver import human_arm_to_exact_sew_target, retarget_exact_sew, solve_exact_sew

if TYPE_CHECKING:
    from .numerical_oracle import NumericalExactSewOracle, NumericalOracleConfig


def __getattr__(name: str) -> Any:
    """Load the Method-3 oracle only when its public names are requested."""
    if name in ("NumericalExactSewOracle", "NumericalOracleConfig"):
        from .numerical_oracle import NumericalExactSewOracle, NumericalOracleConfig

        globals().update(
            NumericalExactSewOracle=NumericalExactSewOracle,
            NumericalOracleConfig=NumericalOracleConfig,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExactSewResiduals",
    "ExactSewCandidate",
    "ExactSewCandidateSet",
    "NumericalExactSewOracle",
    "NumericalOracleConfig",
    "NativeStereoSewTarget",
    "R2R2R2RSearchConfig",
    "enumerate_exact_sew_candidates",
    "robot_exact_sew_residuals",
    "so3_log",
    "to_native_stereo_sew_target",
    "BranchSelectionOutcome",
    "candidate_branch_id",
    "human_arm_to_exact_sew_target",
    "normalized_authoritative_residual",
    "retarget_exact_sew",
    "select_exact_sew_branch",
    "solve_exact_sew",
]
