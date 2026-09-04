"""Typed result structures shared by retargeting methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .status import SolverStatus


Vector = NDArray[np.float64]


def _optional_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite when present")
    return number


@dataclass
class SolverDiagnostics:
    """Common diagnostics, with method-specific values kept in ``metadata``."""

    position_error_m: float | None = None
    orientation_error_rad: float | None = None
    sew_error_rad: float | None = None
    joint_limit_margin_rad: float | None = None
    solve_time_ms: float | None = None
    branch_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "position_error_m",
            "orientation_error_rad",
            "sew_error_rad",
            "joint_limit_margin_rad",
            "solve_time_ms",
        ):
            setattr(self, name, _optional_finite(getattr(self, name), name))
        if self.branch_id is not None and not isinstance(self.branch_id, str):
            raise ValueError("branch_id must be a string when present")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary while preserving method metadata values."""
        return asdict(self)


@dataclass
class SolverResult:
    """A joint solution and its explicit solver outcome."""

    method: str
    status: SolverStatus
    q: Vector | None
    diagnostics: SolverDiagnostics = field(default_factory=SolverDiagnostics)
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a nonempty string")
        if not isinstance(self.status, SolverStatus):
            raise ValueError("status must be a SolverStatus")
        if not isinstance(self.diagnostics, SolverDiagnostics):
            raise ValueError("diagnostics must be SolverDiagnostics")
        if self.message is not None and not isinstance(self.message, str):
            raise ValueError("message must be a string when present")
        if self.q is not None:
            configuration = np.asarray(self.q, dtype=float)
            if configuration.shape != (7,):
                raise ValueError(f"q must have shape (7,), got {configuration.shape}")
            if not np.all(np.isfinite(configuration)):
                raise ValueError("q must contain only finite values")
            self.q = configuration.copy()
        success = self.status in (
            SolverStatus.SUCCESS_EXACT,
            SolverStatus.SUCCESS_APPROX,
        )
        if success and self.q is None:
            raise ValueError("successful solver results must contain q")
        if not success and self.q is not None:
            raise ValueError("failed solver results must not contain q")

    def to_dict(self) -> dict[str, Any]:
        """Normalize the shared fields while preserving metadata values."""
        return {
            "method": self.method,
            "status": self.status.value,
            "q": None if self.q is None else self.q.tolist(),
            "diagnostics": self.diagnostics.to_dict(),
            "message": self.message,
        }
