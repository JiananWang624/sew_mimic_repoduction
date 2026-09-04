"""Shared result and status contracts for retargeting methods."""

from .status import SolverStatus
from .types import SolverDiagnostics, SolverResult

__all__ = ["SolverDiagnostics", "SolverResult", "SolverStatus"]
