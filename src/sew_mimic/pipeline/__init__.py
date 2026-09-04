"""Unified fixed-base Gen3 retargeting/evaluation pipeline."""

from .trajectory import (
    PreparedTrajectory,
    TrajectoryFrame,
    prepare_trajectory,
    sample_frame_indices,
)
from .evaluator import EvaluationRow, evaluate_result
from .benchmark import (
    BenchmarkResult,
    capability_metadata,
    run_benchmark,
    summarize_rows,
)

__all__ = [
    "BenchmarkResult",
    "EvaluationRow",
    "PreparedTrajectory",
    "TrajectoryFrame",
    "capability_metadata",
    "evaluate_result",
    "prepare_trajectory",
    "run_benchmark",
    "sample_frame_indices",
    "summarize_rows",
]
