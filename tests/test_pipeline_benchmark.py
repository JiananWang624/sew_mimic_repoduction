import math

from sew_mimic.common import SolverStatus
from sew_mimic.pipeline.benchmark import summarize_rows
from sew_mimic.pipeline.evaluator import EvaluationRow


def _row(
    frame: int,
    method: str,
    status: SolverStatus,
    *,
    q: tuple[float, ...] = (0.0,) * 7,
    branch: str | None = None,
) -> EvaluationRow:
    success = status in (SolverStatus.SUCCESS_EXACT, SolverStatus.SUCCESS_APPROX)
    nan = float("nan")
    return EvaluationRow(
        frame,
        method,
        status.value,
        q if success else (nan,) * 7,
        float(frame + 1) if success else nan,
        2.0 if success else nan,
        3.0 if success else nan,
        True if success else None,
        4.0 if success else nan,
        branch,
        5.0,
    )


def test_summary_retains_failures_and_reports_all_required_statistics():
    rows = [
        _row(0, "exact_sew", SolverStatus.SUCCESS_EXACT, branch="a"),
        _row(1, "exact_sew", SolverStatus.UNREACHABLE),
    ]
    summary = summarize_rows(rows)["methods"]["exact_sew"]
    assert summary["frames_requested"] == 2
    assert summary["success_fraction"] == 0.5
    assert summary["status_counts"] == {
        "SUCCESS_EXACT": 1,
        "UNREACHABLE": 1,
    }
    assert summary["ee_position_error_mm"]["p99"] == 1.0
    assert summary["joint_limits"]["margin_deg"]["minimum"] == 4.0
    assert summary["solve_time_ms"]["p95"] == 5.0
    assert summary["error_statistics_scope"] == "successful_frames_only"


def test_wrapped_joint_jump_and_branch_switch_skip_failed_rows():
    rows = [
        _row(
            0,
            "exact_sew",
            SolverStatus.SUCCESS_EXACT,
            q=(math.pi - 0.1,) + (0.0,) * 6,
            branch="a",
        ),
        _row(1, "exact_sew", SolverStatus.NUMERICAL_FAILURE),
        _row(
            2,
            "exact_sew",
            SolverStatus.SUCCESS_EXACT,
            q=(-math.pi + 0.1,) + (0.0,) * 6,
            branch="b",
        ),
    ]
    continuity = summarize_rows(rows)["methods"]["exact_sew"][
        "trajectory_continuity"
    ]
    assert math.isclose(
        continuity["wrapped_joint_jump_rad"]["median"], 0.2, abs_tol=1e-12
    )
    assert continuity["branch_switch_count"] == 1


def test_oracle_agreement_has_all_categories_and_flags_correctness_misses():
    rows = [
        _row(0, "exact_sew", SolverStatus.SUCCESS_EXACT),
        _row(0, "numerical_oracle", SolverStatus.SUCCESS_EXACT),
        _row(1, "exact_sew", SolverStatus.SUCCESS_EXACT),
        _row(1, "numerical_oracle", SolverStatus.SUCCESS_APPROX),
        _row(2, "exact_sew", SolverStatus.NO_VALID_BRANCH),
        _row(2, "numerical_oracle", SolverStatus.SUCCESS_EXACT),
        _row(3, "exact_sew", SolverStatus.UNREACHABLE),
        _row(3, "numerical_oracle", SolverStatus.SUCCESS_APPROX),
    ]
    agreement = summarize_rows(rows)["oracle_agreement"]["exact_sew"]
    assert agreement["categories"] == {
        "both_exact": 1,
        "method2_exact_oracle_nonexact": 1,
        "oracle_exact_method2_nonexact": 1,
        "both_nonexact": 1,
    }
    assert agreement["correctness_discrepancy_frames"] == [2]
