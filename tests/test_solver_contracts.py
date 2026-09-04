import numpy as np
import pytest

from sew_mimic.common import SolverDiagnostics, SolverResult, SolverStatus


def test_solver_status_values_are_stable_serializable_names() -> None:
    assert [status.value for status in SolverStatus] == [
        "SUCCESS_EXACT",
        "SUCCESS_APPROX",
        "UNREACHABLE",
        "JOINT_LIMIT",
        "SEW_SINGULAR",
        "NO_VALID_BRANCH",
        "INVALID_INPUT",
        "NUMERICAL_FAILURE",
        "LEGACY_FAILURE",
    ]


def test_solver_result_copies_and_serializes_a_valid_configuration() -> None:
    q = np.arange(7, dtype=float)
    result = SolverResult(
        method="test",
        status=SolverStatus.SUCCESS_EXACT,
        q=q,
        diagnostics=SolverDiagnostics(position_error_m=0.0),
    )
    q[0] = 99.0

    np.testing.assert_array_equal(result.q, np.arange(7, dtype=float))
    assert result.to_dict() == {
        "method": "test",
        "status": "SUCCESS_EXACT",
        "q": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "diagnostics": {
            "position_error_m": 0.0,
            "orientation_error_rad": None,
            "sew_error_rad": None,
            "joint_limit_margin_rad": None,
            "solve_time_ms": None,
            "branch_id": None,
            "metadata": {},
        },
        "message": None,
    }


@pytest.mark.parametrize(
    "q",
    [np.zeros(6), np.zeros(8), np.array([0.0] * 6 + [np.nan])],
)
def test_solver_result_rejects_invalid_configurations(q: np.ndarray) -> None:
    with pytest.raises(ValueError, match="q must"):
        SolverResult(
            method="test",
            status=SolverStatus.SUCCESS_EXACT,
            q=q,
        )


def test_solver_diagnostics_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="orientation_error_rad must be finite"):
        SolverDiagnostics(orientation_error_rad=np.inf)


def test_solver_diagnostics_metadata_defaults_are_independent() -> None:
    first = SolverDiagnostics()
    second = SolverDiagnostics()

    first.metadata["source"] = "first"

    assert second.metadata == {}


@pytest.mark.parametrize(
    ("status", "q"),
    [
        (SolverStatus.SUCCESS_EXACT, None),
        (SolverStatus.SUCCESS_APPROX, None),
        (SolverStatus.INVALID_INPUT, np.zeros(7)),
        (SolverStatus.UNREACHABLE, np.zeros(7)),
    ],
)
def test_solver_result_rejects_status_configuration_mismatches(
    status: SolverStatus, q: np.ndarray | None
) -> None:
    with pytest.raises(ValueError, match="solver results"):
        SolverResult(method="test", status=status, q=q)
