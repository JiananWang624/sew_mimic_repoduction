from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.measure_baseline as measure_baseline
from sew_mimic.common import SolverDiagnostics, SolverResult, SolverStatus
from sew_mimic.csv_adapter import REQUIRED_COLUMNS


def test_baseline_summary_includes_failures_and_signed_margin_violations() -> None:
    table = pd.DataFrame(
        {
            "solver_status": [
                "SUCCESS_EXACT",
                "SUCCESS_APPROX",
                "INVALID_INPUT",
            ],
            "ee_position_error_mm": [1.0, 3.0, np.nan],
            "ee_orientation_error_deg": [2.0, 4.0, np.nan],
            "joint_limit_margin_deg": [10.0, -1.0, np.nan],
            "joint_limit_valid": [True, False, None],
        }
    )

    summary = measure_baseline.summarize_baseline(table)

    assert summary["valid_frame_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["ee_position_mean_mm"] == 2.0
    assert summary["ee_position_median_mm"] == 2.0
    assert summary["ee_orientation_mean_deg"] == 3.0
    assert summary["joint_limit_violation_count"] == 1
    assert summary["joint_limit_violation_fraction"] == 0.5
    assert summary["minimum_joint_limit_margin_deg"] == -1.0


def test_baseline_measurement_records_solver_failure_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {column: 0.0 for column in REQUIRED_COLUMNS}
    row.update(
        {
            "Elbow_X": 100.0,
            "Wrist_X": 200.0,
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "metrics.csv"
    pd.DataFrame([row]).to_csv(input_path, index=False)
    monkeypatch.setattr(
        measure_baseline,
        "solve_legacy_sew_mimic",
        lambda *args: SolverResult(
            method="legacy_sew_mimic",
            status=SolverStatus.LEGACY_FAILURE,
            q=None,
            diagnostics=SolverDiagnostics(
                metadata={"constraint_set": "legacy_sew_direction_orientation"}
            ),
            message="known failure",
        ),
    )

    table = measure_baseline.measure_baseline(input_path, output_path)

    assert output_path.is_file()
    assert tuple(table.columns) == measure_baseline.OUTPUT_COLUMNS
    assert len(table) == 1
    assert table.loc[0, "solver_status"] == "LEGACY_FAILURE"
    assert table.loc[0, "message"] == "known failure"
    assert table.loc[0, [f"q{index}" for index in range(1, 8)]].isna().all()
