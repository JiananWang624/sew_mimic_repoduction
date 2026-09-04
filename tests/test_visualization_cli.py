from pathlib import Path

import sew_mimic.pipeline.benchmark as benchmark
import scripts.replay_compare as replay_compare


def test_cli_headless_smoke_uses_precomputed_results_and_never_calls_ik(
    monkeypatch, capsys
):
    root = Path(__file__).resolve().parents[1]

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not invoke Method-2 IK")

    monkeypatch.setattr(benchmark, "solve_exact_sew", forbidden)
    monkeypatch.setattr(replay_compare, "replay_in_mujoco", forbidden)
    assert replay_compare.main(
        [
            "--input",
            str(root / "data" / "test.csv"),
            "--results",
            str(root / "output" / "comparison_frames.csv"),
            "--method",
            "exact_sew",
            "--start-frame",
            "0",
            "--max-frames",
            "2",
            "--stride",
            "1",
            "--fps",
            "60",
            "--show-sew",
            "--show-human",
            "--show-target",
            "--show-error",
            "--trail-length",
            "2",
            "--human-display-offset",
            "0.1",
            "0",
            "0",
            "--no-viewer",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "precomputed replay" in output
    assert "frame=0 method=exact_sew status=SUCCESS_EXACT" in output
    assert "frame=1 method=exact_sew status=SUCCESS_EXACT" in output
