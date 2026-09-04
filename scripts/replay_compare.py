"""Replay precomputed Phase-7 results with debugging-quality MuJoCo overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.visualization import (  # noqa: E402
    ReplayOptions,
    build_replay_frames,
    format_frame_diagnostics,
    prepare_replay_sequence,
    replay_in_mujoco,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("sew_mimic", "exact_sew", "numerical_oracle", "warp_csew"),
        default="exact_sew",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--show-human", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show-target", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show-sew", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--show-error", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trail-length", type=int, default=0)
    parser.add_argument(
        "--human-display-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="validate and print replay frames without opening a window",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.fps <= 0.0:
        parser.error("--fps must be positive")
    if arguments.stride < 1:
        parser.error("--stride must be at least 1")
    if not arguments.all and arguments.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if arguments.trail_length < 0:
        parser.error("--trail-length must be nonnegative")

    try:
        sequence = prepare_replay_sequence(
            arguments.input,
            arguments.results,
            method=arguments.method,
            start_frame=arguments.start_frame,
            max_frames=None if arguments.all else arguments.max_frames,
            stride=arguments.stride,
        )
        options = ReplayOptions(
            fps=arguments.fps,
            loop=arguments.loop,
            show_human=arguments.show_human,
            show_target=arguments.show_target,
            show_sew=arguments.show_sew,
            show_error=arguments.show_error,
            trail_length=arguments.trail_length,
            human_display_offset_m=tuple(arguments.human_display_offset),
        )
        displays = build_replay_frames(sequence, options)
    except ValueError as error:
        parser.error(str(error))

    if arguments.method == "numerical_oracle":
        print("method role: validation_only")
    print(
        f"precomputed replay: method={arguments.method} frames={len(displays)} "
        f"results={arguments.results.resolve()}"
    )
    if arguments.no_viewer:
        for display in displays:
            print(format_frame_diagnostics(display))
        return 0

    replay_in_mujoco(sequence, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
