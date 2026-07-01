#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

from spirecomm.ai.recorded_run_replay import render_replay_report_summary, replay_recorded_run
from spirecomm.ai.strict_recorded_run_replay import (
    render_strict_replay_report_summary,
    replay_recorded_run_strict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a native-recorded run against the real Communication Mod environment."
    )
    parser.add_argument("--trace", type=Path, required=True, help="Path to a recorded run trace JSON.")
    parser.add_argument(
        "--mode",
        choices=["strict", "bridge"],
        default="strict",
        help="Replay mode; defaults to strict, with bridge retained for smoke/fallback workflows.",
    )
    parser.add_argument("--character", default=None, help="Override character class, e.g. IRONCLAD.")
    parser.add_argument("--no-compare", action="store_true", help="Do not compare live state against trace snapshots.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after mismatches instead of stopping at first divergence.")
    parser.add_argument("--max-steps", type=int, default=None, help="Replay at most this many recorded steps.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the replay report JSON.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report to stdout.")
    args = parser.parse_args()

    if args.mode == "bridge":
        report = replay_recorded_run(
            trace_path=args.trace,
            character=args.character,
            compare_state=not args.no_compare,
            stop_on_mismatch=not args.keep_going,
            max_steps=args.max_steps,
        )
        summary_renderer = render_replay_report_summary
    else:
        report = replay_recorded_run_strict(
            trace_path=args.trace,
            character=args.character,
            max_steps=args.max_steps,
        )
        summary_renderer = render_strict_replay_report_summary

    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = args.output.with_suffix(".txt")
        summary_path.write_text(summary_renderer(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(
        json.dumps(
            {
                "trace": str(args.trace),
                "success": report["success"],
                "steps_replayed": report["steps_replayed"],
                "steps_total": report["steps_total"],
                "first_failure_step": report["first_failure_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
