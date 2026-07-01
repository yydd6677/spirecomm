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

from spirecomm.ai.real_game_first_validation import (
    DEFAULT_CURATED_REPLAY_SEEDS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REPLAY_REPORT_DIR,
    DEFAULT_TRACE_DIR,
    build_real_game_first_report,
    build_seed_corpus,
    render_real_game_first_summary,
)


def _parse_seed_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    values = []
    for part in raw.split(","):
        token = part.strip()
        if token:
            values.append(int(token))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-game-first validation workflow (defaults to native backend v3).")
    parser.add_argument("--mode", choices=["native", "v2", "real", "replay", "all"], default="all")
    parser.add_argument("--native-backend", choices=["v2", "v3"], default="v3", help="Native backend for the validation corpus; defaults to v3.")
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=63)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--curated-replay-seeds", default=None, help="Comma-separated replay seed list.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate"], default="legacy-slot")
    parser.add_argument("--v3-combat-model", type=Path, default=Path("/home/yydd/spirecomm/models/v3_combat_scorer.pt"))
    parser.add_argument("--observation-version", default=None)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional explicit step cap. By default traces/replays run until the game reaches a terminal state.",
    )
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--replay-report-dir", type=Path, default=DEFAULT_REPLAY_REPORT_DIR)
    parser.add_argument("--launch-align", action="store_true", help="Launch the align instance for replay validation.")
    parser.add_argument("--keep-going", action="store_true", help="Keep replaying after a divergence when using --launch-align.")
    parser.add_argument(
        "--pause-on-divergence",
        action="store_true",
        help="Keep the strict replay/game process alive on divergence so v3 can be fixed and resumed.",
    )
    parser.add_argument("--no-xvfb", action="store_true")
    parser.add_argument(
        "--replay-mode",
        choices=["strict", "bridge"],
        default="strict",
        help="Replay validation mode; defaults to strict, with bridge retained for smoke/fallback workflows.",
    )
    parser.add_argument(
        "--trace-policy",
        choices=["model-required", "legacy-fallback"],
        default="model-required",
        help="Trace decision policy; model-required fails instead of silently falling back.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seed_corpus = build_seed_corpus(
        seed_file=args.seeds_file,
        count=args.count,
        random_seed=args.random_seed,
        sequential=args.sequential,
        start_seed=args.start_seed,
    )
    curated_replay_seeds = _parse_seed_list(args.curated_replay_seeds) or list(DEFAULT_CURATED_REPLAY_SEEDS)
    mode = args.mode
    report = build_real_game_first_report(
        seed_corpus=seed_corpus,
        curated_replay_seeds=curated_replay_seeds,
        device=args.device,
        combat_device=args.combat_device,
        combat_selector=args.combat_selector,
        v3_combat_model=args.v3_combat_model,
        observation_version=args.observation_version,
        ascension=args.ascension,
        max_steps=args.max_steps,
        max_floor=args.max_floor,
        native_backend=args.native_backend,
        run_native=mode in {"native", "v2", "all"},
        run_real=mode in {"real", "all"},
        run_replay=mode in {"replay", "all"},
        launch_align=args.launch_align,
        keep_going=args.keep_going,
        use_xvfb=not args.no_xvfb,
        trajectory_dir=args.trajectory_dir,
        trace_dir=args.trace_dir,
        replay_report_dir=args.replay_report_dir,
        replay_mode=args.replay_mode,
        pause_on_divergence=args.pause_on_divergence,
        trace_policy=args.trace_policy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".txt").write_text(render_real_game_first_summary(report), encoding="utf-8")
    if args.json or args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"output": str(args.output), "ok": report["real_game_blocking"]["ok"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
