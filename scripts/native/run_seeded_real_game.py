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

from spirecomm.ai.real_game_runner import run_seeded_real_game
from spirecomm.seed_helper import canonical_seed_string


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one seeded real-game Spirecomm model rollout against Communication Mod.")
    parser.add_argument("--seed", required=True, help="Numeric seed to replay in the real game.")
    parser.add_argument("--character", default="IRONCLAD")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--combat-model", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observation-version", default=None)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_seeded_real_game(
        seed=canonical_seed_string(int(args.seed), str(args.seed)) or str(args.seed),
        player_class=args.character,
        ascension=args.ascension,
        combat_model=args.combat_model,
        device=args.device,
        observation_version=args.observation_version,
        trajectory_dir=args.trajectory_dir,
    )

    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json or args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
