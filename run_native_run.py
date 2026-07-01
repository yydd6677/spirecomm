#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from spirecomm.ai.runtime_decision import (
    build_runtime_selectors,
    choose_modeled_action,
)


def _native_env_cls(backend: str):
    if backend == "v2":
        from spirecomm.native_sim_v2 import NativeRunEnv
        return NativeRunEnv
    if backend == "v3":
        from spirecomm.native_sim_v3 import NativeRunEnv
        return NativeRunEnv
    from spirecomm.native_sim import NativeRunEnv
    return NativeRunEnv

def run_one(seed: int, ascension: int, max_floor: int, max_steps: int, selectors: dict, verbose: bool, backend: str) -> dict:
    env = _native_env_cls(backend)(seed=seed, ascension_level=ascension, enable_neow=bool(selectors.get("enable_neow")))
    steps = 0
    while env.phase not in {"GAME_OVER", "COMPLETE"} and env.floor <= max_floor and steps < max_steps:
        action, scores, source = choose_modeled_action(env, selectors)

        if verbose:
            print(
                f"seed={seed} floor={env.floor:02d} phase={env.phase} hp={env.player.current_hp}/{env.player.max_hp} "
                f"source={source} action={action.get('kind')}:{action.get('name')} scores={[round(float(v), 3) for v in scores[:6]]}",
                flush=True,
            )
        env.step(action)
        steps += 1

    return {
        "seed": seed,
        "phase": env.phase,
        "floor": env.floor,
        "hp": env.player.current_hp,
        "deck_size": len(env.deck),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the spirecomm-native run simulator (defaults to backend v3).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--combat-model", type=Path, default=Path("/home/yydd/spirecomm/models/combat.pt"))
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate"], default="legacy-slot")
    parser.add_argument("--v3-combat-model", type=Path, default=Path("/home/yydd/spirecomm/models/v3_combat_scorer.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("/home/yydd/spirecomm/models/card_reward.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--backend", choices=["v1", "v2", "v3"], default="v3", help="Native backend to use; defaults to v3, with v2 kept for comparison.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--neow", action="store_true", help="Start runs with the approximate native Neow reward phase.")
    args = parser.parse_args()

    selectors = build_runtime_selectors(
        repo_root=Path("/home/yydd/spirecomm"),
        device=args.device,
        combat_device=args.combat_device,
        combat_model=args.combat_model,
        combat_selector=args.combat_selector,
        v3_combat_model=args.v3_combat_model,
        card_reward_model=args.card_reward_model,
    )
    selectors["enable_neow"] = args.neow

    results = []
    for offset in range(args.count):
        result = run_one(
            seed=args.seed + offset,
            ascension=args.ascension,
            max_floor=args.max_floor,
            max_steps=args.max_steps,
            selectors=selectors,
            verbose=args.verbose,
            backend=args.backend,
        )
        results.append(result)
        print(
            f"result seed={result['seed']} phase={result['phase']} floor={result['floor']} "
            f"hp={result['hp']} deck={result['deck_size']} steps={result['steps']}",
            flush=True,
        )
    print(f"summary count={len(results)} avg_floor={sum(r['floor'] for r in results) / max(1, len(results)):.2f}")


if __name__ == "__main__":
    main()
