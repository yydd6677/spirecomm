#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

from spirecomm.ai.lightspeed_combat_model import SerializedCombatSelector


def _native_env_cls(backend: str):
    if backend == "v2":
        from spirecomm.native_sim_v2 import NativeCombatEnv
        return NativeCombatEnv
    if backend == "v3":
        from spirecomm.native_sim_v3 import NativeCombatEnv
        return NativeCombatEnv
    from spirecomm.native_sim import NativeCombatEnv
    return NativeCombatEnv


def choose_fallback(actions):
    for action in actions:
        if action.get("kind") == "card":
            return action
    return actions[0]


def _build_combat_selector(model_path: Path | None, device: str):
    if not model_path:
        return None
    try:
        return SerializedCombatSelector(checkpoint_path=model_path, device=device)
    except ModuleNotFoundError as exc:
        if "torch is required" not in str(exc):
            raise
        return None


def _serialized_combat_state(env) -> dict:
    if hasattr(env, "serialize"):
        return env.serialize()
    if hasattr(env, "to_spirecomm_state"):
        return env.to_spirecomm_state()
    state_attr = getattr(env, "state", None)
    if callable(state_attr):
        return state_attr()
    return state_attr


def _combat_monsters(env):
    monsters = getattr(env, "monsters", None)
    if monsters is not None:
        return monsters
    state_attr = getattr(env, "state", None)
    if state_attr is not None and getattr(state_attr, "monsters", None) is not None:
        return state_attr.monsters
    engine = getattr(env, "engine", None)
    if engine is not None:
        if getattr(engine, "monsters", None) is not None:
            return engine.monsters
        engine_state = getattr(engine, "state", None)
        if engine_state is not None and getattr(engine_state, "monsters", None) is not None:
            return engine_state.monsters
    return []


def run_one(seed: int, ascension: int, max_steps: int, model_path: Path | None, device: str, verbose: bool, backend: str) -> dict:
    env = _native_env_cls(backend)(seed=seed, ascension_level=ascension)
    selector = _build_combat_selector(model_path, device)

    steps = 0
    while env.outcome == "UNDECIDED" and steps < max_steps:
        state = _serialized_combat_state(env)
        actions = env.legal_actions()
        chosen = None
        scores = []
        if selector and selector.available:
            chosen, scores = selector.choose(state, actions)
        if chosen is None:
            chosen = choose_fallback(actions)
        if verbose:
            print(
                f"step={steps:03d} hp={env.player.current_hp}/{env.player.max_hp} "
                f"energy={env.player.energy} action={chosen.get('kind')}:{chosen.get('name')} "
                f"scores={[round(float(value), 3) for value in scores[:8]]}"
            )
        env.step(chosen)
        steps += 1

    return {
        "seed": seed,
        "outcome": env.outcome,
        "hp": env.player.current_hp,
        "steps": steps,
        "monster_hp": [monster.current_hp for monster in _combat_monsters(env)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the spirecomm-native simulator vertical slice (defaults to backend v3).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--model", type=Path, default=Path("/home/yydd/spirecomm/models/combat.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["v1", "v2", "v3"], default="v3", help="Native backend to use; defaults to v3, with v2 kept for comparison.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = []
    for offset in range(args.count):
        result = run_one(
            seed=args.seed + offset,
            ascension=args.ascension,
            max_steps=args.max_steps,
            model_path=args.model,
            device=args.device,
            verbose=args.verbose,
            backend=args.backend,
        )
        results.append(result)
        print(
            f"result seed={result['seed']} outcome={result['outcome']} "
            f"hp={result['hp']} steps={result['steps']} monster_hp={result['monster_hp']}",
            flush=True,
        )
    wins = sum(1 for result in results if result["outcome"] == "PLAYER_VICTORY")
    print(f"summary count={len(results)} wins={wins}")


if __name__ == "__main__":
    main()
