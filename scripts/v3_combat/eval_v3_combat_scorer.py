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
from typing import Any

from scripts.v3_combat.generate_v3_combat_teacher_dataset import curated_envs
from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.ai.v3_combat_selector import V3CandidateCombatSelector
from spirecomm.ai.v3_combat_teacher import TeacherConfig, best_teacher_action, label_env
from spirecomm.native_sim_v3 import NativeRunEnv


SETUP_CARD_IDS = {"Rage", "Inflame", "Flex", "Corruption", "Offering"}


def _action_label(action: dict[str, Any] | None) -> str:
    if not action:
        return "NONE"
    if action.get("kind") == "end":
        return "END_TURN"
    bits = [str(action.get("card_id") or action.get("name") or action.get("kind"))]
    if action.get("target_index") is not None:
        bits.append(f"target={action.get('target_index')}")
    return " ".join(bits)


def eval_curated(model_path: Path, *, device: str) -> dict[str, Any]:
    selector = V3CandidateCombatSelector(model_path, device=device)
    if not selector.available:
        raise SystemExit(f"v3 combat scorer unavailable: {model_path}")
    cases = []
    setup_hits = 0
    setup_total = 0
    for case_id, env in curated_envs():
        labeled = label_env(env, root_id=case_id, source="curated_eval", config=TeacherConfig(beam_width=8, node_budget_per_root=128))
        if labeled is None:
            continue
        teacher_action = best_teacher_action(labeled)
        model_action, scores = selector.choose_env(env)
        teacher_setup = str((teacher_action or {}).get("card_id") or "") in SETUP_CARD_IDS
        if teacher_setup:
            setup_total += 1
            setup_hits += int(str((model_action or {}).get("card_id") or "") == str((teacher_action or {}).get("card_id") or ""))
        cases.append(
            {
                "case": case_id,
                "teacher": _action_label(teacher_action),
                "model": _action_label(model_action),
                "scores": scores,
                "teacher_setup": teacher_setup,
            }
        )
    return {
        "cases": cases,
        "setup_before_attack_hits": setup_hits,
        "setup_before_attack_total": setup_total,
        "setup_before_attack_rate": setup_hits / max(1, setup_total),
    }


def eval_rollout(model_path: Path, *, seeds: list[int], device: str, max_steps: int) -> dict[str, Any]:
    selectors = build_runtime_selectors(
        repo_root=Path("/home/yydd/spirecomm"),
        device=device,
        combat_selector="v3-candidate",
        v3_combat_model=model_path,
    )
    results = []
    for seed in seeds:
        env = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)
        steps = 0
        sources: dict[str, int] = {}
        while env.phase not in {"GAME_OVER", "COMPLETE", "VICTORY"} and steps < max_steps:
            action, _, source = choose_model_required_action(env, selectors)
            sources[source] = sources.get(source, 0) + 1
            env.step(action)
            steps += 1
        results.append(
            {
                "seed": seed,
                "phase": env.phase,
                "floor": int(env.floor),
                "hp": int(env.player.current_hp),
                "steps": steps,
                "sources": sources,
            }
        )
    return {
        "results": results,
        "mean_floor": sum(result["floor"] for result in results) / max(1, len(results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the v3 combat candidate scorer.")
    parser.add_argument("--model", type=Path, default=Path("models/v3_combat_scorer.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-seeds", default="")
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    report = {"curated": eval_curated(args.model, device=args.device)}
    if args.rollout_seeds:
        seeds = [int(token.strip()) for token in args.rollout_seeds.split(",") if token.strip()]
        report["rollout"] = eval_rollout(args.model, seeds=seeds, device=args.device, max_steps=args.max_steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
