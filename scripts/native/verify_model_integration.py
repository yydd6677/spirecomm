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
import sys
from pathlib import Path

from spirecomm.ai.integration_verifier import (
    DEFAULT_LIGHTSPEED_BUILD,
    compare_seed_with_model,
    default_runtime_model_paths,
    validate_combat_model_loads,
    validate_runtime_integration,
)
from spirecomm.ai.learned_policy import CheckpointCombatPolicy
from spirecomm.ai.runtime_decision import build_runtime_selectors
from spirecomm.native_sim_v3 import NativeRunEnv


def _dependency_cli_message(exc: BaseException) -> str | None:
    text = str(exc)
    if "torch is required for model/training operations in spirecomm.ai" in text:
        return (
            "torch is required for scripts/native/verify_model_integration.py. "
            "Install torch to run model-loading or parity checks, or use native-only "
            "entrypoints such as scripts/native/run_native_run.py, scripts/native/run_native_sim.py, or "
            "scripts/native/export_model_run_checklist.py."
        )
    if "CheckpointCombatPolicy requires torch" in text:
        return (
            "torch is required for scripts/native/verify_model_integration.py. "
            "Install torch (for example via the spirecomm-rl environment) to run "
            "model-loading or parity checks, or use native-only entrypoints such as "
            "scripts/native/run_native_run.py, scripts/native/run_native_sim.py, or scripts/native/export_model_run_checklist.py."
        )
    if "slaythespire is required for lightspeed-backed model integration checks" in text:
        return text
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and regress spirecomm model integration against the current default native backend.")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/yydd/spirecomm"))
    parser.add_argument("--lightspeed-build", type=Path, default=DEFAULT_LIGHTSPEED_BUILD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--observation-version", default=None)
    parser.add_argument("--backend-pair", default="lightspeed,v3")
    parser.add_argument("--coverage-count", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=["all", "load", "parity"],
        default="all",
    )
    parser.add_argument(
        "--model-type",
        choices=[
            "all",
            "combat",
            "combat_bc",
            "combat_pref",
        ],
        default="all",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--target-phase", default=None)
    parser.add_argument("--target-floor", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.mode == "all":
            summary = validate_runtime_integration(
                repo_root=args.repo_root,
                device=args.device,
                combat_device=args.combat_device,
                observation_version=args.observation_version,
                backend_pair=args.backend_pair,
                lightspeed_build=args.lightspeed_build,
                include_alternate_combat=True,
                coverage_count=args.coverage_count,
            )
        elif args.mode == "load":
            if args.model_type == "all":
                summary = {
                    "combat_loads": validate_combat_model_loads(
                        repo_root=args.repo_root,
                        device=args.device,
                        combat_device=args.combat_device,
                        observation_version=args.observation_version,
                        include_alternates=True,
                    )
                }
            else:
                repo_paths = default_runtime_model_paths(args.repo_root)
                selected_path = args.model_path or repo_paths[args.model_type]
                selectors = build_runtime_selectors(
                    repo_root=args.repo_root,
                    device=args.device,
                    combat_device=args.combat_device,
                    combat_model=selected_path,
                    observation_version=args.observation_version,
                )
                sample_env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)
                selector = selectors["combat"]
                policy = CheckpointCombatPolicy(
                    checkpoint_path=str(selected_path),
                    device=args.combat_device or args.device,
                    observation_version=args.observation_version,
                )
                choice, scores = selector.choose(sample_env.state(), sample_env.legal_actions())
                scoring = policy.score_state(sample_env.state())
                summary = {
                    "selected_model_type": args.model_type,
                    "selected_model_path": str(selected_path),
                    "selector_available": selector.available,
                    "selector_choice": {
                        "kind": choice.get("kind") if choice else None,
                        "name": choice.get("name") if choice else None,
                        "scores_len": len(scores),
                    },
                    "policy_shapes": {
                        "action_logits": list(scoring["action_logits"].shape),
                        "target_logits": list(scoring["target_logits"].shape),
                    },
                }
        else:
            repo_paths = default_runtime_model_paths(args.repo_root)
            combat_model = args.model_path
            if combat_model is None and args.model_type in {"combat", "combat_bc", "combat_pref"}:
                combat_model = repo_paths[args.model_type]
            selectors = build_runtime_selectors(
                repo_root=args.repo_root,
                device=args.device,
                combat_device=args.combat_device,
                combat_model=combat_model,
                observation_version=args.observation_version,
            )
            summary = compare_seed_with_model(
                seed=args.seed,
                selectors=selectors,
                ascension=0,
                max_steps=args.max_steps,
                backend_pair=args.backend_pair,
                lightspeed_build=args.lightspeed_build,
                target_phase=args.target_phase,
                target_floor=args.target_floor,
            )
            summary["model_type"] = args.model_type
            summary["model_path"] = str(combat_model or repo_paths["combat"])
    except (ModuleNotFoundError, ImportError) as exc:
        message = _dependency_cli_message(exc)
        if message is None:
            raise
        print(message, file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
