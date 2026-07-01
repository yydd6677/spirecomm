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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


COMPARE_FIELDS = (
    "phase",
    "floor",
    "hp",
    "max_hp",
    "gold",
    "deck_size",
    "relic_count",
    "potion_count",
    "steps",
    "won",
    "dead",
    "timed_out",
    "timeout_reason",
    "max_floor_stopped",
    "error",
)


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("candidates") or payload.get("groups") or []
    if not isinstance(payload, list):
        raise SystemExit(f"candidate JSON must be a list or contain candidates/groups: {path}")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise SystemExit(f"candidate {index} is not an object")
        name = str(raw.get("name") or f"candidate_{index:03d}")
        if name in seen:
            name = f"{name}_{index:03d}"
        seen.add(name)
        env = raw.get("env") or raw.get("env_overrides") or {}
        if not isinstance(env, dict):
            raise SystemExit(f"candidate {name} env/env_overrides is not an object")
        candidates.append(
            {
                "index": int(index),
                "name": name,
                "env": {str(key): str(value) for key, value in env.items()},
                "raw": raw,
            }
        )
    if not candidates:
        raise SystemExit(f"no candidates in {path}")
    return candidates


def _parse_candidate_filter(raw: str, candidates: list[dict[str, Any]]) -> list[int]:
    if not str(raw or "").strip():
        return [0]
    by_name = {str(candidate["name"]): int(candidate["index"]) for candidate in candidates}
    selected: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            index = int(token)
        else:
            if token not in by_name:
                raise SystemExit(f"unknown candidate name: {token}")
            index = by_name[token]
        if index < 0 or index >= len(candidates):
            raise SystemExit(f"candidate index out of range: {index}")
        if index not in selected:
            selected.append(index)
    if not selected:
        raise SystemExit("--verify-candidates selected no candidates")
    return selected


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if str(args.seeds or "").strip():
        return [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _run(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"command failed rc={result.returncode} log={log_path}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nseconds={time.time() - started:.3f}\n")


def _shared_cmd(args: argparse.Namespace, shared_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts/v3_combat/run_shared_prefix_sweep.py"),
        "--repo-root",
        str(args.repo_root),
        "--candidate-json",
        str(args.candidate_json),
        "--output-dir",
        str(shared_dir),
        "--affected-phases",
        str(args.affected_phases),
        "--ascension",
        str(args.ascension),
        "--max-floor",
        str(args.max_floor),
        "--max-steps",
        str(args.max_steps),
        "--combat-stall-limit",
        str(args.combat_stall_limit),
        "--workers",
        str(args.shared_workers),
        "--task-batch-size",
        str(args.task_batch_size),
        "--torch-threads",
        str(args.torch_threads),
        "--preload-selectors",
        str(args.preload_selectors),
        "--device",
        str(args.device),
        "--combat-device",
        str(args.combat_device),
        "--combat-selector",
        str(args.combat_selector),
        "--combat-model",
        str(args.combat_model),
        "--v3-combat-model",
        str(args.v3_combat_model),
        "--card-reward-model",
        str(args.card_reward_model),
        "--shop-choice-model",
        str(args.shop_choice_model),
        "--shop-policy",
        str(args.shop_policy),
        "--shop-value-price-cost",
        str(args.shop_value_price_cost),
        "--shop-value-reserve-shortfall-cost",
        str(args.shop_value_reserve_shortfall_cost),
        "--shop-value-future-shop-reserve",
        str(args.shop_value_future_shop_reserve),
        "--shop-value-future-shop-horizon",
        str(args.shop_value_future_shop_horizon),
        "--shop-value-card-scale",
        str(args.shop_value_card_scale),
        "--shop-value-card-reference-price",
        str(args.shop_value_card_reference_price),
        "--shop-value-card-price-factor-min",
        str(args.shop_value_card_price_factor_min),
        "--shop-value-card-price-factor-max",
        str(args.shop_value_card_price_factor_max),
        "--shop-value-potion-scale",
        str(args.shop_value_potion_scale),
        "--shop-value-relic-scale",
        str(args.shop_value_relic_scale),
        "--shop-value-item-scale",
        str(args.shop_value_item_scale),
        "--shop-value-threshold",
        str(args.shop_value_threshold),
        "--shop-prior-weight-override",
        str(args.shop_prior_weight_override),
        "--v3-normal-room-potion-penalty",
        str(args.v3_normal_room_potion_penalty),
        "--summary-interval",
        "0",
    ]
    if str(args.phase_env_keys_json or "").strip():
        cmd.extend(["--phase-env-keys-json", str(args.phase_env_keys_json)])
    if str(args.seeds or "").strip():
        cmd.extend(["--seeds", str(args.seeds)])
    else:
        cmd.extend(["--seed-start", str(args.seed_start), "--count", str(args.count)])
    if args.no_merge_identical_states:
        cmd.append("--no-merge-identical-states")
    return cmd


def _independent_cmd(args: argparse.Namespace, output_dir: Path, seeds: list[int]) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts/v3_combat/evaluate_v3_rollout_batch.py"),
        "--repo-root",
        str(args.repo_root),
        "--output-dir",
        str(output_dir),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--ascension",
        str(args.ascension),
        "--max-floor",
        str(args.max_floor),
        "--max-steps",
        str(args.max_steps),
        "--no-progress-limit",
        "0",
        "--combat-stall-limit",
        str(args.combat_stall_limit),
        "--workers",
        str(args.independent_workers),
        "--trace-mode",
        "none",
        "--metrics-mode",
        "floor",
        "--task-batch-size",
        str(args.task_batch_size),
        "--torch-threads",
        str(args.torch_threads),
        "--preload-selectors",
        str(args.preload_selectors),
        "--device",
        str(args.device),
        "--combat-device",
        str(args.combat_device),
        "--combat-selector",
        str(args.combat_selector),
        "--combat-model",
        str(args.combat_model),
        "--v3-combat-model",
        str(args.v3_combat_model),
        "--card-reward-model",
        str(args.card_reward_model),
        "--shop-choice-model",
        str(args.shop_choice_model),
        "--shop-policy",
        str(args.shop_policy),
        "--shop-value-price-cost",
        str(args.shop_value_price_cost),
        "--shop-value-reserve-shortfall-cost",
        str(args.shop_value_reserve_shortfall_cost),
        "--shop-value-future-shop-reserve",
        str(args.shop_value_future_shop_reserve),
        "--shop-value-future-shop-horizon",
        str(args.shop_value_future_shop_horizon),
        "--shop-value-card-scale",
        str(args.shop_value_card_scale),
        "--shop-value-card-reference-price",
        str(args.shop_value_card_reference_price),
        "--shop-value-card-price-factor-min",
        str(args.shop_value_card_price_factor_min),
        "--shop-value-card-price-factor-max",
        str(args.shop_value_card_price_factor_max),
        "--shop-value-potion-scale",
        str(args.shop_value_potion_scale),
        "--shop-value-relic-scale",
        str(args.shop_value_relic_scale),
        "--shop-value-item-scale",
        str(args.shop_value_item_scale),
        "--shop-value-threshold",
        str(args.shop_value_threshold),
        "--shop-prior-weight-override",
        str(args.shop_prior_weight_override),
        "--v3-normal-room-potion-penalty",
        str(args.v3_normal_room_potion_penalty),
        "--no-write-results-json",
    ]
    return cmd


def _compare_rows(shared_rows: list[dict[str, Any]], independent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shared_by_seed = {int(row["seed"]): row for row in shared_rows}
    independent_by_seed = {int(row["seed"]): row for row in independent_rows}
    mismatches: list[dict[str, Any]] = []
    for seed in sorted(set(shared_by_seed) | set(independent_by_seed)):
        shared = shared_by_seed.get(seed)
        independent = independent_by_seed.get(seed)
        if shared is None or independent is None:
            mismatches.append({"seed": seed, "reason": "missing_result", "shared": shared is not None, "independent": independent is not None})
            continue
        field_diffs = {
            field: {"shared": shared.get(field), "independent": independent.get(field)}
            for field in COMPARE_FIELDS
            if shared.get(field) != independent.get(field)
        }
        if field_diffs:
            mismatches.append({"seed": seed, "reason": "field_mismatch", "fields": field_diffs})
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify shared-prefix sweep results against independent rollout evaluation.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_cache/shared_prefix_consistency"))
    parser.add_argument("--shared-output-dir", type=Path, default=None, help="Use an existing shared-prefix output instead of rerunning it.")
    parser.add_argument("--verify-candidates", default="0", help="Comma-separated candidate names or indices. Default verifies candidate 0.")
    parser.add_argument("--affected-phases", required=True)
    parser.add_argument("--phase-env-keys-json", default="")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--combat-stall-limit", type=int, default=int(os.environ.get("SPIRECOMM_COMBAT_STALL_LIMIT", "250")))
    parser.add_argument("--shared-workers", type=int, default=2)
    parser.add_argument("--independent-workers", type=int, default=2)
    parser.add_argument("--task-batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--preload-selectors", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--no-merge-identical-states", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_scorer.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument("--shop-choice-model", type=Path, default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")))
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--shop-value-price-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")))
    parser.add_argument("--shop-value-reserve-shortfall-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")))
    parser.add_argument("--shop-value-future-shop-reserve", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "120")))
    parser.add_argument("--shop-value-future-shop-horizon", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "5")))
    parser.add_argument("--shop-value-card-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "4.6262945279949435")))
    parser.add_argument("--shop-value-card-reference-price", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "60.0")))
    parser.add_argument("--shop-value-card-price-factor-min", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "0.65")))
    parser.add_argument("--shop-value-card-price-factor-max", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "1.35")))
    parser.add_argument("--shop-value-potion-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "0.5084989138155764")))
    parser.add_argument("--shop-value-relic-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "0.8")))
    parser.add_argument("--shop-value-item-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "1.0")))
    parser.add_argument("--shop-value-threshold", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_THRESHOLD", "0.0")))
    parser.add_argument("--shop-prior-weight-override", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "0.8")))
    parser.add_argument("--v3-normal-room-potion-penalty", type=float, default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5")))
    args = parser.parse_args()

    candidates = _load_candidates(args.candidate_json)
    verify_indices = _parse_candidate_filter(args.verify_candidates, candidates)
    seeds = _parse_seeds(args)
    output_dir = args.output_dir
    if output_dir.exists() and not args.keep_temp:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = args.shared_output_dir or (output_dir / "shared_prefix")
    logs_dir = output_dir / "logs"

    if args.shared_output_dir is None:
        _run(_shared_cmd(args, shared_dir), log_path=logs_dir / "shared_prefix.log")

    shared_rows_all = _read_jsonl(shared_dir / "seed_results.jsonl")
    if not shared_rows_all:
        raise SystemExit(f"missing shared seed results: {shared_dir / 'seed_results.jsonl'}")
    seed_filter = set(int(seed) for seed in seeds)
    shared_by_candidate: dict[int, list[dict[str, Any]]] = {index: [] for index in verify_indices}
    for row in shared_rows_all:
        index = int(row.get("candidate_index", -1))
        if index in shared_by_candidate and int(row.get("seed", -1)) in seed_filter:
            shared_by_candidate[index].append(row)

    reports: list[dict[str, Any]] = []
    total_mismatches = 0
    for index in verify_indices:
        candidate = candidates[index]
        independent_dir = output_dir / f"independent_{index:03d}_{candidate['name']}"
        env = os.environ.copy()
        env.update(candidate["env"])
        _run(_independent_cmd(args, independent_dir, seeds), env=env, log_path=logs_dir / f"independent_{index:03d}_{candidate['name']}.log")
        independent_rows = _read_jsonl(independent_dir / "results.jsonl")
        mismatches = _compare_rows(shared_by_candidate[index], independent_rows)
        total_mismatches += len(mismatches)
        reports.append(
            {
                "candidate_index": index,
                "candidate_name": candidate["name"],
                "seed_count": len(seeds),
                "shared_count": len(shared_by_candidate[index]),
                "independent_count": len(independent_rows),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches[:20],
            }
        )
        if not args.keep_temp and not mismatches:
            shutil.rmtree(independent_dir, ignore_errors=True)

    summary = {
        "ok": total_mismatches == 0,
        "seed_count": len(seeds),
        "verified_candidates": [candidates[index]["name"] for index in verify_indices],
        "mismatch_count": total_mismatches,
        "reports": reports,
        "shared_output_dir": str(shared_dir),
    }
    (output_dir / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if total_mismatches != 0:
        raise SystemExit(f"shared-prefix consistency failed: mismatches={total_mismatches}; details={output_dir / 'verification.json'}")


if __name__ == "__main__":
    main()
