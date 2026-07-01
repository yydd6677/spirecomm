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
from collections import Counter
from pathlib import Path
from typing import Any

from spirecomm.ai.v3_combat_dataset import load_shard, save_shard


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _iter_input_shards(paths: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for path in paths:
        if path.is_dir():
            shards.extend(sorted(path.glob("*.pt")))
        elif path.exists():
            shards.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(shards))


def _root_metadata(root: Any) -> dict[str, Any]:
    metadata = dict(getattr(getattr(root, "root", None), "metadata", {}) or {})
    hard = metadata.get("onpolicy_hard_validation")
    if isinstance(hard, dict):
        return dict(hard)
    teacher_config = dict(getattr(root, "teacher_config", {}) or {})
    hard = teacher_config.get("onpolicy_hard_validation")
    return dict(hard) if isinstance(hard, dict) else {}


def _passes(root: Any, args: argparse.Namespace) -> tuple[bool, str]:
    hard = _root_metadata(root)
    if not hard:
        return False, "missing_hard_metadata"
    categories = set(str(item) for item in hard.get("categories") or [])
    required = set(_split_csv(args.require_categories))
    excluded = set(_split_csv(args.exclude_categories))
    if required and not required.issubset(categories):
        return False, "missing_required_category"
    if excluded and excluded.intersection(categories):
        return False, "excluded_category"
    if args.teacher_kind and str(hard.get("teacher_top_kind") or "") != str(args.teacher_kind):
        return False, "teacher_kind"
    if args.pred_kind and str(hard.get("pred_top_kind") or "") != str(args.pred_kind):
        return False, "pred_kind"
    regret = float(hard.get("regret") or 0.0)
    if regret < float(args.min_regret):
        return False, "min_regret"
    if float(args.max_regret) > 0.0 and regret > float(args.max_regret):
        return False, "max_regret"
    teacher_gap = float(hard.get("teacher_gap") or 0.0)
    if teacher_gap < float(args.min_teacher_gap):
        return False, "min_teacher_gap"
    pred_margin = float(hard.get("pred_margin") or 0.0)
    if pred_margin < float(args.min_pred_margin):
        return False, "min_pred_margin"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter labeled on-policy hard roots into a smaller training shard.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input shard files or directories.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-categories", default="")
    parser.add_argument("--exclude-categories", default="")
    parser.add_argument("--teacher-kind", default="")
    parser.add_argument("--pred-kind", default="")
    parser.add_argument("--min-regret", type=float, default=0.0)
    parser.add_argument("--max-regret", type=float, default=0.0)
    parser.add_argument("--min-teacher-gap", type=float, default=0.0)
    parser.add_argument("--min-pred-margin", type=float, default=0.0)
    parser.add_argument("--max-roots", type=int, default=0)
    args = parser.parse_args()

    shard_paths = _iter_input_shards(list(args.inputs))
    selected: list[Any] = []
    selected_keys: set[str] = set()
    filter_counts: Counter[str] = Counter()
    selected_categories: Counter[str] = Counter()
    selected_transitions: Counter[str] = Counter()
    selected_regret_sum = 0.0

    for shard_path in shard_paths:
        payload = load_shard(shard_path)
        for root in payload.get("roots") or []:
            hard = _root_metadata(root)
            key = str(hard.get("root_id") or getattr(getattr(root, "root", None), "root_id", ""))
            if key and key in selected_keys:
                filter_counts.update(["duplicate_root"])
                continue
            ok, reason = _passes(root, args)
            if not ok:
                filter_counts.update([reason])
                continue
            if key:
                selected_keys.add(key)
            selected.append(root)
            categories = [str(item) for item in hard.get("categories") or []]
            selected_categories.update(categories)
            transition = f"{hard.get('teacher_top_kind')}->{hard.get('pred_top_kind')}"
            selected_transitions.update([transition])
            selected_regret_sum += float(hard.get("regret") or 0.0)
            if int(args.max_roots) > 0 and len(selected) >= int(args.max_roots):
                break
        if int(args.max_roots) > 0 and len(selected) >= int(args.max_roots):
            break

    metadata = {
        "schema": "v3_combat_filtered_hard_roots_v1",
        "input_count": len(shard_paths),
        "root_count": len(selected),
        "require_categories": _split_csv(args.require_categories),
        "exclude_categories": _split_csv(args.exclude_categories),
        "teacher_kind": str(args.teacher_kind),
        "pred_kind": str(args.pred_kind),
        "min_regret": float(args.min_regret),
        "max_regret": float(args.max_regret),
        "min_teacher_gap": float(args.min_teacher_gap),
        "min_pred_margin": float(args.min_pred_margin),
        "filter_counts": dict(filter_counts),
        "selected_category_counts": dict(selected_categories),
        "selected_transition_counts": dict(selected_transitions),
        "selected_mean_regret": selected_regret_sum / max(1, len(selected)),
    }
    if not selected:
        raise SystemExit(f"no roots selected; metadata={json.dumps(metadata, ensure_ascii=False)}")
    save_shard(args.output, selected, metadata=metadata)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metadata}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
