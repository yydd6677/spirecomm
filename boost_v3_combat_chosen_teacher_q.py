#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spirecomm.ai.v3_combat_dataset import load_shard, save_shard


def _boost_root(labeled: Any, *, margin: float) -> bool:
    chosen_key = tuple(getattr(labeled.root, "chosen_action_key", ()) or ())
    if not chosen_key:
        return False
    chosen_candidates = [candidate for candidate in labeled.candidates if tuple(candidate.action_key) == chosen_key]
    if not chosen_candidates:
        return False
    chosen = chosen_candidates[0]
    max_q = max((float(candidate.teacher_q) for candidate in labeled.candidates), default=0.0)
    target_q = max(float(chosen.teacher_q), max_q + float(margin))
    delta = target_q - float(chosen.teacher_q)
    if delta <= 0.0:
        return False
    chosen.teacher_q = target_q
    components = dict(getattr(chosen, "reward_components", {}) or {})
    components["teacher_q"] = target_q
    components["chosen_teacher_q_pseudo_boost"] = float(delta)
    chosen.reward_components = components
    ranked = sorted(range(len(labeled.candidates)), key=lambda index: labeled.candidates[index].teacher_q, reverse=True)
    for rank, index in enumerate(ranked):
        labeled.candidates[index].teacher_rank = rank
    metadata = dict(getattr(labeled.root, "metadata", {}) or {})
    metadata["chosen_teacher_q_pseudo_boost"] = {
        "margin": float(margin),
        "delta": float(delta),
    }
    labeled.root.metadata = metadata
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Boost chosen candidate teacher_q in v3 combat shards.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--glob", default="*.pt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []
    total_roots = 0
    boosted_roots = 0
    for shard_index, shard_path in enumerate(sorted(args.input_dir.glob(args.glob))):
        payload = load_shard(shard_path)
        roots = list(payload.get("roots") or [])
        shard_boosted = 0
        for labeled in roots:
            if _boost_root(labeled, margin=float(args.margin)):
                shard_boosted += 1
        if not roots:
            continue
        out_path = args.output_dir / shard_path.name
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "source_shard": str(shard_path),
                "pseudo_boost_margin": float(args.margin),
                "boosted_roots": int(shard_boosted),
                "root_count": len(roots),
            }
        )
        save_shard(out_path, roots, metadata=metadata)
        total_roots += len(roots)
        boosted_roots += shard_boosted
        shard_records.append({"source": str(shard_path), "path": str(out_path), "root_count": len(roots), "boosted_roots": shard_boosted})
        print(
            f"[boost-chosen] shard={shard_index + 1} roots={len(roots)} boosted={shard_boosted} output={out_path}",
            flush=True,
        )
    summary = {
        "schema": "v3_combat_chosen_teacher_q_boost_v1",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "margin": float(args.margin),
        "shard_count": len(shard_records),
        "root_count": total_roots,
        "boosted_roots": boosted_roots,
        "shards": shard_records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "shards.json").write_text(json.dumps({"shards": shard_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
