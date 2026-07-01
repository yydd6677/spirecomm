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
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import load_shard
from spirecomm.ai.v3_combat_model import load_v3_combat_checkpoint
from spirecomm.ai.v3_combat_transformer import load_v3_combat_transformer_checkpoint


def _default_device(requested: str) -> str:
    if requested != "auto":
        return requested
    torch = require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_validation_chunks(manifest: dict[str, Any], *, seed: int, validation_fraction: float) -> list[dict[str, Any]]:
    chunks = list(manifest.get("chunks") or [])
    if not chunks:
        raise ValueError("transformer tensor dataset is not chunked; this diagnostic expects the training manifest")
    rng = random.Random(int(seed))
    rng.shuffle(chunks)
    val_count = int(len(chunks) * float(validation_fraction))
    return chunks[:val_count]


def _load_chunk(path: Path) -> dict[str, Any]:
    torch = require_torch()
    return torch.load(path, map_location="cpu", weights_only=False)


def _chunk_batch(chunk: dict[str, Any], root_indices: list[int], *, device: str) -> dict[str, Any]:
    torch = require_torch()
    offsets = chunk["candidate_offsets"]
    root_tensor = torch.tensor(root_indices, dtype=torch.long)
    starts = offsets.index_select(0, root_tensor)
    ends = offsets.index_select(0, root_tensor + 1)
    counts = ends - starts
    total_candidates = int(counts.sum().item())
    root_starts = torch.repeat_interleave(starts, counts)
    candidate_offsets = torch.arange(total_candidates, dtype=torch.long) - torch.repeat_interleave(
        torch.cumsum(counts, dim=0) - counts,
        counts,
    )
    candidate_indices = root_starts + candidate_offsets
    return {
        "token_scalar_features": chunk["token_scalar_features"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "token_type_ids": chunk["token_type_ids"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "entity_ids": chunk["entity_ids"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "slot_ids": chunk["slot_ids"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "attention_mask": chunk["attention_mask"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "features": chunk["features"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "teacher_q": chunk["teacher_q"].index_select(0, candidate_indices).to(device, non_blocking=True),
        "candidate_counts": counts,
        "candidate_indices": candidate_indices,
    }


def _action_kind(candidate: Any) -> str:
    action = getattr(candidate, "action", None)
    if isinstance(action, dict):
        return str(action.get("kind") or "unknown")
    return "unknown"


def _action_label(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"kind": "unknown"}
    keys = (
        "kind",
        "card_id",
        "card_name",
        "name",
        "potion_id",
        "potion_name",
        "potion_index",
        "card_index",
        "source_index",
        "target_index",
    )
    return {key: action.get(key) for key in keys if key in action}


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    combat = dict(state.get("combat") or {})
    return {
        "room_type": str(state.get("room_type") or ""),
        "floor": state.get("floor"),
        "act": state.get("act"),
        "turn": combat.get("turn"),
        "current_hp": state.get("current_hp"),
        "max_hp": state.get("max_hp"),
        "gold": state.get("gold"),
    }


def _new_stats() -> dict[str, Any]:
    return {
        "roots": 0,
        "potion_candidate_roots": 0,
        "teacher_top_potion_roots": 0,
        "mlp_top_potion_roots": 0,
        "transformer_top_potion_roots": 0,
        "mlp_top1": 0,
        "transformer_top1": 0,
        "mlp_top1_on_potion_candidate": 0,
        "transformer_top1_on_potion_candidate": 0,
        "mlp_top_potion_when_teacher_top_potion": 0,
        "transformer_top_potion_when_teacher_top_potion": 0,
        "mlp_exact_top_when_teacher_top_potion": 0,
        "transformer_exact_top_when_teacher_top_potion": 0,
        "mlp_best_teacher_potion_rank_sum": 0.0,
        "transformer_best_teacher_potion_rank_sum": 0.0,
        "mlp_best_teacher_potion_gap_sum": 0.0,
        "transformer_best_teacher_potion_gap_sum": 0.0,
        "mlp_best_pred_potion_gap_sum": 0.0,
        "transformer_best_pred_potion_gap_sum": 0.0,
    }


def _mean(stats: dict[str, Any], key: str, denom: str) -> float | None:
    denominator = int(stats.get(denom) or 0)
    if denominator <= 0:
        return None
    return float(stats.get(key) or 0.0) / denominator


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    roots = int(stats["roots"])
    potion_roots = int(stats["potion_candidate_roots"])
    teacher_potion_roots = int(stats["teacher_top_potion_roots"])
    result = dict(stats)
    result.update(
        {
            "mlp_top1_rate": _mean(stats, "mlp_top1", "roots"),
            "transformer_top1_rate": _mean(stats, "transformer_top1", "roots"),
            "potion_candidate_rate": None if roots <= 0 else potion_roots / roots,
            "teacher_top_potion_rate": None if roots <= 0 else teacher_potion_roots / roots,
            "mlp_top_potion_rate": None if roots <= 0 else int(stats["mlp_top_potion_roots"]) / roots,
            "transformer_top_potion_rate": None if roots <= 0 else int(stats["transformer_top_potion_roots"]) / roots,
            "mlp_top1_on_potion_candidate_rate": _mean(stats, "mlp_top1_on_potion_candidate", "potion_candidate_roots"),
            "transformer_top1_on_potion_candidate_rate": _mean(
                stats,
                "transformer_top1_on_potion_candidate",
                "potion_candidate_roots",
            ),
            "mlp_top_potion_recall_when_teacher_top_potion": _mean(
                stats,
                "mlp_top_potion_when_teacher_top_potion",
                "teacher_top_potion_roots",
            ),
            "transformer_top_potion_recall_when_teacher_top_potion": _mean(
                stats,
                "transformer_top_potion_when_teacher_top_potion",
                "teacher_top_potion_roots",
            ),
            "mlp_exact_top_when_teacher_top_potion_rate": _mean(
                stats,
                "mlp_exact_top_when_teacher_top_potion",
                "teacher_top_potion_roots",
            ),
            "transformer_exact_top_when_teacher_top_potion_rate": _mean(
                stats,
                "transformer_exact_top_when_teacher_top_potion",
                "teacher_top_potion_roots",
            ),
            "mlp_best_teacher_potion_mean_rank": _mean(stats, "mlp_best_teacher_potion_rank_sum", "potion_candidate_roots"),
            "transformer_best_teacher_potion_mean_rank": _mean(
                stats,
                "transformer_best_teacher_potion_rank_sum",
                "potion_candidate_roots",
            ),
            "mlp_best_teacher_potion_mean_gap": _mean(stats, "mlp_best_teacher_potion_gap_sum", "potion_candidate_roots"),
            "transformer_best_teacher_potion_mean_gap": _mean(
                stats,
                "transformer_best_teacher_potion_gap_sum",
                "potion_candidate_roots",
            ),
            "mlp_best_pred_potion_mean_gap": _mean(stats, "mlp_best_pred_potion_gap_sum", "potion_candidate_roots"),
            "transformer_best_pred_potion_mean_gap": _mean(
                stats,
                "transformer_best_pred_potion_gap_sum",
                "potion_candidate_roots",
            ),
        }
    )
    return result


def _top_index(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _rank_desc(values: list[float], index: int) -> int:
    target = values[index]
    return 1 + sum(1 for value in values if value > target)


def _best_gap(pred: list[float], potion_indices: list[int], non_potion_indices: list[int]) -> float | None:
    if not potion_indices or not non_potion_indices:
        return None
    return max(pred[index] for index in potion_indices) - max(pred[index] for index in non_potion_indices)


def _teacher_potion_gap(pred: list[float], teacher_q: list[float], potion_indices: list[int], non_potion_indices: list[int]) -> float | None:
    if not potion_indices or not non_potion_indices:
        return None
    best_teacher_potion = max(potion_indices, key=lambda index: teacher_q[index])
    best_teacher_non_potion = max(non_potion_indices, key=lambda index: teacher_q[index])
    return pred[best_teacher_potion] - pred[best_teacher_non_potion]


def _update_stats(stats: dict[str, Any], record: dict[str, Any]) -> None:
    stats["roots"] += 1
    has_potion = bool(record["has_potion_candidate"])
    teacher_top_potion = bool(record["teacher_top_is_potion"])
    mlp_top_potion = bool(record["mlp_top_is_potion"])
    transformer_top_potion = bool(record["transformer_top_is_potion"])
    if has_potion:
        stats["potion_candidate_roots"] += 1
    if teacher_top_potion:
        stats["teacher_top_potion_roots"] += 1
    if mlp_top_potion:
        stats["mlp_top_potion_roots"] += 1
    if transformer_top_potion:
        stats["transformer_top_potion_roots"] += 1
    if record["mlp_top_index"] == record["teacher_top_index"]:
        stats["mlp_top1"] += 1
        if has_potion:
            stats["mlp_top1_on_potion_candidate"] += 1
    if record["transformer_top_index"] == record["teacher_top_index"]:
        stats["transformer_top1"] += 1
        if has_potion:
            stats["transformer_top1_on_potion_candidate"] += 1
    if teacher_top_potion:
        if mlp_top_potion:
            stats["mlp_top_potion_when_teacher_top_potion"] += 1
        if transformer_top_potion:
            stats["transformer_top_potion_when_teacher_top_potion"] += 1
        if record["mlp_top_index"] == record["teacher_top_index"]:
            stats["mlp_exact_top_when_teacher_top_potion"] += 1
        if record["transformer_top_index"] == record["teacher_top_index"]:
            stats["transformer_exact_top_when_teacher_top_potion"] += 1
    if has_potion:
        stats["mlp_best_teacher_potion_rank_sum"] += float(record["mlp_best_teacher_potion_rank"])
        stats["transformer_best_teacher_potion_rank_sum"] += float(record["transformer_best_teacher_potion_rank"])
        stats["mlp_best_teacher_potion_gap_sum"] += float(record["mlp_best_teacher_potion_gap"])
        stats["transformer_best_teacher_potion_gap_sum"] += float(record["transformer_best_teacher_potion_gap"])
        stats["mlp_best_pred_potion_gap_sum"] += float(record["mlp_best_pred_potion_gap"])
        stats["transformer_best_pred_potion_gap_sum"] += float(record["transformer_best_pred_potion_gap"])


def _record_for_root(
    *,
    root: Any,
    candidates: list[Any],
    teacher_q: list[float],
    mlp_pred: list[float],
    transformer_pred: list[float],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    kinds = [_action_kind(candidate) for candidate in candidates]
    potion_indices = [index for index, kind in enumerate(kinds) if kind == "potion"]
    non_potion_indices = [index for index, kind in enumerate(kinds) if kind != "potion"]
    teacher_top = _top_index(teacher_q)
    mlp_top = _top_index(mlp_pred)
    transformer_top = _top_index(transformer_pred)
    best_teacher_potion = max(potion_indices, key=lambda index: teacher_q[index]) if potion_indices else None
    record = {
        "root_id": str(getattr(root.root, "root_id", "")),
        "state": _state_summary(getattr(root.root, "visible_before", {}) or {}),
        "candidate_count": len(candidates),
        "has_potion_candidate": bool(potion_indices),
        "teacher_top_index": teacher_top,
        "mlp_top_index": mlp_top,
        "transformer_top_index": transformer_top,
        "teacher_top_is_potion": kinds[teacher_top] == "potion",
        "mlp_top_is_potion": kinds[mlp_top] == "potion",
        "transformer_top_is_potion": kinds[transformer_top] == "potion",
        "best_teacher_potion_index": best_teacher_potion,
        "teacher_top": {
            "action": _action_label(getattr(candidates[teacher_top], "action", None)),
            "teacher_q": teacher_q[teacher_top],
            "mlp_pred": mlp_pred[teacher_top],
            "transformer_pred": transformer_pred[teacher_top],
        },
        "mlp_top": {
            "action": _action_label(getattr(candidates[mlp_top], "action", None)),
            "teacher_q": teacher_q[mlp_top],
            "mlp_pred": mlp_pred[mlp_top],
            "transformer_pred": transformer_pred[mlp_top],
        },
        "transformer_top": {
            "action": _action_label(getattr(candidates[transformer_top], "action", None)),
            "teacher_q": teacher_q[transformer_top],
            "mlp_pred": mlp_pred[transformer_top],
            "transformer_pred": transformer_pred[transformer_top],
        },
    }
    if best_teacher_potion is not None:
        mlp_gap = _teacher_potion_gap(mlp_pred, teacher_q, potion_indices, non_potion_indices)
        transformer_gap = _teacher_potion_gap(transformer_pred, teacher_q, potion_indices, non_potion_indices)
        mlp_best_gap = _best_gap(mlp_pred, potion_indices, non_potion_indices)
        transformer_best_gap = _best_gap(transformer_pred, potion_indices, non_potion_indices)
        record.update(
            {
                "best_teacher_potion": {
                    "action": _action_label(getattr(candidates[best_teacher_potion], "action", None)),
                    "teacher_q": teacher_q[best_teacher_potion],
                    "mlp_pred": mlp_pred[best_teacher_potion],
                    "transformer_pred": transformer_pred[best_teacher_potion],
                },
                "mlp_best_teacher_potion_rank": _rank_desc(mlp_pred, best_teacher_potion),
                "transformer_best_teacher_potion_rank": _rank_desc(transformer_pred, best_teacher_potion),
                "mlp_best_teacher_potion_gap": float(mlp_gap if mlp_gap is not None else 0.0),
                "transformer_best_teacher_potion_gap": float(transformer_gap if transformer_gap is not None else 0.0),
                "mlp_best_pred_potion_gap": float(mlp_best_gap if mlp_best_gap is not None else 0.0),
                "transformer_best_pred_potion_gap": float(transformer_best_gap if transformer_best_gap is not None else 0.0),
            }
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MLP vs Transformer potion ranking on v3 combat validation roots.")
    parser.add_argument("--transformer-tensor-dataset", type=Path, default=Path("data/v3_combat_tensor/transformer_stage5_v1.pt"))
    parser.add_argument("--mlp-model", type=Path, default=Path("models/v3_combat_scorer_potion_stage5.pt"))
    parser.add_argument("--transformer-model", type=Path, default=Path("models/v3_combat_transformer_stage5_v1.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("_cache/v3_combat_transformer_potion_diagnostics"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--batch-roots", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-disagreements", type=int, default=200)
    args = parser.parse_args()

    torch = require_torch()
    device = _default_device(args.device)
    manifest = torch.load(args.transformer_tensor_dataset, map_location="cpu", weights_only=False)
    validation_chunks = _load_validation_chunks(
        manifest,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    mlp_model, mlp_checkpoint = load_v3_combat_checkpoint(args.mlp_model, device=device)
    transformer_model, transformer_checkpoint = load_v3_combat_transformer_checkpoint(args.transformer_model, device=device)
    mlp_model.eval()
    transformer_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    disagreements_path = args.output_dir / "strict_potion_disagreements.jsonl"

    global_stats = _new_stats()
    by_room: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    by_potion_id: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    room_counts = Counter()
    strict_disagreements: list[dict[str, Any]] = []
    processed_roots = 0
    processed_candidates = 0

    with torch.no_grad():
        for chunk_number, chunk_meta in enumerate(validation_chunks, start=1):
            chunk = _load_chunk(Path(chunk_meta["path"]))
            source_shard = Path(chunk_meta["source_shard"])
            shard_payload = load_shard(source_shard)
            roots = list(shard_payload.get("roots") or [])
            if list(chunk.get("root_ids") or []) != [str(getattr(labeled.root, "root_id", "")) for labeled in roots]:
                raise ValueError(f"root id mismatch between {chunk_meta['path']} and {source_shard}")

            for start in range(0, len(roots), max(1, int(args.batch_roots))):
                root_indices = list(range(start, min(len(roots), start + max(1, int(args.batch_roots)))))
                batch = _chunk_batch(chunk, root_indices, device=device)
                mlp_pred = mlp_model(batch["features"].float()).detach().cpu()
                transformer_pred = transformer_model(batch).detach().cpu()
                teacher_q = batch["teacher_q"].detach().cpu()
                counts = batch["candidate_counts"].tolist()

                offset = 0
                for local_root_index, count in zip(root_indices, counts, strict=False):
                    end = offset + int(count)
                    labeled = roots[local_root_index]
                    record = _record_for_root(
                        root=labeled,
                        candidates=list(labeled.candidates),
                        teacher_q=[float(value) for value in teacher_q[offset:end].tolist()],
                        mlp_pred=[float(value) for value in mlp_pred[offset:end].tolist()],
                        transformer_pred=[float(value) for value in transformer_pred[offset:end].tolist()],
                    )
                    offset = end
                    if record is None:
                        continue
                    room_type = str(record["state"].get("room_type") or "UNKNOWN")
                    room_counts[room_type] += 1
                    _update_stats(global_stats, record)
                    _update_stats(by_room[room_type], record)
                    if record["best_teacher_potion_index"] is not None:
                        potion_action = record["best_teacher_potion"]["action"]
                        potion_id = str(potion_action.get("potion_id") or potion_action.get("name") or "UNKNOWN")
                        _update_stats(by_potion_id[potion_id], record)
                    if (
                        record["teacher_top_is_potion"]
                        and record["mlp_top_is_potion"]
                        and not record["transformer_top_is_potion"]
                    ):
                        strict_disagreements.append(record)
                    processed_roots += 1
                    processed_candidates += int(count)

            print(
                f"processed validation chunk {chunk_number}/{len(validation_chunks)} "
                f"roots={processed_roots} candidates={processed_candidates}",
                flush=True,
            )

    strict_disagreements.sort(
        key=lambda item: (
            float(item["best_teacher_potion"]["teacher_q"]) - float(item["transformer_top"]["teacher_q"]),
            float(item.get("mlp_best_teacher_potion_gap", 0.0)) - float(item.get("transformer_best_teacher_potion_gap", 0.0)),
        ),
        reverse=True,
    )
    with disagreements_path.open("w", encoding="utf-8") as handle:
        for item in strict_disagreements[: max(0, int(args.max_disagreements))]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "transformer_tensor_dataset": str(args.transformer_tensor_dataset),
            "mlp_model": str(args.mlp_model),
            "transformer_model": str(args.transformer_model),
            "output_dir": str(args.output_dir),
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "validation_chunk_count": len(validation_chunks),
            "device": device,
            "batch_roots": args.batch_roots,
        },
        "model_metadata": {
            "mlp_dataset_metadata": mlp_checkpoint.get("dataset_metadata"),
            "transformer_dataset_metadata": transformer_checkpoint.get("dataset_metadata"),
            "transformer_token_schema": transformer_checkpoint.get("token_schema"),
        },
        "processed": {
            "roots": processed_roots,
            "candidates": processed_candidates,
            "room_counts": dict(room_counts),
            "strict_disagreement_count": len(strict_disagreements),
            "strict_disagreements_written": min(len(strict_disagreements), max(0, int(args.max_disagreements))),
        },
        "overall": _finalize_stats(global_stats),
        "by_room": {room: _finalize_stats(stats) for room, stats in sorted(by_room.items())},
        "by_best_teacher_potion_id": {
            potion_id: _finalize_stats(stats)
            for potion_id, stats in sorted(by_potion_id.items(), key=lambda item: (-int(item[1]["potion_candidate_roots"]), item[0]))
        },
        "outputs": {
            "summary": str(summary_path),
            "strict_potion_disagreements": str(disagreements_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["processed"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {disagreements_path}", flush=True)


if __name__ == "__main__":
    main()
