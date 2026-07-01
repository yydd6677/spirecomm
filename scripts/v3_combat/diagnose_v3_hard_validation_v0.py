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
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_features import COMBAT_ROOM_TYPES
from spirecomm.ai.v3_combat_transformer import load_v3_combat_transformer_checkpoint
from scripts.v3_combat.train_v3_combat_transformer_scorer import (
    _is_chunked_payload,
    _is_root_payload,
    _load_chunk,
    _load_tensor_dataset,
    _tensor_batch,
)


ROOM_TYPE_NAMES = {0: "Unknown", **{index + 1: name for index, name in enumerate(COMBAT_ROOM_TYPES)}}


@dataclass
class BucketStats:
    roots: int = 0
    candidates: int = 0
    top1: int = 0
    regret_sum: float = 0.0
    regret_sq_sum: float = 0.0
    gap_sum: float = 0.0
    pred_margin_sum: float = 0.0
    gap_weighted_regret_sum: float = 0.0
    teacher_top_q_sum: float = 0.0
    pred_choice_teacher_q_sum: float = 0.0
    high_conf_disagreements: int = 0
    teacher_top_kind_counts: Counter[str] = field(default_factory=Counter)
    pred_top_kind_counts: Counter[str] = field(default_factory=Counter)
    transition_counts: Counter[str] = field(default_factory=Counter)
    potion_pair_roots: int = 0
    potion_pair_sign_correct: int = 0
    teacher_prefers_potion_roots: int = 0
    pred_prefers_potion_roots: int = 0
    missed_teacher_potion_roots: int = 0
    false_potion_top_roots: int = 0
    baseline_top1: int = 0
    baseline_wrong_roots: int = 0
    model_recovers_baseline_wrong: int = 0
    model_breaks_baseline_correct: int = 0

    def update(
        self,
        *,
        candidate_count: int,
        top1: bool,
        regret: float,
        teacher_gap: float,
        pred_margin: float,
        high_conf_disagreement: bool,
        teacher_top_q: float,
        pred_choice_teacher_q: float,
        teacher_kind: str,
        pred_kind: str,
        potion_pair: bool,
        potion_pair_sign_correct: bool,
        teacher_prefers_potion: bool,
        pred_prefers_potion: bool,
        baseline_top1: bool | None,
    ) -> None:
        self.roots += 1
        self.candidates += int(candidate_count)
        self.top1 += int(top1)
        self.regret_sum += float(regret)
        self.regret_sq_sum += float(regret) * float(regret)
        self.gap_sum += float(teacher_gap)
        self.pred_margin_sum += float(pred_margin)
        self.gap_weighted_regret_sum += float(regret) * max(0.0, float(teacher_gap))
        self.teacher_top_q_sum += float(teacher_top_q)
        self.pred_choice_teacher_q_sum += float(pred_choice_teacher_q)
        self.high_conf_disagreements += int(high_conf_disagreement)
        self.teacher_top_kind_counts.update([teacher_kind])
        self.pred_top_kind_counts.update([pred_kind])
        self.transition_counts.update([f"{teacher_kind}->{pred_kind}"])
        if potion_pair:
            self.potion_pair_roots += 1
            self.potion_pair_sign_correct += int(potion_pair_sign_correct)
            self.teacher_prefers_potion_roots += int(teacher_prefers_potion)
            self.pred_prefers_potion_roots += int(pred_prefers_potion)
        self.missed_teacher_potion_roots += int(teacher_kind == "potion" and pred_kind != "potion")
        self.false_potion_top_roots += int(teacher_kind != "potion" and pred_kind == "potion")
        if baseline_top1 is not None:
            self.baseline_top1 += int(baseline_top1)
            self.baseline_wrong_roots += int(not baseline_top1)
            self.model_recovers_baseline_wrong += int((not baseline_top1) and top1)
            self.model_breaks_baseline_correct += int(baseline_top1 and (not top1))

    def finalize(self) -> dict[str, Any]:
        roots = max(1, self.roots)
        potion_pair_roots = max(1, self.potion_pair_roots)
        baseline_wrong_roots = max(1, self.baseline_wrong_roots)
        baseline_correct_roots = max(1, self.roots - self.baseline_wrong_roots)
        return {
            "roots": self.roots,
            "candidates": self.candidates,
            "avg_candidates": self.candidates / roots,
            "top1_accuracy": self.top1 / roots,
            "mean_regret": self.regret_sum / roots,
            "rmse_regret": (self.regret_sq_sum / roots) ** 0.5,
            "mean_teacher_gap": self.gap_sum / roots,
            "mean_pred_margin": self.pred_margin_sum / roots,
            "gap_weighted_regret": self.gap_weighted_regret_sum / roots,
            "mean_teacher_top_q": self.teacher_top_q_sum / roots,
            "mean_pred_choice_teacher_q": self.pred_choice_teacher_q_sum / roots,
            "high_conf_disagreement_rate": self.high_conf_disagreements / roots,
            "teacher_top_kind_counts": dict(self.teacher_top_kind_counts),
            "pred_top_kind_counts": dict(self.pred_top_kind_counts),
            "teacher_to_pred_kind_counts": dict(self.transition_counts),
            "potion_pair_roots": self.potion_pair_roots,
            "potion_pair_sign_accuracy": None if self.potion_pair_roots <= 0 else self.potion_pair_sign_correct / potion_pair_roots,
            "teacher_prefers_potion_rate_on_pair": None
            if self.potion_pair_roots <= 0
            else self.teacher_prefers_potion_roots / potion_pair_roots,
            "pred_prefers_potion_rate_on_pair": None
            if self.potion_pair_roots <= 0
            else self.pred_prefers_potion_roots / potion_pair_roots,
            "missed_teacher_potion_rate": self.missed_teacher_potion_roots / roots,
            "false_potion_top_rate": self.false_potion_top_roots / roots,
            "baseline_top1_accuracy": None if self.baseline_top1 == 0 and self.baseline_wrong_roots == 0 else self.baseline_top1 / roots,
            "baseline_wrong_roots": self.baseline_wrong_roots,
            "model_recovers_baseline_wrong_rate": None
            if self.baseline_top1 == 0 and self.baseline_wrong_roots == 0
            else self.model_recovers_baseline_wrong / baseline_wrong_roots,
            "model_breaks_baseline_correct_rate": None
            if self.baseline_top1 == 0 and self.baseline_wrong_roots == 0
            else self.model_breaks_baseline_correct / baseline_correct_roots,
        }


def _default_device(requested: str) -> str:
    if requested != "auto":
        return requested
    torch = require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_lines(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _validation_chunks(
    payload: dict[str, Any],
    *,
    seed: int,
    validation_fraction: float,
    validation_source_shards_file: Path | None,
) -> list[dict[str, Any]]:
    chunks = list(payload.get("chunks") or [])
    if not chunks:
        raise ValueError("hard validation v0 expects a chunked tensor dataset manifest")
    validation_sources = _read_lines(validation_source_shards_file)
    if validation_sources:
        validation_chunks = [chunk for chunk in chunks if str(chunk.get("source_shard") or "") in validation_sources]
        if not validation_chunks:
            raise ValueError(f"no validation chunks matched {validation_source_shards_file}")
        return validation_chunks
    rng = random.Random(int(seed))
    rng.shuffle(chunks)
    val_count = int(len(chunks) * float(validation_fraction))
    if val_count <= 0:
        raise ValueError("validation split is empty; increase --validation-fraction or use --validation-source-shards-file")
    return chunks[:val_count]


def _root_batches(root_count: int, batch_roots: int) -> list[list[int]]:
    indices = list(range(int(root_count)))
    size = max(1, int(batch_roots))
    return [indices[start : start + size] for start in range(0, len(indices), size)]


def _action_kinds_from_features(batch: dict[str, Any], state_dim: int) -> list[str]:
    features = batch["features"].detach().cpu()
    end_flag = features[:, state_dim + 0] > 0.5
    card_flag = features[:, state_dim + 1] > 0.5
    potion_flag = features[:, state_dim + 2] > 0.5
    kinds: list[str] = []
    for is_end, is_card, is_potion in zip(end_flag.tolist(), card_flag.tolist(), potion_flag.tolist(), strict=False):
        if is_potion:
            kinds.append("potion")
        elif is_card:
            kinds.append("card")
        elif is_end:
            kinds.append("end")
        else:
            kinds.append("other")
    return kinds


def _top2_gap(values: list[float]) -> tuple[int, float, float]:
    if not values:
        raise ValueError("empty root values")
    best_index = max(range(len(values)), key=values.__getitem__)
    if len(values) == 1:
        return best_index, 0.0, 0.0
    second = max(value for index, value in enumerate(values) if index != best_index)
    return best_index, float(values[best_index] - second), float(second)


def _rank_desc(values: list[float], index: int) -> int:
    target = values[index]
    return 1 + sum(1 for value in values if value > target)


def _best_index(indices: list[int], values: list[float]) -> int:
    return max(indices, key=lambda index: values[index])


def _bucket_key(prefix: str, value: str | int | float) -> str:
    return f"{prefix}:{value}"


def _new_buckets() -> dict[str, BucketStats]:
    return {"overall": BucketStats()}


def _update_bucket(buckets: dict[str, BucketStats], key: str, record: dict[str, Any]) -> None:
    buckets.setdefault(key, BucketStats()).update(**record)


def _format_brief(metrics: dict[str, Any]) -> str:
    return (
        f"roots={metrics['roots']} "
        f"top1={metrics['top1_accuracy']:.4f} "
        f"regret={metrics['mean_regret']:.4f} "
        f"gap={metrics['mean_teacher_gap']:.4f} "
        f"high_conf_bad={metrics['high_conf_disagreement_rate']:.4f}"
    )


def _root_categories(
    *,
    top1: bool,
    regret: float,
    teacher_gap: float,
    pred_margin: float,
    high_conf_disagreement: bool,
    teacher_kind: str,
    pred_kind: str,
    potion_pair: bool,
    teacher_prefers_potion: bool,
    pred_prefers_potion: bool,
    baseline_top1: bool | None,
) -> list[str]:
    categories: list[str] = []
    if top1:
        categories.append("top1_correct")
    else:
        categories.append("top1_wrong")
    if high_conf_disagreement:
        categories.append("high_conf_disagreement")
    if regret >= 25.0:
        categories.append("regret_ge_25")
    elif regret >= 10.0:
        categories.append("regret_ge_10")
    elif regret >= 5.0:
        categories.append("regret_ge_5")
    if teacher_gap >= 5.0 and not top1:
        categories.append("high_teacher_gap_miss_ge_5")
    elif teacher_gap >= 2.0 and not top1:
        categories.append("high_teacher_gap_miss_ge_2")
    if pred_margin >= 1.0 and not top1:
        categories.append("pred_margin_ge_1_wrong")
    if teacher_kind == "potion" and pred_kind != "potion":
        categories.append("missed_teacher_potion")
    if teacher_kind != "potion" and pred_kind == "potion":
        categories.append("false_potion_top")
    if potion_pair and teacher_prefers_potion and not pred_prefers_potion:
        categories.append("potion_pair_wrong_against_potion")
    if potion_pair and (not teacher_prefers_potion) and pred_prefers_potion:
        categories.append("potion_pair_wrong_toward_potion")
    if teacher_kind == "end" and pred_kind != "end":
        categories.append("missed_end")
    if teacher_kind != "end" and pred_kind == "end":
        categories.append("false_end_top")
    if teacher_kind == "card" and pred_kind == "card" and not top1:
        categories.append("card_to_card_mismatch")
    if baseline_top1 is not None:
        if (not baseline_top1) and top1:
            categories.append("baseline_wrong_recovered")
        if baseline_top1 and (not top1):
            categories.append("baseline_correct_broken")
    return categories


def _checkpoint_schema_info(model: Any, checkpoint: dict[str, Any], dataset_payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint_entity_vocab_len = len(checkpoint.get("entity_vocab") or [])
    dataset_entity_vocab_len = len(dataset_payload.get("entity_vocab") or [])
    model_entity_vocab_size = int(getattr(model, "entity_vocab_size", 0) or 0)
    token_schema = dict(checkpoint.get("token_schema") or {})
    dataset_token_schema = dict(dataset_payload.get("token_schema") or {})
    return {
        "checkpoint_entity_vocab_len": checkpoint_entity_vocab_len,
        "model_entity_vocab_size": model_entity_vocab_size,
        "dataset_entity_vocab_len": dataset_entity_vocab_len,
        "checkpoint_vocab_matches_model": checkpoint_entity_vocab_len == model_entity_vocab_size,
        "dataset_vocab_matches_model": dataset_entity_vocab_len == model_entity_vocab_size,
        "checkpoint_token_schema_version": token_schema.get("version") or checkpoint.get("token_schema_version"),
        "dataset_token_schema_version": dataset_token_schema.get("version") or dataset_payload.get("token_schema_version"),
        "checkpoint_token_schema": token_schema,
        "dataset_token_schema": dataset_token_schema,
    }


def _predict(model: Any, batch: dict[str, Any], *, device: str, amp_dtype: Any | None = None) -> Any:
    torch = require_torch()
    with torch.inference_mode():
        if amp_dtype is None or not str(device).startswith("cuda"):
            return model(batch).float()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            return model(batch).float()


def _resolve_amp_dtype(torch: Any, device: str, amp_dtype: str) -> Any | None:
    if not str(device).startswith("cuda") or amp_dtype == "none":
        return None
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype != "auto":
        raise ValueError(f"unsupported amp dtype: {amp_dtype}")
    try:
        if bool(torch.cuda.is_bf16_supported()):
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _set_cuda_runtime(torch: Any, *, device: str, allow_tf32: bool) -> None:
    if not str(device).startswith("cuda"):
        return
    if allow_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def _evaluate_chunk(
    *,
    model: Any,
    baseline_model: Any | None,
    chunk_payload: dict[str, Any],
    chunk_meta: dict[str, Any],
    device: str,
    batch_roots: int,
    state_dim: int,
    gap_thresholds: list[float],
    high_conf_margin: float,
    worst_records: list[dict[str, Any]],
    hard_records: list[dict[str, Any]],
    category_counts: Counter[str],
    transition_category_counts: Counter[str],
    max_worst_records: int,
    amp_dtype: Any | None,
) -> dict[str, BucketStats]:
    torch = require_torch()
    buckets = _new_buckets()
    root_count = int(chunk_payload["metadata"]["root_count"])
    root_ids = list(chunk_payload.get("root_ids") or [str(index) for index in range(root_count)])
    source_shards = list(chunk_payload.get("source_shards") or [chunk_meta.get("source_shard") for _ in range(root_count)])
    for batch_roots_list in _root_batches(root_count, batch_roots):
        batch = _tensor_batch(chunk_payload, batch_roots_list, device=device)
        pred_q = _predict(model, batch, device=device, amp_dtype=amp_dtype).detach().cpu()
        baseline_pred_q = None
        if baseline_model is not None:
            baseline_pred_q = _predict(baseline_model, batch, device=device, amp_dtype=amp_dtype).detach().cpu()
        teacher_q = batch["teacher_q"].detach().cpu()
        counts = batch["candidate_counts"].detach().cpu().tolist()
        room_type_ids = batch.get("room_type_ids")
        room_type_ids_cpu = room_type_ids.detach().cpu().tolist() if room_type_ids is not None else None
        action_is_potion = batch.get("action_is_potion")
        potion_flags = action_is_potion.detach().cpu().tolist() if action_is_potion is not None else None
        action_kinds = _action_kinds_from_features(batch, state_dim)
        offset = 0
        for local_batch_index, count_value in enumerate(counts):
            count = int(count_value)
            root_slice = slice(offset, offset + count)
            teacher_values = [float(value) for value in teacher_q[root_slice].tolist()]
            pred_values = [float(value) for value in pred_q[root_slice].tolist()]
            baseline_values = [float(value) for value in baseline_pred_q[root_slice].tolist()] if baseline_pred_q is not None else None
            local_kinds = action_kinds[offset : offset + count]
            local_potion_flags = (
                [bool(value) for value in potion_flags[offset : offset + count]]
                if potion_flags is not None
                else [kind == "potion" for kind in local_kinds]
            )
            local_room_ids = room_type_ids_cpu[offset : offset + count] if room_type_ids_cpu is not None else [0] * count
            root_room_id = int(local_room_ids[0]) if local_room_ids else 0
            room_name = ROOM_TYPE_NAMES.get(root_room_id, f"room_{root_room_id}")
            teacher_top, teacher_gap, _teacher_second = _top2_gap(teacher_values)
            pred_top, pred_margin, _pred_second = _top2_gap(pred_values)
            top1 = pred_top == teacher_top
            teacher_top_q = float(teacher_values[teacher_top])
            pred_choice_teacher_q = float(teacher_values[pred_top])
            regret = max(0.0, teacher_top_q - pred_choice_teacher_q)
            teacher_kind = local_kinds[teacher_top]
            pred_kind = local_kinds[pred_top]
            high_conf_disagreement = (not top1) and pred_margin >= float(high_conf_margin)
            potion_indices = [index for index, flag in enumerate(local_potion_flags) if flag]
            non_potion_indices = [index for index, flag in enumerate(local_potion_flags) if not flag]
            potion_pair = bool(potion_indices and non_potion_indices)
            potion_pair_sign_correct = False
            teacher_prefers_potion = False
            pred_prefers_potion = False
            if potion_pair:
                teacher_best_potion = _best_index(potion_indices, teacher_values)
                teacher_best_non_potion = _best_index(non_potion_indices, teacher_values)
                pred_best_potion = _best_index(potion_indices, pred_values)
                pred_best_non_potion = _best_index(non_potion_indices, pred_values)
                teacher_potion_gap = teacher_values[teacher_best_potion] - teacher_values[teacher_best_non_potion]
                pred_potion_gap = pred_values[pred_best_potion] - pred_values[pred_best_non_potion]
                teacher_prefers_potion = teacher_potion_gap > 0.0
                pred_prefers_potion = pred_potion_gap > 0.0
                potion_pair_sign_correct = (teacher_potion_gap == 0.0 and pred_potion_gap == 0.0) or (
                    teacher_potion_gap > 0.0
                ) == (pred_potion_gap > 0.0)
            baseline_top1 = None
            if baseline_values is not None:
                baseline_top = max(range(len(baseline_values)), key=baseline_values.__getitem__)
                baseline_top1 = baseline_top == teacher_top
            categories = _root_categories(
                top1=top1,
                regret=regret,
                teacher_gap=teacher_gap,
                pred_margin=pred_margin,
                high_conf_disagreement=high_conf_disagreement,
                teacher_kind=teacher_kind,
                pred_kind=pred_kind,
                potion_pair=potion_pair,
                teacher_prefers_potion=teacher_prefers_potion,
                pred_prefers_potion=pred_prefers_potion,
                baseline_top1=baseline_top1,
            )
            category_counts.update(categories)
            transition_category_counts.update([f"{teacher_kind}->{pred_kind}:{category}" for category in categories])
            record = {
                "candidate_count": count,
                "top1": top1,
                "regret": regret,
                "teacher_gap": teacher_gap,
                "pred_margin": pred_margin,
                "high_conf_disagreement": high_conf_disagreement,
                "teacher_top_q": teacher_top_q,
                "pred_choice_teacher_q": pred_choice_teacher_q,
                "teacher_kind": teacher_kind,
                "pred_kind": pred_kind,
                "potion_pair": potion_pair,
                "potion_pair_sign_correct": potion_pair_sign_correct,
                "teacher_prefers_potion": teacher_prefers_potion,
                "pred_prefers_potion": pred_prefers_potion,
                "baseline_top1": baseline_top1,
            }
            root_bucket_keys = [
                "overall",
                _bucket_key("room", room_name),
                _bucket_key("teacher_top_kind", teacher_kind),
                _bucket_key("pred_top_kind", pred_kind),
                _bucket_key("candidate_count", min(count, 10)),
            ]
            if potion_pair:
                root_bucket_keys.append("potion_pair")
                root_bucket_keys.append(
                    "potion_pair:teacher_prefers_potion" if teacher_prefers_potion else "potion_pair:teacher_prefers_non_potion"
                )
            if any(local_potion_flags):
                root_bucket_keys.append("has_potion_candidate")
            for threshold in gap_thresholds:
                if teacher_gap >= threshold:
                    root_bucket_keys.append(_bucket_key("teacher_gap_ge", threshold))
            if high_conf_disagreement:
                root_bucket_keys.append("high_conf_disagreement")
            if baseline_top1 is not None:
                root_bucket_keys.append("baseline_correct" if baseline_top1 else "baseline_wrong")
            for key in root_bucket_keys:
                _update_bucket(buckets, key, record)
            global_root_index = int(batch_roots_list[local_batch_index])
            root_record = {
                "regret": regret,
                "teacher_gap": teacher_gap,
                "pred_margin": pred_margin,
                "teacher_rank_of_pred": _rank_desc(teacher_values, pred_top),
                "pred_rank_of_teacher": _rank_desc(pred_values, teacher_top),
                "root_id": root_ids[global_root_index] if global_root_index < len(root_ids) else str(global_root_index),
                "source_shard": source_shards[global_root_index] if global_root_index < len(source_shards) else None,
                "chunk_path": chunk_meta.get("path"),
                "local_root_index": global_root_index,
                "candidate_count": count,
                "room_type": room_name,
                "teacher_top_index": teacher_top,
                "pred_top_index": pred_top,
                "teacher_top_kind": teacher_kind,
                "pred_top_kind": pred_kind,
                "teacher_top_q": teacher_top_q,
                "pred_choice_teacher_q": pred_choice_teacher_q,
                "top1": top1,
                "has_potion_candidate": any(local_potion_flags),
                "potion_pair": potion_pair,
                "teacher_prefers_potion": teacher_prefers_potion if potion_pair else None,
                "pred_prefers_potion": pred_prefers_potion if potion_pair else None,
                "baseline_top1": baseline_top1,
                "categories": categories,
            }
            if (not top1) or high_conf_disagreement:
                hard_records.append(root_record)
            if regret > 0.0 or high_conf_disagreement:
                worst_record = dict(root_record)
                worst_records.append(worst_record)
                if len(worst_records) > max_worst_records * 4:
                    worst_records.sort(key=lambda item: (float(item["regret"]), float(item["teacher_gap"])), reverse=True)
                    del worst_records[max_worst_records:]
            offset += count
        del batch, pred_q, teacher_q
        if baseline_pred_q is not None:
            del baseline_pred_q
    return buckets


def _merge_stats(target: dict[str, BucketStats], source: dict[str, BucketStats]) -> None:
    for key, stats in source.items():
        bucket = target.setdefault(key, BucketStats())
        bucket.roots += stats.roots
        bucket.candidates += stats.candidates
        bucket.top1 += stats.top1
        bucket.regret_sum += stats.regret_sum
        bucket.regret_sq_sum += stats.regret_sq_sum
        bucket.gap_sum += stats.gap_sum
        bucket.pred_margin_sum += stats.pred_margin_sum
        bucket.gap_weighted_regret_sum += stats.gap_weighted_regret_sum
        bucket.teacher_top_q_sum += stats.teacher_top_q_sum
        bucket.pred_choice_teacher_q_sum += stats.pred_choice_teacher_q_sum
        bucket.high_conf_disagreements += stats.high_conf_disagreements
        bucket.teacher_top_kind_counts.update(stats.teacher_top_kind_counts)
        bucket.pred_top_kind_counts.update(stats.pred_top_kind_counts)
        bucket.transition_counts.update(stats.transition_counts)
        bucket.potion_pair_roots += stats.potion_pair_roots
        bucket.potion_pair_sign_correct += stats.potion_pair_sign_correct
        bucket.teacher_prefers_potion_roots += stats.teacher_prefers_potion_roots
        bucket.pred_prefers_potion_roots += stats.pred_prefers_potion_roots
        bucket.missed_teacher_potion_roots += stats.missed_teacher_potion_roots
        bucket.false_potion_top_roots += stats.false_potion_top_roots
        bucket.baseline_top1 += stats.baseline_top1
        bucket.baseline_wrong_roots += stats.baseline_wrong_roots
        bucket.model_recovers_baseline_wrong += stats.model_recovers_baseline_wrong
        bucket.model_breaks_baseline_correct += stats.model_breaks_baseline_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard validation v0 for v3 combat transformer checkpoints.")
    parser.add_argument(
        "--tensor-dataset",
        type=Path,
        default=Path("data/v3_combat_tensor/transformer_stage5_v8_potion_pair_200k.pt"),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["auto", "none", "bfloat16", "float16"], default="none")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=11, help="Must match training split seed for comparable validation chunks.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-source-shards-file", type=Path, default=None)
    parser.add_argument("--batch-roots", type=int, default=256)
    parser.add_argument("--limit-chunks", type=int, default=0)
    parser.add_argument("--limit-roots-per-chunk", type=int, default=0)
    parser.add_argument("--gap-thresholds", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--high-conf-margin", type=float, default=1.0)
    parser.add_argument("--max-worst-records", type=int, default=1000)
    parser.add_argument("--progress-interval-chunks", type=int, default=10)
    args = parser.parse_args()

    torch = require_torch()
    device = _default_device(str(args.device))
    _set_cuda_runtime(torch, device=device, allow_tf32=bool(args.allow_tf32))
    amp_dtype = _resolve_amp_dtype(torch, device, str(args.amp_dtype))
    payload = _load_tensor_dataset(args.tensor_dataset)
    if not _is_chunked_payload(payload):
        raise SystemExit("hard validation v0 currently expects chunked tensor datasets.")
    if _is_root_payload(payload):
        raise SystemExit("this v0 script currently expects candidate-level tensor chunks; use a candidate checkpoint/dataset.")
    feature_dims = dict(payload.get("feature_dims") or {})
    state_dim = int(feature_dims.get("state_dim") or 0)
    if state_dim <= 0:
        raise SystemExit("tensor dataset is missing feature_dims.state_dim")
    validation_chunks = _validation_chunks(
        payload,
        seed=int(args.seed),
        validation_fraction=float(args.validation_fraction),
        validation_source_shards_file=args.validation_source_shards_file,
    )
    if int(args.limit_chunks) > 0:
        validation_chunks = validation_chunks[: int(args.limit_chunks)]
    if not validation_chunks:
        raise SystemExit("no validation chunks selected")
    gap_thresholds = [float(value.strip()) for value in str(args.gap_thresholds).split(",") if value.strip()]
    model, checkpoint = load_v3_combat_transformer_checkpoint(args.model, device=device)
    if bool(getattr(model, "expects_root_batch", False)):
        raise SystemExit(f"{args.model} is a root-action-set checkpoint; this v0 run uses candidate-level tensor chunks.")
    model_schema_info = _checkpoint_schema_info(model, checkpoint, payload)
    if not bool(model_schema_info["checkpoint_vocab_matches_model"]):
        print(
            "[hard-val-v0] warning: model checkpoint entity_vocab length does not match model embedding size; "
            "offline tensor validation can still run, but runtime replay would use the bad checkpoint vocab.",
            flush=True,
        )
    baseline_model = None
    baseline_checkpoint: dict[str, Any] | None = None
    baseline_schema_info: dict[str, Any] | None = None
    if args.baseline_model is not None:
        baseline_model, baseline_checkpoint = load_v3_combat_transformer_checkpoint(args.baseline_model, device=device)
        if bool(getattr(baseline_model, "expects_root_batch", False)):
            raise SystemExit(f"{args.baseline_model} is a root-action-set checkpoint; baseline must match candidate-level chunks.")
        baseline_schema_info = _checkpoint_schema_info(baseline_model, baseline_checkpoint, payload)
        if not bool(baseline_schema_info["checkpoint_vocab_matches_model"]):
            print(
                "[hard-val-v0] warning: baseline checkpoint entity_vocab length does not match model embedding size; "
                "baseline offline tensor validation can still run, but runtime replay would use the bad checkpoint vocab.",
                flush=True,
            )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("diagnostics/hard_validation_v0") / args.model.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = _new_buckets()
    worst_records: list[dict[str, Any]] = []
    hard_records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    transition_category_counts: Counter[str] = Counter()
    started_at = time.time()
    processed_roots = 0
    processed_candidates = 0
    for chunk_index, chunk in enumerate(validation_chunks, start=1):
        chunk_payload = _load_chunk(chunk)
        if int(args.limit_roots_per_chunk) > 0:
            root_limit = min(int(args.limit_roots_per_chunk), int(chunk_payload["metadata"]["root_count"]))
            original_offsets = chunk_payload["candidate_offsets"]
            candidate_limit = int(original_offsets[root_limit].item())
            chunk_payload = dict(chunk_payload)
            chunk_payload["candidate_offsets"] = original_offsets[: root_limit + 1].contiguous()
            chunk_payload["token_scalar_features"] = chunk_payload["token_scalar_features"][:candidate_limit].contiguous()
            chunk_payload["token_type_ids"] = chunk_payload["token_type_ids"][:candidate_limit].contiguous()
            chunk_payload["entity_ids"] = chunk_payload["entity_ids"][:candidate_limit].contiguous()
            chunk_payload["slot_ids"] = chunk_payload["slot_ids"][:candidate_limit].contiguous()
            chunk_payload["attention_mask"] = chunk_payload["attention_mask"][:candidate_limit].contiguous()
            chunk_payload["features"] = chunk_payload["features"][:candidate_limit].contiguous()
            chunk_payload["teacher_q"] = chunk_payload["teacher_q"][:candidate_limit].contiguous()
            chunk_payload["chosen"] = chunk_payload["chosen"][:candidate_limit].contiguous()
            if "action_is_potion" in chunk_payload:
                chunk_payload["action_is_potion"] = chunk_payload["action_is_potion"][:candidate_limit].contiguous()
            if "room_type_ids" in chunk_payload:
                chunk_payload["room_type_ids"] = chunk_payload["room_type_ids"][:candidate_limit].contiguous()
            chunk_payload["root_ids"] = list(chunk_payload.get("root_ids") or [])[:root_limit]
            chunk_payload["source_shards"] = list(chunk_payload.get("source_shards") or [])[:root_limit]
            chunk_payload["metadata"] = dict(chunk_payload["metadata"])
            chunk_payload["metadata"]["root_count"] = root_limit
            chunk_payload["metadata"]["candidate_count"] = candidate_limit
        chunk_buckets = _evaluate_chunk(
            model=model,
            baseline_model=baseline_model,
            chunk_payload=chunk_payload,
            chunk_meta=chunk,
            device=device,
            batch_roots=int(args.batch_roots),
            state_dim=state_dim,
            gap_thresholds=gap_thresholds,
            high_conf_margin=float(args.high_conf_margin),
            worst_records=worst_records,
            hard_records=hard_records,
            category_counts=category_counts,
            transition_category_counts=transition_category_counts,
            max_worst_records=int(args.max_worst_records),
            amp_dtype=amp_dtype,
        )
        _merge_stats(buckets, chunk_buckets)
        processed_roots += int(chunk_payload["metadata"]["root_count"])
        processed_candidates += int(chunk_payload["metadata"].get("candidate_count") or 0)
        if int(args.progress_interval_chunks) > 0 and (
            chunk_index == 1 or chunk_index % int(args.progress_interval_chunks) == 0 or chunk_index == len(validation_chunks)
        ):
            elapsed = max(1e-6, time.time() - started_at)
            overall = buckets["overall"].finalize()
            print(
                "[hard-val-v0] "
                f"chunks={chunk_index}/{len(validation_chunks)} "
                f"roots={processed_roots} "
                f"candidates={processed_candidates} "
                f"roots/s={processed_roots / elapsed:.1f} "
                f"{_format_brief(overall)}",
                flush=True,
            )
    finalized = {key: stats.finalize() for key, stats in sorted(buckets.items())}
    worst_records.sort(key=lambda item: (float(item["regret"]), float(item["teacher_gap"])), reverse=True)
    worst_records = worst_records[: int(args.max_worst_records)]
    hard_records.sort(key=lambda item: (str(item["room_type"]), str(item["teacher_top_kind"]), -float(item["regret"])))
    hard_summary = {
        "schema": "v3_combat_hard_validation_v0_hard_root_summary",
        "hard_root_count": len(hard_records),
        "category_counts": dict(category_counts),
        "transition_category_counts": dict(transition_category_counts),
        "top_transition_categories": dict(transition_category_counts.most_common(50)),
    }
    summary = {
        "schema": "v3_combat_hard_validation_v0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tensor_dataset": str(args.tensor_dataset),
        "model": str(args.model),
        "baseline_model": None if args.baseline_model is None else str(args.baseline_model),
        "device": device,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
        "validation_seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "validation_source_shards_file": None
        if args.validation_source_shards_file is None
        else str(args.validation_source_shards_file),
        "validation_chunks": len(validation_chunks),
        "processed_roots": processed_roots,
        "processed_candidates": processed_candidates,
        "gap_thresholds": gap_thresholds,
        "high_conf_margin": float(args.high_conf_margin),
        "model_config": dict(checkpoint.get("model_config") or {}),
        "baseline_model_config": None if baseline_checkpoint is None else dict(baseline_checkpoint.get("model_config") or {}),
        "model_schema": model_schema_info,
        "baseline_model_schema": baseline_schema_info,
        "dataset_metadata": dict(payload.get("metadata") or {}),
        "metrics": finalized,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "hard_root_summary.json").write_text(
        json.dumps(hard_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "worst_roots.jsonl").open("w", encoding="utf-8") as handle:
        for record in worst_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "hard_roots.jsonl").open("w", encoding="utf-8") as handle:
        for record in hard_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[hard-val-v0] wrote {output_dir / 'summary.json'}", flush=True)
    print(f"[hard-val-v0] wrote {output_dir / 'hard_root_summary.json'}", flush=True)
    print(f"[hard-val-v0] wrote {output_dir / 'worst_roots.jsonl'}", flush=True)
    print(f"[hard-val-v0] wrote {output_dir / 'hard_roots.jsonl'}", flush=True)
    print(f"[hard-val-v0] overall {_format_brief(finalized['overall'])}", flush=True)


if __name__ == "__main__":
    main()
