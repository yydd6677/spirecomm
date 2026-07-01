#!/usr/bin/env python3
from __future__ import annotations
# Allow this CLI to run directly from its workflow subdirectory.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import contextlib
import copy
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_dataset import REWARD_COMPONENT_DIM, total_candidate_loss
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION
from spirecomm.ai.v3_combat_model import load_v3_combat_checkpoint
from spirecomm.ai.v3_combat_transformer import (
    CHECKPOINT_VERSION,
    CANDIDATE_HEAD_VARIANT_BASE,
    CANDIDATE_HEAD_VARIANTS,
    ROOT_TOKEN_SCHEMA_VERSION,
    ROOT_HEAD_VARIANTS,
    ROOT_HEAD_VARIANT_BASE,
    ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    SUPPORTED_CANDIDATE_TOKEN_SCHEMA_VERSIONS,
    TOKEN_SCHEMA_VERSION,
    TOKEN_TYPES,
    TRANSFORMER_TENSOR_DATASET_SCHEMA,
    V3CombatRootActionSetTransformerScorer,
    V3CombatTransformerCandidateScorer,
    load_v3_combat_transformer_checkpoint,
    save_v3_combat_transformer_checkpoint,
)


CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA = "v3_combat_transformer_tensor_dataset_chunked_v1"
CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA = "v3_combat_root_transformer_tensor_dataset_chunked_v1"


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def _mem_available_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _rss_kb() -> int | None:
    try:
        status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _memory_snapshot(torch_module=None, *, device: str = "cpu", **extra) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time": time.time(),
        "pid": os.getpid(),
        "rss_kb": _rss_kb(),
        "mem_available_kb": _mem_available_kb(),
    }
    if torch_module is not None and str(device).startswith("cuda") and torch_module.cuda.is_available():
        try:
            index = torch_module.cuda.current_device()
            payload.update(
                {
                    "cuda_device": int(index),
                    "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated(index)),
                    "cuda_reserved_bytes": int(torch_module.cuda.memory_reserved(index)),
                }
            )
        except Exception as exc:
            payload["cuda_error"] = str(exc)
    payload.update(extra)
    return payload


def _append_memory_log(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _optimizer_to_device(optimizer, device: str) -> None:
    torch = require_torch()
    target = torch.device(device)
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target)


def _parse_name_patterns(raw: str) -> list[str]:
    return [pattern.strip() for pattern in str(raw or "").split(",") if pattern.strip()]


def _apply_trainable_name_patterns(model: Any, raw: str) -> dict[str, Any]:
    patterns = _parse_name_patterns(raw)
    total_tensors = 0
    trainable_tensors = 0
    frozen_tensors = 0
    trainable_params = 0
    frozen_params = 0
    matched_names: list[str] = []
    for name, parameter in model.named_parameters():
        total_tensors += 1
        trainable = bool(patterns) and any(pattern in name for pattern in patterns)
        if patterns:
            parameter.requires_grad_(trainable)
        else:
            trainable = bool(parameter.requires_grad)
        if trainable:
            trainable_tensors += 1
            trainable_params += int(parameter.numel())
            matched_names.append(name)
        else:
            frozen_tensors += 1
            frozen_params += int(parameter.numel())
    if patterns and not matched_names:
        raise SystemExit(f"--trainable-name-patterns matched no parameters: {patterns}")
    return {
        "trainable_name_patterns": patterns,
        "parameter_tensors_total": total_tensors,
        "trainable_parameter_tensors": trainable_tensors,
        "frozen_parameter_tensors": frozen_tensors,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "matched_parameter_names": matched_names,
    }


def _configure_torch_runtime(torch_module, *, device: str, allow_tf32: bool) -> None:
    if not str(device).startswith("cuda"):
        return
    if allow_tf32:
        try:
            torch_module.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch_module.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch_module.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def _resolve_amp_dtype(torch_module, *, device: str, amp_dtype: str):
    if not str(device).startswith("cuda") or amp_dtype == "none":
        return None
    if amp_dtype == "bfloat16":
        return torch_module.bfloat16
    if amp_dtype == "float16":
        return torch_module.float16
    if amp_dtype != "auto":
        raise ValueError(f"unsupported amp dtype: {amp_dtype}")
    try:
        if bool(torch_module.cuda.is_bf16_supported()):
            return torch_module.bfloat16
    except Exception:
        pass
    return torch_module.float16


def _autocast_context(torch_module, *, device: str, amp_dtype):
    if amp_dtype is None or not str(device).startswith("cuda"):
        return contextlib.nullcontext()
    try:
        return torch_module.autocast(device_type="cuda", dtype=amp_dtype)
    except TypeError:
        return torch_module.cuda.amp.autocast(dtype=amp_dtype)


def _make_grad_scaler(torch_module, *, device: str, amp_dtype):
    enabled = bool(str(device).startswith("cuda") and amp_dtype == torch_module.float16)
    if not enabled:
        return None
    try:
        return torch_module.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch_module.cuda.amp.GradScaler(enabled=True)


def _load_tensor_dataset(path: Path) -> dict[str, Any]:
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("tensor_dataset_schema") not in {
        TRANSFORMER_TENSOR_DATASET_SCHEMA,
        CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
        ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
        CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    }:
        raise ValueError(f"unsupported v3 combat transformer tensor dataset schema in {path}: {payload.get('tensor_dataset_schema')}")
    if payload.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "v3 combat transformer tensor dataset feature schema mismatch in "
            f"{path}: {payload.get('feature_schema')} != {FEATURE_SCHEMA_VERSION}"
        )
    if payload.get("token_schema_version") not in {*SUPPORTED_CANDIDATE_TOKEN_SCHEMA_VERSIONS, ROOT_TOKEN_SCHEMA_VERSION}:
        raise ValueError(
            "v3 combat transformer token dataset mismatch in "
            f"{path}: {payload.get('token_schema_version')} not in "
            f"{sorted([*SUPPORTED_CANDIDATE_TOKEN_SCHEMA_VERSIONS, ROOT_TOKEN_SCHEMA_VERSION])}"
        )
    return payload


def _attach_tensor_dataset_metadata(model: Any, payload: dict[str, Any]) -> None:
    """Keep checkpoint runtime encoders aligned with the tensor cache used for training."""
    feature_dims = payload.get("feature_dims")
    if isinstance(feature_dims, dict) and feature_dims:
        setattr(model, "checkpoint_feature_schema", dict(feature_dims))
    token_schema = payload.get("token_schema")
    if isinstance(token_schema, dict) and token_schema:
        setattr(model, "checkpoint_token_schema", dict(token_schema))
    entity_vocab = payload.get("entity_vocab")
    if entity_vocab:
        setattr(model, "checkpoint_entity_vocab", list(entity_vocab))


def _is_chunked_payload(payload: dict[str, Any]) -> bool:
    return payload.get("tensor_dataset_schema") in {
        CHUNKED_TRANSFORMER_TENSOR_DATASET_SCHEMA,
        CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    }


def _is_root_payload(payload: dict[str, Any]) -> bool:
    return payload.get("tensor_dataset_schema") in {
        ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
        CHUNKED_ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA,
    }


def _load_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    payload = torch.load(Path(chunk["path"]), map_location="cpu", weights_only=False)
    if payload.get("tensor_dataset_schema") not in {TRANSFORMER_TENSOR_DATASET_SCHEMA, ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA}:
        raise ValueError(f"unsupported v3 combat transformer chunk schema in {chunk['path']}: {payload.get('tensor_dataset_schema')}")
    return payload


def _root_active_lengths(payload: dict[str, Any]) -> list[int] | None:
    if not _is_root_payload(payload) or "attention_mask" not in payload:
        return None
    mask = payload["attention_mask"]
    return [int(value) for value in mask.to(dtype=require_torch().long).sum(dim=1).tolist()]


def _root_batches(
    payload: dict[str, Any],
    root_indices: list[int],
    *,
    batch_size: int,
    rng: random.Random,
    shuffle: bool,
    length_bucket_batches: bool,
    length_bucket_window: int,
) -> list[list[int]]:
    batch_size = max(1, int(batch_size))
    ordered = list(root_indices)
    if shuffle:
        rng.shuffle(ordered)
    if not length_bucket_batches:
        return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]
    lengths = _root_active_lengths(payload)
    if lengths is None:
        return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]

    window_size = max(batch_size, batch_size * max(1, int(length_bucket_window)))
    batches: list[list[int]] = []
    for start in range(0, len(ordered), window_size):
        window = ordered[start : start + window_size]
        window.sort(key=lambda index: (lengths[int(index)], int(index)))
        local_batches = [window[pos : pos + batch_size] for pos in range(0, len(window), batch_size)]
        if shuffle:
            rng.shuffle(local_batches)
        batches.extend(local_batches)
    if shuffle:
        rng.shuffle(batches)
    return batches


def _read_lines(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _move_payload_tensors(payload: dict[str, Any], *, device: str) -> dict[str, Any]:
    torch = require_torch()
    staged: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            staged[key] = value.to(device, non_blocking=True)
        else:
            staged[key] = value
    return staged


def _tensor_batch(payload: dict[str, Any], root_indices: list[int], *, device: str) -> dict[str, Any]:
    torch = require_torch()
    offsets = payload["candidate_offsets"]
    index_device = offsets.device
    root_tensor = torch.tensor(root_indices, dtype=torch.long, device=index_device)
    starts = offsets.index_select(0, root_tensor)
    ends = offsets.index_select(0, root_tensor + 1)
    counts = ends - starts
    if not bool((counts > 0).any().item()):
        raise ValueError("empty transformer tensor root batch")
    root_count = int(root_tensor.numel())
    total_candidates = int(counts.sum().item())
    local_sample_ids = torch.repeat_interleave(torch.arange(root_count, dtype=torch.long, device=index_device), counts)
    root_starts = torch.repeat_interleave(starts, counts)
    candidate_offsets = torch.arange(total_candidates, dtype=torch.long, device=index_device) - torch.repeat_interleave(
        torch.cumsum(counts, dim=0) - counts,
        counts,
    )
    candidate_indices = root_starts + candidate_offsets
    if _is_root_payload(payload):
        batch = {
            "token_scalar_features": payload["token_scalar_features"].index_select(0, root_tensor).to(device, non_blocking=True),
            "token_type_ids": payload["token_type_ids"].index_select(0, root_tensor).to(device, non_blocking=True),
            "entity_ids": payload["entity_ids"].index_select(0, root_tensor).to(device, non_blocking=True),
            "slot_ids": payload["slot_ids"].index_select(0, root_tensor).to(device, non_blocking=True),
            "attention_mask": payload["attention_mask"].index_select(0, root_tensor).to(device, non_blocking=True),
            "before_summary": payload["before_summary"].index_select(0, root_tensor).to(device, non_blocking=True),
            "action_token_positions": payload["action_token_positions"].index_select(0, root_tensor).to(device, non_blocking=True),
            "after_token_positions": payload["after_token_positions"].index_select(0, root_tensor).to(device, non_blocking=True),
            "delta_token_positions": payload["delta_token_positions"].index_select(0, root_tensor).to(device, non_blocking=True),
            "legacy_token_positions": payload["legacy_token_positions"].index_select(0, root_tensor).to(device, non_blocking=True),
            "candidate_mask": payload["candidate_mask"].index_select(0, root_tensor).to(device, non_blocking=True),
            "features": payload["features"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "teacher_q": payload["teacher_q"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "sample_ids": local_sample_ids.to(device, non_blocking=True),
            "chosen": payload["chosen"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "candidate_counts": counts.to(device, non_blocking=True),
            "root_count": len(root_indices),
        }
        active = batch["attention_mask"].any(dim=0)
        if bool(active.any().item()):
            max_len = int(active.nonzero(as_tuple=False).flatten()[-1].item()) + 1
            for key in ("token_scalar_features", "token_type_ids", "entity_ids", "slot_ids", "attention_mask"):
                batch[key] = batch[key][:, :max_len]
    else:
        batch = {
            "token_scalar_features": payload["token_scalar_features"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "token_type_ids": payload["token_type_ids"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "entity_ids": payload["entity_ids"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "slot_ids": payload["slot_ids"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "attention_mask": payload["attention_mask"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "features": payload["features"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "teacher_q": payload["teacher_q"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "sample_ids": local_sample_ids.to(device, non_blocking=True),
            "chosen": payload["chosen"].index_select(0, candidate_indices).to(device, non_blocking=True),
            "candidate_counts": counts.to(device, non_blocking=True),
            "root_count": len(root_indices),
        }
    if "action_is_potion" in payload:
        batch["action_is_potion"] = payload["action_is_potion"].index_select(0, candidate_indices).to(device, non_blocking=True)
    if "room_type_ids" in payload:
        batch["room_type_ids"] = payload["room_type_ids"].index_select(0, candidate_indices).to(device, non_blocking=True)
    if "reward_components" in payload:
        batch["reward_components"] = payload["reward_components"].index_select(0, candidate_indices).to(device, non_blocking=True)
    return batch


def _run_epoch(
    model: V3CombatTransformerCandidateScorer | V3CombatRootActionSetTransformerScorer,
    payload: dict[str, Any],
    root_indices: list[int],
    optimizer,
    *,
    device: str,
    batch_size: int,
    seed: int,
    loss_kwargs: dict[str, float],
    distill_model=None,
    distill_model_kind: str = "mlp",
    amp_dtype=None,
    grad_scaler=None,
    length_bucket_batches: bool = False,
    length_bucket_window: int = 64,
    context: str = "",
) -> dict[str, float]:
    torch = require_torch()
    rng = random.Random(seed)
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch_roots in _root_batches(
        payload,
        root_indices,
        batch_size=batch_size,
        rng=rng,
        shuffle=training,
        length_bucket_batches=length_bucket_batches,
        length_bucket_window=length_bucket_window,
    ):
        batch = _tensor_batch(payload, batch_roots, device=device)
        with torch.set_grad_enabled(training):
            with _autocast_context(torch, device=device, amp_dtype=amp_dtype):
                pred_q = model(batch)
            aux_outputs = getattr(model, "last_aux_outputs", None)
            if aux_outputs is not None:
                batch["aux_reward_pred"] = aux_outputs
            if distill_model is not None:
                with torch.inference_mode():
                    distill_param = next(distill_model.parameters())
                    if str(distill_model_kind) == "transformer":
                        batch["distill_q"] = distill_model(batch).to(device=device, dtype=pred_q.dtype)
                    else:
                        distill_features = batch["features"].to(device=device, dtype=distill_param.dtype)
                        batch["distill_q"] = distill_model(distill_features).to(device=device, dtype=pred_q.dtype)
            if not bool(torch.isfinite(pred_q).all().detach().cpu().item()):
                raise FloatingPointError(
                    "non-finite transformer prediction "
                    f"context={context or 'n/a'} roots={batch_roots[:8]} pred_shape={tuple(pred_q.shape)}"
                )
            loss, metrics = total_candidate_loss(pred_q.float(), batch, **loss_kwargs)
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                teacher_q = batch.get("teacher_q")
                teacher_stats = ""
                if teacher_q is not None:
                    finite = torch.isfinite(teacher_q)
                    bad_count = int((~finite).sum().detach().cpu().item())
                    if bool(finite.any().detach().cpu().item()):
                        finite_q = teacher_q[finite].float()
                        teacher_stats = (
                            f" teacher_q_bad={bad_count}"
                            f" teacher_q_min={float(finite_q.min().detach().cpu().item()):.6g}"
                            f" teacher_q_max={float(finite_q.max().detach().cpu().item()):.6g}"
                        )
                    else:
                        teacher_stats = f" teacher_q_bad={bad_count} teacher_q_all_nonfinite=true"
                raise FloatingPointError(
                    "non-finite transformer loss "
                    f"context={context or 'n/a'} roots={batch_roots[:8]} metrics={metrics}{teacher_stats}"
                )
        if training:
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def _run_epoch_chunked(
    model: V3CombatTransformerCandidateScorer | V3CombatRootActionSetTransformerScorer,
    chunks: list[dict[str, Any]],
    optimizer,
    *,
    device: str,
    batch_size: int,
    seed: int,
    loss_kwargs: dict[str, float],
    distill_model=None,
    distill_model_kind: str = "mlp",
    progress_label: str = "",
    progress_interval_chunks: int = 0,
    progress_interval_seconds: float = 0.0,
    amp_dtype=None,
    grad_scaler=None,
    length_bucket_batches: bool = False,
    length_bucket_window: int = 64,
    stage_chunks_on_device: bool = False,
    memory_log: Path | None = None,
) -> dict[str, float]:
    torch = require_torch()
    rng = random.Random(seed)
    ordered_chunks = list(chunks)
    if optimizer is not None:
        rng.shuffle(ordered_chunks)
        model.train()
    else:
        model.eval()
    totals: dict[str, float] = {}
    batches = 0
    total_roots = sum(int(chunk.get("root_count") or 0) for chunk in ordered_chunks)
    total_candidates = sum(int(chunk.get("candidate_count") or 0) for chunk in ordered_chunks)
    processed_roots = 0
    processed_candidates = 0
    start_time = time.time()
    last_progress = start_time
    for chunk_index, chunk in enumerate(ordered_chunks, start=1):
        chunk_start = time.time()
        _append_memory_log(
            memory_log,
            _memory_snapshot(
                torch,
                device=device,
                event="chunk_start",
                label=progress_label,
                chunk_index=chunk_index,
                chunk_total=len(ordered_chunks),
                chunk_root_count=int(chunk.get("root_count") or 0),
                chunk_candidate_count=int(chunk.get("candidate_count") or 0),
                source_shard=str(chunk.get("source_shard") or ""),
                chunk_path=str(chunk.get("path") or ""),
            ),
        )
        payload = _load_chunk(chunk)
        if stage_chunks_on_device and str(device).startswith("cuda"):
            payload = _move_payload_tensors(payload, device=device)
        root_count = int(payload["metadata"]["root_count"])
        candidate_count = int(payload["metadata"].get("candidate_count") or chunk.get("candidate_count") or 0)
        indices = list(range(root_count))
        if optimizer is not None:
            rng.shuffle(indices)
        metrics = _run_epoch(
            model,
            payload,
            indices,
            optimizer,
            device=device,
            batch_size=batch_size,
            seed=seed,
            loss_kwargs=loss_kwargs,
            distill_model=distill_model,
            distill_model_kind=distill_model_kind,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            length_bucket_batches=length_bucket_batches,
            length_bucket_window=length_bucket_window,
            context=f"chunk={chunk_index}/{len(ordered_chunks)} source_shard={chunk.get('source_shard')} path={chunk.get('path')}",
        )
        chunk_batches = max(1, (root_count + batch_size - 1) // batch_size)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * chunk_batches
        batches += chunk_batches
        processed_roots += root_count
        processed_candidates += candidate_count
        now = time.time()
        should_log = False
        if int(progress_interval_chunks) > 0 and (chunk_index == 1 or chunk_index % int(progress_interval_chunks) == 0 or chunk_index == len(ordered_chunks)):
            should_log = True
        if float(progress_interval_seconds) > 0.0 and now - last_progress >= float(progress_interval_seconds):
            should_log = True
        if should_log and progress_label:
            elapsed = max(1.0e-6, now - start_time)
            chunks_per_second = chunk_index / elapsed
            remaining_chunks = max(0, len(ordered_chunks) - chunk_index)
            eta_min = remaining_chunks / max(1.0e-6, chunks_per_second) / 60.0
            print(
                f"{progress_label} progress chunks={chunk_index}/{len(ordered_chunks)} "
                f"roots={processed_roots}/{total_roots} "
                f"candidates={processed_candidates}/{total_candidates} "
                f"loss={float(metrics.get('loss', 0.0)):.4f} "
                f"chunk_sec={now - chunk_start:.1f} "
                f"elapsed_min={elapsed / 60.0:.1f} "
                f"eta_min={eta_min:.1f}",
                flush=True,
            )
            last_progress = now
        del payload
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _append_memory_log(
            memory_log,
            _memory_snapshot(
                torch,
                device=device,
                event="chunk_done",
                label=progress_label,
                chunk_index=chunk_index,
                chunk_total=len(ordered_chunks),
                root_count=root_count,
                candidate_count=candidate_count,
                chunk_seconds=now - chunk_start,
                loss=float(metrics.get("loss", 0.0)),
                source_shard=str(chunk.get("source_shard") or ""),
                chunk_path=str(chunk.get("path") or ""),
            ),
        )
    return {key: value / max(1, batches) for key, value in totals.items()}


def _checkpoint_metadata(
    *,
    root_count: int,
    train_count: int,
    validation_count: int,
    is_chunked: bool,
    best_validation: float | None,
    best_epoch: int | None,
    stopped_for_low_memory: bool,
    stopped_for_early_stop: bool,
    early_stop_bad_epochs: int,
    early_stop_best_validation: float | None,
    init_checkpoint: Path | None,
    init_checkpoint_metadata: dict[str, Any] | None,
    tensor_dataset: Path,
) -> dict[str, Any]:
    return {
        "root_count": root_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "chunked_dataset": is_chunked,
        "best_validation_loss": best_validation,
        "best_epoch": best_epoch,
        "stopped_for_low_memory": stopped_for_low_memory,
        "stopped_for_early_stop": stopped_for_early_stop,
        "early_stop_bad_epochs": early_stop_bad_epochs,
        "early_stop_best_validation_loss": early_stop_best_validation,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "init_checkpoint_best_validation_loss": init_checkpoint_metadata.get("best_validation_loss") if init_checkpoint_metadata else None,
        "source_tensor_dataset": str(tensor_dataset),
    }


def _save_training_checkpoint(
    path: Path,
    model: V3CombatTransformerCandidateScorer | V3CombatRootActionSetTransformerScorer,
    optimizer,
    scheduler,
    *,
    args: argparse.Namespace,
    root_count: int,
    train_count: int,
    validation_count: int,
    is_chunked: bool,
    current_epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    best_validation: float | None,
    best_epoch: int | None,
    best_state_dict: dict[str, Any] | None,
    stopped_for_low_memory: bool,
    stopped_for_early_stop: bool,
    early_stop_bad_epochs: int,
    early_stop_best_validation: float | None,
    init_checkpoint_metadata: dict[str, Any] | None,
) -> None:
    save_v3_combat_transformer_checkpoint(
        path,
        model,
        training_args=vars(args),
        dataset_metadata=_checkpoint_metadata(
            root_count=root_count,
            train_count=train_count,
            validation_count=validation_count,
            is_chunked=is_chunked,
            best_validation=best_validation,
            best_epoch=best_epoch,
            stopped_for_low_memory=stopped_for_low_memory,
            stopped_for_early_stop=stopped_for_early_stop,
            early_stop_bad_epochs=early_stop_bad_epochs,
            early_stop_best_validation=early_stop_best_validation,
            init_checkpoint=args.init_checkpoint or args.resume_checkpoint,
            init_checkpoint_metadata=init_checkpoint_metadata,
            tensor_dataset=args.tensor_dataset,
        ),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        training_state={
            "current_epoch": int(current_epoch),
            "train_metrics": dict(train_metrics),
            "validation_metrics": dict(validation_metrics),
            "best_validation_loss": best_validation,
            "best_epoch": best_epoch,
            "best_model_state_dict": copy.deepcopy(best_state_dict) if best_state_dict is not None else None,
            "stopped_for_low_memory": bool(stopped_for_low_memory),
            "stopped_for_early_stop": bool(stopped_for_early_stop),
            "early_stop_bad_epochs": int(early_stop_bad_epochs),
            "early_stop_best_validation_loss": early_stop_best_validation,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transformer v3 combat candidate scorer.")
    parser.add_argument("--tensor-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/v3_combat_transformer_stage5_v1.pt"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume model, optimizer, scheduler, and best state from an epoch checkpoint.")
    parser.add_argument("--epoch-output-dir", type=Path, default=None, help="Directory for per-epoch checkpoints. Default: <output>.epochs")
    parser.add_argument("--save-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-source-shards-file",
        type=Path,
        default=None,
        help="For chunked datasets, keep chunks whose source_shard is listed here as validation; all new chunks train.",
    )
    parser.add_argument("--device", default="cuda" if require_torch().cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=["auto", "none", "bfloat16", "float16"], default="auto")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limit-roots", type=int, default=0, help="Optional training smoke-test cap over cached roots.")
    parser.add_argument("--min-roots", type=int, default=0, help="Fail fast if the tensor dataset has fewer roots than this.")
    parser.add_argument("--architecture", choices=["auto", "candidate", "root-action-set"], default="auto")
    parser.add_argument(
        "--root-head-variant",
        choices=sorted(ROOT_HEAD_VARIANTS),
        default=ROOT_HEAD_VARIANT_BASE,
        help="Root-action-set scoring head variant. v4 experiments use this without rebuilding the tensor cache.",
    )
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument(
        "--candidate-head-variant",
        choices=sorted(CANDIDATE_HEAD_VARIANTS),
        default=CANDIDATE_HEAD_VARIANT_BASE,
        help="Candidate-level scoring head variant for v5 two-stage action-set experiments.",
    )
    parser.add_argument("--legacy-dropout", type=float, default=0.0, help="Training-time dropout on the LEGACY vector for heads that use it.")
    parser.add_argument(
        "--disable-token-types",
        default="",
        help="Comma-separated token type names masked out before attention/head extraction, e.g. LEGACY,POWER_DELTA.",
    )
    parser.add_argument("--semantic-delta-scale", type=float, default=1.0)
    parser.add_argument("--semantic-delta-clip", type=float, default=4.0)
    parser.add_argument("--auxiliary-reward-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aux-reward-weight", type=float, default=0.0)
    parser.add_argument("--potion-residual-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--potion-residual-clip", type=float, default=2.0)
    parser.add_argument("--card-residual-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--card-residual-clip", type=float, default=2.0)
    parser.add_argument("--legacy-baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-legacy-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--token-type-vocab-size",
        type=int,
        default=0,
        help="Override token type embedding rows. Use 14 for old candidate-actionset parity; <=0 uses code default.",
    )
    parser.add_argument("--action-set-layers", type=int, default=1, help="Root-level action-set transformer layers over candidates in the same root; 0 keeps the old candidate-independent head.")
    parser.add_argument("--action-set-ffn-dim", type=int, default=0, help="Feed-forward width for the action-set transformer; <=0 reuses --ffn-dim.")
    parser.add_argument("--max-actions", type=int, default=0, help="Root-action-set max actions; <=0 uses the dataset token schema value.")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--memory-log", type=Path, default=None)
    parser.add_argument("--min-mem-available-gb", type=float, default=3.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument(
        "--rank-temperature",
        type=float,
        default=1.0,
        help="Temperature for the teacher distribution in ranking KL; >1 softens noisy teacher gaps.",
    )
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--pair-weight", type=float, default=0.2)
    parser.add_argument("--bc-weight", type=float, default=0.05)
    parser.add_argument("--potion-vs-non-potion-weight", type=float, default=0.5)
    parser.add_argument("--potion-vs-non-potion-margin", type=float, default=0.15)
    parser.add_argument("--potion-vs-non-potion-min-teacher-gap", type=float, default=0.5)
    parser.add_argument("--elite-boss-top-potion-root-weight", type=float, default=6.0)
    parser.add_argument("--critical-loss-weight", type=float, default=0.0)
    parser.add_argument("--critical-loss-margin", type=float, default=0.2)
    parser.add_argument("--critical-loss-min-teacher-gap", type=float, default=1.0)
    parser.add_argument("--critical-loss-floor-max", type=float, default=5.0)
    parser.add_argument("--critical-loss-room-id", type=int, default=1, help="1=MonsterRoom, 2=Elite, 3=Boss, <=0 disables room filtering.")
    parser.add_argument("--critical-loss-non-potion-teacher-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gap-q-weight", type=float, default=0.0)
    parser.add_argument("--gap-q-transform", choices=["none", "raw", "linear", "sqrt", "log"], default="sqrt")
    parser.add_argument("--gap-q-loss", choices=["l1", "abs", "absolute", "smooth_l1", "huber"], default="l1")
    parser.add_argument("--gap-q-hard-negative-threshold", type=float, default=10.0)
    parser.add_argument("--gap-q-hard-negative-weight", type=float, default=5.0)
    parser.add_argument("--hard-top-weight", type=float, default=0.0)
    parser.add_argument("--hard-top-min-teacher-gap", type=float, default=5.0)
    parser.add_argument("--hard-top-topk", type=int, default=2)
    parser.add_argument("--hard-top-margin-base", type=float, default=0.25)
    parser.add_argument("--hard-top-margin-log-scale", type=float, default=0.35)
    parser.add_argument("--hard-top-margin-max", type=float, default=3.0)
    parser.add_argument(
        "--hard-top-kind-filter",
        choices=["all", "card-card", "teacher-potion", "teacher-card", "teacher-end"],
        default="all",
    )
    parser.add_argument("--hard-top-card-card-weight", type=float, default=2.0)
    parser.add_argument("--hard-top-monster-room-weight", type=float, default=1.5)
    parser.add_argument("--hard-top-early-floor-max", type=float, default=16.0)
    parser.add_argument("--hard-top-early-floor-weight", type=float, default=2.0)
    parser.add_argument("--hard-top-large-gap-threshold", type=float, default=25.0)
    parser.add_argument("--hard-top-large-gap-weight", type=float, default=2.0)
    parser.add_argument("--hard-top-root-weight-clip", type=float, default=6.0)
    parser.add_argument("--good-bad-weight", type=float, default=0.0)
    parser.add_argument("--good-bad-good-teacher-gap", type=float, default=2.0)
    parser.add_argument("--good-bad-bad-min-regret", type=float, default=25.0)
    parser.add_argument("--good-bad-min-top-gap", type=float, default=5.0)
    parser.add_argument("--good-bad-bad-topk", type=int, default=2)
    parser.add_argument("--good-bad-margin-base", type=float, default=0.25)
    parser.add_argument("--good-bad-margin-log-scale", type=float, default=0.25)
    parser.add_argument("--good-bad-margin-max", type=float, default=3.0)
    parser.add_argument(
        "--good-bad-kind-filter",
        choices=["all", "card-card", "teacher-card", "teacher-potion", "teacher-end"],
        default="card-card",
    )
    parser.add_argument(
        "--good-bad-room-filter",
        choices=["all", "combat", "monster", "elite-boss"],
        default="combat",
    )
    parser.add_argument("--good-bad-monster-room-weight", type=float, default=1.5)
    parser.add_argument("--good-bad-early-floor-max", type=float, default=16.0)
    parser.add_argument("--good-bad-early-floor-weight", type=float, default=2.0)
    parser.add_argument("--good-bad-large-gap-threshold", type=float, default=50.0)
    parser.add_argument("--good-bad-large-gap-weight", type=float, default=1.5)
    parser.add_argument("--good-bad-root-weight-clip", type=float, default=5.0)
    parser.add_argument("--top1-ce-weight", type=float, default=0.0)
    parser.add_argument("--top1-ce-min-teacher-gap", type=float, default=0.0)
    parser.add_argument("--top1-ce-teacher-gap-log-scale", type=float, default=0.25)
    parser.add_argument("--top1-ce-large-gap-threshold", type=float, default=10.0)
    parser.add_argument("--top1-ce-large-gap-weight", type=float, default=2.0)
    parser.add_argument("--top1-ce-monster-room-weight", type=float, default=1.0)
    parser.add_argument("--top1-ce-early-floor-max", type=float, default=0.0)
    parser.add_argument("--top1-ce-early-floor-weight", type=float, default=1.0)
    parser.add_argument(
        "--top1-ce-kind-filter",
        choices=["all", "teacher-card", "teacher-potion", "teacher-end"],
        default="all",
    )
    parser.add_argument("--top1-ce-root-weight-clip", type=float, default=6.0)
    parser.add_argument(
        "--teacher-q-clip",
        type=float,
        default=0.0,
        help="If >0, replace non-finite teacher_q and clamp teacher_q to +/- this value inside training losses.",
    )
    parser.add_argument("--distill-mlp-checkpoint", type=Path, default=None)
    parser.add_argument("--distill-transformer-checkpoint", type=Path, default=None)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--distill-floor-max", type=float, default=10.0)
    parser.add_argument("--distill-room-id", type=int, default=1, help="1=MonsterRoom, 2=Elite, 3=Boss, <=0 disables room filtering.")
    parser.add_argument("--distill-non-potion-teacher-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--distill-root-mode",
        choices=["early", "all", "baseline_correct", "baseline_correct_non_potion"],
        default="early",
        help="Which roots receive distillation. baseline_correct uses roots where the distill model agrees with teacher top1.",
    )
    parser.add_argument("--anchor-guard-weight", type=float, default=0.0)
    parser.add_argument("--anchor-guard-margin", type=float, default=0.1)
    parser.add_argument("--anchor-guard-min-teacher-gap", type=float, default=0.5)
    parser.add_argument(
        "--trainable-name-patterns",
        default="",
        help=(
            "Comma-separated parameter-name substrings to keep trainable. "
            "If non-empty, every unmatched parameter is frozen."
        ),
    )
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Stop after this many validation epochs without a meaningful improvement; <=0 disables.")
    parser.add_argument("--early-stop-min-delta", type=float, default=5e-4, help="Minimum validation-loss drop that resets early-stop patience.")
    parser.add_argument("--length-bucket-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length-bucket-window", type=int, default=64, help="Shuffle roots, then sort windows of batch_size*N by active token length before batching.")
    parser.add_argument(
        "--stage-chunks-on-device",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For chunked datasets on CUDA, move each chunk to the training device once before batching.",
    )
    parser.add_argument("--progress-interval-chunks", type=int, default=10)
    parser.add_argument("--progress-interval-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise SystemExit("--init-checkpoint and --resume-checkpoint are mutually exclusive.")
    action_set_ffn_dim = None if int(args.action_set_ffn_dim) <= 0 else int(args.action_set_ffn_dim)
    token_type_vocab_size = int(args.token_type_vocab_size) if int(args.token_type_vocab_size) > 0 else len(TOKEN_TYPES)

    torch = require_torch()
    _configure_torch_runtime(torch, device=args.device, allow_tf32=bool(args.allow_tf32))
    amp_dtype = _resolve_amp_dtype(torch, device=args.device, amp_dtype=str(args.amp_dtype))
    grad_scaler = _make_grad_scaler(torch, device=args.device, amp_dtype=amp_dtype)
    amp_dtype_name = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none"
    payload = _load_tensor_dataset(args.tensor_dataset)
    root_count = int(payload["metadata"]["root_count"])
    is_chunked = _is_chunked_payload(payload)
    is_root_dataset = _is_root_payload(payload)
    if int(args.min_roots) > 0 and root_count < int(args.min_roots):
        raise SystemExit(f"tensor dataset has only {root_count} roots, below --min-roots={int(args.min_roots)}")
    architecture = args.architecture
    if architecture == "auto":
        architecture = "root-action-set" if is_root_dataset else "candidate"
    if architecture == "root-action-set" and not is_root_dataset:
        raise SystemExit("--architecture root-action-set requires a root-action-set tensor dataset.")
    if architecture == "candidate" and is_root_dataset:
        raise SystemExit("root-action-set tensor dataset requires --architecture root-action-set or --architecture auto.")
    if args.limit_roots > 0 and not is_chunked:
        root_count = min(root_count, int(args.limit_roots))
    if root_count <= 0:
        raise SystemExit("No v3 combat transformer roots loaded.")
    rng = random.Random(args.seed)
    if is_chunked:
        chunks = list(payload.get("chunks") or [])
        if not chunks:
            raise SystemExit("Chunked transformer dataset has no chunks.")
        validation_sources = _read_lines(args.validation_source_shards_file)
        if validation_sources:
            validation_chunks = [chunk for chunk in chunks if str(chunk.get("source_shard") or "") in validation_sources]
            train_chunks = [chunk for chunk in chunks if str(chunk.get("source_shard") or "") not in validation_sources]
            if not validation_chunks:
                raise SystemExit(f"no validation chunks matched {args.validation_source_shards_file}")
            if not train_chunks:
                train_chunks = chunks
        else:
            rng.shuffle(chunks)
            val_chunk_count = int(len(chunks) * args.validation_fraction)
            validation_chunks = chunks[:val_chunk_count]
            train_chunks = chunks[val_chunk_count:] or chunks
        train_indices = []
        validation_indices = []
    else:
        chunks = []
        validation_chunks = []
        train_chunks = []
        indices = list(range(root_count))
        rng.shuffle(indices)
        val_count = int(root_count * args.validation_fraction)
        validation_indices = indices[:val_count]
        train_indices = indices[val_count:] or indices

    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        model, resume_checkpoint = load_v3_combat_transformer_checkpoint(args.resume_checkpoint, device=args.device)
        init_checkpoint_metadata = dict(resume_checkpoint.get("dataset_metadata") or {})
    elif args.init_checkpoint is not None:
        model, init_checkpoint = load_v3_combat_transformer_checkpoint(args.init_checkpoint, device=args.device)
        init_checkpoint_metadata = dict(init_checkpoint.get("dataset_metadata") or {})
    else:
        feature_dims = dict(payload.get("feature_dims") or {})
        token_schema = dict(payload.get("token_schema") or {})
        state_dim = int(feature_dims.get("state_dim") or 0)
        action_dim = int(feature_dims.get("action_dim") or 0)
        delta_dim = int(feature_dims.get("delta_dim") or 0)
        feature_dim = state_dim * 2 + action_dim + delta_dim if state_dim and action_dim and delta_dim else 0
        if architecture == "root-action-set":
            model = V3CombatRootActionSetTransformerScorer(
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                ffn_dim=args.ffn_dim,
                dropout=args.dropout,
                scalar_dim=int(token_schema.get("scalar_dim") or payload["metadata"].get("scalar_dim") or args.d_model),
                token_type_vocab_size=token_type_vocab_size,
                entity_vocab_size=len(payload.get("entity_vocab") or []),
                max_sequence_length=int(token_schema.get("max_sequence_length") or payload["metadata"].get("sequence_length") or 0) or None,
                max_actions=int(args.max_actions or token_schema.get("max_actions") or payload["metadata"].get("max_actions") or 0)
                or None,
                feature_dim=feature_dim or None,
                state_dim=state_dim or None,
                action_dim=action_dim or None,
                delta_dim=delta_dim or None,
                token_schema_version=str(payload.get("token_schema_version") or token_schema.get("version") or ""),
                root_head_variant=str(args.root_head_variant),
            ).to(args.device)
        else:
            model = V3CombatTransformerCandidateScorer(
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                ffn_dim=args.ffn_dim,
                action_set_layers=args.action_set_layers,
                action_set_ffn_dim=action_set_ffn_dim,
                candidate_head_variant=str(args.candidate_head_variant),
                legacy_dropout=float(args.legacy_dropout),
                disabled_token_types=str(args.disable_token_types),
                semantic_delta_scale=float(args.semantic_delta_scale),
                semantic_delta_clip=float(args.semantic_delta_clip),
                auxiliary_reward_heads=bool(args.auxiliary_reward_heads),
                potion_residual_head_enabled=bool(args.potion_residual_head),
                potion_residual_clip=float(args.potion_residual_clip),
                card_residual_head_enabled=bool(args.card_residual_head),
                card_residual_clip=float(args.card_residual_clip),
                dropout=args.dropout,
                scalar_dim=int(token_schema.get("scalar_dim") or payload["metadata"].get("scalar_dim") or args.d_model),
                token_type_vocab_size=token_type_vocab_size,
                entity_vocab_size=len(payload.get("entity_vocab") or []),
                max_sequence_length=int(token_schema.get("max_sequence_length") or payload["metadata"].get("sequence_length") or 0) or None,
                feature_dim=feature_dim or None,
                state_dim=state_dim or None,
                action_dim=action_dim or None,
                delta_dim=delta_dim or None,
            ).to(args.device)
        init_checkpoint_metadata = None
    _attach_tensor_dataset_metadata(model, payload)
    if bool(args.auxiliary_reward_heads):
        if hasattr(model, "enable_auxiliary_reward_heads"):
            if not bool(getattr(model, "auxiliary_reward_heads", False)):
                model.enable_auxiliary_reward_heads(output_dim=REWARD_COMPONENT_DIM)
        elif not bool(getattr(model, "auxiliary_reward_heads", False)):
            raise SystemExit("--auxiliary-reward-heads is not supported by this architecture")
    if bool(args.potion_residual_head):
        if hasattr(model, "enable_potion_residual_head"):
            if not bool(getattr(model, "potion_residual_head_enabled", False)):
                model.enable_potion_residual_head(clip=float(args.potion_residual_clip))
            else:
                setattr(model, "potion_residual_clip", float(max(0.0, args.potion_residual_clip)))
        else:
            raise SystemExit("--potion-residual-head is not supported by this architecture")
    if bool(args.card_residual_head):
        if hasattr(model, "enable_card_residual_head"):
            if not bool(getattr(model, "card_residual_head_enabled", False)):
                model.enable_card_residual_head(clip=float(args.card_residual_clip))
            else:
                setattr(model, "card_residual_clip", float(max(0.0, args.card_residual_clip)))
        else:
            raise SystemExit("--card-residual-head is not supported by this architecture")
    if args.legacy_baseline_checkpoint is not None:
        if not hasattr(model, "legacy_baseline_head") or getattr(model, "legacy_baseline_head", None) is None:
            raise SystemExit("--legacy-baseline-checkpoint requires candidate-head-variant=legacy-baseline-semantic-delta")
        baseline_model, _baseline_checkpoint = load_v3_combat_checkpoint(args.legacy_baseline_checkpoint, device=args.device)
        model.legacy_baseline_head.load_state_dict(baseline_model.network.state_dict())
        if bool(args.freeze_legacy_baseline):
            for parameter in model.legacy_baseline_head.parameters():
                parameter.requires_grad_(False)
    trainable_report = _apply_trainable_name_patterns(model, str(args.trainable_name_patterns))
    trainable_parameters = [parameter for parameter in model.parameters() if bool(parameter.requires_grad)]
    if not trainable_parameters:
        raise SystemExit("No trainable parameters remain after applying freeze settings.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=max(0.0, float(args.min_learning_rate)),
    )
    distill_model = None
    distill_model_kind = "mlp"
    if float(args.distill_weight) > 0.0 or float(args.anchor_guard_weight) > 0.0:
        if args.distill_mlp_checkpoint is not None and args.distill_transformer_checkpoint is not None:
            raise SystemExit("--distill-mlp-checkpoint and --distill-transformer-checkpoint are mutually exclusive")
        if args.distill_transformer_checkpoint is not None:
            distill_model, _distill_checkpoint = load_v3_combat_transformer_checkpoint(args.distill_transformer_checkpoint, device=args.device)
            if bool(getattr(distill_model, "expects_root_batch", False)) != bool(getattr(model, "expects_root_batch", False)):
                raise SystemExit("distill transformer architecture must match the trained model architecture")
            distill_model_kind = "transformer"
        elif args.distill_mlp_checkpoint is not None:
            distill_model, _distill_checkpoint = load_v3_combat_checkpoint(args.distill_mlp_checkpoint, device=args.device)
            distill_model_kind = "mlp"
        else:
            raise SystemExit(
                "--distill-weight/--anchor-guard-weight requires "
                "--distill-mlp-checkpoint or --distill-transformer-checkpoint"
            )
        distill_model.eval()
        for parameter in distill_model.parameters():
            parameter.requires_grad_(False)
    memory_log = args.memory_log or args.output.with_suffix(args.output.suffix + ".memory_log.jsonl")
    min_mem_available_kb = int(max(0.0, float(args.min_mem_available_gb)) * 1024 * 1024)
    best_validation = None
    best_state_dict = None
    best_epoch = None
    stopped_for_low_memory = False
    stopped_for_early_stop = False
    early_stop_bad_epochs = 0
    early_stop_best_validation = None
    start_epoch = 1
    if resume_checkpoint is not None:
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        scheduler_state = resume_checkpoint.get("scheduler_state_dict")
        training_state = dict(resume_checkpoint.get("training_state") or {})
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            _optimizer_to_device(optimizer, args.device)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        start_epoch = int(training_state.get("current_epoch") or 0) + 1
        best_validation = training_state.get("best_validation_loss")
        if best_validation is not None:
            best_validation = float(best_validation)
        best_epoch = training_state.get("best_epoch")
        if best_epoch is not None:
            best_epoch = int(best_epoch)
        best_state_dict = training_state.get("best_model_state_dict")
        if best_state_dict is None and best_epoch == int(training_state.get("current_epoch") or 0):
            best_state_dict = copy.deepcopy(model.state_dict())
        stopped_for_low_memory = bool(training_state.get("stopped_for_low_memory", False))
        stopped_for_early_stop = bool(training_state.get("stopped_for_early_stop", False))
        early_stop_bad_epochs = int(training_state.get("early_stop_bad_epochs") or 0)
        early_stop_best_validation = training_state.get("early_stop_best_validation_loss")
        if early_stop_best_validation is not None:
            early_stop_best_validation = float(early_stop_best_validation)
        elif best_validation is not None:
            early_stop_best_validation = float(best_validation)
        if start_epoch > int(args.epochs):
            raise SystemExit(f"resume checkpoint is already at epoch {start_epoch - 1}, beyond --epochs={args.epochs}")
    epoch_output_dir = args.epoch_output_dir or args.output.with_suffix(args.output.suffix + ".epochs")
    loss_kwargs = {
        "rank_weight": float(args.rank_weight),
        "temperature": float(args.rank_temperature),
        "q_weight": float(args.q_weight),
        "pair_weight": float(args.pair_weight),
        "bc_weight": float(args.bc_weight),
        "potion_pair_weight": float(args.potion_vs_non_potion_weight),
        "potion_pair_margin": float(args.potion_vs_non_potion_margin),
        "potion_pair_min_teacher_gap": float(args.potion_vs_non_potion_min_teacher_gap),
        "elite_boss_top_potion_root_weight": float(args.elite_boss_top_potion_root_weight),
        "critical_loss_weight": float(args.critical_loss_weight),
        "critical_loss_margin": float(args.critical_loss_margin),
        "critical_loss_min_teacher_gap": float(args.critical_loss_min_teacher_gap),
        "critical_loss_floor_max": float(args.critical_loss_floor_max),
        "critical_loss_room_id": int(args.critical_loss_room_id),
        "critical_loss_non_potion_teacher_only": bool(args.critical_loss_non_potion_teacher_only),
        "gap_q_weight": float(args.gap_q_weight),
        "gap_q_transform": str(args.gap_q_transform),
        "gap_q_loss": str(args.gap_q_loss),
        "gap_q_hard_negative_threshold": float(args.gap_q_hard_negative_threshold),
        "gap_q_hard_negative_weight": float(args.gap_q_hard_negative_weight),
        "hard_top_weight": float(args.hard_top_weight),
        "hard_top_min_teacher_gap": float(args.hard_top_min_teacher_gap),
        "hard_top_topk": int(args.hard_top_topk),
        "hard_top_margin_base": float(args.hard_top_margin_base),
        "hard_top_margin_log_scale": float(args.hard_top_margin_log_scale),
        "hard_top_margin_max": float(args.hard_top_margin_max),
        "hard_top_kind_filter": str(args.hard_top_kind_filter),
        "hard_top_card_card_weight": float(args.hard_top_card_card_weight),
        "hard_top_monster_room_weight": float(args.hard_top_monster_room_weight),
        "hard_top_early_floor_max": float(args.hard_top_early_floor_max),
        "hard_top_early_floor_weight": float(args.hard_top_early_floor_weight),
        "hard_top_large_gap_threshold": float(args.hard_top_large_gap_threshold),
        "hard_top_large_gap_weight": float(args.hard_top_large_gap_weight),
        "hard_top_root_weight_clip": float(args.hard_top_root_weight_clip),
        "good_bad_weight": float(args.good_bad_weight),
        "good_bad_good_teacher_gap": float(args.good_bad_good_teacher_gap),
        "good_bad_bad_min_regret": float(args.good_bad_bad_min_regret),
        "good_bad_min_top_gap": float(args.good_bad_min_top_gap),
        "good_bad_bad_topk": int(args.good_bad_bad_topk),
        "good_bad_margin_base": float(args.good_bad_margin_base),
        "good_bad_margin_log_scale": float(args.good_bad_margin_log_scale),
        "good_bad_margin_max": float(args.good_bad_margin_max),
        "good_bad_kind_filter": str(args.good_bad_kind_filter),
        "good_bad_room_filter": str(args.good_bad_room_filter),
        "good_bad_monster_room_weight": float(args.good_bad_monster_room_weight),
        "good_bad_early_floor_max": float(args.good_bad_early_floor_max),
        "good_bad_early_floor_weight": float(args.good_bad_early_floor_weight),
        "good_bad_large_gap_threshold": float(args.good_bad_large_gap_threshold),
        "good_bad_large_gap_weight": float(args.good_bad_large_gap_weight),
        "good_bad_root_weight_clip": float(args.good_bad_root_weight_clip),
        "top1_ce_weight": float(args.top1_ce_weight),
        "top1_ce_min_teacher_gap": float(args.top1_ce_min_teacher_gap),
        "top1_ce_teacher_gap_log_scale": float(args.top1_ce_teacher_gap_log_scale),
        "top1_ce_large_gap_threshold": float(args.top1_ce_large_gap_threshold),
        "top1_ce_large_gap_weight": float(args.top1_ce_large_gap_weight),
        "top1_ce_monster_room_weight": float(args.top1_ce_monster_room_weight),
        "top1_ce_early_floor_max": float(args.top1_ce_early_floor_max),
        "top1_ce_early_floor_weight": float(args.top1_ce_early_floor_weight),
        "top1_ce_kind_filter": str(args.top1_ce_kind_filter),
        "top1_ce_root_weight_clip": float(args.top1_ce_root_weight_clip),
        "teacher_q_clip": float(args.teacher_q_clip),
        "distill_weight": float(args.distill_weight),
        "distill_temperature": float(args.distill_temperature),
        "distill_floor_max": float(args.distill_floor_max),
        "distill_room_id": int(args.distill_room_id),
        "distill_non_potion_teacher_only": bool(args.distill_non_potion_teacher_only),
        "distill_root_mode": str(args.distill_root_mode),
        "anchor_guard_weight": float(args.anchor_guard_weight),
        "anchor_guard_margin": float(args.anchor_guard_margin),
        "anchor_guard_min_teacher_gap": float(args.anchor_guard_min_teacher_gap),
        "aux_reward_weight": float(args.aux_reward_weight),
    }

    print(
        "loaded transformer "
        f"roots={root_count} "
        f"train={sum(int(chunk.get('root_count') or 0) for chunk in train_chunks) if is_chunked else len(train_indices)} "
        f"validation={sum(int(chunk.get('root_count') or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices)} "
        f"chunked={is_chunked} "
        f"architecture={architecture} "
        f"root_head_variant={getattr(model, 'root_head_variant', 'n/a')} "
        f"candidate_head_variant={getattr(model, 'candidate_head_variant', 'n/a')} "
        f"legacy_dropout={float(getattr(model, 'legacy_dropout', 0.0))} "
        f"disabled_token_types={','.join(getattr(model, 'disabled_token_type_names', ())) or 'none'} "
        f"auxiliary_reward_heads={bool(getattr(model, 'auxiliary_reward_heads', False))} "
        f"potion_residual_head={bool(getattr(model, 'potion_residual_head_enabled', False))} "
        f"potion_residual_clip={float(getattr(model, 'potion_residual_clip', 0.0))} "
        f"card_residual_head={bool(getattr(model, 'card_residual_head_enabled', False))} "
        f"card_residual_clip={float(getattr(model, 'card_residual_clip', 0.0))} "
        f"token_type_vocab_size={int(getattr(model, 'token_type_vocab_size', 0))} "
        f"amp_dtype={amp_dtype_name} "
        f"allow_tf32={bool(args.allow_tf32)} "
        f"length_bucket_batches={bool(args.length_bucket_batches)} "
        f"length_bucket_window={int(args.length_bucket_window)} "
        f"stage_chunks_on_device={bool(args.stage_chunks_on_device)} "
        f"rank_weight={float(args.rank_weight)} "
        f"rank_temperature={float(args.rank_temperature)} "
        f"q_weight={float(args.q_weight)} "
        f"pair_weight={float(args.pair_weight)} "
        f"bc_weight={float(args.bc_weight)} "
        f"critical_loss_weight={float(args.critical_loss_weight)} "
        f"gap_q_weight={float(args.gap_q_weight)} "
        f"gap_q_transform={args.gap_q_transform} "
        f"gap_q_hard_negative_threshold={float(args.gap_q_hard_negative_threshold)} "
        f"gap_q_hard_negative_weight={float(args.gap_q_hard_negative_weight)} "
        f"hard_top_weight={float(args.hard_top_weight)} "
        f"hard_top_min_teacher_gap={float(args.hard_top_min_teacher_gap)} "
        f"hard_top_topk={int(args.hard_top_topk)} "
        f"hard_top_margin_max={float(args.hard_top_margin_max)} "
        f"hard_top_kind_filter={args.hard_top_kind_filter} "
        f"good_bad_weight={float(args.good_bad_weight)} "
        f"good_bad_min_top_gap={float(args.good_bad_min_top_gap)} "
        f"good_bad_bad_min_regret={float(args.good_bad_bad_min_regret)} "
        f"good_bad_kind_filter={args.good_bad_kind_filter} "
        f"good_bad_room_filter={args.good_bad_room_filter} "
        f"top1_ce_weight={float(args.top1_ce_weight)} "
        f"top1_ce_min_teacher_gap={float(args.top1_ce_min_teacher_gap)} "
        f"top1_ce_kind_filter={args.top1_ce_kind_filter} "
        f"teacher_q_clip={float(args.teacher_q_clip)} "
        f"distill_weight={float(args.distill_weight)} "
        f"anchor_guard_weight={float(args.anchor_guard_weight)} "
        f"trainable_parameter_tensors={int(trainable_report['trainable_parameter_tensors'])} "
        f"trainable_parameters={int(trainable_report['trainable_parameters'])} "
        f"trainable_name_patterns={','.join(trainable_report['trainable_name_patterns']) or 'all'} "
        f"candidates={payload['metadata'].get('candidate_count')} checkpoint_version={CHECKPOINT_VERSION}"
    )
    _append_memory_log(
        memory_log,
        _memory_snapshot(
            torch,
            device=args.device,
            event="training_start",
            roots=root_count,
            train=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
            validation=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
            chunked=is_chunked,
            tensor_dataset=str(args.tensor_dataset),
            init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            validation_source_shards_file=str(args.validation_source_shards_file)
            if args.validation_source_shards_file is not None
            else None,
            start_epoch=start_epoch,
            candidate_head_variant=str(getattr(model, "candidate_head_variant", "n/a")),
            legacy_dropout=float(getattr(model, "legacy_dropout", 0.0)),
            disabled_token_types=list(getattr(model, "disabled_token_type_names", ())),
            auxiliary_reward_heads=bool(getattr(model, "auxiliary_reward_heads", False)),
            potion_residual_head=bool(getattr(model, "potion_residual_head_enabled", False)),
            potion_residual_clip=float(getattr(model, "potion_residual_clip", 0.0)),
            card_residual_head=bool(getattr(model, "card_residual_head_enabled", False)),
            card_residual_clip=float(getattr(model, "card_residual_clip", 0.0)),
            token_type_vocab_size=int(getattr(model, "token_type_vocab_size", 0)),
            amp_dtype=amp_dtype_name,
            allow_tf32=bool(args.allow_tf32),
            length_bucket_batches=bool(args.length_bucket_batches),
            length_bucket_window=int(args.length_bucket_window),
            teacher_q_clip=float(args.teacher_q_clip),
            rank_weight=float(args.rank_weight),
            q_weight=float(args.q_weight),
            pair_weight=float(args.pair_weight),
            bc_weight=float(args.bc_weight),
            hard_top_weight=float(args.hard_top_weight),
            top1_ce_weight=float(args.top1_ce_weight),
            top1_ce_min_teacher_gap=float(args.top1_ce_min_teacher_gap),
            anchor_guard_weight=float(args.anchor_guard_weight),
            trainable_report={
                key: value
                for key, value in trainable_report.items()
                if key != "matched_parameter_names"
            },
            trainable_parameter_names=trainable_report["matched_parameter_names"][:80],
        ),
    )

    for epoch in range(start_epoch, args.epochs + 1):
        _append_memory_log(memory_log, _memory_snapshot(torch, device=args.device, event="epoch_start", epoch=epoch))
        if is_chunked:
            train_metrics = _run_epoch_chunked(
                model,
                train_chunks,
                optimizer,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed + epoch,
                loss_kwargs=loss_kwargs,
                distill_model=distill_model,
                distill_model_kind=distill_model_kind,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
            length_bucket_batches=bool(args.length_bucket_batches),
            length_bucket_window=int(args.length_bucket_window),
            stage_chunks_on_device=bool(args.stage_chunks_on_device),
            progress_label=f"epoch {epoch:03d} train",
            progress_interval_chunks=args.progress_interval_chunks,
            progress_interval_seconds=args.progress_interval_seconds,
            memory_log=memory_log,
            )
        else:
            train_metrics = _run_epoch(
                model,
                payload,
                train_indices,
                optimizer,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed + epoch,
                loss_kwargs=loss_kwargs,
                distill_model=distill_model,
                distill_model_kind=distill_model_kind,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
                length_bucket_batches=bool(args.length_bucket_batches),
                length_bucket_window=int(args.length_bucket_window),
            )
        print(f"epoch {epoch:03d} train {_format_metrics(train_metrics)}", flush=True)
        validation_metrics = {}
        if validation_chunks if is_chunked else validation_indices:
            if is_chunked:
                validation_metrics = _run_epoch_chunked(
                    model,
                    validation_chunks,
                    None,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    loss_kwargs=loss_kwargs,
                    distill_model=distill_model,
                    distill_model_kind=distill_model_kind,
                    amp_dtype=amp_dtype,
                    grad_scaler=None,
                    length_bucket_batches=bool(args.length_bucket_batches),
                    length_bucket_window=int(args.length_bucket_window),
                    stage_chunks_on_device=bool(args.stage_chunks_on_device),
                    progress_label=f"epoch {epoch:03d} valid",
                    progress_interval_chunks=args.progress_interval_chunks,
                    progress_interval_seconds=args.progress_interval_seconds,
                    memory_log=memory_log,
                )
            else:
                validation_metrics = _run_epoch(
                    model,
                    payload,
                    validation_indices,
                    None,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    loss_kwargs=loss_kwargs,
                    distill_model=distill_model,
                    distill_model_kind=distill_model_kind,
                    amp_dtype=amp_dtype,
                    grad_scaler=None,
                    length_bucket_batches=bool(args.length_bucket_batches),
                    length_bucket_window=int(args.length_bucket_window),
                )
            print(f"epoch {epoch:03d} valid {_format_metrics(validation_metrics)}", flush=True)
            current = validation_metrics.get("loss")
            if current is not None and (best_validation is None or current < best_validation):
                best_validation = current
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if current is not None:
                if early_stop_best_validation is None or current < early_stop_best_validation - float(args.early_stop_min_delta):
                    early_stop_best_validation = current
                    early_stop_bad_epochs = 0
                else:
                    early_stop_bad_epochs += 1
        scheduler.step()
        _append_memory_log(
            memory_log,
            _memory_snapshot(
                torch,
                device=args.device,
                event="epoch_done",
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            ),
        )
        available = _mem_available_kb()
        if min_mem_available_kb > 0 and available is not None and available < min_mem_available_kb:
            stopped_for_low_memory = True
            print(
                "stopping after epoch "
                f"{epoch}: MemAvailable={available / 1024 / 1024:.2f}GB "
                f"< threshold={args.min_mem_available_gb:.2f}GB",
                flush=True,
            )
            if best_state_dict is None:
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if args.save_each_epoch:
                epoch_output_dir.mkdir(parents=True, exist_ok=True)
                _save_training_checkpoint(
                    epoch_output_dir / f"epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    args=args,
                    root_count=root_count,
                    train_count=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
                    validation_count=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
                    is_chunked=is_chunked,
                    current_epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_validation=best_validation,
                    best_epoch=best_epoch,
                    best_state_dict=best_state_dict,
                    stopped_for_low_memory=stopped_for_low_memory,
                    stopped_for_early_stop=stopped_for_early_stop,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                    early_stop_best_validation=early_stop_best_validation,
                    init_checkpoint_metadata=init_checkpoint_metadata,
                )
                _save_training_checkpoint(
                    epoch_output_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    args=args,
                    root_count=root_count,
                    train_count=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
                    validation_count=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
                    is_chunked=is_chunked,
                    current_epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_validation=best_validation,
                    best_epoch=best_epoch,
                    best_state_dict=best_state_dict,
                    stopped_for_low_memory=stopped_for_low_memory,
                    stopped_for_early_stop=stopped_for_early_stop,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                    early_stop_best_validation=early_stop_best_validation,
                    init_checkpoint_metadata=init_checkpoint_metadata,
                )
            break
        if (
            args.early_stop_patience > 0
            and validation_metrics
            and early_stop_bad_epochs >= int(args.early_stop_patience)
        ):
            stopped_for_early_stop = True
            print(
                "stopping after epoch "
                f"{epoch}: validation loss did not improve by at least {args.early_stop_min_delta:.6f} "
                f"for {early_stop_bad_epochs} epochs "
                f"(best epoch={best_epoch}, best loss={best_validation:.6f})",
                flush=True,
            )
            _append_memory_log(
                memory_log,
                _memory_snapshot(
                    torch,
                    device=args.device,
                    event="early_stop",
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                    early_stop_best_validation_loss=early_stop_best_validation,
                ),
            )
            if args.save_each_epoch:
                epoch_output_dir.mkdir(parents=True, exist_ok=True)
                _save_training_checkpoint(
                    epoch_output_dir / f"epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    args=args,
                    root_count=root_count,
                    train_count=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
                    validation_count=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
                    is_chunked=is_chunked,
                    current_epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_validation=best_validation,
                    best_epoch=best_epoch,
                    best_state_dict=best_state_dict,
                    stopped_for_low_memory=stopped_for_low_memory,
                    stopped_for_early_stop=stopped_for_early_stop,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                    early_stop_best_validation=early_stop_best_validation,
                    init_checkpoint_metadata=init_checkpoint_metadata,
                )
                _save_training_checkpoint(
                    epoch_output_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    args=args,
                    root_count=root_count,
                    train_count=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
                    validation_count=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
                    is_chunked=is_chunked,
                    current_epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_validation=best_validation,
                    best_epoch=best_epoch,
                    best_state_dict=best_state_dict,
                    stopped_for_low_memory=stopped_for_low_memory,
                    stopped_for_early_stop=stopped_for_early_stop,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                    early_stop_best_validation=early_stop_best_validation,
                    init_checkpoint_metadata=init_checkpoint_metadata,
                )
            break
        if args.save_each_epoch:
            epoch_output_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = epoch_output_dir / f"epoch_{epoch:03d}.pt"
            _save_training_checkpoint(
                epoch_path,
                model,
                optimizer,
                scheduler,
                args=args,
                root_count=root_count,
                train_count=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
                validation_count=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
                is_chunked=is_chunked,
                current_epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                best_validation=best_validation,
                best_epoch=best_epoch,
                best_state_dict=best_state_dict,
                stopped_for_low_memory=stopped_for_low_memory,
                stopped_for_early_stop=stopped_for_early_stop,
                early_stop_bad_epochs=early_stop_bad_epochs,
                early_stop_best_validation=early_stop_best_validation,
                init_checkpoint_metadata=init_checkpoint_metadata,
            )
            _save_training_checkpoint(
                epoch_output_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                args=args,
                root_count=root_count,
                train_count=sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
                validation_count=sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
                is_chunked=is_chunked,
                current_epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                best_validation=best_validation,
                best_epoch=best_epoch,
                best_state_dict=best_state_dict,
                stopped_for_low_memory=stopped_for_low_memory,
                stopped_for_early_stop=stopped_for_early_stop,
                early_stop_bad_epochs=early_stop_bad_epochs,
                early_stop_best_validation=early_stop_best_validation,
                init_checkpoint_metadata=init_checkpoint_metadata,
            )
            print(f"epoch {epoch:03d} checkpoint {epoch_path}", flush=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_v3_combat_transformer_checkpoint(
        args.output,
        model,
        training_args=vars(args),
        dataset_metadata={
            "root_count": root_count,
            "train_count": sum(int(chunk.get("root_count") or 0) for chunk in train_chunks) if is_chunked else len(train_indices),
            "validation_count": sum(int(chunk.get("root_count") or 0) for chunk in validation_chunks) if is_chunked else len(validation_indices),
            "chunked_dataset": is_chunked,
            "best_validation_loss": best_validation,
            "best_epoch": best_epoch,
            "stopped_for_low_memory": stopped_for_low_memory,
            "stopped_for_early_stop": stopped_for_early_stop,
            "early_stop_bad_epochs": early_stop_bad_epochs,
            "early_stop_best_validation_loss": early_stop_best_validation,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "init_checkpoint_best_validation_loss": init_checkpoint_metadata.get("best_validation_loss") if init_checkpoint_metadata else None,
            "source_tensor_dataset": str(args.tensor_dataset),
        },
    )
    _append_memory_log(
        memory_log,
        _memory_snapshot(
            torch,
            device=args.device,
            event="training_done",
            best_epoch=best_epoch,
            best_validation_loss=best_validation,
            stopped_for_low_memory=stopped_for_low_memory,
            stopped_for_early_stop=stopped_for_early_stop,
            early_stop_bad_epochs=early_stop_bad_epochs,
            early_stop_best_validation_loss=early_stop_best_validation,
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "roots": root_count,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "stopped_for_low_memory": stopped_for_low_memory,
                "stopped_for_early_stop": stopped_for_early_stop,
                "memory_log": str(memory_log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
