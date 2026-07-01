#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import os
import random
import signal
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.ai.torch_compat import require_torch
from spirecomm.ai.v3_combat_ppo import (
    TERMINAL_PHASES,
    V3PPOCombatSelector,
    collate_ppo_roots,
    compute_gae_for_trajectories,
    count_trainable_parameters,
    export_policy_transformer_checkpoint,
    first_candidate_features_from_records,
    load_base_transformer_for_ppo,
    load_ppo_policy_checkpoint,
    make_ppo_policy_from_transformer,
    ppo_loss,
    save_ppo_policy_checkpoint,
    set_ppo_trainable,
)
from spirecomm.native_sim_v3 import NativeRunEnv


_WORKER_CONFIG: dict[str, Any] = {}
_SELECTORS: dict[str, Any] | None = None
_PPO_SELECTOR: V3PPOCombatSelector | None = None


def _arm_parent_death_signal() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        # PR_SET_PDEATHSIG: terminate rollout workers if the PPO parent process dies.
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)
        if os.getppid() == 1:
            os._exit(143)
    except Exception:
        pass


def _release_large_update_objects() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _apply_runtime_env(config: dict[str, Any]) -> None:
    for env_name, config_key in (
        ("SPIRECOMM_SHOP_POLICY", "shop_policy"),
        ("SPIRECOMM_SHOP_VALUE_PRICE_COST", "shop_value_price_cost"),
        ("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "shop_value_reserve_shortfall_cost"),
        ("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "shop_value_future_shop_reserve"),
        ("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "shop_value_future_shop_horizon"),
        ("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "shop_value_card_scale"),
        ("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "shop_value_card_reference_price"),
        ("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "shop_value_card_price_factor_min"),
        ("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "shop_value_card_price_factor_max"),
        ("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "shop_value_potion_scale"),
        ("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "shop_value_relic_scale"),
        ("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "shop_value_item_scale"),
        ("SPIRECOMM_SHOP_VALUE_THRESHOLD", "shop_value_threshold"),
        ("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "shop_prior_weight_override"),
        ("SPIRECOMM_REWARD_POTION_FULL_REPLACE", "reward_potion_full_replace"),
    ):
        if config_key in config and config[config_key] is not None:
            os.environ[env_name] = str(config[config_key])


def _init_rollout_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _SELECTORS, _PPO_SELECTOR
    _arm_parent_death_signal()
    _WORKER_CONFIG = dict(config)
    _apply_runtime_env(_WORKER_CONFIG)
    torch_threads = int(_WORKER_CONFIG.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
            torch = require_torch()
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(_WORKER_CONFIG["repo_root"]),
        device=str(_WORKER_CONFIG["noncombat_device"]),
        combat_device=str(_WORKER_CONFIG["rollout_device"]),
        combat_selector="v3-teacher",
        card_reward_model=Path(_WORKER_CONFIG["card_reward_model"]),
        shop_model=Path(_WORKER_CONFIG["shop_choice_model"]),
    )
    _PPO_SELECTOR = V3PPOCombatSelector(
        _WORKER_CONFIG["ppo_checkpoint"],
        device=str(_WORKER_CONFIG["rollout_device"]),
        temperature=float(_WORKER_CONFIG["temperature"]),
        normal_room_potion_penalty=float(_WORKER_CONFIG["normal_room_potion_penalty"]),
        sample=bool(_WORKER_CONFIG.get("sample", True)),
        compact_records=bool(_WORKER_CONFIG.get("compact_records", True)),
        seed=int(_WORKER_CONFIG["sample_seed"]) + os.getpid(),
    )
    _SELECTORS["combat"] = _PPO_SELECTOR


def _room_clear_reward(room_type: str, config: dict[str, Any]) -> float:
    normalized = str(room_type or "")
    if "Boss" in normalized:
        return float(config["boss_clear_reward"])
    if "Elite" in normalized:
        return float(config["elite_clear_reward"])
    return float(config["combat_clear_reward"])


def _combat_step_reward(
    *,
    pre_phase: str,
    pre_room_type: str,
    pre_hp: int,
    post_phase: str,
    post_hp: int,
    config: dict[str, Any],
) -> tuple[float, bool]:
    reward = float(config["hp_delta_scale"]) * float(post_hp - pre_hp)
    terminal_applied = False
    if pre_phase == "COMBAT" and post_phase != "COMBAT":
        reward += _room_clear_reward(pre_room_type, config)
    if post_phase in {"COMPLETE", "VICTORY"}:
        reward += float(config["victory_reward"])
        terminal_applied = True
    elif post_phase == "GAME_OVER":
        reward -= float(config["death_penalty"])
        terminal_applied = True
    return reward, terminal_applied


def _bootstrap_value_from_env(env: NativeRunEnv, config: dict[str, Any]) -> float | None:
    assert _SELECTORS is not None
    assert _PPO_SELECTOR is not None
    max_steps = max(0, int(config.get("bootstrap_max_steps") or 0))
    for step_index in range(max_steps + 1):
        phase = str(getattr(env, "phase", ""))
        if phase in TERMINAL_PHASES or int(getattr(env, "floor", 0)) > int(config["max_floor"]):
            return 0.0
        if phase == "COMBAT":
            return _PPO_SELECTOR.value_env(env)
        if step_index >= max_steps:
            return None
        try:
            action, _scores, _source = choose_model_required_action(env, _SELECTORS, return_scores=False)
            env.step(action)
        except Exception:
            return None
    return None


def _collect_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    assert _PPO_SELECTOR is not None
    started = time.time()
    config = _WORKER_CONFIG
    trajectory: list[dict[str, Any]] = []
    action_kinds: Counter[str] = Counter()
    room_types: Counter[str] = Counter()
    error = None
    error_traceback = None
    truncated_for_roots = False
    record_trajectory = bool(config.get("record_trajectory", True))
    root_count = 0
    max_roots_per_seed = int(config.get("max_roots_per_seed") or 0)
    env = NativeRunEnv(seed=int(seed), ascension_level=int(config["ascension"]), enable_neow=True)
    for step_index in range(int(config["max_steps"])):
        if str(env.phase) in TERMINAL_PHASES or int(env.floor) > int(config["max_floor"]):
            break
        if max_roots_per_seed > 0 and root_count >= max_roots_per_seed:
            truncated_for_roots = True
            if trajectory and not bool(trajectory[-1].get("done")):
                bootstrap_value = _bootstrap_value_from_env(env, config)
                if bootstrap_value is not None:
                    trajectory[-1]["bootstrap_value"] = float(bootstrap_value)
            break
        pre_phase = str(env.phase)
        pre_floor = int(env.floor)
        pre_hp = int(env.player.current_hp)
        pre_room_type = str(getattr(env, "current_room_type", "") or pre_phase)
        _PPO_SELECTOR.last_decision = None
        _PPO_SELECTOR.rng.seed(int(config["sample_seed"]) + int(seed) * 1_000_003 + step_index)
        try:
            action, _scores, source = choose_model_required_action(env, _SELECTORS, return_scores=False)
            decision = copy.deepcopy(_PPO_SELECTOR.last_decision) if pre_phase == "COMBAT" and source == "combat" else None
            env.step(action)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            break
        post_phase = str(env.phase)
        post_hp = int(env.player.current_hp)
        if decision is not None:
            root_count += 1
            reward, terminal_applied = _combat_step_reward(
                pre_phase=pre_phase,
                pre_room_type=pre_room_type,
                pre_hp=pre_hp,
                post_phase=post_phase,
                post_hp=post_hp,
                config=config,
            )
            decision.update(
                {
                    "seed": int(seed),
                    "step": int(step_index),
                    "floor": int(pre_floor),
                    "room_type": pre_room_type,
                    "reward": float(reward),
                    "done": post_phase in TERMINAL_PHASES,
                    "terminal_reward_applied": bool(terminal_applied),
                    "post_phase": post_phase,
                    "post_floor": int(env.floor),
                    "post_hp": int(post_hp),
                }
            )
            if record_trajectory:
                trajectory.append(decision)
            action_kinds[str(decision.get("action_kind") or "")] += 1
            room_types[pre_room_type] += 1
    if record_trajectory and trajectory:
        if str(env.phase) in {"COMPLETE", "VICTORY"} and not bool(trajectory[-1].get("terminal_reward_applied")):
            trajectory[-1]["reward"] = float(trajectory[-1].get("reward") or 0.0) + float(config["victory_reward"])
            trajectory[-1]["done"] = True
        elif str(env.phase) == "GAME_OVER" and not bool(trajectory[-1].get("terminal_reward_applied")):
            trajectory[-1]["reward"] = float(trajectory[-1].get("reward") or 0.0) - float(config["death_penalty"])
            trajectory[-1]["done"] = True
    return {
        "seed": int(seed),
        "floor": int(env.floor),
        "phase": str(env.phase),
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "hp": int(env.player.current_hp),
        "roots": int(root_count),
        "truncated_for_roots": bool(truncated_for_roots),
        "trajectory": trajectory if record_trajectory else [],
        "action_kinds": dict(action_kinds),
        "room_types": dict(room_types),
        "seconds": time.time() - started,
        "error": error,
        "error_traceback": error_traceback,
    }


def _collect_rollouts(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    started = time.time()
    phase = str(config.get("progress_phase") or "rollout")
    update_index = int(config.get("update_index") or 0)
    progress_interval = int(config.get("progress_interval_seeds") or 0)

    def emit_progress(results: list[dict[str, Any]], *, force: bool = False) -> None:
        completed = len(results)
        if completed <= 0:
            return
        if not force and (progress_interval <= 0 or completed % progress_interval != 0):
            return
        summary = _summarize_rollout(results)
        payload = {
            "event": "collect_progress",
            "phase": phase,
            "update": update_index,
            "completed": completed,
            "total": len(seeds),
            "mean_floor": float(summary.get("mean_floor") or 0.0),
            "root_count": int(summary.get("root_count") or 0),
            "win_count": int(summary.get("win_count") or 0),
            "death_count": int(summary.get("death_count") or 0),
            "error_count": int(summary.get("error_count") or 0),
            "elapsed_seconds": time.time() - started,
        }
        print(
            f"update={update_index} {phase} progress "
            f"seeds={completed}/{len(seeds)} "
            f"mean_floor={payload['mean_floor']:.2f} "
            f"roots={payload['root_count']} "
            f"wins={payload['win_count']} "
            f"errors={payload['error_count']}",
            flush=True,
        )
        progress_path = str(config.get("progress_path") or "")
        if progress_path:
            _write_jsonl(Path(progress_path), payload)

    workers = max(1, int(config["workers"]))
    if workers <= 1:
        _init_rollout_worker(config)
        results: list[dict[str, Any]] = []
        for seed in seeds:
            results.append(_collect_seed(seed))
            emit_progress(results)
        emit_progress(results, force=True)
        return results
    results: list[dict[str, Any]] = []
    recycle_interval = max(1, int(config.get("rollout_worker_recycle_interval") or 1))
    chunk_size = max(1, workers * recycle_interval)
    for start in range(0, len(seeds), chunk_size):
        seed_chunk = seeds[start : start + chunk_size]
        with ProcessPoolExecutor(
            max_workers=min(workers, len(seed_chunk)),
            initializer=_init_rollout_worker,
            initargs=(config,),
        ) as executor:
            futures = {executor.submit(_collect_seed, seed): seed for seed in seed_chunk}
            for future in as_completed(futures):
                results.append(future.result())
                emit_progress(results)
    results.sort(key=lambda item: int(item["seed"]))
    emit_progress(results, force=True)
    return results


def _summarize_rollout(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    floors = [int(result["floor"]) for result in seed_results]
    roots = [int(result["roots"]) for result in seed_results]
    action_kinds: Counter[str] = Counter()
    room_types: Counter[str] = Counter()
    for result in seed_results:
        action_kinds.update(result.get("action_kinds") or {})
        room_types.update(result.get("room_types") or {})
    bootstrap_values = [
        float(root["bootstrap_value"])
        for result in seed_results
        for root in list(result.get("trajectory") or [])
        if root.get("bootstrap_value") is not None
    ]
    return {
        "seed_count": len(seed_results),
        "root_count": sum(roots),
        "mean_floor": mean(floors) if floors else 0.0,
        "min_floor": min(floors) if floors else 0,
        "max_floor": max(floors) if floors else 0,
        "win_count": sum(1 for result in seed_results if result.get("won")),
        "death_count": sum(1 for result in seed_results if result.get("dead")),
        "error_count": sum(1 for result in seed_results if result.get("error")),
        "truncated_count": sum(1 for result in seed_results if result.get("truncated_for_roots")),
        "bootstrap_count": len(bootstrap_values),
        "mean_bootstrap_value": mean(bootstrap_values) if bootstrap_values else 0.0,
        "mean_roots_per_seed": mean(roots) if roots else 0.0,
        "action_kinds": dict(action_kinds.most_common()),
        "room_types": dict(room_types.most_common()),
        "seconds": sum(float(result.get("seconds") or 0.0) for result in seed_results),
    }


def _batch_roots(roots: list[dict[str, Any]], *, batch_size: int, rng: random.Random) -> list[list[dict[str, Any]]]:
    shuffled = list(roots)
    rng.shuffle(shuffled)
    return [shuffled[index : index + batch_size] for index in range(0, len(shuffled), batch_size)]


def _average_metrics(weighted_metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total = sum(weight for weight, _metrics in weighted_metrics)
    if total <= 0:
        return {}
    keys = sorted({key for _weight, metrics in weighted_metrics for key in metrics})
    return {
        key: sum(float(metrics.get(key, 0.0)) * weight for weight, metrics in weighted_metrics) / total
        for key in keys
    }


def _collate_ppo_value_roots(roots: list[dict[str, Any]], *, device: str, state_dim: int) -> dict[str, Any]:
    torch = require_torch()
    before_summaries: list[list[float]] = []
    returns: list[float] = []
    for root in roots:
        records = root.get("candidate_records")
        if not records:
            continue
        features = first_candidate_features_from_records(records)
        if len(features) < int(state_dim):
            continue
        before_summaries.append([float(value) for value in features[: int(state_dim)]])
        returns.append(float(root["return"]))
    if not before_summaries:
        raise ValueError("cannot collate empty PPO value root batch")
    return {
        "before_summary": torch.tensor(before_summaries, dtype=torch.float32, device=device),
        "returns": torch.tensor(returns, dtype=torch.float32, device=device),
    }


def _run_value_warmup(
    *,
    policy: Any,
    optimizer: Any,
    roots: list[dict[str, Any]],
    args: argparse.Namespace,
    update_index: int,
) -> dict[str, float]:
    epochs = int(args.value_warmup_epochs)
    if epochs <= 0:
        return {}
    torch = require_torch()
    rng = random.Random(int(args.seed) + update_index * 6151 + 97)
    batch_size = int(args.value_warmup_batch_size or args.batch_size)
    policy.train()
    weighted_metrics: list[tuple[int, dict[str, float]]] = []
    progress_interval = int(getattr(args, "progress_interval_batches", 0) or 0)
    completed_batches = 0
    for epoch in range(1, epochs + 1):
        batches = _batch_roots(roots, batch_size=batch_size, rng=rng)
        for root_batch in batches:
            batch = _collate_ppo_value_roots(
                root_batch,
                device=str(args.train_device),
                state_dim=int(getattr(policy, "state_dim", 161)),
            )
            values = policy.root_values_from_before_summary(batch["before_summary"])
            returns = batch["returns"].to(dtype=values.dtype)
            value_loss = torch.nn.functional.smooth_l1_loss(values.float(), returns.float())
            optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in policy.value_head.parameters() if parameter.requires_grad],
                float(args.max_grad_norm),
            )
            optimizer.step()
            with torch.no_grad():
                return_var = torch.var(returns.float(), unbiased=False)
                value_error_var = torch.var((returns.float() - values.float()), unbiased=False)
                explained_variance = 1.0 - value_error_var / torch.clamp(return_var, min=1.0e-8)
            weighted_metrics.append(
                (
                    int(returns.numel()),
                    {
                        "value_warmup_loss": float(value_loss.detach().cpu().item()),
                        "value_warmup_explained_variance": float(explained_variance.detach().cpu().item()),
                    },
                )
            )
            completed_batches += 1
            if progress_interval > 0 and completed_batches % progress_interval == 0:
                payload = {
                    "event": "value_warmup_progress",
                    "update": int(update_index),
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "batch": int(completed_batches),
                    "loss": float(value_loss.detach().cpu().item()),
                    "explained_variance": float(explained_variance.detach().cpu().item()),
                }
                print(
                    f"update={update_index} value_warmup "
                    f"epoch={epoch}/{epochs} batch={completed_batches} "
                    f"loss={payload['loss']:.4f} ev={payload['explained_variance']:.4f}",
                    flush=True,
                )
                progress_path = str(getattr(args, "progress_path", "") or "")
                if progress_path:
                    _write_jsonl(Path(progress_path), payload)
    metrics = _average_metrics(weighted_metrics)
    metrics["value_warmup_epochs_completed"] = float(epochs)
    return metrics


def _run_ppo_update(
    *,
    policy: Any,
    reference_model: Any,
    optimizer: Any,
    roots: list[dict[str, Any]],
    args: argparse.Namespace,
    update_index: int,
) -> dict[str, float]:
    torch = require_torch()
    rng = random.Random(int(args.seed) + update_index * 7919)
    policy.train()
    weighted_metrics: list[tuple[int, dict[str, float]]] = []
    stopped_for_kl = False
    stopped_for_clip_fraction = False
    progress_interval = int(getattr(args, "progress_interval_batches", 0) or 0)
    completed_batches = 0
    for epoch in range(1, int(args.ppo_epochs) + 1):
        for root_batch in _batch_roots(roots, batch_size=int(args.batch_size), rng=rng):
            batch = collate_ppo_roots(root_batch, device=str(args.train_device))
            loss, metrics = ppo_loss(
                policy,
                batch,
                clip_eps=float(args.clip_eps),
                value_coef=float(args.value_coef),
                entropy_coef=float(args.entropy_coef),
                kl_coef=float(args.kl_coef),
                temperature=float(args.temperature),
                reference_model=reference_model,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in policy.parameters() if parameter.requires_grad], float(args.max_grad_norm))
            optimizer.step()
            weighted_metrics.append((len(root_batch), metrics))
            completed_batches += 1
            if progress_interval > 0 and completed_batches % progress_interval == 0:
                payload = {
                    "event": "ppo_batch_progress",
                    "update": int(update_index),
                    "epoch": int(epoch),
                    "epochs": int(args.ppo_epochs),
                    "batch": int(completed_batches),
                    "loss": float(metrics.get("loss", 0.0)),
                    "policy_loss": float(metrics.get("policy_loss", 0.0)),
                    "value_loss": float(metrics.get("value_loss", 0.0)),
                    "approx_kl": float(metrics.get("approx_kl", 0.0)),
                    "kl_to_reference": float(metrics.get("kl_to_reference", 0.0)),
                    "clip_fraction": float(metrics.get("clip_fraction", 0.0)),
                    "entropy": float(metrics.get("entropy", 0.0)),
                }
                print(
                    f"update={update_index} ppo_batch "
                    f"epoch={epoch}/{args.ppo_epochs} batch={completed_batches} "
                    f"loss={payload['loss']:.4f} "
                    f"kl={payload['approx_kl']:.4f} "
                    f"ref_kl={payload['kl_to_reference']:.4f} "
                    f"clip={payload['clip_fraction']:.3f}",
                    flush=True,
                )
                progress_path = str(getattr(args, "progress_path", "") or "")
                if progress_path:
                    _write_jsonl(Path(progress_path), payload)
            if float(args.target_kl) > 0.0 and float(metrics.get("approx_kl", 0.0)) > float(args.target_kl):
                stopped_for_kl = True
                break
            if float(args.max_clip_fraction) > 0.0 and float(metrics.get("clip_fraction", 0.0)) > float(args.max_clip_fraction):
                stopped_for_clip_fraction = True
                break
        if stopped_for_kl or stopped_for_clip_fraction:
            break
    metrics = _average_metrics(weighted_metrics)
    metrics["ppo_epochs_completed"] = float(epoch)
    metrics["stopped_for_kl"] = float(1 if stopped_for_kl else 0)
    metrics["stopped_for_clip_fraction"] = float(1 if stopped_for_clip_fraction else 0)
    return metrics


def _save_all(
    *,
    args: argparse.Namespace,
    policy: Any,
    optimizer: Any,
    update_index: int,
    seed_cursor: int,
    metrics: dict[str, Any],
    best_eval: dict[str, Any] | None,
    ppo_checkpoint: Path,
) -> None:
    dataset_metadata = {
        "algorithm": "ppo",
        "update": int(update_index),
        "seed_cursor": int(seed_cursor),
        "init_checkpoint": str(args.init_checkpoint),
        "reference_checkpoint": str(args.reference_checkpoint or args.init_checkpoint),
        "trainable_mode": str(args.trainable_mode),
        "metrics": dict(metrics),
        "best_eval": dict(best_eval or {}),
    }
    export_policy_transformer_checkpoint(
        args.output,
        policy,
        training_args=vars(args),
        dataset_metadata=dataset_metadata,
    )
    save_ppo_policy_checkpoint(
        ppo_checkpoint,
        policy,
        optimizer_state_dict=optimizer.state_dict(),
        training_state={
            "current_update": int(update_index),
            "seed_cursor": int(seed_cursor),
            "metrics": dict(metrics),
            "best_eval": dict(best_eval or {}),
        },
        training_args=vars(args),
        dataset_metadata=dataset_metadata,
    )


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _default_best_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_best_eval{path.suffix}")


def _unlink_if_unneeded(path: Path, *, keep: bool) -> None:
    if keep:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Combat-only PPO fine-tuning for the old v3 actionset transformer.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("models/cache/v3_combat_transformer_stage5_v8_potion_pair_200k_actionset_best_epoch011.pt"),
    )
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-ppo-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("models/v3_combat_ppo_actionset.pt"))
    parser.add_argument("--ppo-checkpoint", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--seeds-per-update", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=10001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--rollout-worker-recycle-interval",
        type=int,
        default=1,
        help="Restart rollout worker processes after this many seeds per worker. This bounds Python RSS growth during long PPO collection.",
    )
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--max-roots-per-seed",
        type=int,
        default=256,
        help="Stop collecting a seed after this many PPO combat roots; <=0 keeps full runs. This bounds rollout RAM.",
    )
    parser.add_argument(
        "--bootstrap-max-steps",
        type=int,
        default=200,
        help="When max-roots truncates a seed outside combat, advance fixed non-combat policy up to this many steps to bootstrap the next combat value.",
    )
    parser.add_argument("--train-device", default="cuda" if require_torch().cuda.is_available() else "cpu")
    parser.add_argument("--rollout-device", default="cpu")
    parser.add_argument("--noncombat-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--trainable-mode", choices=["heads", "action-set", "full"], default="action-set")
    parser.add_argument("--value-hidden-dim", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--value-warmup-epochs", type=int, default=1)
    parser.add_argument("--value-warmup-batch-size", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.10)
    parser.add_argument("--max-clip-fraction", type=float, default=0.25)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-after-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-seed-start", type=int, default=1)
    parser.add_argument("--eval-seeds", type=int, default=60)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--eval-max-roots-per-seed", type=int, default=0)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--best-output", type=Path, default=None)
    parser.add_argument("--best-ppo-checkpoint", type=Path, default=None)
    parser.add_argument("--keep-rollout-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-eval-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-jsonl", type=Path, default=None)
    parser.add_argument("--progress-interval-seeds", type=int, default=8)
    parser.add_argument("--progress-interval-batches", type=int, default=8)
    parser.add_argument("--normal-room-potion-penalty", type=float, default=1.5)
    parser.add_argument("--combat-clear-reward", type=float, default=1.0)
    parser.add_argument("--elite-clear-reward", type=float, default=1.0)
    parser.add_argument("--boss-clear-reward", type=float, default=2.0)
    parser.add_argument("--victory-reward", type=float, default=20.0)
    parser.add_argument("--death-penalty", type=float, default=3.0)
    parser.add_argument("--hp-delta-scale", type=float, default=0.03)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-each-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument(
        "--shop-choice-model",
        type=Path,
        default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")),
    )
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
    parser.add_argument("--reward-potion-full-replace", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch = require_torch()
    if str(args.train_device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    work_dir = args.work_dir or args.output.with_suffix(args.output.suffix + ".ppo_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    ppo_checkpoint = args.ppo_checkpoint or args.output.with_suffix(args.output.suffix + ".ppo.pt")
    best_output = args.best_output or _default_best_path(args.output)
    best_ppo_checkpoint = args.best_ppo_checkpoint or _default_best_path(ppo_checkpoint)
    metrics_path = work_dir / "ppo_metrics.jsonl"
    progress_path = args.progress_jsonl or (work_dir / "ppo_progress.jsonl")
    setattr(args, "progress_path", str(progress_path))

    start_update = 1
    seed_cursor = int(args.seed_start)
    best_eval: dict[str, Any] = {}
    best_eval_mean_floor = float("-inf")
    if args.resume_ppo_checkpoint is not None:
        policy, resume_checkpoint = load_ppo_policy_checkpoint(args.resume_ppo_checkpoint, device=str(args.train_device))
        training_state = dict(resume_checkpoint.get("training_state") or {})
        start_update = int(training_state.get("current_update") or 0) + 1
        seed_cursor = int(training_state.get("seed_cursor") or seed_cursor)
        best_eval = dict(training_state.get("best_eval") or {})
        if best_eval:
            best_eval_mean_floor = float(best_eval.get("mean_floor", float("-inf")))
    else:
        policy, _init_checkpoint = make_ppo_policy_from_transformer(
            args.init_checkpoint,
            device=str(args.train_device),
            value_hidden_dim=int(args.value_hidden_dim) or None,
        )
    set_ppo_trainable(policy, str(args.trainable_mode))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    if args.resume_ppo_checkpoint is not None:
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if hasattr(value, "to"):
                        state[key] = value.to(str(args.train_device))

    reference_model, _reference_checkpoint = load_base_transformer_for_ppo(
        args.reference_checkpoint or args.init_checkpoint,
        device=str(args.train_device),
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    print(
        f"loaded PPO policy trainable_mode={args.trainable_mode} "
        f"trainable_params={count_trainable_parameters(policy)} "
        f"updates={args.updates} seeds_per_update={args.seeds_per_update} "
        f"max_roots_per_seed={args.max_roots_per_seed} "
        f"temperature={args.temperature} entropy_coef={args.entropy_coef} "
        f"lr={args.learning_rate} eval_after_update={args.eval_after_update} "
        f"train_device={args.train_device} rollout_device={args.rollout_device}",
        flush=True,
    )

    last_metrics: dict[str, Any] = {}
    _save_all(
        args=args,
        policy=policy,
        optimizer=optimizer,
        update_index=start_update - 1,
        seed_cursor=seed_cursor,
        metrics={"event": "initial_export"},
        best_eval=best_eval,
        ppo_checkpoint=ppo_checkpoint,
    )
    if int(args.updates) <= 0:
        print(f"exported initial policy to {args.output} and {ppo_checkpoint}", flush=True)
        return

    for update_index in range(start_update, int(args.updates) + 1):
        rollout_checkpoint = work_dir / f"rollout_policy_update_{update_index:04d}.ppo.pt"
        save_ppo_policy_checkpoint(
            rollout_checkpoint,
            policy,
            optimizer_state_dict=optimizer.state_dict(),
            training_state={"current_update": update_index - 1, "seed_cursor": seed_cursor},
            training_args=vars(args),
        )
        seeds = list(range(seed_cursor, seed_cursor + int(args.seeds_per_update)))
        seed_cursor += int(args.seeds_per_update)
        rollout_config = {
            "repo_root": str(args.repo_root),
            "ppo_checkpoint": str(rollout_checkpoint),
            "ascension": int(args.ascension),
            "max_floor": int(args.max_floor),
            "max_steps": int(args.max_steps),
            "max_roots_per_seed": int(args.max_roots_per_seed),
            "bootstrap_max_steps": int(args.bootstrap_max_steps),
            "workers": int(args.workers),
            "progress_path": str(progress_path),
            "progress_phase": "rollout",
            "progress_interval_seeds": int(args.progress_interval_seeds),
            "update_index": int(update_index),
            "rollout_worker_recycle_interval": int(args.rollout_worker_recycle_interval),
            "torch_threads": int(args.torch_threads),
            "rollout_device": str(args.rollout_device),
            "noncombat_device": str(args.noncombat_device),
            "temperature": float(args.temperature),
            "sample": True,
            "record_trajectory": True,
            "compact_records": True,
            "normal_room_potion_penalty": float(args.normal_room_potion_penalty),
            "sample_seed": int(args.seed) + update_index * 1009,
            "combat_clear_reward": float(args.combat_clear_reward),
            "elite_clear_reward": float(args.elite_clear_reward),
            "boss_clear_reward": float(args.boss_clear_reward),
            "victory_reward": float(args.victory_reward),
            "death_penalty": float(args.death_penalty),
            "hp_delta_scale": float(args.hp_delta_scale),
            "card_reward_model": str(args.card_reward_model),
            "shop_choice_model": str(args.shop_choice_model),
            "shop_policy": str(args.shop_policy),
            "shop_value_price_cost": float(args.shop_value_price_cost),
            "shop_value_reserve_shortfall_cost": float(args.shop_value_reserve_shortfall_cost),
            "shop_value_future_shop_reserve": int(args.shop_value_future_shop_reserve),
            "shop_value_future_shop_horizon": int(args.shop_value_future_shop_horizon),
            "shop_value_card_scale": float(args.shop_value_card_scale),
            "shop_value_card_reference_price": float(args.shop_value_card_reference_price),
            "shop_value_card_price_factor_min": float(args.shop_value_card_price_factor_min),
            "shop_value_card_price_factor_max": float(args.shop_value_card_price_factor_max),
            "shop_value_potion_scale": float(args.shop_value_potion_scale),
            "shop_value_relic_scale": float(args.shop_value_relic_scale),
            "shop_value_item_scale": float(args.shop_value_item_scale),
            "shop_value_threshold": float(args.shop_value_threshold),
            "shop_prior_weight_override": float(args.shop_prior_weight_override),
            "reward_potion_full_replace": str(bool(args.reward_potion_full_replace)).lower(),
        }
        collect_started = time.time()
        seed_results = _collect_rollouts(rollout_config, seeds)
        _unlink_if_unneeded(rollout_checkpoint, keep=bool(args.keep_rollout_checkpoints))
        rollout_summary = _summarize_rollout(seed_results)
        trajectories = [result["trajectory"] for result in seed_results if result.get("trajectory")]
        roots = compute_gae_for_trajectories(
            trajectories,
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            normalize=True,
        )
        if not roots:
            raise RuntimeError(f"update {update_index} collected no PPO combat roots")
        del seed_results, trajectories
        _release_large_update_objects()
        value_warmup_metrics = _run_value_warmup(
            policy=policy,
            optimizer=optimizer,
            roots=roots,
            args=args,
            update_index=update_index,
        )
        train_metrics = _run_ppo_update(
            policy=policy,
            reference_model=reference_model,
            optimizer=optimizer,
            roots=roots,
            args=args,
            update_index=update_index,
        )
        if value_warmup_metrics:
            train_metrics.update(value_warmup_metrics)
        eval_summary: dict[str, Any] = {}
        best_eval_improved = False
        if bool(args.eval_after_update) and int(args.eval_seeds) > 0:
            eval_checkpoint = work_dir / f"eval_policy_update_{update_index:04d}.ppo.pt"
            save_ppo_policy_checkpoint(
                eval_checkpoint,
                policy,
                optimizer_state_dict=None,
                training_state={"current_update": int(update_index), "seed_cursor": int(seed_cursor), "best_eval": dict(best_eval)},
                training_args=vars(args),
            )
            eval_seeds = list(range(int(args.eval_seed_start), int(args.eval_seed_start) + int(args.eval_seeds)))
            eval_config = dict(rollout_config)
            eval_config.update(
                {
                    "ppo_checkpoint": str(eval_checkpoint),
                    "max_roots_per_seed": int(args.eval_max_roots_per_seed),
                    "workers": int(args.eval_workers or args.workers),
                    "progress_phase": "eval",
                    "progress_interval_seeds": int(args.progress_interval_seeds),
                    "update_index": int(update_index),
                    "temperature": float(args.eval_temperature),
                    "sample": False,
                    "record_trajectory": False,
                    "sample_seed": int(args.seed) + 99_991 + update_index,
                }
            )
            eval_results = _collect_rollouts(eval_config, eval_seeds)
            eval_summary = _summarize_rollout(eval_results)
            eval_summary["seed_start"] = int(args.eval_seed_start)
            eval_summary["seed_count"] = int(args.eval_seeds)
            _unlink_if_unneeded(eval_checkpoint, keep=bool(args.keep_eval_checkpoints))
            eval_mean = float(eval_summary.get("mean_floor", 0.0))
            if eval_mean > best_eval_mean_floor:
                best_eval_mean_floor = eval_mean
                best_eval = {
                    "update": int(update_index),
                    "mean_floor": float(eval_mean),
                    "max_floor": int(eval_summary.get("max_floor") or 0),
                    "min_floor": int(eval_summary.get("min_floor") or 0),
                    "win_count": int(eval_summary.get("win_count") or 0),
                    "seed_start": int(args.eval_seed_start),
                    "seed_count": int(args.eval_seeds),
                }
                best_eval_improved = True
                export_policy_transformer_checkpoint(
                    best_output,
                    policy,
                    training_args=vars(args),
                    dataset_metadata={"algorithm": "ppo", "update": int(update_index), "best_eval": dict(best_eval), "eval": eval_summary},
                )
                save_ppo_policy_checkpoint(
                    best_ppo_checkpoint,
                    policy,
                    optimizer_state_dict=optimizer.state_dict(),
                    training_state={"current_update": int(update_index), "seed_cursor": int(seed_cursor), "best_eval": dict(best_eval)},
                    training_args=vars(args),
                    dataset_metadata={"algorithm": "ppo", "update": int(update_index), "best_eval": dict(best_eval), "eval": eval_summary},
                )
        last_metrics = {
            "update": int(update_index),
            "seeds": seeds,
            "rollout": rollout_summary,
            "train": train_metrics,
            "eval": eval_summary,
            "best_eval": best_eval,
            "best_eval_improved": bool(best_eval_improved),
            "collect_seconds": time.time() - collect_started,
        }
        _write_jsonl(metrics_path, last_metrics)
        if int(args.log_interval) > 0 and (update_index == 1 or update_index % int(args.log_interval) == 0):
            print(
                f"update={update_index}/{args.updates} roots={rollout_summary['root_count']} "
                f"mean_floor={rollout_summary['mean_floor']:.2f} "
                f"loss={float(train_metrics.get('loss', 0.0)):.4f} "
                f"approx_kl={float(train_metrics.get('approx_kl', 0.0)):.4f} "
                f"ref_kl={float(train_metrics.get('kl_to_reference', 0.0)):.4f} "
                f"entropy={float(train_metrics.get('entropy', 0.0)):.4f} "
                f"eval_mean={float(eval_summary.get('mean_floor', 0.0)):.2f} "
                f"best_eval={float(best_eval.get('mean_floor', 0.0)):.2f}",
                flush=True,
            )
        if bool(args.save_each_update):
            update_policy_path = work_dir / f"policy_update_{update_index:04d}.pt"
            update_ppo_path = work_dir / f"policy_update_{update_index:04d}.ppo.pt"
            export_policy_transformer_checkpoint(
                update_policy_path,
                policy,
                training_args=vars(args),
                dataset_metadata={"algorithm": "ppo", "update": int(update_index), "metrics": last_metrics},
            )
            save_ppo_policy_checkpoint(
                update_ppo_path,
                policy,
                optimizer_state_dict=optimizer.state_dict(),
                training_state={"current_update": int(update_index), "seed_cursor": int(seed_cursor), "metrics": last_metrics},
                training_args=vars(args),
                dataset_metadata={"algorithm": "ppo", "update": int(update_index), "metrics": last_metrics},
            )
        _save_all(
            args=args,
            policy=policy,
            optimizer=optimizer,
            update_index=update_index,
            seed_cursor=seed_cursor,
            metrics=last_metrics,
            best_eval=best_eval,
            ppo_checkpoint=ppo_checkpoint,
        )
        del roots
        _release_large_update_objects()
    print(json.dumps(last_metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
