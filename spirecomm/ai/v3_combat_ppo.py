from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import F, nn, require_torch, torch
from spirecomm.ai.v3_combat_features import (
    action_keys_are_unique,
    clone_env_blob,
    encode_candidate_with_before_summary,
    encode_state_summary,
    root_combat_actions,
    step_branch_from_blob,
)
from spirecomm.ai.v3_combat_selector import V3CandidateCombatSelector
from spirecomm.ai.v3_combat_transformer import (
    V3CombatTransformerCandidateScorer,
    collate_transformer_records,
    entity_index_from_vocab,
    load_v3_combat_transformer_checkpoint,
    save_v3_combat_transformer_checkpoint,
    token_spec_from_payload,
)


PPO_CHECKPOINT_VERSION = "v3_combat_ppo_v1"
TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
COMPACT_RECORDS_MARKER = "__v3_ppo_compact_candidate_records__"


@dataclass(frozen=True)
class PPOStats:
    log_probs: Any
    entropy: Any
    kl_to_reference: Any | None


class V3CombatPPOPolicy(nn.Module):
    """Policy/value wrapper around the candidate action-set transformer."""

    checkpoint_kind = "v3_combat_ppo"

    def __init__(
        self,
        base_model: V3CombatTransformerCandidateScorer,
        *,
        value_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.state_dim = int(getattr(base_model, "state_dim", 161))
        hidden = int(value_hidden_dim or max(128, int(getattr(base_model, "d_model", 192))))
        self.value_hidden_dim = hidden
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.checkpoint_token_schema = getattr(base_model, "checkpoint_token_schema", None)
        self.checkpoint_feature_schema = getattr(base_model, "checkpoint_feature_schema", None)
        self.checkpoint_entity_vocab = getattr(base_model, "checkpoint_entity_vocab", None)
        self.checkpoint_token_types = getattr(base_model, "checkpoint_token_types", None)

    def policy_logits(self, batch: dict[str, Any]) -> Any:
        return self.base_model(batch)

    def root_values(self, batch: dict[str, Any]) -> Any:
        features = batch["features"]
        candidate_counts = batch["candidate_counts"].to(device=features.device, dtype=torch.long)
        valid_counts = candidate_counts[candidate_counts > 0]
        if int(valid_counts.numel()) <= 0:
            return features.new_zeros((0,))
        starts = torch.cumsum(valid_counts, dim=0) - valid_counts
        before_summary = features.index_select(0, starts)[:, : self.state_dim]
        return self.root_values_from_before_summary(before_summary)

    def root_values_from_before_summary(self, before_summary: Any) -> Any:
        return self.value_head(before_summary).squeeze(-1)

    def policy_and_value(self, batch: dict[str, Any]) -> tuple[Any, Any]:
        return self.policy_logits(batch), self.root_values(batch)


def attach_checkpoint_metadata(model: Any, checkpoint: dict[str, Any]) -> None:
    token_schema = checkpoint.get("token_schema")
    if isinstance(token_schema, dict) and token_schema:
        setattr(model, "checkpoint_token_schema", dict(token_schema))
    feature_schema = checkpoint.get("feature_schema")
    if isinstance(feature_schema, dict) and feature_schema:
        setattr(model, "checkpoint_feature_schema", dict(feature_schema))
    entity_vocab = checkpoint.get("entity_vocab")
    if entity_vocab:
        setattr(model, "checkpoint_entity_vocab", list(entity_vocab))
    token_types = checkpoint.get("token_types")
    if isinstance(token_types, dict) and token_types:
        setattr(model, "checkpoint_token_types", dict(token_types))


def load_base_transformer_for_ppo(path: str | Path, *, device: str) -> tuple[V3CombatTransformerCandidateScorer, dict[str, Any]]:
    model, checkpoint = load_v3_combat_transformer_checkpoint(path, device=device)
    if not isinstance(model, V3CombatTransformerCandidateScorer):
        raise ValueError("PPO currently expects the old candidate action-set transformer, not root_action_set.")
    if bool(getattr(model, "expects_root_batch", False)):
        raise ValueError("PPO currently expects candidate transformer records, not root-action-set records.")
    attach_checkpoint_metadata(model, checkpoint)
    return model, checkpoint


def make_ppo_policy_from_transformer(path: str | Path, *, device: str, value_hidden_dim: int | None = None) -> tuple[V3CombatPPOPolicy, dict[str, Any]]:
    base_model, checkpoint = load_base_transformer_for_ppo(path, device=device)
    policy = V3CombatPPOPolicy(base_model, value_hidden_dim=value_hidden_dim).to(device)
    return policy, checkpoint


def _base_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = dict(checkpoint.get("base_model_config") or {})
    if not config:
        raise ValueError("PPO checkpoint is missing base_model_config")
    config.pop("architecture", None)
    return config


def load_ppo_policy_checkpoint(path: str | Path, *, device: str) -> tuple[V3CombatPPOPolicy, dict[str, Any]]:
    require_torch()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != PPO_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported PPO checkpoint version: {checkpoint.get('checkpoint_version')}")
    base_model = V3CombatTransformerCandidateScorer(**_base_config_from_checkpoint(checkpoint)).to(device)
    attach_checkpoint_metadata(base_model, checkpoint)
    policy = V3CombatPPOPolicy(
        base_model,
        value_hidden_dim=int(checkpoint.get("ppo_config", {}).get("value_hidden_dim") or 0) or None,
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint


def save_ppo_policy_checkpoint(
    path: str | Path,
    policy: V3CombatPPOPolicy,
    *,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
    training_args: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    base = policy.base_model
    entity_vocab_payload = list(getattr(base, "checkpoint_entity_vocab", None) or [])
    entity_vocab_size = int(getattr(base, "entity_vocab_size", len(entity_vocab_payload)) or len(entity_vocab_payload))
    if len(entity_vocab_payload) != entity_vocab_size:
        raise ValueError(
            "refusing to save PPO checkpoint with mismatched entity vocab: "
            f"len(entity_vocab)={len(entity_vocab_payload)} != entity_vocab_size={entity_vocab_size}"
        )
    token_types_payload = dict(getattr(base, "checkpoint_token_types", None) or {})
    token_type_vocab_size = int(getattr(base, "token_type_vocab_size", len(token_types_payload)) or len(token_types_payload))
    if token_types_payload and any(int(index) >= token_type_vocab_size for index in token_types_payload.values()):
        raise ValueError(
            "refusing to save PPO checkpoint with token type id outside vocab: "
            f"token_type_vocab_size={token_type_vocab_size}, token_types={token_types_payload}"
        )
    torch.save(
        {
            "checkpoint_version": PPO_CHECKPOINT_VERSION,
            "model_state_dict": policy.state_dict(),
            "base_model_config": dict(base._config()),
            "token_schema": dict(getattr(base, "checkpoint_token_schema", None) or getattr(base, "token_schema", None) or {}),
            "feature_schema": dict(getattr(base, "checkpoint_feature_schema", None) or {}),
            "entity_vocab": entity_vocab_payload,
            "token_types": token_types_payload,
            "ppo_config": {
                "value_hidden_dim": int(policy.value_hidden_dim),
                "state_dim": int(policy.state_dim),
            },
            "training_args": dict(training_args or {}),
            "dataset_metadata": dict(dataset_metadata or {}),
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "training_state": dict(training_state or {}),
        },
        target,
    )


def export_policy_transformer_checkpoint(
    path: str | Path,
    policy: V3CombatPPOPolicy,
    *,
    training_args: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> None:
    save_v3_combat_transformer_checkpoint(
        path,
        policy.base_model,
        training_args=training_args,
        dataset_metadata=dataset_metadata,
    )


def set_ppo_trainable(policy: V3CombatPPOPolicy, mode: str) -> None:
    normalized = str(mode or "heads").strip().lower().replace("_", "-")
    if normalized not in {"heads", "action-set", "full"}:
        raise ValueError(f"unsupported PPO trainable mode: {mode!r}")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.value_head.parameters():
        parameter.requires_grad_(True)
    if normalized == "full":
        for parameter in policy.base_model.parameters():
            parameter.requires_grad_(True)
        return
    head_names = ("output_head", "semantic_head", "legacy_residual_head", "legacy_gate_head")
    for name in head_names:
        module = getattr(policy.base_model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    if normalized == "action-set":
        for name in ("action_set_input", "action_set_encoder", "action_set_context_projection"):
            module = getattr(policy.base_model, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)


def count_trainable_parameters(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


def _state_from_env(env: Any) -> dict[str, Any]:
    return V3CandidateCombatSelector._state_from_env(env)


def _combat_env_from_root(env: Any) -> Any | None:
    return V3CandidateCombatSelector._combat_env_from_root(env)


def encode_combat_root_for_candidate_transformer(
    env: Any,
    *,
    entity_index: dict[str, int] | None,
    token_spec: Any | None,
    legal_actions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    actions = root_combat_actions(env, legal_actions=legal_actions)
    if not actions:
        raise ValueError("no_root_combat_actions")
    before_state = _state_from_env(env)
    if not action_keys_are_unique(actions, before_state):
        raise ValueError("ambiguous_action_key")

    before_summary = encode_state_summary(before_state)
    combat_env = _combat_env_from_root(env)
    combat_env_blob = clone_env_blob(combat_env, strip_debug_history=True) if combat_env is not None else None
    env_blob: bytes | None = None
    records: list[dict[str, Any]] = []
    for action in actions:
        after_state = None
        if combat_env_blob is not None:
            combat_branch = step_branch_from_blob(combat_env_blob, action, strip_debug_history=True)
            outcome = str(getattr(combat_branch, "outcome", "") or "")
            if not outcome or outcome == "UNDECIDED":
                after_state = _state_from_env(combat_branch)
        if after_state is None:
            if env_blob is None:
                env_blob = clone_env_blob(env, strip_debug_history=True)
            branch = step_branch_from_blob(env_blob, action, strip_debug_history=True)
            after_state = _state_from_env(branch)
        features = encode_candidate_with_before_summary(before_state, before_summary, action, after_state)
        from spirecomm.ai.v3_combat_transformer import encode_transformer_candidate

        records.append(
            encode_transformer_candidate(
                before_state,
                action,
                after_state,
                candidate_features=features,
                entity_index=entity_index,
                spec=token_spec,
            )
        )
    return before_state, actions, records, collate_transformer_records(records, device="cpu", candidate_counts=[len(records)])


def compact_candidate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    return {
        COMPACT_RECORDS_MARKER: True,
        "token_scalar_features": np.asarray([record["token_scalar_features"] for record in records], dtype=np.float32),
        "token_type_ids": np.asarray([record["token_type_ids"] for record in records], dtype=np.int16),
        "entity_ids": np.asarray([record["entity_ids"] for record in records], dtype=np.int16),
        "slot_ids": np.asarray([record["slot_ids"] for record in records], dtype=np.int16),
        "attention_mask": np.asarray([record["attention_mask"] for record in records], dtype=np.bool_),
        "candidate_features": np.asarray([record["candidate_features"] for record in records], dtype=np.float32),
    }


def records_are_compact(records: Any) -> bool:
    return isinstance(records, dict) and bool(records.get(COMPACT_RECORDS_MARKER))


def first_candidate_features_from_records(records: Any) -> list[float] | Any:
    if records_are_compact(records):
        return records["candidate_features"][0]
    return list(records[0].get("candidate_features") or [])


def apply_runtime_potion_penalty(logits: Any, before_state: dict[str, Any], actions: list[dict[str, Any]], penalty: float) -> Any:
    if float(penalty) <= 0.0 or str(before_state.get("room_type") or "") != "MonsterRoom":
        return logits
    adjusted = logits.clone()
    for index, action in enumerate(actions):
        if str(action.get("kind") or "") == "potion":
            adjusted[index] = adjusted[index] - float(penalty)
    return adjusted


def runtime_potion_logit_adjustments(before_state: dict[str, Any], actions: list[dict[str, Any]], penalty: float) -> list[float]:
    if float(penalty) <= 0.0 or str(before_state.get("room_type") or "") != "MonsterRoom":
        return [0.0 for _action in actions]
    return [-float(penalty) if str(action.get("kind") or "") == "potion" else 0.0 for action in actions]


def _sample_index_from_probs(probs: list[float], rng: random.Random) -> int:
    if not probs:
        raise ValueError("cannot sample from empty probability list")
    draw = rng.random()
    total = 0.0
    for index, probability in enumerate(probs):
        total += max(0.0, float(probability))
        if draw <= total:
            return index
    return len(probs) - 1


class V3PPOCombatSelector:
    handles_potions = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        temperature: float = 1.0,
        normal_room_potion_penalty: float = 1.5,
        sample: bool = True,
        compact_records: bool = False,
        seed: int = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.temperature = max(1.0e-6, float(temperature))
        self.normal_room_potion_penalty = max(0.0, float(normal_room_potion_penalty))
        self.sample = bool(sample)
        self.compact_records = bool(compact_records)
        self.rng = random.Random(seed)
        self.last_error: str | None = None
        self.last_decision: dict[str, Any] | None = None
        self.policy, checkpoint = load_ppo_policy_checkpoint(self.checkpoint_path, device=self.device)
        self.policy.eval()
        self.entity_index = entity_index_from_vocab(checkpoint.get("entity_vocab"))
        self.token_spec = token_spec_from_payload(checkpoint.get("token_schema"))

    @property
    def available(self) -> bool:
        return self.policy is not None

    def choose_env(
        self,
        env: Any,
        *,
        return_scores: bool = True,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, list[float]]:
        self.last_error = None
        self.last_decision = None
        try:
            require_torch()
            before_state, actions, records, batch = encode_combat_root_for_candidate_transformer(
                env,
                entity_index=self.entity_index,
                token_spec=self.token_spec,
                legal_actions=legal_actions,
            )
            batch = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
            with torch.inference_mode():
                logits, values = self.policy.policy_and_value(batch)
                adjustments = runtime_potion_logit_adjustments(before_state, actions, self.normal_room_potion_penalty)
                if any(float(value) != 0.0 for value in adjustments):
                    logits = logits + torch.tensor(adjustments, dtype=logits.dtype, device=logits.device)
                policy_logits = logits.float() / self.temperature
                log_probs = F.log_softmax(policy_logits, dim=0)
                probs = torch.exp(log_probs)
                if self.sample:
                    choice_index = _sample_index_from_probs([float(value) for value in probs.detach().cpu().tolist()], self.rng)
                else:
                    choice_index = int(torch.argmax(policy_logits).item())
                entropy = -(probs * log_probs).sum()
            self.last_decision = {
                "candidate_records": compact_candidate_records(records) if self.compact_records else records,
                "candidate_count": len(records),
                "logit_adjustments": adjustments,
                "chosen_index": int(choice_index),
                "old_logprob": float(log_probs[choice_index].detach().cpu().item()),
                "old_value": float(values[0].detach().cpu().item()) if int(values.numel()) else 0.0,
                "entropy": float(entropy.detach().cpu().item()),
                "before_room_type": str(before_state.get("room_type") or ""),
                "before_floor": int(before_state.get("floor") or 0),
                "action_kind": str(actions[choice_index].get("kind") or ""),
                "action_name": str(actions[choice_index].get("name") or actions[choice_index].get("card_id") or ""),
            }
            scores = [float(value) for value in logits.detach().cpu().tolist()] if return_scores else []
            return actions[choice_index], scores
        except Exception as exc:
            self.last_error = f"v3_ppo_combat_scoring_failed:{exc}"
            return None, []

    def value_env(self, env: Any) -> float | None:
        self.last_error = None
        try:
            require_torch()
            _before_state, _actions, _records, batch = encode_combat_root_for_candidate_transformer(
                env,
                entity_index=self.entity_index,
                token_spec=self.token_spec,
            )
            batch = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
            with torch.inference_mode():
                values = self.policy.root_values(batch)
            if int(values.numel()) <= 0:
                return None
            return float(values[0].detach().cpu().item())
        except Exception as exc:
            self.last_error = f"v3_ppo_combat_value_failed:{exc}"
            return None


def grouped_categorical_stats(
    logits: Any,
    candidate_counts: Any,
    chosen_indices: Any,
    *,
    temperature: float = 1.0,
    reference_logits: Any | None = None,
) -> PPOStats:
    log_probs: list[Any] = []
    entropies: list[Any] = []
    kls: list[Any] = []
    start = 0
    temp = max(1.0e-6, float(temperature))
    for root_index, count_value in enumerate(candidate_counts.detach().cpu().tolist()):
        count = int(count_value)
        end = start + count
        root_logits = logits[start:end].float() / temp
        root_log_probs = F.log_softmax(root_logits, dim=0)
        root_probs = torch.exp(root_log_probs)
        chosen = int(chosen_indices[root_index].item())
        log_probs.append(root_log_probs[chosen])
        entropies.append(-(root_probs * root_log_probs).sum())
        if reference_logits is not None:
            reference_log_probs = F.log_softmax(reference_logits[start:end].float() / temp, dim=0)
            kls.append((root_probs * (root_log_probs - reference_log_probs)).sum())
        start = end
    stacked_log_probs = torch.stack(log_probs) if log_probs else logits.new_zeros((0,))
    stacked_entropies = torch.stack(entropies) if entropies else logits.new_zeros((0,))
    stacked_kls = torch.stack(kls) if kls else None
    return PPOStats(log_probs=stacked_log_probs, entropy=stacked_entropies, kl_to_reference=stacked_kls)


def collate_ppo_roots(roots: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    if not roots:
        raise ValueError("cannot collate empty PPO root batch")
    counts = [int(root["candidate_count"]) for root in roots]
    root_records = [root["candidate_records"] for root in roots]
    if all(records_are_compact(records) for records in root_records):
        import numpy as np

        batch = {
            "token_scalar_features": torch.tensor(
                np.concatenate([records["token_scalar_features"] for records in root_records], axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "token_type_ids": torch.tensor(
                np.concatenate([records["token_type_ids"] for records in root_records], axis=0),
                dtype=torch.long,
                device=device,
            ),
            "entity_ids": torch.tensor(
                np.concatenate([records["entity_ids"] for records in root_records], axis=0),
                dtype=torch.long,
                device=device,
            ),
            "slot_ids": torch.tensor(
                np.concatenate([records["slot_ids"] for records in root_records], axis=0),
                dtype=torch.long,
                device=device,
            ),
            "attention_mask": torch.tensor(
                np.concatenate([records["attention_mask"] for records in root_records], axis=0),
                dtype=torch.bool,
                device=device,
            ),
            "features": torch.tensor(
                np.concatenate([records["candidate_features"] for records in root_records], axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "candidate_counts": torch.tensor(counts, dtype=torch.long, device=device),
        }
    else:
        records = [
            record
            for root_records_value in root_records
            for record in (root_records_value if not records_are_compact(root_records_value) else [])
        ]
        if len(records) != sum(counts):
            raise ValueError("mixed compact and non-compact PPO records are not supported")
        batch = collate_transformer_records(records, device=device, candidate_counts=counts)
    batch["chosen_indices"] = torch.tensor([int(root["chosen_index"]) for root in roots], dtype=torch.long, device=device)
    batch["old_logprobs"] = torch.tensor([float(root["old_logprob"]) for root in roots], dtype=torch.float32, device=device)
    batch["old_values"] = torch.tensor([float(root["old_value"]) for root in roots], dtype=torch.float32, device=device)
    batch["returns"] = torch.tensor([float(root["return"]) for root in roots], dtype=torch.float32, device=device)
    batch["advantages"] = torch.tensor([float(root["advantage"]) for root in roots], dtype=torch.float32, device=device)
    adjustments = [
        float(value)
        for root in roots
        for value in list(root.get("logit_adjustments") or [0.0] * int(root["candidate_count"]))
    ]
    batch["logit_adjustments"] = torch.tensor(adjustments, dtype=torch.float32, device=device)
    return batch


def compute_gae_for_trajectories(
    trajectories: list[list[dict[str, Any]]],
    *,
    gamma: float,
    gae_lambda: float,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for trajectory in trajectories:
        advantage = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            root = trajectory[index]
            value = float(root.get("old_value") or 0.0)
            if index + 1 < len(trajectory):
                next_value = float(trajectory[index + 1].get("old_value") or 0.0)
                next_non_terminal = 1.0
            else:
                next_value = float(root.get("bootstrap_value") or 0.0)
                next_non_terminal = 1.0 if root.get("bootstrap_value") is not None else 0.0
            if bool(root.get("done")):
                next_value = 0.0
                next_non_terminal = 0.0
            delta = float(root.get("reward") or 0.0) + float(gamma) * next_value * next_non_terminal - value
            advantage = delta + float(gamma) * float(gae_lambda) * next_non_terminal * advantage
            root["advantage"] = float(advantage)
            root["return"] = float(advantage + value)
        flattened.extend(trajectory)
    if normalize and flattened:
        mean = sum(float(root["advantage"]) for root in flattened) / len(flattened)
        variance = sum((float(root["advantage"]) - mean) ** 2 for root in flattened) / max(1, len(flattened))
        std = math.sqrt(max(1.0e-8, variance))
        for root in flattened:
            root["advantage"] = (float(root["advantage"]) - mean) / std
    return flattened


def ppo_loss(
    policy: V3CombatPPOPolicy,
    batch: dict[str, Any],
    *,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
    temperature: float,
    reference_model: V3CombatTransformerCandidateScorer | None = None,
) -> tuple[Any, dict[str, float]]:
    logits, values = policy.policy_and_value(batch)
    if "logit_adjustments" in batch:
        logits = logits + batch["logit_adjustments"].to(device=logits.device, dtype=logits.dtype)
    reference_logits = None
    if reference_model is not None:
        with torch.inference_mode():
            reference_logits = reference_model(batch)
            if "logit_adjustments" in batch:
                reference_logits = reference_logits + batch["logit_adjustments"].to(
                    device=reference_logits.device,
                    dtype=reference_logits.dtype,
                )
    stats = grouped_categorical_stats(
        logits,
        batch["candidate_counts"],
        batch["chosen_indices"],
        temperature=temperature,
        reference_logits=reference_logits,
    )
    old_logprobs = batch["old_logprobs"].to(dtype=stats.log_probs.dtype)
    advantages = batch["advantages"].to(dtype=stats.log_probs.dtype)
    returns = batch["returns"].to(dtype=values.dtype)
    ratio = torch.exp(stats.log_probs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = F.smooth_l1_loss(values.float(), returns.float())
    entropy = stats.entropy.mean() if int(stats.entropy.numel()) else logits.new_tensor(0.0)
    kl = (
        stats.kl_to_reference.mean()
        if stats.kl_to_reference is not None and int(stats.kl_to_reference.numel())
        else logits.new_tensor(0.0)
    )
    loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy + float(kl_coef) * kl
    approx_kl = (old_logprobs - stats.log_probs).mean()
    clip_fraction = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
    with torch.no_grad():
        return_var = torch.var(returns.float(), unbiased=False)
        value_error_var = torch.var((returns.float() - values.float()), unbiased=False)
        explained_variance = 1.0 - value_error_var / torch.clamp(return_var, min=1.0e-8)
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "kl_to_reference": float(kl.detach().cpu().item()),
        "approx_kl": float(approx_kl.detach().cpu().item()),
        "clip_fraction": float(clip_fraction.detach().cpu().item()),
        "mean_return": float(returns.detach().float().mean().cpu().item()) if int(returns.numel()) else 0.0,
        "mean_advantage": float(advantages.detach().float().mean().cpu().item()) if int(advantages.numel()) else 0.0,
        "explained_variance": float(explained_variance.detach().cpu().item()),
    }
    return loss, metrics
