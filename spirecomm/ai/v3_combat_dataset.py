from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import F, require_torch, torch
from spirecomm.ai.v3_combat_features import (
    FEATURE_SCHEMA_VERSION,
    action_key,
    action_keys_are_unique,
    encode_candidate,
    root_combat_actions,
    schema,
    step_branch,
)


DATASET_SCHEMA_VERSION = "v3_combat_teacher_dataset_v1"

REWARD_COMPONENT_NAMES = (
    "hp_damage",
    "monster_kill",
    "combat_win",
    "death",
    "hp_loss",
    "effective_block",
    "raw_incoming_damage_reduction",
    "energy_spent",
    "playable_hand_count_delta",
    "player_power_delta",
    "monster_power_delta",
    "play_card_constant",
    "power_card_constant",
    "skill_power_turn_constant",
    "potion_adjustment",
    "potion_room_adjustment",
    "potion_cost",
    "immediate_total",
    "continuation_raw",
    "continuation_adjusted",
    "potion_continuation_room_bonus",
    "non_potion_baseline",
    "potion_marginal",
    "teacher_q",
)
REWARD_COMPONENT_DIM = len(REWARD_COMPONENT_NAMES)
REWARD_COMPONENT_TARGET_CLIP = 5000.0


def reward_component_vector(components: dict[str, Any] | None) -> list[float]:
    mapping = components or {}
    values: list[float] = []
    for name in REWARD_COMPONENT_NAMES:
        try:
            values.append(float(mapping.get(name, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


@dataclass
class V3CombatCandidateExample:
    action: dict[str, Any]
    action_key: tuple[Any, ...]
    visible_after: dict[str, Any]
    delta_features: list[float]
    candidate_features: list[float]
    teacher_q: float = 0.0
    teacher_rank: int = -1
    continuation_depth: int = 0
    continuation_nodes: int = 0
    terminal_kind: str = "UNKNOWN"
    debug_best_line: list[dict[str, Any]] = field(default_factory=list)
    reward_components: dict[str, float] = field(default_factory=dict)
    is_chosen: bool = False


@dataclass
class V3CombatRootSample:
    root_id: str
    source: str
    env_blob: bytes
    visible_before: dict[str, Any]
    actions: list[dict[str, Any]]
    action_keys: list[tuple[Any, ...]]
    chosen_action_key: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def legal_action_count(self) -> int:
        return len(self.actions)

    def load_env(self) -> Any:
        return pickle.loads(self.env_blob)


@dataclass
class V3CombatLabeledRoot:
    root: V3CombatRootSample
    candidates: list[V3CombatCandidateExample]
    teacher_config: dict[str, Any] = field(default_factory=dict)


def _state_from_env(env: Any) -> dict[str, Any]:
    state_method = getattr(env, "state", None)
    if callable(state_method):
        return state_method()
    return env.serialize()


def make_root_sample(
    env: Any,
    *,
    root_id: str,
    source: str,
    chosen_action: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> V3CombatRootSample | None:
    visible_before = _state_from_env(env)
    actions = root_combat_actions(env)
    if len(actions) <= 1:
        return None
    if not action_keys_are_unique(actions, visible_before):
        return None
    keys = [action_key(action, visible_before) for action in actions]
    chosen_key = action_key(chosen_action, visible_before) if chosen_action is not None else None
    return V3CombatRootSample(
        root_id=root_id,
        source=source,
        env_blob=pickle.dumps(env),
        visible_before=visible_before,
        actions=actions,
        action_keys=keys,
        chosen_action_key=chosen_key,
        metadata=dict(metadata or {}),
    )


def unlabeled_candidates(root: V3CombatRootSample) -> list[V3CombatCandidateExample]:
    env = root.load_env()
    visible_before = _state_from_env(env)
    examples: list[V3CombatCandidateExample] = []
    for action in root.actions:
        branch = step_branch(env, action)
        visible_after = _state_from_env(branch)
        features = encode_candidate(visible_before, action, visible_after)
        key = action_key(action, visible_before)
        examples.append(
            V3CombatCandidateExample(
                action=dict(action),
                action_key=key,
                visible_after=visible_after,
                delta_features=features[-schema().delta_dim :],
                candidate_features=features,
                is_chosen=root.chosen_action_key == key,
            )
        )
    return examples


def save_shard(path: Path, roots: list[V3CombatLabeledRoot], *, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dims": asdict(schema()),
        "metadata": dict(metadata or {}),
        "roots": roots,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_shard(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("dataset_schema") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported v3 combat dataset schema in {path}: {payload.get('dataset_schema')}")
    if payload.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "v3 combat dataset feature schema mismatch in "
            f"{path}: {payload.get('feature_schema')} != {FEATURE_SCHEMA_VERSION}"
        )
    return payload


def load_labeled_roots(paths: list[Path]) -> list[V3CombatLabeledRoot]:
    roots: list[V3CombatLabeledRoot] = []
    for path in paths:
        payload = load_shard(path)
        roots.extend(payload.get("roots") or [])
    return roots


def flatten_labeled_roots(roots: list[V3CombatLabeledRoot]) -> list[tuple[int, V3CombatCandidateExample]]:
    flattened: list[tuple[int, V3CombatCandidateExample]] = []
    for sample_id, root in enumerate(roots):
        for candidate in root.candidates:
            flattened.append((sample_id, candidate))
    return flattened


def collate_labeled_roots(roots: list[V3CombatLabeledRoot], *, device: str = "cpu") -> dict[str, Any]:
    require_torch()
    candidates = flatten_labeled_roots(roots)
    if not candidates:
        raise ValueError("cannot collate empty v3 combat candidate batch")
    features = [candidate.candidate_features for _, candidate in candidates]
    teacher_q = [float(candidate.teacher_q) for _, candidate in candidates]
    reward_components = [reward_component_vector(getattr(candidate, "reward_components", None)) for _, candidate in candidates]
    sample_ids = [int(sample_id) for sample_id, _ in candidates]
    chosen = [bool(candidate.is_chosen) for _, candidate in candidates]
    return {
        "features": torch.tensor(features, dtype=torch.float32, device=device),
        "teacher_q": torch.tensor(teacher_q, dtype=torch.float32, device=device),
        "reward_components": torch.tensor(reward_components, dtype=torch.float32, device=device),
        "sample_ids": torch.tensor(sample_ids, dtype=torch.long, device=device),
        "chosen": torch.tensor(chosen, dtype=torch.bool, device=device),
        "root_count": len(roots),
    }


def _padded_candidate_values(values: Any, counts: Any, *, fill_value: float = 0.0) -> tuple[Any, Any]:
    require_torch()
    root_count = int(counts.numel())
    max_candidates = int(counts.max().item()) if root_count else 0
    if root_count <= 0 or max_candidates <= 0:
        raise ValueError("cannot build padded candidate values for empty batch")
    mask = torch.arange(max_candidates, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)
    result = torch.full((root_count, max_candidates), float(fill_value), dtype=values.dtype, device=values.device)
    result[mask] = values
    return result, mask


def _weighted_mean(values: Any, weights: Any | None) -> Any:
    require_torch()
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)


def _sanitize_teacher_q(teacher_q: Any, *, clip: float = 0.0) -> tuple[Any, dict[str, float]]:
    require_torch()
    clip_value = float(clip)
    if clip_value <= 0.0:
        return teacher_q, {"teacher_q_nonfinite": 0.0, "teacher_q_clipped": 0.0}
    teacher = teacher_q.to(dtype=torch.float32)
    finite = torch.isfinite(teacher)
    clipped = finite & (teacher.abs() > clip_value)
    sanitized = torch.nan_to_num(teacher, nan=0.0, posinf=clip_value, neginf=-clip_value).clamp(
        min=-clip_value,
        max=clip_value,
    )
    return sanitized.to(device=teacher_q.device, dtype=teacher_q.dtype), {
        "teacher_q_nonfinite": float((~finite).to(dtype=torch.float32).sum().detach().cpu().item()),
        "teacher_q_clipped": float(clipped.to(dtype=torch.float32).sum().detach().cpu().item()),
    }


def ranking_loss_padded(pred_q: Any, teacher_q: Any, counts: Any, *, temperature: float = 1.0, root_weights: Any | None = None) -> Any:
    require_torch()
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=-1.0e9)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    pred_log_probs = F.log_softmax(pred, dim=1)
    target_probs = F.softmax(teacher / max(1.0e-6, float(temperature)), dim=1)
    per_root = (target_probs * (target_probs.clamp_min(1.0e-12).log() - pred_log_probs)).masked_fill(~mask, 0.0).sum(dim=1)
    per_root = per_root / counts.to(dtype=per_root.dtype).clamp_min(1.0)
    return _weighted_mean(per_root, root_weights)


def pairwise_margin_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    *,
    margin: float = 0.1,
    min_teacher_gap: float = 1.0,
    root_weights: Any | None = None,
) -> Any:
    require_torch()
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=0.0)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=0.0)
    valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
    teacher_gap = teacher.unsqueeze(2) - teacher.unsqueeze(1)
    pair_mask = valid_pairs & (teacher_gap >= float(min_teacher_gap))
    if not bool(pair_mask.any().item()):
        return pred_q.sum() * 0.0
    pred_gap = pred.unsqueeze(2) - pred.unsqueeze(1)
    losses = F.relu(torch.tensor(float(margin), device=pred.device) - pred_gap)
    per_root_sum = losses.masked_fill(~pair_mask, 0.0).sum(dim=(1, 2))
    per_root_count = pair_mask.to(dtype=losses.dtype).sum(dim=(1, 2))
    valid_root = per_root_count > 0
    per_root = per_root_sum[valid_root] / per_root_count[valid_root].clamp_min(1.0)
    weights = root_weights[valid_root] if root_weights is not None else None
    return _weighted_mean(per_root, weights)


def behavior_cloning_loss_padded(pred_q: Any, counts: Any, chosen: Any) -> Any:
    require_torch()
    if not bool(chosen.any().item()):
        return pred_q.sum() * 0.0
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=-1.0e9)
    chosen_padded, _ = _padded_candidate_values(chosen.to(dtype=pred_q.dtype), counts, fill_value=0.0)
    chosen_mask = chosen_padded.to(dtype=torch.bool) & mask
    root_has_choice = chosen_mask.any(dim=1)
    if not bool(root_has_choice.any().item()):
        return pred_q.sum() * 0.0
    log_probs = F.log_softmax(pred, dim=1)
    return -(log_probs[chosen_mask]).mean()


def _root_aux_from_padded(batch: dict[str, Any], counts: Any, teacher_q: Any) -> tuple[Any | None, dict[str, float]]:
    require_torch()
    action_is_potion = batch.get("action_is_potion")
    room_type_ids = batch.get("room_type_ids")
    if action_is_potion is None or room_type_ids is None:
        return None, {
            "elite_boss_top_potion_roots": 0.0,
            "potion_candidate_roots": 0.0,
        }
    potion, mask = _padded_candidate_values(action_is_potion.to(dtype=torch.float32), counts, fill_value=0.0)
    room_ids, _ = _padded_candidate_values(room_type_ids.to(dtype=torch.float32), counts, fill_value=0.0)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    has_potion = (potion > 0.5).any(dim=1)
    best_is_potion = (potion.gather(1, teacher.argmax(dim=1, keepdim=True)).squeeze(1) > 0.5) & has_potion
    root_room_ids = room_ids[:, 0].to(dtype=torch.long)
    elite_boss = (root_room_ids == 2) | (root_room_ids == 3)
    highlighted = best_is_potion & elite_boss
    return highlighted, {
        "elite_boss_top_potion_roots": float(highlighted.to(dtype=torch.float32).sum().detach().cpu().item()),
        "potion_candidate_roots": float(has_potion.to(dtype=torch.float32).sum().detach().cpu().item()),
    }


def potion_vs_non_potion_pairwise_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    action_is_potion: Any | None,
    *,
    margin: float = 0.15,
    min_teacher_gap: float = 0.5,
    root_weights: Any | None = None,
) -> Any:
    require_torch()
    if action_is_potion is None:
        return pred_q.sum() * 0.0
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=0.0)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=0.0)
    potion, _ = _padded_candidate_values(action_is_potion.to(dtype=torch.float32), counts, fill_value=0.0)
    potion_mask = mask & (potion > 0.5)
    non_potion_mask = mask & ~potion_mask
    valid_root = potion_mask.any(dim=1) & non_potion_mask.any(dim=1)
    if not bool(valid_root.any().item()):
        return pred_q.sum() * 0.0
    neg_inf = torch.tensor(-1.0e9, dtype=teacher.dtype, device=teacher.device)
    best_potion_index = teacher.masked_fill(~potion_mask, neg_inf).argmax(dim=1)
    best_non_potion_index = teacher.masked_fill(~non_potion_mask, neg_inf).argmax(dim=1)
    row = torch.arange(teacher.shape[0], device=teacher.device)
    teacher_gap = teacher[row, best_potion_index] - teacher[row, best_non_potion_index]
    direction = torch.sign(teacher_gap)
    valid_root = valid_root & (direction != 0) & (teacher_gap.abs() >= float(min_teacher_gap))
    if not bool(valid_root.any().item()):
        return pred_q.sum() * 0.0
    pred_gap = pred[row, best_potion_index] - pred[row, best_non_potion_index]
    losses = F.relu(torch.tensor(float(margin), device=pred.device) - direction * pred_gap)
    losses = losses[valid_root]
    weights = root_weights[valid_root] if root_weights is not None else None
    return _weighted_mean(losses, weights)


def _root_floor_values_from_batch(batch: dict[str, Any], counts: Any) -> Any | None:
    before_summary = batch.get("before_summary")
    if before_summary is not None and int(before_summary.shape[0]) == int(counts.numel()) and int(before_summary.shape[1]) > 1:
        return before_summary[:, 1].to(dtype=torch.float32, device=counts.device) * 60.0
    features = batch.get("features")
    if features is None or int(features.shape[1]) <= 1:
        return None
    floor_values, _ = _padded_candidate_values(features[:, 1].to(dtype=torch.float32), counts, fill_value=0.0)
    return floor_values[:, 0] * 60.0


def _root_room_ids_from_batch(batch: dict[str, Any], counts: Any) -> Any | None:
    room_type_ids = batch.get("room_type_ids")
    if room_type_ids is None:
        return None
    room_ids, _ = _padded_candidate_values(room_type_ids.to(dtype=torch.float32), counts, fill_value=0.0)
    return room_ids[:, 0].to(dtype=torch.long)


def _teacher_top_is_potion(batch: dict[str, Any], counts: Any, teacher_q: Any) -> Any | None:
    action_is_potion = batch.get("action_is_potion")
    if action_is_potion is None:
        return None
    potion, _ = _padded_candidate_values(action_is_potion.to(dtype=torch.float32), counts, fill_value=0.0)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    return potion.gather(1, teacher.argmax(dim=1, keepdim=True)).squeeze(1) > 0.5


def _early_root_mask(
    batch: dict[str, Any],
    counts: Any,
    teacher_q: Any,
    *,
    floor_max: float,
    room_id: int,
    non_potion_teacher_only: bool,
) -> Any | None:
    root_floor = _root_floor_values_from_batch(batch, counts)
    root_room_ids = _root_room_ids_from_batch(batch, counts)
    if root_floor is None or root_room_ids is None:
        return None
    mask = torch.ones_like(root_room_ids, dtype=torch.bool, device=counts.device)
    if float(floor_max) > 0.0:
        mask = mask & (root_floor <= float(floor_max))
    if int(room_id) > 0:
        mask = mask & (root_room_ids == int(room_id))
    if bool(non_potion_teacher_only):
        top_is_potion = _teacher_top_is_potion(batch, counts, teacher_q)
        if top_is_potion is not None:
            mask = mask & ~top_is_potion
    return mask


def critical_top_vs_all_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    root_mask: Any | None,
    *,
    margin: float = 0.2,
    min_teacher_gap: float = 1.0,
) -> Any:
    require_torch()
    if root_mask is None:
        return pred_q.sum() * 0.0
    root_mask = root_mask.to(device=pred_q.device, dtype=torch.bool)
    if not bool(root_mask.any().item()):
        return pred_q.sum() * 0.0
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=0.0)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    row = torch.arange(teacher.shape[0], device=teacher.device)
    top_index = teacher.argmax(dim=1)
    teacher_gap = teacher[row, top_index].unsqueeze(1) - teacher
    candidate_index = torch.arange(teacher.shape[1], device=teacher.device).unsqueeze(0)
    pair_mask = (
        mask
        & (candidate_index != top_index.unsqueeze(1))
        & (teacher_gap >= float(min_teacher_gap))
        & root_mask.unsqueeze(1)
    )
    if not bool(pair_mask.any().item()):
        return pred_q.sum() * 0.0
    pred_gap = pred[row, top_index].unsqueeze(1) - pred
    losses = F.relu(torch.tensor(float(margin), device=pred.device) - pred_gap)
    per_root_count = pair_mask.to(dtype=losses.dtype).sum(dim=1)
    valid_root = per_root_count > 0
    per_root = losses.masked_fill(~pair_mask, 0.0).sum(dim=1)[valid_root] / per_root_count[valid_root].clamp_min(1.0)
    return per_root.mean() if int(per_root.numel()) > 0 else pred_q.sum() * 0.0


def top1_cross_entropy_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    batch: dict[str, Any],
    *,
    min_teacher_gap: float = 0.0,
    teacher_gap_log_scale: float = 0.25,
    large_gap_threshold: float = 10.0,
    large_gap_weight: float = 2.0,
    monster_room_weight: float = 1.0,
    early_floor_max: float = 0.0,
    early_floor_weight: float = 1.0,
    kind_filter: str = "all",
    root_weight_clip: float = 6.0,
) -> tuple[Any, dict[str, float]]:
    require_torch()
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=-1.0e9)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    if int(pred.shape[1]) <= 1:
        return pred_q.sum() * 0.0, {
            "top1_ce_roots": 0.0,
            "top1_ce_mean_weight": 0.0,
        }
    row = torch.arange(teacher.shape[0], device=teacher.device)
    teacher_top_index = teacher.argmax(dim=1)
    second_teacher = teacher.masked_fill(
        torch.arange(teacher.shape[1], device=teacher.device).unsqueeze(0) == teacher_top_index.unsqueeze(1),
        -1.0e9,
    ).max(dim=1).values
    teacher_gap = teacher[row, teacher_top_index] - second_teacher
    root_mask = mask.any(dim=1) & (counts > 1) & (teacher_gap >= float(min_teacher_gap))

    end_flag, card_flag, potion_flag = _padded_action_kind_flags(batch, counts, mask)
    teacher_top_card = card_flag.gather(1, teacher_top_index.unsqueeze(1)).squeeze(1)
    teacher_top_potion = potion_flag.gather(1, teacher_top_index.unsqueeze(1)).squeeze(1)
    teacher_top_end = end_flag.gather(1, teacher_top_index.unsqueeze(1)).squeeze(1)
    kind_filter_value = str(kind_filter or "all").strip().lower().replace("_", "-")
    if kind_filter_value in {"teacher-card", "card-top", "top-card"}:
        root_mask = root_mask & teacher_top_card
    elif kind_filter_value in {"teacher-potion", "potion-top", "top-potion"}:
        root_mask = root_mask & teacher_top_potion
    elif kind_filter_value in {"teacher-end", "end-top", "top-end"}:
        root_mask = root_mask & teacher_top_end
    elif kind_filter_value not in {"", "all"}:
        raise ValueError(f"unsupported top1_ce kind_filter: {kind_filter}")

    if not bool(root_mask.any().item()):
        return pred_q.sum() * 0.0, {
            "top1_ce_roots": 0.0,
            "top1_ce_mean_weight": 0.0,
        }

    losses = F.cross_entropy(pred, teacher_top_index, reduction="none")
    weights = torch.ones_like(losses)
    if float(teacher_gap_log_scale) != 0.0:
        weights = weights + float(teacher_gap_log_scale) * torch.log1p(teacher_gap.clamp_min(0.0))
    if float(large_gap_threshold) > 0.0 and float(large_gap_weight) != 1.0:
        weights = torch.where(teacher_gap >= float(large_gap_threshold), weights * float(large_gap_weight), weights)

    root_room_ids = _root_room_ids_from_batch(batch, counts)
    if root_room_ids is not None and float(monster_room_weight) != 1.0:
        monster_room = root_room_ids.to(device=pred_q.device) == 1
        weights = torch.where(monster_room, weights * float(monster_room_weight), weights)

    root_floor = _root_floor_values_from_batch(batch, counts)
    if root_floor is not None and float(early_floor_max) > 0.0 and float(early_floor_weight) != 1.0:
        early_root = root_floor.to(device=pred_q.device) <= float(early_floor_max)
        weights = torch.where(early_root, weights * float(early_floor_weight), weights)

    if float(root_weight_clip) > 0.0:
        weights = weights.clamp(max=float(root_weight_clip))
    weights = weights.masked_fill(~root_mask, 0.0)
    denominator = weights.sum().clamp_min(1.0e-6)
    return (losses * weights).sum() / denominator, {
        "top1_ce_roots": float(root_mask.to(dtype=torch.float32).sum().detach().cpu().item()),
        "top1_ce_mean_weight": float(weights[root_mask].mean().detach().cpu().item()) if bool(root_mask.any().item()) else 0.0,
    }


def _padded_action_kind_flags(batch: dict[str, Any], counts: Any, mask: Any) -> tuple[Any, Any, Any]:
    require_torch()
    features = batch.get("features")
    if features is None:
        false = torch.zeros_like(mask, dtype=torch.bool)
        action_is_potion = batch.get("action_is_potion")
        if action_is_potion is not None:
            potion, _ = _padded_candidate_values(action_is_potion.to(dtype=torch.float32), counts, fill_value=0.0)
            false = potion > 0.5
        return torch.zeros_like(mask, dtype=torch.bool), torch.zeros_like(mask, dtype=torch.bool), false & mask
    feature_schema = schema()
    state_dim = int(feature_schema.state_dim)
    if int(features.shape[1]) <= state_dim + 2:
        false = torch.zeros_like(mask, dtype=torch.bool)
        return false, false, false
    end_values, _ = _padded_candidate_values(features[:, state_dim + 0].to(dtype=torch.float32), counts, fill_value=0.0)
    card_values, _ = _padded_candidate_values(features[:, state_dim + 1].to(dtype=torch.float32), counts, fill_value=0.0)
    potion_values, _ = _padded_candidate_values(features[:, state_dim + 2].to(dtype=torch.float32), counts, fill_value=0.0)
    action_is_potion = batch.get("action_is_potion")
    if action_is_potion is not None:
        potion_values, _ = _padded_candidate_values(action_is_potion.to(dtype=torch.float32), counts, fill_value=0.0)
    return (end_values > 0.5) & mask, (card_values > 0.5) & mask, (potion_values > 0.5) & mask


def hard_top_contrastive_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    batch: dict[str, Any],
    *,
    min_teacher_gap: float = 5.0,
    topk: int = 2,
    margin_base: float = 0.25,
    margin_log_scale: float = 0.35,
    margin_max: float = 3.0,
    kind_filter: str = "all",
    card_card_weight: float = 2.0,
    monster_room_weight: float = 1.5,
    early_floor_max: float = 16.0,
    early_floor_weight: float = 2.0,
    large_gap_threshold: float = 25.0,
    large_gap_weight: float = 2.0,
    root_weight_clip: float = 6.0,
) -> tuple[Any, dict[str, float]]:
    require_torch()
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=-1.0e9)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    if int(pred.shape[1]) <= 1 or int(topk) <= 0:
        return pred_q.sum() * 0.0, {
            "hard_top_pairs": 0.0,
            "hard_top_roots": 0.0,
            "hard_top_card_card_pairs": 0.0,
        }
    row = torch.arange(teacher.shape[0], device=teacher.device)
    teacher_top_index = teacher.argmax(dim=1)
    candidate_index = torch.arange(teacher.shape[1], device=teacher.device).unsqueeze(0)
    wrong_mask = mask & (candidate_index != teacher_top_index.unsqueeze(1))
    if not bool(wrong_mask.any().item()):
        return pred_q.sum() * 0.0, {
            "hard_top_pairs": 0.0,
            "hard_top_roots": 0.0,
            "hard_top_card_card_pairs": 0.0,
        }
    k = min(int(topk), max(1, int(pred.shape[1]) - 1))
    wrong_pred = pred.masked_fill(~wrong_mask, -1.0e9)
    hard_pred_values, hard_indices = wrong_pred.topk(k, dim=1)
    valid_hard = hard_pred_values > -1.0e8
    teacher_top = teacher[row, teacher_top_index].unsqueeze(1)
    pred_top = pred[row, teacher_top_index].unsqueeze(1)
    hard_teacher = teacher.gather(1, hard_indices)
    hard_pred = pred.gather(1, hard_indices)
    teacher_gap = teacher_top - hard_teacher
    pair_mask = valid_hard & (teacher_gap >= float(min_teacher_gap))
    if not bool(pair_mask.any().item()):
        return pred_q.sum() * 0.0, {
            "hard_top_pairs": 0.0,
            "hard_top_roots": 0.0,
            "hard_top_card_card_pairs": 0.0,
        }

    target_margin = float(margin_base) + float(margin_log_scale) * torch.log1p(teacher_gap.clamp_min(0.0))
    if float(margin_max) > 0.0:
        target_margin = target_margin.clamp(max=float(margin_max))
    losses = F.softplus(target_margin - (pred_top - hard_pred))

    end_flag, card_flag, potion_flag = _padded_action_kind_flags(batch, counts, mask)
    teacher_top_card = card_flag.gather(1, teacher_top_index.unsqueeze(1))
    teacher_top_potion = potion_flag.gather(1, teacher_top_index.unsqueeze(1))
    teacher_top_end = end_flag.gather(1, teacher_top_index.unsqueeze(1))
    hard_card = card_flag.gather(1, hard_indices)
    card_card_pair_raw = teacher_top_card & hard_card
    kind_filter_value = str(kind_filter or "all").strip().lower().replace("_", "-")
    if kind_filter_value in {"card-card", "cardcard"}:
        pair_mask = pair_mask & card_card_pair_raw
    elif kind_filter_value in {"teacher-potion", "potion-top", "top-potion"}:
        pair_mask = pair_mask & teacher_top_potion
    elif kind_filter_value in {"teacher-card", "card-top", "top-card"}:
        pair_mask = pair_mask & teacher_top_card
    elif kind_filter_value in {"teacher-end", "end-top", "top-end"}:
        pair_mask = pair_mask & teacher_top_end
    elif kind_filter_value not in {"", "all"}:
        raise ValueError(f"unsupported hard_top kind_filter: {kind_filter}")
    card_card_pair = card_card_pair_raw & pair_mask

    pair_weights = torch.ones_like(losses)
    if float(card_card_weight) != 1.0:
        pair_weights = torch.where(card_card_pair, pair_weights * float(card_card_weight), pair_weights)

    root_room_ids = _root_room_ids_from_batch(batch, counts)
    if root_room_ids is not None and float(monster_room_weight) != 1.0:
        monster_room = root_room_ids.to(device=pred_q.device).unsqueeze(1) == 1
        pair_weights = torch.where(monster_room & pair_mask, pair_weights * float(monster_room_weight), pair_weights)

    root_floor = _root_floor_values_from_batch(batch, counts)
    if root_floor is not None and float(early_floor_max) > 0.0 and float(early_floor_weight) != 1.0:
        early_root = root_floor.to(device=pred_q.device).unsqueeze(1) <= float(early_floor_max)
        pair_weights = torch.where(early_root & pair_mask, pair_weights * float(early_floor_weight), pair_weights)

    if float(large_gap_threshold) > 0.0 and float(large_gap_weight) != 1.0:
        large_gap = teacher_gap >= float(large_gap_threshold)
        pair_weights = torch.where(large_gap & pair_mask, pair_weights * float(large_gap_weight), pair_weights)

    if float(root_weight_clip) > 0.0:
        pair_weights = pair_weights.clamp(max=float(root_weight_clip))
    pair_weights = pair_weights.masked_fill(~pair_mask, 0.0)
    denominator = pair_weights.sum().clamp_min(1.0e-6)
    loss = (losses * pair_weights).sum() / denominator
    hard_roots = pair_mask.any(dim=1)
    return loss, {
        "hard_top_pairs": float(pair_mask.to(dtype=torch.float32).sum().detach().cpu().item()),
        "hard_top_roots": float(hard_roots.to(dtype=torch.float32).sum().detach().cpu().item()),
        "hard_top_card_card_pairs": float(card_card_pair.to(dtype=torch.float32).sum().detach().cpu().item()),
    }


def good_bad_contrastive_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    batch: dict[str, Any],
    *,
    good_teacher_gap: float = 2.0,
    bad_min_regret: float = 25.0,
    min_top_gap: float = 5.0,
    bad_topk: int = 2,
    margin_base: float = 0.25,
    margin_log_scale: float = 0.25,
    margin_max: float = 3.0,
    kind_filter: str = "card-card",
    room_filter: str = "combat",
    monster_room_weight: float = 1.5,
    early_floor_max: float = 16.0,
    early_floor_weight: float = 2.0,
    large_gap_threshold: float = 50.0,
    large_gap_weight: float = 1.5,
    root_weight_clip: float = 5.0,
) -> tuple[Any, dict[str, float]]:
    require_torch()
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=-1.0e9)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    empty_metrics = {
        "good_bad_pairs": 0.0,
        "good_bad_roots": 0.0,
        "good_bad_card_card_pairs": 0.0,
        "good_bad_mean_good_count": 0.0,
    }
    if int(pred.shape[1]) <= 1 or int(bad_topk) <= 0:
        return pred_q.sum() * 0.0, empty_metrics

    row = torch.arange(teacher.shape[0], device=teacher.device)
    teacher_top_index = teacher.argmax(dim=1)
    candidate_index = torch.arange(teacher.shape[1], device=teacher.device).unsqueeze(0)
    teacher_top = teacher[row, teacher_top_index].unsqueeze(1)
    teacher_second = teacher.masked_fill(candidate_index == teacher_top_index.unsqueeze(1), -1.0e9).max(dim=1).values
    top_gap = teacher_top.squeeze(1) - teacher_second
    root_mask = top_gap >= float(min_top_gap)

    end_flag, card_flag, potion_flag = _padded_action_kind_flags(batch, counts, mask)
    teacher_top_card = card_flag.gather(1, teacher_top_index.unsqueeze(1)).squeeze(1)
    teacher_top_potion = potion_flag.gather(1, teacher_top_index.unsqueeze(1)).squeeze(1)
    teacher_top_end = end_flag.gather(1, teacher_top_index.unsqueeze(1)).squeeze(1)

    teacher_gap = teacher_top - teacher
    good_mask = mask & (teacher_gap <= float(good_teacher_gap))
    bad_mask = mask & (teacher_gap >= float(bad_min_regret))
    kind_filter_value = str(kind_filter or "all").strip().lower().replace("_", "-")
    if kind_filter_value in {"card-card", "cardcard"}:
        root_mask = root_mask & teacher_top_card
        good_mask = good_mask & card_flag
        bad_mask = bad_mask & card_flag
    elif kind_filter_value in {"teacher-card", "card-top", "top-card"}:
        root_mask = root_mask & teacher_top_card
    elif kind_filter_value in {"teacher-potion", "potion-top", "top-potion"}:
        root_mask = root_mask & teacher_top_potion
    elif kind_filter_value in {"teacher-end", "end-top", "top-end"}:
        root_mask = root_mask & teacher_top_end
    elif kind_filter_value not in {"", "all"}:
        raise ValueError(f"unsupported good_bad kind_filter: {kind_filter}")

    root_room_ids = _root_room_ids_from_batch(batch, counts)
    room_filter_value = str(room_filter or "all").strip().lower().replace("_", "-")
    if root_room_ids is not None and room_filter_value not in {"", "all"}:
        room_ids = root_room_ids.to(device=pred_q.device)
        if room_filter_value in {"combat", "combat-only"}:
            root_mask = root_mask & ((room_ids == 1) | (room_ids == 2) | (room_ids == 3))
        elif room_filter_value in {"monster", "monster-room"}:
            root_mask = root_mask & (room_ids == 1)
        elif room_filter_value in {"elite-boss", "boss-elite"}:
            root_mask = root_mask & ((room_ids == 2) | (room_ids == 3))
        else:
            raise ValueError(f"unsupported good_bad room_filter: {room_filter}")

    good_pred = pred.masked_fill(~good_mask, -1.0e9)
    good_best = good_pred.max(dim=1).values.unsqueeze(1)
    good_available = good_best.squeeze(1) > -1.0e8
    bad_pred = pred.masked_fill(~bad_mask, -1.0e9)
    k = min(int(bad_topk), max(1, int(pred.shape[1]) - 1))
    bad_pred_values, bad_indices = bad_pred.topk(k, dim=1)
    valid_bad = bad_pred_values > -1.0e8
    bad_teacher_gap = teacher_gap.gather(1, bad_indices)
    pair_mask = valid_bad & root_mask.unsqueeze(1) & good_available.unsqueeze(1)
    if not bool(pair_mask.any().item()):
        return pred_q.sum() * 0.0, empty_metrics

    target_margin = float(margin_base) + float(margin_log_scale) * torch.log1p(bad_teacher_gap.clamp_min(0.0))
    if float(margin_max) > 0.0:
        target_margin = target_margin.clamp(max=float(margin_max))
    losses = F.softplus(target_margin - (good_best - bad_pred_values))

    bad_card = card_flag.gather(1, bad_indices)
    card_card_pair = teacher_top_card.unsqueeze(1) & bad_card & pair_mask
    pair_weights = torch.ones_like(losses)
    if root_room_ids is not None and float(monster_room_weight) != 1.0:
        monster_room = root_room_ids.to(device=pred_q.device).unsqueeze(1) == 1
        pair_weights = torch.where(monster_room & pair_mask, pair_weights * float(monster_room_weight), pair_weights)
    root_floor = _root_floor_values_from_batch(batch, counts)
    if root_floor is not None and float(early_floor_max) > 0.0 and float(early_floor_weight) != 1.0:
        early_root = root_floor.to(device=pred_q.device).unsqueeze(1) <= float(early_floor_max)
        pair_weights = torch.where(early_root & pair_mask, pair_weights * float(early_floor_weight), pair_weights)
    if float(large_gap_threshold) > 0.0 and float(large_gap_weight) != 1.0:
        large_gap = bad_teacher_gap >= float(large_gap_threshold)
        pair_weights = torch.where(large_gap & pair_mask, pair_weights * float(large_gap_weight), pair_weights)
    if float(root_weight_clip) > 0.0:
        pair_weights = pair_weights.clamp(max=float(root_weight_clip))
    pair_weights = pair_weights.masked_fill(~pair_mask, 0.0)
    denominator = pair_weights.sum().clamp_min(1.0e-6)
    loss = (losses * pair_weights).sum() / denominator
    good_counts = good_mask.to(dtype=torch.float32).sum(dim=1)
    active_roots = pair_mask.any(dim=1)
    return loss, {
        "good_bad_pairs": float(pair_mask.to(dtype=torch.float32).sum().detach().cpu().item()),
        "good_bad_roots": float(active_roots.to(dtype=torch.float32).sum().detach().cpu().item()),
        "good_bad_card_card_pairs": float(card_card_pair.to(dtype=torch.float32).sum().detach().cpu().item()),
        "good_bad_mean_good_count": float(good_counts[active_roots].mean().detach().cpu().item()) if bool(active_roots.any().item()) else 0.0,
    }


def distill_ranking_loss_padded(
    pred_q: Any,
    distill_q: Any | None,
    counts: Any,
    root_mask: Any | None,
    *,
    temperature: float = 1.0,
) -> Any:
    require_torch()
    if distill_q is None or root_mask is None:
        return pred_q.sum() * 0.0
    weights = root_mask.to(device=pred_q.device, dtype=pred_q.dtype)
    if not bool((weights > 0).any().item()):
        return pred_q.sum() * 0.0
    return ranking_loss_padded(pred_q, distill_q.to(device=pred_q.device, dtype=pred_q.dtype), counts, temperature=temperature, root_weights=weights)


def anchor_guard_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    distill_q: Any | None,
    counts: Any,
    *,
    margin: float = 0.1,
    min_teacher_gap: float = 0.5,
) -> tuple[Any, dict[str, float]]:
    require_torch()
    if distill_q is None:
        return pred_q.sum() * 0.0, {
            "anchor_guard_roots": 0.0,
            "anchor_guard_pairs": 0.0,
            "anchor_guard_pred_broken_roots": 0.0,
        }
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=-1.0e9)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    anchor, _ = _padded_candidate_values(distill_q.to(device=teacher_q.device, dtype=teacher_q.dtype), counts, fill_value=-1.0e9)
    if int(pred.shape[1]) <= 1:
        return pred_q.sum() * 0.0, {
            "anchor_guard_roots": 0.0,
            "anchor_guard_pairs": 0.0,
            "anchor_guard_pred_broken_roots": 0.0,
        }
    row = torch.arange(teacher.shape[0], device=teacher.device)
    teacher_top = teacher.argmax(dim=1)
    anchor_top = anchor.argmax(dim=1)
    pred_top = pred.argmax(dim=1)
    second_teacher = teacher.masked_fill(
        torch.arange(teacher.shape[1], device=teacher.device).unsqueeze(0) == teacher_top.unsqueeze(1),
        -1.0e9,
    ).max(dim=1).values
    teacher_gap = teacher[row, teacher_top] - second_teacher
    root_mask = (anchor_top == teacher_top) & (teacher_gap >= float(min_teacher_gap)) & (counts > 1)
    if not bool(root_mask.any().item()):
        return pred_q.sum() * 0.0, {
            "anchor_guard_roots": 0.0,
            "anchor_guard_pairs": 0.0,
            "anchor_guard_pred_broken_roots": 0.0,
        }
    candidate_index = torch.arange(teacher.shape[1], device=teacher.device).unsqueeze(0)
    pair_mask = mask & (candidate_index != teacher_top.unsqueeze(1)) & root_mask.unsqueeze(1)
    pred_gap = pred[row, teacher_top].unsqueeze(1) - pred
    losses = F.relu(torch.tensor(float(margin), device=pred.device) - pred_gap).masked_fill(~pair_mask, 0.0)
    # A single high competing action can flip the root, so use the worst
    # violating competitor per protected root instead of averaging it away.
    per_root = losses.max(dim=1).values[root_mask]
    loss = per_root.mean() if int(per_root.numel()) > 0 else pred_q.sum() * 0.0
    broken = root_mask & (pred_top != teacher_top)
    return loss, {
        "anchor_guard_roots": float(root_mask.to(dtype=torch.float32).sum().detach().cpu().item()),
        "anchor_guard_pairs": float(pair_mask.to(dtype=torch.float32).sum().detach().cpu().item()),
        "anchor_guard_pred_broken_roots": float(broken.to(dtype=torch.float32).sum().detach().cpu().item()),
    }


def _distill_root_mask(
    batch: dict[str, Any],
    counts: Any,
    teacher_q: Any,
    *,
    mode: str,
    floor_max: float,
    room_id: int,
    non_potion_teacher_only: bool,
) -> Any | None:
    require_torch()
    mode = str(mode or "early")
    if mode == "early":
        return _early_root_mask(
            batch,
            counts,
            teacher_q,
            floor_max=floor_max,
            room_id=room_id,
            non_potion_teacher_only=non_potion_teacher_only,
        )
    if mode == "all":
        return torch.ones((int(counts.numel()),), dtype=torch.bool, device=counts.device)
    if mode not in {"baseline_correct", "baseline_correct_non_potion"}:
        raise ValueError(f"unsupported distill root mode: {mode}")
    distill_q = batch.get("distill_q")
    if distill_q is None:
        return None
    teacher, mask = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    distill, _ = _padded_candidate_values(distill_q.to(device=teacher_q.device, dtype=teacher_q.dtype), counts, fill_value=-1.0e9)
    root_mask = teacher.argmax(dim=1) == distill.argmax(dim=1)
    if mode == "baseline_correct_non_potion":
        top_is_potion = _teacher_top_is_potion(batch, counts, teacher_q)
        if top_is_potion is not None:
            root_mask = root_mask & ~top_is_potion
    return root_mask & mask.any(dim=1)


def segment_log_softmax(values: Any, sample_ids: Any) -> Any:
    require_torch()
    result = torch.empty_like(values)
    for sample_id in torch.unique(sample_ids).tolist():
        mask = sample_ids == int(sample_id)
        result[mask] = F.log_softmax(values[mask], dim=0)
    return result


def segment_softmax(values: Any, sample_ids: Any) -> Any:
    return segment_log_softmax(values, sample_ids).exp()


def ranking_loss(pred_q: Any, teacher_q: Any, sample_ids: Any, *, temperature: float = 1.0) -> Any:
    require_torch()
    losses = []
    temp = max(1e-6, float(temperature))
    for sample_id in torch.unique(sample_ids).tolist():
        mask = sample_ids == int(sample_id)
        pred_log_probs = F.log_softmax(pred_q[mask], dim=0)
        target_probs = F.softmax(teacher_q[mask] / temp, dim=0)
        losses.append(F.kl_div(pred_log_probs, target_probs, reduction="batchmean"))
    return torch.stack(losses).mean() if losses else pred_q.sum() * 0.0


def pairwise_margin_loss(pred_q: Any, teacher_q: Any, sample_ids: Any, *, margin: float = 0.1, min_teacher_gap: float = 1.0) -> Any:
    require_torch()
    losses = []
    for sample_id in torch.unique(sample_ids).tolist():
        mask = sample_ids == int(sample_id)
        pred = pred_q[mask]
        teacher = teacher_q[mask]
        count = int(pred.shape[0])
        for i in range(count):
            for j in range(count):
                if float(teacher[i].item() - teacher[j].item()) < min_teacher_gap:
                    continue
                losses.append(F.relu(torch.tensor(float(margin), device=pred.device) - (pred[i] - pred[j])))
    return torch.stack(losses).mean() if losses else pred_q.sum() * 0.0


def q_regression_loss(pred_q: Any, teacher_q: Any, candidate_weights: Any | None = None) -> Any:
    require_torch()
    target = teacher_q
    if target.numel() > 1:
        target = (target - target.mean()) / (target.std(unbiased=False) + 1e-6)
        pred = (pred_q - pred_q.mean()) / (pred_q.std(unbiased=False) + 1e-6)
    else:
        pred = pred_q
    losses = F.smooth_l1_loss(pred, target, reduction="none")
    return _weighted_mean(losses, candidate_weights)


def _signed_gap_transform(values: Any, transform: str) -> Any:
    transform = str(transform or "none").strip().lower()
    if transform in {"", "none", "raw", "linear"}:
        return values
    if transform == "sqrt":
        # `sign(x) * sqrt(abs(x))` has an infinite derivative at the anchor
        # gap x=0. Every root has at least one exact anchor gap, so use the
        # algebraic equivalent with epsilon-stabilized denominator.
        return values / (values.abs() + 1.0e-2).sqrt()
    if transform == "log":
        return values.sign() * torch.log1p(values.abs())
    raise ValueError(f"unsupported gap q transform: {transform}")


def _gap_q_losses(pred_gap: Any, teacher_gap: Any, *, loss: str) -> Any:
    mode = str(loss or "l1").strip().lower()
    if mode in {"l1", "abs", "absolute"}:
        return (pred_gap - teacher_gap).abs()
    if mode in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred_gap, teacher_gap, reduction="none")
    raise ValueError(f"unsupported gap q loss: {loss}")


def root_gap_q_loss_padded(
    pred_q: Any,
    teacher_q: Any,
    counts: Any,
    *,
    transform: str = "sqrt",
    loss: str = "l1",
    hard_negative_gap_threshold: float = 10.0,
    hard_negative_weight: float = 5.0,
    root_weights: Any | None = None,
) -> Any:
    require_torch()
    pred, mask = _padded_candidate_values(pred_q, counts, fill_value=0.0)
    teacher, _ = _padded_candidate_values(teacher_q, counts, fill_value=-1.0e9)
    teacher_top_index = teacher.argmax(dim=1, keepdim=True)
    teacher_anchor = teacher.gather(1, teacher_top_index)
    pred_anchor = pred.gather(1, teacher_top_index)
    raw_teacher_gap = teacher - teacher_anchor
    teacher_gap = raw_teacher_gap
    pred_gap = pred - pred_anchor
    teacher_gap = _signed_gap_transform(teacher_gap, transform)
    pred_gap = _signed_gap_transform(pred_gap, transform)
    losses = _gap_q_losses(pred_gap, teacher_gap, loss=loss).masked_fill(~mask, 0.0)
    candidate_weights = torch.ones_like(losses)
    if float(hard_negative_gap_threshold) > 0.0 and float(hard_negative_weight) > 0.0:
        hard_negative = raw_teacher_gap <= -float(hard_negative_gap_threshold)
        candidate_weights = torch.where(
            hard_negative & mask,
            candidate_weights + float(hard_negative_weight),
            candidate_weights,
        )
    candidate_weights = candidate_weights.masked_fill(~mask, 0.0)
    per_root = (losses * candidate_weights).sum(dim=1) / candidate_weights.sum(dim=1).clamp_min(1.0)
    return _weighted_mean(per_root, root_weights)


def root_gap_q_loss(
    pred_q: Any,
    teacher_q: Any,
    sample_ids: Any,
    *,
    transform: str = "sqrt",
    loss: str = "l1",
    hard_negative_gap_threshold: float = 10.0,
    hard_negative_weight: float = 5.0,
) -> Any:
    require_torch()
    losses = []
    for sample_id in torch.unique(sample_ids).tolist():
        mask = sample_ids == int(sample_id)
        pred = pred_q[mask]
        teacher = teacher_q[mask]
        if int(pred.numel()) <= 0:
            continue
        top_index = int(torch.argmax(teacher).item())
        raw_teacher_gap = teacher - teacher[top_index]
        teacher_gap = raw_teacher_gap
        pred_gap = pred - pred[top_index]
        teacher_gap = _signed_gap_transform(teacher_gap, transform)
        pred_gap = _signed_gap_transform(pred_gap, transform)
        per_candidate = _gap_q_losses(pred_gap, teacher_gap, loss=loss)
        candidate_weights = torch.ones_like(per_candidate)
        if float(hard_negative_gap_threshold) > 0.0 and float(hard_negative_weight) > 0.0:
            candidate_weights = torch.where(
                raw_teacher_gap <= -float(hard_negative_gap_threshold),
                candidate_weights + float(hard_negative_weight),
                candidate_weights,
            )
        losses.append((per_candidate * candidate_weights).sum() / candidate_weights.sum().clamp_min(1.0))
    return torch.stack(losses).mean() if losses else pred_q.sum() * 0.0


def auxiliary_reward_component_loss(aux_pred: Any | None, batch: dict[str, Any]) -> Any | None:
    require_torch()
    if aux_pred is None:
        return None
    reward_components = batch.get("reward_components")
    if reward_components is not None and int(aux_pred.shape[-1]) >= REWARD_COMPONENT_DIM:
        # Old relabeled shards can contain huge finite sentinels from clipped
        # teacher values. Normalize auxiliary targets in fp32 and clip them
        # before computing moments so a single sentinel cannot create NaNs.
        target = reward_components.to(device=aux_pred.device, dtype=torch.float32)
        pred = aux_pred[:, :REWARD_COMPONENT_DIM].float()
        finite = torch.isfinite(target)
        if not bool(finite.any().detach().cpu().item()):
            return pred.sum() * 0.0
        clean_target = torch.where(finite, target.clamp(-REWARD_COMPONENT_TARGET_CLIP, REWARD_COMPONENT_TARGET_CLIP), torch.zeros_like(target))
        counts = finite.to(dtype=target.dtype).sum(dim=0).clamp_min(1.0)
        mean = clean_target.sum(dim=0) / counts
        centered = torch.where(finite, clean_target - mean, torch.zeros_like(clean_target))
        std = (centered.square().sum(dim=0) / counts).sqrt().clamp_min(1.0)
        losses = F.smooth_l1_loss((pred - mean) / std, (clean_target - mean) / std, reduction="none")
        masked = torch.where(finite, losses, torch.zeros_like(losses))
        return masked.sum() / finite.to(dtype=losses.dtype).sum().clamp_min(1.0)
    features = batch.get("features")
    if features is None:
        return None
    feature_schema = schema()
    delta_dim = int(feature_schema.delta_dim)
    if int(features.shape[1]) < delta_dim or int(aux_pred.shape[-1]) < 8:
        return None
    delta = features[:, -delta_dim:].to(device=aux_pred.device, dtype=aux_pred.dtype)
    # Existing normalized delta components: total damage, kill, effective block,
    # incoming change, playable count, victory, death, and HP delta.
    indices = [12, 11, 14, 15, 16, 23, 24, 0]
    target = delta[:, indices]
    return F.smooth_l1_loss(aux_pred[:, : len(indices)], target)


def behavior_cloning_loss(pred_q: Any, sample_ids: Any, chosen: Any) -> Any:
    require_torch()
    if not bool(chosen.any().item()):
        return pred_q.sum() * 0.0
    losses = []
    for sample_id in torch.unique(sample_ids).tolist():
        mask = sample_ids == int(sample_id)
        chosen_mask = chosen[mask]
        if not bool(chosen_mask.any().item()):
            continue
        log_probs = F.log_softmax(pred_q[mask], dim=0)
        chosen_indices = torch.nonzero(chosen_mask, as_tuple=False).flatten()
        losses.append(-log_probs[chosen_indices[0]])
    return torch.stack(losses).mean() if losses else pred_q.sum() * 0.0


def total_candidate_loss(
    pred_q: Any,
    batch: dict[str, Any],
    *,
    rank_weight: float = 1.0,
    q_weight: float = 0.5,
    pair_weight: float = 0.2,
    bc_weight: float = 0.05,
    temperature: float = 1.0,
    potion_pair_weight: float = 0.0,
    potion_pair_margin: float = 0.15,
    potion_pair_min_teacher_gap: float = 0.5,
    elite_boss_top_potion_root_weight: float = 1.0,
    critical_loss_weight: float = 0.0,
    critical_loss_margin: float = 0.2,
    critical_loss_min_teacher_gap: float = 1.0,
    critical_loss_floor_max: float = 5.0,
    critical_loss_room_id: int = 1,
    critical_loss_non_potion_teacher_only: bool = True,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
    distill_floor_max: float = 10.0,
    distill_room_id: int = 1,
    distill_non_potion_teacher_only: bool = True,
    distill_root_mode: str = "early",
    anchor_guard_weight: float = 0.0,
    anchor_guard_margin: float = 0.1,
    anchor_guard_min_teacher_gap: float = 0.5,
    aux_reward_weight: float = 0.0,
    gap_q_weight: float = 0.0,
    gap_q_transform: str = "sqrt",
    gap_q_loss: str = "l1",
    gap_q_hard_negative_threshold: float = 10.0,
    gap_q_hard_negative_weight: float = 5.0,
    hard_top_weight: float = 0.0,
    hard_top_min_teacher_gap: float = 5.0,
    hard_top_topk: int = 2,
    hard_top_margin_base: float = 0.25,
    hard_top_margin_log_scale: float = 0.35,
    hard_top_margin_max: float = 3.0,
    hard_top_kind_filter: str = "all",
    hard_top_card_card_weight: float = 2.0,
    hard_top_monster_room_weight: float = 1.5,
    hard_top_early_floor_max: float = 16.0,
    hard_top_early_floor_weight: float = 2.0,
    hard_top_large_gap_threshold: float = 25.0,
    hard_top_large_gap_weight: float = 2.0,
    hard_top_root_weight_clip: float = 6.0,
    good_bad_weight: float = 0.0,
    good_bad_good_teacher_gap: float = 2.0,
    good_bad_bad_min_regret: float = 25.0,
    good_bad_min_top_gap: float = 5.0,
    good_bad_bad_topk: int = 2,
    good_bad_margin_base: float = 0.25,
    good_bad_margin_log_scale: float = 0.25,
    good_bad_margin_max: float = 3.0,
    good_bad_kind_filter: str = "card-card",
    good_bad_room_filter: str = "combat",
    good_bad_monster_room_weight: float = 1.5,
    good_bad_early_floor_max: float = 16.0,
    good_bad_early_floor_weight: float = 2.0,
    good_bad_large_gap_threshold: float = 50.0,
    good_bad_large_gap_weight: float = 1.5,
    good_bad_root_weight_clip: float = 5.0,
    top1_ce_weight: float = 0.0,
    top1_ce_min_teacher_gap: float = 0.0,
    top1_ce_teacher_gap_log_scale: float = 0.25,
    top1_ce_large_gap_threshold: float = 10.0,
    top1_ce_large_gap_weight: float = 2.0,
    top1_ce_monster_room_weight: float = 1.0,
    top1_ce_early_floor_max: float = 0.0,
    top1_ce_early_floor_weight: float = 1.0,
    top1_ce_kind_filter: str = "all",
    top1_ce_root_weight_clip: float = 6.0,
    teacher_q_clip: float = 0.0,
) -> tuple[Any, dict[str, float]]:
    teacher_q, teacher_q_metrics = _sanitize_teacher_q(batch["teacher_q"], clip=teacher_q_clip)
    sample_ids = batch["sample_ids"]
    counts = batch.get("candidate_counts")
    potion_pair_root_weights = None
    aux_metrics: dict[str, float] = {}
    critical_loss = pred_q.sum() * 0.0
    distill_loss = pred_q.sum() * 0.0
    aux_reward_loss = pred_q.sum() * 0.0
    gap_q = pred_q.sum() * 0.0
    anchor_guard = pred_q.sum() * 0.0
    hard_top_loss = pred_q.sum() * 0.0
    good_bad_loss = pred_q.sum() * 0.0
    top1_ce_loss = pred_q.sum() * 0.0
    hard_top_metrics = {
        "hard_top_pairs": 0.0,
        "hard_top_roots": 0.0,
        "hard_top_card_card_pairs": 0.0,
    }
    top1_ce_metrics = {
        "top1_ce_roots": 0.0,
        "top1_ce_mean_weight": 0.0,
    }
    good_bad_metrics = {
        "good_bad_pairs": 0.0,
        "good_bad_roots": 0.0,
        "good_bad_card_card_pairs": 0.0,
        "good_bad_mean_good_count": 0.0,
    }
    anchor_guard_metrics = {
        "anchor_guard_roots": 0.0,
        "anchor_guard_pairs": 0.0,
        "anchor_guard_pred_broken_roots": 0.0,
    }
    critical_root_count = 0.0
    distill_root_count = 0.0
    if counts is not None:
        highlighted, aux_metrics = _root_aux_from_padded(batch, counts, teacher_q)
        if highlighted is not None and float(elite_boss_top_potion_root_weight) != 1.0:
            # Keep the generic ranking/Q objective representative of all roots.
            # Only the potion-specific auxiliary loss should emphasize sparse
            # elite/boss roots where the teacher prefers potion use.
            potion_pair_root_weights = torch.ones_like(counts, dtype=pred_q.dtype, device=pred_q.device)
            potion_pair_root_weights = torch.where(
                highlighted.to(device=pred_q.device),
                torch.full_like(potion_pair_root_weights, float(elite_boss_top_potion_root_weight)),
                potion_pair_root_weights,
            )
        rank = ranking_loss_padded(pred_q, teacher_q, counts, temperature=temperature)
        pair = pairwise_margin_loss_padded(pred_q, teacher_q, counts)
        bc = behavior_cloning_loss_padded(pred_q, counts, batch["chosen"])
        if float(gap_q_weight) > 0.0:
            gap_q = root_gap_q_loss_padded(
                pred_q,
                teacher_q,
                counts,
                transform=gap_q_transform,
                loss=gap_q_loss,
                hard_negative_gap_threshold=gap_q_hard_negative_threshold,
                hard_negative_weight=gap_q_hard_negative_weight,
            )
        if float(hard_top_weight) > 0.0:
            hard_top_loss, hard_top_metrics = hard_top_contrastive_loss_padded(
                pred_q,
                teacher_q,
                counts,
                batch,
                min_teacher_gap=hard_top_min_teacher_gap,
                topk=int(hard_top_topk),
                margin_base=hard_top_margin_base,
                margin_log_scale=hard_top_margin_log_scale,
                margin_max=hard_top_margin_max,
                kind_filter=hard_top_kind_filter,
                card_card_weight=hard_top_card_card_weight,
                monster_room_weight=hard_top_monster_room_weight,
                early_floor_max=hard_top_early_floor_max,
                early_floor_weight=hard_top_early_floor_weight,
                large_gap_threshold=hard_top_large_gap_threshold,
                large_gap_weight=hard_top_large_gap_weight,
                root_weight_clip=hard_top_root_weight_clip,
            )
        if float(top1_ce_weight) > 0.0:
            top1_ce_loss, top1_ce_metrics = top1_cross_entropy_loss_padded(
                pred_q,
                teacher_q,
                counts,
                batch,
                min_teacher_gap=top1_ce_min_teacher_gap,
                teacher_gap_log_scale=top1_ce_teacher_gap_log_scale,
                large_gap_threshold=top1_ce_large_gap_threshold,
                large_gap_weight=top1_ce_large_gap_weight,
                monster_room_weight=top1_ce_monster_room_weight,
                early_floor_max=top1_ce_early_floor_max,
                early_floor_weight=top1_ce_early_floor_weight,
                kind_filter=top1_ce_kind_filter,
                root_weight_clip=top1_ce_root_weight_clip,
            )
        if float(good_bad_weight) > 0.0:
            good_bad_loss, good_bad_metrics = good_bad_contrastive_loss_padded(
                pred_q,
                teacher_q,
                counts,
                batch,
                good_teacher_gap=good_bad_good_teacher_gap,
                bad_min_regret=good_bad_bad_min_regret,
                min_top_gap=good_bad_min_top_gap,
                bad_topk=int(good_bad_bad_topk),
                margin_base=good_bad_margin_base,
                margin_log_scale=good_bad_margin_log_scale,
                margin_max=good_bad_margin_max,
                kind_filter=good_bad_kind_filter,
                room_filter=good_bad_room_filter,
                monster_room_weight=good_bad_monster_room_weight,
                early_floor_max=good_bad_early_floor_max,
                early_floor_weight=good_bad_early_floor_weight,
                large_gap_threshold=good_bad_large_gap_threshold,
                large_gap_weight=good_bad_large_gap_weight,
                root_weight_clip=good_bad_root_weight_clip,
            )
        potion_pair = potion_vs_non_potion_pairwise_loss_padded(
            pred_q,
            teacher_q,
            counts,
            batch.get("action_is_potion"),
            margin=potion_pair_margin,
            min_teacher_gap=potion_pair_min_teacher_gap,
            root_weights=potion_pair_root_weights,
        )
        if float(critical_loss_weight) > 0.0:
            critical_root_mask = _early_root_mask(
                batch,
                counts,
                teacher_q,
                floor_max=critical_loss_floor_max,
                room_id=int(critical_loss_room_id),
                non_potion_teacher_only=bool(critical_loss_non_potion_teacher_only),
            )
            if critical_root_mask is not None:
                critical_root_count = float(critical_root_mask.to(dtype=torch.float32).sum().detach().cpu().item())
            critical_loss = critical_top_vs_all_loss_padded(
                pred_q,
                teacher_q,
                counts,
                critical_root_mask,
                margin=critical_loss_margin,
                min_teacher_gap=critical_loss_min_teacher_gap,
            )
        if float(distill_weight) > 0.0:
            distill_root_mask = _distill_root_mask(
                batch,
                counts,
                teacher_q,
                mode=distill_root_mode,
                floor_max=distill_floor_max,
                room_id=int(distill_room_id),
                non_potion_teacher_only=bool(distill_non_potion_teacher_only),
            )
            if distill_root_mask is not None:
                distill_root_count = float(distill_root_mask.to(dtype=torch.float32).sum().detach().cpu().item())
            distill_loss = distill_ranking_loss_padded(
                pred_q,
                batch.get("distill_q"),
                counts,
                distill_root_mask,
                temperature=distill_temperature,
            )
        if float(anchor_guard_weight) > 0.0:
            anchor_guard, anchor_guard_metrics = anchor_guard_loss_padded(
                pred_q,
                teacher_q,
                batch.get("distill_q"),
                counts,
                margin=anchor_guard_margin,
                min_teacher_gap=anchor_guard_min_teacher_gap,
            )
    else:
        rank = ranking_loss(pred_q, teacher_q, sample_ids, temperature=temperature)
        pair = pairwise_margin_loss(pred_q, teacher_q, sample_ids)
        bc = behavior_cloning_loss(pred_q, sample_ids, batch["chosen"])
        if float(gap_q_weight) > 0.0:
            gap_q = root_gap_q_loss(
                pred_q,
                teacher_q,
                sample_ids,
                transform=gap_q_transform,
                loss=gap_q_loss,
                hard_negative_gap_threshold=gap_q_hard_negative_threshold,
                hard_negative_weight=gap_q_hard_negative_weight,
            )
        potion_pair = pred_q.sum() * 0.0
    q_loss = q_regression_loss(pred_q, teacher_q)
    if float(aux_reward_weight) > 0.0:
        aux_loss = auxiliary_reward_component_loss(batch.get("aux_reward_pred"), batch)
        if aux_loss is not None:
            aux_reward_loss = aux_loss
    total = (
        rank_weight * rank
        + q_weight * q_loss
        + pair_weight * pair
        + bc_weight * bc
        + float(potion_pair_weight) * potion_pair
        + float(critical_loss_weight) * critical_loss
        + float(distill_weight) * distill_loss
        + float(anchor_guard_weight) * anchor_guard
        + float(aux_reward_weight) * aux_reward_loss
        + float(gap_q_weight) * gap_q
        + float(hard_top_weight) * hard_top_loss
        + float(good_bad_weight) * good_bad_loss
        + float(top1_ce_weight) * top1_ce_loss
    )
    metrics = {
        "loss": float(total.detach().cpu().item()),
        "rank_loss": float(rank.detach().cpu().item()),
        "q_loss": float(q_loss.detach().cpu().item()),
        "pair_loss": float(pair.detach().cpu().item()),
        "bc_loss": float(bc.detach().cpu().item()),
        "potion_pair_loss": float(potion_pair.detach().cpu().item()),
        "critical_loss": float(critical_loss.detach().cpu().item()),
        "critical_roots": critical_root_count,
        "distill_loss": float(distill_loss.detach().cpu().item()),
        "distill_roots": distill_root_count,
        "anchor_guard_loss": float(anchor_guard.detach().cpu().item()),
        "aux_reward_loss": float(aux_reward_loss.detach().cpu().item()),
        "gap_q_loss": float(gap_q.detach().cpu().item()),
        "hard_top_loss": float(hard_top_loss.detach().cpu().item()),
        "good_bad_loss": float(good_bad_loss.detach().cpu().item()),
        "top1_ce_loss": float(top1_ce_loss.detach().cpu().item()),
    }
    metrics.update(hard_top_metrics)
    metrics.update(good_bad_metrics)
    metrics.update(top1_ce_metrics)
    metrics.update(anchor_guard_metrics)
    metrics.update(aux_metrics)
    metrics.update(teacher_q_metrics)
    return total, metrics
