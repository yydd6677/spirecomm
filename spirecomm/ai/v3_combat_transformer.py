from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is an optional eval-speed dependency.
    np = None

from spirecomm.ai.checkpoint_compat import torch_load_portable_path
from spirecomm.ai.torch_compat import nn, require_torch, torch
from spirecomm.ai.v3_combat_dataset import REWARD_COMPONENT_DIM, reward_component_vector
from spirecomm.ai.v3_combat_features import (
    ATTACK_INTENTS,
    CARD_RARITIES,
    CARD_TYPES,
    COMBAT_ROOM_TYPES,
    FEATURE_SCHEMA_VERSION,
    INTENTS,
    PLAYER_POWER_IDS,
    ZONE_NAMES,
    _clip_norm,
    _incoming_damage_from_monsters,
    _is_alive,
    _powers_by_id,
    _selected_card,
    _selected_potion,
    _selected_target,
    _zone_cards,
    _zone_summary,
    action_key,
    card_identity_ids,
    combat_state,
    encode_action_summary_base,
    encode_candidate,
    encode_delta,
    encode_state_summary,
    hand_cards,
    monsters,
    potion_identity_ids,
    schema,
)


CHECKPOINT_VERSION = "v3_combat_transformer_candidate_scorer_v1"
TOKEN_SCHEMA_VERSION_V1 = "v3_combat_transformer_tokens_v1"
TOKEN_SCHEMA_VERSION_V2 = "v3_combat_transformer_tokens_v2_relic32"
TOKEN_SCHEMA_VERSION_V3 = "v3_combat_transformer_tokens_v3_relic24_power12"
TOKEN_SCHEMA_VERSION_V4_STRUCTURED = "v3_combat_transformer_tokens_v4_structured_transition"
TOKEN_SCHEMA_VERSION_V5_STRUCTURED_SELECTED = "v3_combat_transformer_tokens_v5_structured_selected"
TOKEN_SCHEMA_VERSION_V6_STRUCTURED_PHASE2 = "v3_combat_transformer_tokens_v6_structured_phase2"
TOKEN_SCHEMA_VERSION_V7_ACTION_BINDING = "v3_combat_transformer_tokens_v7_action_binding"
TOKEN_SCHEMA_VERSION = TOKEN_SCHEMA_VERSION_V3
ROOT_TOKEN_SCHEMA_VERSION_V1 = "v3_combat_root_action_tokens_v1"
ROOT_TOKEN_SCHEMA_VERSION_V2 = "v3_combat_root_action_tokens_v2_selected_entities"
ROOT_TOKEN_SCHEMA_VERSION = "v3_combat_root_action_tokens_v3_no_legacy"
SUPPORTED_CANDIDATE_TOKEN_SCHEMA_VERSIONS = {
    TOKEN_SCHEMA_VERSION_V1,
    TOKEN_SCHEMA_VERSION_V2,
    TOKEN_SCHEMA_VERSION,
    TOKEN_SCHEMA_VERSION_V4_STRUCTURED,
    TOKEN_SCHEMA_VERSION_V5_STRUCTURED_SELECTED,
    TOKEN_SCHEMA_VERSION_V6_STRUCTURED_PHASE2,
    TOKEN_SCHEMA_VERSION_V7_ACTION_BINDING,
}
SUPPORTED_TOKEN_SCHEMA_VERSIONS = {
    *SUPPORTED_CANDIDATE_TOKEN_SCHEMA_VERSIONS,
    ROOT_TOKEN_SCHEMA_VERSION_V1,
    ROOT_TOKEN_SCHEMA_VERSION_V2,
    ROOT_TOKEN_SCHEMA_VERSION,
}
TRANSFORMER_TENSOR_DATASET_SCHEMA = "v3_combat_transformer_tensor_dataset_v1"
ROOT_TRANSFORMER_TENSOR_DATASET_SCHEMA = "v3_combat_root_transformer_tensor_dataset_v1"

TRANSFORMER_SCALAR_DIM = 48
DEFAULT_MAX_HAND = 10
DEFAULT_MAX_POTIONS = 5
DEFAULT_MAX_MONSTERS = 6
DEFAULT_MAX_PLAYER_POWERS = 12
DEFAULT_MAX_RELICS = 24
DEFAULT_MAX_POWER_DELTAS = 8
DEFAULT_MAX_ACTIONS = 64
ROOT_ACTION_SEGMENT_WIDTH_V1 = 4
ROOT_ACTION_SEGMENT_WIDTH_V2 = 7
ROOT_ACTION_SEGMENT_WIDTH = 6
ROOT_HEAD_VARIANT_BASE = "base"
ROOT_HEAD_VARIANT_SELECTED = "selected-head"
ROOT_HEAD_VARIANT_ACTION200 = "action200-head"
ROOT_HEAD_VARIANT_SELECTED_SHARED_POOL = "selected-shared-pool"
ROOT_HEAD_VARIANTS = {
    ROOT_HEAD_VARIANT_BASE,
    ROOT_HEAD_VARIANT_SELECTED,
    ROOT_HEAD_VARIANT_ACTION200,
    ROOT_HEAD_VARIANT_SELECTED_SHARED_POOL,
}
CANDIDATE_HEAD_VARIANT_BASE = "base"
CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY = "selected-entity-head"
CANDIDATE_HEAD_VARIANT_ACTION200 = "action200-head"
CANDIDATE_HEAD_VARIANT_RELATIVE_RANK = "relative-rank-head"
CANDIDATE_HEAD_VARIANT_DUAL_GATE = "dual-semantic-legacy-gate"
CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200 = "dual-gate-action200"
CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY = "dual-gate-selected-entity"
CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING = "dual-action-binding-gate"
CANDIDATE_HEAD_VARIANT_SEMANTIC_ONLY = "semantic-only"
CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION = "semantic-transition-head"
CANDIDATE_HEAD_VARIANT_LEGACY_DELTA = "legacy-baseline-semantic-delta"
CANDIDATE_HEAD_VARIANTS = {
    CANDIDATE_HEAD_VARIANT_BASE,
    CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY,
    CANDIDATE_HEAD_VARIANT_ACTION200,
    CANDIDATE_HEAD_VARIANT_RELATIVE_RANK,
    CANDIDATE_HEAD_VARIANT_DUAL_GATE,
    CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
    CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY,
    CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
    CANDIDATE_HEAD_VARIANT_SEMANTIC_ONLY,
    CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION,
    CANDIDATE_HEAD_VARIANT_LEGACY_DELTA,
}


TOKEN_TYPES = {
    "PAD": 0,
    "CLS": 1,
    "GLOBAL_BEFORE": 2,
    "PLAYER_CORE": 3,
    "PLAYER_POWER": 4,
    "MONSTER": 5,
    "HAND_CARD": 6,
    "POTION": 7,
    "RELIC": 8,
    "ZONE_SUMMARY": 9,
    "ACTION": 10,
    "AFTER_SUMMARY": 11,
    "DELTA": 12,
    "LEGACY": 13,
    "SELECTED_CARD": 14,
    "SELECTED_TARGET": 15,
    "SELECTED_POTION": 16,
    "PLAYER_DELTA": 17,
    "MONSTER_DELTA": 18,
    "ZONE_DELTA": 19,
    "POTION_SLOT_DELTA": 20,
    "PLAYER_AFTER": 21,
    "MONSTER_AFTER": 22,
    "POWER_DELTA": 23,
    "CARD_TARGET_INTERACTION": 24,
}


@dataclass(frozen=True)
class V3CombatTransformerTokenSpec:
    version: str = TOKEN_SCHEMA_VERSION
    scalar_dim: int = TRANSFORMER_SCALAR_DIM
    max_hand: int = DEFAULT_MAX_HAND
    max_potions: int = DEFAULT_MAX_POTIONS
    max_monsters: int = DEFAULT_MAX_MONSTERS
    max_player_powers: int = DEFAULT_MAX_PLAYER_POWERS
    max_relics: int = DEFAULT_MAX_RELICS
    max_power_deltas: int = DEFAULT_MAX_POWER_DELTAS

    @property
    def uses_structured_transition_tokens(self) -> bool:
        return self.version in {
            TOKEN_SCHEMA_VERSION_V4_STRUCTURED,
            TOKEN_SCHEMA_VERSION_V5_STRUCTURED_SELECTED,
            TOKEN_SCHEMA_VERSION_V6_STRUCTURED_PHASE2,
            TOKEN_SCHEMA_VERSION_V7_ACTION_BINDING,
        }

    @property
    def uses_selected_entity_tokens(self) -> bool:
        return self.version in {
            TOKEN_SCHEMA_VERSION_V5_STRUCTURED_SELECTED,
            TOKEN_SCHEMA_VERSION_V6_STRUCTURED_PHASE2,
            TOKEN_SCHEMA_VERSION_V7_ACTION_BINDING,
        }

    @property
    def uses_phase2_transition_tokens(self) -> bool:
        return self.version == TOKEN_SCHEMA_VERSION_V6_STRUCTURED_PHASE2

    @property
    def uses_action_interaction_token(self) -> bool:
        return self.version == TOKEN_SCHEMA_VERSION_V7_ACTION_BINDING

    @property
    def max_sequence_length(self) -> int:
        length = (
            1  # CLS
            + 1  # GLOBAL_BEFORE
            + 1  # PLAYER_CORE
            + self.max_player_powers
            + self.max_monsters
            + self.max_hand
            + self.max_potions
            + self.max_relics
            + len(ZONE_NAMES)
            + 1  # ACTION
            + 1  # AFTER_SUMMARY
            + 1  # DELTA
            + 1  # LEGACY
        )
        if self.uses_selected_entity_tokens:
            length += 3  # SELECTED_CARD, SELECTED_TARGET, SELECTED_POTION
        if self.uses_action_interaction_token:
            length += 1  # CARD_TARGET_INTERACTION
        if self.uses_structured_transition_tokens:
            length += (
                1  # PLAYER_DELTA
                + self.max_monsters
                + len(ZONE_NAMES)
                + self.max_potions
            )
        if self.uses_phase2_transition_tokens:
            length += (
                1  # PLAYER_AFTER
                + self.max_monsters
                + self.max_power_deltas
            )
        return length

    @property
    def slot_vocab_size(self) -> int:
        return max(self.max_hand, self.max_potions, self.max_monsters) + 2


def token_spec(version: str | None = None) -> V3CombatTransformerTokenSpec:
    return V3CombatTransformerTokenSpec(version=str(version or TOKEN_SCHEMA_VERSION))


@dataclass(frozen=True)
class V3CombatRootTransformerTokenSpec:
    version: str = ROOT_TOKEN_SCHEMA_VERSION
    scalar_dim: int = TRANSFORMER_SCALAR_DIM
    max_hand: int = DEFAULT_MAX_HAND
    max_potions: int = DEFAULT_MAX_POTIONS
    max_monsters: int = DEFAULT_MAX_MONSTERS
    max_player_powers: int = DEFAULT_MAX_PLAYER_POWERS
    max_relics: int = DEFAULT_MAX_RELICS
    max_actions: int = DEFAULT_MAX_ACTIONS

    @property
    def shared_sequence_length(self) -> int:
        return (
            1  # CLS
            + 1  # GLOBAL_BEFORE
            + 1  # PLAYER_CORE
            + self.max_player_powers
            + self.max_monsters
            + self.max_hand
            + self.max_potions
            + self.max_relics
            + len(ZONE_NAMES)
        )

    @property
    def max_sequence_length(self) -> int:
        return self.shared_sequence_length + self.max_actions * self.action_segment_width

    @property
    def action_segment_width(self) -> int:
        if self.version == ROOT_TOKEN_SCHEMA_VERSION_V1:
            return ROOT_ACTION_SEGMENT_WIDTH_V1
        if self.version == ROOT_TOKEN_SCHEMA_VERSION_V2:
            return ROOT_ACTION_SEGMENT_WIDTH_V2
        return ROOT_ACTION_SEGMENT_WIDTH

    @property
    def uses_selected_entity_tokens(self) -> bool:
        return self.version != ROOT_TOKEN_SCHEMA_VERSION_V1

    @property
    def uses_legacy_token(self) -> bool:
        return self.version in {ROOT_TOKEN_SCHEMA_VERSION_V1, ROOT_TOKEN_SCHEMA_VERSION_V2}

    @property
    def slot_vocab_size(self) -> int:
        return max(self.max_hand, self.max_potions, self.max_monsters, self.max_actions) + 2


def root_token_spec() -> V3CombatRootTransformerTokenSpec:
    return V3CombatRootTransformerTokenSpec()


def normalize_candidate_head_variant(value: str | None) -> str:
    raw = str(value or CANDIDATE_HEAD_VARIANT_BASE).strip().lower().replace("_", "-")
    aliases = {
        "default": CANDIDATE_HEAD_VARIANT_BASE,
        "candidate": CANDIDATE_HEAD_VARIANT_BASE,
        "selected": CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY,
        "selected-entity": CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY,
        "selected-head": CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY,
        "action200": CANDIDATE_HEAD_VARIANT_ACTION200,
        "action-200": CANDIDATE_HEAD_VARIANT_ACTION200,
        "action-summary": CANDIDATE_HEAD_VARIANT_ACTION200,
        "relative": CANDIDATE_HEAD_VARIANT_RELATIVE_RANK,
        "relative-rank": CANDIDATE_HEAD_VARIANT_RELATIVE_RANK,
        "dual": CANDIDATE_HEAD_VARIANT_DUAL_GATE,
        "dual-gate": CANDIDATE_HEAD_VARIANT_DUAL_GATE,
        "semantic-legacy": CANDIDATE_HEAD_VARIANT_DUAL_GATE,
        "dual-action200": CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
        "dual-action-200": CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
        "dual-gate-action200": CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
        "dual-gate-action-200": CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
        "dual-gate-action-summary": CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
        "dual-selected": CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY,
        "dual-selected-entity": CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY,
        "dual-gate-selected": CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY,
        "dual-gate-selected-entity": CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY,
        "dual-binding": CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
        "dual-action-binding": CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
        "dual-binding-gate": CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
        "action-binding": CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
        "action-binding-gate": CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
        "semantic": CANDIDATE_HEAD_VARIANT_SEMANTIC_ONLY,
        "semantic-only": CANDIDATE_HEAD_VARIANT_SEMANTIC_ONLY,
        "semantic-transition": CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION,
        "transition": CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION,
        "transition-head": CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION,
        "legacy-delta": CANDIDATE_HEAD_VARIANT_LEGACY_DELTA,
        "residual": CANDIDATE_HEAD_VARIANT_LEGACY_DELTA,
        "semantic-delta": CANDIDATE_HEAD_VARIANT_LEGACY_DELTA,
        "legacy-baseline-delta": CANDIDATE_HEAD_VARIANT_LEGACY_DELTA,
    }
    normalized = aliases.get(raw, raw)
    if normalized not in CANDIDATE_HEAD_VARIANTS:
        raise ValueError(
            f"unsupported candidate head variant: {value!r}; expected one of {sorted(CANDIDATE_HEAD_VARIANTS)}"
        )
    return normalized


def normalize_root_head_variant(value: str | None) -> str:
    raw = str(value or ROOT_HEAD_VARIANT_BASE).strip().lower().replace("_", "-")
    aliases = {
        "default": ROOT_HEAD_VARIANT_BASE,
        "v3": ROOT_HEAD_VARIANT_BASE,
        "v4-selected": ROOT_HEAD_VARIANT_SELECTED,
        "segment": ROOT_HEAD_VARIANT_SELECTED,
        "segment-head": ROOT_HEAD_VARIANT_SELECTED,
        "selected": ROOT_HEAD_VARIANT_SELECTED,
        "action200": ROOT_HEAD_VARIANT_ACTION200,
        "action-200": ROOT_HEAD_VARIANT_ACTION200,
        "action-summary": ROOT_HEAD_VARIANT_ACTION200,
        "shared-pool": ROOT_HEAD_VARIANT_SELECTED_SHARED_POOL,
        "selected-shared": ROOT_HEAD_VARIANT_SELECTED_SHARED_POOL,
    }
    normalized = aliases.get(raw, raw)
    if normalized not in ROOT_HEAD_VARIANTS:
        raise ValueError(f"unsupported root head variant: {value!r}; expected one of {sorted(ROOT_HEAD_VARIANTS)}")
    return normalized


def infer_root_token_schema_version(*, max_sequence_length: int | None, max_actions: int | None) -> str:
    if not max_sequence_length:
        return ROOT_TOKEN_SCHEMA_VERSION
    actions = int(max_actions or DEFAULT_MAX_ACTIONS)
    for version in (ROOT_TOKEN_SCHEMA_VERSION_V1, ROOT_TOKEN_SCHEMA_VERSION_V2, ROOT_TOKEN_SCHEMA_VERSION):
        spec = V3CombatRootTransformerTokenSpec(version=version, max_actions=actions)
        if int(max_sequence_length) == int(spec.max_sequence_length):
            return version
    return ROOT_TOKEN_SCHEMA_VERSION


def token_spec_from_payload(payload: dict[str, Any] | None) -> V3CombatTransformerTokenSpec:
    raw = dict(payload or {})
    return V3CombatTransformerTokenSpec(
        version=str(raw.get("version") or TOKEN_SCHEMA_VERSION),
        scalar_dim=int(raw.get("scalar_dim") or TRANSFORMER_SCALAR_DIM),
        max_hand=int(raw.get("max_hand") or DEFAULT_MAX_HAND),
        max_potions=int(raw.get("max_potions") or DEFAULT_MAX_POTIONS),
        max_monsters=int(raw.get("max_monsters") or DEFAULT_MAX_MONSTERS),
        max_player_powers=int(raw.get("max_player_powers") or DEFAULT_MAX_PLAYER_POWERS),
        max_relics=int(raw.get("max_relics") or DEFAULT_MAX_RELICS),
        max_power_deltas=int(raw.get("max_power_deltas") or DEFAULT_MAX_POWER_DELTAS),
    )


def root_token_spec_from_payload(payload: dict[str, Any] | None) -> V3CombatRootTransformerTokenSpec:
    raw = dict(payload or {})
    return V3CombatRootTransformerTokenSpec(
        version=str(raw.get("version") or ROOT_TOKEN_SCHEMA_VERSION),
        scalar_dim=int(raw.get("scalar_dim") or TRANSFORMER_SCALAR_DIM),
        max_hand=int(raw.get("max_hand") or DEFAULT_MAX_HAND),
        max_potions=int(raw.get("max_potions") or DEFAULT_MAX_POTIONS),
        max_monsters=int(raw.get("max_monsters") or DEFAULT_MAX_MONSTERS),
        max_player_powers=int(raw.get("max_player_powers") or DEFAULT_MAX_PLAYER_POWERS),
        max_relics=int(raw.get("max_relics") or DEFAULT_MAX_RELICS),
        max_actions=int(raw.get("max_actions") or DEFAULT_MAX_ACTIONS),
    )


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _pad(values: list[float], size: int = TRANSFORMER_SCALAR_DIM) -> list[float]:
    if len(values) >= size:
        return [float(value) for value in values[:size]]
    return [float(value) for value in values] + [0.0] * (size - len(values))


def _id_from_payload(payload: dict[str, Any] | None, *keys: str) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


@lru_cache(maxsize=1)
def monster_identity_ids() -> tuple[str, ...]:
    try:
        from spirecomm.native_sim_v3.content.monsters import monster_catalog

        ids = sorted(str(monster_id) for monster_id in monster_catalog().keys())
    except Exception:
        ids = []
    return ("__UNK__", *ids)


@lru_cache(maxsize=1)
def relic_identity_ids() -> tuple[str, ...]:
    try:
        from spirecomm.native_sim_v3.content.relics import relic_catalog

        ids = sorted(str(relic_id) for relic_id in relic_catalog().keys())
    except Exception:
        ids = []
    return ("__UNK__", *ids)


@lru_cache(maxsize=1)
def player_power_identity_ids() -> tuple[str, ...]:
    return ("__UNK__", *sorted(set(PLAYER_POWER_IDS)))


@lru_cache(maxsize=1)
def extra_entity_ids() -> tuple[str, ...]:
    action_kinds = ("end", "card", "potion", "card_select", "card_reward", "unknown")
    zones = tuple(f"zone:{name}" for name in ZONE_NAMES)
    actions = tuple(f"action:{kind}" for kind in action_kinds)
    return ("__NONE__", *zones, *actions)


@lru_cache(maxsize=1)
def transformer_entity_ids() -> tuple[str, ...]:
    ids: list[str] = ["__NONE__", "__UNK__"]
    ids.extend(f"card:{card_id}" for card_id in card_identity_ids()[1:])
    ids.extend(f"potion:{potion_id}" for potion_id in potion_identity_ids()[1:])
    ids.extend(f"monster:{monster_id}" for monster_id in monster_identity_ids()[1:])
    ids.extend(f"relic:{relic_id}" for relic_id in relic_identity_ids()[1:])
    ids.extend(f"power:{power_id}" for power_id in player_power_identity_ids()[1:])
    ids.extend(extra_entity_ids()[1:])
    deduped: list[str] = []
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


@lru_cache(maxsize=1)
def transformer_entity_index() -> dict[str, int]:
    return {entity_id: index for index, entity_id in enumerate(transformer_entity_ids())}


def entity_index_from_vocab(entity_vocab: list[str] | tuple[str, ...] | None) -> dict[str, int] | None:
    if not entity_vocab:
        return None
    return {str(entity_id): index for index, entity_id in enumerate(entity_vocab)}


def _entity_id(namespace: str, value: str | None, entity_index: dict[str, int] | None = None) -> int:
    if not value:
        return 0
    key = f"{namespace}:{value}"
    index = entity_index or transformer_entity_index()
    return index.get(key, index.get("__UNK__", 1))


def _extra_entity_id(value: str, entity_index: dict[str, int] | None = None) -> int:
    index = entity_index or transformer_entity_index()
    return index.get(value, 0)


def _card_cost(card: dict[str, Any] | None) -> int:
    if not card:
        return 0
    cost = card.get("cost_for_turn")
    if cost is None:
        cost = card.get("cost")
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 0


def _tier_flags(tier: str) -> list[float]:
    tiers = ["STARTER", "COMMON", "UNCOMMON", "RARE", "SHOP", "BOSS", "SPECIAL"]
    return [_bool(str(tier or "").upper() == name) for name in tiers]


def _rarity_flags(rarity: str) -> list[float]:
    return [_bool(str(rarity or "").upper() == name) for name in CARD_RARITIES]


def _type_flags(card_type: str) -> list[float]:
    return [_bool(str(card_type or "").upper() == name) for name in CARD_TYPES]


def _intent_flags(intent: str) -> list[float]:
    raw = str(intent or "UNKNOWN")
    if raw not in INTENTS:
        raw = "UNKNOWN"
    return [_bool(raw == name) for name in INTENTS]


def _card_scalars(card: dict[str, Any], *, selected: bool = False) -> list[float]:
    card_type = str(card.get("type") or "")
    rarity = str(card.get("rarity") or "")
    color = str(card.get("color") or "")
    return _pad(
        [
            _clip_norm(_card_cost(card), 4.0, -2.0, 3.0),
            _clip_norm(card.get("base_damage", 0), 50.0, -1.0, 3.0),
            _clip_norm(card.get("base_block", 0), 40.0, 0.0, 3.0),
            _clip_norm(card.get("base_magic", 0), 8.0, -3.0, 3.0),
            _clip_norm(card.get("upgrades", 0), 3.0, 0.0, 3.0),
            _bool(card.get("is_playable", False)),
            _bool(card.get("has_target", False)),
            _bool(card.get("exhausts", False)),
            _bool(card.get("ethereal", False)),
            _bool(card.get("retain", False) or card.get("self_retain", False)),
            _bool(card.get("innate", False)),
            _bool(card.get("free_to_play_once", False)),
            _bool(selected),
            _bool(color == "RED"),
            _bool(color == "COLORLESS"),
            _bool(color == "CURSE"),
            _bool(color == "STATUS"),
            *_type_flags(card_type),
            *_rarity_flags(rarity),
        ]
    )


def _potion_scalars(potion: dict[str, Any], *, selected: bool = False) -> list[float]:
    potion_id = _id_from_payload(potion, "potion_id", "id", "name")
    rarity = ""
    try:
        from spirecomm.native_sim_v3.content.potions import potion_rarity_map

        rarity = str(potion_rarity_map().get(potion_id) or "")
    except Exception:
        rarity = ""
    return _pad(
        [
            _bool(potion_id == "Potion Slot"),
            _bool(potion.get("requires_target", False)),
            _bool(potion.get("can_use", False)),
            _bool(potion.get("can_discard", False)),
            _bool(selected),
            _bool(rarity == "COMMON"),
            _bool(rarity == "UNCOMMON"),
            _bool(rarity == "RARE"),
        ]
    )


def _monster_scalars(monster: dict[str, Any], *, selected: bool = False) -> list[float]:
    max_hp = max(1, int(monster.get("max_hp") or 1))
    current_hp = int(monster.get("current_hp") or 0)
    powers = _powers_by_id(list(monster.get("powers") or []))
    positive_total = sum(abs(amount) for amount in powers.values() if amount > 0)
    negative_total = sum(abs(amount) for amount in powers.values() if amount < 0)
    intent = str(monster.get("intent") or "UNKNOWN")
    return _pad(
        [
            _clip_norm(current_hp, 350.0, 0.0, 5.0),
            _clip_norm(max_hp, 350.0, 0.0, 5.0),
            max(0.0, min(1.5, float(current_hp) / float(max_hp))),
            _clip_norm(monster.get("block", 0), 40.0, 0.0, 5.0),
            _clip_norm(monster.get("move_adjusted_damage", 0), 50.0, -2.0, 3.0),
            _clip_norm(monster.get("move_hits", 0), 5.0, 0.0, 3.0),
            _bool(intent in ATTACK_INTENTS),
            _bool(_is_alive(monster)),
            _bool(monster.get("half_dead", False)),
            _bool(monster.get("is_gone", False)),
            _bool(selected),
            _clip_norm(positive_total, 20.0, 0.0, 5.0),
            _clip_norm(negative_total, 20.0, 0.0, 5.0),
            *_intent_flags(intent),
        ]
    )


def _player_core_scalars(state: dict[str, Any]) -> list[float]:
    combat = combat_state(state)
    player = dict(combat.get("player") or {})
    all_monsters = monsters(state)
    hand = list(combat.get("hand") or [])
    draw_pile = list(combat.get("draw_pile") or [])
    discard_pile = list(combat.get("discard_pile") or [])
    exhaust_pile = list(combat.get("exhaust_pile") or [])
    deck = list(state.get("deck") or [])
    current_hp = int(state.get("current_hp") or player.get("current_hp") or 0)
    max_hp = max(1, int(state.get("max_hp") or player.get("max_hp") or 1))
    room_type = str(state.get("room_type") or "")
    return _pad(
        [
            _clip_norm(state.get("act", 1), 4.0, 0.0, 2.0),
            _clip_norm(state.get("floor", 0), 60.0, 0.0, 2.0),
            _clip_norm(combat.get("turn", 0), 10.0, 0.0, 3.0),
            *[_bool(room_type == candidate) for candidate in COMBAT_ROOM_TYPES],
            _clip_norm(current_hp, 90.0, 0.0, 2.0),
            _clip_norm(max_hp, 90.0, 0.0, 2.0),
            max(0.0, min(1.5, float(current_hp) / float(max_hp))),
            _clip_norm(player.get("block", 0), 40.0, 0.0, 5.0),
            _clip_norm(player.get("energy", 0), 6.0, 0.0, 3.0),
            _clip_norm(state.get("gold", 0), 500.0, 0.0, 5.0),
            _clip_norm(len(hand), 10.0, 0.0, 2.0),
            _clip_norm(len(draw_pile), 25.0, 0.0, 3.0),
            _clip_norm(len(discard_pile), 25.0, 0.0, 3.0),
            _clip_norm(len(exhaust_pile), 15.0, 0.0, 3.0),
            _clip_norm(len(deck), 35.0, 0.0, 3.0),
            _clip_norm(len(state.get("relics") or []), 30.0, 0.0, 3.0),
            _clip_norm(len(state.get("potions") or []), 5.0, 0.0, 2.0),
            _clip_norm(_incoming_damage_from_monsters(all_monsters), 60.0, 0.0, 5.0),
        ]
    )


def _power_scalars(power_id: str, amount: float) -> list[float]:
    return _pad(
        [
            _clip_norm(amount, 10.0, -5.0, 5.0),
            _bool(amount > 0),
            _bool(amount < 0),
            _bool(power_id in {"Weakened", "Weak", "Vulnerable", "Frail", "No Draw", "Draw Reduction", "Entangled"}),
        ]
    )


def _relic_scalars(relic: dict[str, Any]) -> list[float]:
    counter = relic.get("counter", relic.get("counter_value", 0))
    return _pad([_clip_norm(counter, 12.0, -3.0, 5.0), *_tier_flags(str(relic.get("tier") or ""))])


def _monster_raw_incoming(monster: dict[str, Any] | None) -> int:
    if not isinstance(monster, dict) or not _is_alive(monster):
        return 0
    if str(monster.get("intent") or "UNKNOWN") not in ATTACK_INTENTS:
        return 0
    damage = int(monster.get("move_adjusted_damage") or 0)
    hits = max(1, int(monster.get("move_hits") or 1))
    return max(0, damage) * hits


def _player_delta_scalars(before_state: dict[str, Any], action: dict[str, Any], after_state: dict[str, Any]) -> list[float]:
    return _pad(encode_delta(before_state, action, after_state))


def _monster_delta_scalars(
    before_monster: dict[str, Any] | None,
    after_monster: dict[str, Any] | None,
    *,
    selected: bool = False,
) -> list[float]:
    before_monster = before_monster or {}
    after_monster = after_monster or {}
    before_max_hp = max(1, int(before_monster.get("max_hp") or after_monster.get("max_hp") or 1))
    after_max_hp = max(1, int(after_monster.get("max_hp") or before_monster.get("max_hp") or 1))
    before_hp = int(before_monster.get("current_hp") or 0)
    after_hp = int(after_monster.get("current_hp") or 0)
    before_alive = _is_alive(before_monster)
    after_alive = _is_alive(after_monster)
    before_powers = _powers_by_id(list(before_monster.get("powers") or []))
    after_powers = _powers_by_id(list(after_monster.get("powers") or []))
    strength_delta = after_powers.get("Strength", 0.0) - before_powers.get("Strength", 0.0)
    vulnerable_delta = after_powers.get("Vulnerable", 0.0) - before_powers.get("Vulnerable", 0.0)
    weak_delta = after_powers.get("Weak", 0.0) - before_powers.get("Weak", 0.0)
    hp_delta = after_hp - before_hp
    block_delta = int(after_monster.get("block") or 0) - int(before_monster.get("block") or 0)
    incoming_delta = _monster_raw_incoming(after_monster) - _monster_raw_incoming(before_monster)
    affected = (
        abs(hp_delta) > 0
        or abs(block_delta) > 0
        or (before_alive and not after_alive)
        or abs(incoming_delta) > 0
        or bool(selected)
        or abs(strength_delta) > 1.0e-6
        or abs(vulnerable_delta) > 1.0e-6
        or abs(weak_delta) > 1.0e-6
    )
    return _pad(
        [
            _clip_norm(hp_delta, 80.0, -5.0, 5.0),
            max(0.0, min(1.5, float(before_hp) / float(before_max_hp))),
            max(0.0, min(1.5, float(after_hp) / float(after_max_hp))),
            _clip_norm(block_delta, 40.0, -5.0, 5.0),
            _bool(before_alive and not after_alive),
            _bool(after_alive),
            _clip_norm(incoming_delta, 50.0, -5.0, 5.0),
            _bool(selected),
            _clip_norm(strength_delta, 10.0, -5.0, 5.0),
            _clip_norm(vulnerable_delta, 5.0, -5.0, 5.0),
            _clip_norm(weak_delta, 5.0, -5.0, 5.0),
            _bool(str(before_monster.get("intent") or "") != str(after_monster.get("intent") or "")),
            _bool(affected),
        ]
    )


def _zone_delta_scalars(before_state: dict[str, Any], after_state: dict[str, Any], zone_name: str) -> list[float]:
    before_cards = _zone_cards(before_state, zone_name)
    after_cards = _zone_cards(after_state, zone_name)
    before_summary = _zone_summary(before_cards)
    after_summary = _zone_summary(after_cards)
    return _pad(
        [
            _clip_norm(len(after_cards) - len(before_cards), 20.0, -5.0, 5.0),
            *[float(after_value) - float(before_value) for before_value, after_value in zip(before_summary, after_summary, strict=False)],
        ]
    )


def _potion_slot_delta_scalars(before_potion: dict[str, Any] | None, after_potion: dict[str, Any] | None) -> list[float]:
    before_id = _id_from_payload(before_potion, "potion_id", "id", "name")
    after_id = _id_from_payload(after_potion, "potion_id", "id", "name")
    before_empty = not before_id or before_id == "Potion Slot"
    after_empty = not after_id or after_id == "Potion Slot"
    return _pad(
        [
            _bool(before_empty),
            _bool(after_empty),
            _bool(not before_empty and after_empty),
            _bool(before_empty and not after_empty),
            _bool(before_id != after_id),
            _bool(before_id == after_id and not before_empty),
        ]
    )


def _power_delta_records(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    *,
    limit: int,
    entity_index: dict[str, int] | None = None,
) -> list[tuple[list[float], int, int]]:
    records: list[tuple[float, list[float], int, int]] = []

    def add_owner(owner_kind: str, slot_id: int, before_powers: dict[str, float], after_powers: dict[str, float]) -> None:
        for power_id in sorted(set(before_powers) | set(after_powers)):
            before_amount = float(before_powers.get(power_id, 0.0))
            after_amount = float(after_powers.get(power_id, 0.0))
            delta = after_amount - before_amount
            if abs(delta) <= 1.0e-6:
                continue
            scalars = _pad(
                [
                    _bool(owner_kind == "player"),
                    _bool(owner_kind == "monster"),
                    _clip_norm(slot_id, 8.0, 0.0, 8.0),
                    _clip_norm(before_amount, 10.0, -5.0, 5.0),
                    _clip_norm(after_amount, 10.0, -5.0, 5.0),
                    _clip_norm(delta, 10.0, -5.0, 5.0),
                    _bool(abs(before_amount) <= 1.0e-6 and abs(after_amount) > 1.0e-6),
                    _bool(abs(before_amount) > 1.0e-6 and abs(after_amount) <= 1.0e-6),
                ]
            )
            records.append((abs(delta), scalars, _entity_id("power", power_id, entity_index), slot_id))

    before_player = dict(combat_state(before_state).get("player") or {})
    after_player = dict(combat_state(after_state).get("player") or {})
    add_owner("player", 0, _powers_by_id(list(before_player.get("powers") or [])), _powers_by_id(list(after_player.get("powers") or [])))
    before_monsters = monsters(before_state)
    after_monsters = monsters(after_state)
    for index in range(max(len(before_monsters), len(after_monsters))):
        before_monster = before_monsters[index] if index < len(before_monsters) else {}
        after_monster = after_monsters[index] if index < len(after_monsters) else {}
        add_owner(
            "monster",
            index + 1,
            _powers_by_id(list(before_monster.get("powers") or [])),
            _powers_by_id(list(after_monster.get("powers") or [])),
        )

    records.sort(key=lambda item: (-item[0], item[3], item[2]))
    return [(scalars, entity_id, slot_id) for _magnitude, scalars, entity_id, slot_id in records[: max(0, int(limit))]]


def _action_entity_id(before_state: dict[str, Any], action: dict[str, Any], entity_index: dict[str, int] | None = None) -> int:
    kind = str(action.get("kind") or "")
    if kind == "card":
        card = _selected_card(before_state, action)
        return _entity_id("card", _id_from_payload(card or action, "card_id", "id", "name"), entity_index)
    if kind == "potion":
        potion = _selected_potion(before_state, action)
        return _entity_id("potion", _id_from_payload(potion or action, "potion_id", "id", "name"), entity_index)
    return _extra_entity_id(f"action:{kind or 'unknown'}", entity_index)


def _selected_index(action: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = action.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _bounded_indices(length: int, limit: int, selected: int | None = None) -> list[int]:
    indices = list(range(min(length, limit)))
    if selected is not None and 0 <= selected < length and selected not in indices and limit > 0:
        if len(indices) >= limit:
            indices[-1] = selected
        else:
            indices.append(selected)
    return indices


def _action_scalars(before_state: dict[str, Any], action: dict[str, Any], scalar_dim: int = TRANSFORMER_SCALAR_DIM) -> list[float]:
    values = _pad(encode_action_summary_base(before_state, action), scalar_dim)
    try:
        key = action_key(action, before_state)
        values[-4:] = [
            _clip_norm(key[1] if len(key) > 1 and isinstance(key[1], int) else 0, 10.0, -1.0, 3.0),
            _clip_norm(key[3] if len(key) > 3 and isinstance(key[3], int) else 0, 5.0, -1.0, 3.0),
            _clip_norm(key[4] if len(key) > 4 and isinstance(key[4], int) else 0, 10.0, -1.0, 3.0),
            _bool(str(action.get("kind") or "") == "end"),
        ]
    except Exception:
        pass
    return values


def _card_target_interaction_scalars(
    before_state: dict[str, Any],
    action: dict[str, Any],
    after_state: dict[str, Any],
) -> list[float]:
    kind = str(action.get("kind") or "")
    selected_card = _selected_card(before_state, action)
    selected_potion = _selected_potion(before_state, action)
    target_index = _selected_index(action, "target_index")
    selected_target = _selected_target(before_state, action)
    after_monsters = monsters(after_state)
    after_target = after_monsters[target_index] if target_index is not None and 0 <= target_index < len(after_monsters) else None

    target_hp_before = int((selected_target or {}).get("current_hp") or 0)
    target_block_before = int((selected_target or {}).get("block") or 0)
    target_incoming_before = _monster_raw_incoming(selected_target)
    target_hp_after = int((after_target or {}).get("current_hp") or 0)
    before_alive = selected_target is not None and _is_alive(selected_target)
    after_alive = after_target is not None and _is_alive(after_target)
    direct_damage = max(0, target_hp_before - target_hp_after) if selected_target is not None else 0

    before_player = dict(combat_state(before_state).get("player") or {})
    after_player = dict(combat_state(after_state).get("player") or {})
    block_delta = int(after_player.get("block") or 0) - int(before_player.get("block") or 0)
    energy_cost = _card_cost(selected_card) if selected_card is not None else int(action.get("energy_cost") or 0)
    if selected_potion is not None:
        energy_cost = 0

    return _pad(
        [
            _bool(kind == "card"),
            _bool(kind == "potion"),
            _bool(kind == "end"),
            _bool(kind not in {"card", "potion", "end"}),
            _clip_norm(energy_cost, 4.0, -2.0, 3.0),
            _clip_norm(target_hp_before, 350.0, 0.0, 5.0),
            _clip_norm(target_block_before, 60.0, 0.0, 5.0),
            _clip_norm(target_incoming_before, 80.0, 0.0, 5.0),
            _clip_norm(direct_damage, 80.0, 0.0, 5.0),
            _clip_norm(max(0, block_delta), 80.0, 0.0, 5.0),
            _bool(before_alive and not after_alive),
            _bool(selected_target is not None),
            _bool(target_index is not None and target_index >= 0),
            _bool(selected_card is not None),
            _bool(selected_potion is not None),
        ]
    )


def encode_transformer_candidate(
    before_state: dict[str, Any],
    action: dict[str, Any],
    after_state: dict[str, Any],
    *,
    candidate_features: list[float] | None = None,
    spec: V3CombatTransformerTokenSpec | None = None,
    entity_index: dict[str, int] | None = None,
) -> dict[str, Any]:
    spec = spec or token_spec()
    max_len = spec.max_sequence_length
    token_scalar_features = [[0.0] * spec.scalar_dim for _ in range(max_len)]
    token_type_ids = [TOKEN_TYPES["PAD"]] * max_len
    entity_ids = [0] * max_len
    slot_ids = [0] * max_len
    attention_mask = [False] * max_len
    position = 0

    def add_token(token_type: str, scalars: list[float] | None = None, *, entity_id: int = 0, slot_id: int = 0) -> None:
        nonlocal position
        if position >= max_len:
            raise ValueError(f"v3 combat transformer token overflow at {token_type}")
        token_scalar_features[position] = _pad(list(scalars or []), spec.scalar_dim)
        token_type_ids[position] = TOKEN_TYPES[token_type]
        entity_ids[position] = int(entity_id)
        slot_ids[position] = int(slot_id)
        attention_mask[position] = True
        position += 1

    add_token("CLS")
    add_token("GLOBAL_BEFORE")
    add_token("PLAYER_CORE", _player_core_scalars(before_state))

    player = dict(combat_state(before_state).get("player") or {})
    powers = _powers_by_id(list(player.get("powers") or []))
    ordered_powers = sorted(powers.items(), key=lambda item: (-abs(float(item[1])), str(item[0])))[: spec.max_player_powers]
    for power_id, amount in ordered_powers:
        add_token("PLAYER_POWER", _power_scalars(power_id, float(amount)), entity_id=_entity_id("power", power_id, entity_index))

    target_index = _selected_index(action, "target_index")
    all_monsters = monsters(before_state)
    for index in _bounded_indices(len(all_monsters), spec.max_monsters, target_index):
        monster = all_monsters[index]
        monster_id = _id_from_payload(monster, "monster_id", "id", "name")
        add_token(
            "MONSTER",
            _monster_scalars(monster, selected=index == target_index),
            entity_id=_entity_id("monster", monster_id, entity_index),
            slot_id=index + 1,
        )

    selected_card_index = _selected_index(action, "card_index", "source_index")
    all_hand_cards = hand_cards(before_state)
    for index in _bounded_indices(len(all_hand_cards), spec.max_hand, selected_card_index):
        card = all_hand_cards[index]
        card_id = _id_from_payload(card, "card_id", "id", "name")
        add_token(
            "HAND_CARD",
            _card_scalars(card, selected=index == selected_card_index),
            entity_id=_entity_id("card", card_id, entity_index),
            slot_id=index + 1,
        )

    selected_potion_index = _selected_index(action, "potion_index")
    all_potions = list(before_state.get("potions") or [])
    for index in _bounded_indices(len(all_potions), spec.max_potions, selected_potion_index):
        potion = all_potions[index]
        potion_id = _id_from_payload(potion, "potion_id", "id", "name")
        add_token(
            "POTION",
            _potion_scalars(potion, selected=index == selected_potion_index),
            entity_id=_entity_id("potion", potion_id, entity_index),
            slot_id=index + 1,
        )

    for relic in list(before_state.get("relics") or [])[: spec.max_relics]:
        relic_id = _id_from_payload(relic, "relic_id", "id", "name")
        add_token("RELIC", _relic_scalars(relic), entity_id=_entity_id("relic", relic_id, entity_index))

    for zone_name in ZONE_NAMES:
        add_token("ZONE_SUMMARY", _zone_summary(_zone_cards(before_state, zone_name)), entity_id=_extra_entity_id(f"zone:{zone_name}", entity_index))

    action_scalars = _action_scalars(before_state, action, spec.scalar_dim)
    add_token("ACTION", action_scalars, entity_id=_action_entity_id(before_state, action, entity_index))
    if spec.uses_selected_entity_tokens:
        selected_card = _selected_card(before_state, action)
        selected_target = _selected_target(before_state, action)
        selected_potion = _selected_potion(before_state, action)
        add_token(
            "SELECTED_CARD",
            _card_scalars(selected_card, selected=True) if selected_card else None,
            entity_id=_entity_id("card", _id_from_payload(selected_card, "card_id", "id", "name"), entity_index)
            if selected_card
            else 0,
        )
        add_token(
            "SELECTED_TARGET",
            _monster_scalars(selected_target, selected=True) if selected_target else None,
            entity_id=_entity_id("monster", _id_from_payload(selected_target, "monster_id", "id", "name"), entity_index)
            if selected_target
            else 0,
        )
        add_token(
            "SELECTED_POTION",
            _potion_scalars(selected_potion, selected=True) if selected_potion else None,
            entity_id=_entity_id("potion", _id_from_payload(selected_potion, "potion_id", "id", "name"), entity_index)
            if selected_potion
            else 0,
        )
    if spec.uses_action_interaction_token:
        interaction_slot = (target_index + 1) if target_index is not None and target_index >= 0 else 0
        add_token(
            "CARD_TARGET_INTERACTION",
            _card_target_interaction_scalars(before_state, action, after_state),
            entity_id=_action_entity_id(before_state, action, entity_index),
            slot_id=interaction_slot,
        )
    add_token("AFTER_SUMMARY")
    add_token("DELTA")
    if spec.uses_structured_transition_tokens:
        add_token("PLAYER_DELTA", _player_delta_scalars(before_state, action, after_state))
        after_monsters = monsters(after_state)
        for index in _bounded_indices(len(all_monsters), spec.max_monsters, target_index):
            before_monster = all_monsters[index]
            after_monster = after_monsters[index] if index < len(after_monsters) else {}
            monster_id = _id_from_payload(before_monster or after_monster, "monster_id", "id", "name")
            add_token(
                "MONSTER_DELTA",
                _monster_delta_scalars(before_monster, after_monster, selected=index == target_index),
                entity_id=_entity_id("monster", monster_id, entity_index),
                slot_id=index + 1,
            )
        for zone_name in ZONE_NAMES:
            add_token(
                "ZONE_DELTA",
                _zone_delta_scalars(before_state, after_state, zone_name),
                entity_id=_extra_entity_id(f"zone:{zone_name}", entity_index),
            )
        after_potions = list(after_state.get("potions") or [])
        for index in range(spec.max_potions):
            before_potion = all_potions[index] if index < len(all_potions) else None
            after_potion = after_potions[index] if index < len(after_potions) else None
            potion_id = _id_from_payload(after_potion or before_potion, "potion_id", "id", "name")
            add_token(
                "POTION_SLOT_DELTA",
                _potion_slot_delta_scalars(before_potion, after_potion),
                entity_id=_entity_id("potion", potion_id, entity_index),
                slot_id=index + 1,
            )
    if spec.uses_phase2_transition_tokens:
        add_token("PLAYER_AFTER", _player_core_scalars(after_state))
        after_monsters = monsters(after_state)
        for index in _bounded_indices(max(len(all_monsters), len(after_monsters)), spec.max_monsters, target_index):
            before_monster = all_monsters[index] if index < len(all_monsters) else {}
            after_monster = after_monsters[index] if index < len(after_monsters) else {}
            monster_id = _id_from_payload(after_monster or before_monster, "monster_id", "id", "name")
            add_token(
                "MONSTER_AFTER",
                _monster_scalars(after_monster, selected=index == target_index) if after_monster else None,
                entity_id=_entity_id("monster", monster_id, entity_index),
                slot_id=index + 1,
            )
        for scalars, entity_id, slot_id in _power_delta_records(
            before_state,
            after_state,
            limit=spec.max_power_deltas,
            entity_index=entity_index,
        ):
            add_token("POWER_DELTA", scalars, entity_id=entity_id, slot_id=slot_id)
    add_token("LEGACY")

    features = candidate_features if candidate_features is not None else encode_candidate(before_state, action, after_state)
    return {
        "token_scalar_features": token_scalar_features,
        "token_type_ids": token_type_ids,
        "entity_ids": entity_ids,
        "slot_ids": slot_ids,
        "attention_mask": attention_mask,
        "candidate_features": list(features),
    }


def encode_transformer_candidates_shared_before(
    before_state: dict[str, Any],
    actions: list[dict[str, Any]],
    after_states: list[dict[str, Any]],
    *,
    candidate_features: list[list[float]],
    spec: V3CombatTransformerTokenSpec | None = None,
    entity_index: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Encode same-root candidates while reusing before-state token work.

    This fast path is intentionally limited to the v3-style token schema used by
    the current rollout checkpoint. Newer schemas add action-dependent
    transition tokens, so they fall back to encode_transformer_candidate().
    """

    spec = spec or token_spec()
    if (
        spec.uses_selected_entity_tokens
        or spec.uses_structured_transition_tokens
        or spec.uses_phase2_transition_tokens
        or spec.uses_action_interaction_token
        or len(actions) != len(after_states)
        or len(actions) != len(candidate_features)
    ):
        return [
            encode_transformer_candidate(
                before_state,
                action,
                after_state,
                candidate_features=features,
                spec=spec,
                entity_index=entity_index,
            )
            for action, after_state, features in zip(actions, after_states, candidate_features, strict=False)
        ]

    all_monsters = monsters(before_state)
    all_hand_cards = hand_cards(before_state)
    all_potions = list(before_state.get("potions") or [])
    if (
        len(all_monsters) > spec.max_monsters
        or len(all_hand_cards) > spec.max_hand
        or len(all_potions) > spec.max_potions
    ):
        return [
            encode_transformer_candidate(
                before_state,
                action,
                after_state,
                candidate_features=features,
                spec=spec,
                entity_index=entity_index,
            )
            for action, after_state, features in zip(actions, after_states, candidate_features, strict=False)
        ]

    max_len = spec.max_sequence_length
    base_scalars = [[0.0] * spec.scalar_dim for _ in range(max_len)]
    base_token_types = [TOKEN_TYPES["PAD"]] * max_len
    base_entity_ids = [0] * max_len
    base_slot_ids = [0] * max_len
    base_attention = [False] * max_len
    position = 0
    monster_positions: dict[int, int] = {}
    hand_positions: dict[int, int] = {}
    potion_positions: dict[int, int] = {}

    def add_base(token_type: str, scalars: list[float] | None = None, *, entity_id: int = 0, slot_id: int = 0) -> int:
        nonlocal position
        if position >= max_len:
            raise ValueError(f"v3 combat transformer token overflow at {token_type}")
        index = position
        base_scalars[index] = _pad(list(scalars or []), spec.scalar_dim)
        base_token_types[index] = TOKEN_TYPES[token_type]
        base_entity_ids[index] = int(entity_id)
        base_slot_ids[index] = int(slot_id)
        base_attention[index] = True
        position += 1
        return index

    add_base("CLS")
    add_base("GLOBAL_BEFORE")
    add_base("PLAYER_CORE", _player_core_scalars(before_state))

    player = dict(combat_state(before_state).get("player") or {})
    powers = _powers_by_id(list(player.get("powers") or []))
    ordered_powers = sorted(powers.items(), key=lambda item: (-abs(float(item[1])), str(item[0])))[: spec.max_player_powers]
    for power_id, amount in ordered_powers:
        add_base("PLAYER_POWER", _power_scalars(power_id, float(amount)), entity_id=_entity_id("power", power_id, entity_index))

    for index, monster in enumerate(all_monsters):
        monster_id = _id_from_payload(monster, "monster_id", "id", "name")
        monster_positions[index] = add_base(
            "MONSTER",
            _monster_scalars(monster, selected=False),
            entity_id=_entity_id("monster", monster_id, entity_index),
            slot_id=index + 1,
        )

    for index, card in enumerate(all_hand_cards):
        card_id = _id_from_payload(card, "card_id", "id", "name")
        hand_positions[index] = add_base(
            "HAND_CARD",
            _card_scalars(card, selected=False),
            entity_id=_entity_id("card", card_id, entity_index),
            slot_id=index + 1,
        )

    for index, potion in enumerate(all_potions):
        potion_id = _id_from_payload(potion, "potion_id", "id", "name")
        potion_positions[index] = add_base(
            "POTION",
            _potion_scalars(potion, selected=False),
            entity_id=_entity_id("potion", potion_id, entity_index),
            slot_id=index + 1,
        )

    for relic in list(before_state.get("relics") or [])[: spec.max_relics]:
        relic_id = _id_from_payload(relic, "relic_id", "id", "name")
        add_base("RELIC", _relic_scalars(relic), entity_id=_entity_id("relic", relic_id, entity_index))

    for zone_name in ZONE_NAMES:
        add_base("ZONE_SUMMARY", _zone_summary(_zone_cards(before_state, zone_name)), entity_id=_extra_entity_id(f"zone:{zone_name}", entity_index))

    action_position = position

    def make_record(action: dict[str, Any], features: list[float]) -> dict[str, Any]:
        token_scalar_features = [row.copy() for row in base_scalars]
        token_type_ids = list(base_token_types)
        entity_ids = list(base_entity_ids)
        slot_ids = list(base_slot_ids)
        attention_mask = list(base_attention)

        target_index = _selected_index(action, "target_index")
        if target_index is not None and target_index in monster_positions:
            row_index = monster_positions[target_index]
            token_scalar_features[row_index] = token_scalar_features[row_index].copy()
            token_scalar_features[row_index][10] = 1.0
        selected_card_index = _selected_index(action, "card_index", "source_index")
        if selected_card_index is not None and selected_card_index in hand_positions:
            row_index = hand_positions[selected_card_index]
            token_scalar_features[row_index] = token_scalar_features[row_index].copy()
            token_scalar_features[row_index][12] = 1.0
        selected_potion_index = _selected_index(action, "potion_index")
        if selected_potion_index is not None and selected_potion_index in potion_positions:
            row_index = potion_positions[selected_potion_index]
            token_scalar_features[row_index] = token_scalar_features[row_index].copy()
            token_scalar_features[row_index][4] = 1.0

        position = action_position
        for token_type, scalars, entity_id in (
            ("ACTION", _action_scalars(before_state, action, spec.scalar_dim), _action_entity_id(before_state, action, entity_index)),
            ("AFTER_SUMMARY", None, 0),
            ("DELTA", None, 0),
            ("LEGACY", None, 0),
        ):
            if position >= max_len:
                raise ValueError(f"v3 combat transformer token overflow at {token_type}")
            token_scalar_features[position] = _pad(list(scalars or []), spec.scalar_dim)
            token_type_ids[position] = TOKEN_TYPES[token_type]
            entity_ids[position] = int(entity_id)
            slot_ids[position] = 0
            attention_mask[position] = True
            position += 1
        return {
            "token_scalar_features": token_scalar_features,
            "token_type_ids": token_type_ids,
            "entity_ids": entity_ids,
            "slot_ids": slot_ids,
            "attention_mask": attention_mask,
            "candidate_features": list(features),
        }

    return [make_record(action, features) for action, features in zip(actions, candidate_features, strict=False)]


def collate_transformer_candidates_shared_before(
    before_state: dict[str, Any],
    actions: list[dict[str, Any]],
    after_states: list[dict[str, Any]],
    *,
    candidate_features: list[list[float]],
    spec: V3CombatTransformerTokenSpec | None = None,
    entity_index: dict[str, int] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    require_torch()
    spec = spec or token_spec()
    if (
        np is None
        or spec.uses_selected_entity_tokens
        or spec.uses_structured_transition_tokens
        or spec.uses_phase2_transition_tokens
        or spec.uses_action_interaction_token
        or len(actions) != len(after_states)
        or len(actions) != len(candidate_features)
    ):
        return collate_transformer_records(
            encode_transformer_candidates_shared_before(
                before_state,
                actions,
                after_states,
                candidate_features=candidate_features,
                spec=spec,
                entity_index=entity_index,
            ),
            device=device,
        )

    all_monsters = monsters(before_state)
    all_hand_cards = hand_cards(before_state)
    all_potions = list(before_state.get("potions") or [])
    if (
        len(all_monsters) > spec.max_monsters
        or len(all_hand_cards) > spec.max_hand
        or len(all_potions) > spec.max_potions
    ):
        return collate_transformer_records(
            encode_transformer_candidates_shared_before(
                before_state,
                actions,
                after_states,
                candidate_features=candidate_features,
                spec=spec,
                entity_index=entity_index,
            ),
            device=device,
        )

    count = len(actions)
    max_len = spec.max_sequence_length
    base_scalars = np.zeros((max_len, spec.scalar_dim), dtype=np.float32)
    base_token_types = np.full((max_len,), TOKEN_TYPES["PAD"], dtype=np.int64)
    base_entity_ids = np.zeros((max_len,), dtype=np.int64)
    base_slot_ids = np.zeros((max_len,), dtype=np.int64)
    base_attention = np.zeros((max_len,), dtype=np.bool_)
    position = 0
    monster_positions: dict[int, int] = {}
    hand_positions: dict[int, int] = {}
    potion_positions: dict[int, int] = {}

    def add_base(token_type: str, scalars: list[float] | None = None, *, entity_id: int = 0, slot_id: int = 0) -> int:
        nonlocal position
        if position >= max_len:
            raise ValueError(f"v3 combat transformer token overflow at {token_type}")
        index = position
        base_scalars[index, :] = np.asarray(_pad(list(scalars or []), spec.scalar_dim), dtype=np.float32)
        base_token_types[index] = TOKEN_TYPES[token_type]
        base_entity_ids[index] = int(entity_id)
        base_slot_ids[index] = int(slot_id)
        base_attention[index] = True
        position += 1
        return index

    add_base("CLS")
    add_base("GLOBAL_BEFORE")
    add_base("PLAYER_CORE", _player_core_scalars(before_state))

    player = dict(combat_state(before_state).get("player") or {})
    powers = _powers_by_id(list(player.get("powers") or []))
    ordered_powers = sorted(powers.items(), key=lambda item: (-abs(float(item[1])), str(item[0])))[: spec.max_player_powers]
    for power_id, amount in ordered_powers:
        add_base("PLAYER_POWER", _power_scalars(power_id, float(amount)), entity_id=_entity_id("power", power_id, entity_index))

    for index, monster in enumerate(all_monsters):
        monster_id = _id_from_payload(monster, "monster_id", "id", "name")
        monster_positions[index] = add_base(
            "MONSTER",
            _monster_scalars(monster, selected=False),
            entity_id=_entity_id("monster", monster_id, entity_index),
            slot_id=index + 1,
        )

    for index, card in enumerate(all_hand_cards):
        card_id = _id_from_payload(card, "card_id", "id", "name")
        hand_positions[index] = add_base(
            "HAND_CARD",
            _card_scalars(card, selected=False),
            entity_id=_entity_id("card", card_id, entity_index),
            slot_id=index + 1,
        )

    for index, potion in enumerate(all_potions):
        potion_id = _id_from_payload(potion, "potion_id", "id", "name")
        potion_positions[index] = add_base(
            "POTION",
            _potion_scalars(potion, selected=False),
            entity_id=_entity_id("potion", potion_id, entity_index),
            slot_id=index + 1,
        )

    for relic in list(before_state.get("relics") or [])[: spec.max_relics]:
        relic_id = _id_from_payload(relic, "relic_id", "id", "name")
        add_base("RELIC", _relic_scalars(relic), entity_id=_entity_id("relic", relic_id, entity_index))

    for zone_name in ZONE_NAMES:
        add_base("ZONE_SUMMARY", _zone_summary(_zone_cards(before_state, zone_name)), entity_id=_extra_entity_id(f"zone:{zone_name}", entity_index))

    action_position = position
    token_scalar_features = np.repeat(base_scalars[None, :, :], count, axis=0)
    token_type_ids = np.repeat(base_token_types[None, :], count, axis=0)
    entity_ids = np.repeat(base_entity_ids[None, :], count, axis=0)
    slot_ids = np.repeat(base_slot_ids[None, :], count, axis=0)
    attention_mask = np.repeat(base_attention[None, :], count, axis=0)
    for offset, token_type in enumerate(("ACTION", "AFTER_SUMMARY", "DELTA", "LEGACY")):
        token_type_ids[:, action_position + offset] = TOKEN_TYPES[token_type]
        attention_mask[:, action_position + offset] = True

    for row, action in enumerate(actions):
        target_index = _selected_index(action, "target_index")
        if target_index is not None and target_index in monster_positions:
            token_scalar_features[row, monster_positions[target_index], 10] = 1.0
        selected_card_index = _selected_index(action, "card_index", "source_index")
        if selected_card_index is not None and selected_card_index in hand_positions:
            token_scalar_features[row, hand_positions[selected_card_index], 12] = 1.0
        selected_potion_index = _selected_index(action, "potion_index")
        if selected_potion_index is not None and selected_potion_index in potion_positions:
            token_scalar_features[row, potion_positions[selected_potion_index], 4] = 1.0
        token_scalar_features[row, action_position, :] = np.asarray(
            _action_scalars(before_state, action, spec.scalar_dim),
            dtype=np.float32,
        )
        entity_ids[row, action_position] = _action_entity_id(before_state, action, entity_index)

    return {
        "token_scalar_features": torch.as_tensor(token_scalar_features, dtype=torch.float32, device=device),
        "token_type_ids": torch.as_tensor(token_type_ids, dtype=torch.long, device=device),
        "entity_ids": torch.as_tensor(entity_ids, dtype=torch.long, device=device),
        "slot_ids": torch.as_tensor(slot_ids, dtype=torch.long, device=device),
        "attention_mask": torch.as_tensor(attention_mask, dtype=torch.bool, device=device),
        "features": torch.as_tensor(np.asarray(candidate_features, dtype=np.float32), dtype=torch.float32, device=device),
        "candidate_counts": torch.as_tensor(np.asarray([count], dtype=np.int64), dtype=torch.long, device=device),
    }


def encode_root_transformer_actions(
    before_state: dict[str, Any],
    actions: list[dict[str, Any]],
    after_states: list[dict[str, Any]] | None = None,
    *,
    candidate_features: list[list[float]] | None = None,
    spec: V3CombatRootTransformerTokenSpec | None = None,
    entity_index: dict[str, int] | None = None,
    trim_padding: bool = False,
) -> dict[str, Any]:
    spec = spec or root_token_spec()
    if after_states is not None and len(actions) != len(after_states):
        raise ValueError(f"root transformer action/after mismatch: {len(actions)} != {len(after_states)}")
    if candidate_features is None and after_states is None:
        raise ValueError("root transformer needs after_states when candidate_features are not provided")
    if candidate_features is not None and len(actions) != len(candidate_features):
        raise ValueError(f"root transformer action/features mismatch: {len(actions)} != {len(candidate_features)}")
    if len(actions) > spec.max_actions:
        raise ValueError(f"root transformer action overflow: {len(actions)} > max_actions={spec.max_actions}")
    max_len = spec.max_sequence_length
    if trim_padding:
        token_scalar_features: list[list[float]] = []
        token_type_ids: list[int] = []
        entity_ids: list[int] = []
        slot_ids: list[int] = []
        attention_mask: list[bool] = []
    else:
        token_scalar_features = [[0.0] * spec.scalar_dim for _ in range(max_len)]
        token_type_ids = [TOKEN_TYPES["PAD"]] * max_len
        entity_ids = [0] * max_len
        slot_ids = [0] * max_len
        attention_mask = [False] * max_len
    action_token_positions = [0] * spec.max_actions
    after_token_positions = [0] * spec.max_actions
    delta_token_positions = [0] * spec.max_actions
    legacy_token_positions = [0] * spec.max_actions
    candidate_mask = [False] * spec.max_actions
    position = 0

    def add_token(token_type: str, scalars: list[float] | None = None, *, entity_id: int = 0, slot_id: int = 0) -> int:
        nonlocal position
        if position >= max_len:
            raise ValueError(f"v3 root combat transformer token overflow at {token_type}")
        index = position
        if trim_padding:
            token_scalar_features.append(_pad(list(scalars or []), spec.scalar_dim))
            token_type_ids.append(TOKEN_TYPES[token_type])
            entity_ids.append(int(entity_id))
            slot_ids.append(int(slot_id))
            attention_mask.append(True)
        else:
            token_scalar_features[index] = _pad(list(scalars or []), spec.scalar_dim)
            token_type_ids[index] = TOKEN_TYPES[token_type]
            entity_ids[index] = int(entity_id)
            slot_ids[index] = int(slot_id)
            attention_mask[index] = True
        position += 1
        return index

    add_token("CLS")
    add_token("GLOBAL_BEFORE")
    add_token("PLAYER_CORE", _player_core_scalars(before_state))

    player = dict(combat_state(before_state).get("player") or {})
    powers = _powers_by_id(list(player.get("powers") or []))
    ordered_powers = sorted(powers.items(), key=lambda item: (-abs(float(item[1])), str(item[0])))[: spec.max_player_powers]
    for power_id, amount in ordered_powers:
        add_token("PLAYER_POWER", _power_scalars(power_id, float(amount)), entity_id=_entity_id("power", power_id, entity_index))

    all_monsters = monsters(before_state)
    for index in _bounded_indices(len(all_monsters), spec.max_monsters, None):
        monster = all_monsters[index]
        monster_id = _id_from_payload(monster, "monster_id", "id", "name")
        add_token(
            "MONSTER",
            _monster_scalars(monster, selected=False),
            entity_id=_entity_id("monster", monster_id, entity_index),
            slot_id=index + 1,
        )

    all_hand_cards = hand_cards(before_state)
    for index in _bounded_indices(len(all_hand_cards), spec.max_hand, None):
        card = all_hand_cards[index]
        card_id = _id_from_payload(card, "card_id", "id", "name")
        add_token(
            "HAND_CARD",
            _card_scalars(card, selected=False),
            entity_id=_entity_id("card", card_id, entity_index),
            slot_id=index + 1,
        )

    all_potions = list(before_state.get("potions") or [])
    for index in _bounded_indices(len(all_potions), spec.max_potions, None):
        potion = all_potions[index]
        potion_id = _id_from_payload(potion, "potion_id", "id", "name")
        add_token(
            "POTION",
            _potion_scalars(potion, selected=False),
            entity_id=_entity_id("potion", potion_id, entity_index),
            slot_id=index + 1,
        )

    for relic in list(before_state.get("relics") or [])[: spec.max_relics]:
        relic_id = _id_from_payload(relic, "relic_id", "id", "name")
        add_token("RELIC", _relic_scalars(relic), entity_id=_entity_id("relic", relic_id, entity_index))

    for zone_name in ZONE_NAMES:
        add_token("ZONE_SUMMARY", _zone_summary(_zone_cards(before_state, zone_name)), entity_id=_extra_entity_id(f"zone:{zone_name}", entity_index))

    feature_rows: list[list[float]] = []
    use_selected_entity_tokens = spec.uses_selected_entity_tokens
    use_legacy_token = spec.uses_legacy_token
    for action_index, action in enumerate(actions):
        features = (
            list(candidate_features[action_index])
            if candidate_features is not None
            else encode_candidate(before_state, action, after_states[action_index])
        )
        slot_id = action_index + 1
        action_token_positions[action_index] = add_token(
            "ACTION",
            _action_scalars(before_state, action, spec.scalar_dim),
            entity_id=_action_entity_id(before_state, action, entity_index),
            slot_id=slot_id,
        )
        if use_selected_entity_tokens:
            selected_card = _selected_card(before_state, action)
            selected_target = _selected_target(before_state, action)
            selected_potion = _selected_potion(before_state, action)
            add_token(
                "SELECTED_CARD",
                _card_scalars(selected_card, selected=True) if selected_card else None,
                entity_id=_entity_id("card", _id_from_payload(selected_card, "card_id", "id", "name"), entity_index)
                if selected_card
                else 0,
                slot_id=slot_id,
            )
            add_token(
                "SELECTED_TARGET",
                _monster_scalars(selected_target, selected=True) if selected_target else None,
                entity_id=_entity_id("monster", _id_from_payload(selected_target, "monster_id", "id", "name"), entity_index)
                if selected_target
                else 0,
                slot_id=slot_id,
            )
            add_token(
                "SELECTED_POTION",
                _potion_scalars(selected_potion, selected=True) if selected_potion else None,
                entity_id=_entity_id("potion", _id_from_payload(selected_potion, "potion_id", "id", "name"), entity_index)
                if selected_potion
                else 0,
                slot_id=slot_id,
            )
        after_token_positions[action_index] = add_token("AFTER_SUMMARY", slot_id=slot_id)
        delta_token_positions[action_index] = add_token("DELTA", slot_id=slot_id)
        if use_legacy_token:
            legacy_token_positions[action_index] = add_token("LEGACY", slot_id=slot_id)
        candidate_mask[action_index] = True
        feature_rows.append(features)

    return {
        "token_scalar_features": token_scalar_features,
        "token_type_ids": token_type_ids,
        "entity_ids": entity_ids,
        "slot_ids": slot_ids,
        "attention_mask": attention_mask,
        "before_summary": encode_state_summary(before_state),
        "action_token_positions": action_token_positions,
        "after_token_positions": after_token_positions,
        "delta_token_positions": delta_token_positions,
        "legacy_token_positions": legacy_token_positions,
        "candidate_mask": candidate_mask,
        "candidate_features": feature_rows,
    }


def _trim_batch_sequence(batch: dict[str, Any]) -> dict[str, Any]:
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return batch
    active = attention_mask.any(dim=0)
    if not bool(active.any().item()):
        return batch
    max_len = int(active.nonzero(as_tuple=False).flatten()[-1].item()) + 1
    for key in ("token_scalar_features", "token_type_ids", "entity_ids", "slot_ids", "attention_mask"):
        if key in batch:
            batch[key] = batch[key][:, :max_len]
    return batch


def _tensor_from_nested(values: Any, *, dtype: Any, device: str, numpy_dtype: Any | None = None) -> Any:
    if np is not None and numpy_dtype is not None:
        return torch.as_tensor(np.asarray(values, dtype=numpy_dtype), dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, device=device)


def collate_root_transformer_records(
    records: list[dict[str, Any]],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    require_torch()
    if not records:
        raise ValueError("cannot collate empty v3 root combat transformer records")
    candidate_counts = [len(record["candidate_features"]) for record in records]
    features = [features for record in records for features in record["candidate_features"]]
    sample_ids: list[int] = []
    for sample_id, count in enumerate(candidate_counts):
        sample_ids.extend([sample_id] * int(count))
    batch = {
        "token_scalar_features": torch.tensor([record["token_scalar_features"] for record in records], dtype=torch.float32, device=device),
        "token_type_ids": torch.tensor([record["token_type_ids"] for record in records], dtype=torch.long, device=device),
        "entity_ids": torch.tensor([record["entity_ids"] for record in records], dtype=torch.long, device=device),
        "slot_ids": torch.tensor([record["slot_ids"] for record in records], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([record["attention_mask"] for record in records], dtype=torch.bool, device=device),
        "before_summary": torch.tensor([record["before_summary"] for record in records], dtype=torch.float32, device=device),
        "action_token_positions": torch.tensor([record["action_token_positions"] for record in records], dtype=torch.long, device=device),
        "after_token_positions": torch.tensor([record["after_token_positions"] for record in records], dtype=torch.long, device=device),
        "delta_token_positions": torch.tensor([record["delta_token_positions"] for record in records], dtype=torch.long, device=device),
        "legacy_token_positions": torch.tensor([record["legacy_token_positions"] for record in records], dtype=torch.long, device=device),
        "candidate_mask": torch.tensor([record["candidate_mask"] for record in records], dtype=torch.bool, device=device),
        "features": torch.tensor(features, dtype=torch.float32, device=device),
        "sample_ids": torch.tensor(sample_ids, dtype=torch.long, device=device),
        "candidate_counts": torch.tensor(candidate_counts, dtype=torch.long, device=device),
        "root_count": len(records),
    }
    return _trim_batch_sequence(batch)


def collate_transformer_records(
    records: list[dict[str, Any]],
    *,
    device: str = "cpu",
    candidate_counts: list[int] | None = None,
) -> dict[str, Any]:
    require_torch()
    if not records:
        raise ValueError("cannot collate empty v3 combat transformer records")
    if candidate_counts is None:
        candidate_counts = [len(records)]
    batch = {
        "token_scalar_features": _tensor_from_nested(
            [record["token_scalar_features"] for record in records],
            dtype=torch.float32,
            device=device,
            numpy_dtype=np.float32 if np is not None else None,
        ),
        "token_type_ids": _tensor_from_nested(
            [record["token_type_ids"] for record in records],
            dtype=torch.long,
            device=device,
            numpy_dtype=np.int64 if np is not None else None,
        ),
        "entity_ids": _tensor_from_nested(
            [record["entity_ids"] for record in records],
            dtype=torch.long,
            device=device,
            numpy_dtype=np.int64 if np is not None else None,
        ),
        "slot_ids": _tensor_from_nested(
            [record["slot_ids"] for record in records],
            dtype=torch.long,
            device=device,
            numpy_dtype=np.int64 if np is not None else None,
        ),
        "attention_mask": _tensor_from_nested(
            [record["attention_mask"] for record in records],
            dtype=torch.bool,
            device=device,
            numpy_dtype=np.bool_ if np is not None else None,
        ),
        "features": _tensor_from_nested(
            [record["candidate_features"] for record in records],
            dtype=torch.float32,
            device=device,
            numpy_dtype=np.float32 if np is not None else None,
        ),
    }
    batch["candidate_counts"] = _tensor_from_nested(
        candidate_counts,
        dtype=torch.long,
        device=device,
        numpy_dtype=np.int64 if np is not None else None,
    )
    return batch


def collate_transformer_labeled_roots(roots: list[Any], *, device: str = "cpu") -> dict[str, Any]:
    require_torch()
    records: list[dict[str, Any]] = []
    teacher_q: list[float] = []
    reward_components: list[list[float]] = []
    sample_ids: list[int] = []
    chosen: list[bool] = []
    counts: list[int] = []
    for sample_id, labeled in enumerate(roots):
        before_state = labeled.root.visible_before
        root_count = 0
        for candidate in labeled.candidates:
            records.append(
                encode_transformer_candidate(
                    before_state,
                    candidate.action,
                    candidate.visible_after,
                    candidate_features=candidate.candidate_features,
                )
            )
            teacher_q.append(float(candidate.teacher_q))
            reward_components.append(reward_component_vector(getattr(candidate, "reward_components", None)))
            sample_ids.append(sample_id)
            chosen.append(bool(candidate.is_chosen))
            root_count += 1
        counts.append(root_count)
    batch = collate_transformer_records(records, device=device, candidate_counts=counts)
    batch.update(
        {
            "teacher_q": torch.tensor(teacher_q, dtype=torch.float32, device=device),
            "reward_components": torch.tensor(reward_components, dtype=torch.float32, device=device),
            "sample_ids": torch.tensor(sample_ids, dtype=torch.long, device=device),
            "chosen": torch.tensor(chosen, dtype=torch.bool, device=device),
            "candidate_counts": torch.tensor(counts, dtype=torch.long, device=device),
            "root_count": len(roots),
        }
    )
    return batch


class V3CombatTransformerCandidateScorer(nn.Module):
    model_kind = "transformer"

    def __init__(
        self,
        *,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 768,
        dropout: float = 0.05,
        scalar_dim: int = TRANSFORMER_SCALAR_DIM,
        token_type_vocab_size: int = len(TOKEN_TYPES),
        entity_vocab_size: int | None = None,
        slot_vocab_size: int | None = None,
        max_sequence_length: int | None = None,
        feature_dim: int | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        delta_dim: int | None = None,
        action_set_layers: int = 1,
        action_set_ffn_dim: int | None = None,
        candidate_head_variant: str = CANDIDATE_HEAD_VARIANT_BASE,
        legacy_dropout: float = 0.0,
        disabled_token_types: list[str] | tuple[str, ...] | str | None = None,
        semantic_delta_scale: float = 1.0,
        semantic_delta_clip: float = 4.0,
        auxiliary_reward_heads: bool = False,
        auxiliary_reward_dim: int = REWARD_COMPONENT_DIM,
        potion_residual_head_enabled: bool = False,
        potion_residual_clip: float = 2.0,
        card_residual_head_enabled: bool = False,
        card_residual_clip: float = 2.0,
    ) -> None:
        super().__init__()
        feature_schema = schema()
        spec = token_spec()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)
        self.scalar_dim = int(scalar_dim)
        self.token_type_vocab_size = int(token_type_vocab_size)
        self.entity_vocab_size = int(entity_vocab_size or len(transformer_entity_ids()))
        self.slot_vocab_size = int(slot_vocab_size or spec.slot_vocab_size)
        self.max_sequence_length = int(max_sequence_length or spec.max_sequence_length)
        self.feature_dim = int(feature_dim or feature_schema.candidate_dim)
        self.state_dim = int(state_dim or feature_schema.state_dim)
        self.action_dim = int(action_dim or feature_schema.action_dim)
        self.delta_dim = int(delta_dim or feature_schema.delta_dim)
        self.action_set_layers = int(max(0, action_set_layers))
        self.action_set_ffn_dim = int(action_set_ffn_dim or self.ffn_dim)
        self.candidate_head_variant = normalize_candidate_head_variant(candidate_head_variant)
        self.legacy_dropout = float(max(0.0, min(1.0, legacy_dropout)))
        self.disabled_token_type_names = self._normalize_disabled_token_types(disabled_token_types)
        self.disabled_token_type_ids = tuple(
            int(TOKEN_TYPES[name]) for name in self.disabled_token_type_names if name in TOKEN_TYPES
        )
        self.semantic_delta_scale = float(semantic_delta_scale)
        self.semantic_delta_clip = float(max(0.0, semantic_delta_clip))
        self.auxiliary_reward_heads = bool(auxiliary_reward_heads)
        self.auxiliary_reward_dim = int(max(1, auxiliary_reward_dim))
        self.potion_residual_head_enabled = bool(potion_residual_head_enabled)
        self.potion_residual_clip = float(max(0.0, potion_residual_clip))
        self.card_residual_head_enabled = bool(card_residual_head_enabled)
        self.card_residual_clip = float(max(0.0, card_residual_clip))
        self.last_aux_outputs = None
        self.potion_residual_head = None
        self.card_residual_head = None

        self.scalar_projection = nn.Linear(self.scalar_dim, self.d_model)
        self.before_summary_projection = nn.Linear(self.state_dim, self.d_model)
        self.after_summary_projection = nn.Linear(self.state_dim, self.d_model)
        self.delta_projection = nn.Linear(self.delta_dim, self.d_model)
        self.legacy_projection = nn.Linear(self.feature_dim, self.d_model)
        self.action_summary_projection = (
            nn.Linear(self.action_dim, self.d_model)
            if self.candidate_head_variant
            in {CANDIDATE_HEAD_VARIANT_ACTION200, CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200}
            else None
        )
        self.legacy_dropout_layer = nn.Dropout(self.legacy_dropout)
        self.token_type_embedding = nn.Embedding(self.token_type_vocab_size, self.d_model)
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, self.d_model)
        self.slot_embedding = nn.Embedding(self.slot_vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, enable_nested_tensor=False)
        self.action_selected_attention = None
        self.action_delta_attention = None
        self.action_selected_attention_norm = None
        self.action_delta_attention_norm = None
        if self.action_set_layers > 0:
            self.action_set_input = nn.Sequential(
                nn.LayerNorm(self.d_model * 4),
                nn.Linear(self.d_model * 4, self.d_model),
                nn.GELU(),
            )
            action_set_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.action_set_ffn_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_set_encoder = nn.TransformerEncoder(
                action_set_layer,
                num_layers=self.action_set_layers,
                enable_nested_tensor=False,
            )
            self.action_set_context_projection = nn.Linear(self.d_model, self.d_model * 4)
        else:
            self.action_set_input = None
            self.action_set_encoder = None
            self.action_set_context_projection = None
        if self._is_dual_gate_variant():
            semantic_input_dim = self._dual_semantic_input_dim()
            self.output_head = None
            self.semantic_delta_head = None
            self.legacy_baseline_head = None
            if self._is_action_binding_variant():
                self.action_selected_attention = nn.MultiheadAttention(
                    self.d_model,
                    self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                self.action_delta_attention = nn.MultiheadAttention(
                    self.d_model,
                    self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                self.action_selected_attention_norm = nn.LayerNorm(self.d_model)
                self.action_delta_attention_norm = nn.LayerNorm(self.d_model)
            self.semantic_head = nn.Sequential(
                nn.LayerNorm(semantic_input_dim),
                nn.Linear(semantic_input_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            )
            self.legacy_residual_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            )
            self.legacy_gate_head = nn.Sequential(
                nn.LayerNorm(semantic_input_dim),
                nn.Linear(semantic_input_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            )
        elif self._is_legacy_delta_variant():
            semantic_input_dim = self._transition_semantic_input_dim()
            self.output_head = None
            self.semantic_head = None
            self.legacy_residual_head = None
            self.legacy_gate_head = None
            self.semantic_delta_head = nn.Sequential(
                nn.LayerNorm(semantic_input_dim),
                nn.Linear(semantic_input_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            )
            self.legacy_baseline_head = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, 1),
            )
        else:
            self.semantic_head = None
            self.legacy_residual_head = None
            self.legacy_gate_head = None
            self.semantic_delta_head = None
            self.legacy_baseline_head = None
            self.output_head = nn.Sequential(
                nn.LayerNorm(self._head_input_dim()),
                nn.Linear(self._head_input_dim(), self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, 1),
            )
        if self.potion_residual_head_enabled:
            self.enable_potion_residual_head(clip=self.potion_residual_clip)
        if self.card_residual_head_enabled:
            self.enable_card_residual_head(clip=self.card_residual_clip)
        self.aux_reward_head = self._build_aux_reward_head() if self.auxiliary_reward_heads else None

    @staticmethod
    def _normalize_disabled_token_types(value: list[str] | tuple[str, ...] | str | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values = [part.strip() for part in value.replace(";", ",").split(",")]
        else:
            raw_values = [str(part).strip() for part in value]
        names: list[str] = []
        for raw in raw_values:
            if not raw:
                continue
            name = raw.upper().replace("-", "_")
            if name not in TOKEN_TYPES:
                raise ValueError(f"unknown disabled token type: {raw!r}")
            if name not in names:
                names.append(name)
        return tuple(names)

    def _config(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "scalar_dim": self.scalar_dim,
            "token_type_vocab_size": self.token_type_vocab_size,
            "entity_vocab_size": self.entity_vocab_size,
            "slot_vocab_size": self.slot_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "feature_dim": self.feature_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "delta_dim": self.delta_dim,
            "action_set_layers": self.action_set_layers,
            "action_set_ffn_dim": self.action_set_ffn_dim,
            "candidate_head_variant": self.candidate_head_variant,
            "legacy_dropout": self.legacy_dropout,
            "disabled_token_types": list(self.disabled_token_type_names),
            "semantic_delta_scale": self.semantic_delta_scale,
            "semantic_delta_clip": self.semantic_delta_clip,
            "auxiliary_reward_heads": self.auxiliary_reward_heads,
            "auxiliary_reward_dim": self.auxiliary_reward_dim,
            "potion_residual_head_enabled": self.potion_residual_head_enabled,
            "potion_residual_clip": self.potion_residual_clip,
            "card_residual_head_enabled": self.card_residual_head_enabled,
            "card_residual_clip": self.card_residual_clip,
        }

    def _build_aux_reward_head(self):
        return nn.Sequential(
            nn.LayerNorm(self._transition_semantic_input_dim()),
            nn.Linear(self._transition_semantic_input_dim(), self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.auxiliary_reward_dim),
        )

    def enable_auxiliary_reward_heads(self, *, output_dim: int = REWARD_COMPONENT_DIM) -> None:
        self.auxiliary_reward_dim = int(max(1, output_dim))
        self.auxiliary_reward_heads = True
        head = self._build_aux_reward_head()
        try:
            parameter = next(self.parameters())
            head.to(device=parameter.device, dtype=parameter.dtype)
        except StopIteration:
            pass
        self.aux_reward_head = head

    def _build_action_kind_residual_head(self):
        if not self._is_dual_gate_variant():
            raise ValueError("action-kind residual head requires a dual-gate candidate head variant")
        head = nn.Sequential(
            nn.LayerNorm(self._dual_semantic_input_dim()),
            nn.Linear(self._dual_semantic_input_dim(), self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        final_linear = head[-1]
        if hasattr(final_linear, "weight"):
            nn.init.zeros_(final_linear.weight)
        if hasattr(final_linear, "bias") and final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)
        return head

    def enable_potion_residual_head(self, *, clip: float | None = None) -> None:
        if clip is not None:
            self.potion_residual_clip = float(max(0.0, clip))
        self.potion_residual_head_enabled = True
        head = self._build_action_kind_residual_head()
        try:
            parameter = next(self.parameters())
            head.to(device=parameter.device, dtype=parameter.dtype)
        except StopIteration:
            pass
        self.potion_residual_head = head

    def enable_card_residual_head(self, *, clip: float | None = None) -> None:
        if clip is not None:
            self.card_residual_clip = float(max(0.0, clip))
        self.card_residual_head_enabled = True
        head = self._build_action_kind_residual_head()
        try:
            parameter = next(self.parameters())
            head.to(device=parameter.device, dtype=parameter.dtype)
        except StopIteration:
            pass
        self.card_residual_head = head

    def _head_input_dim(self) -> int:
        base_dim = self.d_model * 4
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY:
            return self.d_model * 7
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_ACTION200:
            return base_dim + self.action_dim
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_RELATIVE_RANK:
            return base_dim * 4
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_SEMANTIC_ONLY:
            return self.d_model * 3
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION:
            return self._transition_semantic_input_dim()
        return base_dim

    def _is_dual_gate_variant(self) -> bool:
        return self.candidate_head_variant in {
            CANDIDATE_HEAD_VARIANT_DUAL_GATE,
            CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200,
            CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY,
            CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING,
        }

    def _is_action_binding_variant(self) -> bool:
        return self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_DUAL_ACTION_BINDING

    def _is_legacy_delta_variant(self) -> bool:
        return self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_LEGACY_DELTA

    def _transition_semantic_input_dim(self) -> int:
        return self.d_model * 13

    def _dual_semantic_input_dim(self) -> int:
        if self._is_action_binding_variant():
            return self.d_model * 9
        dim = self.d_model * 3
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200:
            dim += self.action_dim
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY:
            dim += self.d_model * 3
        return dim

    def _special_vector(self, encoded: Any, token_type_ids: Any, token_type_name: str, attention_mask: Any | None = None) -> Any:
        token_type = TOKEN_TYPES[token_type_name]
        mask = token_type_ids == int(token_type)
        if attention_mask is not None:
            mask = mask & attention_mask
        if bool(mask.any().item()):
            indices = mask.to(dtype=encoded.dtype).argmax(dim=1)
            vectors = encoded[torch.arange(encoded.shape[0], device=encoded.device), indices]
            has_match = mask.any(dim=1)
            return vectors * has_match.unsqueeze(1).to(dtype=vectors.dtype)
        return encoded.new_zeros((encoded.shape[0], encoded.shape[2]))

    def _mean_vector(self, encoded: Any, token_type_ids: Any, token_type_name: str, attention_mask: Any) -> Any:
        token_type = TOKEN_TYPES[token_type_name]
        mask = (token_type_ids == int(token_type)) & attention_mask
        weights = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denom

    def _selected_vector(
        self,
        encoded: Any,
        token_scalar_features: Any,
        token_type_ids: Any,
        attention_mask: Any,
        token_type_name: str,
        selected_scalar_index: int,
    ) -> Any:
        token_type = TOKEN_TYPES[token_type_name]
        selected = token_scalar_features[:, :, int(selected_scalar_index)] > 0.5
        mask = (token_type_ids == int(token_type)) & selected & attention_mask
        has_match = mask.any(dim=1)
        indices = mask.to(dtype=torch.long).argmax(dim=1)
        vectors = encoded[torch.arange(encoded.shape[0], device=encoded.device), indices]
        return vectors * has_match.unsqueeze(1).to(dtype=vectors.dtype)

    def _selected_entity_vectors(
        self,
        encoded: Any,
        token_scalar_features: Any,
        token_type_ids: Any,
        attention_mask: Any,
    ) -> tuple[Any, Any, Any]:
        selected_card_vector = self._selected_vector(
            encoded,
            token_scalar_features,
            token_type_ids,
            attention_mask,
            "HAND_CARD",
            12,
        )
        selected_target_vector = self._selected_vector(
            encoded,
            token_scalar_features,
            token_type_ids,
            attention_mask,
            "MONSTER",
            10,
        )
        selected_potion_vector = self._selected_vector(
            encoded,
            token_scalar_features,
            token_type_ids,
            attention_mask,
            "POTION",
            4,
        )
        return selected_card_vector, selected_target_vector, selected_potion_vector

    def _token_type_vectors(
        self,
        encoded: Any,
        token_type_ids: Any,
        attention_mask: Any,
        token_type_names: tuple[str, ...],
    ) -> tuple[Any, Any]:
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for name in token_type_names:
            token_type = TOKEN_TYPES.get(name)
            if token_type is not None:
                mask = mask | (token_type_ids == int(token_type))
        mask = mask & attention_mask
        counts = mask.sum(dim=1)
        max_count = max(1, int(counts.max().item()) if int(counts.numel()) else 1)
        vectors = encoded.new_zeros((encoded.shape[0], max_count, encoded.shape[2]))
        key_padding_mask = torch.ones((encoded.shape[0], max_count), dtype=torch.bool, device=encoded.device)
        for batch_index in range(int(encoded.shape[0])):
            indices = mask[batch_index].nonzero(as_tuple=False).flatten()
            if int(indices.numel()) <= 0:
                key_padding_mask[batch_index, 0] = False
                continue
            indices = indices[:max_count]
            count = int(indices.numel())
            vectors[batch_index, :count] = encoded[batch_index, indices]
            key_padding_mask[batch_index, :count] = False
        return vectors, key_padding_mask

    def _action_binding_vector(
        self,
        attention: Any,
        norm: Any,
        encoded: Any,
        token_type_ids: Any,
        attention_mask: Any,
        action_vector: Any,
        token_type_names: tuple[str, ...],
    ) -> Any:
        if attention is None or norm is None:
            return encoded.new_zeros((encoded.shape[0], encoded.shape[2]))
        key_vectors, key_padding_mask = self._token_type_vectors(
            encoded,
            token_type_ids,
            attention_mask,
            token_type_names,
        )
        query = action_vector.unsqueeze(1)
        bound, _weights = attention(
            query,
            key_vectors,
            key_vectors,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return norm(bound.squeeze(1))

    def _dual_semantic_features(
        self,
        encoded: Any,
        token_scalar_features: Any,
        token_type_ids: Any,
        attention_mask: Any,
        cls_vector: Any,
        action_vector: Any,
        delta_vector: Any,
        action_summary: Any,
    ) -> Any:
        if self._is_action_binding_variant():
            bind_selected = self._action_binding_vector(
                self.action_selected_attention,
                self.action_selected_attention_norm,
                encoded,
                token_type_ids,
                attention_mask,
                action_vector,
                ("SELECTED_CARD", "SELECTED_TARGET", "SELECTED_POTION", "CARD_TARGET_INTERACTION"),
            )
            bind_delta = self._action_binding_vector(
                self.action_delta_attention,
                self.action_delta_attention_norm,
                encoded,
                token_type_ids,
                attention_mask,
                action_vector,
                ("PLAYER_DELTA", "MONSTER_DELTA", "ZONE_DELTA", "POTION_SLOT_DELTA"),
            )
            return torch.cat(
                [
                    cls_vector,
                    action_vector,
                    delta_vector,
                    self._special_vector(encoded, token_type_ids, "CARD_TARGET_INTERACTION", attention_mask),
                    bind_selected,
                    bind_delta,
                    self._special_vector(encoded, token_type_ids, "PLAYER_DELTA", attention_mask),
                    self._mean_vector(encoded, token_type_ids, "MONSTER_DELTA", attention_mask),
                    self._mean_vector(encoded, token_type_ids, "POTION_SLOT_DELTA", attention_mask),
                ],
                dim=1,
            )
        parts = [cls_vector, action_vector]
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_DUAL_GATE_SELECTED_ENTITY:
            parts.extend(self._selected_entity_vectors(encoded, token_scalar_features, token_type_ids, attention_mask))
        parts.append(delta_vector)
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_DUAL_GATE_ACTION200:
            parts.append(action_summary)
        return torch.cat(parts, dim=1)

    def _transition_semantic_features(
        self,
        encoded: Any,
        token_scalar_features: Any,
        token_type_ids: Any,
        attention_mask: Any,
        cls_vector: Any,
        action_vector: Any,
        delta_vector: Any,
    ) -> Any:
        selected_card_vector, selected_target_vector, selected_potion_vector = self._selected_entity_vectors(
            encoded,
            token_scalar_features,
            token_type_ids,
            attention_mask,
        )
        selected_card_vector = selected_card_vector + self._special_vector(encoded, token_type_ids, "SELECTED_CARD", attention_mask)
        selected_target_vector = selected_target_vector + self._special_vector(encoded, token_type_ids, "SELECTED_TARGET", attention_mask)
        selected_potion_vector = selected_potion_vector + self._special_vector(encoded, token_type_ids, "SELECTED_POTION", attention_mask)
        return torch.cat(
            [
                cls_vector,
                action_vector,
                delta_vector,
                selected_card_vector,
                selected_target_vector,
                selected_potion_vector,
                self._special_vector(encoded, token_type_ids, "PLAYER_DELTA", attention_mask),
                self._mean_vector(encoded, token_type_ids, "MONSTER_DELTA", attention_mask),
                self._mean_vector(encoded, token_type_ids, "ZONE_DELTA", attention_mask),
                self._mean_vector(encoded, token_type_ids, "POTION_SLOT_DELTA", attention_mask),
                self._special_vector(encoded, token_type_ids, "PLAYER_AFTER", attention_mask),
                self._mean_vector(encoded, token_type_ids, "MONSTER_AFTER", attention_mask),
                self._mean_vector(encoded, token_type_ids, "POWER_DELTA", attention_mask),
            ],
            dim=1,
        )

    def _root_mean_features(self, candidate_features: Any, candidate_counts: Any | None) -> Any:
        if candidate_counts is None:
            return candidate_features.new_zeros(candidate_features.shape)
        counts = candidate_counts.to(device=candidate_features.device, dtype=torch.long)
        counts = counts[counts > 0]
        if int(counts.numel()) <= 0 or int(counts.sum().item()) != int(candidate_features.shape[0]):
            return candidate_features.new_zeros(candidate_features.shape)
        root_indices = torch.repeat_interleave(torch.arange(int(counts.numel()), device=candidate_features.device), counts)
        sums = candidate_features.new_zeros((int(counts.numel()), int(candidate_features.shape[1])))
        sums.index_add_(0, root_indices, candidate_features)
        means = sums / counts.to(dtype=candidate_features.dtype).unsqueeze(1).clamp_min(1.0)
        return means[root_indices]

    def _action_set_context(self, candidate_features: Any, candidate_counts: Any | None) -> Any | None:
        if self.action_set_layers <= 0 or self.action_set_input is None or self.action_set_encoder is None:
            return None
        if candidate_counts is None:
            return None
        counts = candidate_counts.to(device=candidate_features.device, dtype=torch.long)
        counts = counts[counts > 0]
        if int(counts.numel()) <= 0 or int(counts.sum().item()) != int(candidate_features.shape[0]):
            return None
        max_candidates = int(counts.max().item())
        if max_candidates <= 1:
            return candidate_features.new_zeros((candidate_features.shape[0], self.d_model))
        action_vectors = self.action_set_input(candidate_features)
        mask = torch.arange(max_candidates, device=action_vectors.device).unsqueeze(0) < counts.unsqueeze(1)
        padded = action_vectors.new_zeros((int(counts.numel()), max_candidates, self.d_model))
        padded[mask] = action_vectors
        encoded = self.action_set_encoder(padded, src_key_padding_mask=~mask)
        return encoded[mask]

    def forward(self, batch: Any, **kwargs: Any) -> Any:
        require_torch()
        if isinstance(batch, dict):
            token_scalar_features = batch["token_scalar_features"]
            token_type_ids = batch["token_type_ids"]
            entity_ids = batch["entity_ids"]
            slot_ids = batch["slot_ids"]
            attention_mask = batch["attention_mask"]
            legacy_features = batch.get("features")
            if legacy_features is None:
                legacy_features = batch.get("legacy_features")
            candidate_counts = batch.get("candidate_counts")
        else:
            token_scalar_features = batch
            token_type_ids = kwargs["token_type_ids"]
            entity_ids = kwargs["entity_ids"]
            slot_ids = kwargs["slot_ids"]
            attention_mask = kwargs["attention_mask"]
            legacy_features = kwargs["features"] if "features" in kwargs else kwargs["legacy_features"]
            candidate_counts = kwargs.get("candidate_counts")

        parameter = next(self.parameters())
        token_scalar_features = token_scalar_features.to(dtype=parameter.dtype, device=parameter.device)
        token_type_ids = token_type_ids.to(dtype=torch.long, device=parameter.device)
        entity_ids = entity_ids.to(dtype=torch.long, device=parameter.device).clamp(0, self.entity_vocab_size - 1)
        slot_ids = slot_ids.to(dtype=torch.long, device=parameter.device).clamp(0, self.slot_vocab_size - 1)
        attention_mask = attention_mask.to(dtype=torch.bool, device=parameter.device)
        legacy_features = legacy_features.to(dtype=parameter.dtype, device=parameter.device)
        effective_attention_mask = attention_mask
        if self.disabled_token_type_ids:
            effective_attention_mask = attention_mask.clone()
            for token_type_id in self.disabled_token_type_ids:
                effective_attention_mask = effective_attention_mask & (token_type_ids != int(token_type_id))

        batch_size, sequence_length = token_type_ids.shape
        positions = torch.arange(sequence_length, device=parameter.device).unsqueeze(0).expand(batch_size, sequence_length)
        x = (
            self.scalar_projection(token_scalar_features)
            + self.token_type_embedding(token_type_ids.clamp(0, self.token_type_vocab_size - 1))
            + self.entity_embedding(entity_ids)
            + self.slot_embedding(slot_ids)
            + self.position_embedding(positions.clamp(0, self.max_sequence_length - 1))
        )

        before_summary = legacy_features[:, : self.state_dim]
        after_start = self.state_dim + self.action_dim
        after_summary = legacy_features[:, after_start : after_start + self.state_dim]
        delta_features = legacy_features[:, -self.delta_dim :]
        special_additions = {
            "GLOBAL_BEFORE": self.before_summary_projection(before_summary),
            "AFTER_SUMMARY": self.after_summary_projection(after_summary),
            "DELTA": self.delta_projection(delta_features),
            "LEGACY": self.legacy_projection(legacy_features),
        }
        for token_type_name, vector in special_additions.items():
            mask = ((token_type_ids == TOKEN_TYPES[token_type_name]) & effective_attention_mask).unsqueeze(-1).to(dtype=x.dtype)
            x = x + mask * vector.unsqueeze(1)

        encoded = self.encoder(x, src_key_padding_mask=~effective_attention_mask)
        cls_vector = encoded[:, 0]
        action_vector = self._special_vector(encoded, token_type_ids, "ACTION", effective_attention_mask)
        delta_vector = self._special_vector(encoded, token_type_ids, "DELTA", effective_attention_mask)
        legacy_vector = self._special_vector(encoded, token_type_ids, "LEGACY", effective_attention_mask)
        action_summary = legacy_features[:, self.state_dim : self.state_dim + self.action_dim]
        if self.action_summary_projection is not None:
            action_vector = action_vector + self.action_summary_projection(action_summary)
        score_features = torch.cat([cls_vector, action_vector, delta_vector, legacy_vector], dim=1)
        action_set_context = self._action_set_context(score_features, candidate_counts)
        if action_set_context is not None and self.action_set_context_projection is not None:
            context_parts = self.action_set_context_projection(action_set_context).chunk(4, dim=1)
            cls_vector = cls_vector + context_parts[0]
            action_vector = action_vector + context_parts[1]
            delta_vector = delta_vector + context_parts[2]
            legacy_vector = legacy_vector + context_parts[3]
            score_features = torch.cat([cls_vector, action_vector, delta_vector, legacy_vector], dim=1)
        if self._is_dual_gate_variant():
            semantic_features = self._dual_semantic_features(
                encoded,
                token_scalar_features,
                token_type_ids,
                effective_attention_mask,
                cls_vector,
                action_vector,
                delta_vector,
                action_summary,
            )
            legacy_for_head = self.legacy_dropout_layer(legacy_vector)
            semantic_score = self.semantic_head(semantic_features).squeeze(-1)
            legacy_score = self.legacy_residual_head(legacy_for_head).squeeze(-1)
            legacy_gate = torch.sigmoid(self.legacy_gate_head(semantic_features).squeeze(-1))
            if self.aux_reward_head is not None:
                self.last_aux_outputs = self.aux_reward_head(
                    self._transition_semantic_features(
                        encoded,
                        token_scalar_features,
                        token_type_ids,
                        effective_attention_mask,
                        cls_vector,
                        action_vector,
                        delta_vector,
                    )
                )
            else:
                self.last_aux_outputs = None
            score = semantic_score + legacy_gate * legacy_score
            if self.potion_residual_head is not None:
                potion_mask = (action_summary[:, 2] > 0.5).to(dtype=score.dtype)
                raw_residual = self.potion_residual_head(semantic_features).squeeze(-1)
                residual = torch.tanh(raw_residual) * float(self.potion_residual_clip)
                score = score + potion_mask * residual
            if self.card_residual_head is not None:
                card_mask = (action_summary[:, 1] > 0.5).to(dtype=score.dtype)
                raw_residual = self.card_residual_head(semantic_features).squeeze(-1)
                residual = torch.tanh(raw_residual) * float(self.card_residual_clip)
                score = score + card_mask * residual
            return score
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_SEMANTIC_ONLY:
            score_features = torch.cat([cls_vector, action_vector, delta_vector], dim=1)
            if self.aux_reward_head is not None:
                self.last_aux_outputs = self.aux_reward_head(
                    self._transition_semantic_features(
                        encoded,
                        token_scalar_features,
                        token_type_ids,
                        effective_attention_mask,
                        cls_vector,
                        action_vector,
                        delta_vector,
                    )
                )
            else:
                self.last_aux_outputs = None
            return self.output_head(score_features).squeeze(-1)
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_SEMANTIC_TRANSITION:
            transition_features = self._transition_semantic_features(
                encoded,
                token_scalar_features,
                token_type_ids,
                effective_attention_mask,
                cls_vector,
                action_vector,
                delta_vector,
            )
            self.last_aux_outputs = self.aux_reward_head(transition_features) if self.aux_reward_head is not None else None
            return self.output_head(transition_features).squeeze(-1)
        if self._is_legacy_delta_variant():
            transition_features = self._transition_semantic_features(
                encoded,
                token_scalar_features,
                token_type_ids,
                effective_attention_mask,
                cls_vector,
                action_vector,
                delta_vector,
            )
            self.last_aux_outputs = self.aux_reward_head(transition_features) if self.aux_reward_head is not None else None
            delta = self.semantic_delta_head(transition_features).squeeze(-1)
            if self.semantic_delta_clip > 0.0:
                clip = torch.tensor(float(self.semantic_delta_clip), dtype=delta.dtype, device=delta.device)
                delta = clip * torch.tanh(delta / clip)
            legacy_score = self.legacy_baseline_head(legacy_features).squeeze(-1)
            return legacy_score + float(self.semantic_delta_scale) * delta
        if self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_SELECTED_ENTITY:
            selected_card_vector, selected_target_vector, selected_potion_vector = self._selected_entity_vectors(
                encoded,
                token_scalar_features,
                token_type_ids,
                effective_attention_mask,
            )
            score_features = torch.cat(
                [
                    cls_vector,
                    action_vector,
                    selected_card_vector,
                    selected_target_vector,
                    selected_potion_vector,
                    delta_vector,
                    legacy_vector,
                ],
                dim=1,
            )
        elif self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_ACTION200:
            score_features = torch.cat([score_features, action_summary], dim=1)
        elif self.candidate_head_variant == CANDIDATE_HEAD_VARIANT_RELATIVE_RANK:
            root_mean = self._root_mean_features(score_features, candidate_counts)
            score_features = torch.cat(
                [score_features, score_features - root_mean, score_features * root_mean, root_mean],
                dim=1,
            )
        if self.aux_reward_head is not None:
            self.last_aux_outputs = self.aux_reward_head(
                self._transition_semantic_features(
                    encoded,
                    token_scalar_features,
                    token_type_ids,
                    effective_attention_mask,
                    cls_vector,
                    action_vector,
                    delta_vector,
                )
            )
        else:
            self.last_aux_outputs = None
        return self.output_head(score_features).squeeze(-1)


class V3CombatRootActionSetTransformerScorer(nn.Module):
    model_kind = "transformer"
    transformer_architecture = "root_action_set"
    expects_root_batch = True

    def __init__(
        self,
        *,
        d_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 768,
        dropout: float = 0.05,
        scalar_dim: int = TRANSFORMER_SCALAR_DIM,
        token_type_vocab_size: int = len(TOKEN_TYPES),
        entity_vocab_size: int | None = None,
        slot_vocab_size: int | None = None,
        max_sequence_length: int | None = None,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        feature_dim: int | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        delta_dim: int | None = None,
        token_schema_version: str | None = None,
        root_head_variant: str = ROOT_HEAD_VARIANT_BASE,
    ) -> None:
        super().__init__()
        feature_schema = schema()
        default_spec = root_token_spec()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)
        self.scalar_dim = int(scalar_dim)
        self.token_type_vocab_size = int(token_type_vocab_size)
        self.entity_vocab_size = int(entity_vocab_size or len(transformer_entity_ids()))
        self.max_actions = int(max_actions or DEFAULT_MAX_ACTIONS)
        schema_version = str(token_schema_version or infer_root_token_schema_version(
            max_sequence_length=max_sequence_length,
            max_actions=self.max_actions,
        ))
        model_spec = V3CombatRootTransformerTokenSpec(
            version=schema_version,
            scalar_dim=self.scalar_dim,
            max_hand=default_spec.max_hand,
            max_potions=default_spec.max_potions,
            max_monsters=default_spec.max_monsters,
            max_player_powers=default_spec.max_player_powers,
            max_relics=default_spec.max_relics,
            max_actions=self.max_actions,
        )
        self.token_schema_version = model_spec.version
        self.uses_legacy_token = bool(model_spec.uses_legacy_token)
        self.slot_vocab_size = int(slot_vocab_size or max(model_spec.slot_vocab_size, self.max_actions + 2))
        self.max_sequence_length = int(max_sequence_length or model_spec.max_sequence_length)
        self.feature_dim = int(feature_dim or feature_schema.candidate_dim)
        self.state_dim = int(state_dim or feature_schema.state_dim)
        self.action_dim = int(action_dim or feature_schema.action_dim)
        self.delta_dim = int(delta_dim or feature_schema.delta_dim)
        self.root_head_variant = normalize_root_head_variant(root_head_variant)
        self.token_schema = asdict(model_spec) | {
            "action_segment_width": model_spec.action_segment_width,
            "max_sequence_length": self.max_sequence_length,
            "uses_legacy_token": self.uses_legacy_token,
            "root_head_variant": self.root_head_variant,
        }

        self.scalar_projection = nn.Linear(self.scalar_dim, self.d_model)
        self.before_summary_projection = nn.Linear(self.state_dim, self.d_model)
        self.after_summary_projection = nn.Linear(self.state_dim, self.d_model)
        self.delta_projection = nn.Linear(self.delta_dim, self.d_model)
        self.legacy_projection = nn.Linear(self.feature_dim, self.d_model) if self.uses_legacy_token else None
        self.token_type_embedding = nn.Embedding(self.token_type_vocab_size, self.d_model)
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, self.d_model)
        self.slot_embedding = nn.Embedding(self.slot_vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, enable_nested_tensor=False)
        head_input_dim = self._head_input_dim()
        self.output_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

    def _config(self) -> dict[str, Any]:
        return {
            "architecture": self.transformer_architecture,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "scalar_dim": self.scalar_dim,
            "token_type_vocab_size": self.token_type_vocab_size,
            "entity_vocab_size": self.entity_vocab_size,
            "slot_vocab_size": self.slot_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "max_actions": self.max_actions,
            "feature_dim": self.feature_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "delta_dim": self.delta_dim,
            "token_schema_version": self.token_schema_version,
            "root_head_variant": self.root_head_variant,
        }

    def _head_input_dim(self) -> int:
        if self.uses_legacy_token:
            return self.d_model * 4
        if self.root_head_variant == ROOT_HEAD_VARIANT_SELECTED:
            return self.d_model * 7
        if self.root_head_variant == ROOT_HEAD_VARIANT_ACTION200:
            return self.d_model * 4 + self.action_dim
        if self.root_head_variant == ROOT_HEAD_VARIANT_SELECTED_SHARED_POOL:
            return self.d_model * 10
        return self.d_model * 4

    @staticmethod
    def _flat_candidate_positions(positions: Any, candidate_mask: Any) -> tuple[Any, Any]:
        row_indices = torch.arange(positions.shape[0], device=positions.device).unsqueeze(1).expand_as(positions)
        return row_indices[candidate_mask], positions[candidate_mask]

    def _position_addition(self, x: Any, positions: Any, candidate_mask: Any, values: Any) -> Any:
        root_indices, token_positions = self._flat_candidate_positions(positions, candidate_mask)
        if int(token_positions.numel()) <= 0:
            return x
        addition = torch.zeros_like(x)
        flat_indices = root_indices * x.shape[1] + token_positions
        addition.reshape(-1, x.shape[-1]).index_add_(0, flat_indices, values.to(dtype=x.dtype))
        return x + addition

    def _candidate_vectors(self, encoded: Any, positions: Any, candidate_mask: Any) -> Any:
        root_indices, token_positions = self._flat_candidate_positions(positions, candidate_mask)
        return encoded[root_indices, token_positions]

    @staticmethod
    def _vectors_at(encoded: Any, root_indices: Any, token_positions: Any) -> Any:
        return encoded[root_indices, token_positions]

    def _shared_slot_vectors(
        self,
        encoded: Any,
        token_type_ids: Any,
        slot_ids: Any,
        attention_mask: Any,
        root_indices: Any,
        desired_slots: Any,
        token_type_name: str,
    ) -> Any:
        if int(root_indices.numel()) <= 0:
            return encoded.new_zeros((0, self.d_model))
        root_token_types = token_type_ids[root_indices]
        root_slots = slot_ids[root_indices]
        root_attention = attention_mask[root_indices]
        wanted = desired_slots.to(dtype=root_slots.dtype, device=root_slots.device).unsqueeze(1)
        mask = (
            (root_token_types == int(TOKEN_TYPES[token_type_name]))
            & (root_slots == wanted)
            & (wanted > 0)
            & root_attention
        )
        has_match = mask.any(dim=1)
        positions = mask.to(dtype=torch.long).argmax(dim=1)
        vectors = encoded[root_indices, positions]
        return vectors * has_match.unsqueeze(1).to(dtype=vectors.dtype)

    def _selected_shared_vectors(
        self,
        encoded: Any,
        token_scalar_features: Any,
        token_type_ids: Any,
        slot_ids: Any,
        attention_mask: Any,
        root_indices: Any,
        action_positions: Any,
    ) -> tuple[Any, Any, Any]:
        action_scalars = token_scalar_features[root_indices, action_positions]
        is_card = action_scalars[:, 1] > 0.5
        is_potion = action_scalars[:, 2] > 0.5
        requires_target = action_scalars[:, 5] > 0.5
        source_index = torch.round(action_scalars[:, -4] * 10.0).to(dtype=torch.long)
        card_slots = torch.where(is_card, source_index + 1, torch.zeros_like(source_index))
        potion_slots = torch.where(is_potion, source_index + 1, torch.zeros_like(source_index))
        card_target_index = torch.round(action_scalars[:, -2] * 10.0).to(dtype=torch.long)
        potion_target_index = torch.round(action_scalars[:, -3] * 5.0).to(dtype=torch.long)
        target_index = torch.where(is_potion, potion_target_index, card_target_index)
        target_slots = torch.where(requires_target, target_index + 1, torch.zeros_like(target_index))
        return (
            self._shared_slot_vectors(encoded, token_type_ids, slot_ids, attention_mask, root_indices, card_slots, "HAND_CARD"),
            self._shared_slot_vectors(encoded, token_type_ids, slot_ids, attention_mask, root_indices, target_slots, "MONSTER"),
            self._shared_slot_vectors(encoded, token_type_ids, slot_ids, attention_mask, root_indices, potion_slots, "POTION"),
        )

    def forward(self, batch: Any, **kwargs: Any) -> Any:
        require_torch()
        if not isinstance(batch, dict):
            raise TypeError("root-action transformer expects a dict batch")

        parameter = next(self.parameters())
        token_scalar_features = batch["token_scalar_features"].to(dtype=parameter.dtype, device=parameter.device)
        token_type_ids = batch["token_type_ids"].to(dtype=torch.long, device=parameter.device)
        entity_ids = batch["entity_ids"].to(dtype=torch.long, device=parameter.device).clamp(0, self.entity_vocab_size - 1)
        slot_ids = batch["slot_ids"].to(dtype=torch.long, device=parameter.device).clamp(0, self.slot_vocab_size - 1)
        attention_mask = batch["attention_mask"].to(dtype=torch.bool, device=parameter.device)
        before_summary = batch["before_summary"].to(dtype=parameter.dtype, device=parameter.device)
        legacy_features = batch["features"].to(dtype=parameter.dtype, device=parameter.device)
        candidate_mask = batch["candidate_mask"].to(dtype=torch.bool, device=parameter.device)
        action_positions = batch["action_token_positions"].to(dtype=torch.long, device=parameter.device)
        after_positions = batch["after_token_positions"].to(dtype=torch.long, device=parameter.device)
        delta_positions = batch["delta_token_positions"].to(dtype=torch.long, device=parameter.device)
        legacy_positions = (
            batch.get("legacy_token_positions").to(dtype=torch.long, device=parameter.device)
            if self.uses_legacy_token and batch.get("legacy_token_positions") is not None
            else None
        )

        batch_size, sequence_length = token_type_ids.shape
        positions = torch.arange(sequence_length, device=parameter.device).unsqueeze(0).expand(batch_size, sequence_length)
        x = (
            self.scalar_projection(token_scalar_features)
            + self.token_type_embedding(token_type_ids.clamp(0, self.token_type_vocab_size - 1))
            + self.entity_embedding(entity_ids)
            + self.slot_embedding(slot_ids)
            + self.position_embedding(positions.clamp(0, self.max_sequence_length - 1))
        )

        before_mask = (token_type_ids == TOKEN_TYPES["GLOBAL_BEFORE"]).unsqueeze(-1).to(dtype=x.dtype)
        x = x + before_mask * self.before_summary_projection(before_summary).unsqueeze(1)

        after_start = self.state_dim + self.action_dim
        after_summary = legacy_features[:, after_start : after_start + self.state_dim]
        delta_features = legacy_features[:, -self.delta_dim :]
        x = self._position_addition(x, after_positions, candidate_mask, self.after_summary_projection(after_summary))
        x = self._position_addition(x, delta_positions, candidate_mask, self.delta_projection(delta_features))
        if self.uses_legacy_token and self.legacy_projection is not None and legacy_positions is not None:
            x = self._position_addition(x, legacy_positions, candidate_mask, self.legacy_projection(legacy_features))

        encoded = self.encoder(x, src_key_padding_mask=~attention_mask)
        candidate_root_indices, candidate_action_slots = torch.nonzero(candidate_mask, as_tuple=True)
        flat_action_positions = action_positions[candidate_root_indices, candidate_action_slots]
        flat_after_positions = after_positions[candidate_root_indices, candidate_action_slots]
        flat_delta_positions = delta_positions[candidate_root_indices, candidate_action_slots]
        cls_vector = encoded[:, 0][candidate_root_indices]
        action_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions)
        delta_vector = self._vectors_at(encoded, candidate_root_indices, flat_delta_positions)
        if self.uses_legacy_token and legacy_positions is not None:
            flat_legacy_positions = legacy_positions[candidate_root_indices, candidate_action_slots]
            legacy_vector = self._vectors_at(encoded, candidate_root_indices, flat_legacy_positions)
            score_features = torch.cat([cls_vector, action_vector, delta_vector, legacy_vector], dim=1)
        else:
            after_vector = self._vectors_at(encoded, candidate_root_indices, flat_after_positions)
            if self.root_head_variant == ROOT_HEAD_VARIANT_SELECTED:
                selected_card_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions + 1)
                selected_target_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions + 2)
                selected_potion_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions + 3)
                score_features = torch.cat(
                    [
                        cls_vector,
                        action_vector,
                        selected_card_vector,
                        selected_target_vector,
                        selected_potion_vector,
                        after_vector,
                        delta_vector,
                    ],
                    dim=1,
                )
            elif self.root_head_variant == ROOT_HEAD_VARIANT_ACTION200:
                action_summary = legacy_features[:, self.state_dim : self.state_dim + self.action_dim]
                score_features = torch.cat([cls_vector, action_vector, after_vector, delta_vector, action_summary], dim=1)
            elif self.root_head_variant == ROOT_HEAD_VARIANT_SELECTED_SHARED_POOL:
                selected_card_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions + 1)
                selected_target_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions + 2)
                selected_potion_vector = self._vectors_at(encoded, candidate_root_indices, flat_action_positions + 3)
                shared_card_vector, shared_target_vector, shared_potion_vector = self._selected_shared_vectors(
                    encoded,
                    token_scalar_features,
                    token_type_ids,
                    slot_ids,
                    attention_mask,
                    candidate_root_indices,
                    flat_action_positions,
                )
                score_features = torch.cat(
                    [
                        cls_vector,
                        action_vector,
                        selected_card_vector,
                        selected_target_vector,
                        selected_potion_vector,
                        shared_card_vector,
                        shared_target_vector,
                        shared_potion_vector,
                        after_vector,
                        delta_vector,
                    ],
                    dim=1,
                )
            else:
                score_features = torch.cat([cls_vector, action_vector, after_vector, delta_vector], dim=1)
        return self.output_head(score_features).squeeze(-1)


def save_v3_combat_transformer_checkpoint(
    path: str | Path,
    model: V3CombatTransformerCandidateScorer | V3CombatRootActionSetTransformerScorer,
    *,
    training_args: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    token_schema_payload = dict(
        getattr(model, "checkpoint_token_schema", None)
        or getattr(model, "token_schema", None)
        or asdict(token_spec())
    )
    token_schema_version = str(token_schema_payload.get("version") or TOKEN_SCHEMA_VERSION)
    feature_schema_payload = dict(getattr(model, "checkpoint_feature_schema", None) or schema().__dict__)
    entity_vocab_payload = list(getattr(model, "checkpoint_entity_vocab", None) or transformer_entity_ids())
    entity_vocab_size = int(getattr(model, "entity_vocab_size", len(entity_vocab_payload)) or len(entity_vocab_payload))
    if len(entity_vocab_payload) != entity_vocab_size:
        raise ValueError(
            "refusing to save v3 combat transformer checkpoint with mismatched entity vocab: "
            f"len(entity_vocab)={len(entity_vocab_payload)} != entity_vocab_size={entity_vocab_size}"
        )
    token_types_payload = getattr(model, "checkpoint_token_types", None)
    if token_types_payload is None:
        token_type_vocab_size = int(getattr(model, "token_type_vocab_size", len(TOKEN_TYPES)) or len(TOKEN_TYPES))
        token_types_payload = {name: index for name, index in TOKEN_TYPES.items() if int(index) < token_type_vocab_size}
    else:
        token_types_payload = dict(token_types_payload)
        token_type_vocab_size = int(getattr(model, "token_type_vocab_size", len(token_types_payload)) or len(token_types_payload))
    if any(int(index) >= token_type_vocab_size for index in token_types_payload.values()):
        raise ValueError(
            "refusing to save v3 combat transformer checkpoint with token type id outside vocab: "
            f"token_type_vocab_size={token_type_vocab_size}, token_types={token_types_payload}"
        )
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema": feature_schema_payload,
            "token_schema_version": token_schema_version,
            "token_schema": token_schema_payload,
            "entity_vocab": entity_vocab_payload,
            "token_types": token_types_payload,
            "model_state_dict": model.state_dict(),
            "model_config": model._config(),
            "training_args": dict(training_args or {}),
            "dataset_metadata": dict(dataset_metadata or {}),
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "training_state": dict(training_state or {}),
        },
        target,
    )


def load_v3_combat_transformer_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[V3CombatTransformerCandidateScorer | V3CombatRootActionSetTransformerScorer, dict[str, Any]]:
    require_torch()
    checkpoint = torch_load_portable_path(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported v3 combat transformer checkpoint version: {checkpoint.get('checkpoint_version')}")
    if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "v3 combat transformer checkpoint feature schema mismatch: "
            f"{checkpoint.get('feature_schema_version')} != {FEATURE_SCHEMA_VERSION}"
        )
    if checkpoint.get("token_schema_version") not in SUPPORTED_TOKEN_SCHEMA_VERSIONS:
        raise ValueError(
            "v3 combat transformer token schema mismatch: "
            f"{checkpoint.get('token_schema_version')} not in {sorted(SUPPORTED_TOKEN_SCHEMA_VERSIONS)}"
        )
    config = dict(checkpoint.get("model_config") or {})
    architecture = str(config.pop("architecture", "candidate") or "candidate")
    state_dict = checkpoint["model_state_dict"]
    aux_weight = state_dict.get("aux_reward_head.4.weight") if isinstance(state_dict, dict) else None
    if bool(config.get("auxiliary_reward_heads", False)) and aux_weight is not None:
        config.setdefault("auxiliary_reward_dim", int(aux_weight.shape[0]))
    if architecture == "root_action_set":
        config.setdefault(
            "token_schema_version",
            str((checkpoint.get("token_schema") or {}).get("version") or checkpoint.get("token_schema_version") or ROOT_TOKEN_SCHEMA_VERSION),
        )
        model = V3CombatRootActionSetTransformerScorer(**config)
    else:
        if "action_set_layers" not in config:
            config["action_set_layers"] = 0
        model = V3CombatTransformerCandidateScorer(**config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint
