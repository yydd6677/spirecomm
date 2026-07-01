from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from heapq import nlargest
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterable

from spirecomm.ai.v3_combat_dataset import (
    V3CombatCandidateExample,
    V3CombatLabeledRoot,
    V3CombatRootSample,
)
from spirecomm.ai.v3_combat_features import (
    action_key,
    action_keys_are_unique,
    clear_env_debug_history,
    clone_env_blob,
    encode_action_summary,
    encode_delta,
    encode_state_summary,
    root_combat_actions,
    step_branch,
    step_branch_from_blob,
    _selected_card,
    _selected_potion,
    _is_alive,
)
from spirecomm.native_sim_v3.combat.engine import (
    SUPPORTED_COMBAT_POTION_IDS,
    TARGETED_COMBAT_POTION_IDS,
)
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, RandomXS128, StsRandom
from spirecomm.native_sim_v3.core.state import CombatState, MonsterState, PlayerState
from spirecomm.native_sim_v3.serialize import combat_state as serialize_v3_combat_state


TEACHER_VERSION = "v3_combat_teacher_v12_potion_guard_eval"
TEACHER_VALUE_IS_ZERO = True
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_POTION_ONLY_TEACHER_CONFIG_FIELDS = {
    "potion_monster_room_reward_factor",
    "potion_elite_room_reward_factor",
    "potion_boss_room_reward_factor",
    "potion_cost_scale",
    "potion_buff_adjustment_scale",
    "potion_generation_adjustment_scale",
}
_NON_POTION_ROOT_BEST_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_NON_POTION_CONFIG_KEY_CACHE: dict[int, str] = {}
_STEP_BRANCH_BLOB_CACHE: dict[str, bytes] = {}
_IDENTITY_SENSITIVE_CARD_IDS = {
    "Genetic Algorithm",
    "RitualDagger",
    "Searing Blow",
}
_HAND_ORDER_RANDOM_SENSITIVE_CARD_IDS = {
    # These card effects sample an index from the current hand.  Global
    # equivalent-card dedupe can change non-adjacent remaining hand order.
    "Fiend Fire",
    "Madness",
}


@lru_cache(maxsize=None)
def _teacher_env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in _FALSE_ENV_VALUES


@lru_cache(maxsize=None)
def _teacher_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


CARD_ACTION_TYPES = {"ATTACK", "SKILL", "POWER"}
TURN_ORDER_SKILL_POWER_CARD_IDS = {
    "Battle Trance",
    "Flex",
    "Double Tap",
    "Rage",
    "Shockwave",
    "Spot Weakness",
    "J.A.X.",
    "Trip",
}
POST_COMBAT_PHASES = {"CARD_REWARD", "BOSS_RELIC", "MAP", "EVENT", "SHOP", "CAMPFIRE", "TREASURE", "COMPLETE", "VICTORY"}
ATTACK_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
VISIBLE_POWER_IDS = {
    "FlameBarrier": "Flame Barrier",
    "NextTurnBlock": "Next Turn Block",
}
VISIBLE_MONSTER_IDS = {
    "OrbWalker": "Orb Walker",
    "ShelledParasite": "Shelled Parasite",
}

# Sweep-best values are the defaults. Do not add preset switches here; update
# this dictionary directly when a new sweep result is accepted.
PLAYER_POWER_WEIGHTS = {
    "Strength": 4.486054133555118,
    "Flex": -1.6385851664027493,
    "Rage": 1.6332906750607947,
    "Double Tap": 6.271471430323626,
    "Berserk": 0.9148176194477742,
    "Dark Embrace": 2.6891903495709757,
    "Feel No Pain": 12.868475395409378,
    "Evolve": 0.2519642749819631,
    "Combust": 1.7136738557428957,
    "Metallicize": 18.050839381638962,
    "Rupture": 2.6020408499466665,
    "Fire Breathing": 10.09983821277073,
    "Flame Barrier": 2.8003090967441393,
    "Brutality": 10.167980246573547,
    "Juggernaut": 10.160755129304004,
    "Barricade": 0.8941930544802675,
    "Corruption": 15.869267106946484,
    "Demon Form": 26.733153102226886,
    "No Draw": -3.6373094113436437,
    "Vulnerable": -0.2553872571995335,
    "Artifact": 12.445559791937505,
    "IntangiblePlayer": 8.161931817789254,
    "Panache": 1.0,
    "Sadistic": 1.0,
    "TheBomb": 13.0,
    "Magnetism": 12.0,
    "Mayhem": 13.0,
    "NoBlockPower": -10.0,
}

MONSTER_POWER_WEIGHTS = {
    "Vulnerable": 4.0,
    "Weakened": 2.0,
    "Strength": 4.0,
    "Shackled": 0.5,
}

POTION_COSTS = {
    "Fire Potion": 0.8 * 20.0 + 5.0,
    "Explosive Potion": 0.8 * 20.0,
    "Weak Potion": 6.0 * 0.8,
    "FearPotion": 12.0 * 0.8,
    "Strength Potion": 6.0,
    "SteroidPotion": 8.0,
    "Dexterity Potion": 4.0,
    "SpeedPotion": 7.0,
    "Block Potion": 8.0,
    "Energy Potion": 10.0,
    "Swift Potion": 5.0,
    "Ancient Potion": 3.0,
    "DuplicationPotion": 10.0,
    "EssenceOfSteel": 3.0,
    "LiquidBronze": 6.0,
    "Regen Potion": 10.0,
    "HeartOfIron": 5.0,
    "CultistPotion": 8.0,
    "Fruit Juice": 0.0,
    "AttackPotion": 5.0,
    "SkillPotion": 5.0,
    "PowerPotion": 5.0,
    "ColorlessPotion": 5.0,
    "BlessingOfTheForge": 4.0,
    "LiquidMemories": 5.0,
    "SneckoOil": 5.0,
    "DistilledChaos": 12.0,
    "EntropicBrew": 5.0,
}

# The coefficient above is the value of the unupgraded/base power amount.
# Upgraded cards scale as actual_amount / base_amount.
POWER_BASE_AMOUNTS = {
    "Strength": 1.0,
    "Flex": 1.0,
    "Rage": 3.0,
    "Double Tap": 1.0,
    "Berserk": 1.0,
    "Dark Embrace": 1.0,
    "Feel No Pain": 3.0,
    "Evolve": 1.0,
    "Combust": 5.0,
    "Metallicize": 3.0,
    "Rupture": 1.0,
    "Fire Breathing": 6.0,
    "Flame Barrier": 4.0,
    "Brutality": 1.0,
    "Juggernaut": 5.0,
    "Barricade": 1.0,
    "Corruption": 1.0,
    "Demon Form": 2.0,
    "No Draw": 1.0,
    "Vulnerable": 1.0,
    "Artifact": 1.0,
    "IntangiblePlayer": 1.0,
    "Panache": 5.0,
    "Sadistic": 5.0,
    "TheBomb": 3.0,
    "Magnetism": 1.0,
    "Mayhem": 1.0,
    "NoBlockPower": 1.0,
    "Weakened": 1.0,
    "Shackled": 1.0,
}

NEGATIVE_SENTINEL_POSITIVE_POWERS = {"Barricade", "Corruption"}
NEGATIVE_SENTINEL_DEBUFF_POWERS = {"No Draw"}
TEACHER_POWER_ID_ALIASES = {
    "Weak": "Weakened",
    "Entangle": "Entangled",
    "DemonForm": "Demon Form",
    "FlameBarrier": "Flame Barrier",
    "Intangible": "IntangiblePlayer",
    "NoBlock": "NoBlockPower",
    "SadisticNature": "Sadistic",
    "The Bomb": "TheBomb",
}


TEACHER_REWARD_WEIGHTS = {
    "hp_damage_weight": 0.6464938260661672,
    "monster_kill_weight": 9.778684213036362,
    "combat_win_weight": 19.66808483721789,
    "death_weight": -67.93034920566453,
    "hp_loss_weight": -3.904633760822847,
    "effective_block_weight": 1.645849634046857,
    "raw_incoming_damage_reduction_weight": 0.6942641100479945,
    "playable_hand_count_delta_weight": 8.928984453899307,
}

@dataclass
class TeacherConfig:
    beam_width: int = 24
    node_budget_per_root: int = 768
    max_depth: int = 20
    continuation_action_cap: int = 0
    hp_damage_weight: float = TEACHER_REWARD_WEIGHTS["hp_damage_weight"]
    monster_kill_weight: float = TEACHER_REWARD_WEIGHTS["monster_kill_weight"]
    combat_win_weight: float = TEACHER_REWARD_WEIGHTS["combat_win_weight"]
    death_weight: float = TEACHER_REWARD_WEIGHTS["death_weight"]
    hp_loss_weight: float = TEACHER_REWARD_WEIGHTS["hp_loss_weight"]
    effective_block_weight: float = TEACHER_REWARD_WEIGHTS["effective_block_weight"]
    raw_incoming_damage_reduction_weight: float = TEACHER_REWARD_WEIGHTS["raw_incoming_damage_reduction_weight"]
    energy_spent_weight: float = 0.0
    playable_hand_count_delta_weight: float = TEACHER_REWARD_WEIGHTS["playable_hand_count_delta_weight"]
    play_card_constant: float = 0.0
    power_card_constant: float = 10.0
    skill_power_turn_constant: float = 10.0
    turn_order_decay_per_card: float = 0.2
    potion_monster_room_reward_factor: float = 0.3
    potion_elite_room_reward_factor: float = 1.3
    potion_boss_room_reward_factor: float = 2.0
    potion_cost_scale: float = 1.1
    potion_buff_adjustment_scale: float = 1.0
    potion_generation_adjustment_scale: float = 1.0
    monster_vulnerable_weight: float = 4.0
    monster_weakened_weight: float = 2.0
    monster_strength_weight: float = 4.0
    monster_shackled_weight: float = 0.5
    lethal_check_node_budget: int = 32
    suppress_block_reward_when_lethal_available: bool = True
    lethal_block_suppression_factor: float = 0.75
    lethal_block_low_hp_protection: bool = True
    lethal_block_low_hp_max: int = 8
    lethal_block_low_hp_suppression_factor: float = 1.0
    lethal_block_low_hp_requires_facing_lethal: bool = True
    teacher_survival_guard_enabled: bool = True
    teacher_survival_guard_restrict_safe: bool = True
    teacher_survival_guard_score_margin: float = 10000.0
    player_power_weights: dict[str, float] = field(default_factory=lambda: dict(PLAYER_POWER_WEIGHTS))


@dataclass(frozen=True)
class _TeacherCombatStepSource:
    payload_blob: bytes
    run_env: Any | None = None
    combat_env: Any | None = None
    payload_kind: str = "combat"
    payload_obj: Any | None = field(default=None, compare=False, repr=False)


def default_teacher_config() -> TeacherConfig:
    return TeacherConfig()


def _coerce_teacher_config_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"Cannot parse boolean TeacherConfig value: {value!r}")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError(f"Cannot parse dict TeacherConfig value: {value!r}")
        return {str(key): float(raw_value) for key, raw_value in value.items()}
    return value


def teacher_config_from_mapping(mapping: dict[str, Any] | None = None) -> TeacherConfig:
    """Build a TeacherConfig from a flat mapping or {"teacher_config": {...}} payload."""

    if mapping is None:
        return default_teacher_config()
    payload = mapping.get("teacher_config") if isinstance(mapping.get("teacher_config"), dict) else mapping
    payload = dict(payload)
    player_power_overrides: dict[str, float] = {}
    for key in list(payload):
        prefix = "player_power_weights."
        if str(key).startswith(prefix):
            power_id = str(key)[len(prefix) :]
            if not power_id:
                raise ValueError(f"Invalid TeacherConfig key: {key!r}")
            player_power_overrides[power_id] = float(payload.pop(key))
    if player_power_overrides:
        existing = payload.get("player_power_weights")
        if existing is not None and not isinstance(existing, dict):
            raise ValueError("player_power_weights must be a JSON object when provided")
        merged = dict(existing or {})
        merged.update(player_power_overrides)
        payload["player_power_weights"] = merged
    allowed_fields = {field.name: field for field in fields(TeacherConfig)}
    metadata_keys = {"id", "name", "metadata", "round", "source", "version"}
    unknown_keys = sorted(set(payload) - set(allowed_fields) - metadata_keys)
    if unknown_keys:
        raise ValueError(f"Unknown TeacherConfig keys: {', '.join(unknown_keys)}")
    defaults = default_teacher_config()
    values: dict[str, Any] = {}
    for name in allowed_fields:
        if name in payload:
            default_value = getattr(defaults, name)
            coerced = _coerce_teacher_config_value(payload[name], default_value)
            if isinstance(default_value, dict):
                values[name] = {**default_value, **coerced}
            else:
                values[name] = coerced
    if not values:
        return defaults
    return TeacherConfig(**{**asdict(defaults), **values})


def teacher_config_from_json_path(path: str | os.PathLike[str]) -> TeacherConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher config JSON must contain an object: {path}")
    return teacher_config_from_mapping(payload)


def teacher_config_from_env() -> TeacherConfig:
    raw_json = os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_JSON")
    if raw_json:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("SPIRECOMM_V3_TEACHER_CONFIG_JSON must contain a JSON object")
        return teacher_config_from_mapping(payload)
    config_path = os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_PATH")
    if config_path:
        return teacher_config_from_json_path(config_path)
    if os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_PRESET"):
        raise ValueError(
            "SPIRECOMM_V3_TEACHER_CONFIG_PRESET is no longer supported. "
            "Write sweep-best values into TeacherConfig defaults or pass explicit JSON overrides."
        )
    return default_teacher_config()


@dataclass
class ContinuationResult:
    value: float
    depth: int
    nodes: int
    terminal_kind: str
    debug_best_line: list[dict[str, Any]]
    fully_explored: bool = False
    used_potion_keys: set[tuple[int | None, str]] = field(default_factory=set)


def _state_from_env(env: Any, *, include_run_globals: bool = True) -> dict[str, Any]:
    fast_state = _fast_v3_combat_state_from_env(env, include_run_globals=include_run_globals)
    if fast_state is not None:
        return fast_state
    state_method = getattr(env, "state", None)
    if callable(state_method):
        return _compact_teacher_state(state_method())
    return _compact_teacher_state(env.serialize())


def _scoring_state_from_env(env: Any) -> dict[str, Any]:
    return _state_from_env(env, include_run_globals=False)


def _combat_state_view(state: dict[str, Any]) -> dict[str, Any]:
    combat = state.get("combat_state")
    return combat if isinstance(combat, dict) else {}


def _monsters_view(state: dict[str, Any]) -> list[dict[str, Any]]:
    monsters_value = _combat_state_view(state).get("monsters")
    return monsters_value if isinstance(monsters_value, list) else []


def _monsters_from_combat_view(combat: dict[str, Any]) -> list[dict[str, Any]]:
    monsters_value = combat.get("monsters")
    return monsters_value if isinstance(monsters_value, list) else []


def _hand_cards_view(state: dict[str, Any]) -> list[dict[str, Any]]:
    hand = _combat_state_view(state).get("hand")
    return hand if isinstance(hand, list) else []


def _hand_cards_from_combat_view(combat: dict[str, Any]) -> list[dict[str, Any]]:
    hand = combat.get("hand")
    return hand if isinstance(hand, list) else []


def _player_from_combat_view(combat: dict[str, Any]) -> dict[str, Any]:
    player = combat.get("player")
    return player if isinstance(player, dict) else {}


def _incoming_damage_from_monsters(monsters: list[dict[str, Any]]) -> int:
    total = 0
    for monster in monsters:
        if not _is_alive(monster):
            continue
        if str(monster.get("intent") or "") not in ATTACK_INTENTS:
            continue
        try:
            damage = int(monster.get("move_adjusted_damage") or 0)
            hits = int(monster.get("move_hits") or 1)
        except (TypeError, ValueError):
            continue
        total += max(0, damage) * max(1, hits)
    return total


def _incoming_damage_fast(state: dict[str, Any]) -> int:
    return _incoming_damage_from_monsters(_monsters_view(state))


def _playable_count_fast(state: dict[str, Any]) -> int:
    return sum(1 for card in _hand_cards_view(state) if bool(card.get("is_playable", False)))


def _playable_count_from_hand(hand: list[dict[str, Any]]) -> int:
    return sum(1 for card in hand if bool(card.get("is_playable", False)))


def _compact_teacher_state(state: dict[str, Any]) -> dict[str, Any]:
    state.pop("rng_trace", None)
    state.pop("commands", None)
    return state


def _teacher_visible_power(power: dict[str, Any], *, owner: str | None = None) -> dict[str, Any]:
    payload = dict(power)
    power_id = str(payload.get("power_id") or payload.get("id") or payload.get("name") or "")
    visible_id = "IntangiblePlayer" if owner == "player" and power_id == "Intangible" else VISIBLE_POWER_IDS.get(power_id, power_id)
    if visible_id:
        payload["power_id"] = visible_id
        payload["id"] = visible_id
        payload["name"] = visible_id
    payload["__teacher_branch_key"] = (payload.get("power_id") or payload.get("id") or payload.get("name"), payload.get("amount"))
    return payload


def _teacher_visible_card(card: dict[str, Any]) -> dict[str, Any]:
    payload = dict(card)
    visible_cost = payload.get("cost_for_turn")
    if visible_cost is None:
        visible_cost = payload.get("cost")
    payload["cost"] = visible_cost
    card_id = str(payload.get("card_id") or "")
    if card_id not in _IDENTITY_SENSITIVE_CARD_IDS:
        tags = payload.get("tags")
        tags_key = tuple(str(tag) for tag in tags) if isinstance(tags, list) else ()
        payload["__teacher_branch_key"] = (
            card_id,
            payload.get("name"),
            payload.get("type"),
            payload.get("rarity"),
            payload.get("cost"),
            payload.get("base_cost"),
            payload.get("cost_for_turn"),
            payload.get("cost_for_combat"),
            payload.get("free_to_play_once"),
            payload.get("upgrades"),
            payload.get("misc"),
            payload.get("exhausts"),
            payload.get("has_target"),
            payload.get("is_playable"),
            payload.get("base_damage"),
            payload.get("base_block"),
            payload.get("base_magic"),
            payload.get("ethereal"),
            payload.get("innate"),
            payload.get("retain"),
            payload.get("color"),
            payload.get("target"),
            tags_key,
        )
    return payload


def _pending_select_visible_cards(pending_select: dict[str, Any]) -> list[Any]:
    pending_mode = str(pending_select.get("mode") or "").upper()
    cards = list(pending_select.get("cards") or ())
    if pending_mode not in {"GAMBLING_CHIP", "ELIXIR"}:
        return cards
    selected_source_indexes = {
        int(index) for index in list(pending_select.get("selected_source_indexes") or [])
    }
    source_indexes = list(pending_select.get("source_indexes") or [])
    visible_cards = []
    for index, card in enumerate(cards):
        source_index = source_indexes[index] if index < len(source_indexes) else index
        if int(source_index) not in selected_source_indexes:
            visible_cards.append(card)
    return visible_cards


def _teacher_visible_potion(potion: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "potion_id": potion.get("potion_id") or potion.get("id") or potion.get("name"),
        "id": potion.get("id") or potion.get("potion_id") or potion.get("name"),
        "name": potion.get("name") or potion.get("potion_id") or potion.get("id"),
        "requires_target": bool(potion.get("requires_target", False)),
        "can_use": bool(potion.get("can_use", True)),
        "can_discard": bool(potion.get("can_discard", True)),
    }
    payload["__teacher_branch_key"] = (
        payload.get("potion_id") or payload.get("id") or payload.get("name"),
        payload.get("can_use"),
        payload.get("can_discard"),
        payload.get("requires_target"),
    )
    return payload


def _teacher_monster_state(monster: Any, *, hide_intent: bool = False, copy_powers: bool = True) -> dict[str, Any]:
    meta = getattr(monster, "meta", {}) or {}
    monster_id = str(getattr(monster, "monster_id", "") or "")
    half_dead = bool(meta.get("half_dead", False))
    if monster_id in {"Darkling", "AwakenedOne"} and half_dead:
        return {
            "monster_id": monster_id,
            "name": getattr(monster, "name", None) or monster_id,
            "current_hp": int(getattr(monster, "current_hp", 0) or 0),
            "max_hp": int(getattr(monster, "max_hp", 0) or 0),
            "block": int(getattr(monster, "block", 0) or 0),
            "half_dead": False,
            "is_gone": True,
            "powers": [],
        }
    intent = str(getattr(monster, "intent", "") or "")
    is_attack = intent in ATTACK_INTENTS
    current_hp = int(getattr(monster, "current_hp", 0) or 0)
    return {
        "monster_id": VISIBLE_MONSTER_IDS.get(monster_id, monster_id),
        "name": getattr(monster, "name", None) or monster_id,
        "current_hp": current_hp,
        "max_hp": int(getattr(monster, "max_hp", 0) or 0),
        "block": int(getattr(monster, "block", 0) or 0),
        "intent": "UNKNOWN" if hide_intent else intent,
        "half_dead": half_dead,
        "is_gone": (current_hp <= 0 or bool(meta.get("escaped", False))) and not half_dead,
        "move_adjusted_damage": 0 if hide_intent else int(getattr(monster, "move_adjusted_damage", -1 if not is_attack else 0) or 0),
        "move_hits": 0 if hide_intent else int(getattr(monster, "move_hits", 1) or 1),
        "powers": (
            [_teacher_visible_power(power, owner="monster") for power in (getattr(monster, "powers", None) or ())]
            if copy_powers
            else (getattr(monster, "powers", None) or [])
        ),
    }


def _fast_teacher_v3_combat_state_from_env(
    env: Any,
    *,
    include_run_globals: bool = True,
) -> dict[str, Any] | None:
    if not _teacher_env_flag("SPIRECOMM_V3_TEACHER_FAST_STATE", True):
        return None
    fast_terminal_state = getattr(env, "_teacher_fast_terminal_state", None)
    if isinstance(fast_terminal_state, dict):
        payload = dict(fast_terminal_state)
        payload.pop("rng_trace", None)
        payload.pop("commands", None)
        return payload
    run_env = env
    phase = str(getattr(env, "phase", "COMBAT") or "COMBAT")
    combat_env = None
    if getattr(env, "sim_backend", None) == "v3" and getattr(env, "combat", None) is not None:
        if phase in {"COMBAT", "CARD_SELECT", "CARD_REWARD"}:
            combat_env = getattr(env, "combat", None)
    elif getattr(env, "sim_backend", None) == "v3" and getattr(env, "engine", None) is not None:
        combat_env = env
        phase = "COMBAT"
    if combat_env is None:
        return None
    engine = getattr(combat_env, "engine", None)
    combat_state = getattr(combat_env, "state", None)
    if combat_state is None and engine is not None:
        combat_state = getattr(engine, "state", None)
    if combat_state is None:
        return None
    outcome = str(getattr(engine, "outcome", "UNDECIDED") or "UNDECIDED") if engine is not None else "UNDECIDED"
    pending_select = getattr(engine, "pending_card_select", None) if outcome == "UNDECIDED" else None
    pending_mode = str((pending_select or {}).get("mode") or "").upper()
    if pending_mode in {"DISCOVERY", "NILRYS_CODEX"}:
        phase = "CARD_REWARD"
    elif pending_select is not None:
        phase = "CARD_SELECT"
    screen_state: dict[str, Any] = {}
    if pending_select is not None:
        if phase == "CARD_REWARD":
            screen_state = {
                "cards": [_teacher_visible_card(card) for card in (pending_select.get("cards") or ())],
                "bowl_available": False,
                "skip_available": bool(pending_select.get("can_skip", False)),
            }
        else:
            selected_cards = pending_select.get("selected_cards") or ()
            screen_state = {
                "cards": [_teacher_visible_card(card) for card in _pending_select_visible_cards(pending_select)],
                "num_cards": int(pending_select.get("num_cards") or 1),
                "max_cards": int(pending_select.get("num_cards") or 1),
                "any_number": bool(pending_select.get("any_number", False)),
                "can_pick_zero": bool(pending_select.get("can_pick_zero", False)),
                "for_upgrade": False,
                "for_transform": False,
                "for_purge": False,
                "confirm_up": bool(pending_select.get("confirm_up", False)),
                "selected_cards": [_teacher_visible_card(card) for card in selected_cards],
            }
    player = getattr(combat_state, "player", getattr(engine, "player", getattr(combat_env, "player", None)))
    all_pending_actions = [dict(action) for action in combat_env.legal_actions()] if pending_select is not None else []
    if pending_mode in {"GAMBLING_CHIP", "ELIXIR"}:
        choice_actions = [action for action in all_pending_actions if action.get("kind") != "confirm"]
    else:
        choice_actions = all_pending_actions
    if include_run_globals:
        deck = getattr(engine, "master_deck", None) if engine is not None else None
        if deck is None:
            deck = getattr(combat_env, "master_deck", getattr(run_env, "deck", []))
        deck_value = list(deck or [])
    else:
        deck_value = []
    relics = getattr(engine, "relics", None) if engine is not None else None
    if relics is None:
        relics = getattr(combat_env, "relics", getattr(run_env, "relics", []))
    potions = getattr(engine, "potions", None) if engine is not None else None
    if potions is None:
        potions = getattr(combat_env, "potions", getattr(run_env, "potions", []))
    gold = getattr(engine, "gold", None) if engine is not None else None
    if gold is None:
        gold = getattr(combat_env, "gold", getattr(run_env, "gold", 0))
    hide_intent_cache = getattr(combat_env, "_teacher_hide_intent_cache", None)
    if isinstance(hide_intent_cache, bool):
        hide_intent = hide_intent_cache
    else:
        hide_intent = any(
            str(relic.get("relic_id") or relic.get("id")) == "Runic Dome"
            for relic in (relics or ())
        )
        try:
            setattr(combat_env, "_teacher_hide_intent_cache", bool(hide_intent))
        except Exception:
            pass
    relics_value = list(relics or []) if include_run_globals else []
    reference_sources_value = (
        dict(getattr(combat_env, "reference_sources", getattr(run_env, "reference_sources", {})))
        if include_run_globals
        else {}
    )
    return {
        "backend": "v3",
        "implementation_status": "combat_vertical_slice",
        "phase": phase,
        "screen": phase,
        "screen_type": phase,
        "screen_up": pending_select is not None,
        "ascension_level": int(getattr(combat_env, "ascension_level", getattr(run_env, "ascension_level", 0)) or 0),
        "act": int(getattr(combat_env, "act", getattr(run_env, "act", 1)) or 1),
        "dungeon_id": getattr(combat_env, "dungeon_id", getattr(run_env, "dungeon_id", None)),
        "floor": int(getattr(combat_env, "floor", getattr(run_env, "floor", 0)) or 0),
        "room_type": phase if pending_select is not None else getattr(combat_env, "room_type", getattr(run_env, "current_room_type", None)),
        "current_hp": int(getattr(player, "current_hp", 0) or 0),
        "max_hp": int(getattr(player, "max_hp", 0) or 0),
        "gold": int(gold or 0),
        "act_boss": getattr(combat_env, "act_boss", getattr(run_env, "act_boss", None)),
        "deck": deck_value,
        "relics": relics_value,
        "potions": [_teacher_visible_potion(potion) for potion in (potions or ())],
        "combat_state": {
            "turn": int(getattr(combat_state, "turn", 0) or 0) + 1,
            "cards_played_this_turn": int(getattr(engine, "cards_played_this_turn", 0) or 0) if engine is not None else 0,
            "player": {
                "current_hp": int(getattr(player, "current_hp", 0) or 0),
                "max_hp": int(getattr(player, "max_hp", 0) or 0),
                "block": int(getattr(player, "block", 0) or 0),
                "energy": int(getattr(player, "energy", 0) or 0),
                "powers": [
                    _teacher_visible_power(power, owner="player")
                    for power in (getattr(player, "powers", None) or ())
                ],
            },
            "cards_discarded_this_turn": 0,
            "monsters": [
                _teacher_monster_state(monster, hide_intent=hide_intent, copy_powers=True)
                for monster in (getattr(combat_state, "monsters", None) or ())
            ],
            "hand": [_teacher_visible_card(card) for card in (getattr(combat_state, "hand", None) or ())],
        },
        "screen_state": screen_state,
        "choice_available": pending_select is not None,
        "choice_list": choice_actions,
        "reference_sources": reference_sources_value,
    }


def _fast_v3_combat_state_from_env(env: Any, *, include_run_globals: bool = True) -> dict[str, Any] | None:
    fast_state = _fast_teacher_v3_combat_state_from_env(env, include_run_globals=include_run_globals)
    if fast_state is not None:
        return fast_state
    phase = str(getattr(env, "phase", "COMBAT") or "COMBAT")
    combat_env = None
    if getattr(env, "sim_backend", None) == "v3" and getattr(env, "combat", None) is not None:
        if phase in {"COMBAT", "CARD_SELECT", "CARD_REWARD"}:
            combat_env = getattr(env, "combat", None)
    elif getattr(env, "sim_backend", None) == "v3" and getattr(env, "engine", None) is not None:
        combat_env = env
    if combat_env is None:
        return None
    try:
        return serialize_v3_combat_state(combat_env, include_debug_trace=False, include_commands=False)
    except Exception:
        return None


def _phase(env: Any) -> str:
    return str(getattr(env, "phase", None) or _state_from_env(env).get("phase") or "COMBAT")


def _phase_from_state(env: Any, state: dict[str, Any]) -> str:
    return str(getattr(env, "phase", None) or state.get("phase") or "COMBAT")


def _outcome(env: Any) -> str:
    return str(getattr(env, "outcome", "") or "")


def _combat_turn(env: Any) -> int:
    try:
        return _combat_turn_from_state(_scoring_state_from_env(env))
    except (TypeError, ValueError):
        return 0


def _combat_turn_from_state(state: dict[str, Any]) -> int:
    try:
        return int((_combat_state_view(state) or {}).get("turn") or 0)
    except (TypeError, ValueError):
        return 0


def _is_terminal(env: Any) -> str | None:
    phase = _phase(env)
    state = _scoring_state_from_env(env)
    return _is_terminal_from_state(env, state, phase)


def _is_terminal_from_state(env: Any, state: dict[str, Any], phase: str) -> str | None:
    outcome = _outcome(env)
    if phase in {"COMPLETE", "VICTORY"} or outcome == "VICTORY":
        return "VICTORY"
    if phase == "GAME_OVER" or outcome == "DEFEAT":
        return "DEFEAT"
    if phase in POST_COMBAT_PHASES and not _combat_state_view(state):
        return "VICTORY"
    return None


def _is_next_player_decision(env: Any, *, root_turn: int) -> bool:
    if _phase(env) != "COMBAT":
        return False
    state = _scoring_state_from_env(env)
    return _is_next_player_decision_from_state(state, phase="COMBAT", root_turn=root_turn)


def _is_next_player_decision_from_state(state: dict[str, Any], *, phase: str, root_turn: int) -> bool:
    if phase != "COMBAT":
        return False
    if bool(state.get("choice_available", False)):
        return False
    return _combat_turn_from_state(state) > root_turn


def _legal_teacher_actions(env: Any) -> list[dict[str, Any]]:
    phase = _phase(env)
    state = _scoring_state_from_env(env)
    return _legal_teacher_actions_from_state(env, state, phase)


def _fast_raw_potion_id(potion: dict[str, Any] | None) -> str:
    if not isinstance(potion, dict):
        return ""
    return str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")


def _fast_obj_monster_alive(monster: Any) -> bool:
    return int(getattr(monster, "current_hp", 0) or 0) > 0 and not bool(getattr(monster, "meta", {}).get("escaped", False))


def _fast_obj_monster_power_key(monster: Any) -> tuple[tuple[str, int], ...]:
    powers = getattr(monster, "powers", None)
    if powers is None and isinstance(monster, dict):
        powers = monster.get("powers")
    if isinstance(powers, dict):
        raw_items = powers.items()
    elif isinstance(powers, list):
        raw_items = [
            (
                str(item.get("id") or item.get("power_id") or item.get("name") or ""),
                item.get("amount", 0),
            )
            for item in powers
            if isinstance(item, dict)
        ]
    else:
        raw_items = []
    items: list[tuple[str, int]] = []
    for key, value in raw_items:
        try:
            amount = int(value)
        except (TypeError, ValueError):
            amount = 0
        if amount:
            items.append((str(key), amount))
    return tuple(sorted(items))


def _equivalent_monster_target_key(monster: Any) -> tuple[Any, ...]:
    meta = getattr(monster, "meta", None)
    if meta is None and isinstance(monster, dict):
        meta = monster.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    return (
        str(getattr(monster, "id", "") or getattr(monster, "name", "") or (monster.get("id") if isinstance(monster, dict) else "")),
        int(getattr(monster, "current_hp", 0) if not isinstance(monster, dict) else monster.get("current_hp", 0) or 0),
        int(getattr(monster, "block", 0) if not isinstance(monster, dict) else monster.get("block", 0) or 0),
        str(getattr(monster, "intent", "") if not isinstance(monster, dict) else monster.get("intent", "") or ""),
        int(getattr(monster, "next_damage", 0) if not isinstance(monster, dict) else monster.get("next_damage", 0) or 0),
        int(getattr(monster, "move_hits", 0) if not isinstance(monster, dict) else monster.get("move_hits", 0) or 0),
        bool(meta.get("escaped", False)),
        bool(getattr(monster, "half_dead", False) if not isinstance(monster, dict) else monster.get("half_dead", False)),
        _fast_obj_monster_power_key(monster),
    )


def _equivalent_card_action_key(card: dict[str, Any], target_index: int | None) -> tuple[Any, ...]:
    """Collapse distinct physical copies only when their visible combat semantics match."""

    tags = card.get("tags")
    if isinstance(tags, list):
        tags_key: tuple[Any, ...] = tuple(str(tag) for tag in tags)
    else:
        tags_key = ()
    return (
        str(card.get("card_id") or ""),
        str(card.get("name") or ""),
        str(card.get("type") or ""),
        str(card.get("target") or ""),
        str(card.get("color") or ""),
        str(card.get("rarity") or ""),
        int(card.get("cost") if card.get("cost") is not None else -99),
        int(card.get("cost_for_turn") if card.get("cost_for_turn") is not None else -99),
        int(card.get("cost_for_combat") if card.get("cost_for_combat") is not None else -99),
        int(card.get("base_cost") if card.get("base_cost") is not None else -99),
        int(card.get("base_damage") or 0),
        int(card.get("base_block") or 0),
        int(card.get("base_magic") or 0),
        int(card.get("upgrades") or 0),
        int(card.get("misc") or 0),
        bool(card.get("has_target", False)),
        bool(card.get("is_playable", False)),
        bool(card.get("exhausts", False)),
        bool(card.get("ethereal", False)),
        bool(card.get("retain", False)),
        bool(card.get("innate", False)),
        bool(card.get("free_to_play_once", False)),
        tags_key,
        int(target_index) if target_index is not None else None,
    )


def _has_hand_order_random_sensitive_card(hand: Iterable[dict[str, Any]]) -> bool:
    for card in hand:
        card_id = str(card.get("card_id") or "")
        if card_id in _HAND_ORDER_RANDOM_SENSITIVE_CARD_IDS:
            return True
        if card_id == "True Grit" and int(card.get("upgrades") or 0) <= 0:
            return True
    return False


def _fast_combat_teacher_actions(env: Any, phase: str) -> list[dict[str, Any]] | None:
    if phase != "COMBAT":
        return None
    if not _teacher_env_flag("SPIRECOMM_V3_TEACHER_FAST_LEGAL_ACTIONS", True):
        return None
    combat_env = None
    if getattr(env, "sim_backend", None) == "v3" and getattr(env, "combat", None) is not None:
        combat_env = getattr(env, "combat", None)
    elif getattr(env, "sim_backend", None) == "v3" and getattr(env, "engine", None) is not None:
        combat_env = env
    if combat_env is None:
        return None
    engine = getattr(combat_env, "engine", None)
    state = getattr(engine, "state", None)
    if engine is None or state is None or getattr(engine, "pending_card_select", None) is not None:
        return None
    actions: list[dict[str, Any]] = []
    monsters = getattr(state, "monsters", None) or ()
    dedupe_targets = _teacher_env_flag("SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_TARGETS", True)
    potions = getattr(engine, "potions", getattr(combat_env, "potions", ())) or ()
    for potion_index, potion in enumerate(potions):
        potion_id = _fast_raw_potion_id(potion)
        if potion_id in {"", "Potion Slot"} or not bool((potion or {}).get("can_use", True)):
            continue
        if potion_id not in SUPPORTED_COMBAT_POTION_IDS:
            continue
        base_action = {
            "kind": "potion",
            "name": potion.get("name") or potion_id,
            "potion_id": potion_id,
            "potion_index": potion_index,
            "requires_target": potion_id in TARGETED_COMBAT_POTION_IDS,
        }
        if potion_id in TARGETED_COMBAT_POTION_IDS:
            seen_target_keys: set[tuple[Any, ...]] = set()
            for target_index, monster in enumerate(monsters):
                if not _fast_obj_monster_alive(monster):
                    continue
                if dedupe_targets:
                    target_key = _equivalent_monster_target_key(monster)
                    if target_key in seen_target_keys:
                        continue
                    seen_target_keys.add(target_key)
                actions.append({**base_action, "target_index": target_index, "model_target_index": target_index})
        else:
            actions.append(base_action)

    relics = getattr(engine, "relics", getattr(combat_env, "relics", ())) or ()
    has_blue_candle = False
    has_medical_kit = False
    has_velvet_choker = False
    velvet_counter = 0
    for relic in relics:
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        if relic_id == "Blue Candle":
            has_blue_candle = True
        elif relic_id == "Medical Kit":
            has_medical_kit = True
        elif relic_id == "Velvet Choker":
            has_velvet_choker = True
            velvet_counter = int(relic.get("counter", 0) or 0)
    hand = getattr(state, "hand", None) or ()
    dedupe_card_actions = _teacher_env_flag("SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_CARD_ACTIONS", True)
    if dedupe_card_actions and _has_hand_order_random_sensitive_card(hand):
        dedupe_card_actions = False
    safe_adjacent_dedupe = _teacher_env_flag("SPIRECOMM_V3_TEACHER_DEDUPE_SAFE_ADJACENT_CARD_ACTIONS", True)
    seen_card_action_keys: set[tuple[Any, ...]] = set()
    previous_card_action_index = -2
    previous_card_action_keys: set[tuple[Any, ...]] = set()
    current_card_action_index = -2
    current_card_action_keys: set[tuple[Any, ...]] = set()

    def append_card_action(action: dict[str, Any], card: dict[str, Any], target_index: int | None = None) -> None:
        nonlocal previous_card_action_index, previous_card_action_keys
        nonlocal current_card_action_index, current_card_action_keys
        try:
            card_index = int(action.get("card_index"))
        except (TypeError, ValueError):
            card_index = -1
        key = _equivalent_card_action_key(card, target_index)
        if card_index != current_card_action_index:
            if current_card_action_index >= 0:
                previous_card_action_index = current_card_action_index
                previous_card_action_keys = set(current_card_action_keys)
            if card_index != previous_card_action_index + 1:
                previous_card_action_index = -2
                previous_card_action_keys = set()
            current_card_action_index = card_index
            current_card_action_keys = set()
        if dedupe_card_actions:
            if key in seen_card_action_keys:
                current_card_action_keys.add(key)
                return
            seen_card_action_keys.add(key)
        elif (
            safe_adjacent_dedupe
            and card_index >= 1
            and previous_card_action_index == card_index - 1
            and str(card.get("card_id") or "") not in _IDENTITY_SENSITIVE_CARD_IDS
        ):
            if key in previous_card_action_keys:
                current_card_action_keys.add(key)
                return
        current_card_action_keys.add(key)
        actions.append(action)

    cards_played_this_turn = int(getattr(engine, "cards_played_this_turn", 0) or 0)
    normality_locked = cards_played_this_turn >= 3 and any(card.get("card_id") == "Normality" for card in hand)
    velvet_locked = has_velvet_choker and (velvet_counter >= 6 or cards_played_this_turn >= 6)
    draw_pile = None
    if normality_locked or velvet_locked:
        actions.append({"kind": "end", "name": "END_TURN", "action_index": 0})
        return actions
    for index, card in enumerate(hand):
        card_id = str(card.get("card_id") or "")
        card_type = card.get("type")
        if card_type == "CURSE" and has_blue_candle and card_id != "Necronomicurse":
            append_card_action(
                {
                    "kind": "card",
                    "name": card["name"],
                    "card_id": card["card_id"],
                    "card_type": card_type,
                    "card_index": index,
                    "source_index": index,
                    "requires_target": False,
                },
                card,
            )
            continue
        if card_type == "STATUS" and has_medical_kit:
            append_card_action(
                {
                    "kind": "card",
                    "name": card["name"],
                    "card_id": card["card_id"],
                    "card_type": card_type,
                    "card_index": index,
                    "source_index": index,
                    "requires_target": False,
                },
                card,
            )
            continue
        if not card.get("is_playable"):
            continue
        if card_id == "Clash" and any(hand_card.get("type") != "ATTACK" for hand_card in hand if hand_card is not card):
            continue
        if card_id == "Secret Technique":
            if draw_pile is None:
                draw_pile = getattr(state, "draw_pile", None) or ()
            if not any(draw_card.get("type") == "SKILL" for draw_card in draw_pile):
                continue
        if card_id == "Secret Weapon":
            if draw_pile is None:
                draw_pile = getattr(state, "draw_pile", None) or ()
            if not any(draw_card.get("type") == "ATTACK" for draw_card in draw_pile):
                continue
        if card.get("has_target"):
            seen_target_keys: set[tuple[Any, ...]] = set()
            for target_index, monster in enumerate(monsters):
                if not _fast_obj_monster_alive(monster):
                    continue
                if dedupe_targets:
                    target_key = _equivalent_monster_target_key(monster)
                    if target_key in seen_target_keys:
                        continue
                    seen_target_keys.add(target_key)
                append_card_action(
                    {
                        "kind": "card",
                        "name": card["name"],
                        "card_id": card["card_id"],
                        "card_type": card_type,
                        "card_index": index,
                        "source_index": index,
                        "target_index": target_index,
                        "model_target_index": target_index,
                        "requires_target": True,
                    },
                    card,
                    target_index,
                )
        else:
            append_card_action(
                {
                    "kind": "card",
                    "name": card["name"],
                    "card_id": card["card_id"],
                    "card_type": card_type,
                    "card_index": index,
                    "source_index": index,
                    "requires_target": False,
                },
                card,
            )
    actions.append({"kind": "end", "name": "END_TURN", "action_index": 0})
    return actions


def _legal_teacher_actions_from_state(env: Any, state: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    fast_actions = _fast_combat_teacher_actions(env, phase)
    if fast_actions is not None:
        return fast_actions
    actions = env.legal_actions()
    if phase == "COMBAT":
        return [action for action in actions if action.get("kind") in {"card", "potion", "end"}]
    if phase in {"CARD_SELECT", "CARD_REWARD"} and _combat_state_view(state):
        return list(actions)
    return []


def _approx_action_priority_from_state(state: dict[str, Any], action: dict[str, Any]) -> float:
    kind = str(action.get("kind") or "")
    if kind == "end":
        return -1.0
    if kind == "potion":
        return 40.0
    if kind != "card":
        return 0.0
    combat = _combat_state_view(state)
    hand = _hand_cards_from_combat_view(combat)
    try:
        card_index = int(action.get("card_index", action.get("source_index", -1)))
    except (TypeError, ValueError):
        card_index = -1
    card = hand[card_index] if 0 <= card_index < len(hand) else {}
    if not isinstance(card, dict):
        card = {}
    card_type = str(card.get("type") or action.get("card_type") or "")
    card_id = str(card.get("card_id") or action.get("card_id") or "")
    cost = _int(card.get("cost_for_turn"), _int(card.get("cost"), 1))
    damage = max(
        0.0,
        float(
            _int(
                card.get("damage"),
                _int(card.get("base_damage"), _int(action.get("damage"), 0)),
            )
        ),
    )
    block = max(
        0.0,
        float(
            _int(
                card.get("block"),
                _int(card.get("base_block"), _int(action.get("block"), 0)),
            )
        ),
    )
    magic = max(0.0, float(_int(card.get("magic"), _int(card.get("base_magic"), 0))))
    priority = 0.0
    if card_type == "ATTACK":
        priority += 1.8 * damage + 0.5 * magic
        monsters = _monsters_from_combat_view(combat)
        try:
            target_index = int(action.get("target_index"))
        except (TypeError, ValueError):
            target_index = -1
        if 0 <= target_index < len(monsters):
            target = monsters[target_index]
            target_hp = max(0, _int(target.get("current_hp")) + _int(target.get("block")))
            if target_hp > 0 and damage >= target_hp:
                priority += 80.0
            priority += max(0.0, 1.0 - target_hp / 80.0) * 8.0
    elif card_type == "SKILL":
        incoming = float(_incoming_damage_fast(state))
        priority += 1.3 * min(block, max(incoming, 0.0)) + 0.35 * block + 0.6 * magic
    elif card_type == "POWER":
        priority += 18.0 + 0.5 * magic
    else:
        priority += 0.8 * damage + 0.8 * block + 0.5 * magic
    if cost == 0:
        priority += 5.0
    elif cost > 1:
        priority -= 1.5 * float(cost - 1)
    if card_id in {"Feed", "Reaper", "RitualDagger", "Limit Break", "Impervious", "Flame Barrier"}:
        priority += 10.0
    if bool(card.get("exhausts")):
        priority -= 1.0
    return float(priority)


def _prune_continuation_actions_for_speed(
    state: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    action_cap: int | None = None,
) -> list[dict[str, Any]]:
    cap = (
        _teacher_env_int("SPIRECOMM_V3_TEACHER_CONTINUATION_ACTION_CAP", 0)
        if action_cap is None or int(action_cap) <= 0
        else int(action_cap)
    )
    if cap <= 0 or len(actions) <= cap:
        return actions
    keep_end = _teacher_env_flag("SPIRECOMM_V3_TEACHER_CONTINUATION_ALWAYS_KEEP_END", True)
    indexed = list(enumerate(actions))
    kept_indices: set[int] = set()
    if keep_end:
        for index, action in indexed:
            if str(action.get("kind") or "") == "end":
                kept_indices.add(index)
                break
    slots = max(0, int(cap) - len(kept_indices))
    ranked = sorted(
        (
            (_approx_action_priority_from_state(state, action), -index, index)
            for index, action in indexed
            if index not in kept_indices
        ),
        reverse=True,
    )
    for _priority, _negative_index, index in ranked[:slots]:
        kept_indices.add(index)
    if not kept_indices:
        return actions[: max(1, int(cap))]
    return [action for index, action in indexed if index in kept_indices]


def _root_combat_actions_no_copy(env: Any, *, legal_actions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if str(getattr(env, "phase", "COMBAT")) != "COMBAT":
        return []
    actions = legal_actions if legal_actions is not None else env.legal_actions()
    return [action for action in actions if action.get("kind") in {"card", "potion", "end"}]


def _clone_env_blob_or_none(env: Any) -> bytes | None:
    try:
        slim_branch = _teacher_env_flag("SPIRECOMM_V3_TEACHER_SLIM_BRANCH_CLONE", True)
        clear_env_debug_history(env)
        return clone_env_blob(env, strip_debug_history=False, teacher_branch_slim=slim_branch)
    except Exception:
        return None


def _clone_step_source_or_none(env: Any) -> bytes | _TeacherCombatStepSource | None:
    if _teacher_env_flag("SPIRECOMM_V3_TEACHER_COMBAT_STEP_SOURCE", True):
        try:
            combat_env = None
            run_env = None
            if getattr(env, "sim_backend", None) == "v3" and getattr(env, "combat", None) is not None:
                combat_env = getattr(env, "combat", None)
                if not _teacher_env_flag("SPIRECOMM_V3_TEACHER_COMBAT_BRANCH_ONLY", False):
                    run_env = env
            elif getattr(env, "sim_backend", None) == "v3" and getattr(env, "engine", None) is not None:
                combat_env = env
            if combat_env is not None:
                if not bool(getattr(combat_env, "_teacher_random_debug_disabled", False)):
                    clear_env_debug_history(combat_env)
                engine = getattr(combat_env, "engine", None)
                if (
                    engine is not None
                    and _teacher_env_flag("SPIRECOMM_V3_TEACHER_ENGINE_STEP_SOURCE", True)
                ):
                    had_slim_flag = hasattr(engine, "_teacher_branch_clone_slim")
                    previous_slim_flag = getattr(engine, "_teacher_branch_clone_slim", False)
                    try:
                        setattr(engine, "_teacher_branch_clone_slim", True)
                        engine_blob = pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL)
                    finally:
                        if had_slim_flag:
                            setattr(engine, "_teacher_branch_clone_slim", previous_slim_flag)
                        else:
                            try:
                                delattr(engine, "_teacher_branch_clone_slim")
                            except Exception:
                                pass
                    return _TeacherCombatStepSource(
                        payload_blob=engine_blob,
                        run_env=run_env,
                        combat_env=combat_env,
                        payload_kind="engine",
                        payload_obj=engine,
                    )
                combat_blob = clone_env_blob(
                    combat_env,
                    strip_debug_history=False,
                    teacher_branch_slim=True,
                )
                return _TeacherCombatStepSource(payload_blob=combat_blob, run_env=run_env, payload_kind="combat")
        except Exception:
            pass
    return _clone_env_blob_or_none(env)


def _non_potion_config_cache_key(config: TeacherConfig) -> str:
    object_key = id(config)
    cached = _NON_POTION_CONFIG_KEY_CACHE.get(object_key)
    if cached is not None:
        return cached
    payload: dict[str, Any] = {}
    for field_info in fields(TeacherConfig):
        name = field_info.name
        if name in _POTION_ONLY_TEACHER_CONFIG_FIELDS:
            continue
        value = getattr(config, name)
        if name == "player_power_weights" and isinstance(value, dict):
            payload[name] = tuple(sorted((str(key), float(raw_value)) for key, raw_value in value.items()))
        else:
            payload[name] = value
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    if len(_NON_POTION_CONFIG_KEY_CACHE) > 512:
        _NON_POTION_CONFIG_KEY_CACHE.clear()
    _NON_POTION_CONFIG_KEY_CACHE[object_key] = encoded
    return encoded


def _teacher_step_source_cache_bytes(source: bytes | _TeacherCombatStepSource | None) -> bytes:
    if isinstance(source, _TeacherCombatStepSource):
        prefix = f"{source.payload_kind}:".encode("utf-8", "replace")
        return prefix + source.payload_blob
    if isinstance(source, bytes):
        return source
    return b""


def _action_cache_payload(actions: list[dict[str, Any]]) -> bytes:
    compact_actions = []
    for action in actions:
        compact_actions.append(
            {
                "kind": action.get("kind"),
                "name": action.get("name"),
                "card_id": action.get("card_id"),
                "card_index": action.get("card_index"),
                "source_index": action.get("source_index"),
                "target_index": action.get("target_index"),
                "model_target_index": action.get("model_target_index"),
                "requires_target": action.get("requires_target"),
            }
        )
    return json.dumps(compact_actions, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _teacher_step_source_context_payload(source: bytes | _TeacherCombatStepSource | None) -> bytes:
    if not isinstance(source, _TeacherCombatStepSource):
        return b""
    payload: dict[str, Any] = {
        "has_run_env": source.run_env is not None,
        "has_combat_env": source.combat_env is not None,
        "payload_kind": source.payload_kind,
    }
    for label, obj in (("run", source.run_env), ("combat", source.combat_env)):
        if obj is None:
            continue
        payload[label] = {
            "phase": str(getattr(obj, "phase", "")),
            "floor": int(getattr(obj, "floor", 0) or 0),
            "act": int(getattr(obj, "act", 0) or 0),
            "dungeon_id": getattr(obj, "dungeon_id", None),
            "room_type": getattr(obj, "room_type", getattr(obj, "current_room_type", None)),
            "ascension_level": int(getattr(obj, "ascension_level", 0) or 0),
            "act_boss": getattr(obj, "act_boss", None),
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _step_branch_cache_source_prefix(source: bytes | _TeacherCombatStepSource | None) -> bytes | None:
    # Off by default: serializing every stepped branch is usually more
    # expensive than the cache hits it creates in full-budget sweeps.
    if not _teacher_env_flag("SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE", False):
        return None
    source_bytes = _teacher_step_source_cache_bytes(source)
    if not source_bytes:
        return None
    digest = hashlib.blake2b(digest_size=20)
    digest.update(source_bytes)
    context = _teacher_step_source_context_payload(source)
    if context:
        digest.update(b"\0context\0")
        digest.update(context)
    digest.update(b"\0fast_sync\0")
    digest.update(b"1" if _teacher_env_flag("SPIRECOMM_V3_TEACHER_FAST_BRANCH_SYNC", True) else b"0")
    return digest.digest()


def _step_branch_cache_key(
    source: bytes | _TeacherCombatStepSource | None,
    action: dict[str, Any],
    source_prefix: bytes | None = None,
) -> str | None:
    if source_prefix is None:
        source_prefix = _step_branch_cache_source_prefix(source)
    if source_prefix is None:
        return None
    digest = hashlib.blake2b(digest_size=20)
    digest.update(source_prefix)
    digest.update(b"\0action\0")
    digest.update(_action_cache_payload([action]))
    return digest.hexdigest()


def _non_potion_root_cache_key(
    *,
    config: TeacherConfig,
    source: bytes | _TeacherCombatStepSource | None,
    actions: list[dict[str, Any]],
) -> tuple[str, str] | None:
    source_bytes = _teacher_step_source_cache_bytes(source)
    if not source_bytes:
        return None
    digest = hashlib.blake2b(digest_size=20)
    digest.update(source_bytes)
    digest.update(b"\0actions\0")
    digest.update(_action_cache_payload(actions))
    return _non_potion_config_cache_key(config), digest.hexdigest()


def _get_non_potion_root_cache(key: tuple[str, str] | None) -> dict[str, Any] | None:
    if key is None:
        return None
    cached = _NON_POTION_ROOT_BEST_CACHE.get(key)
    return dict(cached) if cached is not None else None


def _put_non_potion_root_cache(key: tuple[str, str] | None, action: dict[str, Any]) -> None:
    if key is None:
        return
    max_size = _teacher_env_int("SPIRECOMM_V3_TEACHER_NON_POTION_ROOT_CACHE_SIZE", 4096)
    if max_size <= 0:
        return
    if len(_NON_POTION_ROOT_BEST_CACHE) >= max_size:
        trim_count = max(1, max_size // 8)
        for old_key in list(_NON_POTION_ROOT_BEST_CACHE.keys())[:trim_count]:
            _NON_POTION_ROOT_BEST_CACHE.pop(old_key, None)
    _NON_POTION_ROOT_BEST_CACHE[key] = dict(action)


def _state_has_identity_sensitive_card(value: Any) -> bool:
    if isinstance(value, dict):
        card_id = str(value.get("card_id") or "")
        if card_id in _IDENTITY_SENSITIVE_CARD_IDS:
            return True
        return any(_state_has_identity_sensitive_card(item) for item in value.values())
    if isinstance(value, list):
        return any(_state_has_identity_sensitive_card(item) for item in value)
    return False


def _canonical_state_for_branch_merge(value: Any, *, semantic: bool) -> Any:
    if isinstance(value, dict):
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if semantic and key == "uuid":
                continue
            payload[str(key)] = _canonical_state_for_branch_merge(item, semantic=semantic)
        return payload
    if isinstance(value, list):
        return [_canonical_state_for_branch_merge(item, semantic=semantic) for item in value]
    if isinstance(value, tuple):
        return [_canonical_state_for_branch_merge(item, semantic=semantic) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _semantic_canonical_state_and_identity_flag(value: Any) -> tuple[Any, bool]:
    """Build the semantic merge key while detecting cards that require UUIDs.

    The old path first scanned the whole state for identity-sensitive cards and
    then walked the same state again to strip UUIDs. Most states do not contain
    Genetic Algorithm / Ritual Dagger / Searing Blow, so this combines the
    common case into one exact pass and falls back to the UUID-preserving key
    only when needed.
    """

    if isinstance(value, dict):
        has_identity = str(value.get("card_id") or "") in _IDENTITY_SENSITIVE_CARD_IDS
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if key == "uuid":
                continue
            canonical_item, item_has_identity = _semantic_canonical_state_and_identity_flag(item)
            if item_has_identity:
                has_identity = True
            payload[str(key)] = canonical_item
        return payload, has_identity
    if isinstance(value, list):
        has_identity = False
        payload_list = []
        for item in value:
            canonical_item, item_has_identity = _semantic_canonical_state_and_identity_flag(item)
            if item_has_identity:
                has_identity = True
            payload_list.append(canonical_item)
        return payload_list, has_identity
    if isinstance(value, tuple):
        has_identity = False
        payload_list = []
        for item in value:
            canonical_item, item_has_identity = _semantic_canonical_state_and_identity_flag(item)
            if item_has_identity:
                has_identity = True
            payload_list.append(canonical_item)
        return payload_list, has_identity
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value, False
    return str(value), False


def _fast_branch_card_key(card: Any) -> tuple[Any, ...] | None:
    if not isinstance(card, dict):
        return (str(card),)
    cached = card.get("__teacher_branch_key")
    if isinstance(cached, tuple):
        return cached
    card_id = str(card.get("card_id") or "")
    if card_id in _IDENTITY_SENSITIVE_CARD_IDS:
        return None
    tags = card.get("tags")
    tags_key = tuple(str(tag) for tag in tags) if isinstance(tags, list) else ()
    return (
        card_id,
        card.get("name"),
        card.get("type"),
        card.get("rarity"),
        card.get("cost"),
        card.get("base_cost"),
        card.get("cost_for_turn"),
        card.get("cost_for_combat"),
        card.get("free_to_play_once"),
        card.get("upgrades"),
        card.get("misc"),
        card.get("exhausts"),
        card.get("has_target"),
        card.get("is_playable"),
        card.get("base_damage"),
        card.get("base_block"),
        card.get("base_magic"),
        card.get("ethereal"),
        card.get("innate"),
        card.get("retain"),
        card.get("color"),
        card.get("target"),
        tags_key,
    )


def _fast_branch_power_key(power: Any) -> tuple[Any, ...]:
    if isinstance(power, dict):
        cached = power.get("__teacher_branch_key")
        if isinstance(cached, tuple):
            return cached
        return (
            power.get("power_id") or power.get("id") or power.get("name"),
            power.get("amount"),
        )
    return (
        str(getattr(power, "power_id", getattr(power, "id", getattr(power, "name", power))) or ""),
        int(getattr(power, "amount", 0) or 0),
    )


def _fast_branch_potion_key(potion: Any) -> tuple[Any, ...]:
    if isinstance(potion, dict):
        cached = potion.get("__teacher_branch_key")
        if isinstance(cached, tuple):
            return cached
        return (
            potion.get("potion_id") or potion.get("id") or potion.get("name"),
            potion.get("can_use"),
            potion.get("can_discard"),
            potion.get("requires_target"),
        )
    return (
        str(getattr(potion, "potion_id", getattr(potion, "id", getattr(potion, "name", potion))) or ""),
        bool(getattr(potion, "can_use", True)),
        bool(getattr(potion, "can_discard", True)),
        bool(getattr(potion, "requires_target", False)),
    )


def _fast_branch_sort_key(value: Any) -> tuple[Any, ...]:
    """Stable ordering key for heterogeneous branch-signature tuples."""

    if value is None:
        return (0,)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int):
        return (2, int(value))
    if isinstance(value, float):
        return (3, float(value))
    if isinstance(value, str):
        return (4, value)
    if isinstance(value, (tuple, list)):
        return (5, tuple(_fast_branch_sort_key(item) for item in value))
    if isinstance(value, dict):
        return (
            6,
            tuple(
                sorted(
                    ((str(key), _fast_branch_sort_key(item)) for key, item in value.items()),
                    key=lambda item: item[0],
                )
            ),
        )
    return (7, str(value))


def _fast_branch_order_atom(value: Any) -> tuple[Any, ...]:
    """Cheap stable ordering key for known branch-signature scalar tuples."""

    if value is None:
        return (0,)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int):
        return (2, int(value))
    if isinstance(value, float):
        return (3, float(value))
    if isinstance(value, str):
        return (4, value)
    if isinstance(value, tuple):
        return (5, tuple(_fast_branch_order_atom(item) for item in value))
    if isinstance(value, list):
        return (5, tuple(_fast_branch_order_atom(item) for item in value))
    return (7, str(value))


def _fast_branch_card_order_key(card_key: tuple[Any, ...]) -> tuple[Any, ...]:
    return (repr(card_key),)


def _fast_branch_power_order_key(power_key: tuple[Any, ...]) -> tuple[Any, ...]:
    return (repr(power_key),)


def _fast_branch_monster_order_key(monster_key: tuple[Any, ...]) -> tuple[Any, ...]:
    return (repr(monster_key),)


def _fast_branch_state_merge_signature(
    state: dict[str, Any],
    *,
    phase: str,
    terminal: str | None,
    depth: int,
    used_potion_keys: set[tuple[int | None, str]] | None,
) -> tuple[Any, ...] | None:
    if str(state.get("backend") or "") != "v3":
        return None
    combat = _combat_state_view(state)
    player = _player_from_combat_view(combat)
    monsters = _monsters_from_combat_view(combat)
    hand_cards = _hand_cards_from_combat_view(combat)

    hand_key = []
    for card in hand_cards:
        card_key = _fast_branch_card_key(card)
        if card_key is None:
            return None
        hand_key.append(card_key)
    if (
        len(hand_key) > 1
        and _teacher_env_flag("SPIRECOMM_V3_TEACHER_CANONICALIZE_HAND_ORDER", True)
        and not _has_hand_order_random_sensitive_card(hand_cards)
    ):
        # Approximate semantic merge: most card decisions do not depend on
        # physical hand order.  Preserve order only for known random/index
        # sensitive hands.
        hand_key = sorted(hand_key, key=_fast_branch_card_order_key)

    screen_state = state.get("screen_state")
    screen_cards_key: tuple[Any, ...] = ()
    selected_cards_key: tuple[Any, ...] = ()
    if isinstance(screen_state, dict):
        screen_cards = []
        for card in screen_state.get("cards") or ():
            card_key = _fast_branch_card_key(card)
            if card_key is None:
                return None
            screen_cards.append(card_key)
        selected_cards = []
        for card in screen_state.get("selected_cards") or ():
            card_key = _fast_branch_card_key(card)
            if card_key is None:
                return None
            selected_cards.append(card_key)
        screen_cards_key = tuple(screen_cards)
        selected_cards_key = tuple(selected_cards)

    monster_items = [
        (
            monster.get("monster_id") or monster.get("name"),
            monster.get("current_hp"),
            monster.get("max_hp"),
            monster.get("block"),
            monster.get("intent"),
            monster.get("half_dead"),
            monster.get("is_gone"),
            monster.get("move_adjusted_damage"),
            monster.get("move_hits"),
            tuple(_fast_branch_power_key(power) for power in (monster.get("powers") or ())),
        )
        for monster in monsters
        if isinstance(monster, dict)
    ]
    if len(monster_items) > 1 and _teacher_env_flag("SPIRECOMM_V3_TEACHER_CANONICALIZE_MONSTER_ORDER", True):
        # Approximate semantic merge: target identity is already represented by
        # visible monster state, so equivalent reordered monster lists can share
        # continuation subtrees.
        monster_items = sorted(monster_items, key=_fast_branch_monster_order_key)
    monster_key = tuple(monster_items)

    return (
        "fast_v3",
        str(phase),
        terminal,
        int(depth),
        tuple(
            sorted(
                (index if index is not None else -1, str(potion_id))
                for index, potion_id in (used_potion_keys or set())
            )
        ),
        state.get("screen"),
        state.get("screen_type"),
        state.get("screen_up"),
        state.get("floor"),
        state.get("room_type"),
        state.get("current_hp"),
        state.get("max_hp"),
        state.get("gold"),
        tuple(_fast_branch_potion_key(potion) for potion in (state.get("potions") or ())),
        combat.get("turn"),
        combat.get("cards_played_this_turn"),
        combat.get("cards_discarded_this_turn"),
        player.get("current_hp"),
        player.get("max_hp"),
        player.get("block"),
        player.get("energy"),
        tuple(_fast_branch_power_key(power) for power in (player.get("powers") or ())),
        monster_key,
        tuple(hand_key),
        state.get("choice_available"),
        screen_cards_key,
        screen_state.get("num_cards") if isinstance(screen_state, dict) else None,
        screen_state.get("max_cards") if isinstance(screen_state, dict) else None,
        screen_state.get("any_number") if isinstance(screen_state, dict) else None,
        screen_state.get("can_pick_zero") if isinstance(screen_state, dict) else None,
        screen_state.get("confirm_up") if isinstance(screen_state, dict) else None,
        screen_state.get("skip_available") if isinstance(screen_state, dict) else None,
        selected_cards_key,
    )


def _branch_state_merge_signature(
    state: dict[str, Any],
    *,
    phase: str,
    terminal: str | None,
    depth: int,
    used_potion_keys: set[tuple[int | None, str]] | None,
) -> Any:
    if _teacher_env_flag("SPIRECOMM_V3_TEACHER_FAST_BRANCH_SIGNATURE", True):
        fast_key = _fast_branch_state_merge_signature(
            state,
            phase=phase,
            terminal=terminal,
            depth=depth,
            used_potion_keys=used_potion_keys,
        )
        if fast_key is not None:
            return fast_key
    semantic_requested = _teacher_env_flag("SPIRECOMM_V3_TEACHER_SEMANTIC_BRANCH_MERGE", True)
    if semantic_requested:
        canonical_state, has_identity_sensitive_card = _semantic_canonical_state_and_identity_flag(state)
        if has_identity_sensitive_card:
            canonical_state = _canonical_state_for_branch_merge(state, semantic=False)
    else:
        canonical_state = _canonical_state_for_branch_merge(state, semantic=False)
    payload = {
        "phase": str(phase),
        "terminal": terminal,
        "depth": int(depth),
        "used_potion_keys": sorted(
            (index if index is not None else -1, potion_id)
            for index, potion_id in (used_potion_keys or set())
        ),
        "state": canonical_state,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _dedupe_next_beam_entries(
    entries: list[
        tuple[
            float,
            Any,
            dict[str, Any],
            str,
            str | None,
            list[dict[str, Any]],
            int,
            float,
            set[tuple[int | None, str]] | None,
        ]
    ],
) -> list[
    tuple[
        float,
        Any,
        dict[str, Any],
        str,
        str | None,
        list[dict[str, Any]],
        int,
        float,
        set[tuple[int | None, str]] | None,
    ]
]:
    if len(entries) <= 1 or not _teacher_env_flag("SPIRECOMM_V3_TEACHER_BRANCH_STATE_MERGE", True):
        return entries
    merged: dict[
        str,
        tuple[
            float,
            Any,
            dict[str, Any],
            str,
            str | None,
            list[dict[str, Any]],
            int,
            float,
            set[tuple[int | None, str]] | None,
        ],
    ] = {}
    for entry in entries:
        priority, _branch, state, phase, terminal, _path, depth, _value, used_potion_keys = entry
        key = _branch_state_merge_signature(
            state,
            phase=phase,
            terminal=terminal,
            depth=depth,
            used_potion_keys=used_potion_keys,
        )
        previous = merged.get(key)
        if previous is None or float(priority) > float(previous[0]):
            merged[key] = entry
    return list(merged.values())


def _disable_branch_random_debug_recording(env: Any) -> None:
    """Teacher search branches do not need RNG debug traces."""

    if bool(getattr(env, "_teacher_random_debug_disabled", False)):
        return
    try:
        setattr(env, "_teacher_random_debug_disabled", True)
    except Exception:
        pass
    random_sets = [getattr(env, "randoms", None)]
    combat_env = getattr(env, "combat", None)
    if combat_env is not None and combat_env is not env:
        try:
            setattr(combat_env, "_teacher_random_debug_disabled", True)
        except Exception:
            pass
        random_sets.append(getattr(combat_env, "randoms", None))
    seen: set[int] = set()
    for randoms in random_sets:
        streams = getattr(randoms, "streams", None)
        if not isinstance(streams, dict):
            continue
        for stream in streams.values():
            stream_id = id(stream)
            if stream_id in seen:
                continue
            seen.add(stream_id)
            try:
                setattr(stream, "record_calls", False)
                calls = getattr(stream, "calls", None)
                if isinstance(calls, list):
                    calls.clear()
            except Exception:
                continue


def _copy_run_shell_for_teacher_step(run_env: Any, combat_branch: Any) -> Any:
    branch = run_env.__class__.__new__(run_env.__class__)
    branch.__dict__ = dict(getattr(run_env, "__dict__", {}))
    branch.combat = combat_branch
    branch.player = getattr(combat_branch, "player", getattr(run_env, "player", None))
    branch.randoms = getattr(combat_branch, "randoms", getattr(run_env, "randoms", None))
    branch._teacher_fast_combat_sync = True
    branch._teacher_fast_terminal_state = None
    combat_engine = getattr(combat_branch, "engine", None)
    field_sources = {
        "deck": getattr(combat_engine, "master_deck", None) if combat_engine is not None else None,
        "relics": getattr(combat_engine, "relics", None) if combat_engine is not None else None,
        "potions": getattr(combat_engine, "potions", None) if combat_engine is not None else None,
    }
    for attr, source_value in field_sources.items():
        value = source_value if isinstance(source_value, list) else getattr(run_env, attr, None)
        if isinstance(value, list):
            setattr(branch, attr, value)
    reference_sources = getattr(run_env, "reference_sources", None)
    if isinstance(reference_sources, dict):
        branch.reference_sources = reference_sources
    return branch


def _copy_combat_shell_for_teacher_step(combat_env: Any, engine_branch: Any) -> Any:
    branch = combat_env.__class__.__new__(combat_env.__class__)
    branch.__dict__ = dict(getattr(combat_env, "__dict__", {}))
    branch.engine = engine_branch
    branch.randoms = getattr(engine_branch, "randoms", getattr(combat_env, "randoms", None))
    branch.player = getattr(engine_branch, "player", getattr(combat_env, "player", None))
    engine_master_deck = getattr(engine_branch, "master_deck", None)
    engine_relics = getattr(engine_branch, "relics", None)
    engine_potions = getattr(engine_branch, "potions", None)
    branch.master_deck = engine_master_deck if isinstance(engine_master_deck, list) else list(getattr(combat_env, "master_deck", []) or [])
    branch.relics = engine_relics if isinstance(engine_relics, list) else list(getattr(combat_env, "relics", []) or [])
    branch.potions = engine_potions if isinstance(engine_potions, list) else list(getattr(combat_env, "potions", []) or [])
    branch.gold = int(getattr(engine_branch, "gold", getattr(combat_env, "gold", 0)) or 0)
    branch.state = getattr(engine_branch, "state", getattr(combat_env, "state", None))
    branch.outcome = str(getattr(engine_branch, "outcome", getattr(combat_env, "outcome", "UNDECIDED")) or "UNDECIDED")
    branch._teacher_fast_combat_sync = True
    return branch


def _step_branch_in_place(env: Any, action: dict[str, Any]) -> Any:
    _disable_branch_random_debug_recording(env)
    if getattr(env, "combat", None) is not None and str(getattr(env, "phase", "") or "") in {
        "COMBAT",
        "CARD_SELECT",
        "CARD_REWARD",
    }:
        env._step_combat(action)
    else:
        env.step(action)
    return env


def _clone_flat_mapping_for_teacher_engine(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    cloned = dict(value)
    for key, inner in list(cloned.items()):
        if isinstance(inner, list):
            cloned[key] = list(inner)
        elif isinstance(inner, dict):
            cloned[key] = dict(inner)
        elif isinstance(inner, set):
            cloned[key] = set(inner)
    return cloned


def _clone_random_set_for_teacher_engine(randoms: Any) -> Any | None:
    if not isinstance(randoms, NativeRandomSet):
        return None
    cloned = NativeRandomSet.__new__(NativeRandomSet)
    cloned.seed = int(randoms.seed)
    cloned.act = int(randoms.act)
    cloned.floor = int(randoms.floor)
    cloned.streams = {}
    for name, stream in (randoms.streams or {}).items():
        if not isinstance(stream, StsRandom):
            return None
        cloned_stream = StsRandom.__new__(StsRandom)
        cloned_stream.seed = int(stream.seed)
        cloned_stream.stream_name = str(stream.stream_name)
        cloned_stream.counter = int(stream.counter)
        cloned_stream.calls = []
        cloned_stream.record_calls = bool(getattr(stream, "record_calls", True))
        cloned_rng = RandomXS128.__new__(RandomXS128)
        cloned_rng.seed0 = int(stream._random.seed0)
        cloned_rng.seed1 = int(stream._random.seed1)
        cloned_stream._random = cloned_rng
        cloned.streams[str(name)] = cloned_stream
    return cloned


def _clone_player_for_teacher_engine(player: Any) -> PlayerState | None:
    if not isinstance(player, PlayerState):
        return None
    return PlayerState(
        current_hp=int(player.current_hp),
        max_hp=int(player.max_hp),
        block=int(player.block),
        energy=int(player.energy),
        base_energy=int(player.base_energy),
        draw_per_turn=int(player.draw_per_turn),
        powers=[_clone_flat_mapping_for_teacher_engine(power) for power in (player.powers or [])],
    )


def _clone_monster_for_teacher_engine(monster: Any) -> MonsterState | None:
    if not isinstance(monster, MonsterState):
        return None
    return MonsterState(
        monster_id=str(monster.monster_id),
        current_hp=int(monster.current_hp),
        max_hp=int(monster.max_hp),
        name=monster.name,
        block=int(monster.block),
        intent=str(monster.intent),
        powers=[_clone_flat_mapping_for_teacher_engine(power) for power in (monster.powers or [])],
        move_adjusted_damage=int(monster.move_adjusted_damage),
        move_hits=int(monster.move_hits),
        next_move=str(monster.next_move),
        move_history=list(monster.move_history or []),
        meta=dict(monster.meta or {}),
    )


def _fast_clone_engine_for_teacher_step(engine: Any) -> Any | None:
    if engine is None or not _teacher_env_flag("SPIRECOMM_V3_TEACHER_FAST_ENGINE_CLONE", False):
        return None
    state = getattr(engine, "state", None)
    if not isinstance(state, CombatState):
        return None
    if getattr(engine, "pending_card_select", None) is not None:
        return None
    # Deferred effect lists can contain object references back into state.monsters.
    # Fall back to pickle whenever such aliasing may matter.
    for key, value in getattr(engine, "__dict__", {}).items():
        if key.startswith("_pending") and value:
            return None
    if bool(getattr(engine, "_in_monster_turn", False)):
        return None
    randoms = _clone_random_set_for_teacher_engine(getattr(engine, "randoms", None))
    player = _clone_player_for_teacher_engine(getattr(engine, "player", None))
    if randoms is None or player is None:
        return None
    monsters: list[MonsterState] = []
    for monster in state.monsters or []:
        cloned_monster = _clone_monster_for_teacher_engine(monster)
        if cloned_monster is None:
            return None
        monsters.append(cloned_monster)
    cloned_state = CombatState(
        player=player,
        monsters=monsters,
        hand=[_clone_flat_mapping_for_teacher_engine(card) for card in (state.hand or [])],
        draw_pile=[_clone_flat_mapping_for_teacher_engine(card) for card in (state.draw_pile or [])],
        discard_pile=[_clone_flat_mapping_for_teacher_engine(card) for card in (state.discard_pile or [])],
        exhaust_pile=[_clone_flat_mapping_for_teacher_engine(card) for card in (state.exhaust_pile or [])],
        turn=int(state.turn),
        encounter_name=str(state.encounter_name),
        room_type=str(state.room_type),
    )
    cloned_engine = engine.__class__.__new__(engine.__class__)
    payload = dict(getattr(engine, "__dict__", {}))
    payload["randoms"] = randoms
    payload["player"] = player
    payload["state"] = cloned_state
    payload["relics"] = [_clone_flat_mapping_for_teacher_engine(relic) for relic in (getattr(engine, "relics", None) or [])]
    payload["_relic_ids"] = set(getattr(engine, "_relic_ids", set()) or set())
    payload["potions"] = [_clone_flat_mapping_for_teacher_engine(potion) for potion in (getattr(engine, "potions", None) or [])]
    payload["master_deck"] = [_clone_flat_mapping_for_teacher_engine(card) for card in (getattr(engine, "master_deck", None) or [])]
    payload["source_card_pools"] = getattr(engine, "source_card_pools", {}) or {}
    for key, value in list(payload.items()):
        if isinstance(value, set):
            payload[key] = set(value)
        elif key.startswith("_pending") and isinstance(value, list):
            payload[key] = []
    cloned_engine.__dict__.update(payload)
    return cloned_engine


def _step_branch_with_source(
    env: Any,
    source: bytes | _TeacherCombatStepSource | None,
    action: dict[str, Any],
) -> Any:
    if isinstance(source, _TeacherCombatStepSource):
        try:
            payload = (
                _fast_clone_engine_for_teacher_step(source.payload_obj)
                if source.payload_kind == "engine"
                else None
            )
            if payload is None:
                payload = pickle.loads(source.payload_blob)
            if source.payload_kind == "engine" and source.combat_env is not None:
                combat_branch = _copy_combat_shell_for_teacher_step(source.combat_env, payload)
            else:
                combat_branch = payload
            try:
                setattr(combat_branch, "_teacher_fast_combat_sync", True)
                engine = getattr(combat_branch, "engine", None)
                if engine is not None:
                    setattr(engine, "_teacher_fast_step_refresh", True)
            except Exception:
                pass
            if source.run_env is None:
                _disable_branch_random_debug_recording(combat_branch)
                combat_branch.step(action)
                return combat_branch
            branch = _copy_run_shell_for_teacher_step(source.run_env, combat_branch)
            _disable_branch_random_debug_recording(branch)
            if getattr(branch, "combat", None) is not None and str(getattr(branch, "phase", "") or "") in {
                "COMBAT",
                "CARD_SELECT",
                "CARD_REWARD",
            }:
                branch._step_combat(action)
            else:
                branch.step(action)
            return branch
        except Exception:
            return _step_branch_with_blob(env, None, action)
    return _step_branch_with_blob(env, source if isinstance(source, bytes) else None, action)


def _step_branch_with_source_cached(
    env: Any,
    source: bytes | _TeacherCombatStepSource | None,
    action: dict[str, Any],
    *,
    source_prefix: bytes | None = None,
) -> Any:
    if source_prefix is None and not _teacher_env_flag("SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE", False):
        return _step_branch_with_source(env, source, action)
    key = _step_branch_cache_key(source, action, source_prefix=source_prefix)
    if key is None:
        return _step_branch_with_source(env, source, action)
    cached = _STEP_BRANCH_BLOB_CACHE.get(key)
    if cached is not None:
        try:
            return pickle.loads(cached)
        except Exception:
            _STEP_BRANCH_BLOB_CACHE.pop(key, None)
    branch = _step_branch_with_source(env, source, action)
    try:
        branch_blob = pickle.dumps(branch, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return branch
    max_size = _teacher_env_int("SPIRECOMM_V3_TEACHER_STEP_BRANCH_CACHE_SIZE", 4096)
    if max_size <= 0:
        return branch
    if len(_STEP_BRANCH_BLOB_CACHE) >= max_size:
        trim_count = max(1, max_size // 8)
        for old_key in list(_STEP_BRANCH_BLOB_CACHE.keys())[:trim_count]:
            _STEP_BRANCH_BLOB_CACHE.pop(old_key, None)
    _STEP_BRANCH_BLOB_CACHE[key] = branch_blob
    return branch


def _step_branch_with_blob(env: Any, env_blob: bytes | None, action: dict[str, Any]) -> Any:
    fast_combat_sync = _teacher_env_flag("SPIRECOMM_V3_TEACHER_FAST_BRANCH_SYNC", True)
    if env_blob is not None:
        try:
            return step_branch_from_blob(
                env_blob,
                action,
                strip_debug_history=False,
                fast_combat_sync=fast_combat_sync,
            )
        except Exception:
            pass
    return step_branch(env, action, strip_debug_history=True, fast_combat_sync=fast_combat_sync)


def _teacher_action_forced_survival_status(
    before_state: dict[str, Any],
    branch: Any,
    visible_after: dict[str, Any],
) -> bool | None:
    """Return whether the root action is proven safe through a forced END."""

    after_phase = _phase_from_state(branch, visible_after)
    after_terminal = _is_terminal_from_state(branch, visible_after, after_phase)
    if _is_post_combat_victory_from_state(before_state, visible_after, after_terminal):
        return True
    if after_terminal == "DEFEAT" or _player_hp(visible_after) <= 0:
        return False
    if after_phase != "COMBAT":
        return True
    followup_actions = _legal_teacher_actions_from_state(branch, visible_after, after_phase)
    if len(followup_actions) != 1 or str(followup_actions[0].get("kind") or "") != "end":
        return None
    try:
        branch_source = _clone_step_source_or_none(branch)
        branch_source_prefix = _step_branch_cache_source_prefix(branch_source)
        ended = _step_branch_with_source_cached(
            branch,
            branch_source,
            followup_actions[0],
            source_prefix=branch_source_prefix,
        )
        ended_state = _scoring_state_from_env(ended)
        ended_phase = _phase_from_state(ended, ended_state)
        ended_terminal = _is_terminal_from_state(ended, ended_state, ended_phase)
    except Exception:
        return None
    if _is_post_combat_victory_from_state(visible_after, ended_state, ended_terminal):
        return True
    if ended_terminal == "DEFEAT" or _player_hp(ended_state) <= 0:
        return False
    return True


def _apply_teacher_survival_guard_to_scores(
    before_state: dict[str, Any],
    expanded_actions: list[tuple[dict[str, Any], Any, dict[str, Any]]],
    scores: list[float],
    cfg: TeacherConfig,
    status_cache: dict[int, bool | None] | None = None,
) -> tuple[list[float], dict[int, float]]:
    if not cfg.teacher_survival_guard_enabled or len(scores) <= 1:
        return scores, {}
    if len(expanded_actions) != len(scores):
        return scores, {}

    def status_for(index: int) -> bool | None:
        if status_cache is not None and index in status_cache:
            return status_cache[index]
        _action, branch, visible_after = expanded_actions[index]
        status = _teacher_action_forced_survival_status(before_state, branch, visible_after)
        if status_cache is not None:
            status_cache[index] = status
        return status

    best_index = max(range(len(scores)), key=lambda index: scores[index])
    best_status = status_for(best_index)
    if best_status is not False:
        return scores, {}
    safe_indices: list[int] = []
    for index, (action, branch, visible_after) in enumerate(expanded_actions):
        if index == best_index:
            continue
        if str(action.get("kind") or "") == "end":
            continue
        status = status_for(index)
        if status is True:
            safe_indices.append(index)
    if not safe_indices:
        return scores, {}
    adjusted = list(scores)
    safe_best = max(adjusted[index] for index in safe_indices)
    guard_floor = safe_best - abs(float(cfg.teacher_survival_guard_score_margin))
    adjustments: dict[int, float] = {}
    if cfg.teacher_survival_guard_restrict_safe:
        safe_set = set(safe_indices)
        for index in range(len(adjusted)):
            if index in safe_set:
                continue
            new_score = min(adjusted[index], guard_floor)
            if new_score != adjusted[index]:
                adjustments[index] = float(new_score - adjusted[index])
                adjusted[index] = float(new_score)
    else:
        new_score = min(adjusted[best_index], guard_floor)
        if new_score != adjusted[best_index]:
            adjustments[best_index] = float(new_score - adjusted[best_index])
            adjusted[best_index] = float(new_score)
    return adjusted, adjustments


def _root_turn_lethal_available(env: Any, *, config: TeacherConfig | None = None) -> bool:
    cfg = config or default_teacher_config()
    root_turn = _combat_turn(env)
    initial_branches: list[tuple[dict[str, Any], Any, dict[str, Any] | None]] = []
    env_blob = _clone_step_source_or_none(env)
    env_blob_prefix = _step_branch_cache_source_prefix(env_blob)
    for action in _legal_teacher_actions(env):
        if (action.get("kind") or "") == "end":
            continue
        try:
            initial_branches.append(
                (
                    action,
                    _step_branch_with_source_cached(env, env_blob, action, source_prefix=env_blob_prefix),
                    None,
                )
            )
        except Exception:
            continue
    return _root_turn_lethal_available_from_branches(initial_branches, root_turn=root_turn, config=cfg)


def _root_turn_lethal_available_from_branches(
    initial_branches: list[tuple[dict[str, Any], Any, dict[str, Any] | None]],
    *,
    root_turn: int,
    config: TeacherConfig | None = None,
) -> bool:
    cfg = config or default_teacher_config()
    stack: list[tuple[Any, dict[str, Any], str, str | None, int]] = []
    seen_signatures: set[str] = set()
    nodes = 0
    for action, branch, branch_state in initial_branches:
        if (action.get("kind") or "") == "end":
            continue
        if branch_state is None:
            branch_state = _scoring_state_from_env(branch)
        branch_phase = _phase_from_state(branch, branch_state)
        branch_terminal = _is_terminal_from_state(branch, branch_state, branch_phase)
        nodes += 1
        if branch_terminal == "VICTORY":
            return True
        signature = _branch_state_merge_signature(
            branch_state,
            phase=branch_phase,
            terminal=branch_terminal,
            depth=1,
            used_potion_keys=None,
        )
        if signature in seen_signatures:
            if nodes >= cfg.lethal_check_node_budget:
                break
            continue
        seen_signatures.add(signature)
        stack.append((branch, branch_state, branch_phase, branch_terminal, 1))
        if nodes >= cfg.lethal_check_node_budget:
            break
    while stack and nodes < cfg.lethal_check_node_budget:
        current_env, current_state, phase, terminal, depth = stack.pop()
        if terminal == "VICTORY":
            return True
        if (
            terminal is not None
            or _is_next_player_decision_from_state(current_state, phase=phase, root_turn=root_turn)
            or depth >= cfg.max_depth
        ):
            continue
        actions = _legal_teacher_actions_from_state(current_env, current_state, phase)
        if not actions:
            continue
        current_env_blob = _clone_step_source_or_none(current_env)
        current_env_blob_prefix = _step_branch_cache_source_prefix(current_env_blob)
        for action in actions:
            if (action.get("kind") or "") == "end":
                continue
            try:
                branch = _step_branch_with_source_cached(
                    current_env,
                    current_env_blob,
                    action,
                    source_prefix=current_env_blob_prefix,
                )
            except Exception:
                continue
            branch_state = _scoring_state_from_env(branch)
            branch_phase = _phase_from_state(branch, branch_state)
            branch_terminal = _is_terminal_from_state(branch, branch_state, branch_phase)
            nodes += 1
            if branch_terminal == "VICTORY":
                return True
            signature = _branch_state_merge_signature(
                branch_state,
                phase=branch_phase,
                terminal=branch_terminal,
                depth=depth + 1,
                used_potion_keys=None,
            )
            if signature in seen_signatures:
                if nodes >= cfg.lethal_check_node_budget:
                    break
                continue
            seen_signatures.add(signature)
            stack.append((branch, branch_state, branch_phase, branch_terminal, depth + 1))
            if nodes >= cfg.lethal_check_node_budget:
                break
    return False


def _canonical_power_id(power_id: str) -> str:
    power_id = str(power_id)
    return TEACHER_POWER_ID_ALIASES.get(power_id, power_id)


def _power_amounts(powers: list[dict[str, Any]] | None) -> dict[str, float]:
    if not powers:
        return {}
    totals: dict[str, float] = {}
    aliases = TEACHER_POWER_ID_ALIASES
    positive_sentinel = NEGATIVE_SENTINEL_POSITIVE_POWERS
    debuff_sentinel = NEGATIVE_SENTINEL_DEBUFF_POWERS
    for power in powers:
        raw_id = str(power.get("power_id") or power.get("id") or power.get("name") or "")
        power_id = aliases.get(raw_id)
        if power_id is None:
            power_id = raw_id
        try:
            amount = float(power.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        if power_id in positive_sentinel:
            amount = abs(amount) if amount else 1.0
        elif power_id in debuff_sentinel:
            amount = abs(amount) if amount else 1.0
        totals[power_id] = totals.get(power_id, 0.0) + amount
    return totals


def _player_power_amounts(state: dict[str, Any]) -> dict[str, float]:
    player = _combat_state_view(state).get("player")
    if not isinstance(player, dict):
        return {}
    return _power_amounts(player.get("powers"))


def _monster_power_amounts(monster: dict[str, Any]) -> dict[str, float]:
    return _power_amounts(monster.get("powers"))


def _player_has_any_power(state: dict[str, Any]) -> bool:
    player = _combat_state_view(state).get("player")
    return isinstance(player, dict) and bool(player.get("powers"))


def _monsters_have_any_power(state: dict[str, Any]) -> bool:
    for monster in _monsters_view(state):
        if isinstance(monster, dict) and monster.get("powers"):
            return True
    return False


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _monster_transition_stats(
    before_mons: list[dict[str, Any]],
    after_mons: list[dict[str, Any]],
    *,
    victory_after: bool,
    after_has_combat: bool,
) -> tuple[int, int, int, int, int, bool, bool]:
    before_hp_total = 0
    after_hp_total = 0
    kills = 0
    incoming_before = 0
    incoming_after = 0
    before_has_power = False
    after_has_power = False

    for index, before_monster in enumerate(before_mons):
        if before_monster.get("powers"):
            before_has_power = True
        if not _is_alive(before_monster):
            continue
        before_hp_total += max(0, _int(before_monster.get("current_hp")))
        if (before_monster.get("intent") or "") in ATTACK_INTENTS:
            try:
                damage = int(before_monster.get("move_adjusted_damage") or 0)
                hits = int(before_monster.get("move_hits") or 1)
            except (TypeError, ValueError):
                damage = 0
                hits = 1
            incoming_before += max(0, damage) * max(1, hits)
        if victory_after:
            kills += 1
            continue
        after_monster = after_mons[index] if index < len(after_mons) else {}
        if not after_monster or not _is_alive(after_monster):
            kills += 1

    if after_has_combat:
        for after_monster in after_mons:
            if after_monster.get("powers"):
                after_has_power = True
            if not _is_alive(after_monster):
                continue
            after_hp_total += max(0, _int(after_monster.get("current_hp")))
            if (after_monster.get("intent") or "") not in ATTACK_INTENTS:
                continue
            try:
                damage = int(after_monster.get("move_adjusted_damage") or 0)
                hits = int(after_monster.get("move_hits") or 1)
            except (TypeError, ValueError):
                continue
            incoming_after += max(0, damage) * max(1, hits)

    return (
        before_hp_total,
        after_hp_total,
        kills,
        incoming_before,
        incoming_after,
        before_has_power,
        after_has_power,
    )


def _is_post_combat_victory(before_state: dict[str, Any], after_state: dict[str, Any], after_env: Any) -> bool:
    if _is_terminal(after_env) == "VICTORY":
        return True
    before_phase = str(before_state.get("phase") or "")
    after_phase = str(after_state.get("phase") or "")
    return before_phase == "COMBAT" and after_phase in POST_COMBAT_PHASES and not _combat_state_view(after_state)


def _is_post_combat_victory_from_state(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    after_terminal: str | None,
) -> bool:
    if after_terminal == "VICTORY":
        return True
    before_phase = str(before_state.get("phase") or "")
    after_phase = str(after_state.get("phase") or "")
    return before_phase == "COMBAT" and after_phase in POST_COMBAT_PHASES and not _combat_state_view(after_state)


def _alive_hp_total(state: dict[str, Any]) -> int:
    return _alive_hp_total_from_monsters(_monsters_view(state))


def _alive_hp_total_from_monsters(monsters: list[dict[str, Any]]) -> int:
    return sum(max(0, _int(monster.get("current_hp"))) for monster in monsters if _is_alive(monster))


def _alive_monster_count(state: dict[str, Any]) -> int:
    return sum(1 for monster in _monsters_view(state) if _is_alive(monster))


def _monster_kills(before_state: dict[str, Any], after_state: dict[str, Any], *, victory_after: bool) -> int:
    return _monster_kills_from_monsters(
        _monsters_view(before_state),
        _monsters_view(after_state),
        victory_after=victory_after,
    )


def _monster_kills_from_monsters(
    before_mons: list[dict[str, Any]],
    after_mons: list[dict[str, Any]],
    *,
    victory_after: bool,
) -> int:
    kills = 0
    for index, before_monster in enumerate(before_mons):
        if not _is_alive(before_monster):
            continue
        if victory_after:
            kills += 1
            continue
        after_monster = after_mons[index] if index < len(after_mons) else {}
        if not after_monster or not _is_alive(after_monster):
            kills += 1
    return kills


def _player(state: dict[str, Any]) -> dict[str, Any]:
    player = _combat_state_view(state).get("player")
    return player if isinstance(player, dict) else {}


def _player_hp(state: dict[str, Any]) -> int:
    player = _player(state)
    return _int(state.get("current_hp"), _int(player.get("current_hp")))


def _player_block(state: dict[str, Any]) -> int:
    return max(0, _int(_player(state).get("block")))


def _player_energy(state: dict[str, Any]) -> int:
    return _int(_player(state).get("energy"))


def _fight_remaining(state: dict[str, Any]) -> float:
    live = [monster for monster in _monsters_view(state) if _is_alive(monster)]
    max_hp = sum(max(1, _int(monster.get("max_hp"), 1)) for monster in live)
    if max_hp <= 0:
        return 0.0
    hp = sum(max(0, _int(monster.get("current_hp"))) for monster in live)
    return max(0.0, min(1.0, float(hp) / float(max_hp)))


def _potion_room_reward_factor(state: dict[str, Any], config: TeacherConfig) -> float:
    room_type = str(state.get("room_type") or "")
    if room_type == "MonsterRoomElite":
        return float(config.potion_elite_room_reward_factor)
    if room_type == "MonsterRoomBoss":
        return float(config.potion_boss_room_reward_factor)
    if room_type in {"MonsterRoom", "EventRoom"}:
        return float(config.potion_monster_room_reward_factor)
    return 1.0


def _potion_id(before_state: dict[str, Any], action: dict[str, Any]) -> str:
    potion = _selected_potion(before_state, action)
    return str(action.get("potion_id") or (potion or {}).get("potion_id") or (potion or {}).get("id") or (potion or {}).get("name") or "")


def _action_potion_id(action: dict[str, Any]) -> str:
    potion = action.get("potion")
    if not isinstance(potion, dict):
        potion = {}
    return str(action.get("potion_id") or potion.get("potion_id") or potion.get("id") or potion.get("name") or "")


def _action_potion_index(action: dict[str, Any]) -> int | None:
    index = action.get("potion_index")
    if index is None:
        return None
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def _is_same_potion_action(action: dict[str, Any], blocked_potion_action: dict[str, Any] | None) -> bool:
    if blocked_potion_action is None or (action.get("kind") or "") != "potion":
        return False
    blocked_index = _action_potion_index(blocked_potion_action)
    action_index = _action_potion_index(action)
    blocked_id = _action_potion_id(blocked_potion_action)
    action_id = _action_potion_id(action)
    if blocked_index is not None and action_index is not None and blocked_index != action_index:
        return False
    if blocked_id and action_id and blocked_id != action_id:
        return False
    return blocked_index is not None or bool(blocked_id)


def _missing_hp(state: dict[str, Any]) -> int:
    return max(0, _int(state.get("max_hp")) - _player_hp(state))


def _uncovered_incoming(state: dict[str, Any]) -> int:
    return max(0, _incoming_damage_fast(state) - _player_block(state))


def _deferred_effective_block(
    root_state: dict[str, Any] | None,
    end_state: dict[str, Any],
    *,
    lethal_available: bool,
    cfg: TeacherConfig,
) -> float:
    if root_state is None:
        return 0.0
    final_incoming = float(_incoming_damage_fast(end_state))
    if final_incoming <= 0.0:
        return 0.0
    root_block = float(_player_block(root_state))
    end_block = float(_player_block(end_state))
    effective_block = max(0.0, min(end_block, final_incoming) - min(root_block, final_incoming))
    if lethal_available and cfg.suppress_block_reward_when_lethal_available:
        suppression_factor = float(cfg.lethal_block_suppression_factor)
        if (
            cfg.lethal_block_low_hp_protection
            and effective_block > 0.0
            and _player_hp(root_state) <= int(cfg.lethal_block_low_hp_max)
        ):
            root_uncovered = _uncovered_incoming(root_state)
            end_uncovered = _uncovered_incoming(end_state)
            facing_lethal = root_uncovered >= _player_hp(root_state)
            improves_survival = end_uncovered < root_uncovered
            if improves_survival and (facing_lethal or not cfg.lethal_block_low_hp_requires_facing_lethal):
                suppression_factor = float(cfg.lethal_block_low_hp_suppression_factor)
        return effective_block * suppression_factor
    return effective_block


def _enemy_attack_hit_count(state: dict[str, Any]) -> int:
    total = 0
    for monster in _monsters_view(state):
        if not _is_alive(monster):
            continue
        if str(monster.get("intent") or "") not in {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}:
            continue
        if _int(monster.get("move_adjusted_damage")) <= 0:
            continue
        total += max(1, _int(monster.get("move_hits"), 1))
    return total


def _empty_potion_slot_count_after_use(before_state: dict[str, Any], action: dict[str, Any]) -> int:
    consumed_index = action.get("potion_index")
    try:
        consumed_index = int(consumed_index)
    except (TypeError, ValueError):
        consumed_index = -1
    count = 0
    for index, potion in enumerate(before_state.get("potions") or ()):
        potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
        if potion_id == "Potion Slot" or index == consumed_index:
            count += 1
    return count


def _potion_cost(before_state: dict[str, Any], action: dict[str, Any], config: TeacherConfig) -> float:
    potion_id = _potion_id(before_state, action)
    if potion_id == "BloodPotion":
        base_cost = float(max(1, _int(before_state.get("max_hp")))) * 0.2
    else:
        base_cost = float(POTION_COSTS.get(potion_id, 0.0))
    return float(base_cost) * float(config.potion_cost_scale)


def _potion_reward_adjustment(before_state: dict[str, Any], action: dict[str, Any], config: TeacherConfig) -> float:
    """Return the delta between the generic transition reward and potion-specific direct reward."""

    potion_id = _potion_id(before_state, action)
    power = _fight_remaining(before_state)
    uncovered = float(_uncovered_incoming(before_state))
    scale = float(config.potion_buff_adjustment_scale)
    if potion_id == "Strength Potion":
        return (8.0 * power - 8.0) * scale
    if potion_id == "SteroidPotion":
        return -10.0 * scale
    if potion_id == "Dexterity Potion":
        return 6.0 * power * scale
    if potion_id == "Ancient Potion":
        return (4.0 * power - 4.0) * scale
    if potion_id == "DuplicationPotion":
        return -8.0 * scale
    if potion_id == "EssenceOfSteel":
        return min(4.0, uncovered) * power * scale
    if potion_id == "LiquidBronze":
        return float(_enemy_attack_hit_count(before_state)) * 3.0 * 0.8 * power * scale
    if potion_id == "Regen Potion":
        return min(float(_missing_hp(before_state)), 15.0) * power * scale
    if potion_id == "HeartOfIron":
        return (min(6.0, uncovered) * power - 16.0) * scale
    if potion_id == "CultistPotion":
        return 10.0 * power * scale
    if potion_id == "BloodPotion":
        return float(_missing_hp(before_state)) * scale
    if potion_id == "Fruit Juice":
        return 10.0 * scale
    if potion_id == "EntropicBrew":
        return (
            float(_empty_potion_slot_count_after_use(before_state, action))
            * 5.0
            * float(config.potion_generation_adjustment_scale)
        )
    return 0.0


def _target_hp_fraction(monster: dict[str, Any]) -> float:
    max_hp = max(1, _int(monster.get("max_hp"), 1))
    return max(0.0, min(1.0, float(max(0, _int(monster.get("current_hp")))) / float(max_hp)))


def _cards_played_this_turn(env: Any) -> int:
    candidates = [
        getattr(getattr(getattr(env, "combat", None), "engine", None), "cards_played_this_turn", None),
        getattr(getattr(env, "engine", None), "cards_played_this_turn", None),
    ]
    for value in candidates:
        if value is None:
            continue
        return max(0, _int(value))
    return 0


def _turn_order_factor(env: Any, config: TeacherConfig) -> float:
    return max(0.0, 1.0 - float(config.turn_order_decay_per_card) * float(_cards_played_this_turn(env)))


def _cards_played_this_turn_from_state(state: dict[str, Any], env: Any | None = None) -> int:
    combat = _combat_state_view(state)
    value = combat.get("cards_played_this_turn")
    if value is None:
        value = state.get("cards_played_this_turn")
    if value is not None:
        return max(0, _int(value))
    return _cards_played_this_turn(env) if env is not None else 0


def _turn_order_factor_from_state(state: dict[str, Any], env: Any | None, config: TeacherConfig) -> float:
    return max(
        0.0,
        1.0 - float(config.turn_order_decay_per_card) * float(_cards_played_this_turn_from_state(state, env)),
    )


def _scaled_power_delta_value(
    *,
    before_amount: float,
    after_amount: float,
    power_id: str,
    weight: float,
    multiplier: float = 1.0,
) -> float:
    base = max(1e-6, float(POWER_BASE_AMOUNTS.get(power_id, 1.0)))
    delta = float(after_amount) - float(before_amount)
    return float(weight) * delta / base * float(multiplier)


def _player_power_delta_value(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    *,
    config: TeacherConfig,
    victory_after: bool,
) -> float:
    before_powers = _player_power_amounts(before_state)
    after_powers = before_powers if victory_after and not _combat_state_view(after_state) else _player_power_amounts(after_state)
    value = 0.0
    weights = config.player_power_weights
    for power_id in set(before_powers) | set(after_powers):
        weight = weights.get(power_id)
        if weight is None:
            continue
        value += _scaled_power_delta_value(
            before_amount=float(before_powers.get(power_id, 0.0)),
            after_amount=float(after_powers.get(power_id, 0.0)),
            power_id=power_id,
            weight=weight,
        )
    return value


def _player_power_delta_features(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    *,
    victory_after: bool,
) -> tuple[tuple[str, float], ...]:
    before_powers = _player_power_amounts(before_state)
    after_powers = before_powers if victory_after and not _combat_state_view(after_state) else _player_power_amounts(after_state)
    features: list[tuple[str, float]] = []
    for power_id in set(before_powers) | set(after_powers):
        base = max(1e-6, float(POWER_BASE_AMOUNTS.get(power_id, 1.0)))
        delta = float(after_powers.get(power_id, 0.0)) - float(before_powers.get(power_id, 0.0))
        features.append((str(power_id), float(delta / base)))
    return tuple(features)


def _player_power_delta_value_from_features(
    features: tuple[tuple[str, float], ...],
    config: TeacherConfig,
) -> float:
    return _player_power_delta_value_from_features_and_weights(features, config.player_power_weights)


def _player_power_delta_value_from_features_and_weights(
    features: tuple[tuple[str, float], ...],
    weights: dict[str, float],
) -> float:
    value = 0.0
    for power_id, normalized_delta in features:
        weight = weights.get(power_id)
        if weight is None:
            continue
        value += float(weight) * float(normalized_delta)
    return float(value)


def _monster_power_weights(config: TeacherConfig) -> dict[str, float]:
    return {
        "Vulnerable": float(config.monster_vulnerable_weight),
        "Weakened": float(config.monster_weakened_weight),
        "Strength": float(config.monster_strength_weight),
        "Shackled": float(config.monster_shackled_weight),
    }


def _monster_power_weight_items(config: TeacherConfig) -> tuple[tuple[str, float], ...]:
    return (
        ("Vulnerable", float(config.monster_vulnerable_weight)),
        ("Weakened", float(config.monster_weakened_weight)),
        ("Strength", float(config.monster_strength_weight)),
        ("Shackled", float(config.monster_shackled_weight)),
    )


def _monster_power_delta_value(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    *,
    config: TeacherConfig,
    victory_after: bool,
) -> float:
    if victory_after and not _combat_state_view(after_state):
        return 0.0
    before_mons = _monsters_view(before_state)
    after_mons = _monsters_view(after_state)
    value = 0.0
    for index, before_monster in enumerate(before_mons):
        if not _is_alive(before_monster):
            continue
        after_monster = after_mons[index] if index < len(after_mons) else {}
        before_powers = _monster_power_amounts(before_monster)
        after_powers = _monster_power_amounts(after_monster)
        hp_factor = _target_hp_fraction(before_monster)
        for power_id, weight in _monster_power_weight_items(config):
            before_amount = float(before_powers.get(power_id, 0.0))
            after_amount = float(after_powers.get(power_id, 0.0))
            if power_id == "Strength":
                delta = after_amount - before_amount
                value += -float(weight) * delta / max(1e-6, POWER_BASE_AMOUNTS.get(power_id, 1.0)) * hp_factor
            elif power_id == "Shackled":
                delta = after_amount - before_amount
                value += -float(weight) * delta / max(1e-6, POWER_BASE_AMOUNTS.get(power_id, 1.0)) * hp_factor
            else:
                value += _scaled_power_delta_value(
                    before_amount=before_amount,
                    after_amount=after_amount,
                    power_id=power_id,
                    weight=weight,
                    multiplier=hp_factor,
                )
    return value


def _selected_card_type(before_state: dict[str, Any], action: dict[str, Any]) -> str:
    action_card_type = action.get("card_type")
    if action_card_type is not None:
        return str(action_card_type or "")
    card = _selected_card(before_state, action)
    if not isinstance(card, dict):
        card = action
    return str((card or {}).get("type") or "")


def _selected_card_id(before_state: dict[str, Any], action: dict[str, Any]) -> str:
    action_card_id = action.get("card_id")
    if action_card_id is not None:
        return str(action_card_id or "")
    card = _selected_card(before_state, action)
    return str(action.get("card_id") or (card or {}).get("card_id") or (card or {}).get("name") or "")


def transition_reward_components(
    before_env: Any,
    before_state: dict[str, Any],
    action: dict[str, Any],
    after_env: Any,
    after_state: dict[str, Any],
    config: TeacherConfig | None = None,
    *,
    lethal_available: bool = False,
    block_reward_base_state: dict[str, Any] | None = None,
    after_phase: str | None = None,
    after_terminal: str | None = None,
) -> dict[str, float]:
    cfg = config or default_teacher_config()
    if after_phase is None:
        after_phase = _phase_from_state(after_env, after_state)
    if after_terminal is None:
        after_terminal = _is_terminal_from_state(after_env, after_state, after_phase)
    victory_after = _is_post_combat_victory_from_state(before_state, after_state, after_terminal)
    defeat_after = after_terminal == "DEFEAT"

    before_combat = _combat_state_view(before_state)
    after_combat = _combat_state_view(after_state)
    before_monsters = _monsters_from_combat_view(before_combat)
    after_monsters = _monsters_from_combat_view(after_combat)
    before_player = _player_from_combat_view(before_combat)
    after_player = _player_from_combat_view(after_combat)
    after_has_combat = bool(after_combat)

    (
        before_hp_total,
        raw_after_hp_total,
        monster_kill_count,
        incoming_before_int,
        incoming_after_int,
        before_monsters_have_power,
        after_monsters_have_power,
    ) = _monster_transition_stats(
        before_monsters,
        after_monsters,
        victory_after=victory_after,
        after_has_combat=after_has_combat,
    )
    after_hp_total = 0 if victory_after and not after_has_combat else raw_after_hp_total
    hp_damage = max(0.0, float(before_hp_total - after_hp_total))
    monster_kill = float(monster_kill_count)
    incoming_after = float(incoming_after_int)
    incoming_before = float(incoming_before_int)
    effective_block = 0.0
    if (action.get("kind") or "") == "end":
        effective_block = _deferred_effective_block(
            block_reward_base_state,
            before_state,
            lethal_available=lethal_available,
            cfg=cfg,
        )
    # incoming_damage is enemy-side raw incoming, independent of player block.
    raw_incoming_damage_reduction = max(0.0, incoming_before - incoming_after)

    action_kind = action.get("kind") or ""
    is_card_action = action_kind == "card"
    is_potion_action = action_kind == "potion"
    energy_spent = 0.0
    hp_loss = 0.0
    playable_hand_count_delta = 0.0
    if is_card_action:
        hp_loss = max(0.0, float(_player_hp(before_state) - _player_hp(after_state)))
        energy_spent = max(0.0, float(_player_energy(before_state) - _player_energy(after_state)))
        playable_before = _playable_count_fast(before_state)
        playable_after = _playable_count_fast(after_state) if _combat_state_view(after_state) else 0
        # Reward newly unlocked playable cards from draw/cost effects, but do
        # not punish normal card play for spending the last available energy.
        playable_hand_count_delta = max(0.0, float(playable_after - max(0, playable_before - 1)))

    after_has_combat = bool(_combat_state_view(after_state))
    player_power_delta = 0.0
    if (not victory_after or after_has_combat) and (
        _player_has_any_power(before_state) or _player_has_any_power(after_state)
    ):
        player_power_delta = _player_power_delta_value(
            before_state,
            after_state,
            config=cfg,
            victory_after=victory_after,
        )
    monster_power_delta = 0.0
    if (not victory_after or after_has_combat) and (
        _monsters_have_any_power(before_state) or _monsters_have_any_power(after_state)
    ):
        monster_power_delta = _monster_power_delta_value(
            before_state,
            after_state,
            config=cfg,
            victory_after=victory_after,
        )

    components: dict[str, float] = {
        "hp_damage": float(cfg.hp_damage_weight * hp_damage),
        "monster_kill": float(cfg.monster_kill_weight * monster_kill),
        "combat_win": float(cfg.combat_win_weight if victory_after else 0.0),
        "death": float(cfg.death_weight if defeat_after else 0.0),
        "hp_loss": float(cfg.hp_loss_weight * hp_loss),
        "effective_block": float(cfg.effective_block_weight * effective_block),
        "raw_incoming_damage_reduction": float(
            cfg.raw_incoming_damage_reduction_weight * raw_incoming_damage_reduction
        ),
        "energy_spent": float(cfg.energy_spent_weight * energy_spent),
        "playable_hand_count_delta": float(cfg.playable_hand_count_delta_weight * playable_hand_count_delta),
        "player_power_delta": float(player_power_delta),
        "monster_power_delta": float(monster_power_delta),
        "play_card_constant": 0.0,
        "power_card_constant": 0.0,
        "skill_power_turn_constant": 0.0,
        "potion_adjustment": 0.0,
        "potion_room_adjustment": 0.0,
        "potion_cost": 0.0,
    }

    if is_card_action:
        card_type = _selected_card_type(before_state, action)
        card_id = _selected_card_id(before_state, action)
        if card_type in CARD_ACTION_TYPES:
            components["play_card_constant"] = float(cfg.play_card_constant)
        if card_type == "POWER":
            components["power_card_constant"] = float(cfg.power_card_constant * _fight_remaining(before_state))
        if card_id in TURN_ORDER_SKILL_POWER_CARD_IDS:
            components["skill_power_turn_constant"] = float(
                cfg.skill_power_turn_constant * _turn_order_factor_from_state(before_state, before_env, cfg)
            )

    value = float(sum(components.values()))
    if is_potion_action:
        potion_adjustment = float(_potion_reward_adjustment(before_state, action, cfg))
        components["potion_adjustment"] = potion_adjustment
        value += potion_adjustment
        if value > 0.0:
            room_adjustment = float(value * (_potion_room_reward_factor(before_state, cfg) - 1.0))
            components["potion_room_adjustment"] = room_adjustment
            value += room_adjustment
        potion_cost = float(-_potion_cost(before_state, action, cfg))
        components["potion_cost"] = potion_cost
        value += potion_cost

    components["immediate_total"] = float(value)
    return components


def transition_reward(
    before_env: Any,
    before_state: dict[str, Any],
    action: dict[str, Any],
    after_env: Any,
    after_state: dict[str, Any],
    config: TeacherConfig | None = None,
    *,
    lethal_available: bool = False,
    block_reward_base_state: dict[str, Any] | None = None,
    after_phase: str | None = None,
    after_terminal: str | None = None,
) -> float:
    cfg = config or default_teacher_config()
    if after_phase is None:
        after_phase = _phase_from_state(after_env, after_state)
    if after_terminal is None:
        after_terminal = _is_terminal_from_state(after_env, after_state, after_phase)
    victory_after = _is_post_combat_victory_from_state(before_state, after_state, after_terminal)
    defeat_after = after_terminal == "DEFEAT"

    before_combat = _combat_state_view(before_state)
    after_combat = _combat_state_view(after_state)
    before_monsters = _monsters_from_combat_view(before_combat)
    after_monsters = _monsters_from_combat_view(after_combat)
    before_player = _player_from_combat_view(before_combat)
    after_player = _player_from_combat_view(after_combat)
    after_has_combat = bool(after_combat)

    (
        before_hp_total,
        raw_after_hp_total,
        monster_kill_count,
        incoming_before_int,
        incoming_after_int,
        before_monsters_have_power,
        after_monsters_have_power,
    ) = _monster_transition_stats(
        before_monsters,
        after_monsters,
        victory_after=victory_after,
        after_has_combat=after_has_combat,
    )
    after_hp_total = 0 if victory_after and not after_has_combat else raw_after_hp_total
    hp_damage = max(0.0, float(before_hp_total - after_hp_total))
    monster_kill = float(monster_kill_count)
    incoming_after = float(incoming_after_int)
    incoming_before = float(incoming_before_int)
    effective_block = 0.0
    if str(action.get("kind") or "") == "end":
        effective_block = _deferred_effective_block(
            block_reward_base_state,
            before_state,
            lethal_available=lethal_available,
            cfg=cfg,
        )
    raw_incoming_damage_reduction = max(0.0, incoming_before - incoming_after)

    action_kind = action.get("kind") or ""
    is_card_action = action_kind == "card"
    is_potion_action = action_kind == "potion"
    energy_spent = 0.0
    hp_loss = 0.0
    playable_hand_count_delta = 0.0
    if is_card_action:
        before_player_hp = _int(before_state.get("current_hp"), _int(before_player.get("current_hp")))
        after_player_hp = _int(after_state.get("current_hp"), _int(after_player.get("current_hp")))
        hp_loss = max(0.0, float(before_player_hp - after_player_hp))
        energy_spent = max(0.0, float(_int(before_player.get("energy")) - _int(after_player.get("energy"))))
        playable_before = _playable_count_from_hand(_hand_cards_from_combat_view(before_combat))
        playable_after = _playable_count_from_hand(_hand_cards_from_combat_view(after_combat)) if after_has_combat else 0
        playable_hand_count_delta = max(0.0, float(playable_after - max(0, playable_before - 1)))

    value = 0.0
    value += float(cfg.hp_damage_weight * hp_damage)
    value += float(cfg.monster_kill_weight * monster_kill)
    value += float(cfg.combat_win_weight if victory_after else 0.0)
    value += float(cfg.death_weight if defeat_after else 0.0)
    value += float(cfg.hp_loss_weight * hp_loss)
    value += float(cfg.effective_block_weight * effective_block)
    value += float(cfg.raw_incoming_damage_reduction_weight * raw_incoming_damage_reduction)
    value += float(cfg.energy_spent_weight * energy_spent)
    value += float(cfg.playable_hand_count_delta_weight * playable_hand_count_delta)
    if (not victory_after or after_has_combat) and (
        bool(before_player.get("powers")) or bool(after_player.get("powers"))
    ):
        value += float(_player_power_delta_value(before_state, after_state, config=cfg, victory_after=victory_after))
    if (not victory_after or after_has_combat) and (
        before_monsters_have_power or after_monsters_have_power
    ):
        value += float(_monster_power_delta_value(before_state, after_state, config=cfg, victory_after=victory_after))

    if is_card_action:
        card_type = _selected_card_type(before_state, action)
        card_id = _selected_card_id(before_state, action)
        if card_type in CARD_ACTION_TYPES:
            value += float(cfg.play_card_constant)
        if card_type == "POWER":
            value += float(cfg.power_card_constant * _fight_remaining(before_state))
        if card_id in TURN_ORDER_SKILL_POWER_CARD_IDS:
            value += float(cfg.skill_power_turn_constant * _turn_order_factor_from_state(before_state, before_env, cfg))

    if is_potion_action:
        value += float(_potion_reward_adjustment(before_state, action, cfg))
        if value > 0.0:
            value += float(value * (_potion_room_reward_factor(before_state, cfg) - 1.0))
        value += float(-_potion_cost(before_state, action, cfg))

    return float(value)


def _root_potion_continuation_value(
    action: dict[str, Any],
    before_state: dict[str, Any],
    immediate_reward: float,
    continuation_value: float,
    config: TeacherConfig,
    *,
    non_potion_baseline: float = 0.0,
) -> float:
    if (action.get("kind") or "") != "potion":
        return float(continuation_value)
    potion_total = float(immediate_reward) + float(continuation_value)
    marginal = potion_total - float(non_potion_baseline)
    if marginal > 0.0:
        return float(continuation_value) + (_potion_room_reward_factor(before_state, config) - 1.0) * marginal
    return float(continuation_value)


def _complete_reward_components(
    immediate_components: dict[str, float],
    *,
    action: dict[str, Any],
    immediate_reward: float,
    continuation_raw: float,
    continuation_adjusted: float,
    non_potion_baseline: float,
    teacher_q: float,
) -> dict[str, float]:
    components = {str(key): float(value) for key, value in dict(immediate_components).items()}
    components["immediate_total"] = float(immediate_reward)
    components["continuation_raw"] = float(continuation_raw)
    components["continuation_adjusted"] = float(continuation_adjusted)
    components["potion_continuation_room_bonus"] = float(continuation_adjusted - continuation_raw)
    components["non_potion_baseline"] = float(non_potion_baseline)
    components["potion_marginal"] = (
        float(immediate_reward + continuation_raw - non_potion_baseline)
        if (action.get("kind") or "") == "potion"
        else 0.0
    )
    components["teacher_q"] = float(teacher_q)
    return components


def _continuation_uses_same_potion(result: ContinuationResult, blocked_potion_action: dict[str, Any]) -> bool:
    if result.used_potion_keys:
        return (_action_potion_index(blocked_potion_action), _action_potion_id(blocked_potion_action)) in result.used_potion_keys
    return any(_is_same_potion_action(action, blocked_potion_action) for action in result.debug_best_line)


def _can_reuse_unblocked_continuation(result: ContinuationResult, blocked_potion_action: dict[str, Any]) -> bool:
    # Reuse is exact only when the unblocked search saw the whole search tree.
    # Beam pruning can make a blocked search explore paths that were pruned in
    # the unblocked search, even if the unblocked best line did not use potion.
    return bool(result.fully_explored and not _continuation_uses_same_potion(result, blocked_potion_action))


def teacher_value(env: Any, config: TeacherConfig | None = None) -> float:
    return 0.0


def _leaf_result(
    env: Any,
    *,
    path: list[dict[str, Any]],
    depth: int,
    nodes: int,
    config: TeacherConfig,
    cumulative_value: float = 0.0,
    state: dict[str, Any] | None = None,
    phase: str | None = None,
    terminal: str | None = None,
    used_potion_keys: set[tuple[int | None, str]] | None = None,
) -> ContinuationResult:
    if state is None:
        state = _scoring_state_from_env(env)
    if phase is None:
        phase = _phase_from_state(env, state)
    if terminal is None:
        terminal = _is_terminal_from_state(env, state, phase)
    if terminal is None and phase == "COMBAT":
        terminal = "NEXT_TURN" if depth > 0 else "COMBAT"
    if terminal is None:
        terminal = phase or "UNKNOWN"
    return ContinuationResult(
        value=float(cumulative_value if TEACHER_VALUE_IS_ZERO else cumulative_value + teacher_value(env, config)),
        depth=depth,
        nodes=nodes,
        terminal_kind=terminal,
        debug_best_line=[dict(action) for action in path],
        used_potion_keys=set(used_potion_keys or set()),
    )


def continuation_search(
    env: Any,
    *,
    root_turn: int,
    config: TeacherConfig | None = None,
    lethal_available: bool = False,
    block_reward_base_state: dict[str, Any] | None = None,
    blocked_potion_action: dict[str, Any] | None = None,
    track_debug_line: bool = True,
    track_potion_keys: bool = True,
    initial_state: dict[str, Any] | None = None,
    initial_phase: str | None = None,
    initial_terminal: str | None = None,
) -> ContinuationResult:
    cfg = config or default_teacher_config()
    if initial_state is None:
        initial_state = _scoring_state_from_env(env)
    if initial_phase is None:
        initial_phase = _phase_from_state(env, initial_state)
    if initial_terminal is None:
        initial_terminal = _is_terminal_from_state(env, initial_state, initial_phase)
    beam: list[
        tuple[Any, dict[str, Any], str, str | None, list[dict[str, Any]], int, float, set[tuple[int | None, str]] | None]
    ] = [
        (env, initial_state, initial_phase, initial_terminal, [], 0, 0.0, None)
    ]
    best: ContinuationResult | None = None
    nodes = 0
    fully_explored = True

    while beam and nodes < cfg.node_budget_per_root:
        safe_single_action_inplace = _teacher_env_flag("SPIRECOMM_V3_TEACHER_SAFE_SINGLE_ACTION_INPLACE", True)
        beam_env_ref_counts: dict[int, int] | None = None
        if safe_single_action_inplace and len(beam) > 1:
            beam_env_ref_counts = {}
            for entry in beam:
                env_id = id(entry[0])
                beam_env_ref_counts[env_id] = beam_env_ref_counts.get(env_id, 0) + 1
        next_beam: list[
            tuple[
                float,
                Any,
                dict[str, Any],
                str,
                str | None,
                list[dict[str, Any]],
                int,
                float,
                set[tuple[int | None, str]] | None,
            ]
        ] = []
        for current_env, current_state, phase, terminal, path, depth, cumulative_value, used_potion_keys in beam:
            if nodes >= cfg.node_budget_per_root:
                fully_explored = False
                break
            if (
                terminal is not None
                or _is_next_player_decision_from_state(current_state, phase=phase, root_turn=root_turn)
                or depth >= cfg.max_depth
            ):
                candidate = _leaf_result(
                    current_env,
                    path=path,
                    depth=depth,
                    nodes=nodes,
                    config=cfg,
                    cumulative_value=cumulative_value,
                    state=current_state,
                    phase=phase,
                    terminal=terminal,
                    used_potion_keys=used_potion_keys,
                )
                if best is None or candidate.value > best.value:
                    best = candidate
                continue
            actions = _legal_teacher_actions_from_state(current_env, current_state, phase)
            if blocked_potion_action is not None:
                actions = [action for action in actions if not _is_same_potion_action(action, blocked_potion_action)]
            actions = _prune_continuation_actions_for_speed(
                current_state,
                actions,
                action_cap=int(getattr(cfg, "continuation_action_cap", 0) or 0),
            )
            if not actions:
                candidate = _leaf_result(
                    current_env,
                    path=path,
                    depth=depth,
                    nodes=nodes,
                    config=cfg,
                    cumulative_value=cumulative_value,
                    state=current_state,
                    phase=phase,
                    terminal=terminal,
                    used_potion_keys=used_potion_keys,
                )
                if best is None or candidate.value > best.value:
                    best = candidate
                continue
            can_step_single_in_place = (
                len(actions) == 1
                and depth > 0
                and safe_single_action_inplace
                and "cards_played_this_turn" in _combat_state_view(current_state)
                and (
                    beam_env_ref_counts is None
                    or beam_env_ref_counts.get(id(current_env), 0) == 1
                )
            )
            current_env_blob = None if can_step_single_in_place else _clone_step_source_or_none(current_env)
            current_env_blob_prefix = None if can_step_single_in_place else _step_branch_cache_source_prefix(current_env_blob)
            for action in actions:
                if nodes >= cfg.node_budget_per_root:
                    fully_explored = False
                    break
                try:
                    branch = (
                        _step_branch_in_place(current_env, action)
                        if can_step_single_in_place
                        else _step_branch_with_source_cached(
                            current_env,
                            current_env_blob,
                            action,
                            source_prefix=current_env_blob_prefix,
                        )
                    )
                except Exception:
                    continue
                nodes += 1
                branch_state = _scoring_state_from_env(branch)
                branch_phase = _phase_from_state(branch, branch_state)
                branch_terminal = _is_terminal_from_state(branch, branch_state, branch_phase)
                reward = transition_reward(
                    current_env,
                    current_state,
                    action,
                    branch,
                    branch_state,
                    cfg,
                    lethal_available=lethal_available,
                    block_reward_base_state=block_reward_base_state,
                    after_phase=branch_phase,
                    after_terminal=branch_terminal,
                )
                new_value = float(cumulative_value + reward)
                new_path = path + [action] if track_debug_line else path
                new_used_potion_keys = used_potion_keys if track_potion_keys else None
                if track_potion_keys and (action.get("kind") or "") == "potion":
                    new_used_potion_keys = set(used_potion_keys or set())
                    new_used_potion_keys.add((_action_potion_index(action), _action_potion_id(action)))
                priority = new_value if TEACHER_VALUE_IS_ZERO else new_value + teacher_value(branch, cfg)
                next_beam.append(
                    (
                        priority,
                        branch,
                        branch_state,
                        branch_phase,
                        branch_terminal,
                        new_path,
                        depth + 1,
                        new_value,
                        new_used_potion_keys,
                    )
                )
        raw_next_beam_len = len(next_beam)
        if raw_next_beam_len > 1:
            next_beam = _dedupe_next_beam_entries(next_beam)
        if raw_next_beam_len > cfg.beam_width:
            # Keep the old fully_explored semantics even if equivalent-state
            # merging brings the beam back under width.
            fully_explored = False
        if len(next_beam) > cfg.beam_width:
            next_beam = nlargest(cfg.beam_width, next_beam, key=itemgetter(0))
        else:
            next_beam.sort(key=itemgetter(0), reverse=True)
        beam = [
            (branch, state, phase, terminal, path, depth, value, used_potion_keys)
            for _, branch, state, phase, terminal, path, depth, value, used_potion_keys in next_beam[: cfg.beam_width]
        ]

    if nodes >= cfg.node_budget_per_root:
        fully_explored = False
    if best is None:
        if beam:
            best_env, best_state, best_phase, best_terminal, best_path, best_depth, best_value, best_used_potion_keys = beam[0]
            best = _leaf_result(
                best_env,
                path=best_path,
                depth=best_depth,
                nodes=nodes,
                config=cfg,
                cumulative_value=best_value,
                state=best_state,
                phase=best_phase,
                terminal=best_terminal,
                used_potion_keys=best_used_potion_keys,
            )
        else:
            best = _leaf_result(env, path=[], depth=0, nodes=nodes, config=cfg)
    best.nodes = nodes
    best.fully_explored = bool(fully_explored)
    return best


def score_root_actions_fast(
    root_env: Any,
    *,
    config: TeacherConfig | None = None,
    legal_actions: list[dict[str, Any]] | None = None,
    root_step_source: bytes | _TeacherCombatStepSource | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Runtime-only teacher scores without dataset feature construction."""

    cfg = config or default_teacher_config()
    if _teacher_env_flag("SPIRECOMM_V3_TEACHER_SINGLE_CONFIG_MANY_FAST", False):
        scores_by_config = score_root_actions_many_configs_fast(root_env, [cfg], legal_actions=legal_actions)
        if scores_by_config is not None:
            return scores_by_config[0]
    before_state = _scoring_state_from_env(root_env)
    actions = _root_combat_actions_no_copy(root_env, legal_actions=legal_actions)
    if len(actions) <= 1:
        return []
    root_turn = _combat_turn_from_state(before_state)
    initial_expanded: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    root_env_blob = root_step_source if root_step_source is not None else _clone_step_source_or_none(root_env)
    root_env_blob_prefix = _step_branch_cache_source_prefix(root_env_blob)
    for action in actions:
        immediate_branch = _step_branch_with_source_cached(
            root_env,
            root_env_blob,
            action,
            source_prefix=root_env_blob_prefix,
        )
        visible_after = _scoring_state_from_env(immediate_branch)
        initial_expanded.append((action, immediate_branch, visible_after))
    root_lethal_available = _root_turn_lethal_available_from_branches(
        [(action, branch, visible_after) for action, branch, visible_after in initial_expanded],
        root_turn=root_turn,
        config=cfg,
    )
    has_root_potion = any((action.get("kind") or "") == "potion" for action in actions)
    expanded: list[tuple[dict[str, Any], Any, dict[str, Any], float, ContinuationResult]] = []
    continuation_cache: dict[str, ContinuationResult] = {}
    for action, immediate_branch, visible_after in initial_expanded:
        after_phase = _phase_from_state(immediate_branch, visible_after)
        after_terminal = _is_terminal_from_state(immediate_branch, visible_after, after_phase)
        immediate_reward = transition_reward(
            root_env,
            before_state,
            action,
            immediate_branch,
            visible_after,
            cfg,
            lethal_available=root_lethal_available,
            block_reward_base_state=before_state,
            after_phase=after_phase,
            after_terminal=after_terminal,
        )
        continuation_key = (
            _branch_state_merge_signature(
                visible_after,
                phase=after_phase,
                terminal=after_terminal,
                depth=1,
                used_potion_keys={
                    (_action_potion_index(action), _action_potion_id(action))
                }
                if has_root_potion and (action.get("kind") or "") == "potion"
                else None,
            )
            if _teacher_env_flag("SPIRECOMM_V3_TEACHER_ROOT_CONTINUATION_CACHE", True)
            else ""
        )
        cached_result = continuation_cache.get(continuation_key) if continuation_key else None
        if cached_result is not None:
            result = cached_result
        else:
            result = continuation_search(
                immediate_branch,
                root_turn=root_turn,
                config=cfg,
                lethal_available=root_lethal_available,
                block_reward_base_state=before_state,
                track_debug_line=False,
                track_potion_keys=has_root_potion,
                initial_state=visible_after,
                initial_phase=after_phase,
                initial_terminal=after_terminal,
            )
            if continuation_key:
                continuation_cache[continuation_key] = result
        expanded.append((action, immediate_branch, visible_after, float(immediate_reward), result))

    non_potion_expanded = [
        (action, branch, visible_after, immediate_reward, result)
        for action, branch, visible_after, immediate_reward, result in expanded
        if (action.get("kind") or "") != "potion"
    ]
    blocked_baseline_cache: dict[tuple[int | None, str], float] = {}

    def non_potion_baseline_for(root_potion_action: dict[str, Any]) -> float:
        cache_key = (_action_potion_index(root_potion_action), _action_potion_id(root_potion_action))
        cached = blocked_baseline_cache.get(cache_key)
        if cached is not None:
            return cached
        values = []
        for _action, branch, _visible_after, immediate_reward, result in non_potion_expanded:
            if _can_reuse_unblocked_continuation(result, root_potion_action):
                baseline_result = result
            else:
                baseline_result = continuation_search(
                    branch,
                    root_turn=root_turn,
                    config=cfg,
                    lethal_available=root_lethal_available,
                    block_reward_base_state=before_state,
                    blocked_potion_action=root_potion_action,
                    track_debug_line=False,
                    track_potion_keys=False,
                    initial_state=_visible_after,
                )
            values.append(float(immediate_reward) + float(baseline_result.value))
        baseline = max(values) if values else 0.0
        blocked_baseline_cache[cache_key] = float(baseline)
        return float(baseline)

    scores: list[tuple[dict[str, Any], float]] = []
    for action, _immediate_branch, _visible_after, immediate_reward, result in expanded:
        non_potion_baseline = (
            non_potion_baseline_for(action) if (action.get("kind") or "") == "potion" else 0.0
        )
        continuation_value = _root_potion_continuation_value(
            action,
            before_state,
            immediate_reward,
            result.value,
            cfg,
            non_potion_baseline=non_potion_baseline,
        )
        scores.append((dict(action), float(immediate_reward + continuation_value)))
    adjusted_scores, _adjustments = _apply_teacher_survival_guard_to_scores(
        before_state,
        [(action, branch, visible_after) for action, branch, visible_after, _immediate_reward, _result in expanded],
        [score for _action, score in scores],
        cfg,
    )
    if adjusted_scores != [score for _action, score in scores]:
        scores = [(action, float(adjusted_scores[index])) for index, (action, _score) in enumerate(scores)]
    return scores


def _teacher_batch_configs_compatible(configs: list[TeacherConfig]) -> bool:
    if not configs:
        return False
    first = configs[0]
    keys = (
        "beam_width",
        "node_budget_per_root",
        "max_depth",
        "continuation_action_cap",
        "lethal_check_node_budget",
    )
    return all(all(getattr(config, key) == getattr(first, key) for key in keys) for config in configs[1:])


def _dedupe_batch_next_entries(
    entries: list[tuple[float, int, int, float]],
    nodes: list[tuple[Any, dict[str, Any], str, str | None]],
    signature_cache: dict[tuple[int, int], Any] | None = None,
) -> list[tuple[float, int, int, float]]:
    if len(entries) <= 1 or not _teacher_env_flag("SPIRECOMM_V3_TEACHER_BRANCH_STATE_MERGE", True):
        return entries
    merged: dict[str, tuple[float, int, int, float]] = {}
    for entry in entries:
        priority, node_id, depth, _value = entry
        _env, state, phase, terminal = nodes[node_id]
        cache_key = (int(node_id), int(depth))
        if signature_cache is not None and cache_key in signature_cache:
            key = signature_cache[cache_key]
        else:
            key = _branch_state_merge_signature(
                state,
                phase=phase,
                terminal=terminal,
                depth=depth,
                used_potion_keys=None,
            )
            if signature_cache is not None:
                signature_cache[cache_key] = key
        previous = merged.get(key)
        if previous is None or float(priority) > float(previous[0]):
            merged[key] = entry
    return list(merged.values())


def _teacher_action_exact_cache_key(action: dict[str, Any]) -> Any:
    items = tuple(sorted(action.items()))
    try:
        hash(items)
        return items
    except TypeError:
        return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _teacher_semantic_state_key_for_transposition(
    state: dict[str, Any],
    *,
    phase: str,
    terminal: str | None,
) -> Any | None:
    return _branch_state_merge_signature(
        state,
        phase=phase,
        terminal=terminal,
        depth=0,
        used_potion_keys=None,
    )


def _teacher_semantic_action_key_for_state(state: dict[str, Any], action: dict[str, Any]) -> Any:
    kind = str(action.get("kind") or "")
    combat = _combat_state_view(state)
    if kind == "card":
        hand = _hand_cards_from_combat_view(combat)
        try:
            card_index = int(action.get("card_index", action.get("source_index", -1)))
        except (TypeError, ValueError):
            card_index = -1
        card_key: Any
        if 0 <= card_index < len(hand):
            card_key = _fast_branch_card_key(hand[card_index])
        else:
            card_key = None
        if card_key is None:
            card_key = (
                str(action.get("card_id") or ""),
                str(action.get("name") or ""),
                str(action.get("card_type") or ""),
            )
        target_key: Any = None
        if action.get("target_index") is not None:
            monsters = _monsters_from_combat_view(combat)
            try:
                target_index = int(action.get("target_index"))
            except (TypeError, ValueError):
                target_index = -1
            if 0 <= target_index < len(monsters):
                target_key = _equivalent_monster_target_key(monsters[target_index])
            else:
                target_key = int(target_index)
        return ("card", card_key, target_key, bool(action.get("requires_target", False)))
    if kind == "potion":
        target_key = None
        if action.get("target_index") is not None:
            monsters = _monsters_from_combat_view(combat)
            try:
                target_index = int(action.get("target_index"))
            except (TypeError, ValueError):
                target_index = -1
            if 0 <= target_index < len(monsters):
                target_key = _equivalent_monster_target_key(monsters[target_index])
            else:
                target_key = int(target_index)
        potion_index = _action_potion_index(action)
        potion_id = _action_potion_id(action)
        # Prefer potion id over physical slot for transposition; using a
        # semantically equal potion from another slot is an accepted approximate
        # speed tradeoff for sweeps.
        return ("potion", str(potion_id), target_key, potion_index is None)
    if kind == "end":
        return ("end",)
    return _teacher_action_exact_cache_key(action)


_BATCH_LINEAR_REWARD_CONSTANT_FIELDS = (
    "suppress_block_reward_when_lethal_available",
    "lethal_block_suppression_factor",
    "lethal_block_low_hp_protection",
    "lethal_block_low_hp_max",
    "lethal_block_low_hp_suppression_factor",
    "lethal_block_low_hp_requires_facing_lethal",
)

_POTION_REWARD_CARD_ONLY_FIELDS = {
    "energy_spent_weight",
    "playable_hand_count_delta_weight",
    "play_card_constant",
    "power_card_constant",
    "skill_power_turn_constant",
    "turn_order_decay_per_card",
}


def _teacher_batch_linear_reward_compatible(configs: list[TeacherConfig]) -> bool:
    if not configs:
        return False
    first = configs[0]
    return all(
        all(getattr(config, key) == getattr(first, key) for key in _BATCH_LINEAR_REWARD_CONSTANT_FIELDS)
        for config in configs[1:]
    )


def _teacher_batch_potion_reward_compatible(configs: list[TeacherConfig]) -> bool:
    if not configs:
        return False
    first = configs[0]
    for config in configs[1:]:
        for field_info in fields(TeacherConfig):
            name = field_info.name
            if name in _POTION_REWARD_CARD_ONLY_FIELDS:
                continue
            if getattr(config, name) != getattr(first, name):
                return False
    return True


def _transition_reward_linear_features(
    before_env: Any,
    before_state: dict[str, Any],
    action: dict[str, Any],
    after_env: Any,
    after_state: dict[str, Any],
    config: TeacherConfig,
    *,
    lethal_available: bool,
    block_reward_base_state: dict[str, Any] | None,
    after_phase: str,
    after_terminal: str | None,
) -> tuple[Any, ...] | None:
    if (action.get("kind") or "") == "potion":
        return None
    victory_after = _is_post_combat_victory_from_state(before_state, after_state, after_terminal)
    defeat_after = after_terminal == "DEFEAT"
    before_combat = _combat_state_view(before_state)
    after_combat = _combat_state_view(after_state)
    before_monsters = _monsters_from_combat_view(before_combat)
    after_monsters = _monsters_from_combat_view(after_combat)
    before_player = _player_from_combat_view(before_combat)
    after_player = _player_from_combat_view(after_combat)
    after_has_combat = bool(after_combat)
    (
        before_hp_total,
        raw_after_hp_total,
        monster_kill_count,
        incoming_before_int,
        incoming_after_int,
        before_monsters_have_power,
        after_monsters_have_power,
    ) = _monster_transition_stats(
        before_monsters,
        after_monsters,
        victory_after=victory_after,
        after_has_combat=after_has_combat,
    )
    after_hp_total = 0 if victory_after and not after_has_combat else raw_after_hp_total
    hp_damage = max(0.0, float(before_hp_total - after_hp_total))
    effective_block = 0.0
    if str(action.get("kind") or "") == "end":
        effective_block = _deferred_effective_block(
            block_reward_base_state,
            before_state,
            lethal_available=lethal_available,
            cfg=config,
        )
    raw_incoming_damage_reduction = max(0.0, float(incoming_before_int - incoming_after_int))

    is_card_action = (action.get("kind") or "") == "card"
    hp_loss = 0.0
    energy_spent = 0.0
    playable_hand_count_delta = 0.0
    if is_card_action:
        before_player_hp = _int(before_state.get("current_hp"), _int(before_player.get("current_hp")))
        after_player_hp = _int(after_state.get("current_hp"), _int(after_player.get("current_hp")))
        hp_loss = max(0.0, float(before_player_hp - after_player_hp))
        energy_spent = max(0.0, float(_int(before_player.get("energy")) - _int(after_player.get("energy"))))
        playable_before = _playable_count_from_hand(_hand_cards_from_combat_view(before_combat))
        playable_after = _playable_count_from_hand(_hand_cards_from_combat_view(after_combat)) if after_has_combat else 0
        playable_hand_count_delta = max(0.0, float(playable_after - max(0, playable_before - 1)))

    player_power_features: tuple[tuple[str, float], ...] = ()
    if (not victory_after or after_has_combat) and (
        bool(before_player.get("powers")) or bool(after_player.get("powers"))
    ):
        player_power_features = _player_power_delta_features(
            before_state,
            after_state,
            victory_after=victory_after,
        )

    monster_vulnerable = 0.0
    monster_weakened = 0.0
    monster_strength = 0.0
    monster_shackled = 0.0
    if (not victory_after or after_has_combat) and (before_monsters_have_power or after_monsters_have_power):
        after_mons = _monsters_view(after_state)
        for index, before_monster in enumerate(_monsters_view(before_state)):
            if not _is_alive(before_monster):
                continue
            after_monster = after_mons[index] if index < len(after_mons) else {}
            before_powers = _monster_power_amounts(before_monster)
            after_powers = _monster_power_amounts(after_monster)
            hp_factor = _target_hp_fraction(before_monster)
            for power_id in ("Vulnerable", "Weakened", "Strength", "Shackled"):
                before_amount = float(before_powers.get(power_id, 0.0))
                after_amount = float(after_powers.get(power_id, 0.0))
                delta = (after_amount - before_amount) / max(1e-6, POWER_BASE_AMOUNTS.get(power_id, 1.0)) * hp_factor
                if power_id == "Vulnerable":
                    monster_vulnerable += delta
                elif power_id == "Weakened":
                    monster_weakened += delta
                elif power_id == "Strength":
                    monster_strength -= delta
                elif power_id == "Shackled":
                    monster_shackled -= delta

    play_card = 0.0
    power_card = 0.0
    skill_power_turn = 0.0
    skill_power_cards_played = 0.0
    if is_card_action:
        card_type = _selected_card_type(before_state, action)
        card_id = _selected_card_id(before_state, action)
        if card_type in CARD_ACTION_TYPES:
            play_card = 1.0
        if card_type == "POWER":
            power_card = float(_fight_remaining(before_state))
        if card_id in TURN_ORDER_SKILL_POWER_CARD_IDS:
            skill_power_turn = 1.0
            skill_power_cards_played = float(_cards_played_this_turn_from_state(before_state, before_env))

    return (
        hp_damage,
        float(monster_kill_count),
        1.0 if victory_after else 0.0,
        1.0 if defeat_after else 0.0,
        hp_loss,
        float(effective_block),
        raw_incoming_damage_reduction,
        energy_spent,
        playable_hand_count_delta,
        player_power_features,
        monster_vulnerable,
        monster_weakened,
        monster_strength,
        monster_shackled,
        play_card,
        power_card,
        skill_power_turn,
        skill_power_cards_played,
    )


def _transition_reward_from_linear_features(features: tuple[Any, ...], config: TeacherConfig) -> float:
    return _transition_reward_from_linear_features_and_coefficients(
        features,
        _linear_reward_coefficients(config),
    )


def _linear_reward_coefficients(config: TeacherConfig) -> tuple[Any, ...]:
    return (
        float(config.hp_damage_weight),
        float(config.monster_kill_weight),
        float(config.combat_win_weight),
        float(config.death_weight),
        float(config.hp_loss_weight),
        float(config.effective_block_weight),
        float(config.raw_incoming_damage_reduction_weight),
        float(config.energy_spent_weight),
        float(config.playable_hand_count_delta_weight),
        config.player_power_weights,
        float(config.monster_vulnerable_weight),
        float(config.monster_weakened_weight),
        float(config.monster_strength_weight),
        float(config.monster_shackled_weight),
        float(config.play_card_constant),
        float(config.power_card_constant),
        float(config.skill_power_turn_constant),
        float(config.turn_order_decay_per_card),
    )


def _transition_reward_from_linear_features_and_coefficients(
    features: tuple[Any, ...],
    coeffs: tuple[Any, ...],
) -> float:
    return float(
        coeffs[0] * features[0]
        + coeffs[1] * features[1]
        + coeffs[2] * features[2]
        + coeffs[3] * features[3]
        + coeffs[4] * features[4]
        + coeffs[5] * features[5]
        + coeffs[6] * features[6]
        + coeffs[7] * features[7]
        + coeffs[8] * features[8]
        + _player_power_delta_value_from_features_and_weights(features[9], coeffs[9])
        + coeffs[10] * features[10]
        + coeffs[11] * features[11]
        + coeffs[12] * features[12]
        + coeffs[13] * features[13]
        + coeffs[14] * features[14]
        + coeffs[15] * features[15]
        + coeffs[16]
        * features[16]
        * max(0.0, 1.0 - coeffs[17] * float(features[17] if len(features) > 17 else 0.0))
    )


_LINEAR_REWARD_SCALAR_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15)


def _linear_reward_batch_plan(coefficients_by_config: list[tuple[Any, ...]]) -> tuple[tuple[int, ...], tuple[int, ...], bool, bool]:
    if not coefficients_by_config:
        return (), (), True, True
    first = coefficients_by_config[0]
    fixed_indices = []
    variable_indices = []
    for index in _LINEAR_REWARD_SCALAR_INDICES:
        if all(coeffs[index] == first[index] for coeffs in coefficients_by_config[1:]):
            fixed_indices.append(index)
        else:
            variable_indices.append(index)
    player_power_fixed = all(coeffs[9] == first[9] for coeffs in coefficients_by_config[1:])
    skill_turn_fixed = all(
        coeffs[16] == first[16] and coeffs[17] == first[17]
        for coeffs in coefficients_by_config[1:]
    )
    return tuple(fixed_indices), tuple(variable_indices), bool(player_power_fixed), bool(skill_turn_fixed)


def _skill_power_turn_linear_term(features: tuple[Any, ...], coeffs: tuple[Any, ...]) -> float:
    return float(
        coeffs[16]
        * features[16]
        * max(0.0, 1.0 - coeffs[17] * float(features[17] if len(features) > 17 else 0.0))
    )


def _transition_reward_values_from_linear_features_and_coefficients(
    features: tuple[Any, ...],
    coefficients_by_config: list[tuple[Any, ...]],
    batch_plan: tuple[tuple[int, ...], tuple[int, ...], bool, bool],
) -> list[float]:
    if not coefficients_by_config:
        return []
    first = coefficients_by_config[0]
    fixed_indices, variable_indices, player_power_fixed, skill_turn_fixed = batch_plan
    fixed_value = 0.0
    for index in fixed_indices:
        fixed_value += float(first[index] * features[index])
    if player_power_fixed:
        fixed_value += float(_player_power_delta_value_from_features_and_weights(features[9], first[9]))
    if skill_turn_fixed:
        fixed_value += _skill_power_turn_linear_term(features, first)

    values: list[float] = []
    for coeffs in coefficients_by_config:
        value = fixed_value
        for index in variable_indices:
            value += float(coeffs[index] * features[index])
        if not player_power_fixed:
            value += float(_player_power_delta_value_from_features_and_weights(features[9], coeffs[9]))
        if not skill_turn_fixed:
            value += _skill_power_turn_linear_term(features, coeffs)
        values.append(float(value))
    return values


def _continuation_search_many_configs_no_potion_tracking(
    initial_nodes: list[tuple[Any, dict[str, Any], str, str | None]],
    *,
    root_turn: int,
    configs: list[TeacherConfig],
    lethal_available: bool,
    block_reward_base_state: dict[str, Any],
    blocked_potion_action: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Exact shared continuation search for compatible configs.

    This preserves each config's own beam, node budget, reward weights and
    pruning order, but shares environment expansion for identical state/action
    pairs across configs and root actions. Potion history is intentionally not
    tracked; blocked-potion baselines are computed by rerunning this function
    with the same potion action filtered from every continuation node.
    """

    cfg0 = configs[0]
    nodes: list[tuple[Any, dict[str, Any], str, str | None]] = list(initial_nodes)
    expansion_cache: dict[tuple[int, Any], tuple[int, dict[str, Any]] | None] = {}
    reward_feature_cache: dict[tuple[int, Any], tuple[Any, ...] | None] = {}
    reward_values_cache: dict[tuple[int, Any], list[float] | None] = {}
    legal_actions_base_cache: dict[int, list[dict[str, Any]]] = {}
    legal_actions_cache: dict[tuple[int, tuple[int | None, str] | None], list[dict[str, Any]]] = {}
    branch_signature_cache: dict[tuple[int, int], Any] = {}
    node_step_source_cache: dict[int, tuple[bytes | _TeacherCombatStepSource | None, bytes | None]] = {}
    action_exact_key_cache: dict[int, Any] = {}
    semantic_node_key_cache: dict[int, Any | None] = {}
    semantic_action_key_cache: dict[tuple[int, int], Any] = {}
    semantic_expansion_cache: dict[tuple[Any, Any], tuple[int, dict[str, Any]] | None] = {}
    semantic_reward_feature_cache: dict[tuple[Any, Any], tuple[Any, ...] | None] = {}
    semantic_reward_values_cache: dict[tuple[Any, Any], list[float] | None] = {}
    semantic_transposition_enabled = _teacher_env_flag("SPIRECOMM_V3_TEACHER_SEMANTIC_TRANSPOSITION_CACHE", True)
    linear_reward_compatible = _teacher_batch_linear_reward_compatible(configs) and _teacher_env_flag(
        "SPIRECOMM_V3_TEACHER_BATCH_LINEAR_REWARD_VALUES",
        True,
    )
    potion_reward_compatible = _teacher_batch_potion_reward_compatible(configs)
    vector_reward_values = linear_reward_compatible and _teacher_env_flag(
        "SPIRECOMM_V3_TEACHER_VECTOR_REWARD_VALUES",
        True,
    )
    linear_coefficients_by_config = (
        [_linear_reward_coefficients(cfg) for cfg in configs]
        if linear_reward_compatible
        else []
    )
    linear_reward_batch_plan = (
        _linear_reward_batch_plan(linear_coefficients_by_config)
        if linear_coefficients_by_config
        else ((), (), True, True)
    )
    potion_reward_cache: dict[tuple[int, Any], float] = {}
    blocked_potion_key = (
        (_action_potion_index(blocked_potion_action), _action_potion_id(blocked_potion_action))
        if blocked_potion_action is not None
        else None
    )

    def action_exact_key(action: dict[str, Any]) -> Any:
        object_id = id(action)
        cached = action_exact_key_cache.get(object_id)
        if cached is not None:
            return cached
        key = _teacher_action_exact_cache_key(action)
        action_exact_key_cache[object_id] = key
        return key

    def semantic_node_key(node_id: int) -> Any | None:
        node_id = int(node_id)
        if node_id in semantic_node_key_cache:
            return semantic_node_key_cache[node_id]
        _env, state, phase, terminal = nodes[node_id]
        key = _teacher_semantic_state_key_for_transposition(
            state,
            phase=phase,
            terminal=terminal,
        )
        semantic_node_key_cache[node_id] = key
        return key

    def semantic_action_key(node_id: int, action: dict[str, Any]) -> Any:
        key = (int(node_id), id(action))
        cached = semantic_action_key_cache.get(key)
        if cached is not None:
            return cached
        _env, state, _phase, _terminal = nodes[int(node_id)]
        value = _teacher_semantic_action_key_for_state(state, action)
        semantic_action_key_cache[key] = value
        return value

    def expand(node_id: int, action: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
        key = (int(node_id), action_exact_key(action))
        if key in expansion_cache:
            return expansion_cache[key]
        transposition_key: tuple[Any, Any] | None = None
        if semantic_transposition_enabled:
            source_key = semantic_node_key(int(node_id))
            if source_key is not None:
                transposition_key = (source_key, semantic_action_key(int(node_id), action))
                if transposition_key in semantic_expansion_cache:
                    result = semantic_expansion_cache[transposition_key]
                    expansion_cache[key] = result
                    return result
        current_env, _current_state, _phase, _terminal = nodes[node_id]
        try:
            source_entry = node_step_source_cache.get(int(node_id))
            if source_entry is None and int(node_id) not in node_step_source_cache:
                current_env_blob = _clone_step_source_or_none(current_env)
                current_env_blob_prefix = _step_branch_cache_source_prefix(current_env_blob)
                source_entry = (current_env_blob, current_env_blob_prefix)
                node_step_source_cache[int(node_id)] = source_entry
            else:
                current_env_blob, current_env_blob_prefix = source_entry
            branch = _step_branch_with_source_cached(
                current_env,
                current_env_blob,
                action,
                source_prefix=current_env_blob_prefix,
            )
        except Exception:
            expansion_cache[key] = None
            if transposition_key is not None:
                semantic_expansion_cache[transposition_key] = None
            return None
        branch_state = _scoring_state_from_env(branch)
        branch_phase = _phase_from_state(branch, branch_state)
        branch_terminal = _is_terminal_from_state(branch, branch_state, branch_phase)
        child_id = len(nodes)
        nodes.append((branch, branch_state, branch_phase, branch_terminal))
        result = (child_id, branch_state)
        expansion_cache[key] = result
        if transposition_key is not None:
            semantic_expansion_cache[transposition_key] = result
        return result

    def legal_actions_for(node_id: int, current_env: Any, current_state: dict[str, Any], phase: str) -> list[dict[str, Any]]:
        key = (int(node_id), blocked_potion_key)
        cached = legal_actions_cache.get(key)
        if cached is not None:
            return cached
        base_actions = legal_actions_base_cache.get(int(node_id))
        if base_actions is None:
            base_actions = _legal_teacher_actions_from_state(current_env, current_state, phase)
            legal_actions_base_cache[int(node_id)] = base_actions
        if blocked_potion_action is not None:
            actions = [
                action
                for action in base_actions
                if not _is_same_potion_action(action, blocked_potion_action)
            ]
        else:
            actions = base_actions
        actions = _prune_continuation_actions_for_speed(
            current_state,
            actions,
            action_cap=int(getattr(cfg0, "continuation_action_cap", 0) or 0),
        )
        legal_actions_cache[key] = actions
        return actions

    def reward_for(
        current_node_id: int,
        action: dict[str, Any],
        branch: Any,
        branch_state: dict[str, Any],
        config_index: int,
        cfg: TeacherConfig,
        coeffs: tuple[Any, ...] | None,
        *,
        branch_phase: str,
        branch_terminal: str | None,
    ) -> float:
        current_env, current_state, _phase, _terminal = nodes[current_node_id]
        exact_key = action_exact_key(action)
        if potion_reward_compatible and (action.get("kind") or "") == "potion":
            key = (int(current_node_id), exact_key)
            if key in potion_reward_cache:
                return float(potion_reward_cache[key])
            value = transition_reward(
                current_env,
                current_state,
                action,
                branch,
                branch_state,
                cfg,
                lethal_available=lethal_available,
                block_reward_base_state=block_reward_base_state,
                after_phase=branch_phase,
                after_terminal=branch_terminal,
            )
            potion_reward_cache[key] = float(value)
            return float(value)
        semantic_reward_key: tuple[Any, Any] | None = None
        if semantic_transposition_enabled:
            source_key = semantic_node_key(int(current_node_id))
            if source_key is not None:
                semantic_reward_key = (source_key, semantic_action_key(int(current_node_id), action))
        if vector_reward_values and (action.get("kind") or "") != "potion":
            key = (int(current_node_id), exact_key)
            values = reward_values_cache.get(key)
            if values is not None:
                return float(values[config_index])
            if semantic_reward_key is not None:
                semantic_values = semantic_reward_values_cache.get(semantic_reward_key)
                if semantic_values is not None:
                    reward_values_cache[key] = semantic_values
                    return float(semantic_values[config_index])
                if semantic_reward_key in semantic_reward_values_cache:
                    reward_values_cache[key] = None
                    return transition_reward(
                        current_env,
                        current_state,
                        action,
                        branch,
                        branch_state,
                        cfg,
                        lethal_available=lethal_available,
                        block_reward_base_state=block_reward_base_state,
                        after_phase=branch_phase,
                        after_terminal=branch_terminal,
                    )
            if key in reward_values_cache:
                return transition_reward(
                    current_env,
                    current_state,
                    action,
                    branch,
                    branch_state,
                    cfg,
                    lethal_available=lethal_available,
                    block_reward_base_state=block_reward_base_state,
                    after_phase=branch_phase,
                    after_terminal=branch_terminal,
                )
            features = reward_feature_cache.get(key)
            if features is None and key not in reward_feature_cache:
                if semantic_reward_key is not None and semantic_reward_key in semantic_reward_feature_cache:
                    features = semantic_reward_feature_cache[semantic_reward_key]
                else:
                    features = _transition_reward_linear_features(
                        current_env,
                        current_state,
                        action,
                        branch,
                        branch_state,
                        cfg0,
                        lethal_available=lethal_available,
                        block_reward_base_state=block_reward_base_state,
                        after_phase=branch_phase,
                        after_terminal=branch_terminal,
                    )
                    if semantic_reward_key is not None:
                        semantic_reward_feature_cache[semantic_reward_key] = features
                reward_feature_cache[key] = features
            if features is not None:
                values = _transition_reward_values_from_linear_features_and_coefficients(
                    features,
                    linear_coefficients_by_config,
                    linear_reward_batch_plan,
                )
                reward_values_cache[key] = values
                if semantic_reward_key is not None:
                    semantic_reward_values_cache[semantic_reward_key] = values
                return float(values[config_index])
            reward_values_cache[key] = None
            if semantic_reward_key is not None:
                semantic_reward_values_cache[semantic_reward_key] = None
        if linear_reward_compatible and (action.get("kind") or "") != "potion":
            key = (int(current_node_id), exact_key)
            features = reward_feature_cache.get(key)
            if features is None and key not in reward_feature_cache:
                if semantic_reward_key is not None and semantic_reward_key in semantic_reward_feature_cache:
                    features = semantic_reward_feature_cache[semantic_reward_key]
                else:
                    features = _transition_reward_linear_features(
                        current_env,
                        current_state,
                        action,
                        branch,
                        branch_state,
                        cfg0,
                        lethal_available=lethal_available,
                        block_reward_base_state=block_reward_base_state,
                        after_phase=branch_phase,
                        after_terminal=branch_terminal,
                    )
                    if semantic_reward_key is not None:
                        semantic_reward_feature_cache[semantic_reward_key] = features
                reward_feature_cache[key] = features
            if features is not None:
                return _transition_reward_from_linear_features_and_coefficients(features, coeffs)
        return transition_reward(
            current_env,
            current_state,
            action,
            branch,
            branch_state,
            cfg,
            lethal_available=lethal_available,
            block_reward_base_state=block_reward_base_state,
            after_phase=branch_phase,
            after_terminal=branch_terminal,
        )

    values_by_config: list[list[float]] = []
    for config_index, cfg in enumerate(configs):
        coeffs = linear_coefficients_by_config[config_index] if linear_reward_compatible else None
        cfg_values: list[float] = []
        for initial_node_id in range(len(initial_nodes)):
            beam: list[tuple[int, int, float]] = [(initial_node_id, 0, 0.0)]
            best_value: float | None = None
            nodes_used = 0
            while beam and nodes_used < cfg.node_budget_per_root:
                next_beam: list[tuple[float, int, int, float]] = []
                for current_node_id, depth, cumulative_value in beam:
                    if nodes_used >= cfg.node_budget_per_root:
                        break
                    current_env, current_state, phase, terminal = nodes[current_node_id]
                    if (
                        terminal is not None
                        or _is_next_player_decision_from_state(current_state, phase=phase, root_turn=root_turn)
                        or depth >= cfg.max_depth
                    ):
                        if best_value is None or cumulative_value > best_value:
                            best_value = float(cumulative_value)
                        continue
                    actions = legal_actions_for(current_node_id, current_env, current_state, phase)
                    if not actions:
                        if best_value is None or cumulative_value > best_value:
                            best_value = float(cumulative_value)
                        continue
                    for action in actions:
                        if nodes_used >= cfg.node_budget_per_root:
                            break
                        expanded = expand(current_node_id, action)
                        if expanded is None:
                            continue
                        child_node_id, branch_state = expanded
                        branch, _state, branch_phase, branch_terminal = nodes[child_node_id]
                        nodes_used += 1
                        reward = reward_for(
                            current_node_id,
                            action,
                            branch,
                            branch_state,
                            config_index,
                            cfg,
                            coeffs,
                            branch_phase=branch_phase,
                            branch_terminal=branch_terminal,
                        )
                        new_value = float(cumulative_value + reward)
                        next_beam.append((new_value, child_node_id, depth + 1, new_value))
                raw_next_beam_len = len(next_beam)
                if raw_next_beam_len > 1:
                    next_beam = _dedupe_batch_next_entries(next_beam, nodes, branch_signature_cache)
                if len(next_beam) > cfg.beam_width:
                    next_beam = nlargest(cfg.beam_width, next_beam, key=itemgetter(0))
                else:
                    next_beam.sort(key=itemgetter(0), reverse=True)
                beam = [
                    (node_id, depth, value)
                    for _priority, node_id, depth, value in next_beam[: cfg.beam_width]
                ]
            if best_value is None:
                best_value = float(beam[0][2]) if beam else 0.0
            cfg_values.append(float(best_value))
        values_by_config.append(cfg_values)
    return values_by_config


def score_root_actions_many_configs_fast(
    root_env: Any,
    configs: list[TeacherConfig],
    *,
    legal_actions: list[dict[str, Any]] | None = None,
) -> list[list[tuple[dict[str, Any], float]]] | None:
    configs = [config or default_teacher_config() for config in configs]
    if not configs or not _teacher_batch_configs_compatible(configs):
        return None
    before_state = _scoring_state_from_env(root_env)
    actions = _root_combat_actions_no_copy(root_env, legal_actions=legal_actions)
    if len(actions) <= 1:
        return [[] for _config in configs]
    has_root_potion = any((action.get("kind") or "") == "potion" for action in actions)

    root_turn = _combat_turn_from_state(before_state)
    root_step_source = _clone_step_source_or_none(root_env)
    root_step_source_prefix = _step_branch_cache_source_prefix(root_step_source)
    expanded: list[tuple[dict[str, Any], Any, dict[str, Any], str, str | None]] = []
    for action in actions:
        try:
            immediate_branch = _step_branch_with_source_cached(
                root_env,
                root_step_source,
                action,
                source_prefix=root_step_source_prefix,
            )
        except Exception:
            return None
        visible_after = _scoring_state_from_env(immediate_branch)
        after_phase = _phase_from_state(immediate_branch, visible_after)
        after_terminal = _is_terminal_from_state(immediate_branch, visible_after, after_phase)
        expanded.append((dict(action), immediate_branch, visible_after, after_phase, after_terminal))

    root_lethal_available = _root_turn_lethal_available_from_branches(
        [(action, branch, visible_after) for action, branch, visible_after, _phase, _terminal in expanded],
        root_turn=root_turn,
        config=configs[0],
    )
    continuation_values = _continuation_search_many_configs_no_potion_tracking(
        [(branch, visible_after, after_phase, after_terminal) for _action, branch, visible_after, after_phase, after_terminal in expanded],
        root_turn=root_turn,
        configs=configs,
        lethal_available=root_lethal_available,
        block_reward_base_state=before_state,
    )

    linear_reward_compatible = _teacher_batch_linear_reward_compatible(configs) and _teacher_env_flag(
        "SPIRECOMM_V3_TEACHER_BATCH_LINEAR_REWARD_VALUES",
        True,
    )
    potion_reward_compatible = _teacher_batch_potion_reward_compatible(configs)
    vector_reward_values = linear_reward_compatible and _teacher_env_flag(
        "SPIRECOMM_V3_TEACHER_VECTOR_REWARD_VALUES",
        True,
    )
    linear_coefficients_by_config = (
        [_linear_reward_coefficients(cfg) for cfg in configs]
        if linear_reward_compatible
        else []
    )
    linear_reward_batch_plan = (
        _linear_reward_batch_plan(linear_coefficients_by_config)
        if linear_coefficients_by_config
        else ((), (), True, True)
    )
    immediate_feature_cache: dict[int, tuple[Any, ...] | None] = {}
    immediate_reward_values_cache: dict[int, list[float] | None] = {}
    immediate_potion_reward_cache: dict[int, float] = {}
    immediate_rewards_by_config: list[list[float]] = []
    for config_index, cfg in enumerate(configs):
        coeffs = linear_coefficients_by_config[config_index] if linear_reward_compatible else None
        rewards: list[float] = []
        for action_index, (action, immediate_branch, visible_after, after_phase, after_terminal) in enumerate(expanded):
            if potion_reward_compatible and (action.get("kind") or "") == "potion":
                if action_index in immediate_potion_reward_cache:
                    rewards.append(float(immediate_potion_reward_cache[action_index]))
                    continue
                value = transition_reward(
                    root_env,
                    before_state,
                    action,
                    immediate_branch,
                    visible_after,
                    cfg,
                    lethal_available=root_lethal_available,
                    block_reward_base_state=before_state,
                    after_phase=after_phase,
                    after_terminal=after_terminal,
                )
                immediate_potion_reward_cache[action_index] = float(value)
                rewards.append(float(value))
                continue
            if vector_reward_values and (action.get("kind") or "") != "potion":
                values = immediate_reward_values_cache.get(action_index)
                if values is not None:
                    rewards.append(float(values[config_index]))
                    continue
                if action_index in immediate_reward_values_cache:
                    values = None
                features = immediate_feature_cache.get(action_index)
                if action_index not in immediate_feature_cache:
                    features = _transition_reward_linear_features(
                        root_env,
                        before_state,
                        action,
                        immediate_branch,
                        visible_after,
                        configs[0],
                        lethal_available=root_lethal_available,
                        block_reward_base_state=before_state,
                        after_phase=after_phase,
                        after_terminal=after_terminal,
                    )
                    immediate_feature_cache[action_index] = features
                if features is not None:
                    values = _transition_reward_values_from_linear_features_and_coefficients(
                        features,
                        linear_coefficients_by_config,
                        linear_reward_batch_plan,
                    )
                    immediate_reward_values_cache[action_index] = values
                    rewards.append(float(values[config_index]))
                    continue
                immediate_reward_values_cache[action_index] = None
            if linear_reward_compatible and (action.get("kind") or "") != "potion":
                features = immediate_feature_cache.get(action_index)
                if action_index not in immediate_feature_cache:
                    features = _transition_reward_linear_features(
                        root_env,
                        before_state,
                        action,
                        immediate_branch,
                        visible_after,
                        configs[0],
                        lethal_available=root_lethal_available,
                        block_reward_base_state=before_state,
                        after_phase=after_phase,
                        after_terminal=after_terminal,
                    )
                    immediate_feature_cache[action_index] = features
                if features is not None:
                    rewards.append(_transition_reward_from_linear_features_and_coefficients(features, coeffs))
                    continue
            rewards.append(
                transition_reward(
                    root_env,
                    before_state,
                    action,
                    immediate_branch,
                    visible_after,
                    cfg,
                    lethal_available=root_lethal_available,
                    block_reward_base_state=before_state,
                    after_phase=after_phase,
                    after_terminal=after_terminal,
                )
            )
        immediate_rewards_by_config.append(rewards)

    non_potion_indices = [
        index for index, (action, _branch, _state, _phase, _terminal) in enumerate(expanded)
        if (action.get("kind") or "") != "potion"
    ]
    blocked_baselines: dict[int, list[float]] = {}
    use_batch_potion_baseline_cache = _teacher_env_flag("SPIRECOMM_V3_TEACHER_BATCH_POTION_BASELINE_CACHE", True)
    blocked_baseline_by_potion_key: dict[tuple[int | None, str], list[float]] = {}
    if has_root_potion and non_potion_indices:
        non_potion_nodes = [
            (expanded[index][1], expanded[index][2], expanded[index][3], expanded[index][4])
            for index in non_potion_indices
        ]
        for action_index, (action, _branch, _state, _phase, _terminal) in enumerate(expanded):
            if (action.get("kind") or "") != "potion":
                continue
            potion_key = (_action_potion_index(action), _action_potion_id(action))
            baselines = blocked_baseline_by_potion_key.get(potion_key) if use_batch_potion_baseline_cache else None
            if baselines is None:
                blocked_values = _continuation_search_many_configs_no_potion_tracking(
                    non_potion_nodes,
                    root_turn=root_turn,
                    configs=configs,
                    lethal_available=root_lethal_available,
                    block_reward_base_state=before_state,
                    blocked_potion_action=action,
                )
                baselines = []
                for config_index in range(len(configs)):
                    values = [
                        float(immediate_rewards_by_config[config_index][root_action_index])
                        + float(blocked_values[config_index][non_potion_pos])
                        for non_potion_pos, root_action_index in enumerate(non_potion_indices)
                    ]
                    baselines.append(max(values) if values else 0.0)
                blocked_baseline_by_potion_key[potion_key] = baselines
            blocked_baselines[action_index] = baselines

    all_scores: list[list[tuple[dict[str, Any], float]]] = []
    survival_status_cache: dict[int, bool | None] | None = (
        {} if _teacher_env_flag("SPIRECOMM_V3_TEACHER_SHARED_SURVIVAL_GUARD_CACHE", True) else None
    )
    expanded_action_views = [
        (action, branch, visible_after)
        for action, branch, visible_after, _phase, _terminal in expanded
    ]
    for config_index, cfg in enumerate(configs):
        score_actions: list[dict[str, Any]] = []
        score_values: list[float] = []
        for action_index, (action, immediate_branch, visible_after, after_phase, after_terminal) in enumerate(expanded):
            immediate_reward = float(immediate_rewards_by_config[config_index][action_index])
            continuation_value = float(continuation_values[config_index][action_index])
            if (action.get("kind") or "") == "potion":
                continuation_value = _root_potion_continuation_value(
                    action,
                    before_state,
                    immediate_reward,
                    continuation_value,
                    cfg,
                    non_potion_baseline=float(blocked_baselines.get(action_index, [0.0] * len(configs))[config_index]),
                )
            score_actions.append(action)
            score_values.append(float(immediate_reward + continuation_value))
        adjusted_scores, _adjustments = _apply_teacher_survival_guard_to_scores(
            before_state,
            expanded_action_views,
            score_values,
            cfg,
            status_cache=survival_status_cache,
        )
        if adjusted_scores != score_values:
            score_values = [float(score) for score in adjusted_scores]
        all_scores.append(list(zip(score_actions, score_values, strict=True)))
    return all_scores


def best_teacher_actions_env_many_configs(
    env: Any,
    configs: list[TeacherConfig],
    *,
    legal_actions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any] | None] | None:
    scores_by_config = score_root_actions_many_configs_fast(env, configs, legal_actions=legal_actions)
    if scores_by_config is None:
        return None
    best_actions: list[dict[str, Any] | None] = []
    for scores in scores_by_config:
        if not scores:
            best_actions.append(None)
        else:
            best_actions.append(dict(max(scores, key=lambda item: item[1])[0]))
    return best_actions


def best_teacher_action_env(
    env: Any,
    *,
    config: TeacherConfig | None = None,
    legal_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    cfg = config or default_teacher_config()
    actions = _root_combat_actions_no_copy(env, legal_actions=legal_actions)
    if len(actions) <= 1:
        return None
    root_step_source: bytes | _TeacherCombatStepSource | None = None
    cache_key: tuple[str, str] | None = None
    use_non_potion_cache = (
        _teacher_env_flag("SPIRECOMM_V3_TEACHER_NON_POTION_ROOT_CACHE", True)
        and not any((action.get("kind") or "") == "potion" for action in actions)
    )
    if use_non_potion_cache:
        root_step_source = _clone_step_source_or_none(env)
        cache_key = _non_potion_root_cache_key(config=cfg, source=root_step_source, actions=actions)
        cached = _get_non_potion_root_cache(cache_key)
        if cached is not None:
            return cached
    scores = score_root_actions_fast(env, config=cfg, legal_actions=actions, root_step_source=root_step_source)
    if not scores:
        return None
    best = dict(max(scores, key=lambda item: item[1])[0])
    if use_non_potion_cache:
        _put_non_potion_root_cache(cache_key, best)
    return best


def _label_root_sample_from_env(
    root_env: Any,
    root: V3CombatRootSample,
    *,
    config: TeacherConfig | None = None,
) -> V3CombatLabeledRoot:
    cfg = config or default_teacher_config()
    before_state = _state_from_env(root_env)
    before_features = encode_state_summary(before_state)
    root_turn = _combat_turn(root_env)
    initial_expanded: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    root_env_blob = _clone_env_blob_or_none(root_env)
    if root_env_blob is not None:
        root.env_blob = root_env_blob
    root_step_source = _clone_step_source_or_none(root_env)
    root_step_source_prefix = _step_branch_cache_source_prefix(root_step_source)
    for action in root.actions:
        immediate_branch = _step_branch_with_source_cached(
            root_env,
            root_step_source,
            action,
            source_prefix=root_step_source_prefix,
        )
        visible_after = _state_from_env(immediate_branch)
        initial_expanded.append((dict(action), immediate_branch, visible_after))
    root_lethal_available = _root_turn_lethal_available_from_branches(
        [(action, branch, visible_after) for action, branch, visible_after in initial_expanded],
        root_turn=root_turn,
        config=cfg,
    )
    has_root_potion = any((action.get("kind") or "") == "potion" for action in root.actions)
    candidates: list[V3CombatCandidateExample] = []
    expanded: list[tuple[dict[str, Any], Any, dict[str, Any], float, dict[str, float], ContinuationResult]] = []
    for action, immediate_branch, visible_after in initial_expanded:
        after_phase = _phase_from_state(immediate_branch, visible_after)
        after_terminal = _is_terminal_from_state(immediate_branch, visible_after, after_phase)
        immediate_components = transition_reward_components(
            root_env,
            before_state,
            action,
            immediate_branch,
            visible_after,
            cfg,
            lethal_available=root_lethal_available,
            block_reward_base_state=before_state,
            after_phase=after_phase,
            after_terminal=after_terminal,
        )
        immediate_reward = float(immediate_components.get("immediate_total", 0.0))
        result = continuation_search(
            immediate_branch,
            root_turn=root_turn,
            config=cfg,
            lethal_available=root_lethal_available,
            block_reward_base_state=before_state,
            track_potion_keys=has_root_potion,
            initial_state=visible_after,
            initial_phase=after_phase,
            initial_terminal=after_terminal,
        )
        expanded.append((dict(action), immediate_branch, visible_after, float(immediate_reward), immediate_components, result))

    non_potion_expanded = [
        (action, branch, visible_after, immediate_reward, result)
        for action, branch, visible_after, immediate_reward, _immediate_components, result in expanded
        if (action.get("kind") or "") != "potion"
    ]
    blocked_baseline_cache: dict[tuple[int | None, str], float] = {}
    baseline_search_stats = {"blocked_search_calls": 0, "blocked_reused_unblocked": 0}

    def non_potion_baseline_for(root_potion_action: dict[str, Any]) -> float:
        cache_key = (_action_potion_index(root_potion_action), _action_potion_id(root_potion_action))
        cached = blocked_baseline_cache.get(cache_key)
        if cached is not None:
            return cached
        values = []
        for _action, branch, _visible_after, immediate_reward, result in non_potion_expanded:
            if _can_reuse_unblocked_continuation(result, root_potion_action):
                baseline_result = result
                baseline_search_stats["blocked_reused_unblocked"] += 1
            else:
                baseline_search_stats["blocked_search_calls"] += 1
                baseline_result = continuation_search(
                    branch,
                    root_turn=root_turn,
                    config=cfg,
                    lethal_available=root_lethal_available,
                    block_reward_base_state=before_state,
                    blocked_potion_action=root_potion_action,
                    track_potion_keys=False,
                    initial_state=_visible_after,
                )
            values.append(float(immediate_reward) + float(baseline_result.value))
        baseline = max(values) if values else 0.0
        blocked_baseline_cache[cache_key] = float(baseline)
        return float(baseline)

    for action, _immediate_branch, visible_after, immediate_reward, immediate_components, result in expanded:
        debug_line = [dict(action), *result.debug_best_line]
        non_potion_baseline = (
            non_potion_baseline_for(action) if (action.get("kind") or "") == "potion" else 0.0
        )
        continuation_value = _root_potion_continuation_value(
            action,
            before_state,
            immediate_reward,
            result.value,
            cfg,
            non_potion_baseline=non_potion_baseline,
        )
        teacher_q = float(immediate_reward + continuation_value)
        reward_components = _complete_reward_components(
            immediate_components,
            action=action,
            immediate_reward=immediate_reward,
            continuation_raw=float(result.value),
            continuation_adjusted=float(continuation_value),
            non_potion_baseline=float(non_potion_baseline),
            teacher_q=teacher_q,
        )
        after_features = encode_state_summary(visible_after)
        delta_features = encode_delta(before_state, action, visible_after)
        features = before_features + encode_action_summary(before_state, action) + after_features + delta_features
        key = action_key(action, before_state)
        candidates.append(
            V3CombatCandidateExample(
                action=dict(action),
                action_key=key,
                visible_after=visible_after,
                delta_features=delta_features,
                candidate_features=features,
                teacher_q=teacher_q,
                continuation_depth=int(result.depth),
                continuation_nodes=int(result.nodes),
                terminal_kind=result.terminal_kind,
                debug_best_line=debug_line,
                reward_components=reward_components,
                is_chosen=root.chosen_action_key == key,
            )
        )
    adjusted_scores, survival_guard_adjustments = _apply_teacher_survival_guard_to_scores(
        before_state,
        [
            (action, immediate_branch, visible_after)
            for action, immediate_branch, visible_after, _immediate_reward, _immediate_components, _result in expanded
        ],
        [float(candidate.teacher_q) for candidate in candidates],
        cfg,
    )
    if survival_guard_adjustments:
        for index, candidate in enumerate(candidates):
            old_score = float(candidate.teacher_q)
            new_score = float(adjusted_scores[index])
            candidate.teacher_q = new_score
            candidate.reward_components["teacher_q"] = new_score
            candidate.reward_components["teacher_survival_guard_adjustment"] = float(new_score - old_score)
    ranked = sorted(range(len(candidates)), key=lambda index: candidates[index].teacher_q, reverse=True)
    for rank, index in enumerate(ranked):
        candidates[index].teacher_rank = rank
    return V3CombatLabeledRoot(
        root=root,
        candidates=candidates,
        teacher_config={"version": TEACHER_VERSION, **asdict(cfg), **baseline_search_stats},
    )


def label_root_sample(root: V3CombatRootSample, config: TeacherConfig | None = None) -> V3CombatLabeledRoot:
    return _label_root_sample_from_env(root.load_env(), root, config=config)


def label_env(
    env: Any,
    *,
    root_id: str,
    source: str,
    config: TeacherConfig | None = None,
    legal_actions: list[dict[str, Any]] | None = None,
    validate_action_keys: bool = True,
) -> V3CombatLabeledRoot | None:
    before_state = _state_from_env(env)
    actions = root_combat_actions(env, legal_actions=legal_actions)
    if len(actions) <= 1:
        return None
    if validate_action_keys and not action_keys_are_unique(actions, before_state):
        return None
    env_blob = _clone_env_blob_or_none(env)
    if env_blob is None:
        from spirecomm.ai.v3_combat_dataset import make_root_sample

        root = make_root_sample(env, root_id=root_id, source=source)
        if root is None:
            return None
        return label_root_sample(root, config=config)
    root = V3CombatRootSample(
        root_id=root_id,
        source=source,
        env_blob=env_blob,
        visible_before=before_state,
        actions=actions,
        action_keys=[action_key(action, before_state) for action in actions]
        if validate_action_keys
        else [(index,) for index, _action in enumerate(actions)],
        chosen_action_key=None,
        metadata={},
    )
    return _label_root_sample_from_env(env, root, config=config)


def best_teacher_action(labeled: V3CombatLabeledRoot) -> dict[str, Any] | None:
    if not labeled.candidates:
        return None
    return min(labeled.candidates, key=lambda candidate: candidate.teacher_rank).action


def candidate_rank_by_card(labeled: V3CombatLabeledRoot, card_id: str) -> int | None:
    wanted = str(card_id)
    ranks = [
        candidate.teacher_rank
        for candidate in labeled.candidates
        if str(candidate.action.get("card_id") or "") == wanted
    ]
    if not ranks:
        return None
    return min(ranks)
