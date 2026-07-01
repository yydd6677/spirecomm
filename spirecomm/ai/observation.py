from __future__ import annotations

from typing import Any

from spirecomm.native_sim.cards import CARD_LIBRARY


CANONICAL_STATE_VERSION = "canonical_v2"
LEGACY_COMBAT_OBSERVATION_VERSION = "legacy_v1"
EXTENDED_COMBAT_OBSERVATION_VERSION = "extended_v1"


def _value_from_like(value_like: Any, *names: str, default: Any = None) -> Any:
    if value_like is None:
        return default
    if isinstance(value_like, dict):
        for name in names:
            if name in value_like:
                return value_like[name]
        return default
    for name in names:
        if hasattr(value_like, name):
            return getattr(value_like, name)
    return default


def _bool_from_like(value_like: Any, *names: str, default: bool = False) -> bool:
    return bool(_value_from_like(value_like, *names, default=default))


def _int_from_like(value_like: Any, *names: str, default: int | None = None) -> int | None:
    value = _value_from_like(value_like, *names, default=default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_power_id(power_id: str | None) -> str:
    if not power_id:
        return ""
    if power_id == "Anger":
        return "Angry"
    return str(power_id)


def _base_cost_from_library(card_id: str | None, upgrades: int, fallback_cost: int | None) -> int | None:
    if not card_id:
        return fallback_cost
    card_def = CARD_LIBRARY.get(str(card_id))
    if card_def is None:
        return fallback_cost
    if upgrades > 0 and card_def.upgraded_cost is not None:
        return int(card_def.upgraded_cost)
    return int(card_def.cost)


def canonicalize_card(card_like: Any) -> dict[str, Any]:
    if card_like is None:
        return {}

    card_id = str(_value_from_like(card_like, "card_id", "id", default="") or "")
    upgrades = _int_from_like(card_like, "upgrades", default=0) or 0
    raw_cost = _int_from_like(card_like, "cost", default=None)
    raw_cost_for_turn = _int_from_like(card_like, "cost_for_turn", default=None)
    raw_cost_for_combat = _int_from_like(card_like, "cost_for_combat", default=None)
    base_cost = _int_from_like(card_like, "base_cost", default=None)
    if base_cost is None:
        base_cost = _base_cost_from_library(card_id, upgrades, raw_cost)

    effective_cost = raw_cost_for_turn if raw_cost_for_turn is not None else raw_cost
    if effective_cost is None:
        effective_cost = base_cost

    return {
        "card_id": card_id,
        "name": str(_value_from_like(card_like, "name", default=card_id) or card_id),
        "uuid": str(_value_from_like(card_like, "uuid", default="") or ""),
        "type": str(_value_from_like(card_like, "type", default="") or ""),
        "rarity": str(_value_from_like(card_like, "rarity", default="") or ""),
        "cost": effective_cost,
        "base_cost": base_cost,
        "cost_for_turn": effective_cost,
        "cost_for_combat": raw_cost_for_combat,
        "free_to_play_once": _bool_from_like(card_like, "free_to_play_once", default=False),
        "upgrades": upgrades,
        "misc": _int_from_like(card_like, "misc", default=0) or 0,
        "has_target": _bool_from_like(card_like, "has_target", default=False),
        "is_playable": _bool_from_like(card_like, "is_playable", default=False),
        "exhausts": _bool_from_like(card_like, "exhausts", default=False),
    }


def canonicalize_power(power_like: Any) -> dict[str, Any]:
    if power_like is None:
        return {}
    power_id = _normalize_power_id(
        str(
            _value_from_like(
                power_like,
                "power_id",
                "id",
                "name",
                default="",
            )
            or ""
        )
    )
    amount = _int_from_like(power_like, "amount", "misc", default=0) or 0
    return {
        "power_id": power_id,
        "id": power_id,
        "name": power_id,
        "amount": amount,
        "card": _value_from_like(power_like, "card", default=None),
        "damage": _int_from_like(power_like, "damage", default=0) or 0,
        "just_applied": _bool_from_like(power_like, "just_applied", default=False),
        "misc": _int_from_like(power_like, "misc", default=amount) or amount,
    }


def canonicalize_powers(powers: list[Any] | None) -> list[dict[str, Any]]:
    return [canonicalize_power(power) for power in powers or [] if power is not None]


def canonicalize_monster(monster_like: Any) -> dict[str, Any]:
    if monster_like is None:
        return {}
    return {
        "monster_index": _int_from_like(monster_like, "monster_index", default=0) or 0,
        "name": str(_value_from_like(monster_like, "name", default="") or ""),
        "monster_id": str(_value_from_like(monster_like, "monster_id", default="") or ""),
        "current_hp": _int_from_like(monster_like, "current_hp", default=0) or 0,
        "max_hp": _int_from_like(monster_like, "max_hp", default=0) or 0,
        "block": _int_from_like(monster_like, "block", default=0) or 0,
        "intent": str(_value_from_like(monster_like, "intent", default="UNKNOWN") or "UNKNOWN"),
        "half_dead": _bool_from_like(monster_like, "half_dead", default=False),
        "is_gone": _bool_from_like(monster_like, "is_gone", default=False),
        "move_id": _value_from_like(monster_like, "move_id", default=None),
        "last_move_id": _value_from_like(monster_like, "last_move_id", default=None),
        "second_last_move_id": _value_from_like(monster_like, "second_last_move_id", default=None),
        "move_base_damage": _int_from_like(monster_like, "move_base_damage", default=0) or 0,
        "move_adjusted_damage": _int_from_like(monster_like, "move_adjusted_damage", default=0) or 0,
        "move_hits": _int_from_like(monster_like, "move_hits", default=0) or 0,
        "powers": canonicalize_powers(_value_from_like(monster_like, "powers", default=[])),
    }


def canonicalize_serialized_state(serialized_state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(serialized_state or {})
    state["observation_version"] = str(
        state.get("observation_version")
        or state.get("schema_version")
        or CANONICAL_STATE_VERSION
    )
    state["deck"] = [canonicalize_card(card) for card in state.get("deck") or []]

    combat_state = state.get("combat_state")
    if combat_state is not None:
        combat_state = dict(combat_state or {})
        player = dict(combat_state.get("player") or {})
        player["powers"] = canonicalize_powers(player.get("powers"))
        combat_state["player"] = player
        combat_state["hand"] = [canonicalize_card(card) for card in combat_state.get("hand") or []]
        combat_state["draw_pile"] = [canonicalize_card(card) for card in combat_state.get("draw_pile") or []]
        combat_state["discard_pile"] = [canonicalize_card(card) for card in combat_state.get("discard_pile") or []]
        combat_state["exhaust_pile"] = [canonicalize_card(card) for card in combat_state.get("exhaust_pile") or []]
        combat_state["limbo"] = [canonicalize_card(card) for card in combat_state.get("limbo") or []]
        card_in_play = combat_state.get("card_in_play")
        combat_state["card_in_play"] = canonicalize_card(card_in_play) if card_in_play else None
        combat_state["monsters"] = [canonicalize_monster(monster) for monster in combat_state.get("monsters") or []]
        state["combat_state"] = combat_state
    else:
        state["combat_state"] = None

    return state


def legacy_monster_power_alias(power_id: str) -> str:
    if power_id == "Anger":
        return "Angry"
    return power_id


__all__ = [
    "CANONICAL_STATE_VERSION",
    "LEGACY_COMBAT_OBSERVATION_VERSION",
    "EXTENDED_COMBAT_OBSERVATION_VERSION",
    "canonicalize_card",
    "canonicalize_monster",
    "canonicalize_powers",
    "canonicalize_serialized_state",
    "legacy_monster_power_alias",
]
