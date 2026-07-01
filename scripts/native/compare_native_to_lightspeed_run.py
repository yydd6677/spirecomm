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
import os
import random
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

def _card_cost_from_state(action: dict[str, Any], state: dict[str, Any] | None) -> Any:
    if state is None:
        return None
    combat = state.get("combat_state") if isinstance(state, dict) else None
    if not isinstance(combat, dict):
        return None
    hand = combat.get("hand")
    if not isinstance(hand, list):
        return None
    card_index = action.get("card_index")
    if card_index is None:
        card_index = action.get("source_index")
    if card_index is None:
        card_index = action.get("select_index")
    try:
        card_index = int(card_index)
    except (TypeError, ValueError):
        return None
    if not (0 <= card_index < len(hand)):
        return None
    card = hand[card_index]
    if not isinstance(card, dict):
        return None
    cost_for_turn = card.get("cost_for_turn")
    if cost_for_turn is not None:
        return cost_for_turn
    return card.get("cost")


def _norm_action(action: dict[str, Any], state: dict[str, Any] | None = None) -> tuple[Any, ...]:
    def _id_token(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        token = "".join(ch.lower() for ch in value if ch.isalnum())
        if token == "paperfrog":
            return "paperphrog"
        if token == "ghostly":
            return "apparition"
        return token

    def _select_type_token(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        mapping = {
            "NEOW_UPGRADE": "UPGRADE",
            "EVENT_UPGRADE": "UPGRADE",
            "CAMPFIRE_SMITH": "UPGRADE",
            "NEOW_REMOVE": "REMOVE",
            "EVENT_REMOVE": "REMOVE",
            "CAMPFIRE_TOKE": "REMOVE",
            "NEOW_TRANSFORM": "TRANSFORM",
            "EVENT_TRANSFORM": "TRANSFORM",
        }
        return mapping.get(value, value)

    def _event_choice_token(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("OPTION_"):
            suffix = value.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
        return value

    kind = action.get("kind")
    if kind == "map":
        symbol = action.get("symbol")
        if symbol == "E_GREEN":
            symbol = "E"
        if symbol == "ACT4_ELITE":
            symbol = "E"
        return (kind, symbol, action.get("x"))
    if kind == "card_reward":
        return (kind, _id_token(action.get("card_id")))
    if kind in {"skip", "reward_skip"}:
        return ("skip", "SKIP")
    if kind == "boss_relic":
        return (kind, _id_token(action.get("relic_id")), _id_token(action.get("name")))
    if kind == "reward_relic":
        return (kind, _id_token(action.get("relic_id")), _id_token(action.get("name")))
    if kind == "reward_potion":
        return (kind, _id_token(action.get("potion_id")))
    if kind == "reward_gold":
        return (kind, action.get("name"))
    if kind == "reward_key":
        return (kind, action.get("name"), None)
    if kind == "campfire":
        return (kind, action.get("name"))
    if kind == "event":
        event_id = action.get("event_id")
        if event_id == "Transmorgrifier":
            event_id = "Transmogrifier"
        if isinstance(event_id, str) and event_id.endswith("(?)"):
            event_id = event_id[:-3]
        if event_id == "NEOW":
            return (kind, _event_choice_token(action.get("name")), event_id)
        choice_index = action.get("choice_index")
        token = choice_index if choice_index is not None else _event_choice_token(action.get("name"))
        return (kind, token, event_id)
    if kind == "neow":
        return ("event", _event_choice_token(action.get("name")), "NEOW")
    if kind == "shop":
        item_kind = action.get("item_kind")
        item_id = action.get("item_id")
        if item_kind in {"leave", "purge"}:
            item_id = item_kind
        return (kind, item_kind, _id_token(item_id), int(action.get("price", 0) or 0))
    if kind == "treasure":
        return (kind, action.get("name"), None)
    if kind == "card_select":
        deck_index = action.get("deck_index")
        if deck_index is None or int(deck_index) < 0:
            deck_index = action.get("select_index")
        if deck_index is None or int(deck_index) < 0:
            deck_index = action.get("choice_index")
        if deck_index is None or int(deck_index) < 0:
            deck_index = action.get("idx1")
        select_type = _select_type_token(action.get("select_type") or action.get("name"))
        return ("card_select", select_type, int(deck_index) if deck_index is not None else None)
    if kind in {"single_card_select", "multi_card_select"}:
        deck_index = action.get("deck_index")
        if deck_index is None or int(deck_index) < 0:
            deck_index = action.get("select_index")
        if deck_index is None or int(deck_index) < 0:
            deck_index = action.get("choice_index")
        if deck_index is None or int(deck_index) < 0:
            deck_index = action.get("idx1")
        select_type = _select_type_token(action.get("select_type") or action.get("name"))
        return ("card_select", select_type, int(deck_index) if deck_index is not None else None)
    if kind == "end":
        return (kind, action.get("name"))
    if kind == "card":
        cost = _card_cost_from_state(action, state)
        return (kind, _id_token(action.get("card_id")), cost, action.get("target_index"))
    if kind == "potion":
        potion_index = action.get("potion_index")
        if potion_index is None:
            potion_index = action.get("source_index")
        if potion_index is None:
            potion_index = action.get("select_index")
        return (kind, action.get("action"), _id_token(action.get("potion_id")), int(potion_index) if potion_index is not None else None, action.get("target_index"))
    return (kind, action.get("name"), action.get("choice_index"))


def _choice_list_signature(choices: list[dict[str, Any]], state: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
    return sorted(set(_norm_action(choice, state) for choice in choices))


def _pick_action_by_signature(actions: list[dict[str, Any]], signature: tuple[Any, ...], state: dict[str, Any] | None = None) -> dict[str, Any]:
    for action in actions:
        if _norm_action(action, state) == signature:
            return action
    raise KeyError(f"signature not found in actions: {signature!r}")


def _battle_signature(state: dict[str, Any]) -> dict[str, Any]:
    combat = state["combat_state"]

    def _norm_card_id(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        token = "".join(ch.lower() for ch in value if ch.isalnum())
        if token == "ghostly":
            return "apparition"
        return token

    def _power_sig(entity: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
        powers = []
        for power in entity.get("powers", []):
            powers.append(
                (
                    power.get("power_id") or power.get("id") or power.get("name"),
                    power.get("amount", 0),
                    bool(power.get("just_applied", False)),
                )
            )
        return tuple(sorted(powers))

    return {
        "player_hp": combat["player"]["current_hp"],
        "player_block": combat["player"]["block"],
        "energy": combat["player"]["energy"],
        "player_powers": _power_sig(combat["player"]),
        "hand": [_norm_card_id(card["card_id"]) for card in combat["hand"]],
        "draw_pile": [_norm_card_id(card["card_id"]) for card in combat["draw_pile"]],
        "discard_pile": [_norm_card_id(card["card_id"]) for card in combat["discard_pile"]],
        "exhaust_pile": [_norm_card_id(card["card_id"]) for card in combat.get("exhaust_pile", [])],
        "monsters": [
            (
                monster["monster_id"],
                monster.get("current_hp"),
                monster.get("block"),
                monster.get("move_name"),
                monster.get("intent"),
                monster.get("move_base_damage"),
                monster.get("move_hits"),
                _power_sig(monster),
            )
            for monster in combat["monsters"]
        ],
    }


def _categorize_mismatch(result: dict[str, Any]) -> str:
    reason = result.get("reason")
    if reason == "finished":
        return "run outcome / map progression"
    if reason == "battle_phase_mismatch":
        return "run outcome / map progression"
    if reason == "battle_state_mismatch":
        lightspeed_state = result.get("lightspeed", {}) if isinstance(result.get("lightspeed"), dict) else {}
        native_state = result.get("native", {}) if isinstance(result.get("native"), dict) else {}
        if (
            lightspeed_state.get("hand") == native_state.get("hand")
            and lightspeed_state.get("draw_pile") == native_state.get("draw_pile")
            and lightspeed_state.get("discard_pile") == native_state.get("discard_pile")
            and lightspeed_state.get("exhaust_pile") == native_state.get("exhaust_pile")
            and lightspeed_state.get("monsters") != native_state.get("monsters")
        ):
            return "monster AI / intent"
        trace_text = json.dumps(result.get("trace_tail", []), ensure_ascii=False).lower()
        lightspeed = json.dumps(result.get("lightspeed", {}), ensure_ascii=False).lower()
        native = json.dumps(result.get("native", {}), ensure_ascii=False).lower()
        haystack = "\n".join([trace_text, lightspeed, native])
        if any(
            token in haystack
            for token in (
                "armaments",
                "burningpact",
                "doubletap",
                "discovery",
                "exhume",
                "forethought",
                "havoc",
                "headbutt",
                "inkbottle",
                "letteropener",
                "necronomicon",
                "unceasingtop",
                "warcry",
            )
        ):
            return "autoplay/after_use"
        if any(token in haystack for token in ("draw_pile", "discard_pile", "exhaust", "dazed", "burn", "slimed", "wound")):
            return "draw/discard/exhaust"
        if any(token in haystack for token in ("move_name", "intent", "slime", "sentry", "chosen", "byrd", "cultist")):
            return "monster AI / intent"
        return "battle-state mismatch"
    if reason == "legal_action_mismatch":
        lightspeed_actions = json.dumps(result.get("lightspeed_actions", []), ensure_ascii=False).lower()
        native_actions = json.dumps(result.get("native_actions", []), ensure_ascii=False).lower()
        haystack = "\n".join([lightspeed_actions, native_actions])
        if "reward_potion" in haystack or "reward_gold" in haystack or "card_reward" in haystack:
            return "reward/potion"
        return "action legality"
    return "other"


def _serialize_signature(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize_signature(item) for item in value]
    if isinstance(value, list):
        return [_serialize_signature(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_signature(item)
            for key, item in value.items()
        }
    return value


def _stable_key(value: Any) -> str:
    return json.dumps(_serialize_signature(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_step(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _phase_signature(result: dict[str, Any]) -> str:
    if isinstance(result.get("lightspeed_phase"), str):
        return str(result.get("lightspeed_phase"))
    if isinstance(result.get("native_phase"), str):
        return str(result.get("native_phase"))
    trace_tail = result.get("trace_tail")
    if isinstance(trace_tail, list) and trace_tail:
        phase = trace_tail[-1].get("phase")
        if isinstance(phase, str):
            return phase
    return "UNKNOWN"


def _last_action_signature(result: dict[str, Any]) -> list[Any] | None:
    trace_tail = result.get("trace_tail")
    if not isinstance(trace_tail, list) or not trace_tail:
        return None
    choice = trace_tail[-1].get("choice")
    if not isinstance(choice, (list, tuple)):
        return None
    return _serialize_signature(choice)


def _compact_choice_signature(choice: Any) -> list[Any] | None:
    if not isinstance(choice, (list, tuple)) or not choice:
        return None
    kind = choice[0]
    if kind in {"card", "card_select"}:
        return [kind, choice[1] if len(choice) > 1 else None]
    if kind == "potion":
        payload = choice[2] if len(choice) > 2 else (choice[1] if len(choice) > 1 else None)
        return [kind, payload]
    if kind in {"event", "map", "end", "boss_relic", "reward_relic", "card_reward"}:
        return [kind, choice[1] if len(choice) > 1 else None]
    if kind == "skip":
        return ["skip", "SKIP"]
    return [kind]


def _recent_action_window_signature(result: dict[str, Any], *, limit: int = 3) -> list[list[Any]]:
    trace_tail = result.get("trace_tail")
    if not isinstance(trace_tail, list):
        return []
    window: list[list[Any]] = []
    for entry in reversed(trace_tail):
        compact = _compact_choice_signature(entry.get("choice"))
        if compact is None:
            continue
        if compact[0] in {"reward_gold", "reward_potion"}:
            continue
        window.append(compact)
        if len(window) >= limit:
            break
    window.reverse()
    return window


def _battle_trigger_signature(result: dict[str, Any], *, limit: int = 2) -> list[list[Any]]:
    trace_tail = result.get("trace_tail")
    if not isinstance(trace_tail, list):
        return []
    battle_entries = [entry for entry in trace_tail if entry.get("phase") == "BATTLE"]
    source_entries = battle_entries if battle_entries else trace_tail
    trigger: list[list[Any]] = []
    for entry in reversed(source_entries):
        compact = _compact_choice_signature(entry.get("choice"))
        if compact is None:
            continue
        if compact[0] not in {"card", "card_select", "potion", "end"}:
            continue
        trigger.append(compact)
        if len(trigger) >= limit:
            break
    trigger.reverse()
    if trigger:
        return trigger
    return _recent_action_window_signature(result, limit=limit)


def _encounter_signature_from_state(state: Any) -> list[str]:
    if not isinstance(state, dict):
        return []
    monsters = state.get("monsters")
    if not isinstance(monsters, list):
        return []
    encounter: list[str] = []
    for monster in monsters:
        if not isinstance(monster, (list, tuple)) or not monster:
            continue
        monster_id = monster[0]
        if monster_id == "INVALID = 0":
            continue
        encounter.append(str(monster_id))
    return encounter


def _encounter_signature(result: dict[str, Any]) -> list[str]:
    for key in ("lightspeed", "native"):
        encounter = _encounter_signature_from_state(result.get(key))
        if encounter:
            return encounter
    return []


def _summarize_counter_delta(left: Counter[Any], right: Counter[Any]) -> dict[str, Any]:
    left_only = sorted((key, count) for key, count in (left - right).items())
    right_only = sorted((key, count) for key, count in (right - left).items())
    return {
        "left_only": _serialize_signature(left_only[:4]),
        "right_only": _serialize_signature(right_only[:4]),
    }


def _list_delta(left: Any, right: Any) -> dict[str, Any] | None:
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    if left == right:
        return None
    if Counter(left) == Counter(right):
        mismatch_index = next(
            (
                index
                for index, (left_value, right_value) in enumerate(zip(left, right))
                if left_value != right_value
            ),
            min(len(left), len(right)),
        )
        left_value = left[mismatch_index] if mismatch_index < len(left) else None
        right_value = right[mismatch_index] if mismatch_index < len(right) else None
        return {
            "kind": "order",
            "index": mismatch_index,
            "left": left_value,
            "right": right_value,
            "len_left": len(left),
            "len_right": len(right),
        }
    return {
        "kind": "multiset",
        "len_left": len(left),
        "len_right": len(right),
        **_summarize_counter_delta(Counter(left), Counter(right)),
    }


def _power_delta(left: Any, right: Any) -> dict[str, Any] | None:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return None
    left_items = [tuple(item) for item in left]
    right_items = [tuple(item) for item in right]
    if left_items == right_items:
        return None
    return _summarize_counter_delta(Counter(left_items), Counter(right_items))


def _monster_diff_entries(left: Any, right: Any) -> list[dict[str, Any]]:
    if not isinstance(left, list) or not isinstance(right, list):
        return []
    diffs: list[dict[str, Any]] = []
    max_len = max(len(left), len(right))
    for index in range(max_len):
        left_monster = left[index] if index < len(left) else None
        right_monster = right[index] if index < len(right) else None
        if left_monster == right_monster:
            continue
        left_id = left_monster[0] if isinstance(left_monster, (list, tuple)) and left_monster else None
        right_id = right_monster[0] if isinstance(right_monster, (list, tuple)) and right_monster else None
        if left_monster is None or right_monster is None:
            diffs.append(
                {
                    "index": index,
                    "monster_id": left_id or right_id,
                    "kind": "missing_slot",
                    "delta": {
                        "lightspeed": _serialize_signature(left_monster),
                        "native": _serialize_signature(right_monster),
                    },
                }
            )
            continue
        if left_id != right_id:
            diffs.append(
                {
                    "index": index,
                    "monster_id": left_id or right_id,
                    "kind": "identity",
                    "delta": {
                        "lightspeed": _serialize_signature(left_monster[:7]),
                        "native": _serialize_signature(right_monster[:7]),
                    },
                }
            )
            continue
        field_changes: list[str] = []
        if left_monster[1] != right_monster[1]:
            field_changes.append("hp")
        if left_monster[2] != right_monster[2]:
            field_changes.append("block")
        if left_monster[3] != right_monster[3]:
            field_changes.append("move_name")
        if left_monster[4] != right_monster[4]:
            field_changes.append("intent")
        if left_monster[5] != right_monster[5] or left_monster[6] != right_monster[6]:
            field_changes.append("damage_profile")
        if left_monster[7] != right_monster[7]:
            field_changes.append("powers")
        if field_changes == ["powers"]:
            kind = "power_only"
            delta = _power_delta(left_monster[7], right_monster[7])
        elif field_changes == ["hp"]:
            kind = "hp_only"
            delta = {"lightspeed": left_monster[1], "native": right_monster[1]}
        elif field_changes == ["block"]:
            kind = "block_only"
            delta = {"lightspeed": left_monster[2], "native": right_monster[2]}
        elif set(field_changes).issubset({"move_name", "intent", "damage_profile"}):
            kind = "move_intent"
            delta = {
                "lightspeed": _serialize_signature(left_monster[3:7]),
                "native": _serialize_signature(right_monster[3:7]),
            }
        elif set(field_changes).issubset({"hp", "block"}):
            kind = "hp_block"
            delta = {
                "lightspeed": _serialize_signature(left_monster[1:3]),
                "native": _serialize_signature(right_monster[1:3]),
            }
        else:
            kind = "mixed"
            delta = {
                "changed_fields": field_changes,
                "lightspeed": _serialize_signature(left_monster[:8]),
                "native": _serialize_signature(right_monster[:8]),
            }
        diffs.append(
            {
                "index": index,
                "monster_id": left_id,
                "kind": kind,
                "delta": _serialize_signature(delta),
            }
        )
    return diffs


def _battle_state_delta(result: dict[str, Any]) -> dict[str, Any]:
    lightspeed = result.get("lightspeed")
    native = result.get("native")
    if not isinstance(lightspeed, dict) or not isinstance(native, dict):
        return {"changed_fields": [], "primary_field": None, "primary_delta": None}
    field_order = (
        "player_hp",
        "player_block",
        "energy",
        "player_powers",
        "hand",
        "draw_pile",
        "discard_pile",
        "exhaust_pile",
        "monsters",
    )
    field_deltas: dict[str, Any] = {}
    for field in field_order:
        left_value = lightspeed.get(field)
        right_value = native.get(field)
        if left_value == right_value:
            continue
        if field in {"hand", "draw_pile", "discard_pile", "exhaust_pile"}:
            field_deltas[field] = _list_delta(left_value, right_value)
        elif field == "player_powers":
            field_deltas[field] = _power_delta(left_value, right_value)
        elif field == "monsters":
            field_deltas[field] = _monster_diff_entries(left_value, right_value)
        else:
            field_deltas[field] = {
                "lightspeed": _serialize_signature(left_value),
                "native": _serialize_signature(right_value),
            }
    changed_fields = [field for field in field_order if field in field_deltas]
    primary_field = changed_fields[0] if changed_fields else None
    return {
        "changed_fields": changed_fields,
        "primary_field": primary_field,
        "primary_delta": _serialize_signature(field_deltas.get(primary_field)),
        "field_deltas": _serialize_signature(field_deltas),
    }


def _delta_shape(primary_field: Any, primary_delta: Any) -> dict[str, Any]:
    if primary_field is None:
        return {"field": None, "kind": "none"}
    if not isinstance(primary_delta, dict):
        return {
            "field": primary_field,
            "kind": "value",
            "value": _serialize_signature(primary_delta),
        }
    if primary_field in {"player_hp", "player_block", "energy"}:
        left_value = primary_delta.get("lightspeed")
        right_value = primary_delta.get("native")
        relation = "same"
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            if right_value < left_value:
                relation = "native_less"
            elif right_value > left_value:
                relation = "native_more"
        return {
            "field": primary_field,
            "kind": "scalar",
            "relation": relation,
        }
    if primary_field == "player_powers":
        left_only = [item[0][0] for item in primary_delta.get("left_only", []) if item and item[0]]
        right_only = [item[0][0] for item in primary_delta.get("right_only", []) if item and item[0]]
        return {
            "field": primary_field,
            "kind": "power_delta",
            "left_only": left_only[:3],
            "right_only": right_only[:3],
        }
    kind = primary_delta.get("kind")
    if kind == "multiset":
        return {
            "field": primary_field,
            "kind": kind,
            "left_only": [item[0] for item in primary_delta.get("left_only", [])][:3],
            "right_only": [item[0] for item in primary_delta.get("right_only", [])][:3],
        }
    if kind == "order":
        return {
            "field": primary_field,
            "kind": kind,
            "left": primary_delta.get("left"),
            "right": primary_delta.get("right"),
        }
    return {
        "field": primary_field,
        "kind": kind or "delta",
    }


def _legal_action_delta(result: dict[str, Any]) -> dict[str, Any]:
    lightspeed_actions = [_serialize_signature(action) for action in result.get("lightspeed_actions", [])]
    native_actions = [_serialize_signature(action) for action in result.get("native_actions", [])]
    lightspeed_set = {_stable_key(action): action for action in lightspeed_actions}
    native_set = {_stable_key(action): action for action in native_actions}
    missing = [lightspeed_set[key] for key in sorted(set(lightspeed_set) - set(native_set))]
    extra = [native_set[key] for key in sorted(set(native_set) - set(lightspeed_set))]
    return {
        "missing_actions": missing[:4],
        "extra_actions": extra[:4],
        "missing_count": len(missing),
        "extra_count": len(extra),
    }


def _cluster_key(parts: list[Any]) -> str:
    return " | ".join(_stable_key(part) for part in parts)


def _build_cluster_metadata(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("match", False):
        return {}
    category = result.get("category") or _categorize_mismatch(result)
    encounter_sig = _encounter_signature(result)
    phase_sig = _phase_signature(result)
    last_action_sig = _last_action_signature(result)
    trigger_window_sig = _recent_action_window_signature(result)
    floor_sig = result.get("native_floor", result.get("lightspeed_floor"))
    common = {
        "encounter_sig": encounter_sig,
        "phase_sig": phase_sig,
        "last_action_sig": last_action_sig,
        "trigger_window_sig": trigger_window_sig,
        "floor_sig": floor_sig,
    }
    if category == "monster AI / intent":
        battle_delta = _battle_state_delta(result)
        monster_diffs = battle_delta.get("field_deltas", {}).get("monsters", [])
        primary = monster_diffs[0] if monster_diffs else {"monster_id": None, "kind": "unknown", "delta": None}
        delta_sig = {
            "kind": "monster_diff",
            "primary": primary,
            "monster_diffs": monster_diffs[:3],
        }
        cluster_key = _cluster_key(
            [
                category,
                encounter_sig,
                primary.get("monster_id"),
                primary.get("kind"),
                primary.get("delta"),
            ]
        )
        cluster_features = {
            **common,
            "diff_kind": primary.get("kind"),
            "focus_monster_id": primary.get("monster_id"),
            "delta_sig": delta_sig,
        }
        return {
            "encounter_sig": encounter_sig,
            "last_action_sig": last_action_sig,
            "trigger_window_sig": trigger_window_sig,
            "delta_sig": delta_sig,
            "cluster_features": cluster_features,
            "cluster_key": cluster_key,
        }
    if category in {"autoplay/after_use", "draw/discard/exhaust"}:
        battle_delta = _battle_state_delta(result)
        trigger_sig = _battle_trigger_signature(result)
        preferred_fields = (
            "hand",
            "draw_pile",
            "discard_pile",
            "exhaust_pile",
            "player_hp",
            "player_block",
            "energy",
            "player_powers",
            "monsters",
        )
        primary_field = next(
            (
                field
                for field in preferred_fields
                if field in battle_delta.get("changed_fields", [])
            ),
            battle_delta.get("primary_field"),
        )
        field_deltas = battle_delta.get("field_deltas", {})
        primary_delta = field_deltas.get(primary_field)
        delta_shape = _delta_shape(primary_field, primary_delta)
        delta_sig = {
            "kind": "battle_state_delta",
            "primary_field": primary_field,
            "primary_delta": primary_delta,
            "changed_fields": battle_delta.get("changed_fields", []),
            "delta_shape": delta_shape,
        }
        cluster_key = _cluster_key(
            [
                category,
                phase_sig,
                trigger_sig,
                delta_shape,
            ]
        )
        cluster_features = {
            **common,
            "trigger_sig": trigger_sig,
            "delta_kind": primary_field,
            "delta_sig": delta_sig,
        }
        return {
            "encounter_sig": encounter_sig,
            "last_action_sig": last_action_sig,
            "trigger_window_sig": trigger_window_sig,
            "trigger_sig": trigger_sig,
            "delta_sig": delta_sig,
            "cluster_features": cluster_features,
            "cluster_key": cluster_key,
        }
    if category == "action legality":
        legal_delta = _legal_action_delta(result)
        delta_sig = {
            "kind": "legal_action_delta",
            **legal_delta,
        }
        cluster_key = _cluster_key(
            [
                category,
                phase_sig,
                encounter_sig,
                legal_delta.get("missing_actions", []),
                legal_delta.get("extra_actions", []),
            ]
        )
        cluster_features = {
            **common,
            "delta_sig": delta_sig,
        }
        return {
            "encounter_sig": encounter_sig,
            "last_action_sig": last_action_sig,
            "trigger_window_sig": trigger_window_sig,
            "delta_sig": delta_sig,
            "cluster_features": cluster_features,
            "cluster_key": cluster_key,
        }
    if category == "run outcome / map progression":
        delta_sig = {
            "kind": str(result.get("reason")),
            "lightspeed_outcome": result.get("lightspeed_outcome"),
            "native_phase": result.get("native_phase"),
        }
        cluster_key = _cluster_key(
            [
                category,
                phase_sig,
                floor_sig,
                last_action_sig,
                result.get("reason"),
            ]
        )
        cluster_features = {
            **common,
            "delta_sig": delta_sig,
        }
        return {
            "encounter_sig": encounter_sig,
            "last_action_sig": last_action_sig,
            "trigger_window_sig": trigger_window_sig,
            "delta_sig": delta_sig,
            "cluster_features": cluster_features,
            "cluster_key": cluster_key,
        }
    delta_sig = {
        "kind": str(result.get("reason")),
        "native_phase": result.get("native_phase"),
    }
    cluster_key = _cluster_key(
        [
            category,
            phase_sig,
            last_action_sig,
            result.get("reason"),
        ]
    )
    cluster_features = {
        **common,
        "delta_sig": delta_sig,
    }
    return {
        "encounter_sig": encounter_sig,
        "last_action_sig": last_action_sig,
        "trigger_window_sig": trigger_window_sig,
        "delta_sig": delta_sig,
        "cluster_features": cluster_features,
        "cluster_key": cluster_key,
    }


def _attach_failure_metadata(
    result: dict[str, Any],
    *,
    backend: str | None = None,
    source_random_seed: int | None = None,
) -> dict[str, Any]:
    if backend is not None:
        result["backend"] = backend
    if source_random_seed is not None:
        result["source_random_seed"] = int(source_random_seed)
    if result.get("match", False):
        return result
    result["category"] = _categorize_mismatch(result)
    result.update(_build_cluster_metadata(result))
    return result


def _cluster_exemplars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        step = _safe_step(row.get("step"))
        return (
            row.get("reason") == "timeout",
            step if step is not None else 10**9,
            int(row.get("seed", 0)),
        )

    if not rows:
        return []
    ordered = sorted(rows, key=_row_sort_key)
    stepped = [row for row in ordered if _safe_step(row.get("step")) is not None]
    if stepped:
        sorted_steps = sorted(_safe_step(row.get("step")) for row in stepped if _safe_step(row.get("step")) is not None)
        target = sorted_steps[len(sorted_steps) // 2]
        prototype = min(
            stepped,
            key=lambda row: (
                abs((_safe_step(row.get("step")) or target) - target),
                _row_sort_key(row),
            ),
        )
    else:
        prototype = ordered[0]
    earliest_clean = next((row for row in ordered if row.get("reason") != "timeout"), ordered[0])
    variant = next(
        (
            row
            for row in ordered
            if row.get("seed") != prototype.get("seed")
            and (
                row.get("source_random_seed") != prototype.get("source_random_seed")
                or row.get("encounter_sig") != prototype.get("encounter_sig")
                or row.get("delta_sig") != prototype.get("delta_sig")
            )
        ),
        None,
    )
    exemplars: list[dict[str, Any]] = []
    seen: set[int] = set()
    for label, row in (
        ("prototype", prototype),
        ("earliest_clean", earliest_clean),
        ("variant", variant),
    ):
        if row is None:
            continue
        seed = int(row.get("seed", -1))
        if seed in seen:
            continue
        seen.add(seed)
        exemplars.append(
            {
                "label": label,
                "seed": seed,
                "step": _safe_step(row.get("step")),
                "source_random_seed": row.get("source_random_seed"),
                "encounter_sig": row.get("encounter_sig", []),
                "last_action_sig": row.get("last_action_sig"),
            }
        )
    return exemplars


def _summarize_clusters(rows: list[dict[str, Any]], *, top_n: int = 25) -> dict[str, Any]:
    failed_rows = [row for row in rows if not row.get("match", False)]
    cluster_rows: dict[str, list[dict[str, Any]]] = {}
    for row in failed_rows:
        _attach_failure_metadata(row, backend=row.get("backend"), source_random_seed=row.get("source_random_seed"))
        cluster_key = str(row.get("cluster_key") or "unknown")
        cluster_rows.setdefault(cluster_key, []).append(row)
    ordered_keys = sorted(cluster_rows, key=lambda key: (-len(cluster_rows[key]), key))
    cluster_counts = {
        key: len(cluster_rows[key])
        for key in ordered_keys
    }
    top_clusters: list[dict[str, Any]] = []
    for key in ordered_keys[:max(0, int(top_n))]:
        rows_for_key = cluster_rows[key]
        steps = sorted(step for step in (_safe_step(row.get("step")) for row in rows_for_key) if step is not None)
        source_random_seeds = sorted(
            {
                int(row["source_random_seed"])
                for row in rows_for_key
                if row.get("source_random_seed") is not None
            }
        )
        encounter_counts = Counter(
            _stable_key(row.get("encounter_sig", []))
            for row in rows_for_key
        )
        trigger_counts = Counter(
            _stable_key(row.get("trigger_window_sig", []))
            for row in rows_for_key
        )
        delta_counts = Counter(
            _stable_key(row.get("delta_sig", {}))
            for row in rows_for_key
        )
        prototype_row = _cluster_exemplars(rows_for_key)[0]["seed"] if rows_for_key else None
        prototype = next((row for row in rows_for_key if int(row.get("seed", -1)) == prototype_row), rows_for_key[0])
        top_clusters.append(
            {
                "cluster_key": key,
                "category": prototype.get("category"),
                "count": len(rows_for_key),
                "source_coverage": len(source_random_seeds),
                "source_random_seeds": source_random_seeds,
                "min_step": steps[0] if steps else None,
                "median_step": steps[len(steps) // 2] if steps else None,
                "max_step": steps[-1] if steps else None,
                "encounter_examples": [
                    json.loads(sig)
                    for sig, _ in encounter_counts.most_common(3)
                ],
                "trigger_examples": [
                    json.loads(sig)
                    for sig, _ in trigger_counts.most_common(3)
                ],
                "delta_examples": [
                    json.loads(sig)
                    for sig, _ in delta_counts.most_common(3)
                ],
                "cluster_features": prototype.get("cluster_features"),
                "exemplars": _cluster_exemplars(rows_for_key),
            }
        )
    return {
        "cluster_count": len(cluster_rows),
        "cluster_counts": cluster_counts,
        "top_clusters": top_clusters,
    }


def _archive_result(archive_dir: Path, result: dict[str, Any]) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    seed = result.get("seed", "unknown")
    reason = result.get("reason", "unknown")
    category = result.get("category", "uncategorized").replace("/", "_").replace(" ", "_")
    out_path = archive_dir / f"{seed}_{reason}_{category}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_checkpoint(checkpoint_path: Path, result: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def _load_checkpoint_results(checkpoint_path: Path) -> list[dict[str, Any]]:
    if not checkpoint_path.exists():
        return []
    results: list[dict[str, Any]] = []
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    matches = sum(1 for result in results if result.get("match"))
    failures = len(results) - matches
    reason_counts = Counter(result.get("reason", "unknown") for result in results if not result.get("match"))
    category_counts = Counter(result.get("category", "unknown") for result in results if not result.get("match"))
    first_fail = next((result for result in results if not result.get("match")), None)
    summary = {
        "count": len(results),
        "matched": matches,
        "failed": failures,
        "reason_counts": dict(sorted(reason_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "first_fail": first_fail,
    }
    summary.update(_summarize_clusters(results))
    return summary


def _emit_progress(start_time: float, completed: int, total: int, matches: int, failures: int) -> None:
    elapsed = max(0.0, time.time() - start_time)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    eta_text = f"{eta_seconds:.1f}s" if eta_seconds != float("inf") else "unknown"
    print(
        f"[progress] completed={completed}/{total} matched={matches} failed={failures} "
        f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta_text}",
        flush=True,
    )


def _native_env_cls(backend: str):
    if backend == "v3":
        from spirecomm.native_sim_v3.env import NativeRunEnv
        return NativeRunEnv
    if backend == "v2":
        from spirecomm.native_sim_v2.env import NativeRunEnv
        return NativeRunEnv
    from spirecomm.native_sim.env import NativeRunEnv
    return NativeRunEnv


def _load_lightspeed_runtime():
    try:
        import slaythespire as sts
    except ModuleNotFoundError as exc:
        if exc.name != "slaythespire":
            raise
        raise ModuleNotFoundError(
            "slaythespire is required for lightspeed/native comparison workflows. "
            "Install the lightspeed Python package or use native-only entrypoints such as "
            "scripts/native/run_native_run.py, scripts/native/run_native_sim.py, or scripts/native/export_model_run_checklist.py."
        ) from exc
    return sts


def compare_seed(seed: int, ascension: int, max_steps: int, backend: str = "v1"):
    sts = _load_lightspeed_runtime()

    ls_env = sts.ModelDrivenEnv(seed, ascension)
    native = _native_env_cls(backend)(seed=seed, ascension_level=ascension, enable_neow=True)

    trace: list[dict[str, Any]] = []

    for step in range(max_steps):
        ls_in_battle = bool(ls_env.in_battle)
        native_in_battle = native.phase == "COMBAT"

        if ls_env.outcome != sts.GameOutcome.UNDECIDED or native.phase in {"GAME_OVER", "COMPLETE"}:
            return {
                "seed": seed,
                "match": ls_env.outcome == sts.GameOutcome.PLAYER_VICTORY and native.phase == "COMPLETE"
                or ls_env.outcome == sts.GameOutcome.PLAYER_LOSS and native.phase == "GAME_OVER",
                "reason": "finished",
                "step": step,
                "trace_tail": trace[-5:],
                "lightspeed_outcome": str(ls_env.outcome),
                "native_phase": native.phase,
                "lightspeed_floor": int(ls_env.floor_num),
                "native_floor": int(native.floor),
            }

        if ls_in_battle != native_in_battle:
            return {
                "seed": seed,
                "match": False,
                "reason": "battle_phase_mismatch",
                "step": step,
                "lightspeed_in_battle": ls_in_battle,
                "native_phase": native.phase,
                "lightspeed_floor": int(ls_env.floor_num),
                "native_floor": int(native.floor),
                "trace_tail": trace[-5:],
            }

        if ls_in_battle:
            ls_state = sts.get_battle_state(ls_env)
            native_state = native.state()
            ls_sig = _battle_signature(ls_state)
            native_sig = _battle_signature(native_state)
            if ls_sig != native_sig:
                return {
                    "seed": seed,
                    "match": False,
                    "reason": "battle_state_mismatch",
                    "step": step,
                    "trace_tail": trace[-5:],
                    "lightspeed": ls_sig,
                    "native": native_sig,
                }
            ls_actions = [dict(action) for action in sts.get_battle_actions(ls_env)]
            native_actions = [dict(action) for action in native.legal_actions()]
        else:
            ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
            native_actions = [dict(action) for action in native.legal_actions()]

        current_ls_state = ls_state if ls_in_battle else None
        current_native_state = native_state if ls_in_battle else None
        ls_norm = _choice_list_signature(ls_actions, current_ls_state)
        native_norm = _choice_list_signature(native_actions, current_native_state)
        if ls_norm != native_norm:
            mismatch_payload = {
                "seed": seed,
                "match": False,
                "reason": "legal_action_mismatch",
                "step": step,
                "trace_tail": trace[-5:],
                "lightspeed_phase": "BATTLE" if ls_in_battle else str(ls_env.screen_state),
                "native_phase": native.phase,
                "lightspeed_floor": int(ls_env.floor_num),
                "native_floor": int(native.floor),
                "lightspeed_actions": ls_norm,
                "native_actions": native_norm,
            }
            if ls_in_battle:
                mismatch_payload["lightspeed"] = ls_sig
                mismatch_payload["native"] = native_sig
            return mismatch_payload

        chosen_sig = ls_norm[0]
        ls_chosen = _pick_action_by_signature(ls_actions, chosen_sig, current_ls_state)
        native_chosen = _pick_action_by_signature(native_actions, chosen_sig, current_native_state)
        trace.append(
            {
                "step": step,
                "phase": "BATTLE" if ls_in_battle else native.phase,
                "choice": chosen_sig,
                "floor": int(native.floor),
            }
        )
        if ls_in_battle:
            sts.execute_battle_action_bits(ls_env, int(ls_chosen["bits"]))
        else:
            sts.execute_action_bits(ls_env, int(ls_chosen["bits"]))
        native.step(native_chosen)

    return {
        "seed": seed,
        "match": True,
        "reason": "max_steps_reached",
        "step": max_steps,
        "trace_tail": trace[-5:],
        "lightspeed_floor": int(ls_env.floor_num),
        "native_floor": int(native.floor),
    }


class _SeedTimeout(Exception):
    pass


def _compare_seed_with_timeout(seed: int, ascension: int, max_steps: int, backend: str, seed_timeout: int):
    if seed_timeout <= 0:
        return compare_seed(seed, ascension, max_steps, backend=backend)

    def _raise_timeout(signum, frame):
        raise _SeedTimeout()

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seed_timeout))
    try:
        return compare_seed(seed, ascension, max_steps, backend=backend)
    except _SeedTimeout:
        return {
            "seed": seed,
            "match": False,
            "reason": "timeout",
            "step": None,
            "trace_tail": [],
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the same simple action script on lightspeed and native sim (defaults to backend v3).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--backend", choices=["v1", "v2", "v3"], default="v3", help="Native backend to use; defaults to v3, with v2 kept for comparison.")
    parser.add_argument("--archive-dir", type=str, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed-timeout", type=int, default=0, help="Per-seed timeout in seconds; 0 disables timeout.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds: list[int]
    if args.seed is not None:
        seeds = [int(args.seed)]
    else:
        rng = random.Random(args.random_seed)
        seeds = [rng.randint(0, (1 << 63) - 1) for _ in range(max(1, args.count))]

    archive_dir = Path(args.archive_dir).expanduser() if args.archive_dir else None
    checkpoint_path = Path(args.checkpoint_path).expanduser() if args.checkpoint_path else None
    if checkpoint_path is not None and checkpoint_path.exists() and not args.resume:
        checkpoint_path.unlink()

    results: list[dict[str, Any]] = []
    completed_seeds: set[int] = set()
    if checkpoint_path is not None and args.resume:
        results = _load_checkpoint_results(checkpoint_path)
        completed_seeds = {int(result["seed"]) for result in results if "seed" in result}
        if results:
            summary = _summarize_results(results)
            print(
                f"[resume] loaded={len(results)} matched={summary['matched']} failed={summary['failed']} "
                f"from={checkpoint_path}",
                flush=True,
            )

    pending_seeds = [seed for seed in seeds if seed not in completed_seeds]
    total_count = len(seeds)
    completed_count = len(results)
    matches = sum(1 for result in results if result.get("match"))
    failures = completed_count - matches
    start_time = time.time()

    def _handle_result(result: dict[str, Any]) -> None:
        nonlocal completed_count, matches, failures
        _attach_failure_metadata(result, backend=args.backend)
        results.append(result)
        completed_count += 1
        if result.get("match"):
            matches += 1
        else:
            failures += 1
        if archive_dir is not None and not result.get("match", False):
            _archive_result(archive_dir, result)
        if checkpoint_path is not None and args.checkpoint_every > 0:
            _append_checkpoint(checkpoint_path, result)
        if args.progress_every > 0 and (
            completed_count % args.progress_every == 0 or completed_count == total_count
        ):
            _emit_progress(start_time, completed_count, total_count, matches, failures)

    try:
        if args.jobs <= 1:
            for seed in pending_seeds:
                _handle_result(_compare_seed_with_timeout(seed, args.ascension, args.max_steps, args.backend, args.seed_timeout))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                future_to_seed = {
                    executor.submit(_compare_seed_with_timeout, seed, args.ascension, args.max_steps, args.backend, args.seed_timeout): seed
                    for seed in pending_seeds
                }
                for future in as_completed(future_to_seed):
                    _handle_result(future.result())
    except ModuleNotFoundError as exc:
        if "slaythespire is required for lightspeed/native comparison workflows" not in str(exc):
            raise
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        summary = _summarize_results(results)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        raise

    if args.summary_only:
        summary = _summarize_results(results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
