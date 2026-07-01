from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


STRICT_TRACE_SCHEMA = "spirecomm.v3_strict_trace.v3"
LEGACY_TRACE_SCHEMA = "spirecomm.native_run_trace.legacy_v1"

_CHOICE_PHASES = {
    "NEOW",
    "EVENT",
    "CARD_REWARD",
    "CARD_SELECT",
    "MAP",
    "SHOP",
    "CAMPFIRE",
    "TREASURE",
    "CHEST",
    "BOSS_RELIC",
}


@dataclass(frozen=True)
class StrictTraceStepPayload:
    strict_action: dict[str, Any]
    strict_pre_state: dict[str, Any]
    strict_post_state: dict[str, Any]


def _normalize_phase(value: Any, *, screen_type: Any = None, room_type: Any = None) -> str:
    phase = str(value or "").upper()
    screen = str(screen_type or "").upper()
    room = str(room_type or "")
    if phase == "BATTLE":
        return "COMBAT"
    if phase in {"COMBAT_REWARD", "CARD_REWARD"} or screen in {"COMBAT_REWARD", "CARD_REWARD"}:
        return "CARD_REWARD"
    if phase in {"BOSS_REWARD", "BOSS_RELIC"} or screen in {"BOSS_REWARD", "BOSS_RELIC"}:
        return "BOSS_RELIC"
    if phase in {"SHOP_ROOM", "SHOP_SCREEN", "SHOP"} or screen in {"SHOP_ROOM", "SHOP_SCREEN", "SHOP"}:
        return "SHOP"
    if phase in {"GRID", "HAND_SELECT", "CARD_SELECT"} or screen in {"GRID", "HAND_SELECT", "CARD_SELECT"}:
        return "CARD_SELECT"
    if phase in {"CHEST", "TREASURE"} or screen in {"CHEST", "TREASURE"}:
        return "TREASURE"
    if phase == "EVENT" or screen == "EVENT":
        screen_state = None
        if isinstance(room, str) and room == "NeowRoom":
            return "NEOW"
        return "EVENT"
    if phase == "MAP" or screen == "MAP":
        return "MAP"
    if phase == "REST" or screen == "REST":
        return "CAMPFIRE"
    if phase == "GAME_OVER" or screen == "GAME_OVER":
        return "GAME_OVER"
    if phase == "COMPLETE" or screen == "COMPLETE":
        return "COMPLETE"
    if phase:
        return phase
    return "UNKNOWN"


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_phrase(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"#\w", " ", text)
    text = re.sub(r"[\[\]\.,:%']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _canonical_room_type(phase: str, room_type: Any) -> str | None:
    raw = _normalize_text(room_type)
    if phase == "NEOW":
        return "NeowRoom"
    if phase == "CARD_REWARD":
        return "CARD_REWARD"
    if phase == "CARD_SELECT":
        return "CARD_SELECT"
    if phase == "MAP":
        return "Map"
    if phase == "SHOP":
        return "ShopRoom"
    if phase == "CAMPFIRE":
        return "RestRoom"
    if phase == "TREASURE":
        return "TreasureRoom"
    if phase == "BOSS_RELIC":
        return "TreasureRoomBoss"
    if phase == "EVENT":
        return raw or "EventRoom"
    return raw


def _canonical_screen_type(phase: str, screen_type: Any) -> str:
    raw = _normalize_text(screen_type)
    if phase in {"NEOW", "EVENT"}:
        return "EVENT"
    if phase == "MAP":
        return "MAP"
    if phase == "SHOP":
        return "SHOP"
    if phase == "CAMPFIRE":
        return "REST"
    if phase == "TREASURE":
        return "TREASURE"
    if phase == "BOSS_RELIC":
        return "BOSS_REWARD"
    if phase == "CARD_REWARD":
        return "CARD_REWARD"
    if phase == "CARD_SELECT":
        return "CARD_SELECT"
    if phase == "COMBAT":
        return "COMBAT"
    return raw or phase


def _card_fingerprint(payload: Any, *, pile: str | None = None) -> dict[str, Any]:
    if payload is None:
        return {}
    exhausts = bool(payload.get("exhausts")) if payload.get("exhausts") is not None else None
    if pile == "exhaust_pile" and bool(payload.get("ethereal")):
        # Comm's live surface reports ethereal cards as exhausts=True after
        # they have moved to the exhaust pile, while the native simulator keeps
        # the printed card exhaust flag unchanged. This is metadata only; the
        # pile membership is the semantic state strict replay needs to compare.
        exhausts = True
    result = {
        "card_id": payload.get("card_id") or payload.get("id") or payload.get("name"),
        "name": payload.get("card_id") or payload.get("id") or payload.get("name"),
        "upgrades": int(payload.get("upgrades", 0) or 0),
        "type": payload.get("type"),
        "rarity": payload.get("rarity"),
        "cost": payload.get("cost"),
        "has_target": bool(payload.get("has_target")) if payload.get("has_target") is not None else None,
        "exhausts": exhausts,
    }
    return result


def _card_fingerprint_sort_key(card: dict[str, Any]) -> tuple[Any, ...]:
    return (
        card.get("card_id"),
        card.get("upgrades"),
        card.get("type"),
        card.get("rarity"),
        card.get("cost"),
        card.get("has_target"),
        card.get("exhausts"),
    )


def _relic_fingerprint(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    result = {
        "relic_id": payload.get("relic_id") or payload.get("id") or payload.get("name"),
        "name": payload.get("name") or payload.get("relic_id") or payload.get("id"),
    }
    if payload.get("counter") is not None:
        counter = int(payload.get("counter") or 0)
        if counter > 0:
            result["counter"] = counter
    if payload.get("price") is not None:
        result["price"] = int(payload.get("price") or 0)
    return result


def _potion_fingerprint(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    result = {
        "potion_id": payload.get("potion_id") or payload.get("id") or payload.get("name"),
        "name": payload.get("name") or payload.get("potion_id") or payload.get("id"),
    }
    if payload.get("price") is not None:
        result["price"] = int(payload.get("price") or 0)
    if payload.get("requires_target") is not None:
        result["requires_target"] = bool(payload.get("requires_target"))
    return result


def _visible_potions(payloads: list[Any]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for potion in payloads:
        fingerprint = _potion_fingerprint(potion)
        if fingerprint.get("potion_id") == "Potion Slot":
            continue
        visible.append(fingerprint)
    return visible


def _map_current_node_fingerprint(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    y = node.get("y")
    if y is not None and int(y) < 0:
        return None
    return {
        "x": node.get("x"),
        "y": y,
    }


def _normalize_map_symbol(value: Any) -> Any:
    if value is None:
        return None
    symbol = str(value)
    if symbol.endswith("_GREEN"):
        return symbol[:-6]
    return symbol


def _infer_neow_bonus(item: dict[str, Any], *, max_hp: Any = None) -> str | None:
    candidates = [
        item.get("bonus"),
        item.get("bonus_text"),
        item.get("text"),
        item.get("label"),
        item.get("name"),
    ]
    phrases = [_normalize_phrase(candidate) for candidate in candidates if candidate is not None]
    text = " ".join(phrase for phrase in phrases if phrase)
    if not text:
        return None

    if "random boss relic" in text:
        return "BOSS_RELIC"
    if "rare colorless" in text:
        return "RANDOM_COLORLESS_2"
    if "colorless card" in text:
        return "RANDOM_COLORLESS"
    if "3 random potions" in text:
        return "THREE_SMALL_POTIONS"
    if "random common relic" in text:
        return "RANDOM_COMMON_RELIC"
    if "random rare relic" in text:
        return "ONE_RARE_RELIC"
    if "250 gold" in text:
        return "TWO_FIFTY_GOLD"
    if "100 gold" in text:
        return "HUNDRED_GOLD"
    if (
        "first 3 combats have 1 hp" in text
        or "first three combats have 1 hp" in text
        or "next 3 combats have 1 hp" in text
        or "next three combats have 1 hp" in text
    ):
        return "THREE_ENEMY_KILL"
    if "transform 2 cards" in text:
        return "TRANSFORM_TWO_CARDS"
    if "remove 2 cards" in text:
        return "REMOVE_TWO"
    if "choose a rare card to obtain" in text or "choose 1 of 3 rare cards" in text:
        return "THREE_RARE_CARDS"
    if "obtain a random rare card" in text:
        return "ONE_RANDOM_RARE_CARD"
    if "choose 1 of 3 cards" in text or "choose a card to obtain" in text:
        return "THREE_CARDS"
    if "remove a card from your deck" in text or "remove a card" in text:
        return "REMOVE_CARD"
    if "upgrade a card in your deck" in text or "upgrade a card" in text:
        return "UPGRADE_CARD"
    if "transform a card" in text:
        return "TRANSFORM_CARD"
    if "proceed" in text or "leave" in text or "continue" in text:
        return "CONTINUE"
    if "max hp" in text and ("gain" in text or "+" in text):
        try:
            hp = int(max_hp or 0)
        except (TypeError, ValueError):
            hp = 0
        numbers = [int(match) for match in re.findall(r"\b\d+\b", text)]
        if hp > 0 and numbers:
            ten_percent = int(hp * 0.1)
            if numbers[0] == ten_percent * 2:
                return "TWENTY_PERCENT_HP_BONUS"
            if numbers[0] == ten_percent:
                return "TEN_PERCENT_HP_BONUS"
        if numbers and numbers[0] >= 12:
            return "TWENTY_PERCENT_HP_BONUS"
        return "TEN_PERCENT_HP_BONUS"
    return None


def _infer_neow_drawback(item: dict[str, Any], *, max_hp: Any = None, current_hp: Any = None) -> str | None:
    candidates = [
        item.get("drawback"),
        item.get("drawback_text"),
        item.get("text"),
        item.get("label"),
        item.get("name"),
    ]
    phrases = [_normalize_phrase(candidate) for candidate in candidates if candidate is not None]
    text = " ".join(phrase for phrase in phrases if phrase)
    if not text:
        return None

    if "lose all gold" in text:
        return "NO_GOLD"
    if "obtain a curse" in text:
        return "CURSE"
    if "lose your starter relic" in text:
        return "NONE"
    if "lose" in text and "max hp" in text:
        return "TEN_PERCENT_HP_LOSS"
    if ("lose" in text and "hp" in text) or ("take" in text and "damage" in text):
        try:
            hp = int(current_hp or 0)
        except (TypeError, ValueError):
            hp = 0
        numbers = [int(match) for match in re.findall(r"\b\d+\b", text)]
        if hp > 0 and numbers and numbers[0] == (hp // 10 * 3):
            return "PERCENT_DAMAGE"
        return "PERCENT_DAMAGE"
    return "NONE"


def _infer_wheel_event_stage(item: dict[str, Any]) -> str | None:
    candidates = [
        item.get("stage"),
        item.get("name"),
        item.get("label"),
        item.get("text"),
    ]
    text = " ".join(_normalize_phrase(candidate) for candidate in candidates if candidate is not None)
    if not text:
        return None
    if "prize" in text or "result" in text:
        return "RESULT"
    if "spin" in text:
        return "SPIN"
    if "play" in text:
        return "PLAY"
    if "leave" in text or "continue" in text or "proceed" in text:
        return "LEAVE"
    return None


def _power_fingerprint(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    return {
        "power_id": payload.get("power_id") or payload.get("id") or payload.get("name"),
        "amount": int(payload.get("amount", 0) or 0),
    }


def _monster_fingerprint(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    current_hp = int(payload.get("current_hp", 0) or 0)
    half_dead = bool(payload.get("half_dead", False))
    is_gone = bool(payload.get("is_gone", False)) or (current_hp <= 0 and not half_dead)
    if is_gone:
        return {
            "monster_id": payload.get("monster_id") or payload.get("id") or payload.get("name"),
            "name": payload.get("name") or payload.get("monster_id") or payload.get("id"),
            "current_hp": current_hp,
            "max_hp": int(payload.get("max_hp", 0) or 0),
            "block": int(payload.get("block", 0) or 0),
            "half_dead": False,
            "is_gone": True,
        }
    return {
        "monster_id": payload.get("monster_id") or payload.get("id") or payload.get("name"),
        "name": payload.get("name") or payload.get("monster_id") or payload.get("id"),
        "current_hp": current_hp,
        "max_hp": int(payload.get("max_hp", 0) or 0),
        "block": int(payload.get("block", 0) or 0),
        "half_dead": half_dead,
        "is_gone": False,
        "intent": payload.get("intent"),
        "move_adjusted_damage": payload.get("move_adjusted_damage"),
        "move_hits": payload.get("move_hits"),
        "powers": [_power_fingerprint(power) for power in list(payload.get("powers") or [])],
    }


def _normalize_choice_item(item: Any, *, phase: str) -> dict[str, Any]:
    if isinstance(item, str):
        if phase == "NEOW":
            return {
                "kind": "neow",
                "bonus": _infer_neow_bonus({"text": item}),
                "drawback": _infer_neow_drawback({"text": item}),
            }
        return {"kind": "raw", "label": item}
    if not isinstance(item, dict):
        return {"kind": "raw", "label": str(item)}

    kind = str(item.get("kind") or "").lower()
    choice_index = item.get("choice_index")
    base: dict[str, Any] = {}
    if choice_index is not None:
        base["choice_index"] = int(choice_index)

    if kind in {"card_reward", "card_select"}:
        base["kind"] = kind
        base["card"] = _card_fingerprint(item.get("card") or item)
        if item.get("mode") is not None:
            base["mode"] = item.get("mode")
        if item.get("target_index") is not None:
            base["target_index"] = int(item.get("target_index") or 0)
        return base

    if kind == "map":
        node_id = item.get("node_id")
        if node_id is None and item.get("x") is not None and item.get("floor") is not None:
            act = int(item.get("act") or 1)
            floor = int(item.get("floor"))
            node_id = f"a{act}-r{floor - 1}-x{int(item.get('x'))}"
        base.update(
            {
                "kind": "map",
                "symbol": _normalize_map_symbol(item.get("symbol") or item.get("name")),
                "x": item.get("x"),
                "floor": item.get("floor"),
                "node_id": node_id,
            }
        )
        return base

    if kind == "shop":
        base["kind"] = "shop"
        base["item_kind"] = item.get("item_kind")
        base["name"] = item.get("name") or item.get("item_id") or item.get("label")
        if item.get("price") is not None:
            base["price"] = int(item.get("price") or 0)
        if item.get("shop_index") is not None:
            base["shop_index"] = int(item.get("shop_index") or 0)
        if item.get("item_kind") == "card":
            base["card"] = _card_fingerprint(item.get("card") or item)
        if item.get("item_kind") == "relic":
            base["relic"] = _relic_fingerprint(item.get("relic") or item)
        if item.get("item_kind") == "potion":
            base["potion"] = _potion_fingerprint(item.get("potion") or item)
        return base

    if kind == "reward_gold":
        base.update({"kind": "reward_gold", "amount": int(item.get("amount", 0) or 0)})
        return base
    if kind == "reward_relic":
        base.update({"kind": "reward_relic", "relic": _relic_fingerprint(item.get("relic") or item)})
        return base
    if kind == "reward_potion":
        base.update({"kind": "reward_potion", "potion": _potion_fingerprint(item.get("potion") or item)})
        return base
    if kind == "reward_key":
        base.update({"kind": "reward_key", "key": item.get("key"), "relic_id": item.get("relic_id")})
        return base
    if kind == "boss_relic":
        base.update({"kind": "boss_relic", "relic": _relic_fingerprint(item.get("relic") or item)})
        return base
    if kind in {"event", "neow"}:
        base["kind"] = kind
        if phase == "NEOW" or kind == "neow" or item.get("bonus") is not None or item.get("drawback") is not None:
            bonus = item.get("bonus") or _infer_neow_bonus(item)
            base["bonus"] = bonus
            base["drawback"] = item.get("drawback") or _infer_neow_drawback(item)
            # Hidden Neow intro Talk has no semantic bonus/drawback identity of
            # its own, so preserve the visible label/text. Regular reward
            # options and CONTINUE are compared by their semantic
            # bonus/drawback pair instead of raw UI phrasing.
            if bonus is None and item.get("label") is not None:
                base["label"] = item.get("label")
            if bonus is None and item.get("text") is not None:
                base["text"] = item.get("text")
        else:
            base["kind"] = "event"
            if str(item.get("event_id") or "") == "Wheel of Change":
                stage = _infer_wheel_event_stage(item)
                if stage is not None:
                    base["event_stage"] = stage
        return base
    if kind in {"campfire", "treasure", "skip"}:
        base.update({"kind": kind, "name": item.get("name") or item.get("label")})
        return base
    if phase == "NEOW" and kind == "":
        base.update(
            {
                "kind": "neow",
                "bonus": _infer_neow_bonus(item),
                "drawback": _infer_neow_drawback(item),
            }
        )
        return base

    base.update({"kind": kind or "raw", "label": item.get("label") or item.get("name") or str(item)})
    return base


def _normalize_choices_from_screen_state(state: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    screen_state = state.get("screen_state") or {}
    choices: list[dict[str, Any]] = []
    if phase in {"EVENT", "NEOW"}:
        for index, option in enumerate(list(screen_state.get("options") or [])):
            option_payload = dict(option)
            option_payload.setdefault("kind", "neow" if phase == "NEOW" else "event")
            option_payload.setdefault("choice_index", index)
            option_payload.setdefault("event_id", screen_state.get("event_id") or state.get("event_id"))
            choices.append(_normalize_choice_item(option_payload, phase=phase))
        if choices:
            return choices

    if phase == "CARD_REWARD":
        for index, card in enumerate(list(screen_state.get("cards") or [])):
            card_payload = {"kind": "card_reward", "choice_index": index, "card": card}
            choices.append(_normalize_choice_item(card_payload, phase=phase))
        if choices:
            return choices
        for index, reward in enumerate(list(screen_state.get("rewards") or [])):
            reward_type = str(reward.get("reward_type") or "").upper()
            reward_payload: dict[str, Any]
            if reward_type in {"GOLD", "STOLEN_GOLD"}:
                reward_payload = {"kind": "reward_gold", "choice_index": index, "amount": reward.get("gold", 0)}
            elif reward_type == "RELIC":
                reward_payload = {"kind": "reward_relic", "choice_index": index, "relic": reward.get("relic")}
            elif reward_type == "POTION":
                reward_payload = {"kind": "reward_potion", "choice_index": index, "potion": reward.get("potion")}
            elif reward_type == "SAPPHIRE_KEY":
                reward_payload = {"kind": "reward_key", "choice_index": index, "key": "sapphire", "relic_id": (reward.get("link") or {}).get("id") or (reward.get("link") or {}).get("relic_id")}
            else:
                reward_payload = {"kind": "raw", "choice_index": index, "label": reward_type}
            choices.append(_normalize_choice_item(reward_payload, phase=phase))
        if choices:
            return choices

    if phase == "MAP":
        for index, node in enumerate(list(screen_state.get("next_nodes") or [])):
            choices.append(
                _normalize_choice_item(
                    {
                        "kind": "map",
                        "choice_index": index,
                        "symbol": node.get("symbol"),
                        "x": node.get("x"),
                        "floor": (node.get("y") + 1) if node.get("y") is not None else None,
                        "act": state.get("act"),
                    },
                    phase=phase,
                )
            )
        if screen_state.get("boss_available"):
            choices.append({"kind": "map_boss", "choice_index": len(choices), "name": "boss"})
        if choices:
            return choices

    if phase == "SHOP":
        index = 0
        for card_index, card in enumerate(list(screen_state.get("cards") or [])):
            choices.append(
                _normalize_choice_item(
                    {
                        "kind": "shop",
                        "choice_index": index,
                        "item_kind": "card",
                        "name": card.get("name") or card.get("id"),
                        "price": card.get("price"),
                        "shop_index": card_index,
                        "card": card,
                    },
                    phase=phase,
                )
            )
            index += 1
        for relic_index, relic in enumerate(list(screen_state.get("relics") or [])):
            choices.append(
                _normalize_choice_item(
                    {
                        "kind": "shop",
                        "choice_index": index,
                        "item_kind": "relic",
                        "name": relic.get("name") or relic.get("id"),
                        "price": relic.get("price"),
                        "shop_index": relic_index,
                        "relic": relic,
                    },
                    phase=phase,
                )
            )
            index += 1
        for potion_index, potion in enumerate(list(screen_state.get("potions") or [])):
            choices.append(
                _normalize_choice_item(
                    {
                        "kind": "shop",
                        "choice_index": index,
                        "item_kind": "potion",
                        "name": potion.get("name") or potion.get("id"),
                        "price": potion.get("price"),
                        "shop_index": potion_index,
                        "potion": potion,
                    },
                    phase=phase,
                )
            )
            index += 1
        if screen_state.get("purge_available"):
            choices.append(
                _normalize_choice_item(
                    {
                        "kind": "shop",
                        "choice_index": index,
                        "item_kind": "purge",
                        "name": "Purge",
                        "price": screen_state.get("purge_cost"),
                    },
                    phase=phase,
                )
            )
            index += 1
        choices.append({"kind": "shop", "choice_index": index, "item_kind": "leave", "name": "Leave"})
        return choices

    if phase == "CAMPFIRE":
        for index, option in enumerate(list(screen_state.get("rest_options") or [])):
            choices.append(_normalize_choice_item({"kind": "campfire", "choice_index": index, "name": option}, phase=phase))
        if choices:
            return choices

    if phase == "TREASURE":
        if screen_state:
            choices.append({"kind": "treasure", "choice_index": 0, "name": "OPEN_CHEST"})
            return choices

    if phase == "BOSS_RELIC":
        for index, relic in enumerate(list(screen_state.get("relics") or [])):
            choices.append(_normalize_choice_item({"kind": "boss_relic", "choice_index": index, "relic": relic}, phase=phase))
        if screen_state.get("relics") is not None:
            choices.append({"kind": "skip", "choice_index": len(choices), "name": "SKIP"})
        if choices:
            return choices

    if phase == "CARD_SELECT":
        cards = list(screen_state.get("cards") or screen_state.get("hand") or [])
        for index, card in enumerate(cards):
            choices.append(
                _normalize_choice_item(
                    {
                        "kind": "card_select",
                        "choice_index": index,
                        "mode": "unknown",
                        "card": card,
                    },
                    phase=phase,
                )
            )
        if choices:
            return choices

    choice_list = state.get("choice_list")
    if isinstance(choice_list, list) and choice_list and isinstance(choice_list[0], dict):
        normalized_choices = []
        for item in choice_list:
            item_payload = dict(item)
            if phase == "MAP":
                item_payload.setdefault("act", state.get("act"))
            normalized_choices.append(_normalize_choice_item(item_payload, phase=phase))
        return normalized_choices

    if isinstance(choice_list, list):
        return [_normalize_choice_item(item, phase=phase) for item in choice_list]
    return choices


def normalize_verbose_state_for_strict(
    state: dict[str, Any] | None,
    *,
    phase_hint: str | None = None,
    snapshot_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if state is None:
        return {"phase": "OUT_OF_GAME"}

    phase_source = phase_hint or state.get("phase") or state.get("screen") or state.get("screen_type")
    if (
        phase_hint is None
        and str(phase_source or "").upper() in {"", "NONE"}
        and str(state.get("room_phase") or "").upper() == "COMBAT"
        and state.get("combat_state") is not None
    ):
        phase_source = "COMBAT"
    phase = _normalize_phase(
        phase_source,
        screen_type=state.get("screen_type"),
        room_type=state.get("room_type"),
    )
    screen_state = dict(state.get("screen_state") or {})
    normalized: dict[str, Any] = {
        "phase": phase,
        "floor": state.get("floor"),
        "act": state.get("act"),
        "current_hp": state.get("current_hp"),
        "max_hp": state.get("max_hp"),
        "gold": state.get("gold"),
        "room_type": _canonical_room_type(phase, state.get("room_type")),
        "screen_type": _canonical_screen_type(phase, state.get("screen_type") or state.get("screen") or phase),
        "deck": [_card_fingerprint(card) for card in list(state.get("deck") or [])],
        "relics": [_relic_fingerprint(relic) for relic in list(state.get("relics") or [])],
        "potions": _visible_potions(list(state.get("potions") or [])),
    }

    event_id = state.get("event_id") or screen_state.get("event_id")
    event_name = None
    if phase != "EVENT":
        event_id = None
        event_name = None

    normalized["screen_state"] = {
        "event_id": event_id,
        "event_name": event_name,
        "card_select_context": (snapshot_hint or {}).get("card_select_context") or state.get("card_select_context"),
        "choices": _normalize_choices_from_screen_state(state, phase=phase),
    }

    if phase == "MAP":
        normalized["screen_state"]["current_node"] = _map_current_node_fingerprint(
            screen_state.get("current_node")
        )
        normalized["screen_state"]["boss_available"] = bool(screen_state.get("boss_available"))
    if phase == "SHOP":
        normalized["screen_state"]["purge_cost"] = screen_state.get("purge_cost")
        normalized["screen_state"]["purge_available"] = bool(screen_state.get("purge_available"))
    if phase == "CARD_SELECT":
        num_cards = screen_state.get("num_cards") or screen_state.get("max_cards")
        selected_cards = list(screen_state.get("selected") or screen_state.get("selected_cards") or [])
        hand_select_confirm = bool(num_cards is not None and len(selected_cards) >= int(num_cards))
        normalized["screen_state"]["confirm_up"] = bool(screen_state.get("confirm_up")) or hand_select_confirm
        normalized["screen_state"]["num_cards"] = num_cards
        normalized["screen_state"]["any_number"] = bool(screen_state.get("any_number"))
        normalized["screen_state"]["can_pick_zero"] = bool(screen_state.get("can_pick_zero"))
        normalized["screen_state"]["for_upgrade"] = bool(screen_state.get("for_upgrade"))
        normalized["screen_state"]["for_transform"] = bool(screen_state.get("for_transform"))
        normalized["screen_state"]["for_purge"] = bool(screen_state.get("for_purge"))
    if phase == "CARD_REWARD":
        skip_available = screen_state.get("skip_available")
        bowl_available = screen_state.get("bowl_available")
        # Reward screens in live StS expose skip by default; strict traces
        # should not self-diverge just because an exporter omitted the bool.
        normalized["screen_state"]["skip_available"] = True if skip_available is None else bool(skip_available)
        normalized["screen_state"]["bowl_available"] = False if bowl_available is None else bool(bowl_available)
    if phase == "TREASURE":
        normalized["screen_state"]["chest_type"] = screen_state.get("chest_type")
        normalized["screen_state"]["chest_open"] = bool(screen_state.get("chest_open"))

    combat_state = state.get("combat_state")
    if phase == "COMBAT" and combat_state is not None:
        player = combat_state.get("player") or {}
        normalized["combat_state"] = {
            "turn": combat_state.get("turn"),
            "player": {
                "current_hp": player.get("current_hp"),
                "max_hp": player.get("max_hp"),
                "block": player.get("block"),
                "energy": player.get("energy"),
                "powers": [_power_fingerprint(power) for power in list(player.get("powers") or [])],
            },
            "hand": [_card_fingerprint(card) for card in list(combat_state.get("hand") or [])],
            "draw_pile": [_card_fingerprint(card) for card in list(combat_state.get("draw_pile") or [])],
            "discard_pile": [_card_fingerprint(card) for card in list(combat_state.get("discard_pile") or [])],
            # End-turn ethereal triggers are shuffled by java.util.Collections
            # rather than game-seeded RNG, so exhaust pile order is not
            # reproducible across strict replay processes.
            "exhaust_pile": sorted(
                [_card_fingerprint(card, pile="exhaust_pile") for card in list(combat_state.get("exhaust_pile") or [])],
                key=_card_fingerprint_sort_key,
            ),
            "monsters": [_monster_fingerprint(monster) for monster in list(combat_state.get("monsters") or [])],
        }
    else:
        normalized["combat_state"] = None

    return normalized


def build_strict_action(
    action: dict[str, Any],
    *,
    pre_state: dict[str, Any],
    phase: str,
    strict_pre_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_pre_state = strict_pre_state or normalize_verbose_state_for_strict(pre_state, phase_hint=phase)
    normalized_phase = _normalize_phase(phase)
    kind = str(action.get("kind") or "").lower()
    result: dict[str, Any] = {
        "phase": normalized_phase,
        "kind": kind,
        "raw_kind": action.get("kind"),
    }

    if kind == "skip":
        result.update({"kind": "skip", "name": action.get("name") or "SKIP"})
        return result
    if kind == "proceed":
        result.update({"kind": "proceed", "name": action.get("name") or "PROCEED"})
        return result
    if kind == "confirm":
        result.update({"kind": "confirm", "name": action.get("name") or "CONFIRM"})
        return result

    if normalized_phase == "COMBAT":
        if kind == "card":
            hand_index = int(action.get("card_index") if action.get("card_index") is not None else action.get("source_index", -1))
            hand = list((strict_pre_state.get("combat_state") or {}).get("hand") or [])
            result.update(
                {
                    "kind": "play_card",
                    "hand_index": hand_index,
                    "target_index": action.get("target_index"),
                    "card": hand[hand_index] if 0 <= hand_index < len(hand) else _card_fingerprint(action.get("card") or action),
                }
            )
            return result
        if kind == "potion":
            result.update(
                {
                    "kind": "use_potion",
                    "potion_index": action.get("potion_index"),
                    "target_index": action.get("target_index"),
                    "potion": _potion_fingerprint(action.get("potion") or action),
                }
            )
            return result
        if kind == "end":
            result["kind"] = "end_turn"
            return result

    choice_index = action.get("choice_index")
    choices = list((strict_pre_state.get("screen_state") or {}).get("choices") or [])
    if normalized_phase == "CARD_REWARD" and kind == "card_reward":
        card_choice_index = action.get("card_index")
        if card_choice_index is None:
            wanted_card = action.get("card") if isinstance(action.get("card"), dict) else {}
            wanted_id = wanted_card.get("card_id") or wanted_card.get("id") or action.get("card_id") or action.get("name")
            if wanted_id is not None:
                for index, choice in enumerate(choices):
                    if str(choice.get("kind") or "").lower() != "card_reward":
                        continue
                    choice_card = choice.get("card") or {}
                    choice_id = choice_card.get("card_id") or choice_card.get("id") or choice_card.get("name")
                    if choice_id == wanted_id:
                        card_choice_index = index
                        break
        if card_choice_index is not None:
            card_choice_index = int(card_choice_index)
            result.update(
                {
                    "kind": "choose_by_index",
                    "choice_index": card_choice_index,
                    "choice": choices[card_choice_index] if 0 <= card_choice_index < len(choices) else None,
                }
            )
            return result

    if (
        normalized_phase == "CARD_REWARD"
        and kind == "raw"
        and str(action.get("label") or action.get("name") or "").upper() == "CARD"
    ):
        for index, choice in enumerate(choices):
            if (
                str(choice.get("kind") or "").lower() == "raw"
                and str(choice.get("label") or choice.get("name") or "").upper() == "CARD"
            ):
                result.update(
                    {
                        "kind": "choose_by_index",
                        "choice_index": index,
                        "choice": choice,
                    }
                )
                return result

    if normalized_phase == "CARD_REWARD" and kind == "discard_potion":
        result.update(
            {
                "kind": "discard_potion",
                "potion_index": action.get("potion_index"),
                "potion": _potion_fingerprint(action.get("potion") or action),
            }
        )
        return result

    if choice_index is not None:
        choice_index = int(choice_index)
        result.update(
            {
                "kind": "choose_by_index",
                "choice_index": choice_index,
                "choice": choices[choice_index] if 0 <= choice_index < len(choices) else None,
            }
        )
        return result

    if normalized_phase == "SHOP":
        if str(action.get("item_kind") or "").lower() == "leave":
            result.update(
                {
                    "kind": "leave",
                    "name": action.get("name") or action.get("label") or "Leave",
                    "item_kind": "leave",
                }
            )
            return result
        result.update(
            {
                "kind": "choose_by_name",
                "name": action.get("name") or action.get("label") or action.get("item_id"),
                "item_kind": action.get("item_kind"),
            }
        )
        if str(action.get("item_kind") or "").lower() == "purge" and action.get("target_index") is not None:
            result["target_index"] = int(action.get("target_index") or 0)
        return result

    result.update({"kind": "unsupported", "action": dict(action)})
    return result


def build_strict_step_payload(
    *,
    action: dict[str, Any],
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    phase: str,
    post_phase: str | None = None,
    pre_snapshot: dict[str, Any] | None = None,
) -> StrictTraceStepPayload:
    strict_pre = normalize_verbose_state_for_strict(pre_state, phase_hint=phase, snapshot_hint=pre_snapshot)
    strict_post = normalize_verbose_state_for_strict(post_state, phase_hint=post_phase or phase, snapshot_hint=pre_snapshot)
    strict_action = build_strict_action(action, pre_state=pre_state, phase=phase, strict_pre_state=strict_pre)
    return StrictTraceStepPayload(
        strict_action=strict_action,
        strict_pre_state=strict_pre,
        strict_post_state=strict_post,
    )
