from __future__ import annotations

from typing import Any

from spirecomm.native_sim_v3.core.state import CombatState, MonsterState, RunState


ATTACK_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
VISIBLE_POWER_IDS = {
    "FlameBarrier": "Flame Barrier",
    "NextTurnBlock": "Next Turn Block",
}
VISIBLE_MONSTER_IDS = {
    "OrbWalker": "Orb Walker",
    "ShelledParasite": "Shelled Parasite",
}


def _comm_mod_move_damage(monster: MonsterState) -> tuple[int, int]:
    if str(monster.intent) not in ATTACK_INTENTS:
        return -1, 1
    return int(monster.move_adjusted_damage), int(monster.move_hits)


def _serialize_potion(potion: dict[str, Any], *, in_combat: bool) -> dict[str, Any]:
    return {
        "potion_id": potion.get("potion_id") or potion.get("id") or potion.get("name"),
        "id": potion.get("id") or potion.get("potion_id") or potion.get("name"),
        "name": potion.get("name") or potion.get("potion_id") or potion.get("id"),
        "requires_target": bool(potion.get("requires_target", False)),
        "can_use": bool(potion.get("can_use", True)) if in_combat else False,
        "can_discard": bool(potion.get("can_discard", True)),
    }


def _serialize_power(power: dict[str, Any], *, owner: str | None = None) -> dict[str, Any]:
    payload = dict(power)
    power_id = str(payload.get("power_id") or payload.get("id") or payload.get("name") or "")
    visible_id = "IntangiblePlayer" if owner == "player" and power_id == "Intangible" else VISIBLE_POWER_IDS.get(power_id, power_id)
    if visible_id:
        payload["power_id"] = visible_id
        payload["id"] = visible_id
        payload["name"] = visible_id
    return payload


def _serialize_powers(powers: list[dict[str, Any]], *, owner: str | None = None) -> list[dict[str, Any]]:
    return [_serialize_power(power, owner=owner) for power in powers]


def _serialize_card(card: dict[str, Any]) -> dict[str, Any]:
    payload = dict(card)
    visible_cost = payload.get("cost_for_turn")
    if visible_cost is None:
        visible_cost = payload.get("cost")
    payload["cost"] = visible_cost
    return payload


def _run_room_type(phase: str) -> str:
    return {
        "EVENT": "EventRoom",
        "MAP": "Map",
        "SHOP": "ShopRoom",
        "CAMPFIRE": "RestRoom",
        "TREASURE": "TreasureRoom",
        "BOSS_RELIC": "TreasureRoomBoss",
    }.get(phase, phase)


def _serialize_monster(monster: MonsterState, *, hide_intent: bool = False) -> dict[str, Any]:
    move_adjusted_damage, move_hits = _comm_mod_move_damage(monster)
    half_dead = bool(monster.meta.get("half_dead", False))
    if monster.monster_id == "Darkling" and half_dead:
        return {
            "monster_id": "Darkling",
            "name": monster.name or monster.monster_id,
            "current_hp": int(monster.current_hp),
            "max_hp": int(monster.max_hp),
            "block": int(monster.block),
            "half_dead": False,
            "is_gone": True,
        }
    if monster.monster_id == "AwakenedOne" and half_dead:
        return {
            "monster_id": "AwakenedOne",
            "name": monster.name or monster.monster_id,
            "current_hp": int(monster.current_hp),
            "max_hp": int(monster.max_hp),
            "block": int(monster.block),
            "half_dead": False,
            "is_gone": True,
        }
    is_gone = (int(monster.current_hp) <= 0 or bool(monster.meta.get("escaped", False))) and not half_dead
    monster_id = VISIBLE_MONSTER_IDS.get(monster.monster_id, monster.monster_id)
    return {
        "monster_id": monster_id,
        "name": monster.name or monster.monster_id,
        "current_hp": int(monster.current_hp),
        "max_hp": int(monster.max_hp),
        "block": int(monster.block),
        "intent": "NONE" if hide_intent else monster.intent,
        "half_dead": half_dead,
        "is_gone": is_gone,
        "move_adjusted_damage": None if hide_intent else move_adjusted_damage,
        "move_hits": None if hide_intent else move_hits,
        "powers": _serialize_powers(list(monster.powers), owner="monster"),
    }


def _serialize_combat_state(combat: CombatState, *, hide_intent: bool = False) -> dict[str, Any]:
    return {
        "turn": int(combat.turn) + 1,
        "player": {
            "current_hp": int(combat.player.current_hp),
            "max_hp": int(combat.player.max_hp),
            "block": int(combat.player.block),
            "energy": int(combat.player.energy),
            "powers": _serialize_powers(list(combat.player.powers), owner="player"),
        },
        "cards_discarded_this_turn": int(getattr(combat, "cards_discarded_this_turn", 0) or 0),
        "monsters": [_serialize_monster(monster, hide_intent=hide_intent) for monster in combat.monsters],
        "hand": [_serialize_card(card) for card in combat.hand],
        "draw_pile": [_serialize_card(card) for card in combat.draw_pile],
        "discard_pile": [_serialize_card(card) for card in combat.discard_pile],
        "exhaust_pile": [_serialize_card(card) for card in combat.exhaust_pile],
    }


def _pending_select_visible_cards(pending_card_select: dict[str, Any]) -> list[Any]:
    pending_mode = str(pending_card_select.get("mode") or "").upper()
    cards = list(pending_card_select.get("cards") or [])
    if pending_mode not in {"GAMBLING_CHIP", "ELIXIR"}:
        return cards
    selected_source_indexes = {
        int(index) for index in list(pending_card_select.get("selected_source_indexes") or [])
    }
    source_indexes = list(pending_card_select.get("source_indexes") or [])
    visible_cards = []
    for index, card in enumerate(cards):
        source_index = source_indexes[index] if index < len(source_indexes) else index
        if int(source_index) not in selected_source_indexes:
            visible_cards.append(card)
    return visible_cards


def combat_state(env: Any, *, include_debug_trace: bool = True, include_commands: bool = True) -> dict[str, Any]:
    hide_intent = any(str(relic.get("relic_id") or relic.get("id")) == "Runic Dome" for relic in list(getattr(env, "relics", []) or []))
    pending_card_select = getattr(getattr(env, "engine", None), "pending_card_select", None)
    if str(getattr(getattr(env, "engine", None), "outcome", "UNDECIDED") or "UNDECIDED") != "UNDECIDED":
        pending_card_select = None
    pending_mode = str((pending_card_select or {}).get("mode") or "").upper()
    card_reward_mode = pending_mode in {"DISCOVERY", "NILRYS_CODEX"}
    phase = "CARD_REWARD" if card_reward_mode else "CARD_SELECT" if pending_card_select is not None else "COMBAT"
    screen_state = None
    if pending_card_select is not None:
        if card_reward_mode:
            screen_state = {
                "cards": [_serialize_card(card) for card in list(pending_card_select.get("cards") or [])],
                "bowl_available": False,
                "skip_available": bool(pending_card_select.get("can_skip", False)),
            }
        else:
            screen_state = {
                "cards": [_serialize_card(card) for card in _pending_select_visible_cards(pending_card_select)],
                "num_cards": int(pending_card_select.get("num_cards") or 1),
                "max_cards": int(pending_card_select.get("num_cards") or 1),
                "any_number": bool(pending_card_select.get("any_number", False)),
                "can_pick_zero": bool(pending_card_select.get("can_pick_zero", False)),
                "for_upgrade": False,
                "for_transform": False,
                "for_purge": False,
                "confirm_up": bool(pending_card_select.get("confirm_up", False)),
                "selected_cards": [
                    _serialize_card(card) for card in list(pending_card_select.get("selected_cards") or [])
                ],
            }
    all_pending_actions = env.legal_actions() if pending_card_select is not None else []
    if pending_mode in {"GAMBLING_CHIP", "ELIXIR"}:
        choice_actions = [action for action in all_pending_actions if action.get("kind") != "confirm"]
    else:
        choice_actions = all_pending_actions
    command_actions = all_pending_actions if pending_card_select is not None else env.legal_actions() if include_commands else []
    payload = {
        "backend": "v3",
        "implementation_status": "combat_vertical_slice",
        "phase": phase,
        "screen": phase,
        "screen_type": phase,
        "screen_up": pending_card_select is not None,
        "ascension_level": int(getattr(env, "ascension_level", 0) or 0),
        "act": int(getattr(env, "act", 1) or 1),
        "dungeon_id": getattr(env, "dungeon_id", None),
        "floor": int(getattr(env, "floor", 0) or 0),
        "room_type": phase if pending_card_select is not None else getattr(env, "room_type", None),
        "current_hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(getattr(env, "gold", 99)),
        "act_boss": getattr(env, "act_boss", None),
        "deck": list(getattr(env, "master_deck", []) or []),
        "relics": list(getattr(env, "relics", []) or []),
        "potions": [_serialize_potion(potion, in_combat=True) for potion in list(getattr(env, "potions", []) or [])],
        "combat_state": _serialize_combat_state(env.state, hide_intent=hide_intent),
        "screen_state": screen_state or {},
        "choice_available": pending_card_select is not None,
        "choice_list": choice_actions,
        "reference_sources": dict(getattr(env, "reference_sources", {})),
    }
    if include_commands:
        payload["commands"] = {
            "play": [action for action in command_actions if action.get("kind") == "card"],
            "end": any(action.get("kind") == "end" for action in command_actions),
        }
    if include_debug_trace:
        payload["rng_trace"] = env.randoms.debug_trace()
    return payload


def run_state(state: RunState) -> dict[str, Any]:
    hide_intent = any(str(relic.get("relic_id") or relic.get("id")) == "Runic Dome" for relic in list(state.relics))
    return {
        "backend": "v3",
        "implementation_status": state.implementation_status,
        "seed": int(state.seed),
        "ascension_level": int(state.ascension_level),
        "act": int(state.act),
        "dungeon_id": state.dungeon_id,
        "floor": int(state.floor),
        "phase": state.phase,
        "screen": state.phase,
        "screen_up": state.phase in {"NEOW", "CARD_REWARD", "MAP", "EVENT", "CAMPFIRE", "SHOP", "TREASURE", "CHEST", "BOSS_RELIC"},
        "room_phase": "COMPLETE" if state.phase in {"NEOW", "CARD_REWARD", "MAP", "EVENT", "CAMPFIRE", "SHOP", "TREASURE", "CHEST", "BOSS_RELIC"} else state.phase,
        "room_type": _run_room_type(state.phase),
        "current_hp": int(state.player.current_hp),
        "max_hp": int(state.player.max_hp),
        "gold": int(state.gold),
        "has_ruby_key": bool(state.has_ruby_key),
        "has_emerald_key": bool(state.has_emerald_key),
        "has_sapphire_key": bool(state.has_sapphire_key),
        "common_card_pool": list(state.common_card_pool),
        "uncommon_card_pool": list(state.uncommon_card_pool),
        "rare_card_pool": list(state.rare_card_pool),
        "colorless_card_pool": list(state.colorless_card_pool),
        "curse_card_pool": list(state.curse_card_pool),
        "common_relic_pool": list(state.common_relic_pool),
        "uncommon_relic_pool": list(state.uncommon_relic_pool),
        "rare_relic_pool": list(state.rare_relic_pool),
        "shop_relic_pool": list(state.shop_relic_pool),
        "src_common_card_pool": list(state.src_common_card_pool),
        "src_uncommon_card_pool": list(state.src_uncommon_card_pool),
        "src_rare_card_pool": list(state.src_rare_card_pool),
        "src_colorless_card_pool": list(state.src_colorless_card_pool),
        "src_curse_card_pool": list(state.src_curse_card_pool),
        "act_boss": state.act_boss,
        "boss_relic_pool": list(state.boss_relic_pool),
        "event_id": state.event_id,
        "deck": list(state.deck),
        "relics": list(state.relics),
        "potions": [_serialize_potion(potion, in_combat=state.phase == "COMBAT") for potion in list(state.potions)],
        "combat_state": _serialize_combat_state(state.combat, hide_intent=hide_intent) if state.combat is not None else None,
        "commands": {},
        "choice_available": False,
        "choice_list": [],
    }
