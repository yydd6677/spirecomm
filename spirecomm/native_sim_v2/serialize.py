"""Serializer bridge for the v2 backend.

This module owns the spirecomm-facing state serialization so v2 no longer
needs to call v1's ``state()`` / ``to_spirecomm_state()`` directly.
"""

from __future__ import annotations

from typing import Any

from spirecomm.ai.observation import CANONICAL_STATE_VERSION
from spirecomm.native_sim.potions import potions_to_spirecomm
from spirecomm.native_sim_v2.helpers_cards import card_to_spirecomm
from spirecomm.native_sim_v2.helpers_serialize import _serialize_move_name, _serialize_named_power, _spirecomm_monster_id
from spirecomm.native_sim_v2.monster_support import monster_adjusted_damage


def _card_to_spirecomm_in_combat(combat: Any, card: Any) -> dict[str, Any]:
    serialized = card_to_spirecomm(card, is_playable=combat.playable(card))
    if card.card_def.x_cost:
        # Lightspeed keeps turn-only X-cost freebies such as Infernal Blade
        # marked as X, but preserves an explicit combat-cost override from
        # generators like Metamorphosis.
        if card.cost_for_combat is not None:
            serialized["cost_for_turn"] = combat._card_display_cost(card)
        else:
            serialized["cost_for_turn"] = -1
    elif card.card_def.card_type not in {"STATUS", "CURSE"}:
        serialized["cost_for_turn"] = combat._card_display_cost(card)
    elif card.card_def.card_type == "CURSE" and combat._has_relic("Blue Candle"):
        # lightspeed exposes Blue Candle curses with a special playable cost marker
        serialized["cost_for_turn"] = -3
    return serialized


def _player_powers_to_spirecomm(player: Any) -> list[dict[str, Any]]:
    powers: list[dict[str, Any]] = []
    for power_id, amount in (
        ("Strength", player.power("Strength")),
        ("Dexterity", player.power("Dexterity")),
        ("Vulnerable", player.power("Vulnerable")),
        ("Weakened", player.power("Weakened")),
        ("Frail", player.power("Frail")),
        ("Artifact", player.power("Artifact")),
        ("Metallicize", player.power("Metallicize")),
        ("Rage", player.power("Rage")),
        ("Barricade", 1 if player.power("Barricade") > 0 else 0),
        ("Demon Form", player.power("Demon Form")),
        ("Berserk", player.power("Berserk")),
        ("Flame Barrier", player.power("Flame Barrier")),
        ("Thorns", player.power("Thorns")),
        ("Plated Armor", player.power("Plated Armor")),
        ("No Draw", 1 if player.power("No Draw") > 0 else 0),
    ):
        if amount:
            powers.append(_serialize_named_power(power_id, amount))
    return powers


def _monster_powers_to_spirecomm(monster: Any) -> list[dict[str, Any]]:
    powers: list[dict[str, Any]] = []
    invalid_slot = getattr(monster, "monster_id", None) == "INVALID = 0"
    leader_summoned = bool(getattr(monster, "ai_state", {}).get("leader_summoned", 0))
    for power_id, amount in (
        ("Strength", monster.power("Strength")),
        ("Dexterity", monster.power("Dexterity")),
        ("Vulnerable", monster.power("Vulnerable")),
        ("Weakened", monster.power("Weakened")),
        ("Artifact", monster.power("Artifact")),
        ("Metallicize", monster.power("Metallicize")),
        ("Ritual", monster.power("Ritual")),
        ("Angry", monster.power("Angry")),
        ("Sharp Hide", monster.power("Sharp Hide")),
        ("Thorns", monster.power("Thorns")),
        ("Curl Up", monster.power("Curl Up")),
        ("Mode Shift", monster.power("Mode Shift")),
        ("Regenerate", monster.power("Regenerate")),
        ("Flight", monster.power("Flight")),
        ("Malleable", monster.power("Malleable")),
        ("Plated Armor", monster.power("Plated Armor")),
    ):
        if power_id == "Flight":
            if "Flight" in monster.powers and amount != 0:
                powers.append(_serialize_named_power(power_id, amount))
            continue
        if invalid_slot and power_id not in {"Strength", "Metallicize", "Regenerate"}:
            continue
        if leader_summoned and power_id == "Angry":
            continue
        if amount:
            powers.append(_serialize_named_power(power_id, amount))
    return powers


def _monster_to_spirecomm(combat: Any, index: int, monster: Any) -> dict[str, Any]:
    adjusted_damage = monster_adjusted_damage(
        monster,
        combat.player,
        vulnerable_multiplier=combat._monster_vulnerable_multiplier(),
    )
    return {
        "block": monster.block,
        "current_hp": monster.current_hp,
        "half_dead": monster.half_dead,
        "intent": monster.intent,
        "is_gone": monster.is_gone,
        "last_move_id": monster.move_history[0] if monster.move_history else None,
        "max_hp": monster.max_hp,
        "monster_id": _spirecomm_monster_id(monster.monster_id),
        "monster_index": index,
        "move_adjusted_damage": adjusted_damage,
        "move_base_damage": monster.move_base_damage,
        "move_hits": monster.move_hits,
        "move_id": _serialize_move_name(monster.move),
        "move_name": _serialize_move_name(monster.move),
        "name": monster.name,
        "powers": _monster_powers_to_spirecomm(monster),
        "second_last_move_id": monster.move_history[1] if len(monster.move_history) > 1 else None,
    }


def combat_state(combat: Any) -> dict[str, Any]:
    visible_monsters = list(combat.monsters)
    combat_state_payload = {
        "card_in_play": None,
        "cards_discarded_this_turn": combat.cards_discarded_this_turn,
        "discard_pile": [card_to_spirecomm(card) for card in combat.discard_pile],
        "draw_pile": [card_to_spirecomm(card) for card in combat.draw_pile],
        "exhaust_pile": [card_to_spirecomm(card) for card in combat.exhaust_pile],
        "hand": [_card_to_spirecomm_in_combat(combat, card) for card in combat.hand],
        "limbo": [],
        "monsters": [_monster_to_spirecomm(combat, index, monster) for index, monster in enumerate(visible_monsters)],
        "player": {
            "block": combat.player.block,
            "current_hp": combat.player.current_hp,
            "energy": combat.player.energy,
            "max_hp": combat.player.max_hp,
            "orbs": [],
            "powers": _player_powers_to_spirecomm(combat.player),
        },
        "turn": combat.turn,
    }
    return {
        "act": combat.act,
        "act_boss": combat.act_boss,
        "ascension_level": combat.ascension_level,
        "character": "IRONCLAD",
        "observation_version": CANONICAL_STATE_VERSION,
        "choice_available": False,
        "choice_list": [],
        "combat_state": combat_state_payload,
        "commands": {
            "cancel": False,
            "end": combat.outcome == "UNDECIDED",
            "play": combat.outcome == "UNDECIDED",
            "potion": combat.outcome == "UNDECIDED" and any(potion.can_use for potion in combat.potions),
            "proceed": False,
        },
        "current_hp": combat.player.current_hp,
        "deck": [card_to_spirecomm(card) for card in combat.deck],
        "floor": combat.floor,
        "gold": combat.gold,
        "in_combat": combat.outcome == "UNDECIDED",
        "max_hp": combat.player.max_hp,
        "potions": potions_to_spirecomm(combat.potions),
        "relics": combat.relics,
        "room_phase": "COMBAT",
        "room_type": "MonsterRoom",
        "screen": "COMBAT",
        "screen_up": False,
        "seed": combat.seed,
    }


def run_state(env: Any) -> dict[str, Any]:
    if env.phase == "COMBAT":
        return combat_state(env.combat)
    return {
        "act": env.act,
        "act_boss": env.act_boss,
        "ascension_level": env.ascension_level,
        "character": "IRONCLAD",
        "observation_version": CANONICAL_STATE_VERSION,
        "choice_available": env.phase in {"NEOW", "CARD_REWARD", "CARD_SELECT", "BOSS_RELIC", "MAP", "CAMPFIRE", "SHOP", "EVENT", "TREASURE", "CHEST"},
        "choice_list": env._choice_list(),
        "combat_state": None,
        "commands": {
            "cancel": False,
            "end": False,
            "play": False,
            "potion": False,
            "proceed": env.phase == "CARD_REWARD",
        },
        "current_hp": env.player.current_hp,
        "deck": [card_to_spirecomm(card) for card in env.deck],
        "floor": env.floor,
        "gold": env.gold,
        "in_combat": False,
        "keys": sorted(env.keys),
        "max_hp": env.player.max_hp,
        "potions": potions_to_spirecomm(env.potions),
        "relics": env.relics,
        "room_phase": "COMPLETE" if env.phase in {"NEOW", "CARD_REWARD", "CARD_SELECT", "BOSS_RELIC", "MAP", "CAMPFIRE", "SHOP", "EVENT", "TREASURE", "CHEST"} else env.phase,
        "room_type": env._room_type(),
        "screen": env.phase,
        "screen_up": env.phase in {"NEOW", "CARD_REWARD", "CARD_SELECT", "BOSS_RELIC", "MAP", "CAMPFIRE", "SHOP", "EVENT", "TREASURE", "CHEST"},
        "seed": env.seed,
    }


__all__ = ["combat_state", "run_state"]
