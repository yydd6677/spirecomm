from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
import struct

from spirecomm.native_sim_v3.content.act_progression import dungeon_id_for_act
from spirecomm.native_sim_v3.content.cards import can_upgrade_card, card_library_random_curse_pool, card_pools, class_reward_pool_key, make_card
from spirecomm.native_sim_v3.content.events import (
    abstract_dungeon_event_gate_rules,
    abstract_dungeon_shrine_gate_rules,
    dungeon_event_ids,
    dungeon_shrine_ids,
    event_catalog,
    note_for_yourself_defaults,
    note_for_yourself_available,
    special_one_time_event_ids,
)
from spirecomm.native_sim_v3.content.potions import make_potion
from spirecomm.native_sim_v3.combat.engine import build_encounter
from spirecomm.native_sim_v3.content.relics import (
    draw_random_relic,
    draw_random_screenless_relic,
    make_relic,
    roll_random_relic_tier,
)
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, java_shuffle_in_place
from spirecomm.native_sim_v3.run.rewards import apply_reward_preview_relics, generate_colorless_reward_group
from spirecomm.native_sim_v3.source_paths import sts_source_path

EVENT_HELPER_SOURCE = sts_source_path("helpers/EventHelper.java")
ABSTRACT_DUNGEON_SOURCE = sts_source_path("dungeons/AbstractDungeon.java")
EVENT_HELPER_FLOAT_PATTERN = re.compile(r"private static final float ([A-Z_]+) = ([0-9.]+)f;")
SHRINE_CHANCE_PATTERN = re.compile(r"shrineChance = ([0-9.]+)f;")


def _is_potion_slot(potion: dict[str, object]) -> bool:
    potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
    return potion_id == "Potion Slot"


@lru_cache(maxsize=1)
def _event_helper_probability_constants() -> dict[str, float]:
    text = EVENT_HELPER_SOURCE.read_text(encoding="utf-8")
    parsed = {name: float(value) for name, value in EVENT_HELPER_FLOAT_PATTERN.findall(text)}
    required = {
        "BASE_ELITE_CHANCE",
        "BASE_MONSTER_CHANCE",
        "BASE_SHOP_CHANCE",
        "BASE_TREASURE_CHANCE",
        "RAMP_ELITE_CHANCE",
        "RAMP_MONSTER_CHANCE",
        "RAMP_SHOP_CHANCE",
        "RAMP_TREASURE_CHANCE",
        "RESET_ELITE_CHANCE",
        "RESET_MONSTER_CHANCE",
        "RESET_SHOP_CHANCE",
        "RESET_TREASURE_CHANCE",
    }
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"missing EventHelper constants {sorted(missing)!r} in {EVENT_HELPER_SOURCE}")
    return parsed


def _java_float(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _java_float_chance_size(chance: float) -> int:
    return int(_java_float(_java_float(chance) * _java_float(100.0)))


def _mathutils_round_positive(value: float) -> int:
    return int(float(value) + 0.5)


@lru_cache(maxsize=1)
def _abstract_dungeon_shrine_chance() -> float:
    text = ABSTRACT_DUNGEON_SOURCE.read_text(encoding="utf-8")
    match = SHRINE_CHANCE_PATTERN.search(text)
    if match is None:
        raise ValueError(f"could not locate shrineChance in {ABSTRACT_DUNGEON_SOURCE}")
    return float(match.group(1))


_EVENT_PROBABILITIES = {key: _java_float(value) for key, value in _event_helper_probability_constants().items()}
BASE_ELITE_CHANCE = _EVENT_PROBABILITIES["BASE_ELITE_CHANCE"]
BASE_MONSTER_CHANCE = _EVENT_PROBABILITIES["BASE_MONSTER_CHANCE"]
BASE_SHOP_CHANCE = _EVENT_PROBABILITIES["BASE_SHOP_CHANCE"]
BASE_TREASURE_CHANCE = _EVENT_PROBABILITIES["BASE_TREASURE_CHANCE"]
RAMP_ELITE_CHANCE = _EVENT_PROBABILITIES["RAMP_ELITE_CHANCE"]
RAMP_MONSTER_CHANCE = _EVENT_PROBABILITIES["RAMP_MONSTER_CHANCE"]
RAMP_SHOP_CHANCE = _EVENT_PROBABILITIES["RAMP_SHOP_CHANCE"]
RAMP_TREASURE_CHANCE = _EVENT_PROBABILITIES["RAMP_TREASURE_CHANCE"]
RESET_ELITE_CHANCE = _EVENT_PROBABILITIES["RESET_ELITE_CHANCE"]
RESET_MONSTER_CHANCE = _EVENT_PROBABILITIES["RESET_MONSTER_CHANCE"]
RESET_SHOP_CHANCE = _EVENT_PROBABILITIES["RESET_SHOP_CHANCE"]
RESET_TREASURE_CHANCE = _EVENT_PROBABILITIES["RESET_TREASURE_CHANCE"]
SHRINE_CHANCE = _abstract_dungeon_shrine_chance()


@dataclass(slots=True)
class EventState:
    event_id: str
    screen: str = "INTRO"
    rewards_opened: bool = False
    result: int | None = None
    data: dict[str, object] = field(default_factory=dict)

    def actions(
        self,
        *,
        ascension_level: int,
        max_hp: int,
        gold: int = 0,
        deck: list[dict[str, object]] | None = None,
        relics: list[dict[str, object]] | None = None,
        potions: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        purgeable = _purgeable_candidate_indexes(deck or [])
        upgradable = _upgradable_candidate_indexes(deck or [])
        if self.event_id == "Scrap Ooze":
            self.data.setdefault("relic_chance", 25)
            self.data.setdefault("dmg", 5 if ascension_level >= 15 else 3)
            self.data.setdefault("total_damage", 0)
            if self.screen == "INTRO":
                return [
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": f"Open ({int(self.data['dmg'])} HP, {int(self.data['relic_chance'])}%)",
                        "label": f"Open ({int(self.data['dmg'])} HP, {int(self.data['relic_chance'])}%)",
                        "choice_index": 0,
                    },
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Shining Light":
            if self.screen == "INTRO":
                actions: list[dict[str, object]] = []
                if upgradable:
                    damage = _mathutils_round_positive(max_hp * (0.3 if ascension_level >= 15 else 0.2))
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Enter ({damage} HP)", "label": f"Enter ({damage} HP)", "choice_index": 0})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Golden Wing":
            if self.screen == "INTRO":
                can_attack = _has_card_with_x_damage(deck or [], 10)
                actions: list[dict[str, object]] = [
                    {"kind": "event", "event_id": self.event_id, "name": "Pray", "label": "Pray", "choice_index": 0},
                ]
                if can_attack:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Smash", "label": "Smash", "choice_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2 if can_attack else 1})
                return actions
            if self.screen == "PURGE":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Liars Game":
            if self.screen == "INTRO":
                gold_reward = 150 if ascension_level >= 15 else 175
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Agree ({gold_reward} Gold)", "label": f"Agree ({gold_reward} Gold)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "WeMeetAgain":
            if self.screen == "INTRO":
                actions: list[dict[str, object]] = []
                enabled_index = 0
                if self.data.get("potion_index") is not None:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Give Potion", "label": "Give Potion", "choice_index": enabled_index, "button_index": 0})
                    enabled_index += 1
                if int(self.data.get("gold_amount") or 0) > 0:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Pay {int(self.data['gold_amount'])} Gold", "label": f"Pay {int(self.data['gold_amount'])} Gold", "choice_index": enabled_index, "button_index": 1})
                    enabled_index += 1
                if self.data.get("card_uuid") is not None:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Give Card", "label": "Give Card", "choice_index": enabled_index, "button_index": 2})
                    enabled_index += 1
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": enabled_index, "button_index": 3})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Lab":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Search", "label": "Search", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Bonfire Elementals":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Approach", "label": "Approach", "choice_index": 0}]
            if self.screen == "CHOOSE":
                return [{"kind": "event", "event_id": self.event_id, "name": "Offer", "label": "Offer", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "FaceTrader":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Approach", "label": "Approach", "choice_index": 0}]
            if self.screen == "MAIN":
                damage = max(1, max_hp // 10)
                gold_reward = 50 if ascension_level >= 15 else 75
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Touch ({damage} HP for {gold_reward} Gold)", "label": f"Touch ({damage} HP for {gold_reward} Gold)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Trade", "label": "Trade", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "World of Goop":
            gold_loss = int(self.data.get("gold_loss") or 0)
            if self.screen == "INTRO":
                return [
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Gather Gold",
                        "label": "Gather Gold",
                        "text": "[Gather Gold] Gain 75 Gold. Lose 11 HP.",
                        "choice_index": 0,
                    },
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Leave It",
                        "label": "Leave It",
                        "text": f"[Leave It] Lose {gold_loss} Gold.",
                        "choice_index": 1,
                    },
                ]
            return [
                {
                    "kind": "event",
                    "event_id": self.event_id,
                    "name": "Leave",
                    "label": "Leave",
                    "text": "[Leave]",
                    "choice_index": 0,
                }
            ]
        if self.event_id == "Mushrooms":
            if self.screen == "INTRO":
                heal_amt = int(max_hp * 0.25)
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Fight", "label": "Fight", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": f"Heal ({heal_amt} HP) and take Parasite", "label": f"Heal ({heal_amt} HP) and take Parasite", "choice_index": 1},
                ]
            if self.screen == "FIGHT":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Golden Shrine":
            if self.screen == "INTRO":
                gold_amt = 50 if ascension_level >= 15 else 100
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Pray ({gold_amt} Gold)", "label": f"Pray ({gold_amt} Gold)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Desecrate (275 Gold + Regret)", "label": "Desecrate (275 Gold + Regret)", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Match and Keep!":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Play", "label": "Play", "choice_index": 0}]
            if self.screen == "RULE_EXPLANATION":
                return [{"kind": "event", "event_id": self.event_id, "name": "Start", "label": "Start", "choice_index": 0}]
            if self.screen == "PLAY":
                actions: list[dict[str, object]] = []
                revealed = {int(index) for index in list(self.data.get("revealed_card_indexes") or [])}
                cards = list(self.data.get("cards") or [])
                for choice_index, card_index in enumerate(_match_and_keep_pickable_indexes(self)):
                    card = dict(cards[int(card_index)])
                    position = _match_and_keep_card_position(int(card_index))
                    known = int(card_index) in revealed
                    label = str(card.get("card_id") or card.get("name") or f"card{position}") if known else f"card{position}"
                    actions.append(
                        {
                            "kind": "event",
                            "event_id": self.event_id,
                            "name": label,
                            "label": label,
                            "choice_index": choice_index,
                            "match_card_index": int(card_index),
                            "match_position": position,
                            "known": known,
                            "card_id": str(card.get("card_id") or "") if known else None,
                        }
                    )
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Dead Adventurer":
            chance = int(self.data.get("encounter_chance") or (35 if ascension_level >= 15 else 25))
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Search ({chance}%)", "label": f"Search ({chance}%)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            if self.screen == "FAIL":
                return [{"kind": "event", "event_id": self.event_id, "name": "Fight", "label": "Fight", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Falling":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            if self.screen == "CHOICE":
                if not any(self.data.get(key) for key in ("skill_uuid", "power_uuid", "attack_uuid")):
                    return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
                actions: list[dict[str, object]] = []
                if self.data.get("skill_uuid"):
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Lose Skill: {str(self.data.get('skill_name') or 'Skill')}", "label": f"Lose Skill: {str(self.data.get('skill_name') or 'Skill')}", "choice_index": 0})
                if self.data.get("power_uuid"):
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Lose Power: {str(self.data.get('power_name') or 'Power')}", "label": f"Lose Power: {str(self.data.get('power_name') or 'Power')}", "choice_index": 1})
                if self.data.get("attack_uuid"):
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Lose Attack: {str(self.data.get('attack_name') or 'Attack')}", "label": f"Lose Attack: {str(self.data.get('attack_name') or 'Attack')}", "choice_index": 2})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Fountain of Cleansing":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Drink", "label": "Drink", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Knowing Skull":
            if self.screen == "INTRO_1":
                return [{"kind": "event", "event_id": self.event_id, "name": "Approach", "label": "Approach", "choice_index": 0}]
            if self.screen == "ASK":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Potion ({int(self.data.get('potion_cost') or 6)} HP)", "label": f"Potion ({int(self.data.get('potion_cost') or 6)} HP)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": f"90 Gold ({int(self.data.get('gold_cost') or 6)} HP)", "label": f"90 Gold ({int(self.data.get('gold_cost') or 6)} HP)", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": f"Colorless Card ({int(self.data.get('card_cost') or 6)} HP)", "label": f"Colorless Card ({int(self.data.get('card_cost') or 6)} HP)", "choice_index": 2},
                    {"kind": "event", "event_id": self.event_id, "name": f"Leave ({int(self.data.get('leave_cost') or 6)} HP)", "label": f"Leave ({int(self.data.get('leave_cost') or 6)} HP)", "choice_index": 3},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Back to Basics":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Elegance", "label": "Elegance", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Simplicity", "label": "Simplicity", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Designer":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Approach", "label": "Approach", "choice_index": 0}]
            if self.screen == "MAIN":
                actions: list[dict[str, object]] = []
                non_bottled_count = _non_bottled_card_count(deck or [])
                adjust_cost = int(self.data["adjust_cost"])
                clean_cost = int(self.data["clean_cost"])
                full_cost = int(self.data["full_cost"])
                if int(gold) >= adjust_cost and upgradable:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Adjust ({adjust_cost} Gold)", "label": f"Adjust ({adjust_cost} Gold)", "choice_index": 0})
                if bool(self.data.get("clean_up_removes_cards")):
                    clean_enabled = non_bottled_count > 0
                else:
                    clean_enabled = non_bottled_count >= 2
                if int(gold) >= clean_cost and clean_enabled:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Clean Up ({clean_cost} Gold)", "label": f"Clean Up ({clean_cost} Gold)", "choice_index": 1})
                if int(gold) >= full_cost and non_bottled_count > 0:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Full Service ({full_cost} Gold)", "label": f"Full Service ({full_cost} Gold)", "choice_index": 2})
                actions.append({"kind": "event", "event_id": self.event_id, "name": f"Punch ({int(self.data['hp_loss'])} HP)", "label": f"Punch ({int(self.data['hp_loss'])} HP)", "choice_index": 3})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Forgotten Altar":
            if self.screen == "INTRO":
                has_idol = any(str(relic.get("relic_id") or relic.get("id")) == "Golden Idol" for relic in (relics or []))
                if has_idol:
                    return [
                        {"kind": "event", "event_id": self.event_id, "name": "Offer Idol", "label": "Offer Idol", "choice_index": 0},
                        {"kind": "event", "event_id": self.event_id, "name": f"Shed Blood (+5 Max HP, {int(self.data['hp_loss'])} HP)", "label": f"Shed Blood (+5 Max HP, {int(self.data['hp_loss'])} HP)", "choice_index": 1},
                        {"kind": "event", "event_id": self.event_id, "name": "Smash (Decay)", "label": "Smash (Decay)", "choice_index": 2},
                    ]
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Shed Blood (+5 Max HP, {int(self.data['hp_loss'])} HP)", "label": f"Shed Blood (+5 Max HP, {int(self.data['hp_loss'])} HP)", "choice_index": 0, "button_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": "Smash (Decay)", "label": "Smash (Decay)", "choice_index": 1, "button_index": 2},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Cursed Tome":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Read", "label": "Read", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            if self.screen in {"PAGE_1", "PAGE_2", "PAGE_3"}:
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            if self.screen == "LAST_PAGE":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Take Book ({int(self.data['final_dmg'])} HP)", "label": f"Take Book ({int(self.data['final_dmg'])} HP)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Stop", "label": "Stop", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Duplicator":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Pray", "label": "Pray", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Beggar":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Give 75 Gold", "label": "Give 75 Gold", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            if self.screen == "GAVE_MONEY":
                return [{"kind": "event", "event_id": self.event_id, "name": "Purge", "label": "Purge", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "The Library":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Read", "label": "Read", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": f"Sleep ({int(self.data['heal_amt'])} HP)", "label": f"Sleep ({int(self.data['heal_amt'])} HP)", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Masked Bandits":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Pay All Gold", "label": "Pay All Gold", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Fight", "label": "Fight", "choice_index": 1},
                ]
            if self.screen in {"PAID_1", "PAID_2", "PAID_3"}:
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Nest":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            if self.screen == "CHOOSE":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Steal {int(self.data.get('gold_gain') or 99)} Gold", "label": f"Steal {int(self.data.get('gold_gain') or 99)} Gold", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Join the Cult (6 HP)", "label": "Join the Cult (6 HP)", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "N'loth":
            if self.screen == "INTRO":
                actions = []
                if self.data.get("choice1_name"):
                    actions.append({
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": f"Offer {str(self.data['choice1_name'])}",
                        "label": f"Offer {str(self.data['choice1_name'])}",
                        "choice_index": 0,
                    })
                if self.data.get("choice2_name"):
                    actions.append({
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": f"Offer {str(self.data['choice2_name'])}",
                        "label": f"Offer {str(self.data['choice2_name'])}",
                        "choice_index": 1,
                    })
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "The Joust":
            if self.screen == "HALT":
                return [{"kind": "event", "event_id": self.event_id, "name": "Approach", "label": "Approach", "choice_index": 0}]
            if self.screen == "EXPLANATION":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Bet Against (50 Gold -> 100 Gold)", "label": "Bet Against (50 Gold -> 100 Gold)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Bet For (50 Gold -> 250 Gold)", "label": "Bet For (50 Gold -> 250 Gold)", "choice_index": 1},
                ]
            if self.screen == "PRE_JOUST":
                return [{"kind": "event", "event_id": self.event_id, "name": "Begin Joust", "label": "Begin Joust", "choice_index": 0}]
            if self.screen == "JOUST":
                return [{"kind": "event", "event_id": self.event_id, "name": "Resolve Joust", "label": "Resolve Joust", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "The Moai Head":
            if self.screen == "INTRO":
                actions = [
                    {"kind": "event", "event_id": self.event_id, "name": f"Heal to Full (-{int(self.data.get('hp_amt') or 0)} Max HP)", "label": f"Heal to Full (-{int(self.data.get('hp_amt') or 0)} Max HP)", "choice_index": 0},
                ]
                if "Golden Idol" in _relic_ids(relics or []):
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Offer Golden Idol (333 Gold)", "label": "Offer Golden Idol (333 Gold)", "choice_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2 if len(actions) > 1 else 1})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Tomb of Lord Red Mask":
            if self.screen == "INTRO":
                actions = []
                if "Red Mask" in _relic_ids(relics or []):
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Wear the Red Mask (222 Gold)", "label": "Wear the Red Mask (222 Gold)", "choice_index": 0})
                else:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Offer All Gold ({gold} Gold) for Red Mask", "label": f"Offer All Gold ({gold} Gold) for Red Mask", "choice_index": 0})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Winding Halls":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            if self.screen == "CHOICE":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Embrace Madness ({int(self.data.get('hp_amt') or 0)} HP)", "label": f"Embrace Madness ({int(self.data.get('hp_amt') or 0)} HP)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": f"Writhe ({int(self.data.get('heal_amt') or 0)} HP)", "label": f"Writhe ({int(self.data.get('heal_amt') or 0)} HP)", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": f"Max HP ({int(self.data.get('max_hp_amt') or 0)} Max HP)", "label": f"Max HP ({int(self.data.get('max_hp_amt') or 0)} Max HP)", "choice_index": 2},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "MindBloom":
            if self.screen == "INTRO":
                late_branch = bool(self.data.get("late_branch"))
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "I Am War", "label": "I Am War", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "I Am Awake", "label": "I Am Awake", "choice_index": 1},
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "I Am Rich" if not late_branch else "I Am Healthy",
                        "label": "I Am Rich" if not late_branch else "I Am Healthy",
                        "choice_index": 2,
                    },
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Addict":
            if self.screen == "INTRO":
                actions: list[dict[str, object]] = []
                if gold >= 85:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Pay 85 Gold", "label": "Pay 85 Gold", "choice_index": 0, "button_index": 0})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Take Shame", "label": "Take Shame", "choice_index": len(actions), "button_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": len(actions), "button_index": 2})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Drug Dealer":
            if self.screen == "INTRO":
                raw_purgeable = _raw_purgeable_candidate_indexes(deck or [])
                actions = [
                    {"kind": "event", "event_id": self.event_id, "name": "Obtain J.A.X.", "label": "Obtain J.A.X.", "choice_index": 0},
                ]
                if len(raw_purgeable) >= 2:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Become Test Subject", "label": "Become Test Subject", "choice_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Inject Mutagens", "label": "Inject Mutagens", "choice_index": 2})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Ghosts":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Accept ({int(self.data['hp_loss'])} Max HP)", "label": f"Accept ({int(self.data['hp_loss'])} Max HP)", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Vampires":
            if self.screen == "INTRO":
                actions = [
                    {"kind": "event", "event_id": self.event_id, "name": f"Accept ({int(self.data['max_hp_loss'])} Max HP)", "label": f"Accept ({int(self.data['max_hp_loss'])} Max HP)", "choice_index": 0},
                ]
                if bool(self.data.get("has_vial")):
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Offer Blood Vial", "label": "Offer Blood Vial", "choice_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2 if self.data.get('has_vial') else 1})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "NoteForYourself":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Read", "label": "Read", "choice_index": 0}]
            if self.screen == "CHOOSE":
                obtain_card = dict(self.data.get("obtain_card") or make_card("Iron Wave"))
                return [
                    {"kind": "event", "event_id": self.event_id, "name": f"Take {obtain_card['name']}", "label": f"Take {obtain_card['name']}", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Wheel of Change":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Play", "label": "Play", "choice_index": 0}]
            if self.screen == "SPIN":
                return [{"kind": "event", "event_id": self.event_id, "name": "Spin", "label": "Spin", "choice_index": 0}]
            if self.screen == "RESULT":
                return [{"kind": "event", "event_id": self.event_id, "name": "Prize?", "label": "Prize?", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Big Fish":
            if self.screen == "INTRO":
                heal_amt = max_hp // 3
                return [
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Banana",
                        "label": "Banana",
                        "text": f"[Banana] Heal {heal_amt} HP.",
                        "choice_index": 0,
                    },
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Donut",
                        "label": "Donut",
                        "text": "[Donut] Max HP +5.",
                        "choice_index": 1,
                    },
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Box",
                        "label": "Box",
                        "text": "[Box] Obtain a Relic. Become Cursed - Regret.",
                        "choice_index": 2,
                    },
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "The Cleric":
            if self.screen == "INTRO":
                heal_amt = int(max_hp * 0.25)
                actions: list[dict[str, object]] = []
                if gold >= 35:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": f"Heal ({heal_amt} HP)", "label": f"Heal ({heal_amt} HP)", "choice_index": 0})
                if gold >= (75 if ascension_level >= 15 else 50) and purgeable:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Purify", "label": "Purify", "choice_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Living Wall":
            if self.screen == "INTRO":
                actions: list[dict[str, object]] = []
                if purgeable:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Forget", "label": "Forget", "choice_index": 0})
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Change", "label": "Change", "choice_index": 1})
                if upgradable:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Grow", "label": "Grow", "choice_index": 2})
                return actions or [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 3}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "The Woman in Blue":
            if self.screen == "INTRO":
                ignored = "Ignored"
                if ascension_level >= 15:
                    ignored = f"Punch ({max(1, (max_hp + 19) // 20)} HP)"
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Buy 1 Potion", "label": "Buy 1 Potion", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Buy 2 Potions", "label": "Buy 2 Potions", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": "Buy 3 Potions", "label": "Buy 3 Potions", "choice_index": 2},
                    {"kind": "event", "event_id": self.event_id, "name": ignored, "label": ignored, "choice_index": 3},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Golden Idol":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Take Golden Idol", "label": "Take Golden Idol", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Ignore", "label": "Ignore", "choice_index": 1},
                ]
            if self.screen == "BOULDER":
                hp_loss = _golden_idol_hp_loss(max_hp=max_hp, ascension_level=ascension_level)
                max_hp_loss = _golden_idol_max_hp_loss(max_hp=max_hp, ascension_level=ascension_level)
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Take Injury", "label": "Take Injury", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": f"Take {hp_loss} Damage", "label": f"Take {hp_loss} Damage", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": f"Lose {max_hp_loss} Max HP", "label": f"Lose {max_hp_loss} Max HP", "choice_index": 2},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "The Mausoleum":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Opened", "label": "Opened", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Ignored", "label": "Ignored", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Accursed Blacksmith":
            if self.screen == "INTRO":
                actions: list[dict[str, object]] = []
                if upgradable:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Forge", "label": "Forge", "choice_index": 0})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Rummage", "label": "Rummage", "choice_index": 1})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 2})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Purifier":
            if self.screen == "INTRO":
                return [
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Pray",
                        "label": "Pray",
                        "text": "[Pray] Remove a card from your deck.",
                        "choice_index": 0,
                    },
                    {
                        "kind": "event",
                        "event_id": self.event_id,
                        "name": "Leave",
                        "label": "Leave",
                        "text": "[Leave]",
                        "choice_index": 1,
                    },
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "text": "[Leave]", "choice_index": 0}]
        if self.event_id == "Upgrade Shrine":
            if self.screen == "INTRO":
                actions: list[dict[str, object]] = []
                if upgradable:
                    actions.append({"kind": "event", "event_id": self.event_id, "name": "Pray", "label": "Pray", "choice_index": 0})
                actions.append({"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1})
                return actions
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Transmorgrifier":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Transmogrify", "label": "Transmogrify", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Colosseum":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Approach", "label": "Approach", "choice_index": 0}]
            if self.screen == "FIGHT":
                return [{"kind": "event", "event_id": self.event_id, "name": "Fight", "label": "Fight", "choice_index": 0}]
            if self.screen == "POST_COMBAT":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Flee", "label": "Flee", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Fight Nobs", "label": "Fight Nobs", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Mysterious Sphere":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Open", "label": "Open", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            if self.screen == "PRE_COMBAT":
                return [{"kind": "event", "event_id": self.event_id, "name": "Fight", "label": "Fight", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "SecretPortal":
            if self.screen == "INTRO":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Enter Portal", "label": "Enter Portal", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 1},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
        if self.event_id == "SensoryStone":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Touch", "label": "Touch", "choice_index": 0}]
            if self.screen == "INTRO_2":
                return [
                    {"kind": "event", "event_id": self.event_id, "name": "Memory 1", "label": "Memory 1", "choice_index": 0},
                    {"kind": "event", "event_id": self.event_id, "name": "Memory 2 (5 HP)", "label": "Memory 2 (5 HP)", "choice_index": 1},
                    {"kind": "event", "event_id": self.event_id, "name": "Memory 3 (10 HP)", "label": "Memory 3 (10 HP)", "choice_index": 2},
                ]
            return [{"kind": "event", "event_id": self.event_id, "name": "Leave", "label": "Leave", "choice_index": 0}]
        if self.event_id == "Spire Heart":
            if self.screen == "INTRO":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            if self.screen == "MIDDLE":
                return [{"kind": "event", "event_id": self.event_id, "name": "Strike", "label": "Strike", "choice_index": 0}]
            if self.screen == "MIDDLE_2":
                return [{"kind": "event", "event_id": self.event_id, "name": "Continue", "label": "Continue", "choice_index": 0}]
            if self.screen == "GO_TO_ENDING":
                return [{"kind": "event", "event_id": self.event_id, "name": "Enter", "label": "Enter", "choice_index": 0}]
            return [{"kind": "event", "event_id": self.event_id, "name": "Die", "label": "Die", "choice_index": 0}]
        return [{"kind": "not_implemented", "event_id": self.event_id, "name": "NOT_IMPLEMENTED", "label": "NOT_IMPLEMENTED", "choice_index": 0}]


def roll_question_room_result(
    randoms: NativeRandomSet,
    *,
    floor: int,
    current_room_type: str | None,
    relics: list[dict[str, object]] | None,
    elite_chance: float,
    monster_chance: float,
    shop_chance: float,
    treasure_chance: float,
) -> tuple[str, dict[str, float]]:
    event_rng = randoms.stream("event")
    relic_ids = {
        str(relic.get("relic_id") or relic.get("id") or "")
        for relic in (relics or [])
    }
    force_treasure = False
    if "Tiny Chest" in relic_ids:
        for relic in relics or []:
            relic_id = str(relic.get("relic_id") or relic.get("id") or "")
            if relic_id != "Tiny Chest":
                continue
            counter = int(relic.get("counter") or 0) + 1
            relic["counter"] = counter
            if counter >= 4:
                relic["counter"] = 0
                force_treasure = True
            break

    roll = float(event_rng.random(1.0))
    # Vanilla StS only allows ? rooms to roll elites when the DeadlyEvents mod
    # is enabled. We model the unmodded game, so the accumulated elite chance
    # is saved/ramped but never contributes to the room-result table.
    elite_size = 0
    monster_size = _java_float_chance_size(monster_chance)
    shop_size = 0 if str(current_room_type or "") == "ShopRoom" else _java_float_chance_size(shop_chance)
    treasure_size = _java_float_chance_size(treasure_chance)

    possible_results = ["EVENT"] * 100
    fill_index = 0
    if elite_size > 0:
        for index in range(fill_index, min(100, fill_index + elite_size)):
            possible_results[index] = "ELITE"
        fill_index += elite_size
    if monster_size > 0:
        for index in range(fill_index, min(100, fill_index + monster_size)):
            possible_results[index] = "MONSTER"
        fill_index += monster_size
    if shop_size > 0:
        for index in range(fill_index, min(100, fill_index + shop_size)):
            possible_results[index] = "SHOP"
        fill_index += shop_size
    if treasure_size > 0:
        for index in range(fill_index, min(100, fill_index + treasure_size)):
            possible_results[index] = "TREASURE"

    choice = possible_results[min(99, int(roll * 100.0))]
    if force_treasure:
        choice = "TREASURE"

    next_elite_chance = RESET_ELITE_CHANCE if choice == "ELITE" else _java_float(elite_chance + RAMP_ELITE_CHANCE)
    next_monster_chance = RESET_MONSTER_CHANCE if choice == "MONSTER" else _java_float(monster_chance + RAMP_MONSTER_CHANCE)
    next_shop_chance = RESET_SHOP_CHANCE if choice == "SHOP" else _java_float(shop_chance + RAMP_SHOP_CHANCE)
    next_treasure_chance = RESET_TREASURE_CHANCE if choice == "TREASURE" else _java_float(treasure_chance + RAMP_TREASURE_CHANCE)

    if choice == "MONSTER" and "Juzu Bracelet" in relic_ids:
        choice = "EVENT"

    return choice, {
        "elite": next_elite_chance,
        "monster": next_monster_chance,
        "shop": next_shop_chance,
        "treasure": next_treasure_chance,
    }


def initialize_special_one_time_event_list(
    *,
    ascension_level: int,
    is_daily_run: bool = False,
    highest_unlocked_ascension: int | None = None,
) -> list[str]:
    include_note = note_for_yourself_available(
        ascension_level=ascension_level,
        is_daily_run=is_daily_run,
        highest_unlocked_ascension=highest_unlocked_ascension,
    )
    return list(special_one_time_event_ids(include_note_for_yourself=include_note))


def initialize_event_pools_for_act(act: int) -> tuple[list[str], list[str]]:
    return initialize_event_pools_for_dungeon(dungeon_id_for_act(act))


def initialize_event_pools_for_dungeon(dungeon_id: str) -> tuple[list[str], list[str]]:
    return list(dungeon_event_ids(str(dungeon_id))), list(dungeon_shrine_ids(str(dungeon_id)))


_IS_CURSED_PROTECTED_CARD_IDS = {"AscendersBane", "CurseOfTheBell", "Necronomicurse"}
_UNPURGEABLE_CARD_IDS = {"AscendersBane", "CurseOfTheBell", "Necronomicurse"}


def _has_curse(deck: list[dict[str, object]]) -> bool:
    return any(
        str(card.get("type") or "") == "CURSE"
        and str(card.get("card_id") or card.get("id") or "") not in _IS_CURSED_PROTECTED_CARD_IDS
        for card in deck
    )


def _relic_ids(relics: list[dict[str, object]]) -> set[str]:
    return {
        str(relic.get("relic_id") or relic.get("id") or "")
        for relic in relics
    }


def _apply_event_hp_loss(current_hp: int, amount: int, relics: list[dict[str, object]] | None = None) -> int:
    loss = max(0, int(amount))
    if loss > 0 and "TungstenRod" in _relic_ids(relics or []):
        loss = max(0, loss - 1)
    return max(0, int(current_hp) - loss)


def _apply_event_normal_damage(current_hp: int, amount: int, relics: list[dict[str, object]] | None = None) -> int:
    damage = max(0, int(amount))
    relic_ids = _relic_ids(relics or [])
    if 1 < damage <= 5 and "Torii" in relic_ids:
        damage = 1
    if damage > 0 and "TungstenRod" in relic_ids:
        damage = max(0, damage - 1)
    return max(0, int(current_hp) - damage)


def _is_event_available(
    event_id: str,
    *,
    floor: int,
    gold: int,
    relic_ids: set[str],
    current_hp: int,
    max_hp: int,
    current_node_y: int | None,
    map_height: int,
) -> bool:
    rule = abstract_dungeon_event_gate_rules().get(event_id)
    if rule is None:
        return True
    if rule.floor_gt is not None and int(floor) <= int(rule.floor_gt):
        return False
    if rule.gold_ge is not None and int(gold) < int(rule.gold_ge):
        return False
    if rule.current_node_y_gt_half and (current_node_y is None or int(current_node_y) <= (int(map_height) // 2)):
        return False
    if rule.required_relic_id is not None and rule.hp_ratio_le is not None:
        if rule.required_relic_id not in relic_ids and (float(current_hp) / float(max(1, max_hp))) > float(rule.hp_ratio_le):
            return False
    elif rule.required_relic_id is not None and rule.required_relic_id not in relic_ids:
        return False
    return True


def _is_special_event_available(
    event_id: str,
    *,
    dungeon_id: str,
    gold: int,
    deck: list[dict[str, object]],
    relic_count: int,
    current_hp: int,
    playtime_seconds: float,
) -> bool:
    rule = abstract_dungeon_shrine_gate_rules().get(event_id)
    if rule is None:
        return True
    if rule.require_curse and not _has_curse(deck):
        return False
    if rule.dungeon_ids and dungeon_id not in set(rule.dungeon_ids):
        return False
    if rule.gold_ge is not None and int(gold) < int(rule.gold_ge):
        return False
    if rule.current_hp_gt is not None and int(current_hp) <= int(rule.current_hp_gt):
        return False
    if rule.relic_count_ge is not None and int(relic_count) < int(rule.relic_count_ge):
        return False
    if rule.playtime_seconds_ge is not None and float(playtime_seconds) < float(rule.playtime_seconds_ge):
        return False
    return True


def _choose_shrine(
    event_rng,
    *,
    dungeon_id: str,
    shrine_list: list[str],
    special_one_time_event_list: list[str],
    gold: int,
    deck: list[dict[str, object]],
    relic_count: int,
    current_hp: int,
    playtime_seconds: float,
) -> EventState | None:
    choices = list(shrine_list)
    for event_id in list(special_one_time_event_list):
        if _is_special_event_available(
            event_id,
            dungeon_id=dungeon_id,
            gold=gold,
            deck=deck,
            relic_count=relic_count,
            current_hp=current_hp,
            playtime_seconds=playtime_seconds,
        ):
            choices.append(event_id)
    if not choices:
        return None
    chosen = choices[int(event_rng.random(len(choices) - 1))]
    if chosen in shrine_list:
        shrine_list.remove(chosen)
    if chosen in special_one_time_event_list:
        special_one_time_event_list.remove(chosen)
    return EventState(chosen)


def _choose_event(
    event_rng,
    *,
    event_list: list[str],
    floor: int,
    gold: int,
    relic_ids: set[str],
    current_hp: int,
    max_hp: int,
    current_node_y: int | None,
    map_height: int,
) -> EventState | None:
    candidates = [
        event_id for event_id in event_list
        if _is_event_available(
            event_id,
            floor=floor,
            gold=gold,
            relic_ids=relic_ids,
            current_hp=current_hp,
            max_hp=max_hp,
            current_node_y=current_node_y,
            map_height=map_height,
        )
    ]
    if not candidates:
        return None
    chosen = candidates[int(event_rng.random(len(candidates) - 1))]
    event_list.remove(chosen)
    return EventState(chosen)


def _initialize_event_state(
    event: EventState,
    *,
    randoms: NativeRandomSet,
    ascension_level: int,
    floor: int,
    gold: int,
    deck: list[dict[str, object]],
    relics: list[dict[str, object]],
    potions: list[dict[str, object]],
    max_hp: int,
    relic_drawer=None,
    screenless_relic_drawer=None,
    runtime_card_pools: dict[str, list[str]] | None = None,
    player_class: str = "IRONCLAD",
    note_for_yourself_card_id: str | None = None,
    note_for_yourself_upgrades: int | None = None,
) -> EventState:
    draw_relic = relic_drawer or (lambda tier: draw_random_relic(randoms, tier))
    if event.event_id == "WeMeetAgain":
        potion_index = None
        potion_indexes = [index for index, potion in enumerate(potions) if not _is_potion_slot(potion)]
        if potion_indexes:
            java_shuffle_in_place(potion_indexes, int(randoms.stream("misc").random_long()))
            potion_index = int(potion_indexes[0])
        gold_amount = 0
        if gold >= 50:
            gold_amount = int(randoms.stream("misc").random(50, min(150, gold)))
        non_basic = [
            card for card in deck
            if str(card.get("rarity") or "") != "BASIC" and str(card.get("type") or "") != "CURSE"
        ]
        card_uuid = None
        if non_basic:
            ordered = list(non_basic)
            java_shuffle_in_place(ordered, int(randoms.stream("misc").random_long()))
            card_uuid = str(ordered[0].get("uuid"))
        event.data.update({
            "potion_index": potion_index,
            "gold_amount": gold_amount,
            "card_uuid": card_uuid,
        })
    elif event.event_id == "World of Goop":
        gold_loss = int(randoms.stream("misc").random(35, 75) if ascension_level >= 15 else randoms.stream("misc").random(20, 50))
        event.data.update({
            "gold_loss": min(gold_loss, int(gold)),
        })
    elif event.event_id == "Dead Adventurer":
        rewards = ["GOLD", "NOTHING", "RELIC"]
        java_shuffle_in_place(rewards, int(randoms.stream("misc").random_long()))
        event.data.update({
            "rewards": rewards,
            "encounter_chance": 35 if ascension_level >= 15 else 25,
            "num_rewards": 0,
            "enemy": int(randoms.stream("misc").random(0, 2)),
        })
    elif event.event_id == "Falling":
        for card_type, key_prefix in (("ATTACK", "attack"), ("SKILL", "skill"), ("POWER", "power")):
            candidates = [
                card for card in deck
                if str(card.get("type") or "") == card_type and not _is_bottled_card(card)
            ]
            if candidates:
                chosen = candidates[int(randoms.stream("misc").random(0, len(candidates) - 1))]
                event.data[f"{key_prefix}_uuid"] = str(chosen.get("uuid") or "")
                event.data[f"{key_prefix}_name"] = str(chosen.get("name") or chosen.get("card_id") or key_prefix.title())
    elif event.event_id == "Knowing Skull":
        event.screen = "INTRO_1"
        event.data.update({
            "potion_cost": 6,
            "gold_cost": 6,
            "card_cost": 6,
            "leave_cost": 6,
        })
    elif event.event_id == "Designer":
        if ascension_level >= 15:
            adjust_cost, clean_cost, full_cost, hp_loss = 50, 75, 110, 5
        else:
            adjust_cost, clean_cost, full_cost, hp_loss = 40, 60, 90, 3
        event.data.update({
            "adjustment_upgrades_one": bool(randoms.stream("misc").random_boolean()),
            "clean_up_removes_cards": bool(randoms.stream("misc").random_boolean()),
            "adjust_cost": adjust_cost,
            "clean_cost": clean_cost,
            "full_cost": full_cost,
            "hp_loss": hp_loss,
        })
    elif event.event_id == "Forgotten Altar":
        hp_loss = _mathutils_round_positive(max_hp * (0.35 if ascension_level >= 15 else 0.25))
        event.data.update({"hp_loss": hp_loss})
    elif event.event_id == "The Library":
        heal_amt = _mathutils_round_positive(max_hp * (0.2 if ascension_level >= 15 else 0.33))
        event.data.update({"heal_amt": heal_amt})
    elif event.event_id == "Masked Bandits":
        # The real event constructor prebuilds the encounter before the player
        # chooses Pay/Fight. Reuse those monsters later so monsterHpRng is not
        # consumed a second time when the Fight option is selected.
        event.data["prebuilt_monsters"] = build_encounter("Masked Bandits", randoms, ascension_level)
    elif event.event_id == "Ghosts":
        hp_loss = max(1, min(max_hp - 1, int(-(-max_hp // 2))))
        event.data.update({"hp_loss": hp_loss})
    elif event.event_id == "Vampires":
        max_hp_loss = max(1, min(max_hp - 1, int(-(-((max_hp * 3) / 10)))))
        event.data.update({
            "max_hp_loss": max_hp_loss,
            "has_vial": "Blood Vial" in _relic_ids(relics),
        })
    elif event.event_id == "Nest":
        event.data.update({
            "gold_gain": 50 if ascension_level >= 15 else 99,
        })
    elif event.event_id == "N'loth":
        relic_choices = [dict(relic) for relic in relics]
        java_shuffle_in_place(relic_choices, int(randoms.stream("misc").random_long()))
        if relic_choices:
            event.data["choice1_id"] = str(relic_choices[0].get("relic_id") or relic_choices[0].get("id") or "")
            event.data["choice1_name"] = str(relic_choices[0].get("name") or event.data["choice1_id"])
        if len(relic_choices) > 1:
            event.data["choice2_id"] = str(relic_choices[1].get("relic_id") or relic_choices[1].get("id") or "")
            event.data["choice2_name"] = str(relic_choices[1].get("name") or event.data["choice2_id"])
    elif event.event_id == "The Joust":
        event.screen = "HALT"
    elif event.event_id == "The Moai Head":
        hp_amt = _mathutils_round_positive(max_hp * (0.18 if ascension_level >= 15 else 0.125))
        event.data.update({"hp_amt": hp_amt})
    elif event.event_id == "Winding Halls":
        event.data.update({
            "hp_amt": _mathutils_round_positive(max_hp * (0.18 if ascension_level >= 15 else 0.125)),
            "heal_amt": _mathutils_round_positive(max_hp * (0.2 if ascension_level >= 15 else 0.25)),
            "max_hp_amt": _mathutils_round_positive(max_hp * 0.05),
        })
    elif event.event_id == "MindBloom":
        event.data.update({
            "late_branch": int(floor) % 50 > 40,
        })
    elif event.event_id == "Cursed Tome":
        event.data.update({
            "final_dmg": 15 if ascension_level >= 15 else 10,
            "damage_taken": 0,
        })
    elif event.event_id == "Match and Keep!":
        event.data.update({
            "cards": _build_match_and_keep_cards(
                randoms,
                ascension_level=ascension_level,
                runtime_card_pools=runtime_card_pools,
                player_class=player_class,
                relics=relics,
            ),
            "attempt_count": 5,
            "matched_cards": [],
        })
    elif event.event_id == "NoteForYourself":
        defaults = note_for_yourself_defaults()
        event.data.update({
            "obtain_card": make_card(
                str(note_for_yourself_card_id or defaults.default_card_id),
                upgrades=int(defaults.default_upgrades if note_for_yourself_upgrades is None else note_for_yourself_upgrades),
            ),
        })
    return event


def generate_event_for_act(
    randoms: NativeRandomSet,
    *,
    ascension_level: int,
    act: int,
    floor: int,
    gold: int,
    relics: list[dict[str, object]],
    deck: list[dict[str, object]],
    potions: list[dict[str, object]],
    current_hp: int,
    max_hp: int,
    current_node_y: int | None,
    map_height: int,
    event_list: list[str],
    shrine_list: list[str],
    special_one_time_event_list: list[str],
    playtime_seconds: float = 0.0,
    relic_drawer=None,
    screenless_relic_drawer=None,
    runtime_card_pools: dict[str, list[str]] | None = None,
    player_class: str = "IRONCLAD",
    note_for_yourself_card_id: str | None = None,
    note_for_yourself_upgrades: int | None = None,
) -> EventState:
    dungeon_id = dungeon_id_for_act(act)
    relic_ids = _relic_ids(relics)
    event_rng = randoms.duplicate_stream("event", alias="event_duplicate")
    if float(event_rng.random(1.0)) < SHRINE_CHANCE:
        shrine = _choose_shrine(
            event_rng,
            dungeon_id=dungeon_id,
            shrine_list=shrine_list,
            special_one_time_event_list=special_one_time_event_list,
            gold=gold,
            deck=deck,
            relic_count=len(relics),
            current_hp=current_hp,
            playtime_seconds=playtime_seconds,
        )
        if shrine is not None:
            return _initialize_event_state(
                shrine,
                randoms=randoms,
                ascension_level=ascension_level,
                floor=floor,
                gold=gold,
                deck=deck,
                relics=relics,
                potions=potions,
                max_hp=max_hp,
                relic_drawer=relic_drawer,
                screenless_relic_drawer=screenless_relic_drawer,
                runtime_card_pools=runtime_card_pools,
                player_class=player_class,
                note_for_yourself_card_id=note_for_yourself_card_id,
                note_for_yourself_upgrades=note_for_yourself_upgrades,
            )
        event = _choose_event(
            event_rng,
            event_list=event_list,
            floor=floor,
            gold=gold,
            relic_ids=relic_ids,
            current_hp=current_hp,
            max_hp=max_hp,
            current_node_y=current_node_y,
            map_height=map_height,
        )
        if event is not None:
            return _initialize_event_state(
                event,
                randoms=randoms,
                ascension_level=ascension_level,
                floor=floor,
                gold=gold,
                deck=deck,
                relics=relics,
                potions=potions,
                max_hp=max_hp,
                relic_drawer=relic_drawer,
                screenless_relic_drawer=screenless_relic_drawer,
                runtime_card_pools=runtime_card_pools,
                player_class=player_class,
                note_for_yourself_card_id=note_for_yourself_card_id,
                note_for_yourself_upgrades=note_for_yourself_upgrades,
            )
        raise NotImplementedError("native_sim_v3 could not generate an event or shrine for the current floor.")
    event = _choose_event(
        event_rng,
        event_list=event_list,
        floor=floor,
        gold=gold,
        relic_ids=relic_ids,
        current_hp=current_hp,
        max_hp=max_hp,
        current_node_y=current_node_y,
        map_height=map_height,
    )
    if event is not None:
        return _initialize_event_state(
            event,
            randoms=randoms,
            ascension_level=ascension_level,
            floor=floor,
            gold=gold,
            deck=deck,
            relics=relics,
            potions=potions,
            max_hp=max_hp,
            relic_drawer=relic_drawer,
            screenless_relic_drawer=screenless_relic_drawer,
            runtime_card_pools=runtime_card_pools,
            player_class=player_class,
            note_for_yourself_card_id=note_for_yourself_card_id,
            note_for_yourself_upgrades=note_for_yourself_upgrades,
        )
    shrine = _choose_shrine(
        event_rng,
        dungeon_id=dungeon_id,
        shrine_list=shrine_list,
        special_one_time_event_list=special_one_time_event_list,
        gold=gold,
        deck=deck,
        relic_count=len(relics),
        current_hp=current_hp,
        playtime_seconds=playtime_seconds,
    )
    if shrine is not None:
        return _initialize_event_state(
            shrine,
            randoms=randoms,
            ascension_level=ascension_level,
            floor=floor,
            gold=gold,
            deck=deck,
            relics=relics,
            potions=potions,
            max_hp=max_hp,
            relic_drawer=relic_drawer,
            screenless_relic_drawer=screenless_relic_drawer,
            runtime_card_pools=runtime_card_pools,
            player_class=player_class,
            note_for_yourself_card_id=note_for_yourself_card_id,
            note_for_yourself_upgrades=note_for_yourself_upgrades,
        )
    raise NotImplementedError("native_sim_v3 could not generate an event or shrine for the current floor.")


def resolve_event_choice(
    event: EventState,
    *,
    action_index: int,
    randoms: NativeRandomSet,
    ascension_level: int,
    dungeon_id: str | None = None,
    current_hp: int,
    max_hp: int,
    gold: int,
    deck: list[dict[str, object]] | None = None,
    relics: list[dict[str, object]] | None = None,
    potions: list[dict[str, object]] | None = None,
    relic_drawer=None,
    screenless_relic_drawer=None,
    runtime_card_pools: dict[str, list[str]] | None = None,
    player_class: str = "IRONCLAD",
    final_act_available: bool = False,
    has_ruby_key: bool = False,
    has_emerald_key: bool = False,
    has_sapphire_key: bool = False,
) -> dict[str, object]:
    draw_relic = relic_drawer or (lambda tier: draw_random_relic(randoms, tier))
    draw_screenless_relic = screenless_relic_drawer or (
        lambda tier=None, exclude=None: draw_random_screenless_relic(randoms, exclude=exclude)
    )
    current_dungeon_id = str(dungeon_id or "Exordium")
    if event.event_id == "FaceTrader":
        if event.screen == "INTRO":
            event.screen = "MAIN"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "MAIN":
            event.screen = "RESULT"
            if action_index == 0:
                damage = max(1, max_hp // 10)
                gold_reward = 50 if ascension_level >= 15 else 75
                return {"gold": gold + gold_reward, "hp": _apply_event_hp_loss(current_hp, damage, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
            if action_index == 1:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [_random_face_relic(randoms, relic_ids=_relic_ids(relics or []))],
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "World of Goop":
        if event.screen == "INTRO":
            event.screen = "RESULT"
            if action_index == 0:
                return {"gold": gold + 75, "hp": _apply_event_hp_loss(current_hp, 11, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
            return {
                "gold": max(0, gold - int(event.data.get("gold_loss") or 0)),
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Mushrooms":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "FIGHT"
                event.data["prebuilt_monsters"] = build_encounter("The Mushroom Lair", randoms, ascension_level)
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "RESULT"
            heal_amt = int(max_hp * 0.25)
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": [make_card("Parasite")],
                "heal_player": heal_amt,
            }
        if event.screen == "FIGHT":
            event.screen = "RESULT"
            gold_reward = int(randoms.stream("misc").random(20, 30))
            reward_relic = make_relic("Circlet") if any(str(relic.get("relic_id") or relic.get("id")) == "Odd Mushroom" for relic in (relics or [])) else make_relic("Odd Mushroom")
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_combat": True,
                "encounter_name": "The Mushroom Lair",
                "prebuilt_monsters": list(event.data.get("prebuilt_monsters") or []),
                "event_rewards": {
                    "gold": gold_reward,
                    "relics": [reward_relic],
                    "potions": [],
                    "cards": [],
                    "card_groups": [],
                },
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Golden Shrine":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0:
                gold_amt = 50 if ascension_level >= 15 else 100
                return {"gold": gold + gold_amt, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            if action_index == 1:
                return {
                    "gold": gold + 275,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Regret")],
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Match and Keep!":
        if event.screen == "INTRO":
            event.screen = "RULE_EXPLANATION"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "RULE_EXPLANATION":
            event.screen = "PLAY"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PLAY":
            pickable = _match_and_keep_pickable_indexes(event)
            if not 0 <= int(action_index) < len(pickable):
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            card_index = int(pickable[int(action_index)])
            cards = list(event.data.get("cards") or [])
            if not 0 <= card_index < len(cards):
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            revealed = {int(index) for index in list(event.data.get("revealed_card_indexes") or [])}
            revealed.add(card_index)
            event.data["revealed_card_indexes"] = sorted(revealed)
            first_index = event.data.get("first_card_index")
            if first_index is None:
                event.data["first_card_index"] = card_index
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            first_index = int(first_index)
            event.data["first_card_index"] = None
            attempts = max(0, int(event.data.get("attempt_count") or 0) - 1)
            event.data["attempt_count"] = attempts
            add_cards: list[dict[str, object]] = []
            if 0 <= first_index < len(cards):
                first_card = dict(cards[first_index])
                second_card = dict(cards[card_index])
                if str(first_card.get("card_id")) == str(second_card.get("card_id")):
                    removed = {int(index) for index in list(event.data.get("removed_card_indexes") or [])}
                    removed.update({first_index, card_index})
                    event.data["removed_card_indexes"] = sorted(removed)
                    matched_ids = [str(card_id) for card_id in list(event.data.get("matched_cards") or [])]
                    matched_ids.append(str(first_card.get("card_id")))
                    event.data["matched_cards"] = matched_ids
                    add_cards.append(first_card)
            if attempts <= 0 or not _match_and_keep_pickable_indexes(event):
                event.screen = "COMPLETE"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": add_cards,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Dead Adventurer":
        if event.screen == "INTRO":
            if action_index == 0:
                threshold = int(event.data.get("encounter_chance") or (35 if ascension_level >= 15 else 25))
                if int(randoms.stream("misc").random(0, 99)) < threshold:
                    event.screen = "FAIL"
                    event.data["event_gold"] = int(randoms.stream("misc").random(25, 35))
                    event.data["prebuilt_monsters"] = build_encounter(
                        _dead_adventurer_encounter_name(int(event.data.get("enemy") or 0)),
                        randoms,
                        ascension_level,
                    )
                    return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
                event.data["num_rewards"] = int(event.data.get("num_rewards") or 0) + 1
                event.data["encounter_chance"] = threshold + 25
                rewards = list(event.data.get("rewards") or ["NOTHING"])
                reward = rewards.pop(0) if rewards else "NOTHING"
                event.data["rewards"] = rewards
                if reward == "GOLD":
                    result = {"gold": gold + 30, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
                elif reward == "RELIC":
                    result = {
                        "gold": gold,
                        "hp": current_hp,
                        "max_hp": max_hp,
                        "potions": [],
                        "open_rewards": False,
                        "add_relics": [draw_screenless_relic(roll_random_relic_tier(randoms))],
                    }
                else:
                    result = {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
                if int(event.data["num_rewards"]) >= 3:
                    event.screen = "SUCCESS"
                else:
                    event.screen = "INTRO"
                return result
            event.screen = "ESCAPE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "FAIL":
            rewards = list(event.data.get("rewards") or [])
            if event.data.get("event_gold") is not None:
                event_gold = int(event.data.get("event_gold") or 0)
            else:
                event_gold = int(randoms.stream("misc").random(25, 35))
            reward_gold = event_gold + 30 * sum(1 for reward in rewards if reward == "GOLD")
            reward_relics = [
                draw_relic(roll_random_relic_tier(randoms))
                for reward in rewards
                if reward == "RELIC"
            ]
            event.screen = "COMPLETE"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_combat": True,
                "encounter_name": _dead_adventurer_encounter_name(int(event.data.get("enemy") or 0)),
                "prebuilt_monsters": event.data.get("prebuilt_monsters"),
                "elite_trigger": True,
                "event_rewards": {
                    "gold": reward_gold,
                    "relics": reward_relics,
                    "potions": [],
                    "cards": [],
                    "card_groups": [],
                },
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Falling":
        if event.screen == "INTRO":
            event.screen = "CHOICE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "CHOICE":
            event.screen = "RESULT"
            if not any(event.data.get(key) for key in ("skill_uuid", "power_uuid", "attack_uuid")):
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            remove_uuid = None
            if action_index == 0:
                remove_uuid = event.data.get("skill_uuid")
            elif action_index == 1:
                remove_uuid = event.data.get("power_uuid")
            elif action_index == 2:
                remove_uuid = event.data.get("attack_uuid")
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "remove_card_uuid": str(remove_uuid) if remove_uuid else None,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Fountain of Cleansing":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0:
                removable = [
                    str(card.get("uuid"))
                    for card in (deck or [])
                    if str(card.get("type") or "") == "CURSE"
                    and str(card.get("card_id") or "") not in {"AscendersBane", "CurseOfTheBell", "Necronomicurse"}
                ]
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "remove_card_uuids": removable,
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Knowing Skull":
        if event.screen == "INTRO_1":
            event.screen = "ASK"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "stay_event": True}
        if event.screen == "ASK":
            if action_index == 0:
                cost = int(event.data.get("potion_cost") or 6)
                event.data["potion_cost"] = cost + 1
                result = {
                    "gold": gold,
                    "hp": _apply_event_hp_loss(current_hp, cost, relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "stay_event": True,
                }
                if "Sozu" not in _relic_ids(relics or []):
                    result["add_potions"] = [_random_potion(randoms)]
                return result
            if action_index == 1:
                cost = int(event.data.get("gold_cost") or 6)
                event.data["gold_cost"] = cost + 1
                return {
                    "gold": gold + 90,
                    "hp": _apply_event_hp_loss(current_hp, cost, relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "stay_event": True,
                }
            if action_index == 2:
                cost = int(event.data.get("card_cost") or 6)
                event.data["card_cost"] = cost + 1
                return {
                    "gold": gold,
                    "hp": _apply_event_hp_loss(current_hp, cost, relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "stay_event": True,
                    "add_cards": [_random_colorless_card(randoms, "COLORLESS_UNCOMMON", runtime_card_pools=runtime_card_pools)],
                }
            event.screen = "COMPLETE"
            leave_cost = int(event.data.get("leave_cost") or 6)
            return {
                "gold": gold,
                "hp": _apply_event_hp_loss(current_hp, leave_cost, relics),
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "stay_event": True,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Back to Basics":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "purge",
                    "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "requires_confirm": False,
                }
            upgrade_indexes = [
                index for index, card in enumerate(deck or [])
                if {"STARTER_STRIKE", "STARTER_DEFEND"} & set(card.get("tags") or [])
                and can_upgrade_card(card)
            ]
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "upgrade_indexes": upgrade_indexes,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Designer":
        if event.screen == "INTRO":
            event.screen = "MAIN"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "MAIN":
            event.screen = "DONE"
            if action_index == 0:
                adjust_cost = int(event.data["adjust_cost"])
                if bool(event.data.get("adjustment_upgrades_one")):
                    return {
                        "gold": max(0, gold - adjust_cost),
                        "hp": current_hp,
                        "max_hp": max_hp,
                        "potions": [],
                        "open_rewards": False,
                        "open_card_select": True,
                        "card_select_mode": "upgrade",
                        "candidate_indexes": _upgradable_candidate_indexes(deck or []),
                        "return_phase": "EVENT",
                        "clear_event_on_finish": False,
                    }
                upgrade_indexes = _pick_random_upgradable_indexes(randoms, deck or [], count=2)
                return {
                    "gold": max(0, gold - adjust_cost),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "upgrade_indexes": upgrade_indexes,
                }
            if action_index == 1:
                clean_cost = int(event.data["clean_cost"])
                if bool(event.data.get("clean_up_removes_cards")):
                    return {
                        "gold": max(0, gold - clean_cost),
                        "hp": current_hp,
                        "max_hp": max_hp,
                        "potions": [],
                        "open_rewards": False,
                        "open_card_select": True,
                        "card_select_mode": "purge",
                        "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                        "return_phase": "EVENT",
                        "clear_event_on_finish": False,
                    }
                return {
                    "gold": max(0, gold - clean_cost),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "transform",
                    "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "remaining_picks": 2,
                }
            if action_index == 2:
                return {
                    "gold": max(0, gold - int(event.data["full_cost"])),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "remove",
                    "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "card_select_effect": "designer_remove_and_upgrade",
                }
            return {
                "gold": gold,
                "hp": _apply_event_hp_loss(current_hp, int(event.data["hp_loss"]), relics),
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Forgotten Altar":
        if event.screen == "INTRO":
            event.screen = "RESULT"
            if action_index == 0:
                relic_ids = _relic_ids(relics or [])
                reward_relic = make_relic("Circlet") if "Bloody Idol" in relic_ids else make_relic("Bloody Idol")
                result = {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [reward_relic],
                }
                if "Bloody Idol" not in relic_ids:
                    result["remove_relic_id"] = "Golden Idol"
                return result
            if action_index == 1:
                new_max_hp = max_hp + 5
                hp_after_max_gain = min(new_max_hp, current_hp + 5)
                return {
                    "gold": gold,
                    "hp": _apply_event_hp_loss(hp_after_max_gain, int(event.data["hp_loss"]), relics),
                    "max_hp": new_max_hp,
                    "potions": [],
                    "open_rewards": False,
                }
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": [make_card("Decay")],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Cursed Tome":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "PAGE_1"
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "END"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PAGE_1":
            event.screen = "PAGE_2"
            event.data["damage_taken"] = int(event.data.get("damage_taken") or 0) + 1
            return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, 1, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PAGE_2":
            event.screen = "PAGE_3"
            event.data["damage_taken"] = int(event.data.get("damage_taken") or 0) + 2
            return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, 2, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PAGE_3":
            event.screen = "LAST_PAGE"
            event.data["damage_taken"] = int(event.data.get("damage_taken") or 0) + 3
            return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, 3, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "LAST_PAGE":
            event.screen = "END"
            if action_index == 0:
                books = [relic_id for relic_id in ("Necronomicon", "Enchiridion", "Nilry's Codex") if relic_id not in _relic_ids(relics or [])]
                if not books:
                    books = ["Circlet"]
                reward_relic = make_relic(books[int(randoms.stream("misc").random(0, len(books) - 1))])
                return {
                    "gold": gold,
                    "hp": _apply_event_hp_loss(current_hp, int(event.data["final_dmg"]), relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": True,
                    "relic_rewards": [reward_relic],
                }
            return {
                "gold": gold,
                "hp": _apply_event_hp_loss(current_hp, 3, relics),
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Duplicator":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "COMPLETE"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "duplicate",
                    "candidate_indexes": list(range(len(deck or []))),
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "requires_confirm": False,
                }
            event.screen = "COMPLETE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Beggar":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "GAVE_MONEY"
                return {
                    "gold": max(0, gold - 75),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                }
            event.screen = "LEAVE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "GAVE_MONEY":
            event.screen = "LEAVE"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_card_select": True,
                "card_select_mode": "purge",
                "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                "return_phase": "MAP",
                "clear_event_on_finish": True,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Addict":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0 and gold >= 85:
                reward_relic = draw_screenless_relic(exclude=_relic_ids(relics or []))
                return {
                    "gold": max(0, gold - 85),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [reward_relic],
                }
            if action_index == 1:
                reward_relic = draw_screenless_relic(exclude=_relic_ids(relics or []))
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [reward_relic],
                    "add_cards": [make_card("Shame")],
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Drug Dealer":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("J.A.X.")],
                }
            if action_index == 1:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "transform",
                    "candidate_indexes": _raw_purgeable_candidate_indexes(deck or []),
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "remaining_picks": 2,
                }
            reward_relic = make_relic("Circlet") if "MutagenicStrength" in _relic_ids(relics or []) else make_relic("MutagenicStrength")
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_relics": [reward_relic],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "The Library":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0:
                library_randomizer = (
                    int(event.data["card_blizz_randomizer"])
                    if "card_blizz_randomizer" in event.data
                    else 5
                )
                library_cards, next_randomizer = _generate_library_cards(
                    randoms,
                    card_blizz_randomizer=library_randomizer,
                    owned_relic_ids=_relic_ids(relics or []),
                    runtime_card_pools=runtime_card_pools,
                )
                event.data["card_blizz_randomizer"] = next_randomizer
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "library",
                    "card_select_cards": list(reversed(library_cards)),
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "requires_confirm": False,
                    "card_blizz_randomizer": next_randomizer,
                }
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "heal_player": int(event.data.get("heal_amt") or 0),
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Masked Bandits":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "PAID_1"
                return {
                    "gold": 0,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                }
            event.screen = "END"
            reward_relic = make_relic("Circlet") if "Red Mask" in _relic_ids(relics or []) else make_relic("Red Mask")
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_combat": True,
                "encounter_name": "Masked Bandits",
                "prebuilt_monsters": list(event.data.get("prebuilt_monsters") or []),
                "event_rewards": {
                    "gold": int(randoms.stream("misc").random(25, 35)),
                    "relics": [reward_relic],
                    "potions": [],
                    "cards": [],
                    "card_groups": [],
                },
            }
        if event.screen == "PAID_1":
            event.screen = "PAID_2"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PAID_2":
            event.screen = "PAID_3"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Nest":
        if event.screen == "INTRO":
            event.screen = "CHOOSE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "CHOOSE":
            event.screen = "COMPLETE"
            if action_index == 0:
                return {
                    "gold": gold + int(event.data.get("gold_gain") or 99),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                }
            return {
                "gold": gold,
                "hp": _apply_event_hp_loss(current_hp, 6, relics),
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": [make_card("RitualDagger")],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "N'loth":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index in {0, 1}:
                choice_id = str(event.data.get(f"choice{action_index + 1}_id") or "")
                has_nloths_gift = "Nloth's Gift" in _relic_ids(relics or [])
                gift_relic = "Circlet" if has_nloths_gift else "Nloth's Gift"
                result = {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [make_relic(gift_relic)],
                }
                if choice_id and not has_nloths_gift:
                    result["remove_relic_id"] = choice_id
                return result
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "The Joust":
        if event.screen == "HALT":
            event.screen = "EXPLANATION"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "EXPLANATION":
            event.data["bet_for"] = action_index == 1
            event.screen = "PRE_JOUST"
            return {"gold": gold - 50, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PRE_JOUST":
            event.data["owner_wins"] = bool(randoms.stream("misc").random_boolean(0.3))
            event.screen = "JOUST"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "JOUST":
            event.screen = "COMPLETE"
            owner_wins = bool(event.data.get("owner_wins"))
            bet_for = bool(event.data.get("bet_for"))
            gold_gain = 0
            if owner_wins and bet_for:
                gold_gain = 250
            elif (not owner_wins) and (not bet_for):
                gold_gain = 100
            return {"gold": gold + gold_gain, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "The Moai Head":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            hp_amt = int(event.data.get("hp_amt") or 0)
            if action_index == 0:
                new_max_hp = max(1, max_hp - hp_amt)
                return {
                    "gold": gold,
                    "hp": min(current_hp, new_max_hp),
                    "max_hp": new_max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "heal_player": new_max_hp,
                }
            if action_index == 1 and "Golden Idol" in _relic_ids(relics or []):
                return {
                    "gold": gold + 333,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "remove_relic_id": "Golden Idol",
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Tomb of Lord Red Mask":
        if event.screen == "INTRO":
            event.screen = "RESULT"
            if action_index != 0:
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
            if "Red Mask" in _relic_ids(relics or []):
                return {"gold": gold + 222, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            return {
                "gold": 0,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_relics": [make_relic("Red Mask")],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Winding Halls":
        if event.screen == "INTRO":
            event.screen = "CHOICE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "CHOICE":
            event.screen = "COMPLETE"
            if action_index == 0:
                return {
                    "gold": gold,
                    "hp": _apply_event_hp_loss(current_hp, int(event.data.get("hp_amt") or 0), relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Madness"), make_card("Madness")],
                }
            if action_index == 1:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Writhe")],
                    "heal_player": int(event.data.get("heal_amt") or 0),
                }
            new_max_hp = max(1, max_hp - int(event.data.get("max_hp_amt") or 0))
            return {
                "gold": gold,
                "hp": min(current_hp, new_max_hp),
                "max_hp": new_max_hp,
                "potions": [],
                "open_rewards": False,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "MindBloom":
        if event.screen == "INTRO":
            late_branch = bool(event.data.get("late_branch"))
            if action_index == 0:
                event.screen = "FIGHT"
                boss_choices = ["The Guardian", "Hexaghost", "Slime Boss"]
                java_shuffle_in_place(boss_choices, int(randoms.stream("misc").random_long()))
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_combat": True,
                    "encounter_name": str(boss_choices[0]),
                    "event_rewards": {
                        "gold": 25 if ascension_level >= 13 else 50,
                        "relics": [draw_relic("RARE")],
                        "potions": [],
                        "cards": [],
                        "card_groups": [],
                    },
                }
            event.screen = "LEAVE"
            if action_index == 1:
                upgrade_indexes = [
                    index for index, card in enumerate(deck or [])
                    if int(card.get("upgrades") or 0) <= 0 and str(card.get("type") or "") not in {"STATUS", "CURSE"}
                ]
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "upgrade_indexes": upgrade_indexes,
                    "add_relics": [make_relic("Mark of the Bloom")],
                }
            if not late_branch:
                return {
                    "gold": gold + 999,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Normality"), make_card("Normality")],
                }
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": [make_card("Doubt")],
                "heal_player": max_hp,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Ghosts":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "COMPLETE"
                apparition_count = 3 if ascension_level >= 15 else 5
                hp_loss = int(event.data.get("hp_loss") or 0)
                return {
                    "gold": gold,
                    "hp": min(max_hp - hp_loss, current_hp),
                    "max_hp": max(1, max_hp - hp_loss),
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Ghostly") for _ in range(apparition_count)],
                }
            event.screen = "LEAVE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Vampires":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0:
                max_hp_loss = int(event.data.get("max_hp_loss") or 0)
                new_max_hp = max(1, max_hp - max_hp_loss)
                new_current_hp = min(new_max_hp, current_hp)
                return {
                    "gold": gold,
                    "hp": new_current_hp,
                    "max_hp": new_max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "remove_starter_strikes": True,
                    "add_cards": [make_card("Bite") for _ in range(5)],
                }
            if bool(event.data.get("has_vial")) and action_index == 1:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "remove_relic_id": "Blood Vial",
                    "remove_starter_strikes": True,
                    "add_cards": [make_card("Bite") for _ in range(5)],
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "NoteForYourself":
        if event.screen == "INTRO":
            event.screen = "CHOOSE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "CHOOSE":
            event.screen = "COMPLETE"
            if action_index == 0:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [dict(event.data.get("obtain_card") or make_card("Iron Wave"))],
                    "open_card_select": True,
                    "card_select_mode": "purge",
                    "candidate_indexes": None,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                    "visible_for_purge": False,
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Scrap Ooze":
        event.data.setdefault("relic_chance", 25)
        event.data.setdefault("dmg", 5 if ascension_level >= 15 else 3)
        event.data.setdefault("total_damage", 0)
        if event.screen == "INTRO":
            if action_index == 0:
                dmg = int(event.data["dmg"])
                event.data["total_damage"] = int(event.data["total_damage"]) + dmg
                success = int(randoms.stream("misc").random(0, 99)) >= 99 - int(event.data["relic_chance"])
                if success:
                    event.screen = "RESULT"
                    return {
                        "gold": gold,
                    "hp": _apply_event_hp_loss(current_hp, dmg, relics),
                        "max_hp": max_hp,
                        "potions": [],
                        "open_rewards": False,
                        "add_relics": [draw_screenless_relic()],
                    }
                event.data["relic_chance"] = int(event.data["relic_chance"]) + 10
                event.data["dmg"] = int(event.data["dmg"]) + 1
                return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, dmg, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "RESULT"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Shining Light":
        if event.screen == "INTRO":
            upgradable = list(_upgradable_candidate_indexes(deck or []))
            if action_index == 0 and upgradable:
                event.screen = "COMPLETE"
                damage = _mathutils_round_positive(max_hp * (0.3 if ascension_level >= 15 else 0.2))
                ordered = list(upgradable)
                java_shuffle_in_place(ordered, int(randoms.stream("misc").random_long()))
                upgrade_indexes = [ordered[0]]
                if len(ordered) > 1:
                    upgrade_indexes.append(ordered[1])
                return {
                    "gold": gold,
                    "hp": _apply_event_normal_damage(current_hp, damage, relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "upgrade_indexes": upgrade_indexes,
                }
            event.screen = "COMPLETE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Golden Wing":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "PURGE"
                return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, 7, relics), "max_hp": max_hp, "potions": [], "open_rewards": False}
            if action_index == 1 and _has_card_with_x_damage(deck or [], 10):
                event.screen = "MAP"
                gold_gain = int(randoms.stream("misc").random(50, 80))
                return {"gold": gold + gold_gain, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "MAP"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PURGE":
            event.screen = "MAP"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_card_select": True,
                "card_select_mode": "purge",
                "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                "return_phase": "EVENT",
                "clear_event_on_finish": False,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Liars Game":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "AGREE"
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "DISAGREE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "AGREE":
            event.screen = "COMPLETE"
            gold_reward = 150 if ascension_level >= 15 else 175
            return {
                "gold": gold + gold_reward,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": [make_card("Doubt")],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "WeMeetAgain":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            if action_index == 0 and event.data.get("potion_index") is not None:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "remove_potion_index": int(event.data["potion_index"]),
                    "add_relics": [draw_screenless_relic()],
                }
            if action_index == 1 and int(event.data.get("gold_amount") or 0) > 0:
                return {
                    "gold": gold - int(event.data["gold_amount"]),
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [draw_screenless_relic()],
                }
            if action_index == 2 and event.data.get("card_uuid") is not None:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "remove_card_uuid": str(event.data["card_uuid"]),
                    "add_relics": [draw_screenless_relic()],
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Lab":
        if event.screen == "INTRO":
            event.screen = "COMPLETE"
            potion_count = 2 if ascension_level >= 15 else 3
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [_random_potion(randoms) for _ in range(potion_count)],
                "open_rewards": True,
                "leave_event_after_rewards": True,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Bonfire Elementals":
        if event.screen == "INTRO":
            event.screen = "CHOOSE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "CHOOSE":
            event.screen = "COMPLETE"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_card_select": True,
                "card_select_mode": "purge",
                "candidate_indexes": _purgeable_candidate_indexes(deck or []),
                "return_phase": "EVENT",
                "clear_event_on_finish": False,
                "card_select_effect": "bonfire",
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Wheel of Change":
        if event.screen == "INTRO":
            result = int(randoms.stream("misc").random(0, 5))
            event.result = result
            event.screen = "SPIN"
            return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False}
        if event.screen == "SPIN":
            result = int(event.result if event.result is not None else randoms.stream("misc").random(0, 5))
            event.result = result
            event.screen = "RESULT"
            if result == 0:
                gold_amount = 100
                if current_dungeon_id == "TheCity":
                    gold_amount = 200
                elif current_dungeon_id == "TheBeyond":
                    gold_amount = 300
                return {"gold": gold + gold_amount, "hp": current_hp, "potions": [], "open_rewards": False}
            return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False}
        if event.screen == "RESULT":
            result = int(event.result if event.result is not None else randoms.stream("misc").random(0, 5))
            event.result = result
            event.screen = "LEAVE"
            if result == 0:
                return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False}
            if result == 1:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "potions": [],
                    "open_rewards": True,
                    "relic_rewards": [draw_screenless_relic()],
                }
            if result == 2:
                return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False, "heal_player": max_hp}
            if result == 3:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Decay")],
                }
            if result == 4:
                candidate_indexes = _purgeable_candidate_indexes(deck or [])
                if candidate_indexes:
                    return {
                        "gold": gold,
                        "hp": current_hp,
                        "potions": [],
                        "open_rewards": False,
                        "open_card_select": True,
                        "card_select_mode": "purge",
                        "candidate_indexes": candidate_indexes,
                        "return_phase": "EVENT",
                        "clear_event_on_finish": False,
                    }
                return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False}
            if result == 5:
                return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, max(1, int(max_hp * 0.1)), relics), "potions": [], "open_rewards": False}
            raise NotImplementedError(
                f"native_sim_v3 Wheel of Change result {result} is not implemented yet."
            )
        return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Big Fish":
        if event.screen == "INTRO":
            event.screen = "RESULT"
            if action_index == 0:
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "heal_player": max_hp // 3}
            if action_index == 1:
                new_max_hp = max_hp + 5
                return {"gold": gold, "hp": current_hp, "max_hp": new_max_hp, "potions": [], "open_rewards": False, "heal_player": 5}
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "add_cards": [make_card("Regret")],
                "add_relics": [draw_screenless_relic()],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "The Cleric":
        purgeable = _purgeable_candidate_indexes(deck or [])
        purify_cost = 75 if ascension_level >= 15 else 50
        if event.screen == "INTRO":
            if action_index == 0 and gold >= 35:
                event.screen = "RESULT"
                return {
                    "gold": gold - 35,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "heal_player": int(max_hp * 0.25),
                }
            if action_index == 1 and gold >= purify_cost and purgeable:
                event.screen = "RESULT"
                return {
                    "gold": gold - purify_cost,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "purge",
                    "candidate_indexes": purgeable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            event.screen = "RESULT"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Living Wall":
        purgeable = _purgeable_candidate_indexes(deck or [])
        upgradable = _upgradable_candidate_indexes(deck or [])
        if event.screen == "INTRO":
            event.screen = "RESULT"
            if action_index == 0 and purgeable:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "remove",
                    "candidate_indexes": purgeable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            if action_index == 1 and purgeable:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "transform",
                    "candidate_indexes": purgeable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            if action_index == 2 and upgradable:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "upgrade",
                    "candidate_indexes": upgradable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "The Woman in Blue":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "RESULT"
                event.rewards_opened = True
                return {"gold": max(0, gold - 20), "hp": current_hp, "potions": [_random_potion(randoms, player_class=player_class)], "open_rewards": True}
            if action_index == 1:
                event.screen = "RESULT"
                event.rewards_opened = True
                return {"gold": max(0, gold - 30), "hp": current_hp, "potions": [_random_potion(randoms, player_class=player_class), _random_potion(randoms, player_class=player_class)], "open_rewards": True}
            if action_index == 2:
                event.screen = "RESULT"
                event.rewards_opened = True
                return {"gold": max(0, gold - 40), "hp": current_hp, "potions": [_random_potion(randoms, player_class=player_class), _random_potion(randoms, player_class=player_class), _random_potion(randoms, player_class=player_class)], "open_rewards": True}
            hp_loss = max(1, (max_hp + 19) // 20) if ascension_level >= 15 else 0
            event.screen = "RESULT"
            return {"gold": gold, "hp": _apply_event_hp_loss(current_hp, hp_loss, relics), "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Golden Idol":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "BOULDER"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [{"relic_id": "Golden Idol", "name": "Golden Idol"}],
                }
            event.screen = "RESULT"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "BOULDER":
            event.screen = "RESULT"
            if action_index == 0:
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Injury")],
                }
            if action_index == 1:
                return {
                    "gold": gold,
                    "hp": _apply_event_hp_loss(current_hp, _golden_idol_hp_loss(max_hp=max_hp, ascension_level=ascension_level), relics),
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                }
            if action_index == 2:
                max_hp_loss = _golden_idol_max_hp_loss(max_hp=max_hp, ascension_level=ascension_level)
                new_max_hp = max(1, max_hp - max_hp_loss)
                return {
                    "gold": gold,
                    "hp": min(current_hp, new_max_hp),
                    "max_hp": new_max_hp,
                    "potions": [],
                    "open_rewards": False,
                }
            raise NotImplementedError(f"native_sim_v3 Golden Idol action {action_index} is not implemented.")
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "The Mausoleum":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "RESULT"
                result = bool(randoms.stream("misc").random_boolean())
                if ascension_level >= 15:
                    result = True
                rewards: dict[str, object] = {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_relics": [draw_screenless_relic(roll_random_relic_tier(randoms))],
                }
                if result:
                    rewards["add_cards"] = [make_card("Writhe")]
                return rewards
            event.screen = "RESULT"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Accursed Blacksmith":
        if event.screen == "INTRO":
            upgradable = _upgradable_candidate_indexes(deck)
            if action_index == 0 and upgradable:
                event.screen = "RESULT"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "upgrade",
                    "candidate_indexes": upgradable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            if action_index == 1:
                event.screen = "RESULT"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "add_cards": [make_card("Pain")],
                    "add_relics": [make_relic("WarpedTongs")],
                }
            event.screen = "RESULT"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Purifier":
        if event.screen == "INTRO":
            purgeable = _purgeable_candidate_indexes(deck)
            if action_index == 0 and purgeable:
                event.screen = "COMPLETE"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "purge",
                    "candidate_indexes": purgeable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            event.screen = "COMPLETE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Upgrade Shrine":
        if event.screen == "INTRO":
            upgradable = _upgradable_candidate_indexes(deck)
            if action_index == 0 and upgradable:
                event.screen = "COMPLETE"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "upgrade",
                    "candidate_indexes": upgradable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            event.screen = "COMPLETE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Transmorgrifier":
        if event.screen == "INTRO":
            purgeable = _purgeable_candidate_indexes(deck)
            if action_index == 0 and purgeable:
                event.screen = "COMPLETE"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_card_select": True,
                    "card_select_mode": "transform",
                    "candidate_indexes": purgeable,
                    "return_phase": "EVENT",
                    "clear_event_on_finish": False,
                }
            event.screen = "COMPLETE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Colosseum":
        if event.screen == "INTRO":
            event.screen = "FIGHT"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "FIGHT":
            event.screen = "POST_COMBAT"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_combat": True,
                "encounter_name": "Colosseum Slavers",
                "event_rewards": {
                    "gold": 0,
                    "relics": [],
                    "potions": [],
                    "cards": [],
                    "card_groups": [],
                    "reopen_event": EventState("Colosseum", screen="POST_COMBAT"),
                },
            }
        if event.screen == "POST_COMBAT":
            if action_index == 1:
                event.screen = "LEAVE"
                return {
                    "gold": gold,
                    "hp": current_hp,
                    "max_hp": max_hp,
                    "potions": [],
                    "open_rewards": False,
                    "open_combat": True,
                    "encounter_name": "Colosseum Nobs",
                    "elite_trigger": True,
                    "event_rewards": {
                        "gold": 100,
                        "relics": [
                            draw_relic("RARE"),
                            draw_relic("UNCOMMON"),
                        ],
                        "potions": [],
                        "cards": [],
                        "card_groups": [],
                    },
                }
            event.screen = "LEAVE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Mysterious Sphere":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "PRE_COMBAT"
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "END"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "PRE_COMBAT":
            event.screen = "END"
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_combat": True,
                "encounter_name": "2 Orb Walkers",
                "event_rewards": {
                    "gold": int(randoms.stream("misc").random(45, 55)),
                    "relics": [draw_screenless_relic("RARE")],
                    "potions": [],
                    "cards": [],
                    "card_groups": [],
                },
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "SecretPortal":
        if event.screen == "INTRO":
            if action_index == 0:
                event.screen = "ACCEPT"
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "LEAVE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "ACCEPT":
            return {
                "gold": gold,
                "hp": current_hp,
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "warp_to_boss": True,
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "SensoryStone":
        if event.screen == "INTRO":
            event.screen = "INTRO_2"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "INTRO_2":
            event.screen = "LEAVE"
            choice = 1 if action_index == 0 else 2 if action_index == 1 else 3
            hp_loss = 0 if choice == 1 else 5 if choice == 2 else 10
            randoms.stream("misc").random_long()
            reward_groups = [
                generate_colorless_reward_group(
                    randoms,
                    owned_relic_ids=_relic_ids(relics or []),
                    runtime_card_pools=runtime_card_pools,
                )
                for _ in range(choice)
            ]
            return {
                "gold": gold,
                "hp": _apply_event_hp_loss(current_hp, hp_loss, relics),
                "max_hp": max_hp,
                "potions": [],
                "open_rewards": False,
                "open_event_card_reward": True,
                "reward_cards": list(reward_groups[0]) if reward_groups else [],
                "reward_card_groups": reward_groups[1:],
            }
        return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "leave_event": True}
    if event.event_id == "Spire Heart":
        if event.screen == "INTRO":
            event.screen = "MIDDLE"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "MIDDLE":
            event.screen = "MIDDLE_2"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "MIDDLE_2":
            if final_act_available and has_ruby_key and has_emerald_key and has_sapphire_key:
                event.screen = "GO_TO_ENDING"
                return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
            event.screen = "DEATH"
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False}
        if event.screen == "GO_TO_ENDING":
            return {"gold": gold, "hp": current_hp, "max_hp": max_hp, "potions": [], "open_rewards": False, "advance_to_final_act": True}
        return {
            "gold": gold,
            "hp": current_hp,
            "max_hp": max_hp,
            "potions": [],
            "open_rewards": False,
            "victory": True,
        }
    raise NotImplementedError(f"native_sim_v3 event {event.event_id!r} is not implemented yet.")


def _random_potion(randoms: NativeRandomSet, *, player_class: str = "IRONCLAD") -> dict[str, object]:
    from spirecomm.native_sim_v3.content.potions import draw_random_potion

    return draw_random_potion(randoms, player_class=player_class)


def _random_colorless_card(
    randoms: NativeRandomSet,
    pool_key: str,
    *,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    pool = list((runtime_card_pools or card_pools()).get(pool_key, []))
    if not pool:
        raise NotImplementedError(f"native_sim_v3 has no card pool {pool_key!r} for event reward.")
    card_id = str(pool[int(randoms.stream("card").random(0, len(pool) - 1))])
    return make_card(card_id)


def _is_bottled_card(card: dict[str, object]) -> bool:
    return bool(card.get("bottled") or card.get("in_bottle_flame") or card.get("in_bottle_lightning") or card.get("in_bottle_tornado"))


def _non_bottled_card_count(deck: list[dict[str, object]]) -> int:
    return sum(1 for card in deck if not _is_bottled_card(card))


def _purgeable_candidate_indexes(deck: list[dict[str, object]]) -> list[int]:
    return [
        index for index, card in enumerate(deck)
        if str(card.get("type") or "") != "STATUS"
        and str(card.get("card_id") or card.get("id") or "") not in _UNPURGEABLE_CARD_IDS
        and not _is_bottled_card(card)
    ]


def _raw_purgeable_candidate_indexes(deck: list[dict[str, object]]) -> list[int]:
    return [
        index for index, card in enumerate(deck)
        if str(card.get("card_id") or card.get("id") or "") not in _UNPURGEABLE_CARD_IDS
    ]


def _upgradable_candidate_indexes(deck: list[dict[str, object]]) -> list[int]:
    return [
        index for index, card in enumerate(deck)
        if can_upgrade_card(card)
    ]


def _golden_idol_hp_loss(*, max_hp: int, ascension_level: int) -> int:
    return int(max_hp * (0.35 if ascension_level >= 15 else 0.25))


def _golden_idol_max_hp_loss(*, max_hp: int, ascension_level: int) -> int:
    loss = int(max_hp * (0.1 if ascension_level >= 15 else 0.08))
    return max(1, loss)


def _has_card_with_x_damage(deck: list[dict[str, object]], required_damage: int) -> bool:
    for card in deck:
        if str(card.get("type") or "") != "ATTACK":
            continue
        if int(card.get("base_damage") or 0) >= int(required_damage):
            return True
    return False


def _random_face_relic(randoms: NativeRandomSet, *, relic_ids: set[str]) -> dict[str, object]:
    choices = [
        relic_id for relic_id in ("CultistMask", "FaceOfCleric", "GremlinMask", "NlothsMask", "SsserpentHead")
        if relic_id not in relic_ids
    ]
    if not choices:
        choices = ["Circlet"]
    ordered = list(choices)
    java_shuffle_in_place(ordered, int(randoms.stream("misc").random_long()))
    return make_relic(ordered[0])


def _dead_adventurer_encounter_name(enemy: int) -> str:
    if enemy == 0:
        return "3 Sentries"
    if enemy == 1:
        return "Gremlin Nob"
    return "Lagavulin Event"


def _draw_reward_card_by_pool(
    randoms: NativeRandomSet,
    pool_key: str,
    *,
    uuid_prefix: str,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    pool = list((runtime_card_pools or card_pools()).get(pool_key, []))
    if not pool:
        raise NotImplementedError(f"native_sim_v3 has no card pool {pool_key!r} for Match and Keep.")
    card_id = pool[int(randoms.stream("card").random(len(pool) - 1))]
    return make_card(card_id, uuid=f"{uuid_prefix}-{card_id}")


def _draw_random_curse(
    randoms: NativeRandomSet,
    *,
    uuid_prefix: str,
) -> dict[str, object]:
    pool = card_library_random_curse_pool()
    card_id = pool[int(randoms.stream("card").random(len(pool) - 1))]
    return make_card(card_id, uuid=f"{uuid_prefix}-{card_id}")


def _draw_match_colorless_card(
    randoms: NativeRandomSet,
    rarity: str,
    *,
    uuid_prefix: str,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    pools = runtime_card_pools or card_pools()
    uncommon = list(pools.get("COLORLESS_UNCOMMON", []))
    rare = list(pools.get("COLORLESS_RARE", []))
    combined = [*uncommon, *rare]
    if not combined:
        return make_card("Swift Strike", uuid=f"{uuid_prefix}-Swift Strike")
    java_shuffle_in_place(combined, int(randoms.stream("shuffle").random_long()))
    if runtime_card_pools is not None:
        runtime_card_pools["COLORLESS_UNCOMMON"] = [card_id for card_id in combined if card_id in set(uncommon)]
        runtime_card_pools["COLORLESS_RARE"] = [card_id for card_id in combined if card_id in set(rare)]
    preferred = "COLORLESS_RARE" if str(rarity) == "RARE" else "COLORLESS_UNCOMMON"
    fallback = "COLORLESS_UNCOMMON" if preferred == "COLORLESS_RARE" else preferred
    preferred_set = set(rare if preferred == "COLORLESS_RARE" else uncommon)
    fallback_set = set(uncommon if fallback == "COLORLESS_UNCOMMON" else rare)
    for card_id in combined:
        if card_id in preferred_set:
            return make_card(card_id, uuid=f"{uuid_prefix}-{card_id}")
    for card_id in combined:
        if card_id in fallback_set:
            return make_card(card_id, uuid=f"{uuid_prefix}-{card_id}")
    return make_card("Swift Strike", uuid=f"{uuid_prefix}-Swift Strike")


def _build_match_and_keep_cards(
    randoms: NativeRandomSet,
    *,
    ascension_level: int,
    runtime_card_pools: dict[str, list[str]] | None = None,
    player_class: str = "IRONCLAD",
    relics: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    pools = runtime_card_pools if runtime_card_pools is not None else {
        key: list(values) for key, values in card_pools(player_class).items()
    }
    base_cards = [
        _draw_reward_card_by_pool(randoms, class_reward_pool_key("RARE"), uuid_prefix="match-rare", runtime_card_pools=pools),
        _draw_reward_card_by_pool(randoms, class_reward_pool_key("UNCOMMON"), uuid_prefix="match-uncommon", runtime_card_pools=pools),
        _draw_reward_card_by_pool(randoms, class_reward_pool_key("COMMON"), uuid_prefix="match-common", runtime_card_pools=pools),
    ]
    if ascension_level >= 15:
        base_cards.append(_draw_random_curse(randoms, uuid_prefix="match-curse-a15"))
        base_cards.append(_draw_random_curse(randoms, uuid_prefix="match-curse"))
    else:
        base_cards.append(_draw_match_colorless_card(randoms, "UNCOMMON", uuid_prefix="match-colorless", runtime_card_pools=pools))
        base_cards.append(_draw_random_curse(randoms, uuid_prefix="match-curse"))
    start_card_id = _match_and_keep_start_card_id(player_class)
    base_cards.append(make_card(start_card_id, uuid=f"match-starter-{start_card_id}"))

    previewed = apply_reward_preview_relics(base_cards, owned_relic_ids=_relic_ids(relics or []))
    duplicates = []
    for index, card in enumerate(previewed):
        duplicate = dict(card)
        duplicate["uuid"] = f"match-copy-{index}-{card['card_id']}"
        duplicates.append(duplicate)
    board = previewed + duplicates
    java_shuffle_in_place(board, int(randoms.stream("misc").random_long()))
    return board


def _match_and_keep_start_card_id(player_class: str) -> str:
    return {
        "IRONCLAD": "Bash",
        "THE_SILENT": "Neutralize",
        "DEFECT": "Zap",
        "WATCHER": "Eruption",
    }.get(str(player_class), "Bash")


def _match_and_keep_card_position(card_index: int) -> int:
    # GremlinMatchGamePatch exposes choices in visual grid order, not in the
    # shuffled CardGroup order used internally by the event.
    return int(card_index) % 4 + 4 * (int(card_index) % 3)


def _match_and_keep_pickable_indexes(event: EventState) -> list[int]:
    cards = list(event.data.get("cards") or [])
    removed = {int(index) for index in list(event.data.get("removed_card_indexes") or [])}
    first_index = event.data.get("first_card_index")
    first = int(first_index) if first_index is not None else None
    indexes = [
        index
        for index in range(len(cards))
        if index not in removed and (first is None or index != first)
    ]
    return sorted(indexes, key=_match_and_keep_card_position)


def _resolve_match_and_keep_autoplay(cards: list[dict[str, object]], *, attempts: int) -> list[dict[str, object]]:
    memory: dict[str, int] = {}
    matched: list[dict[str, object]] = []
    available = list(range(len(cards)))
    cursor = 0
    for _ in range(max(0, int(attempts))):
        while cursor < len(cards) and cursor not in available:
            cursor += 1
        if cursor >= len(cards):
            break
        first = cursor
        first_card = cards[first]
        available.remove(first)
        first_id = str(first_card.get("card_id"))
        second = None
        remembered = memory.get(first_id)
        if remembered is not None and remembered in available:
            second = remembered
        else:
            for candidate in available:
                second = candidate
                break
        if second is None:
            break
        second_card = cards[second]
        available.remove(second)
        second_id = str(second_card.get("card_id"))
        if first_id == second_id:
            matched.append(dict(first_card))
            memory.pop(first_id, None)
        else:
            memory[first_id] = first
            memory[second_id] = second
        cursor = 0
    return matched


def _generate_library_cards(
    randoms: NativeRandomSet,
    *,
    card_blizz_randomizer: int,
    owned_relic_ids: set[str] | None = None,
    count: int = 20,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, object]], int]:
    reward: list[dict[str, object]] = []
    current_randomizer = int(card_blizz_randomizer)
    pools = runtime_card_pools or card_pools()
    while len(reward) < int(count):
        roll = int(randoms.stream("card").random(99)) + current_randomizer
        if roll < 3:
            rarity = "RARE"
        elif roll < 40:
            rarity = "UNCOMMON"
        else:
            rarity = "COMMON"
        pool = list(pools.get(f"RED_{rarity}", []))
        if not pool:
            break
        card_id = pool[int(randoms.stream("card").random(len(pool) - 1))]
        if any(existing["card_id"] == card_id for existing in reward):
            continue
        reward.append(make_card(card_id, uuid=f"library-{card_id}-{len(reward)}"))
    return apply_reward_preview_relics(reward, owned_relic_ids=owned_relic_ids), current_randomizer


def _pick_random_upgradable_indexes(randoms: NativeRandomSet, deck: list[dict[str, object]], *, count: int) -> list[int]:
    candidates = _upgradable_candidate_indexes(deck)
    ordered = list(candidates)
    java_shuffle_in_place(ordered, int(randoms.stream("misc").random_long()))
    return ordered[: max(0, int(count))]
