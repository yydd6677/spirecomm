from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from spirecomm.native_sim_v3.content import make_card, truly_random_card_from_source_pools
from spirecomm.native_sim_v3.content.cards import (
    card_catalog,
    card_library_random_curse_pool,
    card_pools,
    class_reward_pool_key,
    source_card_pools,
)
from spirecomm.native_sim_v3.source_paths import sts_source_path


class NeowRewardType(str, Enum):
    RANDOM_COLORLESS_2 = "RANDOM_COLORLESS_2"
    THREE_CARDS = "THREE_CARDS"
    ONE_RANDOM_RARE_CARD = "ONE_RANDOM_RARE_CARD"
    REMOVE_CARD = "REMOVE_CARD"
    UPGRADE_CARD = "UPGRADE_CARD"
    RANDOM_COLORLESS = "RANDOM_COLORLESS"
    TRANSFORM_CARD = "TRANSFORM_CARD"
    THREE_SMALL_POTIONS = "THREE_SMALL_POTIONS"
    RANDOM_COMMON_RELIC = "RANDOM_COMMON_RELIC"
    TEN_PERCENT_HP_BONUS = "TEN_PERCENT_HP_BONUS"
    HUNDRED_GOLD = "HUNDRED_GOLD"
    THREE_ENEMY_KILL = "THREE_ENEMY_KILL"
    REMOVE_TWO = "REMOVE_TWO"
    TRANSFORM_TWO_CARDS = "TRANSFORM_TWO_CARDS"
    ONE_RARE_RELIC = "ONE_RARE_RELIC"
    THREE_RARE_CARDS = "THREE_RARE_CARDS"
    TWO_FIFTY_GOLD = "TWO_FIFTY_GOLD"
    TWENTY_PERCENT_HP_BONUS = "TWENTY_PERCENT_HP_BONUS"
    BOSS_RELIC = "BOSS_RELIC"


class NeowDrawback(str, Enum):
    NONE = "NONE"
    TEN_PERCENT_HP_LOSS = "TEN_PERCENT_HP_LOSS"
    NO_GOLD = "NO_GOLD"
    CURSE = "CURSE"
    PERCENT_DAMAGE = "PERCENT_DAMAGE"


NEOW_REWARD_SOURCE = sts_source_path("neow/NeowReward.java")
NEOW_CASE_HEADER_PATTERN = re.compile(r"case (\d+): \{")
NEOW_REWARD_ADD_PATTERN = re.compile(r"rewardOptions\.add\(new NeowRewardDef\(NeowRewardType\.([A-Z0-9_]+),")
NEOW_DRAWBACK_OPTION_PATTERN = re.compile(r"NeowRewardDrawbackDef\(NeowRewardDrawback\.([A-Z_]+),")
NEOW_MINI_REWARD_PATTERN = re.compile(
    r"firstMini \? new NeowRewardDef\(NeowRewardType\.([A-Z0-9_]+),.*?\) : new NeowRewardDef\(NeowRewardType\.([A-Z0-9_]+),",
    re.DOTALL,
)


def _neow_source() -> str:
    return NEOW_REWARD_SOURCE.read_text(encoding="utf-8")


def _extract_method_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise RuntimeError(f"native_sim_v3 could not find Neow method signature {signature!r}.")
    brace_index = source.find("{", start)
    if brace_index < 0:
        raise RuntimeError(f"native_sim_v3 could not find opening brace for {signature!r}.")
    depth = 1
    cursor = brace_index + 1
    while cursor < len(source) and depth > 0:
        char = source[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        cursor += 1
    if depth != 0:
        raise RuntimeError(f"native_sim_v3 could not match braces for {signature!r}.")
    return source[brace_index + 1 : cursor - 1]


@lru_cache(maxsize=1)
def _neow_reward_option_specs() -> dict[int, list[tuple[NeowRewardType, NeowDrawback | None]]]:
    source = _neow_source()
    body = _extract_method_body(source, "private ArrayList<NeowRewardDef> getRewardOptions(int category)")
    case_matches = list(NEOW_CASE_HEADER_PATTERN.finditer(body))
    case_blocks: dict[int, str] = {}
    for index, match in enumerate(case_matches):
        start = match.end()
        end = case_matches[index + 1].start() if index + 1 < len(case_matches) else len(body)
        case_blocks[int(match.group(1))] = body[start:end]
    specs: dict[int, list[tuple[NeowRewardType, NeowDrawback | None]]] = {}
    for category, body in case_blocks.items():
        entries: list[tuple[NeowRewardType, NeowDrawback | None]] = []
        exclude_on_drawback: NeowDrawback | None = None
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            conditional_not = re.search(r"if \(this\.drawback != NeowRewardDrawback\.([A-Z_]+)\) \{", line)
            if conditional_not:
                exclude_on_drawback = NeowDrawback[conditional_not.group(1)]
                continue
            conditional_break = re.search(r"if \(this\.drawback == NeowRewardDrawback\.([A-Z_]+)\) break;", line)
            if conditional_break:
                exclude_on_drawback = NeowDrawback[conditional_break.group(1)]
                continue
            reward_add = NEOW_REWARD_ADD_PATTERN.search(line)
            if reward_add:
                entries.append((NeowRewardType[reward_add.group(1)], exclude_on_drawback))
                exclude_on_drawback = None
        specs[category] = entries
    return specs


@lru_cache(maxsize=1)
def _neow_drawback_order() -> tuple[NeowDrawback, ...]:
    source = _neow_source()
    body = _extract_method_body(source, "private ArrayList<NeowRewardDrawbackDef> getRewardDrawbackOptions()")
    return tuple(NeowDrawback[name] for name in NEOW_DRAWBACK_OPTION_PATTERN.findall(body))


@lru_cache(maxsize=1)
def _neow_mini_reward_types() -> tuple[NeowRewardType, NeowRewardType]:
    source = _neow_source()
    constructor_body = _extract_method_body(source, "public NeowReward(boolean firstMini)")
    match = NEOW_MINI_REWARD_PATTERN.search(constructor_body)
    if not match:
        raise RuntimeError("native_sim_v3 could not parse Neow mini blessing reward source.")
    return (NeowRewardType[match.group(1)], NeowRewardType[match.group(2)])


@dataclass(slots=True)
class NeowOption:
    choice_index: int
    bonus: NeowRewardType
    drawback: NeowDrawback
    bonus_text: str
    drawback_text: str = ""

    def to_action(self) -> dict[str, Any]:
        return {
            "kind": "neow",
            "name": f"OPTION_{self.choice_index}",
            "label": f"OPTION_{self.choice_index}",
            "choice_index": self.choice_index,
            "bonus": self.bonus.value,
            "drawback": self.drawback.value,
            "bonus_text": self.bonus_text,
            "drawback_text": self.drawback_text,
        }


def _reward_text_map(max_hp: int) -> dict[NeowRewardType, str]:
    hp_bonus = int(max_hp * 0.1)
    return {
        NeowRewardType.THREE_CARDS: "Choose 1 of 3 cards.",
        NeowRewardType.ONE_RANDOM_RARE_CARD: "Obtain a random rare card.",
        NeowRewardType.REMOVE_CARD: "Remove a card from your deck.",
        NeowRewardType.UPGRADE_CARD: "Upgrade a card in your deck.",
        NeowRewardType.TRANSFORM_CARD: "Transform a card.",
        NeowRewardType.RANDOM_COLORLESS: "Choose 1 of 3 random colorless cards.",
        NeowRewardType.THREE_SMALL_POTIONS: "Obtain 3 random potions.",
        NeowRewardType.RANDOM_COMMON_RELIC: "Obtain a random common relic.",
        NeowRewardType.TEN_PERCENT_HP_BONUS: f"Gain {hp_bonus} Max HP.",
        NeowRewardType.THREE_ENEMY_KILL: "Enemies in your first 3 combats have 1 HP.",
        NeowRewardType.HUNDRED_GOLD: "Gain 100 gold.",
        NeowRewardType.RANDOM_COLORLESS_2: "Choose 1 of 3 rare colorless cards.",
        NeowRewardType.REMOVE_TWO: "Remove 2 cards.",
        NeowRewardType.ONE_RARE_RELIC: "Obtain a random rare relic.",
        NeowRewardType.THREE_RARE_CARDS: "Choose 1 of 3 rare cards.",
        NeowRewardType.TWO_FIFTY_GOLD: "Gain 250 gold.",
        NeowRewardType.TRANSFORM_TWO_CARDS: "Transform 2 cards.",
        NeowRewardType.TWENTY_PERCENT_HP_BONUS: f"Gain {hp_bonus * 2} Max HP.",
        NeowRewardType.BOSS_RELIC: "Lose your starter relic. Obtain a random boss relic.",
    }


def _reward_defs(max_hp: int) -> dict[int, list[tuple[NeowRewardType, str]]]:
    reward_text = _reward_text_map(max_hp)
    return {
        category: [(reward_type, reward_text[reward_type]) for reward_type, _ in entries]
        for category, entries in _neow_reward_option_specs().items()
    }


def _drawback_defs(current_hp: int, max_hp: int) -> list[tuple[NeowDrawback, str]]:
    hp_bonus = int(max_hp * 0.1)
    drawback_text = {
        NeowDrawback.TEN_PERCENT_HP_LOSS: f"Lose {hp_bonus} Max HP.",
        NeowDrawback.NO_GOLD: "Lose all gold.",
        NeowDrawback.CURSE: "Obtain a curse.",
        NeowDrawback.PERCENT_DAMAGE: f"Lose {current_hp // 10 * 3} HP.",
    }
    return [(drawback, drawback_text[drawback]) for drawback in _neow_drawback_order()]


def generate_blessing_options(randoms: Any, *, current_hp: int, max_hp: int, mini_blessing: bool = False) -> list[NeowOption]:
    reward_defs = _reward_defs(max_hp)
    drawbacks = _drawback_defs(current_hp, max_hp)
    reward_specs = _neow_reward_option_specs()
    reward_text = _reward_text_map(max_hp)
    rng = randoms.stream("neow")
    if mini_blessing:
        return [
            NeowOption(
                choice_index=index,
                bonus=reward_type,
                drawback=NeowDrawback.NONE,
                bonus_text=reward_text[reward_type],
                drawback_text="",
            )
            for index, reward_type in enumerate(_neow_mini_reward_types())
        ]
    options: list[NeowOption] = []
    for category in range(4):
        pool = list(reward_defs[category])
        if category == 2:
            drawback, drawback_text = drawbacks[int(rng.random(0, len(drawbacks) - 1))]
            allowed = {
                reward_type
                for reward_type, excluded_drawback in reward_specs[category]
                if excluded_drawback is None or excluded_drawback != drawback
            }
            pool = [entry for entry in pool if entry[0] in allowed]
        else:
            drawback, drawback_text = NeowDrawback.NONE, ""
        reward, reward_text = pool[int(rng.random(0, len(pool) - 1))]
        options.append(
            NeowOption(
                choice_index=category,
                bonus=reward,
                drawback=drawback,
                bonus_text=reward_text,
                drawback_text=drawback_text,
            )
        )
    return options


def draw_neow_cards(
    randoms: Any,
    *,
    rare_only: bool = False,
    colorless: bool = False,
    runtime_pools: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    pools = runtime_pools or card_pools()
    chosen: list[str] = []
    neow_rng = randoms.stream("neow")
    card_rng = randoms.stream("card")
    for _ in range(3):
        if colorless:
            rolled_rarity = "UNCOMMON" if neow_rng.random_boolean(0.33) else "COMMON"
            if rare_only:
                key = "COLORLESS_RARE"
            else:
                if rolled_rarity == "COMMON":
                    rolled_rarity = "UNCOMMON"
                key = f"COLORLESS_{rolled_rarity}"
            pick_rng = card_rng
            pool = sorted(pools[key])
        else:
            rolled_rarity = "UNCOMMON" if neow_rng.random_boolean(0.33) else "COMMON"
            if rare_only:
                key = class_reward_pool_key("RARE")
            else:
                key = class_reward_pool_key(rolled_rarity)
            pick_rng = neow_rng
            pool = pools[key]
        card_id = pool[int(pick_rng.random(0, len(pool) - 1))]
        while card_id in chosen:
            card_id = pool[int(pick_rng.random(0, len(pool) - 1))]
        chosen.append(card_id)
    prefix = "neow-colorless" if colorless else "neow-reward"
    return [make_card(card_id, uuid=f"{prefix}-{index}-{card_id}") for index, card_id in enumerate(chosen)]


def draw_neow_rare_card(randoms: Any, *, runtime_pools: dict[str, list[str]] | None = None) -> dict[str, Any]:
    pools = runtime_pools or card_pools()
    pool = pools[class_reward_pool_key("RARE")]
    card_id = pool[int(randoms.stream("neow").random(0, len(pool) - 1))]
    return make_card(card_id, uuid=f"neow-rare-{card_id}")


def draw_neow_curse(randoms: Any, *, runtime_pools: dict[str, list[str]] | None = None) -> dict[str, Any]:
    pool = card_library_random_curse_pool()
    card_id = pool[int(randoms.stream("card").random(0, len(pool) - 1))]
    return make_card(card_id, uuid=f"neow-curse-{card_id}")


def transform_card(
    randoms: Any,
    card: dict[str, Any],
    *,
    auto_upgrade: bool = False,
    runtime_pools: dict[str, list[str]] | None = None,
    source_pools: dict[str, list[str]] | None = None,
    rng_stream: str = "neow",
) -> dict[str, Any]:
    pools = runtime_pools or card_pools()
    source = source_pools or source_card_pools()
    catalog = card_catalog()
    card_id = str(card.get("card_id") or "")
    color = str(card.get("color") or catalog[card_id].color)
    transformed: dict[str, Any] | None
    if color == "COLORLESS":
        candidates = [
            *list(pools.get("COLORLESS_UNCOMMON", [])),
            *list(pools.get("COLORLESS_RARE", [])),
        ]
        candidates = [candidate for candidate in candidates if candidate != card_id and candidate in catalog]
        transformed = None
        if candidates:
            chosen_id = candidates[int(randoms.stream(rng_stream).random(0, len(candidates) - 1))]
            transformed = make_card(chosen_id, uuid=f"source-random-{chosen_id}")
    elif color == "CURSE":
        transformed = truly_random_card_from_source_pools(
            randoms,
            source_pools=source,
            color_mode="CURSE",
            prohibited_id=card_id,
            rng_stream=rng_stream,
        )
        if transformed is None:
            pool = [candidate for candidate in pools["CURSE"] if candidate != card_id]
            if not pool:
                pool = list(pools["CURSE"])
            chosen_id = pool[int(randoms.stream(rng_stream).random(0, len(pool) - 1))]
            transformed = make_card(
                chosen_id,
                upgrades=1 if auto_upgrade else 0,
                uuid=f"neow-transform-{chosen_id}",
            )
    else:
        candidates = [
            *list(pools.get("CLASS_COMMON", pools.get("RED_COMMON", []))),
            *list(source.get("SRC_UNCOMMON", [])),
            *list(source.get("SRC_RARE", [])),
        ]
        candidates = [candidate for candidate in candidates if candidate != card_id and candidate in catalog]
        transformed = None
        if candidates:
            chosen_id = candidates[int(randoms.stream(rng_stream).random(0, len(candidates) - 1))]
            transformed = make_card(chosen_id, uuid=f"source-random-{chosen_id}")
    if transformed is None:
        raise RuntimeError(f"native_sim_v3 failed to transform card {card_id!r} from source pools")
    return make_card(
        str(transformed["card_id"]),
        upgrades=1 if auto_upgrade else 0,
        uuid=f"neow-transform-{transformed['card_id']}",
    )
