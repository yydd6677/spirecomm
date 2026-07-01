from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Callable

from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.content.chests import chest_def
from spirecomm.native_sim_v3.content.relics import draw_random_relic
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet


def _java_round(value: float) -> int:
    return int(value + 0.5)


def _relic_id(relic: dict[str, Any]) -> str:
    return str(relic.get("relic_id") or relic.get("id") or "")


def _remove_first_reward_relic_order_item(reward_order: list[dict[str, object]]) -> list[dict[str, object]]:
    removed = False
    filtered: list[dict[str, object]] = []
    for item in reward_order:
        if not removed and str(item.get("kind") or "") == "reward_relic":
            removed = True
            continue
        filtered.append(item)
    return filtered


@dataclass(slots=True)
class TreasureState:
    chest_type: str
    gold_amount: int
    relic_tier: str
    opened: bool = False
    gold_reward: bool | None = None

    def actions(self) -> list[dict[str, object]]:
        if not self.opened:
            return [{"kind": "treasure", "name": "OPEN_CHEST", "label": "Open Chest", "choice_index": 0}]
        return [{"kind": "treasure", "name": "PROCEED", "label": "Proceed", "choice_index": 0}]

def generate_treasure(
    randoms: NativeRandomSet,
    *,
    act: int | str = 1,
    dungeon_id: str | None = None,
) -> TreasureState:
    chances = act_chances(str(dungeon_id) if dungeon_id is not None else act)
    treasure_rng = randoms.stream("treasure")
    roll = int(randoms.stream("treasure").random(0, 99))
    small_limit = chances.small_chest_chance
    medium_limit = chances.small_chest_chance + chances.medium_chest_chance
    if roll < small_limit:
        chest_type = "SmallChest"
    elif roll < medium_limit:
        chest_type = "MediumChest"
    else:
        chest_type = "LargeChest"
    definition = chest_def(chest_type)
    reward_roll = int(treasure_rng.random(0, 99))
    gold_reward = reward_roll < definition.gold_chance
    if reward_roll < definition.common_chance:
        tier = "COMMON"
    elif reward_roll < definition.common_chance + definition.uncommon_chance:
        tier = "UNCOMMON"
    else:
        tier = "RARE"
    return TreasureState(chest_type=chest_type, gold_amount=0, relic_tier=tier, gold_reward=gold_reward)


def open_treasure(
    treasure: TreasureState,
    randoms: NativeRandomSet,
    *,
    relic_drawer: Callable[[str], dict[str, Any]] | None = None,
    player_relics: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    treasure.opened = True
    draw_relic = relic_drawer or (lambda tier: draw_random_relic(randoms, tier))
    reward_relics: list[dict[str, Any]] = []
    reward_order: list[dict[str, object]] = []
    for relic in list(player_relics or []):
        if str(relic.get("relic_id") or relic.get("id")) != "Matryoshka":
            continue
        counter = int(relic.get("counter", 0) or 0)
        if counter <= 0:
            continue
        counter -= 1
        if counter <= 0:
            relic["counter"] = -2
            relic["used_up"] = True
        else:
            relic["counter"] = counter
        tier = "COMMON" if randoms.stream("relic").random_boolean(0.75) else "UNCOMMON"
        bonus_relic = draw_relic(tier)
        reward_relics.append(bonus_relic)
        reward_order.append({"kind": "reward_relic", "relic_id": str(bonus_relic["relic_id"])})
    if treasure.gold_reward and int(treasure.gold_amount or 0) <= 0:
        definition = chest_def(treasure.chest_type)
        treasure.gold_amount = _java_round(float(randoms.stream("treasure").random(definition.gold_amount * 0.9, definition.gold_amount * 1.1)))
    elif treasure.gold_reward is None:
        treasure.gold_reward = int(treasure.gold_amount or 0) > 0

    if int(treasure.gold_amount):
        reward_order.append({"kind": "reward_gold"})
    relic = draw_relic(treasure.relic_tier)
    reward_relics.append(relic)
    reward_order.append({"kind": "reward_relic", "relic_id": str(relic["relic_id"])})
    for relic in list(player_relics or []):
        if _relic_id(relic) != "NlothsMask":
            continue
        counter = int(relic.get("counter", 0) or 0)
        if counter <= 0:
            continue
        counter -= 1
        relic["counter"] = -2 if counter <= 0 else counter
        if counter <= 0:
            relic["used_up"] = True
        if reward_relics:
            reward_relics.pop(0)
            reward_order = _remove_first_reward_relic_order_item(reward_order)
        break
    rewards: dict[str, object] = {
        "gold": int(treasure.gold_amount),
        "relics": reward_relics,
        "order": reward_order,
    }
    return rewards
