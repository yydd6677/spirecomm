from __future__ import annotations

from typing import Any

from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.content import draw_random_relic, make_card, upgrade_card
from spirecomm.native_sim_v3.content.cards import card_pools, class_reward_pool_key
from spirecomm.native_sim_v3.content.pricing import reward_rarity_rules
from spirecomm.native_sim_v3.content.room_reward_rules import room_reward_rules
from spirecomm.native_sim_v3.content.reward_rules import card_blizz_rules, post_combat_potion_rules
from spirecomm.native_sim_v3.content.potions import roll_random_potion
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet


def _roll_rarity(
    randoms: NativeRandomSet,
    *,
    rare_chance: int,
    uncommon_chance: int,
    card_blizz_randomizer: int = 0,
) -> str:
    roll = int(randoms.stream("card").random(99)) + int(card_blizz_randomizer)
    if rare_chance >= 100:
        return "RARE"
    if roll < rare_chance:
        return "RARE"
    if roll < rare_chance + uncommon_chance:
        return "UNCOMMON"
    return "COMMON"


def _reward_pool_key(rarity: str) -> str:
    return class_reward_pool_key(rarity)


def _reward_pool_key_for_relics(rarity: str, owned_relic_ids: set[str] | None = None) -> str:
    if "PrismaticShard" in set(owned_relic_ids or ()):
        return f"ANY_COLOR_{str(rarity)}"
    return _reward_pool_key(rarity)


def _roll_room_relic_tier(randoms: NativeRandomSet, *, room_type: str) -> str:
    common_cutoff, rare_gt = room_reward_rules().relic_tier_thresholds(room_type)
    roll = int(randoms.stream("relic").random(99))
    if roll < common_cutoff:
        return "COMMON"
    if roll > rare_gt:
        return "RARE"
    return "UNCOMMON"


def _adjust_reward_card_count(count: int, owned_relic_ids: set[str] | None = None) -> int:
    relic_ids = set(owned_relic_ids or ())
    adjusted = int(count)
    if "Question Card" in relic_ids:
        adjusted += 1
    if "Busted Crown" in relic_ids:
        adjusted -= 2
    return max(0, adjusted)


def _adjust_reward_rarity_chances(
    *,
    rare_chance: int,
    uncommon_chance: int,
    owned_relic_ids: set[str] | None = None,
) -> tuple[int, int]:
    relic_ids = set(owned_relic_ids or ())
    adjusted_rare = int(rare_chance)
    adjusted_uncommon = int(uncommon_chance)
    if "Nloth's Gift" in relic_ids:
        adjusted_rare *= 3
    return adjusted_rare, adjusted_uncommon


def apply_reward_preview_relics(
    cards: list[dict[str, Any]],
    *,
    owned_relic_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    relic_ids = set(owned_relic_ids or ())
    upgraded_cards: list[dict[str, Any]] = []
    for card in cards:
        card_type = str(card.get("type") or "")
        should_upgrade = False
        if "Molten Egg 2" in relic_ids and card_type == "ATTACK":
            should_upgrade = True
        if "Toxic Egg 2" in relic_ids and card_type == "SKILL":
            should_upgrade = True
        if "Frozen Egg 2" in relic_ids and card_type == "POWER":
            should_upgrade = True
        if should_upgrade:
            upgraded_cards.append(upgrade_card(card))
        else:
            upgraded_cards.append(dict(card))
    return upgraded_cards


def generate_card_reward(randoms: NativeRandomSet, *, count: int = 3) -> list[dict[str, Any]]:
    reward, _ = generate_card_reward_with_state(
        randoms,
        count=count,
        card_blizz_randomizer=card_blizz_rules().start_offset,
    )
    return reward


def generate_card_reward_with_state(
    randoms: NativeRandomSet,
    *,
    count: int = 3,
    card_blizz_randomizer: int | None = None,
    card_upgraded_chance: float = 0.0,
    rare_chance: int | None = None,
    uncommon_chance: int | None = None,
    owned_relic_ids: set[str] | None = None,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    blizz_rules = card_blizz_rules()
    pools = runtime_card_pools or card_pools()
    reward: list[dict[str, Any]] = []
    current_randomizer = blizz_rules.start_offset if card_blizz_randomizer is None else int(card_blizz_randomizer)
    target_count = _adjust_reward_card_count(count, owned_relic_ids)
    adjusted_rare_chance, adjusted_uncommon_chance = _adjust_reward_rarity_chances(
        rare_chance=reward_rarity_rules().rare_chance if rare_chance is None else int(rare_chance),
        uncommon_chance=reward_rarity_rules().uncommon_chance if uncommon_chance is None else int(uncommon_chance),
        owned_relic_ids=owned_relic_ids,
    )
    while len(reward) < target_count:
        rarity = _roll_rarity(
            randoms,
            rare_chance=adjusted_rare_chance,
            uncommon_chance=adjusted_uncommon_chance,
            card_blizz_randomizer=current_randomizer,
        )
        if rarity == "RARE":
            current_randomizer = blizz_rules.start_offset
        elif rarity == "COMMON":
            current_randomizer = max(blizz_rules.max_offset, current_randomizer - blizz_rules.growth)
        pool_key = _reward_pool_key_for_relics(rarity, owned_relic_ids)
        pool = list(pools.get(pool_key, []))
        if not pool:
            break
        reward_ids = {str(existing["card_id"]) for existing in reward}
        if all(str(card_id) in reward_ids for card_id in pool):
            break
        while True:
            if "PrismaticShard" in set(owned_relic_ids or ()):
                # CardLibrary.getAnyColorCard() shuffles with cardRng before
                # getRandomCard(true, rarity) sorts by cardID and picks again.
                # The shuffle order is overwritten, but randomLong is consumed.
                randoms.stream("card").random_long()
            card_id = pool[int(randoms.stream("card").random(len(pool) - 1))]
            if str(card_id) not in reward_ids:
                break
        reward.append(make_card(card_id, uuid=f"reward-{card_id}-{len(reward)}"))
    upgraded_reward: list[dict[str, Any]] = []
    for card in reward:
        if str(card.get("rarity")) != "RARE":
            if randoms.stream("card").random_boolean(card_upgraded_chance) and bool(card.get("can_upgrade", True)):
                card = upgrade_card(card)
        upgraded_reward.append(card)
    return apply_reward_preview_relics(upgraded_reward, owned_relic_ids=owned_relic_ids), current_randomizer


def generate_card_reward_groups_with_state(
    randoms: NativeRandomSet,
    *,
    group_count: int,
    card_blizz_randomizer: int | None = None,
    card_upgraded_chance: float = 0.0,
    rare_chance: int | None = None,
    uncommon_chance: int | None = None,
    owned_relic_ids: set[str] | None = None,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> tuple[list[list[dict[str, Any]]], int]:
    groups: list[list[dict[str, Any]]] = []
    current_randomizer = card_blizz_rules().start_offset if card_blizz_randomizer is None else int(card_blizz_randomizer)
    for _ in range(max(0, int(group_count))):
        cards, current_randomizer = generate_card_reward_with_state(
            randoms,
            count=3,
            card_blizz_randomizer=current_randomizer,
            card_upgraded_chance=card_upgraded_chance,
            rare_chance=rare_chance,
            uncommon_chance=uncommon_chance,
            owned_relic_ids=owned_relic_ids,
            runtime_card_pools=runtime_card_pools,
        )
        if cards:
            groups.append(cards)
    return groups, current_randomizer


def generate_colorless_reward_group(
    randoms: NativeRandomSet,
    *,
    act: int | str = 1,
    dungeon_id: str | None = None,
    count: int = 3,
    owned_relic_ids: set[str] | None = None,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    pools = runtime_card_pools or card_pools()
    rare_chance = act_chances(str(dungeon_id) if dungeon_id is not None else act).colorless_rare_chance
    reward: list[dict[str, Any]] = []
    target_count = _adjust_reward_card_count(count, owned_relic_ids)
    while len(reward) < target_count:
        rarity = "RARE" if randoms.stream("card").random_boolean(rare_chance) else "UNCOMMON"
        pool = list(pools.get(f"COLORLESS_{rarity}", []))
        if not pool:
            break
        reward_ids = {str(existing["card_id"]) for existing in reward}
        if all(str(card_id) in reward_ids for card_id in pool):
            break
        while True:
            card_id = pool[int(randoms.stream("card").random(len(pool) - 1))]
            if str(card_id) not in reward_ids:
                break
        reward.append(make_card(card_id, uuid=f"colorless-reward-{card_id}-{len(reward)}"))
    return apply_reward_preview_relics(reward, owned_relic_ids=owned_relic_ids)


def roll_post_combat_potion(
    randoms: NativeRandomSet,
    *,
    reward_count: int,
    blizzard_potion_mod: int,
    owned_relic_ids: set[str] | None = None,
    player_class: str = "IRONCLAD",
    chance_override: int | None = None,
) -> tuple[dict[str, object] | None, int]:
    relic_ids = set(owned_relic_ids or ())
    rules = post_combat_potion_rules()
    chance = rules.base_chance + int(blizzard_potion_mod) if chance_override is None else int(chance_override)
    if "White Beast Statue" in relic_ids:
        chance = rules.white_beast_chance
    if int(reward_count) >= rules.reward_cap:
        chance = 0
    if int(randoms.stream("potion").random(0, 99)) < chance:
        return roll_random_potion(randoms, player_class=player_class), int(blizzard_potion_mod) - rules.blizzard_mod_amount
    return None, int(blizzard_potion_mod) + rules.blizzard_mod_amount


def generate_monster_room_rewards(
    randoms: NativeRandomSet,
    *,
    act: int,
    dungeon_id: str | None = None,
    room_type: str = "MonsterRoom",
    ascension_level: int = 0,
    card_blizz_randomizer: int,
    blizzard_potion_mod: int,
    owned_relic_ids: set[str] | None = None,
    reward_count: int = 1,
    player_class: str = "IRONCLAD",
    prayer_wheel: bool = False,
    include_gold: bool = True,
    potion_chance_override: int | None = None,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    reward_rules = room_reward_rules()
    gold_min, gold_max = reward_rules.gold_range(room_type)
    rare_chance, uncommon_chance = reward_rules.card_rarity_thresholds(room_type)
    gold = int(randoms.stream("treasure").random(gold_min, gold_max)) if include_gold else 0
    if gold and owned_relic_ids and "Golden Idol" in owned_relic_ids:
        gold += int(float(gold) * 0.25 + 0.5)
    potion, next_blizzard_potion_mod = roll_post_combat_potion(
        randoms,
        reward_count=reward_count,
        blizzard_potion_mod=blizzard_potion_mod,
        owned_relic_ids=owned_relic_ids,
        player_class=player_class,
        chance_override=potion_chance_override,
    )
    card_upgraded_chance = act_chances(str(dungeon_id) if dungeon_id is not None else act).card_upgraded_chance(ascension_level)
    card_groups, next_card_blizz_randomizer = generate_card_reward_groups_with_state(
        randoms,
        group_count=2 if prayer_wheel else 1,
        card_blizz_randomizer=card_blizz_randomizer,
        card_upgraded_chance=card_upgraded_chance,
        rare_chance=rare_chance,
        uncommon_chance=uncommon_chance,
        owned_relic_ids=owned_relic_ids,
        runtime_card_pools=runtime_card_pools,
    )
    cards = list(card_groups[0]) if card_groups else []
    return {
        "gold": gold,
        "potion": potion,
        "cards": cards,
        "card_groups": card_groups,
        "card_blizz_randomizer": next_card_blizz_randomizer,
        "blizzard_potion_mod": next_blizzard_potion_mod,
    }


def generate_elite_relic_rewards(
    randoms: NativeRandomSet,
    *,
    act: int | str = 1,
    owned_relic_ids: set[str] | None = None,
    black_star: bool = False,
    relic_drawer: Any | None = None,
) -> list[dict[str, object]]:
    reward_relics: list[dict[str, object]] = []
    exclude = set(owned_relic_ids or ())
    draw_relic = relic_drawer or (lambda tier: draw_random_relic(randoms, tier, exclude=exclude))
    first_relic = draw_relic(_roll_room_relic_tier(randoms, room_type="MonsterRoomElite"))
    reward_relics.append(first_relic)
    exclude.add(str(first_relic["relic_id"]))
    if black_star:
        second_relic = draw_relic(_roll_room_relic_tier(randoms, room_type="MonsterRoomElite"))
        reward_relics.append(second_relic)
    return reward_relics


def generate_combat_rewards(randoms: NativeRandomSet, *, reward_count: int = 0) -> dict[str, object]:
    rewards = generate_monster_room_rewards(
        randoms,
        act=1,
        ascension_level=0,
        card_blizz_randomizer=card_blizz_rules().start_offset,
        blizzard_potion_mod=0,
        reward_count=reward_count,
    )
    return {
        "gold": rewards["gold"],
        "potion": rewards["potion"],
        "cards": rewards["cards"],
    }
