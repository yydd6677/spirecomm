from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from spirecomm.native_sim_v3.content.act_progression import act_for_dungeon_id
from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.content.cards import card_catalog, card_pools, make_card
from spirecomm.native_sim_v3.content.pricing import card_price_for_rarity
from spirecomm.native_sim_v3.content.potions import roll_random_potion
from spirecomm.native_sim_v3.content.shop import shop_rules
from spirecomm.native_sim_v3.run.rewards import apply_reward_preview_relics
from spirecomm.native_sim_v3.content.relics import draw_random_relic_end, is_banned_relic_id, price_for_relic_tier
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet

@dataclass(slots=True)
class ShopState:
    cards: list[dict[str, object]] = field(default_factory=list)
    relics: list[dict[str, object]] = field(default_factory=list)
    potions: list[dict[str, object]] = field(default_factory=list)
    purge_base_cost: int = 75
    purge_cost: int = 75
    purge_available: bool = True

    def actions(self) -> list[dict[str, object]]:
        actions: list[dict[str, object]] = []
        for index, card in enumerate(self.cards):
            card_name = _visible_shop_card_name(card)
            actions.append(
                {
                    "kind": "shop",
                    "item_kind": "card",
                    "item_id": card["card_id"],
                    "name": card_name,
                    "price": int(card["price"]),
                    "shop_index": index,
                    "label": card_name,
                }
            )
        for index, relic in enumerate(self.relics):
            if is_banned_relic_id(relic.get("relic_id") or relic.get("id")):
                continue
            actions.append(
                {
                    "kind": "shop",
                    "item_kind": "relic",
                    "item_id": relic["relic_id"],
                    "name": relic["name"],
                    "price": int(relic["price"]),
                    "shop_index": index,
                    "label": relic["name"],
                }
            )
        for index, potion in enumerate(self.potions):
            actions.append(
                {
                    "kind": "shop",
                    "item_kind": "potion",
                    "item_id": potion["potion_id"],
                    "potion_id": potion["potion_id"],
                    "name": potion["name"],
                    "price": int(potion["price"]),
                    "shop_index": index,
                    "label": potion["name"],
                }
            )
        if self.purge_available:
            actions.append(
                {
                    "kind": "shop",
                    "item_kind": "purge",
                    "item_id": "purge",
                    "name": "Purge",
                    "price": int(self.purge_cost),
                    "label": "Purge",
                }
            )
        actions.append({"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave", "price": 0, "label": "Leave"})
        return actions


def _visible_shop_card_name(card: dict[str, object]) -> str:
    name = str(card.get("name") or card.get("card_id") or card.get("id") or "")
    upgrades = int(card.get("upgrades") or 0)
    if upgrades > 0 and "+" not in name:
        return f"{name}+"
    return name


def generate_shop(
    randoms: NativeRandomSet,
    *,
    player_class: str = "IRONCLAD",
    act: int | str = 1,
    dungeon_id: str | None = None,
    card_blizz_randomizer: int = 0,
    floor_num: int | None = None,
    ascension_level: int = 0,
    owned_relic_ids: set[str] | None = None,
    relic_drawer: Callable[[str], dict[str, Any]] | None = None,
    runtime_card_pools: dict[str, list[str]] | None = None,
    purge_base_cost: int | None = None,
) -> ShopState:
    rules = shop_rules()
    dungeon_key = str(dungeon_id) if dungeon_id is not None else act
    act_num = act_for_dungeon_id(dungeon_key) if dungeon_id is not None else (int(act) if str(act).isdigit() else None)
    relic_ids = set(owned_relic_ids or ())
    merchant_rng = randoms.stream("merchant")
    cards = _generate_merchant_colored_cards(
        randoms,
        player_class=player_class,
        card_blizz_randomizer=card_blizz_randomizer,
        runtime_card_pools=runtime_card_pools,
    )
    used_card_ids = {str(card.get("card_id")) for card in cards}
    for rarity in rules.colorless_card_rarities:
        card = _generate_colorless_shop_card(randoms, rarity=rarity, exclude=used_card_ids, runtime_card_pools=runtime_card_pools)
        if card is None:
            continue
        cards.append(card)
    cards = apply_reward_preview_relics(cards, owned_relic_ids=relic_ids)
    for card in cards:
        card["price"] = _price_with_jitter(
            merchant_rng,
            card_price_for_rarity(str(card["rarity"]))
            * (rules.colorless_price_bump if str(card["color"]) == "COLORLESS" else 1.0),
            lower=1.0 - rules.card_price_jitter,
            upper=1.0 + rules.card_price_jitter,
        )
    colored_indexes = [index for index, card in enumerate(cards) if str(card.get("color") or "") != "COLORLESS"]
    if colored_indexes:
        sale_index = colored_indexes[int(merchant_rng.random(len(colored_indexes) - 1))]
        cards[sale_index]["price"] = max(0, int(cards[sale_index]["price"]) // 2)
        cards[sale_index]["on_sale"] = True
    for card in cards:
        card["price"] = _apply_shop_price_modifiers(
            int(card["price"]),
            ascension_level=ascension_level,
            owned_relic_ids=relic_ids,
        )

    relics: list[dict[str, object]] = []
    used_relics: set[str] = set()
    draw_relic = relic_drawer or (
        lambda tier: draw_random_relic_end(
            randoms,
            tier,
            character=player_class,
            exclude=used_relics,
            floor_num=floor_num,
            current_room_type="ShopRoom",
            owned_relic_ids=relic_ids,
            act=act_num,
        )
    )
    for index in range(3):
        tier = _roll_shop_relic_tier(randoms) if index != 2 else "SHOP"
        relic = draw_relic(tier)
        used_relics.add(str(relic["relic_id"]))
        relic["price"] = _apply_shop_price_modifiers(
            _price_with_jitter(
                merchant_rng,
                price_for_relic_tier(str(relic["tier"])),
                lower=1.0 - rules.relic_price_jitter,
                upper=1.0 + rules.relic_price_jitter,
                rounding="round",
            ),
            ascension_level=ascension_level,
            owned_relic_ids=relic_ids,
        )
        relics.append(relic)

    potions: list[dict[str, object]] = []
    for index in range(3):
        potion = generate_shop_replacement_potion(
            randoms,
            shop_index=index,
            player_class=player_class,
            dungeon_id=dungeon_id,
            ascension_level=ascension_level,
            owned_relic_ids=relic_ids,
        )
        potions.append(potion)

    base_purge_cost = int(rules.purge_cost if purge_base_cost is None else purge_base_cost)
    purge_cost = base_purge_cost
    purge_cost = _apply_shop_price_modifiers(purge_cost, ascension_level=ascension_level, owned_relic_ids=relic_ids)
    if "Smiling Mask" in relic_ids:
        purge_cost = 50

    return ShopState(
        cards=cards,
        relics=relics,
        potions=potions,
        purge_base_cost=base_purge_cost,
        purge_cost=int(purge_cost),
    )


def _generate_merchant_colored_cards(
    randoms: NativeRandomSet,
    *,
    player_class: str,
    card_blizz_randomizer: int,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> list[dict[str, object]]:
    del player_class
    cards: list[dict[str, object]] = []
    first_attack = _roll_merchant_colored_card(
        randoms,
        card_type="ATTACK",
        card_blizz_randomizer=card_blizz_randomizer,
        runtime_card_pools=runtime_card_pools,
    )
    if first_attack is not None:
        cards.append(first_attack)
    second_attack = _roll_merchant_colored_card(
        randoms,
        card_type="ATTACK",
        card_blizz_randomizer=card_blizz_randomizer,
        runtime_card_pools=runtime_card_pools,
        previous_card_id=str(first_attack.get("card_id")) if first_attack is not None else None,
    )
    if second_attack is not None:
        cards.append(second_attack)
    first_skill = _roll_merchant_colored_card(
        randoms,
        card_type="SKILL",
        card_blizz_randomizer=card_blizz_randomizer,
        runtime_card_pools=runtime_card_pools,
    )
    if first_skill is not None:
        cards.append(first_skill)
    second_skill = _roll_merchant_colored_card(
        randoms,
        card_type="SKILL",
        card_blizz_randomizer=card_blizz_randomizer,
        runtime_card_pools=runtime_card_pools,
        previous_card_id=str(first_skill.get("card_id")) if first_skill is not None else None,
    )
    if second_skill is not None:
        cards.append(second_skill)
    first_power = _roll_merchant_colored_card(
        randoms,
        card_type="POWER",
        card_blizz_randomizer=card_blizz_randomizer,
        runtime_card_pools=runtime_card_pools,
    )
    if first_power is not None:
        cards.append(first_power)
    return cards


def _roll_merchant_colored_card(
    randoms: NativeRandomSet,
    *,
    card_type: str,
    card_blizz_randomizer: int,
    runtime_card_pools: dict[str, list[str]] | None = None,
    previous_card_id: str | None = None,
) -> dict[str, object] | None:
    for _ in range(64):
        candidate = _generate_colored_shop_card(
            randoms,
            card_type=card_type,
            card_blizz_randomizer=card_blizz_randomizer,
            reject_ids=set(),
            runtime_card_pools=runtime_card_pools,
        )
        if candidate is None:
            return None
        if str(candidate.get("color") or "") == "COLORLESS":
            continue
        if previous_card_id is not None and str(candidate.get("card_id")) == previous_card_id:
            continue
        return candidate
    return None


def _draw_unique(rng, pool: list[str], used: set[str]) -> str:
    available = [item for item in pool if item not in used]
    choice = available[int(rng.random(len(available) - 1))]
    used.add(choice)
    return choice


def _java_round_positive(value: float) -> int:
    return int(float(value) + 0.5)


def _price_with_jitter(
    rng,
    base_price: float,
    *,
    lower: float,
    upper: float,
    rounding: str = "floor",
) -> int:
    value = float(base_price) * float(rng.random(lower, upper))
    if rounding == "round":
        # libGDX MathUtils.round is floor(value + 0.5) for positive prices.
        return _java_round_positive(value)
    return int(value)


def _apply_shop_price_modifiers(
    base_price: float | int,
    *,
    ascension_level: int = 0,
    owned_relic_ids: set[str] | None = None,
) -> int:
    price = _java_round_positive(float(base_price))
    relic_ids = set(owned_relic_ids or ())
    if ascension_level >= 16:
        price = _java_round_positive(float(price) * 1.1)
    if "The Courier" in relic_ids:
        price = _java_round_positive(float(price) * 0.8)
    if "Membership Card" in relic_ids:
        price = _java_round_positive(float(price) * 0.5)
    return int(price)


def _generate_colored_shop_card(
    randoms: NativeRandomSet,
    *,
    card_type: str,
    card_blizz_randomizer: int,
    reject_ids: set[str],
    runtime_card_pools: dict[str, list[str]] | None = None,
    draw_rng_stream: str = "card",
) -> dict[str, object] | None:
    rarity = _roll_shop_card_rarity(randoms, card_blizz_randomizer=card_blizz_randomizer)
    card_id = _draw_card_from_filtered_pool(
        randoms,
        card_type=card_type,
        color="CLASS",
        rarity=rarity,
        reject_ids=reject_ids,
        runtime_card_pools=runtime_card_pools,
        draw_rng_stream=draw_rng_stream,
    )
    if card_id is None:
        return None
    return make_card(card_id, uuid=f"shop-{card_id}-{len(reject_ids)}")


def _generate_colorless_shop_card(
    randoms: NativeRandomSet,
    *,
    rarity: str,
    exclude: set[str],
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> dict[str, object] | None:
    pools = runtime_card_pools or card_pools()
    pool = sorted(card_id for card_id in pools.get(f"COLORLESS_{rarity}", []) if card_id not in exclude)
    if not pool:
        return None
    card_id = pool[int(randoms.stream("card").random(len(pool) - 1))]
    exclude.add(card_id)
    return make_card(card_id, uuid=f"shop-{card_id}-{len(exclude)}")


def _draw_card_from_filtered_pool(
    randoms: NativeRandomSet,
    *,
    card_type: str,
    color: str,
    rarity: str,
    reject_ids: set[str],
    runtime_card_pools: dict[str, list[str]] | None = None,
    draw_rng_stream: str = "card",
) -> str | None:
    pools = runtime_card_pools or card_pools()
    if color == "CLASS":
        # The runtime surface still exposes RED_* as the backward-compatible
        # class-card alias, and many higher-level call sites mutate that alias.
        # Prefer it here so shop generation tracks the effective runtime pool.
        rarity_pool_key = f"RED_{rarity}" if f"RED_{rarity}" in pools else f"CLASS_{rarity}"
    elif color == "COLORLESS":
        rarity_pool_key = f"COLORLESS_{rarity}"
    else:
        rarity_pool_key = f"{color}_{rarity}"
    pool = sorted(
        card_id
        for card_id in pools.get(rarity_pool_key, [])
        if card_id not in reject_ids
        and card_catalog()[card_id].type == card_type
    )
    if not pool and card_type == "POWER":
        fallback_rarity = "RARE" if rarity == "UNCOMMON" else "UNCOMMON"
        if color == "CLASS":
            fallback_pool_key = (
                f"RED_{fallback_rarity}" if f"RED_{fallback_rarity}" in pools else f"CLASS_{fallback_rarity}"
            )
        elif color == "COLORLESS":
            fallback_pool_key = f"COLORLESS_{fallback_rarity}"
        else:
            fallback_pool_key = f"{color}_{fallback_rarity}"
        pool = sorted(
            card_id
            for card_id in pools.get(fallback_pool_key, [])
            if card_id not in reject_ids
            and card_catalog()[card_id].type == card_type
        )
    if not pool:
        return None
    card_id = pool[int(randoms.stream(draw_rng_stream).random(len(pool) - 1))]
    return card_id


def _roll_shop_card_rarity(randoms: NativeRandomSet, *, card_blizz_randomizer: int = 0) -> str:
    rules = shop_rules()
    roll = int(randoms.stream("card").random(99)) + int(card_blizz_randomizer)
    if roll < rules.card_rare_chance:
        return "RARE"
    if roll < rules.card_rare_chance + rules.card_uncommon_chance:
        return "UNCOMMON"
    return "COMMON"


def _roll_shop_relic_tier(randoms: NativeRandomSet) -> str:
    rules = shop_rules()
    roll = int(randoms.stream("merchant").random(99))
    if roll < rules.relic_common_cutoff:
        return "COMMON"
    if roll < rules.relic_uncommon_cutoff:
        return "UNCOMMON"
    return "RARE"


def generate_shop_replacement_card(
    randoms: NativeRandomSet,
    *,
    purchased_card: dict[str, Any],
    existing_cards: list[dict[str, Any]],
    act: int | str = 1,
    dungeon_id: str | None = None,
    card_blizz_randomizer: int = 0,
    ascension_level: int = 0,
    owned_relic_ids: set[str] | None = None,
    runtime_card_pools: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    rules = shop_rules()
    dungeon_key = str(dungeon_id) if dungeon_id is not None else act
    relic_ids = set(owned_relic_ids or ())
    del existing_cards
    if str(purchased_card.get("color") or "") == "COLORLESS":
        rarity = "RARE" if randoms.stream("merchant").random_boolean(act_chances(dungeon_key).colorless_rare_chance) else "UNCOMMON"
        replacement = _generate_colorless_shop_card(
            randoms,
            rarity=rarity,
            exclude=set(),
            runtime_card_pools=runtime_card_pools,
        )
    else:
        replacement = _generate_colored_shop_card(
            randoms,
            card_type=str(purchased_card.get("type") or "SKILL"),
            card_blizz_randomizer=card_blizz_randomizer,
            reject_ids=set(),
            runtime_card_pools=runtime_card_pools,
            # STS ShopScreen uses rollRarity() for rarity, then
            # getCardFromPool(..., useRng=false) for the actual card.
            # That selection uses MathUtils.random, not cardRng.
            draw_rng_stream="misc",
        )
    if replacement is None:
        return dict(purchased_card)
    base_price = _price_with_jitter(
        randoms.stream("merchant"),
        card_price_for_rarity(str(replacement.get("rarity") or ""))
        * (rules.colorless_price_bump if str(replacement.get("color") or "") == "COLORLESS" else 1.0),
        lower=1.0 - rules.card_price_jitter,
        upper=1.0 + rules.card_price_jitter,
    )
    replacement["price"] = _apply_shop_price_modifiers(
        base_price,
        ascension_level=ascension_level,
        owned_relic_ids=relic_ids,
    )
    replacement.pop("on_sale", None)
    return replacement


def generate_shop_replacement_relic(
    randoms: NativeRandomSet,
    *,
    player_class: str = "IRONCLAD",
    floor_num: int | None = None,
    act: int | str = 1,
    dungeon_id: str | None = None,
    ascension_level: int = 0,
    owned_relic_ids: set[str] | None = None,
    relic_drawer: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, object]:
    rules = shop_rules()
    act_num = act_for_dungeon_id(dungeon_id) if dungeon_id is not None else (int(act) if str(act).isdigit() else None)
    relic_ids = set(owned_relic_ids or ())
    draw_relic = relic_drawer or (
        lambda tier: draw_random_relic_end(
            randoms,
            tier,
            character=player_class,
            floor_num=floor_num,
            current_room_type="ShopRoom",
            owned_relic_ids=relic_ids,
            act=act_num,
        )
    )
    replacement = draw_relic(_roll_shop_relic_tier(randoms))
    replacement["price"] = _apply_shop_price_modifiers(
        _price_with_jitter(
            randoms.stream("merchant"),
            price_for_relic_tier(str(replacement["tier"])),
            lower=1.0 - rules.relic_price_jitter,
            upper=1.0 + rules.relic_price_jitter,
            rounding="round",
        ),
        ascension_level=ascension_level,
        owned_relic_ids=relic_ids,
    )
    return replacement


def generate_shop_replacement_potion(
    randoms: NativeRandomSet,
    *,
    shop_index: int,
    player_class: str = "IRONCLAD",
    dungeon_id: str | None = None,
    ascension_level: int = 0,
    owned_relic_ids: set[str] | None = None,
) -> dict[str, object]:
    del dungeon_id
    rules = shop_rules()
    relic_ids = set(owned_relic_ids or ())
    potion = roll_random_potion(randoms, player_class=player_class)
    potion["price"] = _apply_shop_price_modifiers(
        _price_with_jitter(
            randoms.stream("merchant"),
            int(potion.get("price") or 0),
            lower=1.0 - rules.potion_price_jitter,
            upper=1.0 + rules.potion_price_jitter,
            rounding="round",
        ),
        ascension_level=ascension_level,
        owned_relic_ids=relic_ids,
    )
    potion["shop_index"] = int(shop_index)
    return potion
