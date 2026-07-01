from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from spirecomm.native_sim.randoms import StsRandom, java_collections_shuffle


@dataclass(frozen=True)
class RelicDef:
    relic_id: str
    name: str
    rarity: str = "COMMON"


RELIC_LIBRARY: dict[str, RelicDef] = {
    "Burning Blood": RelicDef("Burning Blood", "Burning Blood", "STARTER"),
    "Akabeko": RelicDef("Akabeko", "Akabeko", "COMMON"),
    "Anchor": RelicDef("Anchor", "Anchor"),
    "Ancient Tea Set": RelicDef("Ancient Tea Set", "Ancient Tea Set", "COMMON"),
    "Art of War": RelicDef("Art of War", "Art of War", "COMMON"),
    "Bag of Marbles": RelicDef("Bag of Marbles", "Bag of Marbles", "COMMON"),
    "Bag of Preparation": RelicDef("Bag of Preparation", "Bag of Preparation"),
    "Blood Vial": RelicDef("Blood Vial", "Blood Vial"),
    "Bronze Scales": RelicDef("Bronze Scales", "Bronze Scales", "COMMON"),
    "Ceramic Fish": RelicDef("Ceramic Fish", "Ceramic Fish", "COMMON"),
    "Centennial Puzzle": RelicDef("Centennial Puzzle", "Centennial Puzzle", "COMMON"),
    "Dream Catcher": RelicDef("Dream Catcher", "Dream Catcher", "COMMON"),
    "Happy Flower": RelicDef("Happy Flower", "Happy Flower", "COMMON"),
    "Juzu Bracelet": RelicDef("Juzu Bracelet", "Juzu Bracelet", "COMMON"),
    "Lantern": RelicDef("Lantern", "Lantern"),
    "Maw Bank": RelicDef("Maw Bank", "Maw Bank", "COMMON"),
    "Meal Ticket": RelicDef("Meal Ticket", "Meal Ticket", "COMMON"),
    "Nunchaku": RelicDef("Nunchaku", "Nunchaku", "COMMON"),
    "Oddly Smooth Stone": RelicDef("Oddly Smooth Stone", "Oddly Smooth Stone", "COMMON"),
    "Omamori": RelicDef("Omamori", "Omamori", "COMMON"),
    "Orichalcum": RelicDef("Orichalcum", "Orichalcum", "COMMON"),
    "Pen Nib": RelicDef("Pen Nib", "Pen Nib", "COMMON"),
    "Potion Belt": RelicDef("Potion Belt", "Potion Belt", "COMMON"),
    "Preserved Insect": RelicDef("Preserved Insect", "Preserved Insect", "COMMON"),
    "Regal Pillow": RelicDef("Regal Pillow", "Regal Pillow", "COMMON"),
    "Smiling Mask": RelicDef("Smiling Mask", "Smiling Mask", "COMMON"),
    "Strawberry": RelicDef("Strawberry", "Strawberry", "COMMON"),
    "The Boot": RelicDef("The Boot", "The Boot", "COMMON"),
    "Tiny Chest": RelicDef("Tiny Chest", "Tiny Chest", "COMMON"),
    "Toy Ornithopter": RelicDef("Toy Ornithopter", "Toy Ornithopter", "COMMON"),
    "Vajra": RelicDef("Vajra", "Vajra", "COMMON"),
    "War Paint": RelicDef("War Paint", "War Paint", "COMMON"),
    "Whetstone": RelicDef("Whetstone", "Whetstone", "COMMON"),
    "Blue Candle": RelicDef("Blue Candle", "Blue Candle", "UNCOMMON"),
    "Bottled Flame": RelicDef("Bottled Flame", "Bottled Flame", "UNCOMMON"),
    "Bottled Lightning": RelicDef("Bottled Lightning", "Bottled Lightning", "UNCOMMON"),
    "Bottled Tornado": RelicDef("Bottled Tornado", "Bottled Tornado", "UNCOMMON"),
    "Darkstone Periapt": RelicDef("Darkstone Periapt", "Darkstone Periapt", "UNCOMMON"),
    "Clockwork Souvenir": RelicDef("Clockwork Souvenir", "Clockwork Souvenir", "SHOP"),
    "Eternal Feather": RelicDef("Eternal Feather", "Eternal Feather", "UNCOMMON"),
    "Gremlin Horn": RelicDef("Gremlin Horn", "Gremlin Horn", "UNCOMMON"),
    "Horn Cleat": RelicDef("Horn Cleat", "Horn Cleat", "UNCOMMON"),
    "Ink Bottle": RelicDef("Ink Bottle", "Ink Bottle", "UNCOMMON"),
    "Kunai": RelicDef("Kunai", "Kunai", "UNCOMMON"),
    "Letter Opener": RelicDef("Letter Opener", "Letter Opener", "UNCOMMON"),
    "Matryoshka": RelicDef("Matryoshka", "Matryoshka", "UNCOMMON"),
    "Meat on the Bone": RelicDef("Meat on the Bone", "Meat on the Bone", "UNCOMMON"),
    "Membership Card": RelicDef("Membership Card", "Membership Card", "SHOP"),
    "Medical Kit": RelicDef("Medical Kit", "Medical Kit", "SHOP"),
    "Mercury Hourglass": RelicDef("Mercury Hourglass", "Mercury Hourglass", "UNCOMMON"),
    "Molten Egg": RelicDef("Molten Egg", "Molten Egg", "UNCOMMON"),
    "Frozen Egg": RelicDef("Frozen Egg", "Frozen Egg", "UNCOMMON"),
    "Mummified Hand": RelicDef("Mummified Hand", "Mummified Hand", "UNCOMMON"),
    "Ornamental Fan": RelicDef("Ornamental Fan", "Ornamental Fan", "UNCOMMON"),
    "Pantograph": RelicDef("Pantograph", "Pantograph", "UNCOMMON"),
    "Pear": RelicDef("Pear", "Pear", "UNCOMMON"),
    "Question Card": RelicDef("Question Card", "Question Card", "UNCOMMON"),
    "Paper Frog": RelicDef("Paper Frog", "Paper Phrog", "UNCOMMON"),
    "Red Skull": RelicDef("Red Skull", "Red Skull", "COMMON"),
    "Shuriken": RelicDef("Shuriken", "Shuriken", "UNCOMMON"),
    "Singing Bowl": RelicDef("Singing Bowl", "Singing Bowl", "UNCOMMON"),
    "Sundial": RelicDef("Sundial", "Sundial", "UNCOMMON"),
    "Strike Dummy": RelicDef("Strike Dummy", "Strike Dummy", "UNCOMMON"),
    "The Courier": RelicDef("The Courier", "The Courier", "UNCOMMON"),
    "Toxic Egg": RelicDef("Toxic Egg", "Toxic Egg", "UNCOMMON"),
    "White Beast Statue": RelicDef("White Beast Statue", "White Beast Statue", "UNCOMMON"),
    "Bird Faced Urn": RelicDef("Bird Faced Urn", "Bird-Faced Urn", "RARE"),
    "Calipers": RelicDef("Calipers", "Calipers", "RARE"),
    "Captain's Wheel": RelicDef("Captain's Wheel", "Captain's Wheel", "RARE"),
    "Champion Belt": RelicDef("Champion Belt", "Champion Belt", "RARE"),
    "Charon's Ashes": RelicDef("Charon's Ashes", "Charon's Ashes", "RARE"),
    "Dead Branch": RelicDef("Dead Branch", "Dead Branch", "RARE"),
    "Du-Vu Doll": RelicDef("Du-Vu Doll", "Du-Vu Doll", "RARE"),
    "Fossilized Helix": RelicDef("Fossilized Helix", "Fossilized Helix", "RARE"),
    "Gambling Chip": RelicDef("Gambling Chip", "Gambling Chip", "RARE"),
    "Ginger": RelicDef("Ginger", "Ginger", "RARE"),
    "Girya": RelicDef("Girya", "Girya", "RARE"),
    "Ice Cream": RelicDef("Ice Cream", "Ice Cream", "RARE"),
    "Incense Burner": RelicDef("Incense Burner", "Incense Burner", "RARE"),
    "Lizard Tail": RelicDef("Lizard Tail", "Lizard Tail", "RARE"),
    "Mango": RelicDef("Mango", "Mango", "RARE"),
    "Old Coin": RelicDef("Old Coin", "Old Coin", "RARE"),
    "Peace Pipe": RelicDef("Peace Pipe", "Peace Pipe", "RARE"),
    "Pocketwatch": RelicDef("Pocketwatch", "Pocketwatch", "RARE"),
    "Prayer Wheel": RelicDef("Prayer Wheel", "Prayer Wheel", "RARE"),
    "Self-Forming Clay": RelicDef("Self-Forming Clay", "Self-Forming Clay", "UNCOMMON"),
    "Shovel": RelicDef("Shovel", "Shovel", "RARE"),
    "Stone Calendar": RelicDef("Stone Calendar", "Stone Calendar", "RARE"),
    "Thread and Needle": RelicDef("Thread and Needle", "Thread and Needle", "RARE"),
    "Torii": RelicDef("Torii", "Torii", "RARE"),
    "Tungsten Rod": RelicDef("Tungsten Rod", "Tungsten Rod", "RARE"),
    "Turnip": RelicDef("Turnip", "Turnip", "RARE"),
    "Unceasing Top": RelicDef("Unceasing Top", "Unceasing Top", "RARE"),
    "Wing Boots": RelicDef("Wing Boots", "Wing Boots", "RARE"),
    "Golden Idol": RelicDef("Golden Idol", "Golden Idol", "EVENT"),
    "Bloody Idol": RelicDef("Bloody Idol", "Bloody Idol", "EVENT"),
    "Odd Mushroom": RelicDef("Odd Mushroom", "Odd Mushroom", "EVENT"),
    "Brimstone": RelicDef("Brimstone", "Brimstone", "SHOP"),
    "Cauldron": RelicDef("Cauldron", "Cauldron", "SHOP"),
    "Chemical X": RelicDef("Chemical X", "Chemical X", "SHOP"),
    "Dolly's Mirror": RelicDef("Dolly's Mirror", "Dolly's Mirror", "SHOP"),
    "Face Of Cleric": RelicDef("Face Of Cleric", "Face of Cleric", "EVENT"),
    "Hand Drill": RelicDef("Hand Drill", "Hand Drill", "SHOP"),
    "Lee's Waffle": RelicDef("Lee's Waffle", "Lee's Waffle", "SHOP"),
    "Magic Flower": RelicDef("Magic Flower", "Magic Flower", "RARE"),
    "Necronomicon": RelicDef("Necronomicon", "Necronomicon", "EVENT"),
    "Enchiridion": RelicDef("Enchiridion", "Enchiridion", "EVENT"),
    "Nilry's Codex": RelicDef("Nilry's Codex", "Nilry's Codex", "EVENT"),
    "Orange Pellets": RelicDef("Orange Pellets", "Orange Pellets", "SHOP"),
    "Orrery": RelicDef("Orrery", "Orrery", "SHOP"),
    "Mutagenic Strength": RelicDef("Mutagenic Strength", "Mutagenic Strength", "EVENT"),
    "N'loth's Gift": RelicDef("N'loth's Gift", "N'loth's Gift", "EVENT"),
    "Sling of Courage": RelicDef("Sling of Courage", "Sling of Courage", "SHOP"),
    "Strange Spoon": RelicDef("Strange Spoon", "Strange Spoon", "SHOP"),
    "Toolbox": RelicDef("Toolbox", "Toolbox", "SHOP"),
    "Warped Tongs": RelicDef("Warped Tongs", "Warped Tongs", "EVENT"),
    "Red Mask": RelicDef("Red Mask", "Red Mask", "EVENT"),
    "Prismatic Shard": RelicDef("Prismatic Shard", "Prismatic Shard", "SHOP"),
    "Frozen Eye": RelicDef("Frozen Eye", "Frozen Eye", "SHOP"),
    "The Abacus": RelicDef("The Abacus", "The Abacus", "SHOP"),
    "Runic Cube": RelicDef("Runic Cube", "Runic Cube", "BOSS"),
    "Astrolabe": RelicDef("Astrolabe", "Astrolabe", "BOSS"),
    "Black Star": RelicDef("Black Star", "Black Star", "BOSS"),
    "Black Blood": RelicDef("Black Blood", "Black Blood", "BOSS"),
    "Busted Crown": RelicDef("Busted Crown", "Busted Crown", "BOSS"),
    "Calling Bell": RelicDef("Calling Bell", "Calling Bell", "BOSS"),
    "Coffee Dripper": RelicDef("Coffee Dripper", "Coffee Dripper", "BOSS"),
    "Cursed Key": RelicDef("Cursed Key", "Cursed Key", "BOSS"),
    "Ectoplasm": RelicDef("Ectoplasm", "Ectoplasm", "BOSS"),
    "Empty Cage": RelicDef("Empty Cage", "Empty Cage", "BOSS"),
    "Fusion Hammer": RelicDef("Fusion Hammer", "Fusion Hammer", "BOSS"),
    "Mark of Pain": RelicDef("Mark of Pain", "Mark of Pain", "BOSS"),
    "Pandora's Box": RelicDef("Pandora's Box", "Pandora's Box", "BOSS"),
    "Philosopher's Stone": RelicDef("Philosopher's Stone", "Philosopher's Stone", "BOSS"),
    "Runic Dome": RelicDef("Runic Dome", "Runic Dome", "BOSS"),
    "Runic Pyramid": RelicDef("Runic Pyramid", "Runic Pyramid", "BOSS"),
    "Slaver's Collar": RelicDef("Slaver's Collar", "Slaver's Collar", "BOSS"),
    "Sacred Bark": RelicDef("Sacred Bark", "Sacred Bark", "BOSS"),
    "Snecko Eye": RelicDef("Snecko Eye", "Snecko Eye", "BOSS"),
    "Sozu": RelicDef("Sozu", "Sozu", "BOSS"),
    "Tiny House": RelicDef("Tiny House", "Tiny House", "BOSS"),
    "Velvet Choker": RelicDef("Velvet Choker", "Velvet Choker", "BOSS"),
    "Circlet": RelicDef("Circlet", "Circlet", "SPECIAL"),
    "Red Circlet": RelicDef("Red Circlet", "Red Circlet", "SPECIAL"),
}

IRONCLAD_COMMON_RELIC_POOL_ORDER = (
    "Whetstone", "The Boot", "Blood Vial", "Meal Ticket", "Pen Nib", "Akabeko", "Lantern",
    "Regal Pillow", "Bag of Preparation", "Ancient Tea Set", "Smiling Mask", "Potion Belt",
    "Preserved Insect", "Omamori", "Maw Bank", "Art of War", "Toy Ornithopter",
    "Ceramic Fish", "Vajra", "Centennial Puzzle", "Strawberry", "Happy Flower",
    "Oddly Smooth Stone", "War Paint", "Bronze Scales", "Juzu Bracelet", "Dream Catcher",
    "Nunchaku", "Tiny Chest", "Orichalcum", "Anchor", "Bag of Marbles", "Red Skull",
)

IRONCLAD_UNCOMMON_RELIC_POOL_ORDER = (
    "Bottled Tornado", "Sundial", "Kunai", "Pear", "Blue Candle", "Eternal Feather",
    "Strike Dummy", "Singing Bowl", "Matryoshka", "Ink Bottle", "The Courier",
    "Frozen Egg", "Ornamental Fan", "Bottled Lightning", "Gremlin Horn", "Horn Cleat",
    "Toxic Egg", "Letter Opener", "Question Card", "Bottled Flame", "Shuriken",
    "Molten Egg", "Meat on the Bone", "Darkstone Periapt", "Mummified Hand", "Pantograph",
    "White Beast Statue", "Mercury Hourglass", "Self-Forming Clay", "Paper Frog",
)

IRONCLAD_RARE_RELIC_POOL_ORDER = (
    "Ginger", "Old Coin", "Bird Faced Urn", "Unceasing Top", "Torii", "Stone Calendar",
    "Shovel", "Wing Boots", "Thread and Needle", "Turnip", "Ice Cream", "Calipers",
    "Lizard Tail", "Prayer Wheel", "Girya", "Dead Branch", "Du-Vu Doll", "Pocketwatch",
    "Mango", "Incense Burner", "Gambling Chip", "Peace Pipe", "Captain's Wheel",
    "Fossilized Helix", "Tungsten Rod", "Magic Flower", "Charon's Ashes", "Champion Belt",
)

IRONCLAD_SHOP_RELIC_POOL_ORDER = (
    "Sling of Courage", "Hand Drill", "Toolbox", "Chemical X", "Lee's Waffle", "Orrery",
    "Dolly's Mirror", "Orange Pellets", "Prismatic Shard", "Clockwork Souvenir",
    "Frozen Eye", "The Abacus", "Medical Kit", "Cauldron", "Strange Spoon",
    "Membership Card", "Brimstone",
)

IRONCLAD_BOSS_RELIC_POOL_ORDER = (
    "Fusion Hammer", "Velvet Choker", "Runic Dome", "Slaver's Collar", "Snecko Eye",
    "Pandora's Box", "Cursed Key", "Busted Crown", "Ectoplasm", "Tiny House", "Sozu",
    "Philosopher's Stone", "Astrolabe", "Black Star", "Sacred Bark", "Empty Cage",
    "Runic Pyramid", "Calling Bell", "Coffee Dripper", "Black Blood", "Mark of Pain",
    "Runic Cube",
)

BOSS_RELIC_IDS = set(IRONCLAD_BOSS_RELIC_POOL_ORDER)

POOL_ORDER_BY_TIER = {
    "COMMON": IRONCLAD_COMMON_RELIC_POOL_ORDER,
    "UNCOMMON": IRONCLAD_UNCOMMON_RELIC_POOL_ORDER,
    "RARE": IRONCLAD_RARE_RELIC_POOL_ORDER,
    "SHOP": IRONCLAD_SHOP_RELIC_POOL_ORDER,
    "BOSS": IRONCLAD_BOSS_RELIC_POOL_ORDER,
}

IRONCLAD_RELIC_UNLOCK_BUNDLES: tuple[tuple[str, ...], ...] = (
    ("Omamori", "Prayer Wheel", "Shovel"),
    ("Blue Candle", "Dead Branch", "Singing Bowl"),
    ("Du-Vu Doll", "Smiling Mask", "Tiny Chest"),
    ("Art of War", "The Courier", "Pandora's Box"),
)


def make_relic(relic_id: str, *, price: int = 0, counter: int = -1) -> dict[str, Any]:
    relic = RELIC_LIBRARY[relic_id]
    return {
        "relic_id": relic.relic_id,
        "id": relic.relic_id,
        "name": relic.name,
        "counter": counter,
        "price": price,
        "tier": relic.rarity,
    }


def ironclad_locked_relic_ids(unlock_level: int = 5) -> set[str]:
    unlocked_bundles = max(0, min(len(IRONCLAD_RELIC_UNLOCK_BUNDLES), int(unlock_level)))
    locked: set[str] = set()
    for bundle in IRONCLAD_RELIC_UNLOCK_BUNDLES[unlocked_bundles:]:
        locked.update(bundle)
    return locked


def init_ironclad_relic_pools(rng: StsRandom, *, locked_relic_ids: set[str] | None = None) -> dict[str, list[str]]:
    """Create lightspeed/official-style persistent relic pools.

    The game shuffles every tier once at run initialization using
    ``java.util.Random(relicRng.nextLong())``. ``nextLong`` mutates RNG state
    but does not increment the save-file counter in lightspeed.
    """

    locked_relic_ids = locked_relic_ids or set()
    pools = {
        tier: [relic_id for relic_id in order if relic_id not in locked_relic_ids]
        for tier, order in POOL_ORDER_BY_TIER.items()
    }
    for tier in ("COMMON", "UNCOMMON", "RARE", "SHOP", "BOSS"):
        java_collections_shuffle(pools[tier], rng.next_long())
    return pools


def _has_less_than_two_campfire_relics(owned: set[str]) -> bool:
    return sum(1 for relic_id in ("Girya", "Peace Pipe", "Shovel") if relic_id in owned) < 2


def relic_can_spawn(
    relic_id: str,
    *,
    owned: set[str] | None = None,
    floor: int = 1,
    shop_room: bool = False,
    deck: list[Any] | None = None,
) -> bool:
    owned = owned or set()
    deck = deck or []
    if relic_id == "Bottled Flame":
        return any(
            getattr(card, "card_def", None) is not None
            and getattr(card.card_def, "card_type", None) == "ATTACK"
            and getattr(card.card_def, "rarity", None) != "BASIC"
            for card in deck
        )
    if relic_id == "Bottled Lightning":
        return any(
            getattr(card, "card_def", None) is not None
            and getattr(card.card_def, "card_type", None) == "SKILL"
            and getattr(card.card_def, "rarity", None) != "BASIC"
            for card in deck
        )
    if relic_id == "Bottled Tornado":
        return any(
            getattr(card, "card_def", None) is not None
            and getattr(card.card_def, "card_type", None) == "POWER"
            for card in deck
        )
    if relic_id == "Black Blood":
        return "Burning Blood" in owned
    if relic_id == "Tiny Chest":
        return floor <= 35
    if relic_id in {"Wing Boots", "Matryoshka"}:
        return floor <= 40
    if relic_id in {"Girya", "Peace Pipe", "Shovel"}:
        return floor < 48 and _has_less_than_two_campfire_relics(owned)
    if relic_id in {"Maw Bank", "Old Coin", "Smiling Mask"}:
        return floor <= 48 and not shop_room
    if relic_id in {
        "Ancient Tea Set", "Ceramic Fish", "Darkstone Periapt", "Dream Catcher",
        "Frozen Egg", "Juzu Bracelet", "Meal Ticket", "Meat on the Bone",
        "Molten Egg", "Omamori", "Potion Belt", "Prayer Wheel", "Question Card",
        "Regal Pillow", "Singing Bowl", "The Courier", "Toxic Egg",
    }:
        return floor <= 48
    if relic_id == "Preserved Insect":
        return floor <= 52
    return True


def draw_relic_from_pool(
    pools: dict[str, list[str]],
    tier: str,
    *,
    owned: set[str] | None = None,
    floor: int = 1,
    shop_room: bool = False,
    from_front: bool = True,
    deck: list[Any] | None = None,
) -> dict[str, Any]:
    owned = owned or set()
    tier = tier.upper()
    if tier == "COMMON" and not pools.get("COMMON"):
        return draw_relic_from_pool(
            pools, "UNCOMMON", owned=owned, floor=floor, shop_room=shop_room, from_front=from_front, deck=deck
        )
    if tier == "UNCOMMON" and not pools.get("UNCOMMON"):
        return draw_relic_from_pool(
            pools, "RARE", owned=owned, floor=floor, shop_room=shop_room, from_front=from_front, deck=deck
        )
    if tier == "RARE" and not pools.get("RARE"):
        return make_relic("Circlet")
    if tier == "SHOP" and not pools.get("SHOP"):
        return draw_relic_from_pool(
            pools, "UNCOMMON", owned=owned, floor=floor, shop_room=shop_room, from_front=from_front, deck=deck
        )
    if tier == "BOSS" and not pools.get("BOSS"):
        return make_relic("Red Circlet")

    pool = pools[tier]
    relic_id = pool.pop(0 if from_front else -1)
    if relic_can_spawn(relic_id, owned=owned, floor=floor, shop_room=shop_room, deck=deck):
        return make_relic(relic_id)
    # The game discards an invalid front draw and retries from the back.
    return draw_relic_from_pool(
        pools, tier, owned=owned, floor=floor, shop_room=shop_room, from_front=False, deck=deck
    )


def _roll_percent(rng: random.Random) -> int:
    try:
        return int(rng.random(0, 99))
    except TypeError:
        return int(rng.random() * 100)


def _relic_candidates(
    owned: set[str] | None,
    *,
    rarities: set[str],
    include_ironclad_as_uncommon: bool = True,
) -> list[str]:
    owned = owned or set()
    candidates = []
    for relic_id, relic in RELIC_LIBRARY.items():
        if relic_id in owned or relic.rarity in {"STARTER", "BOSS", "EVENT"}:
            continue
        if relic.rarity in rarities:
            candidates.append(relic_id)
        elif include_ironclad_as_uncommon and relic.rarity == "IRONCLAD" and "UNCOMMON" in rarities:
            candidates.append(relic_id)
    return candidates


def roll_relic_of_tier(rng: random.Random, rarity: str, owned: set[str] | None = None) -> dict[str, Any]:
    owned = owned or set()
    candidates = _relic_candidates(owned, rarities={rarity})
    if not candidates:
        candidates = _relic_candidates(owned, rarities={"COMMON", "UNCOMMON", "RARE", "SHOP"})
    return make_relic(rng.choice(candidates))


def roll_relic(rng: random.Random, owned: set[str] | None = None) -> dict[str, Any]:
    owned = owned or set()
    roll = _roll_percent(rng)
    target_rarity = "COMMON" if roll < 50 else "UNCOMMON" if roll < 83 else "RARE"
    candidates = _relic_candidates(owned, rarities={target_rarity})
    if not candidates:
        candidates = _relic_candidates(owned, rarities={"COMMON", "UNCOMMON", "RARE", "SHOP"})
    return make_relic(rng.choice(candidates))


def roll_shop_only_relic(rng: random.Random, owned: set[str] | None = None) -> dict[str, Any]:
    owned = owned or set()
    candidates = [
        relic_id for relic_id, relic in RELIC_LIBRARY.items()
        if relic_id not in owned and relic.rarity == "SHOP"
    ]
    if not candidates:
        candidates = _relic_candidates(owned, rarities={"COMMON", "UNCOMMON", "RARE", "SHOP"})
    return make_relic(rng.choice(candidates))


def roll_shop_relic(rng: random.Random, owned: set[str] | None = None) -> dict[str, Any]:
    owned = owned or set()
    roll = _roll_percent(rng)
    if roll < 18:
        candidates = [
            relic_id for relic_id, relic in RELIC_LIBRARY.items()
            if relic_id not in owned and relic.rarity == "SHOP"
        ]
    else:
        candidates = _relic_candidates(owned, rarities={"COMMON", "UNCOMMON", "RARE"})
    if not candidates:
        candidates = _relic_candidates(owned, rarities={"COMMON", "UNCOMMON", "RARE", "SHOP"})
    return make_relic(rng.choice(candidates))


def roll_boss_relics(rng: random.Random, owned: set[str] | None = None, count: int = 3) -> list[dict[str, Any]]:
    owned = owned or set()
    candidates = [relic_id for relic_id in BOSS_RELIC_IDS if relic_id not in owned]
    rng.shuffle(candidates)
    return [make_relic(relic_id) for relic_id in candidates[:count]]
