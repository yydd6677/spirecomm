from __future__ import annotations

import random

from spirecomm.native_sim.schema import PotionDef, PotionInstance


POTION_LIBRARY: dict[str, PotionDef] = {
    "Fire Potion": PotionDef("Fire Potion", "Fire Potion", "COMMON", requires_target=True),
    "Explosive Potion": PotionDef("Explosive Potion", "Explosive Potion", "COMMON"),
    "Weak Potion": PotionDef("Weak Potion", "Weak Potion", "COMMON", requires_target=True),
    "Fear Potion": PotionDef("Fear Potion", "Fear Potion", "COMMON", requires_target=True),
    "Strength Potion": PotionDef("Strength Potion", "Strength Potion", "COMMON"),
    "Dexterity Potion": PotionDef("Dexterity Potion", "Dexterity Potion", "COMMON"),
    "Block Potion": PotionDef("Block Potion", "Block Potion", "COMMON"),
    "Energy Potion": PotionDef("Energy Potion", "Energy Potion", "COMMON"),
    "Swift Potion": PotionDef("Swift Potion", "Swift Potion", "COMMON"),
    "Blood Potion": PotionDef("Blood Potion", "Blood Potion", "COMMON"),
    "Flex Potion": PotionDef("Flex Potion", "Flex Potion", "COMMON"),
    "Attack Potion": PotionDef("Attack Potion", "Attack Potion", "COMMON"),
    "Skill Potion": PotionDef("Skill Potion", "Skill Potion", "COMMON"),
    "Power Potion": PotionDef("Power Potion", "Power Potion", "COMMON"),
    "Ancient Potion": PotionDef("Ancient Potion", "Ancient Potion", "UNCOMMON"),
    "Blessing of the Forge": PotionDef("Blessing of the Forge", "Blessing of the Forge", "COMMON"),
    "Colorless Potion": PotionDef("Colorless Potion", "Colorless Potion", "COMMON"),
    "Duplication Potion": PotionDef("Duplication Potion", "Duplication Potion", "UNCOMMON"),
    "Elixir Potion": PotionDef("ElixirPotion", "Elixir", "UNCOMMON"),
    "Essence of Steel": PotionDef("Essence of Steel", "Essence of Steel", "UNCOMMON"),
    "Speed Potion": PotionDef("Speed Potion", "Speed Potion", "COMMON"),
    "Steroid Potion": PotionDef("Steroid Potion", "Steroid Potion", "COMMON"),
    "Regen Potion": PotionDef("Regen Potion", "Regen Potion", "UNCOMMON"),
    "Liquid Bronze": PotionDef("Liquid Bronze", "Liquid Bronze", "UNCOMMON"),
    "Liquid Memories": PotionDef("Liquid Memories", "Liquid Memories", "UNCOMMON"),
    "Gambler's Brew": PotionDef("Gambler's Brew", "Gambler's Brew", "UNCOMMON"),
    "Entropic Brew": PotionDef("Entropic Brew", "Entropic Brew", "RARE"),
    "Fruit Juice": PotionDef("Fruit Juice", "Fruit Juice", "RARE"),
    "Heart of Iron": PotionDef("Heart of Iron", "Heart of Iron", "RARE"),
    "Distilled Chaos": PotionDef("Distilled Chaos", "Distilled Chaos", "UNCOMMON"),
    "Cultist Potion": PotionDef("Cultist Potion", "Cultist Potion", "RARE"),
    "Snecko Oil": PotionDef("Snecko Oil", "Snecko Oil", "RARE"),
    "Smoke Bomb": PotionDef("Smoke Bomb", "Smoke Bomb", "RARE"),
    "Fairy in a Bottle": PotionDef("FairyPotion", "Fairy in a Bottle", "RARE"),
}


IRONCLAD_POTION_POOL_ORDER = [
    "Blood Potion", "Elixir Potion", "Heart of Iron", "Block Potion", "Dexterity Potion",
    "Energy Potion", "Explosive Potion", "Fire Potion", "Strength Potion", "Swift Potion",
    "Weak Potion", "Fear Potion", "Attack Potion", "Skill Potion", "Power Potion",
    "Colorless Potion", "Flex Potion", "Speed Potion", "Blessing of the Forge",
    "Regen Potion", "Ancient Potion", "Liquid Bronze", "Gambler's Brew",
    "Essence of Steel", "Duplication Potion", "Distilled Chaos", "Liquid Memories",
    "Cultist Potion", "Fruit Juice", "Snecko Oil", "Fairy in a Bottle", "Smoke Bomb",
    "Entropic Brew",
]

POTION_POOL_BY_CLASS = {
    "IRONCLAD": IRONCLAD_POTION_POOL_ORDER,
}


def empty_potion_slots(count: int = 3) -> list[PotionInstance]:
    return [PotionInstance() for _ in range(count)]


def make_potion(potion_id: str, *, price: int = 0) -> PotionInstance:
    if potion_id == "FairyPotion":
        potion_id = "Fairy in a Bottle"
    return PotionInstance(POTION_LIBRARY[potion_id], price=price)


def _roll_percent(rng: random.Random) -> int:
    try:
        return int(rng.random(0, 99))
    except TypeError:
        return int(rng.random() * 100)


def get_random_potion(rng: random.Random, character_class: str = "IRONCLAD") -> PotionInstance:
    pool = POTION_POOL_BY_CLASS.get(character_class, IRONCLAD_POTION_POOL_ORDER)
    if not pool:
        pool = list(POTION_LIBRARY)
    return make_potion(pool[int(rng.random(len(pool) - 1))])


def roll_potion(rng: random.Random, *, limited: bool = False) -> PotionInstance:
    roll = _roll_percent(rng)
    rarity = "COMMON" if roll < 65 else "UNCOMMON" if roll < 90 else "RARE"
    pool = POTION_POOL_BY_CLASS.get("IRONCLAD", IRONCLAD_POTION_POOL_ORDER)
    if not pool:
        pool = list(POTION_LIBRARY)
    while True:
        potion_id = pool[int(rng.random(len(pool) - 1))]
        if POTION_LIBRARY[potion_id].rarity != rarity:
            continue
        if limited and potion_id == "Fruit Juice":
            continue
        return make_potion(potion_id)


def potions_to_spirecomm(potions: list[PotionInstance]) -> list[dict]:
    return [potion.to_spirecomm() for potion in potions]
