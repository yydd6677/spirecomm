from __future__ import annotations

from spirecomm.native_sim.schema import CardDef, CardInstance


def attack(name: str, rarity: str, cost: int, *, card_id: str | None = None, upgraded_cost: int | None = None, exhausts: bool = False, ethereal: bool = False, x_cost: bool = False, has_target: bool = True) -> CardDef:
    return CardDef(card_id or name, name, "ATTACK", rarity, cost, has_target=has_target, exhausts=exhausts, upgraded_cost=upgraded_cost, ethereal=ethereal, x_cost=x_cost)


def skill(name: str, rarity: str, cost: int, *, card_id: str | None = None, upgraded_cost: int | None = None, exhausts: bool = False, ethereal: bool = False, x_cost: bool = False, has_target: bool = False) -> CardDef:
    return CardDef(card_id or name, name, "SKILL", rarity, cost, has_target=has_target, upgraded_cost=upgraded_cost, exhausts=exhausts, ethereal=ethereal, x_cost=x_cost)


def power(name: str, rarity: str, cost: int, *, card_id: str | None = None, upgraded_cost: int | None = None) -> CardDef:
    return CardDef(card_id or name, name, "POWER", rarity, cost, upgraded_cost=upgraded_cost)


def status(name: str, *, card_id: str | None = None, ethereal: bool = False) -> CardDef:
    return CardDef(card_id or name, name, "STATUS", "SPECIAL", -2, ethereal=ethereal)


def curse(name: str, *, card_id: str | None = None) -> CardDef:
    return CardDef(card_id or name, name, "CURSE", "CURSE", -2)


CARD_LIBRARY: dict[str, CardDef] = {
    # Starter/basic
    "Strike_R": attack("Strike", "BASIC", 1, card_id="Strike_R"),
    "Defend_R": skill("Defend", "BASIC", 1, card_id="Defend_R"),
    "Bash": attack("Bash", "BASIC", 2),

    # Ironclad attacks
    "Anger": attack("Anger", "COMMON", 0),
    "Body Slam": attack("Body Slam", "COMMON", 1, upgraded_cost=0),
    "Clash": attack("Clash", "COMMON", 0),
    "Cleave": attack("Cleave", "COMMON", 1, card_id="Cleave", has_target=False),
    "Clothesline": attack("Clothesline", "COMMON", 2),
    "Headbutt": attack("Headbutt", "COMMON", 1),
    "Heavy Blade": attack("Heavy Blade", "COMMON", 2),
    "Iron Wave": attack("Iron Wave", "COMMON", 1),
    "Perfected Strike": attack("Perfected Strike", "COMMON", 2),
    "Pommel Strike": attack("Pommel Strike", "COMMON", 1),
    "Sword Boomerang": attack("Sword Boomerang", "COMMON", 1, has_target=False),
    "Thunderclap": attack("Thunderclap", "COMMON", 1, has_target=False),
    "Twin Strike": attack("Twin Strike", "COMMON", 1),
    "Wild Strike": attack("Wild Strike", "COMMON", 1),
    "Blood for Blood": attack("Blood for Blood", "UNCOMMON", 4, upgraded_cost=3),
    "Carnage": attack("Carnage", "UNCOMMON", 2, ethereal=True),
    "Dropkick": attack("Dropkick", "UNCOMMON", 1),
    "Hemokinesis": attack("Hemokinesis", "UNCOMMON", 1),
    "Pummel": attack("Pummel", "UNCOMMON", 1, exhausts=True),
    "Rampage": attack("Rampage", "UNCOMMON", 1),
    "Reckless Charge": attack("Reckless Charge", "UNCOMMON", 0),
    "Searing Blow": attack("Searing Blow", "UNCOMMON", 2),
    "Sever Soul": attack("Sever Soul", "UNCOMMON", 2),
    "Uppercut": attack("Uppercut", "UNCOMMON", 2),
    "Whirlwind": attack("Whirlwind", "UNCOMMON", -1, x_cost=True, has_target=False),
    "Bludgeon": attack("Bludgeon", "RARE", 3),
    "Feed": attack("Feed", "RARE", 1, exhausts=True),
    "Fiend Fire": attack("Fiend Fire", "RARE", 2, exhausts=True),
    "Immolate": attack("Immolate", "RARE", 2, has_target=False),
    "Reaper": attack("Reaper", "RARE", 2, exhausts=True, has_target=False),

    # Ironclad skills
    "Armaments": skill("Armaments", "COMMON", 1),
    "Flex": skill("Flex", "COMMON", 0),
    "Havoc": skill("Havoc", "COMMON", 1, upgraded_cost=0),
    "Shrug It Off": skill("Shrug It Off", "COMMON", 1),
    "True Grit": skill("True Grit", "COMMON", 1),
    "Warcry": skill("Warcry", "COMMON", 0, exhausts=True),
    "Battle Trance": skill("Battle Trance", "UNCOMMON", 0),
    "Bloodletting": skill("Bloodletting", "UNCOMMON", 0),
    "Burning Pact": skill("Burning Pact", "UNCOMMON", 1),
    "Disarm": skill("Disarm", "UNCOMMON", 1, exhausts=True, has_target=True),
    "Dual Wield": skill("Dual Wield", "UNCOMMON", 1),
    "Entrench": skill("Entrench", "UNCOMMON", 2, upgraded_cost=1),
    "Flame Barrier": skill("Flame Barrier", "UNCOMMON", 2),
    "Ghostly Armor": skill("Ghostly Armor", "UNCOMMON", 1, ethereal=True),
    "Infernal Blade": skill("Infernal Blade", "UNCOMMON", 1, upgraded_cost=0, exhausts=True),
    "Intimidate": skill("Intimidate", "UNCOMMON", 0, exhausts=True),
    "Power Through": skill("Power Through", "UNCOMMON", 1),
    "Rage": skill("Rage", "UNCOMMON", 0, upgraded_cost=0),
    "Second Wind": skill("Second Wind", "UNCOMMON", 1),
    "Seeing Red": skill("Seeing Red", "UNCOMMON", 1, upgraded_cost=0, exhausts=True),
    "Sentinel": skill("Sentinel", "UNCOMMON", 1),
    "Shockwave": skill("Shockwave", "UNCOMMON", 2, exhausts=True),
    "Spot Weakness": skill("Spot Weakness", "UNCOMMON", 1, has_target=True),
    "Double Tap": skill("Double Tap", "RARE", 1),
    "Exhume": skill("Exhume", "RARE", 1, upgraded_cost=0, exhausts=True),
    "Impervious": skill("Impervious", "RARE", 2, exhausts=True),
    "Limit Break": skill("Limit Break", "RARE", 1, exhausts=True),
    "Offering": skill("Offering", "RARE", 0, exhausts=True),

    # Ironclad powers
    "Barricade": power("Barricade", "RARE", 3, upgraded_cost=2),
    "Berserk": power("Berserk", "RARE", 0),
    "Brutality": power("Brutality", "RARE", 0),
    "Combust": power("Combust", "UNCOMMON", 1),
    "Corruption": power("Corruption", "RARE", 3, upgraded_cost=2),
    "Dark Embrace": power("Dark Embrace", "UNCOMMON", 2, upgraded_cost=1),
    "Demon Form": power("Demon Form", "RARE", 3),
    "Evolve": power("Evolve", "UNCOMMON", 1),
    "Feel No Pain": power("Feel No Pain", "UNCOMMON", 1),
    "Fire Breathing": power("Fire Breathing", "UNCOMMON", 1),
    "Inflame": power("Inflame", "UNCOMMON", 1),
    "Juggernaut": power("Juggernaut", "RARE", 2),
    "Metallicize": power("Metallicize", "UNCOMMON", 1),
    "Rupture": power("Rupture", "UNCOMMON", 1),

    # Status/curse cards needed by Ironclad and Act 1.
    "Wound": status("Wound"),
    "Burn": status("Burn"),
    "Dazed": status("Dazed", ethereal=True),
    "Slimed": CardDef("Slimed", "Slimed", "STATUS", "SPECIAL", 1, exhausts=True),
    "Void": status("Void"),
    "AscendersBane": CardDef("AscendersBane", "AscendersBane", "CURSE", "CURSE", -2, ethereal=True),
    "Clumsy": CardDef("Clumsy", "Clumsy", "CURSE", "CURSE", -2, ethereal=True),
    "CurseOfTheBell": curse("Curse of the Bell", card_id="CurseOfTheBell"),
    "Decay": curse("Decay"),
    "Doubt": curse("Doubt"),
    "Injury": curse("Injury"),
    "Necronomicurse": curse("Necronomicurse"),
    "Normality": curse("Normality"),
    "Pain": curse("Pain"),
    "Parasite": curse("Parasite"),
    "Regret": curse("Regret"),
    "Shame": curse("Shame"),
    "Writhe": curse("Writhe"),
    "J.A.X.": skill("J.A.X.", "SPECIAL", 0, card_id="J.A.X."),

    # Colorless cards Ironclad can see from shops, events, potions, and relics.
    "Bandage Up": skill("Bandage Up", "UNCOMMON", 0, exhausts=True),
    "Blind": skill("Blind", "UNCOMMON", 0, has_target=True),
    "Dark Shackles": skill("Dark Shackles", "UNCOMMON", 0, exhausts=True, has_target=True),
    "Deep Breath": skill("Deep Breath", "UNCOMMON", 0),
    "Discovery": skill("Discovery", "UNCOMMON", 1, exhausts=True),
    "Dramatic Entrance": attack("Dramatic Entrance", "UNCOMMON", 0, exhausts=True, has_target=False),
    "Enlightenment": skill("Enlightenment", "UNCOMMON", 0),
    "Finesse": skill("Finesse", "UNCOMMON", 0),
    "Flash of Steel": attack("Flash of Steel", "UNCOMMON", 0),
    "Forethought": skill("Forethought", "UNCOMMON", 0),
    "Good Instincts": skill("Good Instincts", "UNCOMMON", 0),
    "Impatience": skill("Impatience", "UNCOMMON", 0),
    "Jack Of All Trades": skill("Jack Of All Trades", "UNCOMMON", 0, exhausts=True),
    "Madness": skill("Madness", "UNCOMMON", 1, upgraded_cost=0, exhausts=True),
    "Mind Blast": attack("Mind Blast", "UNCOMMON", 2, upgraded_cost=1),
    "Panacea": skill("Panacea", "UNCOMMON", 0, exhausts=True),
    "Panic Button": skill("Panic Button", "UNCOMMON", 0, exhausts=True),
    "Purity": skill("Purity", "UNCOMMON", 0, exhausts=True),
    "Swift Strike": attack("Swift Strike", "UNCOMMON", 0),
    "Trip": skill("Trip", "UNCOMMON", 0, upgraded_cost=0, has_target=True),
    "Apotheosis": skill("Apotheosis", "RARE", 2, upgraded_cost=1, exhausts=True),
    "Chrysalis": skill("Chrysalis", "RARE", 2, exhausts=True),
    "Hand of Greed": attack("Hand of Greed", "RARE", 2, card_id="HandOfGreed"),
    "Magnetism": power("Magnetism", "RARE", 2, upgraded_cost=1),
    "Master of Strategy": skill("Master of Strategy", "RARE", 0, exhausts=True),
    "Mayhem": power("Mayhem", "RARE", 2, upgraded_cost=1),
    "Metamorphosis": skill("Metamorphosis", "RARE", 2, exhausts=True),
    "Panache": power("Panache", "RARE", 0),
    "Sadistic Nature": power("Sadistic Nature", "RARE", 0),
    "Secret Technique": skill("Secret Technique", "RARE", 0, exhausts=True),
    "Secret Weapon": skill("Secret Weapon", "RARE", 0, exhausts=True),
    "The Bomb": skill("The Bomb", "RARE", 2),
    "Thinking Ahead": skill("Thinking Ahead", "RARE", 0, exhausts=True),
    "Transmutation": skill("Transmutation", "RARE", -1, exhausts=True, x_cost=True),
    "Violence": skill("Violence", "RARE", 0, exhausts=True),
    # Event/special colorless cards.
    "Apparition": skill("Apparition", "SPECIAL", 1, exhausts=True, ethereal=True),
    "Bite": attack("Bite", "SPECIAL", 1),
    "Ritual Dagger": attack("Ritual Dagger", "SPECIAL", 1, exhausts=True),
}

COLORLESS_CARD_ID_ORDER: tuple[str, ...] = (
    "Bandage Up",
    "Blind",
    "Dark Shackles",
    "Deep Breath",
    "Discovery",
    "Dramatic Entrance",
    "Enlightenment",
    "Finesse",
    "Flash of Steel",
    "Forethought",
    "Good Instincts",
    "Impatience",
    "Jack Of All Trades",
    "Madness",
    "Mind Blast",
    "Panacea",
    "Panic Button",
    "Purity",
    "Swift Strike",
    "Trip",
    "Apotheosis",
    "Chrysalis",
    "Hand of Greed",
    "Magnetism",
    "Master of Strategy",
    "Mayhem",
    "Metamorphosis",
    "Panache",
    "Sadistic Nature",
    "Secret Technique",
    "Secret Weapon",
    "The Bomb",
    "Thinking Ahead",
    "Transmutation",
    "Violence",
)

COLORLESS_CARD_IDS = set(COLORLESS_CARD_ID_ORDER)

IRONCLAD_RARITY_CARD_IDS: dict[str, tuple[str, ...]] = {
    "COMMON": (
        "Anger",
        "Cleave",
        "Warcry",
        "Flex",
        "Iron Wave",
        "Body Slam",
        "True Grit",
        "Shrug It Off",
        "Clash",
        "Thunderclap",
        "Pommel Strike",
        "Twin Strike",
        "Clothesline",
        "Armaments",
        "Havoc",
        "Headbutt",
        "Wild Strike",
        "Heavy Blade",
        "Perfected Strike",
        "Sword Boomerang",
    ),
    "UNCOMMON": (
        "Spot Weakness",
        "Inflame",
        "Power Through",
        "Dual Wield",
        "Infernal Blade",
        "Reckless Charge",
        "Hemokinesis",
        "Intimidate",
        "Blood for Blood",
        "Flame Barrier",
        "Pummel",
        "Burning Pact",
        "Metallicize",
        "Shockwave",
        "Rampage",
        "Sever Soul",
        "Whirlwind",
        "Combust",
        "Dark Embrace",
        "Seeing Red",
        "Disarm",
        "Feel No Pain",
        "Rage",
        "Entrench",
        "Sentinel",
        "Battle Trance",
        "Searing Blow",
        "Second Wind",
        "Rupture",
        "Bloodletting",
        "Carnage",
        "Dropkick",
        "Fire Breathing",
        "Ghostly Armor",
        "Uppercut",
        "Evolve",
    ),
    "RARE": (
        "Immolate",
        "Offering",
        "Exhume",
        "Reaper",
        "Brutality",
        "Juggernaut",
        "Impervious",
        "Berserk",
        "Fiend Fire",
        "Barricade",
        "Corruption",
        "Limit Break",
        "Feed",
        "Bludgeon",
        "Demon Form",
        "Double Tap",
    ),
}

IRONCLAD_TYPE_RARITY_CARD_IDS: dict[str, dict[str, tuple[str, ...]]] = {
    "ATTACK": {
        "COMMON": (
            "Anger",
            "Body Slam",
            "Clash",
            "Cleave",
            "Clothesline",
            "Headbutt",
            "Heavy Blade",
            "Iron Wave",
            "Perfected Strike",
            "Pommel Strike",
            "Sword Boomerang",
            "Thunderclap",
            "Twin Strike",
            "Wild Strike",
        ),
        "UNCOMMON": (
            "Blood for Blood",
            "Carnage",
            "Dropkick",
            "Hemokinesis",
            "Pummel",
            "Rampage",
            "Reckless Charge",
            "Searing Blow",
            "Sever Soul",
            "Uppercut",
            "Whirlwind",
        ),
        "RARE": (
            "Bludgeon",
            "Feed",
            "Fiend Fire",
            "Immolate",
            "Reaper",
        ),
    },
    "SKILL": {
        "COMMON": (
            "Armaments",
            "Flex",
            "Havoc",
            "Shrug It Off",
            "True Grit",
            "Warcry",
        ),
        "UNCOMMON": (
            "Battle Trance",
            "Bloodletting",
            "Burning Pact",
            "Disarm",
            "Dual Wield",
            "Entrench",
            "Flame Barrier",
            "Ghostly Armor",
            "Infernal Blade",
            "Intimidate",
            "Power Through",
            "Rage",
            "Second Wind",
            "Seeing Red",
            "Sentinel",
            "Shockwave",
            "Spot Weakness",
        ),
        "RARE": (
            "Double Tap",
            "Exhume",
            "Impervious",
            "Limit Break",
            "Offering",
        ),
    },
    "POWER": {
        "COMMON": (),
        "UNCOMMON": (
            "Combust",
            "Dark Embrace",
            "Evolve",
            "Feel No Pain",
            "Fire Breathing",
            "Inflame",
            "Metallicize",
            "Rupture",
        ),
        "RARE": (
            "Barricade",
            "Berserk",
            "Brutality",
            "Corruption",
            "Demon Form",
            "Juggernaut",
        ),
    },
}

IRONCLAD_UNLOCK_BUNDLES: tuple[tuple[str, ...], ...] = (
    ("Heavy Blade", "Spot Weakness", "Limit Break"),
    ("Wild Strike", "Evolve", "Immolate"),
    ("Havoc", "Sentinel", "Exhume"),
)


def ironclad_cards_by_rarity(rarity: str) -> list[CardDef]:
    return [CARD_LIBRARY[card_id] for card_id in IRONCLAD_RARITY_CARD_IDS.get(rarity, ())]


def ironclad_locked_card_ids(unlock_level: int = 5) -> set[str]:
    unlocked_bundles = max(0, min(3, int(unlock_level)))
    locked: set[str] = set()
    for bundle in IRONCLAD_UNLOCK_BUNDLES[unlocked_bundles:]:
        locked.update(bundle)
    return locked


def ironclad_card_pool(
    *,
    card_type: str | None = None,
    rarity: str | None = None,
    exclude_ids: set[str] | None = None,
) -> list[CardDef]:
    rarities = (rarity,) if rarity else ("COMMON", "UNCOMMON", "RARE")
    pool = [card for item in rarities for card in ironclad_cards_by_rarity(item)]
    if exclude_ids:
        pool = [card for card in pool if card.card_id not in exclude_ids]
    if card_type is not None:
        pool = [card for card in pool if card.card_type == card_type]
    return pool


def ironclad_type_rarity_card_pool(
    card_type: str,
    rarity: str,
    *,
    exclude_ids: set[str] | None = None,
) -> list[CardDef]:
    pool = [CARD_LIBRARY[card_id] for card_id in IRONCLAD_TYPE_RARITY_CARD_IDS.get(card_type, {}).get(rarity, ())]
    if exclude_ids:
        pool = [card for card in pool if card.card_id not in exclude_ids]
    return pool


def ironclad_reward_pool() -> list[CardDef]:
    return ironclad_card_pool()


def colorless_reward_pool() -> list[CardDef]:
    return [CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER if CARD_LIBRARY[card_id].rarity in {"UNCOMMON", "RARE"}]


def reward_rarity(rng) -> str:
    roll = rng.random()
    if roll < 0.60:
        return "COMMON"
    if roll < 0.92:
        return "UNCOMMON"
    return "RARE"


def roll_card_reward(rng, count: int = 3) -> list[CardInstance]:
    pool = ironclad_reward_pool()
    options: list[CardInstance] = []
    seen: set[str] = set()
    for _ in range(count):
        rarity = reward_rarity(rng)
        rarity_pool = [card for card in pool if card.rarity == rarity]
        if not rarity_pool:
            rarity_pool = pool
        chosen = rng.choice(rarity_pool)
        while chosen.card_id in seen:
            chosen = rng.choice(rarity_pool)
        seen.add(chosen.card_id)
        options.append(CardInstance(chosen, uuid=f"reward-{chosen.card_id}-{len(options)}"))
    return options


def roll_colorless_card(rng) -> CardInstance:
    pool = colorless_reward_pool()
    rarity = "UNCOMMON" if rng.random() < 0.72 else "RARE"
    rarity_pool = [card for card in pool if card.rarity == rarity] or pool
    chosen = rng.choice(rarity_pool)
    return CardInstance(chosen, uuid=f"colorless-{chosen.card_id}")


def make_card(card_id: str, upgrades: int = 0, uuid: str = "") -> CardInstance:
    if card_id == "HandOfGreed":
        card_id = "Hand of Greed"
    try:
        card_def = CARD_LIBRARY[card_id]
    except KeyError as exc:
        raise KeyError(f"unsupported native simulator card: {card_id}") from exc
    return CardInstance(card_def=card_def, upgrades=upgrades, uuid=uuid)


def clone_card(card: CardInstance, *, reset_cost_for_turn: bool = True) -> CardInstance:
    return CardInstance(
        card_def=card.card_def,
        upgrades=card.upgrades,
        misc=card.misc,
        uuid=card.uuid,
        cost_for_combat=card.cost_for_combat,
        cost_for_turn=None if reset_cost_for_turn else card.cost_for_turn,
        free_to_play_once=False if reset_cost_for_turn else card.free_to_play_once,
    )


def starter_deck() -> list[CardInstance]:
    deck: list[CardInstance] = []
    counter = 0
    for _ in range(5):
        deck.append(make_card("Strike_R", uuid=f"starter-{counter}"))
        counter += 1
    for _ in range(4):
        deck.append(make_card("Defend_R", uuid=f"starter-{counter}"))
        counter += 1
    deck.append(make_card("Bash", uuid=f"starter-{counter}"))
    return deck


def card_to_spirecomm(card: CardInstance, *, is_playable: bool = False) -> dict:
    base_cost = card.card_def.upgraded_cost if card.upgrades > 0 and card.card_def.upgraded_cost is not None else card.card_def.cost
    display_cost_for_turn = None
    if card.cost_for_turn is not None or card.cost_for_combat is not None:
        display_cost_for_turn = card.cost
    return {
        "card_id": card.card_id,
        "name": card.name,
        "type": card.card_def.card_type,
        "rarity": card.card_def.rarity,
        "cost": card.cost,
        "base_cost": base_cost,
        "cost_for_turn": display_cost_for_turn,
        "cost_for_combat": card.cost_for_combat,
        "free_to_play_once": card.free_to_play_once,
        "upgrades": card.upgrades,
        "misc": card.misc,
        "exhausts": card.card_def.exhausts,
        "has_target": card.card_def.has_target,
        "is_playable": is_playable,
        "uuid": card.uuid,
    }
