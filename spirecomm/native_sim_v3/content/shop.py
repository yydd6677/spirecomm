from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

MERCHANT_SOURCE = sts_source_path("shop/Merchant.java")
SHOP_SCREEN_SOURCE = sts_source_path("shop/ShopScreen.java")
SHOP_ROOM_SOURCE = sts_source_path("rooms/ShopRoom.java")

MERCHANT_COLORED_SLOT_PATTERN = re.compile(
    r"c = AbstractDungeon\.getCardFromPool\(AbstractDungeon\.rollRarity\(\), AbstractCard\.CardType\.(ATTACK|SKILL|POWER), true\)\.makeCopy\(\);"
    r".*?this\.cards1\.add\(c\);",
    re.S,
)
MERCHANT_COLORLESS_SLOT_PATTERN = re.compile(
    r"getColorlessCardFromPool\(AbstractCard\.CardRarity\.(UNCOMMON|RARE)\)"
)
FLOAT_CONSTANT_PATTERN = re.compile(r"private static final float ([A-Z_]+) = ([0-9.]+)f;")
INT_PURGE_ASSIGNMENT_PATTERN = re.compile(r"public static int purgeCost = (\d+);")
INT_PURGE_RAMP_PATTERN = re.compile(r"private static final int PURGE_COST_RAMP = (\d+);")
ROLL_RELIC_TIER_PATTERN = re.compile(
    r"if \(roll < (\d+)\) \{\s*return AbstractRelic\.RelicTier\.COMMON;\s*\}\s*"
    r"if \(roll < (\d+)\) \{\s*return AbstractRelic\.RelicTier\.UNCOMMON;\s*\}\s*"
    r"return AbstractRelic\.RelicTier\.RARE;",
    re.S,
)


@dataclass(frozen=True, slots=True)
class ShopRules:
    colored_card_types: tuple[str, ...]
    colorless_card_rarities: tuple[str, ...]
    card_rare_chance: int
    card_uncommon_chance: int
    relic_common_cutoff: int
    relic_uncommon_cutoff: int
    card_price_jitter: float
    relic_price_jitter: float
    potion_price_jitter: float
    colorless_price_bump: float
    purge_cost: int
    purge_cost_ramp: int
    merchant_source_path: str
    shop_screen_source_path: str
    shop_room_source_path: str


def _extract_method_body(text: str, method_name: str, source_path: Path) -> str:
    declaration = re.search(
        rf"\b(?:public|protected|private)\s+[^{{;\n]*\b{re.escape(method_name)}\s*\(",
        text,
    )
    if declaration is None:
        raise ValueError(f"could not locate method {method_name!r} in {source_path}")
    brace_start = text.find("{", declaration.end())
    if brace_start == -1:
        raise ValueError(f"could not locate body for method {method_name!r} in {source_path}")
    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index]
    raise ValueError(f"unterminated method body for {method_name!r} in {source_path}")


@lru_cache(maxsize=1)
def shop_rules() -> ShopRules:
    merchant_text = MERCHANT_SOURCE.read_text(encoding="utf-8")
    shop_screen_text = SHOP_SCREEN_SOURCE.read_text(encoding="utf-8")
    shop_room_text = SHOP_ROOM_SOURCE.read_text(encoding="utf-8")
    float_constants = {
        name: float(value)
        for name, value in FLOAT_CONSTANT_PATTERN.findall(shop_screen_text)
    }
    purge_cost_match = INT_PURGE_ASSIGNMENT_PATTERN.search(shop_screen_text)
    purge_ramp_match = INT_PURGE_RAMP_PATTERN.search(shop_screen_text)
    relic_roll_match = ROLL_RELIC_TIER_PATTERN.search(
        _extract_method_body(shop_screen_text, "rollRelicTier", SHOP_SCREEN_SOURCE)
    )
    if purge_cost_match is None or purge_ramp_match is None or relic_roll_match is None:
        raise ValueError("could not parse ShopScreen core constants from source")
    colored_card_types = tuple(MERCHANT_COLORED_SLOT_PATTERN.findall(merchant_text))
    colorless_card_rarities = tuple(MERCHANT_COLORLESS_SLOT_PATTERN.findall(merchant_text)[:2])
    rare_match = re.search(r"this\.baseRareCardChance = (\d+);", shop_room_text)
    uncommon_match = re.search(r"this\.baseUncommonCardChance = (\d+);", shop_room_text)
    if (
        len(colored_card_types) != 5
        or len(colorless_card_rarities) != 2
        or rare_match is None
        or uncommon_match is None
    ):
        raise ValueError("could not parse Merchant card slot structure from source")
    return ShopRules(
        colored_card_types=colored_card_types,
        colorless_card_rarities=colorless_card_rarities,
        card_rare_chance=int(rare_match.group(1)),
        card_uncommon_chance=int(uncommon_match.group(1)),
        relic_common_cutoff=int(relic_roll_match.group(1)),
        relic_uncommon_cutoff=int(relic_roll_match.group(2)),
        card_price_jitter=float_constants["CARD_PRICE_JITTER"],
        relic_price_jitter=float_constants["RELIC_PRICE_JITTER"],
        potion_price_jitter=float_constants["POTION_PRICE_JITTER"],
        colorless_price_bump=float_constants["COLORLESS_PRICE_BUMP"],
        purge_cost=int(purge_cost_match.group(1)),
        purge_cost_ramp=int(purge_ramp_match.group(1)),
        merchant_source_path=str(MERCHANT_SOURCE),
        shop_screen_source_path=str(SHOP_SCREEN_SOURCE),
        shop_room_source_path=str(SHOP_ROOM_SOURCE),
    )
