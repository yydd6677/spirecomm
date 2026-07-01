from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

ABSTRACT_CARD_SOURCE = sts_source_path("cards/AbstractCard.java")
ABSTRACT_RELIC_SOURCE = sts_source_path("relics/AbstractRelic.java")
ABSTRACT_POTION_SOURCE = sts_source_path("potions/AbstractPotion.java")
ABSTRACT_DUNGEON_SOURCE = sts_source_path("dungeons/AbstractDungeon.java")

CASE_RETURN_PATTERN = re.compile(r"case ([A-Z_]+): \{.*?return (-?\d+);", re.S)
RARE_RATE_PATTERN = re.compile(r"rareRate = (\d+);")
UNCOMMON_THRESHOLD_PATTERN = re.compile(r"if \(roll < (\d+)\) \{\s*return AbstractCard\.CardRarity\.UNCOMMON;", re.S)


@dataclass(frozen=True, slots=True)
class RewardRarityRules:
    rare_roll_threshold: int
    uncommon_roll_threshold: int
    source_path: str

    @property
    def rare_chance(self) -> int:
        return int(self.rare_roll_threshold)

    @property
    def uncommon_chance(self) -> int:
        return int(self.uncommon_roll_threshold) - int(self.rare_roll_threshold)


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


def _parse_price_cases(source_path: Path, method_name: str) -> dict[str, int]:
    body = _extract_method_body(source_path.read_text(encoding="utf-8"), method_name, source_path)
    return {case_name: int(value) for case_name, value in CASE_RETURN_PATTERN.findall(body)}


@lru_cache(maxsize=1)
def card_price_by_rarity() -> dict[str, int]:
    return _parse_price_cases(ABSTRACT_CARD_SOURCE, "getPrice")


@lru_cache(maxsize=1)
def relic_price_by_tier() -> dict[str, int]:
    return _parse_price_cases(ABSTRACT_RELIC_SOURCE, "getPrice")


@lru_cache(maxsize=1)
def potion_price_by_rarity() -> dict[str, int]:
    return _parse_price_cases(ABSTRACT_POTION_SOURCE, "getPrice")


@lru_cache(maxsize=1)
def reward_rarity_rules() -> RewardRarityRules:
    body = _extract_method_body(ABSTRACT_DUNGEON_SOURCE.read_text(encoding="utf-8"), "getCardRarityFallback", ABSTRACT_DUNGEON_SOURCE)
    rare_rate_match = RARE_RATE_PATTERN.search(body)
    uncommon_threshold_match = UNCOMMON_THRESHOLD_PATTERN.search(body)
    if rare_rate_match is None or uncommon_threshold_match is None:
        raise ValueError("could not parse AbstractDungeon.getCardRarityFallback thresholds from source")
    return RewardRarityRules(
        rare_roll_threshold=int(rare_rate_match.group(1)),
        uncommon_roll_threshold=int(uncommon_threshold_match.group(1)),
        source_path=str(ABSTRACT_DUNGEON_SOURCE),
    )


def card_price_for_rarity(rarity: str) -> int:
    return int(card_price_by_rarity().get(str(rarity), 0))


def relic_price_for_tier(tier: str) -> int:
    return int(relic_price_by_tier().get(str(tier), 150))


def potion_price_for_rarity(rarity: str) -> int:
    return int(potion_price_by_rarity().get(str(rarity), 999))
