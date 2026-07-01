from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

POTION_HELPER_SOURCE = sts_source_path("helpers/PotionHelper.java")
ABSTRACT_DUNGEON_SOURCE = sts_source_path("dungeons/AbstractDungeon.java")
ABSTRACT_ROOM_SOURCE = sts_source_path("rooms/AbstractRoom.java")

POTION_CHANCE_PATTERN = re.compile(r"public static int (POTION_COMMON_CHANCE|POTION_UNCOMMON_CHANCE) = (\d+);")
CARD_BLIZZ_ASSIGNMENT_PATTERN = re.compile(r"cardBlizz(Randomizer|StartOffset|Growth|MaxOffset)\s*=\s*(-?\d+);")
BLIZZARD_POTION_MOD_PATTERN = re.compile(r"private static final int BLIZZARD_POTION_MOD_AMT = (\d+);")
POST_COMBAT_POTION_BASE_PATTERN = re.compile(r"chance\s*=\s*(\d+);")
POST_COMBAT_POTION_REWARD_CAP_PATTERN = re.compile(r"this\.rewards\.size\(\)\s*>=\s*(\d+)")


@dataclass(frozen=True, slots=True)
class PotionRollRules:
    common_chance: int
    uncommon_chance: int
    source_path: str

    @property
    def rare_chance(self) -> int:
        return max(0, 100 - int(self.common_chance) - int(self.uncommon_chance))


@dataclass(frozen=True, slots=True)
class CardBlizzRules:
    start_offset: int
    growth: int
    max_offset: int
    source_path: str


@dataclass(frozen=True, slots=True)
class PostCombatPotionRules:
    base_chance: int
    white_beast_chance: int
    reward_cap: int
    blizzard_mod_amount: int
    source_path: str


def _extract_method_body(text: str, signature_pattern: str, source_path: Path) -> str:
    declaration = re.search(signature_pattern, text)
    if declaration is None:
        raise ValueError(f"could not locate method pattern {signature_pattern!r} in {source_path}")
    brace_start = text.find("{", declaration.end())
    if brace_start == -1:
        raise ValueError(f"could not locate body for method pattern {signature_pattern!r} in {source_path}")
    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index]
    raise ValueError(f"unterminated method body for pattern {signature_pattern!r} in {source_path}")


@lru_cache(maxsize=1)
def potion_roll_rules() -> PotionRollRules:
    text = POTION_HELPER_SOURCE.read_text(encoding="utf-8")
    assignments = {name: int(value) for name, value in POTION_CHANCE_PATTERN.findall(text)}
    required = {"POTION_COMMON_CHANCE", "POTION_UNCOMMON_CHANCE"}
    missing = required - set(assignments)
    if missing:
        raise ValueError(f"missing potion chance constants {sorted(missing)!r} in {POTION_HELPER_SOURCE}")
    return PotionRollRules(
        common_chance=assignments["POTION_COMMON_CHANCE"],
        uncommon_chance=assignments["POTION_UNCOMMON_CHANCE"],
        source_path=str(POTION_HELPER_SOURCE),
    )


@lru_cache(maxsize=1)
def card_blizz_rules() -> CardBlizzRules:
    text = ABSTRACT_DUNGEON_SOURCE.read_text(encoding="utf-8")
    assignments = {name: int(value) for name, value in CARD_BLIZZ_ASSIGNMENT_PATTERN.findall(text)}
    required = {"StartOffset", "Growth", "MaxOffset"}
    missing = required - set(assignments)
    if missing:
        raise ValueError(f"missing cardBlizz constants {sorted(missing)!r} in {ABSTRACT_DUNGEON_SOURCE}")
    return CardBlizzRules(
        start_offset=assignments["StartOffset"],
        growth=assignments["Growth"],
        max_offset=assignments["MaxOffset"],
        source_path=str(ABSTRACT_DUNGEON_SOURCE),
    )


@lru_cache(maxsize=1)
def post_combat_potion_rules() -> PostCombatPotionRules:
    text = ABSTRACT_ROOM_SOURCE.read_text(encoding="utf-8")
    body = _extract_method_body(
        text,
        r"\bpublic\s+void\s+addPotionToRewards\s*\(\s*\)",
        ABSTRACT_ROOM_SOURCE,
    )
    positive_chances = sorted({int(value) for value in POST_COMBAT_POTION_BASE_PATTERN.findall(body) if int(value) > 0})
    reward_cap_match = POST_COMBAT_POTION_REWARD_CAP_PATTERN.search(body)
    blizzard_mod_match = BLIZZARD_POTION_MOD_PATTERN.search(text)
    if reward_cap_match is None or blizzard_mod_match is None or len(positive_chances) < 2:
        raise ValueError(f"could not parse post-combat potion rules from {ABSTRACT_ROOM_SOURCE}")
    return PostCombatPotionRules(
        base_chance=positive_chances[0],
        white_beast_chance=positive_chances[-1],
        reward_cap=int(reward_cap_match.group(1)),
        blizzard_mod_amount=int(blizzard_mod_match.group(1)),
        source_path=str(ABSTRACT_ROOM_SOURCE),
    )
