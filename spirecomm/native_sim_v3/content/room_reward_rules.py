from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

ABSTRACT_ROOM_SOURCE = sts_source_path("rooms/AbstractRoom.java")
MONSTER_ROOM_SOURCE = sts_source_path("rooms/MonsterRoom.java")
MONSTER_ROOM_ELITE_SOURCE = sts_source_path("rooms/MonsterRoomElite.java")

BASE_RARE_PATTERN = re.compile(r"public int baseRareCardChance = (\d+);")
BASE_UNCOMMON_PATTERN = re.compile(r"public int baseUncommonCardChance = (\d+);")
ELITE_BASE_RARE_PATTERN = re.compile(r"this\.baseRareCardChance = (\d+);")
ELITE_BASE_UNCOMMON_PATTERN = re.compile(r"this\.baseUncommonCardChance = (\d+);")
RELIC_COMMON_PATTERN = re.compile(r"if \(roll < (\d+)\) \{\s*return AbstractRelic\.RelicTier\.COMMON;\s*\}", re.S)
RELIC_RARE_GT_PATTERN = re.compile(r"if \(roll > (\d+)\) \{\s*return AbstractRelic\.RelicTier\.RARE;\s*\}", re.S)
MONSTER_GOLD_PATTERN = re.compile(r"this\.addGoldToRewards\(AbstractDungeon\.treasureRng\.random\((\d+), (\d+)\)\);")
ELITE_GOLD_PATTERN = re.compile(r"this\.addGoldToRewards\(AbstractDungeon\.treasureRng\.random\((\d+), (\d+)\)\);")


@dataclass(frozen=True, slots=True)
class RoomRewardRules:
    normal_rare_chance: int
    normal_uncommon_chance: int
    elite_rare_chance: int
    elite_uncommon_chance: int
    normal_relic_common_cutoff: int
    normal_relic_rare_gt: int
    elite_relic_common_cutoff: int
    elite_relic_rare_gt: int
    normal_gold_min: int
    normal_gold_max: int
    elite_gold_min: int
    elite_gold_max: int
    abstract_room_source_path: str
    monster_room_source_path: str
    monster_room_elite_source_path: str

    def card_rarity_thresholds(self, room_type: str) -> tuple[int, int]:
        room_name = str(room_type or "")
        if room_name == "MonsterRoomElite":
            return self.elite_rare_chance, self.elite_uncommon_chance
        return self.normal_rare_chance, self.normal_uncommon_chance

    def relic_tier_thresholds(self, room_type: str) -> tuple[int, int]:
        room_name = str(room_type or "")
        if room_name == "MonsterRoomElite":
            return self.elite_relic_common_cutoff, self.elite_relic_rare_gt
        return self.normal_relic_common_cutoff, self.normal_relic_rare_gt

    def gold_range(self, room_type: str) -> tuple[int, int]:
        room_name = str(room_type or "")
        if room_name == "MonsterRoomElite":
            return self.elite_gold_min, self.elite_gold_max
        return self.normal_gold_min, self.normal_gold_max


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
def room_reward_rules() -> RoomRewardRules:
    abstract_text = ABSTRACT_ROOM_SOURCE.read_text(encoding="utf-8")
    monster_text = MONSTER_ROOM_SOURCE.read_text(encoding="utf-8")
    elite_text = MONSTER_ROOM_ELITE_SOURCE.read_text(encoding="utf-8")
    abstract_update_body = _extract_method_body(abstract_text, "update", ABSTRACT_ROOM_SOURCE)
    monster_relic_body = _extract_method_body(monster_text, "returnRandomRelicTier", MONSTER_ROOM_SOURCE)
    elite_relic_body = _extract_method_body(elite_text, "returnRandomRelicTier", MONSTER_ROOM_ELITE_SOURCE)

    normal_rare_match = BASE_RARE_PATTERN.search(abstract_text)
    normal_uncommon_match = BASE_UNCOMMON_PATTERN.search(abstract_text)
    elite_rare_match = ELITE_BASE_RARE_PATTERN.search(elite_text)
    elite_uncommon_match = ELITE_BASE_UNCOMMON_PATTERN.search(elite_text)
    normal_common_cutoff_match = RELIC_COMMON_PATTERN.search(monster_relic_body)
    normal_rare_gt_match = RELIC_RARE_GT_PATTERN.search(monster_relic_body)
    elite_common_cutoff_match = RELIC_COMMON_PATTERN.search(elite_relic_body)
    elite_rare_gt_match = RELIC_RARE_GT_PATTERN.search(elite_relic_body)

    gold_matches = MONSTER_GOLD_PATTERN.findall(abstract_update_body)
    elite_gold_match = ELITE_GOLD_PATTERN.search(abstract_update_body)
    if not all(
        (
            normal_rare_match,
            normal_uncommon_match,
            elite_rare_match,
            elite_uncommon_match,
            normal_common_cutoff_match,
            normal_rare_gt_match,
            elite_common_cutoff_match,
            elite_rare_gt_match,
            elite_gold_match,
        )
    ):
        raise ValueError("could not parse room reward rules from source")
    if len(gold_matches) < 2:
        raise ValueError("could not parse monster and elite gold reward ranges from AbstractRoom.update")
    normal_gold_min, normal_gold_max = map(int, gold_matches[-1])
    elite_gold_min, elite_gold_max = map(int, gold_matches[-2])
    return RoomRewardRules(
        normal_rare_chance=int(normal_rare_match.group(1)),
        normal_uncommon_chance=int(normal_uncommon_match.group(1)),
        elite_rare_chance=int(elite_rare_match.group(1)),
        elite_uncommon_chance=int(elite_uncommon_match.group(1)),
        normal_relic_common_cutoff=int(normal_common_cutoff_match.group(1)),
        normal_relic_rare_gt=int(normal_rare_gt_match.group(1)),
        elite_relic_common_cutoff=int(elite_common_cutoff_match.group(1)),
        elite_relic_rare_gt=int(elite_rare_gt_match.group(1)),
        normal_gold_min=normal_gold_min,
        normal_gold_max=normal_gold_max,
        elite_gold_min=elite_gold_min,
        elite_gold_max=elite_gold_max,
        abstract_room_source_path=str(ABSTRACT_ROOM_SOURCE),
        monster_room_source_path=str(MONSTER_ROOM_SOURCE),
        monster_room_elite_source_path=str(MONSTER_ROOM_ELITE_SOURCE),
    )
