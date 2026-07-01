from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

MONSTER_ROOT = sts_source_path("monsters")

ID_PATTERN = re.compile(r'public static final String ID = "([^"]+)";')
NAME_PATTERN = re.compile(r'public static final String NAME = ([A-Za-z0-9_]+)\.monsterStrings\.NAME;')
CLASS_PATTERN = re.compile(r"class\s+([A-Za-z0-9_]+)\s+extends\s+AbstractMonster")
SET_HP_PATTERN = re.compile(r"this\.setHp\((\d+),\s*(\d+)\);")


@dataclass(frozen=True, slots=True)
class MonsterDef:
    monster_id: str
    class_name: str
    area: str
    hp_ranges: tuple[tuple[int, int], ...]
    source_path: str


def _parse_monster_file(path: Path) -> MonsterDef | None:
    text = path.read_text(encoding="utf-8")
    id_match = ID_PATTERN.search(text)
    class_match = CLASS_PATTERN.search(text)
    if not id_match or not class_match:
        return None
    hp_ranges = tuple((int(low), int(high)) for low, high in SET_HP_PATTERN.findall(text))
    return MonsterDef(
        monster_id=id_match.group(1),
        class_name=class_match.group(1),
        area=path.parent.name,
        hp_ranges=hp_ranges,
        source_path=str(path),
    )


@lru_cache(maxsize=1)
def monster_catalog() -> dict[str, MonsterDef]:
    catalog: dict[str, MonsterDef] = {}
    for path in sorted(MONSTER_ROOT.glob("*/*.java")):
        monster = _parse_monster_file(path)
        if monster is not None:
            catalog[monster.monster_id] = monster
    return catalog


def monster_ids_for_area(area: str) -> list[str]:
    return sorted(monster_id for monster_id, monster in monster_catalog().items() if monster.area == area)
