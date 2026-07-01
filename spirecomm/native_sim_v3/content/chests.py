from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

CHEST_PATHS = {
    "SmallChest": sts_source_path("rewards/chests/SmallChest.java"),
    "MediumChest": sts_source_path("rewards/chests/MediumChest.java"),
    "LargeChest": sts_source_path("rewards/chests/LargeChest.java"),
}
INT_ASSIGNMENT_PATTERN = re.compile(r"this\.(COMMON_CHANCE|UNCOMMON_CHANCE|RARE_CHANCE|GOLD_CHANCE|GOLD_AMT) = (\d+);")


@dataclass(frozen=True, slots=True)
class ChestDef:
    chest_type: str
    common_chance: int
    uncommon_chance: int
    rare_chance: int
    gold_chance: int
    gold_amount: int
    source_path: str


@lru_cache(maxsize=1)
def chest_catalog() -> dict[str, ChestDef]:
    catalog: dict[str, ChestDef] = {}
    for chest_type, path in CHEST_PATHS.items():
        text = path.read_text(encoding="utf-8")
        assignments = {name: int(value) for name, value in INT_ASSIGNMENT_PATTERN.findall(text)}
        required = {"COMMON_CHANCE", "UNCOMMON_CHANCE", "RARE_CHANCE", "GOLD_CHANCE", "GOLD_AMT"}
        missing = required - set(assignments)
        if missing:
            raise ValueError(f"missing chest constants {sorted(missing)!r} in {path}")
        catalog[chest_type] = ChestDef(
            chest_type=chest_type,
            common_chance=assignments["COMMON_CHANCE"],
            uncommon_chance=assignments["UNCOMMON_CHANCE"],
            rare_chance=assignments["RARE_CHANCE"],
            gold_chance=assignments["GOLD_CHANCE"],
            gold_amount=assignments["GOLD_AMT"],
            source_path=str(path),
        )
    return catalog


def chest_def(chest_type: str) -> ChestDef:
    return chest_catalog()[str(chest_type)]
