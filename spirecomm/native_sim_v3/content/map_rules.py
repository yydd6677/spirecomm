from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

ABSTRACT_DUNGEON_SOURCE = sts_source_path("dungeons/AbstractDungeon.java")

MAP_DIMENSION_PATTERN = re.compile(r"int (mapHeight|mapWidth|mapPathDensity) = (\d+);")
ROW_ASSIGN_PATTERN = re.compile(
    r"RoomTypeAssigner\.assignRowAsRoomType\(map\.get\((?P<row>.+?)\),\s*(?P<room>[A-Za-z0-9_]+)\.class\);"
)
MIMIC_BRANCH_PATTERN = re.compile(
    r'if \(Settings\.isEndless && player\.hasBlight\("MimicInfestation"\)\) \{\s*'
    r'RoomTypeAssigner\.assignRowAsRoomType\(map\.get\((?P<endless_row>\d+)\),\s*(?P<endless_room>[A-Za-z0-9_]+)\.class\);\s*'
    r'\} else \{\s*'
    r'RoomTypeAssigner\.assignRowAsRoomType\(map\.get\((?P<normal_row>\d+)\),\s*(?P<normal_room>[A-Za-z0-9_]+)\.class\);',
    re.S,
)


@dataclass(frozen=True, slots=True)
class MapRuleDef:
    map_height: int
    map_width: int
    map_path_density: int
    first_row_room_class: str
    last_row_room_class: str
    special_row_index: int
    special_row_room_class: str
    endless_mimic_row_index: int
    endless_mimic_room_class: str
    source_path: str


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
def map_rules() -> MapRuleDef:
    source = ABSTRACT_DUNGEON_SOURCE.read_text(encoding="utf-8")
    body = _extract_method_body(source, "generateMap", ABSTRACT_DUNGEON_SOURCE)
    dimensions = {name: int(value) for name, value in MAP_DIMENSION_PATTERN.findall(body)}
    if {"mapHeight", "mapWidth", "mapPathDensity"} - set(dimensions):
        raise ValueError(f"could not parse map dimensions from {ABSTRACT_DUNGEON_SOURCE}")

    row_assignments = list(ROW_ASSIGN_PATTERN.finditer(body))
    first_row_room_class = None
    last_row_room_class = None
    for match in row_assignments:
        row_expr = match.group("row").strip()
        room_class = match.group("room")
        if row_expr == "0":
            first_row_room_class = room_class
        elif row_expr == "map.size() - 1":
            last_row_room_class = room_class
    mimic_match = MIMIC_BRANCH_PATTERN.search(body)
    if first_row_room_class is None or last_row_room_class is None or mimic_match is None:
        raise ValueError(f"could not parse map row assignment rules from {ABSTRACT_DUNGEON_SOURCE}")

    return MapRuleDef(
        map_height=dimensions["mapHeight"],
        map_width=dimensions["mapWidth"],
        map_path_density=dimensions["mapPathDensity"],
        first_row_room_class=first_row_room_class,
        last_row_room_class=last_row_room_class,
        special_row_index=int(mimic_match.group("normal_row")),
        special_row_room_class=mimic_match.group("normal_room"),
        endless_mimic_row_index=int(mimic_match.group("endless_row")),
        endless_mimic_room_class=mimic_match.group("endless_room"),
        source_path=str(ABSTRACT_DUNGEON_SOURCE),
    )
