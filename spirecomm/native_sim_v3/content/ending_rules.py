from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

THE_ENDING_SOURCE = sts_source_path("dungeons/TheEnding.java")

NODE_DECL_PATTERN = re.compile(r"MapRoomNode\s+(\w+)\s*=\s*new MapRoomNode\(([-]?\d+),\s*(\d+)\);")
ROOM_ASSIGN_PATTERN = re.compile(r"(\w+)\.room\s*=\s*new\s+(\w+)\(\);")
CONNECT_PATTERN = re.compile(r"this\.connectNode\((\w+),\s*(\w+)\);")
DIRECT_EDGE_PATTERN = re.compile(r"(\w+)\.addEdge\(new MapEdge\([^;]*?,\s*(\w+)\.x,\s*(\w+)\.y,[^;]*?\)\);")
ELITE_ADD_PATTERN = re.compile(r"eliteMonsterList\.add\(\"([^\"]+)\"\);")
BOSS_ADD_PATTERN = re.compile(r"bossList\.add\(\"([^\"]+)\"\);")


@dataclass(frozen=True, slots=True)
class EndingRoomDef:
    node_name: str
    x: int
    y: int
    room_class: str


@dataclass(frozen=True, slots=True)
class EndingEdgeDef:
    src_name: str
    dst_name: str


@dataclass(frozen=True, slots=True)
class EndingRules:
    rooms: tuple[EndingRoomDef, ...]
    edges: tuple[EndingEdgeDef, ...]
    elite_encounters: tuple[str, ...]
    boss_encounters: tuple[str, ...]
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
def ending_rules() -> EndingRules:
    text = THE_ENDING_SOURCE.read_text(encoding="utf-8")
    map_body = _extract_method_body(text, "generateSpecialMap", THE_ENDING_SOURCE)
    monster_body = _extract_method_body(text, "generateMonsters", THE_ENDING_SOURCE)
    boss_body = _extract_method_body(text, "initializeBoss", THE_ENDING_SOURCE)

    node_positions = {name: (int(x), int(y)) for name, x, y in NODE_DECL_PATTERN.findall(map_body)}
    room_assignments = {name: room_class for name, room_class in ROOM_ASSIGN_PATTERN.findall(map_body)}
    rooms = tuple(
        EndingRoomDef(node_name=name, x=coords[0], y=coords[1], room_class=room_assignments[name])
        for name, coords in node_positions.items()
        if name in room_assignments
    )

    direct_edges = {(src, dst) for src, dst in CONNECT_PATTERN.findall(map_body)}
    direct_edges.update((src, dst) for src, dst, _ in DIRECT_EDGE_PATTERN.findall(map_body))

    elite_encounters: list[str] = []
    for encounter_name in ELITE_ADD_PATTERN.findall(monster_body):
        if encounter_name not in elite_encounters:
            elite_encounters.append(encounter_name)

    boss_encounters: list[str] = []
    for encounter_name in BOSS_ADD_PATTERN.findall(boss_body):
        if encounter_name not in boss_encounters:
            boss_encounters.append(encounter_name)

    if not rooms or not direct_edges or not elite_encounters or not boss_encounters:
        raise ValueError(f"could not parse The Ending rules from {THE_ENDING_SOURCE}")

    return EndingRules(
        rooms=rooms,
        edges=tuple(EndingEdgeDef(src_name=src, dst_name=dst) for src, dst in sorted(direct_edges)),
        elite_encounters=tuple(elite_encounters),
        boss_encounters=tuple(boss_encounters),
        source_path=str(THE_ENDING_SOURCE),
    )
