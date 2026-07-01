from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

TREASURE_ROOM_BOSS_SOURCE = sts_source_path("rooms/TreasureRoomBoss.java")
_DUNGEON_ID_BY_ACT = {
    1: "Exordium",
    2: "TheCity",
    3: "TheBeyond",
    4: "TheEnding",
}
_ACT_BY_DUNGEON_ID = {dungeon_id: act for act, dungeon_id in _DUNGEON_ID_BY_ACT.items()}


@dataclass(frozen=True, slots=True)
class ActProgressionDef:
    standard_next_by_dungeon_id: dict[str, str | None]
    endless_next_by_dungeon_id: dict[str, str | None]
    source_path: str

    def next_dungeon_id(self, current_dungeon_id: str, *, endless: bool = False) -> str | None:
        mapping = self.endless_next_by_dungeon_id if endless else self.standard_next_by_dungeon_id
        return mapping.get(str(current_dungeon_id))


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
    raise ValueError(f"unterminated method body for method {method_name!r} in {source_path}")


@lru_cache(maxsize=1)
def act_progression() -> ActProgressionDef:
    text = TREASURE_ROOM_BOSS_SOURCE.read_text(encoding="utf-8")
    body = _extract_method_body(text, "getNextDungeonName", TREASURE_ROOM_BOSS_SOURCE)
    standard: dict[str, str | None] = {}
    endless: dict[str, str | None] = {}
    case_pattern = re.compile(r'case "([^"]+)":\s*\{')
    for case_match in case_pattern.finditer(body):
        dungeon_id = case_match.group(1)
        brace_start = body.find("{", case_match.start())
        depth = 0
        case_end = -1
        for index in range(brace_start, len(body)):
            char = body[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    case_end = index
                    break
        if case_end == -1:
            raise ValueError(f"unterminated case body for dungeon {dungeon_id!r} in {TREASURE_ROOM_BOSS_SOURCE}")
        case_body = body[brace_start + 1 : case_end]
        endless_match = re.search(r'if \(Settings\.isEndless\)\s*\{\s*return "([^"]+)";\s*\}', case_body)
        standard_case_body = re.sub(
            r'if \(Settings\.isEndless\)\s*\{\s*return "([^"]+)";\s*\}',
            "",
            case_body,
            flags=re.DOTALL,
        )
        standard_match = re.search(r'return "([^"]+)";', standard_case_body)
        standard[dungeon_id] = standard_match.group(1) if standard_match is not None else None
        endless[dungeon_id] = endless_match.group(1) if endless_match is not None else standard[dungeon_id]
    return ActProgressionDef(
        standard_next_by_dungeon_id=standard,
        endless_next_by_dungeon_id=endless,
        source_path=str(TREASURE_ROOM_BOSS_SOURCE),
    )


def dungeon_id_for_act(act: int | str) -> str:
    if isinstance(act, str):
        if act in _ACT_BY_DUNGEON_ID:
            return act
        return _DUNGEON_ID_BY_ACT[int(act)]
    return _DUNGEON_ID_BY_ACT[int(act)]


def act_for_dungeon_id(dungeon_id: str) -> int:
    return _ACT_BY_DUNGEON_ID[str(dungeon_id)]


def next_dungeon_id(current_dungeon_id: str, *, endless: bool = False) -> str | None:
    return act_progression().next_dungeon_id(current_dungeon_id, endless=endless)
