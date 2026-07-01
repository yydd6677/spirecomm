from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

ACT_PATHS = {
    "Exordium": sts_source_path("dungeons/Exordium.java"),
    "TheCity": sts_source_path("dungeons/TheCity.java"),
    "TheBeyond": sts_source_path("dungeons/TheBeyond.java"),
}
MONSTER_INFO_PATTERN = re.compile(r'MonsterInfo\("([^"]+)",\s*([0-9.]+)f\)')
BOSS_ADD_PATTERN = re.compile(r'bossList\.add\("([^"]+)"\);')
COUNT_PATTERN = re.compile(r"this\.generate(WeakEnemies|StrongEnemies|Elites)\((\d+)\);")
CASE_PATTERN = re.compile(r'case "([^"]+)": \{(?P<body>.*?)break;', re.S)
RETVAL_ADD_PATTERN = re.compile(r'retVal\.add\("([^"]+)"\);')


@dataclass(frozen=True, slots=True)
class EncounterWeightDef:
    encounter_name: str
    weight: float


@dataclass(frozen=True, slots=True)
class EncounterExclusionDef:
    trigger_name: str
    excluded_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActEncounterDef:
    act_id: str
    weak: tuple[EncounterWeightDef, ...]
    strong: tuple[EncounterWeightDef, ...]
    elite: tuple[EncounterWeightDef, ...]
    weak_count: int
    strong_count: int
    elite_count: int
    strong_exclusions: tuple[EncounterExclusionDef, ...]
    bosses: tuple[str, ...]
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


def _parse_weighted_encounters(text: str, method_name: str, source_path: Path) -> tuple[EncounterWeightDef, ...]:
    body = _extract_method_body(text, method_name, source_path)
    return tuple(
        EncounterWeightDef(encounter_name=name, weight=float(weight))
        for name, weight in MONSTER_INFO_PATTERN.findall(body)
    )


def _parse_generate_counts(text: str, source_path: Path) -> tuple[int, int, int]:
    body = _extract_method_body(text, "generateMonsters", source_path)
    counts: dict[str, int] = {}
    for count_kind, count_value in COUNT_PATTERN.findall(body):
        counts[count_kind] = int(count_value)
    missing = {"WeakEnemies", "StrongEnemies", "Elites"} - set(counts)
    if missing:
        raise ValueError(f"missing generateMonsters counts {sorted(missing)!r} in {source_path}")
    return counts["WeakEnemies"], counts["StrongEnemies"], counts["Elites"]


def _parse_exclusions(text: str, source_path: Path) -> tuple[EncounterExclusionDef, ...]:
    body = _extract_method_body(text, "generateExclusions", source_path)
    exclusions: list[EncounterExclusionDef] = []
    for trigger_name, case_body in CASE_PATTERN.findall(body):
        excluded_names = tuple(RETVAL_ADD_PATTERN.findall(case_body))
        if excluded_names:
            exclusions.append(EncounterExclusionDef(trigger_name=trigger_name, excluded_names=excluded_names))
    return tuple(exclusions)


def _parse_bosses(text: str, source_path: Path) -> tuple[str, ...]:
    body = _extract_method_body(text, "initializeBoss", source_path)
    ordered: list[str] = []
    seen: set[str] = set()
    for boss_id in BOSS_ADD_PATTERN.findall(body):
        if boss_id not in seen:
            seen.add(boss_id)
            ordered.append(boss_id)
    return tuple(ordered)


@lru_cache(maxsize=1)
def encounter_catalog() -> dict[str, ActEncounterDef]:
    catalog: dict[str, ActEncounterDef] = {}
    for act_id, source_path in ACT_PATHS.items():
        text = source_path.read_text(encoding="utf-8")
        weak_count, strong_count, elite_count = _parse_generate_counts(text, source_path)
        catalog[act_id] = ActEncounterDef(
            act_id=act_id,
            weak=_parse_weighted_encounters(text, "generateWeakEnemies", source_path),
            strong=_parse_weighted_encounters(text, "generateStrongEnemies", source_path),
            elite=_parse_weighted_encounters(text, "generateElites", source_path),
            weak_count=weak_count,
            strong_count=strong_count,
            elite_count=elite_count,
            strong_exclusions=_parse_exclusions(text, source_path),
            bosses=_parse_bosses(text, source_path),
            source_path=str(source_path),
        )
    return catalog


def act_encounter_def(act_id: str) -> ActEncounterDef:
    return encounter_catalog()[act_id]
