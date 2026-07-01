from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from spirecomm.native_sim_v3.source_paths import sts_source_path

CHARACTER_ROOT = sts_source_path("characters")
CHARACTER_SOURCE_FILES = {
    "IRONCLAD": "Ironclad.java",
    "THE_SILENT": "TheSilent.java",
    "DEFECT": "Defect.java",
    "WATCHER": "Watcher.java",
}

METHOD_PATTERN_TEMPLATE = r"public [^\n]+ {method}\([^)]*\) \{{(?P<body>.*?)^\s*\}}"
RETVAL_ADD_PATTERN = re.compile(r'retVal\.add\("([^"]+)"\);')
LOADOUT_PATTERN = re.compile(
    r"new CharSelectInfo\([^,]+,\s*[^,]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),",
    re.DOTALL,
)
ENERGY_MANAGER_PATTERN = re.compile(r"new EnergyManager\((\d+)\)")


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    character: str
    max_hp: int
    current_hp: int
    orb_slots: int
    gold: int
    card_draw: int
    base_energy: int
    starter_relic_ids: tuple[str, ...]
    starter_deck_ids: tuple[str, ...]


def _extract_method_body(source: str, method_name: str) -> str:
    pattern = re.compile(METHOD_PATTERN_TEMPLATE.format(method=re.escape(method_name)), re.DOTALL | re.MULTILINE)
    match = pattern.search(source)
    return match.group("body") if match else ""


def _character_source(character: str) -> str:
    character_key = str(character)
    filename = CHARACTER_SOURCE_FILES.get(character_key)
    if filename is None:
        raise NotImplementedError(f"native_sim_v3 only supports character {character!r} for now.")
    return (CHARACTER_ROOT / filename).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def starting_profile(character: str = "IRONCLAD") -> CharacterProfile:
    source = _character_source(character)
    loadout_match = LOADOUT_PATTERN.search(source)
    if not loadout_match:
        raise RuntimeError(f"native_sim_v3 could not parse CharSelectInfo loadout for {character!r}")
    energy_match = ENERGY_MANAGER_PATTERN.search(source)
    if not energy_match:
        raise RuntimeError(f"native_sim_v3 could not parse EnergyManager base energy for {character!r}")
    max_hp, current_hp, orb_slots, gold, card_draw = (int(group) for group in loadout_match.groups())
    starting_relics = tuple(RETVAL_ADD_PATTERN.findall(_extract_method_body(source, "getStartingRelics")))
    starting_deck = tuple(RETVAL_ADD_PATTERN.findall(_extract_method_body(source, "getStartingDeck")))
    return CharacterProfile(
        character=str(character),
        max_hp=max_hp,
        current_hp=current_hp,
        orb_slots=orb_slots,
        gold=gold,
        card_draw=card_draw,
        base_energy=int(energy_match.group(1)),
        starter_relic_ids=starting_relics,
        starter_deck_ids=starting_deck,
    )
