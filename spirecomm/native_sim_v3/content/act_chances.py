from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.content.act_progression import dungeon_id_for_act
from spirecomm.native_sim_v3.source_paths import sts_source_path


ACT_CHANCE_PATHS = {
    "Exordium": sts_source_path("dungeons/Exordium.java"),
    "TheCity": sts_source_path("dungeons/TheCity.java"),
    "TheBeyond": sts_source_path("dungeons/TheBeyond.java"),
    "TheEnding": sts_source_path("dungeons/TheEnding.java"),
}

INT_ASSIGNMENT_PATTERN = re.compile(r"(smallChestChance|mediumChestChance|largeChestChance|commonRelicChance|uncommonRelicChance|rareRelicChance) = (\d+);")
FLOAT_ROOM_ASSIGNMENT_PATTERN = re.compile(
    r"(shopRoomChance|restRoomChance|treasureRoomChance|eventRoomChance|eliteRoomChance) = ([0-9.]+)f;"
)
FLOAT_ASSIGNMENT_PATTERN = re.compile(r"colorlessRareChance = ([0-9.]+)f;")
TERNARY_CARD_UPGRADE_PATTERN = re.compile(
    r"cardUpgradedChance = AbstractDungeon\.ascensionLevel >= 12 \? ([0-9.]+)f : ([0-9.]+)f;"
)
CONSTANT_CARD_UPGRADE_PATTERN = re.compile(r"cardUpgradedChance = ([0-9.]+)f;")


@dataclass(frozen=True, slots=True)
class ActChanceDef:
    act_id: str
    shop_room_chance: float
    rest_room_chance: float
    treasure_room_chance: float
    event_room_chance: float
    elite_room_chance: float
    small_chest_chance: int
    medium_chest_chance: int
    large_chest_chance: int
    common_relic_chance: int
    uncommon_relic_chance: int
    rare_relic_chance: int
    colorless_rare_chance: float
    card_upgraded_chance_low_ascension: float
    card_upgraded_chance_high_ascension: float
    source_path: str

    def card_upgraded_chance(self, ascension_level: int) -> float:
        return self.card_upgraded_chance_high_ascension if int(ascension_level) >= 12 else self.card_upgraded_chance_low_ascension


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


def _act_id(value: int | str) -> str:
    return dungeon_id_for_act(value)


@lru_cache(maxsize=1)
def act_chance_catalog() -> dict[str, ActChanceDef]:
    catalog: dict[str, ActChanceDef] = {}
    for act_id, source_path in ACT_CHANCE_PATHS.items():
        text = source_path.read_text(encoding="utf-8")
        body = _extract_method_body(text, "initializeLevelSpecificChances", source_path)
        int_assignments = {name: int(value) for name, value in INT_ASSIGNMENT_PATTERN.findall(body)}
        room_assignments = {name: float(value) for name, value in FLOAT_ROOM_ASSIGNMENT_PATTERN.findall(body)}
        colorless_match = FLOAT_ASSIGNMENT_PATTERN.search(body)
        if colorless_match is None:
            raise ValueError(f"could not locate colorlessRareChance in {source_path}")
        ternary_match = TERNARY_CARD_UPGRADE_PATTERN.search(body)
        if ternary_match is not None:
            high_ascension = float(ternary_match.group(1))
            low_ascension = float(ternary_match.group(2))
        else:
            constant_match = CONSTANT_CARD_UPGRADE_PATTERN.search(body)
            if constant_match is None:
                raise ValueError(f"could not locate cardUpgradedChance in {source_path}")
            low_ascension = high_ascension = float(constant_match.group(1))
        catalog[act_id] = ActChanceDef(
            act_id=act_id,
            shop_room_chance=room_assignments["shopRoomChance"],
            rest_room_chance=room_assignments["restRoomChance"],
            treasure_room_chance=room_assignments["treasureRoomChance"],
            event_room_chance=room_assignments["eventRoomChance"],
            elite_room_chance=room_assignments["eliteRoomChance"],
            small_chest_chance=int_assignments["smallChestChance"],
            medium_chest_chance=int_assignments["mediumChestChance"],
            large_chest_chance=int_assignments["largeChestChance"],
            common_relic_chance=int_assignments["commonRelicChance"],
            uncommon_relic_chance=int_assignments["uncommonRelicChance"],
            rare_relic_chance=int_assignments["rareRelicChance"],
            colorless_rare_chance=float(colorless_match.group(1)),
            card_upgraded_chance_low_ascension=low_ascension,
            card_upgraded_chance_high_ascension=high_ascension,
            source_path=str(source_path),
        )
    return catalog


def act_chances(act: int | str) -> ActChanceDef:
    return act_chance_catalog()[_act_id(act)]
