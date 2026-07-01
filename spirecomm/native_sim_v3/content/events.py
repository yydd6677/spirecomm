from __future__ import annotations

import re
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

EVENT_ROOT = sts_source_path("events")
DUNGEON_ROOT = sts_source_path("dungeons")
NOTE_FOR_YOURSELF_SOURCE = EVENT_ROOT / "shrines" / "NoteForYourself.java"

DEFAULT_STS_PREFERENCE_DIRS = (
    Path("/home/yydd/sts_instances/align/game/preferences"),
    Path("/home/yydd/sts_instances/autodl/game/preferences"),
    Path("/home/yydd/sts/sts_autodl/sts_instances/autodl/game/preferences"),
    Path("/home/yydd/.local/share/Steam/steamapps/common/SlayTheSpire/preferences"),
)

ID_PATTERN = re.compile(r'public static final String ID = "([^"]+)";')
CLASS_PATTERN = re.compile(r"class\s+([A-Za-z0-9_]+)\s+extends\s+")
ADD_PATTERN_TEMPLATE = r'{container}\.add\("([^"]+)"\);'
CASE_IF_PATTERN = re.compile(r'case "([^"]+)": \{\s*if \((.*?)\) continue block\d+;', re.DOTALL)

_DUNGEON_FILE_BY_ID = {
    "Exordium": DUNGEON_ROOT / "Exordium.java",
    "TheCity": DUNGEON_ROOT / "TheCity.java",
    "TheBeyond": DUNGEON_ROOT / "TheBeyond.java",
    "TheEnding": DUNGEON_ROOT / "TheEnding.java",
}
_ABSTRACT_DUNGEON_PATH = DUNGEON_ROOT / "AbstractDungeon.java"


@dataclass(frozen=True, slots=True)
class EventDef:
    event_id: str
    class_name: str
    area: str
    source_path: str


@dataclass(frozen=True, slots=True)
class AvailabilityRule:
    event_id: str
    dungeon_ids: tuple[str, ...] = ()
    floor_gt: int | None = None
    gold_ge: int | None = None
    current_hp_gt: int | None = None
    relic_count_ge: int | None = None
    playtime_seconds_ge: float | None = None
    require_curse: bool = False
    current_node_y_gt_half: bool = False
    required_relic_id: str | None = None
    hp_ratio_le: float | None = None
    source_condition: str = ""


@dataclass(frozen=True, slots=True)
class NoteForYourselfDefaults:
    card_pref_key: str
    upgrade_pref_key: str
    default_card_id: str
    default_upgrades: int
    source_path: str


@dataclass(frozen=True, slots=True)
class NoteForYourselfPreference:
    card_id: str
    upgrades: int
    source_path: str | None


@dataclass(frozen=True, slots=True)
class NoteForYourselfAvailability:
    daily_run_disables: bool
    disabled_at_ascension_ge: int
    enabled_at_ascension_eq: int
    unlocked_ascension_pref_key: str
    source_path: str


def _area_for_path(path: Path) -> str:
    return path.parent.name


def _parse_event_file(path: Path) -> EventDef | None:
    text = path.read_text(encoding="utf-8")
    id_match = ID_PATTERN.search(text)
    class_match = CLASS_PATTERN.search(text)
    if not id_match or not class_match:
        return None
    return EventDef(
        event_id=id_match.group(1),
        class_name=class_match.group(1),
        area=_area_for_path(path),
        source_path=str(path),
    )


def _extract_method_body(text: str, signature: str) -> str:
    declaration = re.search(
        rf"\b(?:public|protected|private)\s+[^{{;\n]*{re.escape(signature)}\s*\{{",
        text,
    )
    if declaration is None:
        raise KeyError(f"Could not find method signature {signature!r}.")
    brace_index = text.find("{", declaration.start())
    if brace_index < 0:
        raise KeyError(f"Could not find opening brace for {signature!r}.")
    depth = 1
    index = brace_index + 1
    while index < len(text) and depth > 0:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    if depth != 0:
        raise ValueError(f"Unbalanced braces while parsing {signature!r}.")
    return text[brace_index + 1:index - 1]


def _extract_added_strings(text: str, *, signature: str, container: str) -> list[str]:
    body = _extract_method_body(text, signature)
    pattern = re.compile(ADD_PATTERN_TEMPLATE.format(container=re.escape(container)))
    return pattern.findall(body)


@lru_cache(maxsize=1)
def event_catalog() -> dict[str, EventDef]:
    catalog: dict[str, EventDef] = {}
    for path in sorted(EVENT_ROOT.glob("*/*.java")):
        event = _parse_event_file(path)
        if event is not None:
            catalog[event.event_id] = event
    return catalog


def event_ids_for_area(area: str) -> list[str]:
    return sorted(event_id for event_id, event in event_catalog().items() if event.area == area)


@lru_cache(maxsize=None)
def dungeon_event_ids(dungeon_id: str) -> list[str]:
    path = _DUNGEON_FILE_BY_ID[dungeon_id]
    text = path.read_text(encoding="utf-8")
    return _extract_added_strings(text, signature="initializeEventList()", container="eventList")


@lru_cache(maxsize=None)
def dungeon_shrine_ids(dungeon_id: str) -> list[str]:
    path = _DUNGEON_FILE_BY_ID[dungeon_id]
    text = path.read_text(encoding="utf-8")
    return _extract_added_strings(text, signature="initializeShrineList()", container="shrineList")


@lru_cache(maxsize=2)
def special_one_time_event_ids(include_note_for_yourself: bool = True) -> list[str]:
    text = _ABSTRACT_DUNGEON_PATH.read_text(encoding="utf-8")
    ids = _extract_added_strings(text, signature="initializeSpecialOneTimeEventList()", container="specialOneTimeEventList")
    if include_note_for_yourself:
        return ids
    return [event_id for event_id in ids if event_id != "NoteForYourself"]


def note_for_yourself_available(
    *,
    ascension_level: int,
    is_daily_run: bool = False,
    highest_unlocked_ascension: int | None = None,
) -> bool:
    rules = note_for_yourself_availability()
    if rules.daily_run_disables and is_daily_run:
        return False
    if int(ascension_level) >= int(rules.disabled_at_ascension_ge):
        return False
    if int(ascension_level) == int(rules.enabled_at_ascension_eq):
        return True
    if highest_unlocked_ascension is not None and int(ascension_level) < int(highest_unlocked_ascension):
        return True
    return False


@lru_cache(maxsize=1)
def note_for_yourself_defaults() -> NoteForYourselfDefaults:
    text = NOTE_FOR_YOURSELF_SOURCE.read_text(encoding="utf-8")
    pref_match = re.search(r'getString\("([^"]+)",\s*"([^"]+)"\)', text)
    upgrade_match = re.search(r'getInteger\("([^"]+)",\s*(\d+)\)', text)
    if pref_match is None or upgrade_match is None:
        raise ValueError(f"could not parse NoteForYourself defaults from {NOTE_FOR_YOURSELF_SOURCE}")
    return NoteForYourselfDefaults(
        card_pref_key=str(pref_match.group(1)),
        upgrade_pref_key=str(upgrade_match.group(1)),
        default_card_id=str(pref_match.group(2)),
        default_upgrades=int(upgrade_match.group(2)),
        source_path=str(NOTE_FOR_YOURSELF_SOURCE),
    )


def _read_sts_pref_file(path: Path) -> dict[str, str] | None:
    for candidate in (path, path.with_suffix(path.suffix + ".backUp")):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        return {str(key): str(value) for key, value in payload.items()}
    return None


def _active_save_slot(preferences_dir: Path) -> int:
    save_slot_pref = _read_sts_pref_file(preferences_dir / "STSSaveSlots") or {}
    try:
        return int(str(save_slot_pref.get("DEFAULT_SLOT", "0")).strip())
    except ValueError:
        return 0


def _slot_pref_name(base_name: str, slot: int) -> str:
    return base_name if int(slot) == 0 else f"{int(slot)}_{base_name}"


def note_for_yourself_preference(
    preferences_dir: str | Path | None = None,
) -> NoteForYourselfPreference:
    defaults = note_for_yourself_defaults()
    candidate_dirs = (
        (Path(preferences_dir),)
        if preferences_dir is not None
        else tuple(path for path in DEFAULT_STS_PREFERENCE_DIRS if path.exists())
    )
    for candidate_dir in candidate_dirs:
        slot = _active_save_slot(candidate_dir)
        player_pref_path = candidate_dir / _slot_pref_name("STSPlayer", slot)
        player_pref = _read_sts_pref_file(player_pref_path)
        if player_pref is None:
            continue
        raw_card_id = player_pref.get(defaults.card_pref_key, defaults.default_card_id)
        raw_upgrades = player_pref.get(defaults.upgrade_pref_key, str(defaults.default_upgrades))
        try:
            upgrades = int(str(raw_upgrades).strip())
        except ValueError:
            upgrades = defaults.default_upgrades
        return NoteForYourselfPreference(
            card_id=str(raw_card_id or defaults.default_card_id),
            upgrades=upgrades,
            source_path=str(player_pref_path),
        )
    return NoteForYourselfPreference(
        card_id=defaults.default_card_id,
        upgrades=defaults.default_upgrades,
        source_path=None,
    )


@lru_cache(maxsize=1)
def note_for_yourself_availability() -> NoteForYourselfAvailability:
    text = _ABSTRACT_DUNGEON_PATH.read_text(encoding="utf-8")
    body = _extract_method_body(text, "isNoteForYourselfAvailable()")
    disabled_match = re.search(r'ascensionLevel >= (\d+)', body)
    enabled_match = re.search(r'ascensionLevel == (\d+)', body)
    pref_match = re.search(r'getInteger\("([^"]+)"\)', body)
    if disabled_match is None or enabled_match is None or pref_match is None:
        raise ValueError(f"could not parse NoteForYourself availability from {_ABSTRACT_DUNGEON_PATH}")
    return NoteForYourselfAvailability(
        daily_run_disables="Settings.isDailyRun" in body,
        disabled_at_ascension_ge=int(disabled_match.group(1)),
        enabled_at_ascension_eq=int(enabled_match.group(1)),
        unlocked_ascension_pref_key=str(pref_match.group(1)),
        source_path=str(_ABSTRACT_DUNGEON_PATH),
    )


def _parse_availability_rule(event_id: str, condition: str) -> AvailabilityRule:
    dungeon_ids = tuple(re.findall(r'id\.equals\("([^"]+)"\)', condition))
    floor_match = re.search(r'floorNum <= (\d+)', condition)
    gold_match = re.search(r'gold < (\d+)', condition)
    current_hp_match = re.search(r'currentHealth <= (\d+)', condition)
    relic_count_match = re.search(r'relics\.size\(\) < (\d+)', condition)
    playtime_match = re.search(r'playtime >= ([0-9.]+)f', condition)
    required_relic_match = re.search(r'hasRelic\("([^"]+)"\)', condition)
    hp_ratio_match = re.search(r'currentHealth / \(float\)AbstractDungeon\.player\.maxHealth > ([0-9.]+)f', condition)
    return AvailabilityRule(
        event_id=event_id,
        dungeon_ids=dungeon_ids,
        floor_gt=int(floor_match.group(1)) if floor_match else None,
        gold_ge=int(gold_match.group(1)) if gold_match else None,
        current_hp_gt=int(current_hp_match.group(1)) if current_hp_match else None,
        relic_count_ge=int(relic_count_match.group(1)) if relic_count_match else None,
        playtime_seconds_ge=float(playtime_match.group(1)) if playtime_match else None,
        require_curse="!player.isCursed()" in condition,
        current_node_y_gt_half="currMapNode == null" in condition and "currMapNode.y <= map.size() / 2" in condition,
        required_relic_id=required_relic_match.group(1) if required_relic_match else None,
        hp_ratio_le=float(hp_ratio_match.group(1)) if hp_ratio_match else None,
        source_condition=condition.strip(),
    )


@lru_cache(maxsize=1)
def abstract_dungeon_event_gate_rules() -> dict[str, AvailabilityRule]:
    text = _ABSTRACT_DUNGEON_PATH.read_text(encoding="utf-8")
    body = _extract_method_body(text, "getEvent(Random rng)")
    return {
        event_id: _parse_availability_rule(event_id, condition)
        for event_id, condition in CASE_IF_PATTERN.findall(body)
    }


@lru_cache(maxsize=1)
def abstract_dungeon_shrine_gate_rules() -> dict[str, AvailabilityRule]:
    text = _ABSTRACT_DUNGEON_PATH.read_text(encoding="utf-8")
    body = _extract_method_body(text, "getShrine(Random rng)")
    return {
        event_id: _parse_availability_rule(event_id, condition)
        for event_id, condition in CASE_IF_PATTERN.findall(body)
    }
