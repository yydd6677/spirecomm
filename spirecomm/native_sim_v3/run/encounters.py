from __future__ import annotations

from dataclasses import dataclass

from spirecomm.native_sim_v3.content.act_progression import dungeon_id_for_act
from spirecomm.native_sim_v3.content import act_encounter_def
from spirecomm.native_sim_v3.content.ending_rules import ending_rules
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, java_shuffle_in_place


@dataclass(frozen=True, slots=True)
class MonsterInfoDef:
    name: str
    weight: float


def _defs_for_bucket(act_id: str, bucket: str) -> list[MonsterInfoDef]:
    encounter_def = act_encounter_def(act_id)
    weighted_defs = getattr(encounter_def, bucket)
    return [MonsterInfoDef(item.encounter_name, item.weight) for item in weighted_defs]


def _normalize(defs: list[MonsterInfoDef]) -> list[MonsterInfoDef]:
    sorted_defs = sorted(defs, key=lambda item: item.weight)
    total = sum(item.weight for item in sorted_defs)
    return [MonsterInfoDef(item.name, item.weight / total) for item in sorted_defs]


def _roll_name(defs: list[MonsterInfoDef], roll: float) -> str:
    current = 0.0
    for item in defs:
        current += item.weight
        if roll < current:
            return item.name
    return defs[-1].name


def _populate_monster_list(
    defs: list[MonsterInfoDef],
    count: int,
    randoms: NativeRandomSet,
    *,
    output: list[str] | None = None,
) -> list[str]:
    normalized = _normalize(defs)
    output = [] if output is None else output
    monster_rng = randoms.stream("monster")
    target_size = len(output) + count
    while len(output) < target_size:
        roll = float(monster_rng.random(0.0, 0.999999))
        candidate = _roll_name(normalized, roll)
        if not output:
            output.append(candidate)
            continue
        if candidate == output[-1]:
            continue
        if len(output) > 1 and candidate == output[-2]:
            continue
        output.append(candidate)
    return output


def generate_exordium_weak_list(randoms: NativeRandomSet, count: int = 3) -> list[str]:
    return _populate_monster_list(_defs_for_bucket("Exordium", "weak"), count, randoms)


def _populate_elite_list(defs: list[MonsterInfoDef], count: int, randoms: NativeRandomSet) -> list[str]:
    normalized = _normalize(defs)
    output: list[str] = []
    monster_rng = randoms.stream("monster")
    while len(output) < count:
        roll = float(monster_rng.random(0.0, 0.999999))
        candidate = _roll_name(normalized, roll)
        if output and candidate == output[-1]:
            continue
        output.append(candidate)
    return output


def _strong_exclusions_for_act(act_id: str, last_weak_encounter: str) -> set[str]:
    encounter_def = act_encounter_def(act_id)
    for rule in encounter_def.strong_exclusions:
        if rule.trigger_name == last_weak_encounter:
            return set(rule.excluded_names)
    return set()


def _populate_first_strong_enemy(defs: list[MonsterInfoDef], randoms: NativeRandomSet, exclusions: set[str]) -> str:
    normalized = _normalize(defs)
    monster_rng = randoms.stream("monster")
    while True:
        roll = float(monster_rng.random(0.0, 0.999999))
        candidate = _roll_name(normalized, roll)
        if candidate not in exclusions:
            return candidate


def generate_monster_lists_for_act(
    randoms: NativeRandomSet,
    act: int,
    *,
    weak_count: int | None = None,
    strong_count: int | None = None,
    elite_count: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    return generate_monster_lists_for_dungeon(
        randoms,
        dungeon_id_for_act(act),
        weak_count=weak_count,
        strong_count=strong_count,
        elite_count=elite_count,
    )


def generate_monster_lists_for_dungeon(
    randoms: NativeRandomSet,
    dungeon_id: str,
    *,
    weak_count: int | None = None,
    strong_count: int | None = None,
    elite_count: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    act_id = str(dungeon_id)
    if act_id == "TheEnding":
        rules = ending_rules()
        elite_list = list(rules.elite_encounters) or ["Shield and Spear"]
        boss_list = list(rules.boss_encounters) or ["The Heart"]
        return list(elite_list), list(elite_list), list(boss_list)
    encounter_def = act_encounter_def(act_id)
    weak_defs = _defs_for_bucket(act_id, "weak")
    strong_defs = _defs_for_bucket(act_id, "strong")
    elite_defs = _defs_for_bucket(act_id, "elite")
    boss_defs = list(encounter_def.bosses)
    if weak_count is None:
        weak_count = encounter_def.weak_count
    if strong_count is None:
        strong_count = encounter_def.strong_count
    if elite_count is None:
        elite_count = encounter_def.elite_count
    monster_list = _populate_monster_list(weak_defs, weak_count, randoms)
    exclusions = _strong_exclusions_for_act(act_id, monster_list[-1])
    monster_list.append(_populate_first_strong_enemy(strong_defs, randoms, exclusions))
    _populate_monster_list(strong_defs, strong_count, randoms, output=monster_list)
    elite_list = _populate_elite_list(elite_defs, elite_count, randoms)
    boss_list = list(boss_defs)
    java_shuffle_in_place(boss_list, randoms.stream("monster").random_long())
    if len(boss_list) == 1:
        boss_list.append(boss_list[0])
    return monster_list, elite_list, boss_list


def generate_exordium_monster_lists(
    randoms: NativeRandomSet,
    *,
    weak_count: int | None = None,
    strong_count: int | None = None,
    elite_count: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    return generate_monster_lists_for_act(
        randoms,
        1,
        weak_count=weak_count,
        strong_count=strong_count,
        elite_count=elite_count,
    )
