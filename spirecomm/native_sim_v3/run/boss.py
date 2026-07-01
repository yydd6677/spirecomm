from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.content import make_relic, relic_pools
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, java_shuffle_in_place
from spirecomm.native_sim_v3.source_paths import sts_source_path


RELIC_ROOT = sts_source_path("relics")
BOSS_RELIC_SELECT_SOURCE = sts_source_path("screens/select/BossRelicSelectScreen.java")
ABSTRACT_RELIC_SOURCE = sts_source_path("relics/AbstractRelic.java")
RELIC_ID_PATTERN = re.compile(r'public static final String ID = "([^"]+)";')
HAS_RELIC_PATTERN = re.compile(r'return AbstractDungeon\.player\.hasRelic\("([^"]+)"\);')
RELIC_EQUALS_PATTERN = re.compile(r'relicId\.equals\("([^"]+)"\)')


@lru_cache(maxsize=1)
def _relic_source_paths_by_id() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in RELIC_ROOT.glob("*.java"):
        text = path.read_text(encoding="utf-8")
        match = RELIC_ID_PATTERN.search(text)
        if match:
            result[match.group(1)] = path
    return result


@lru_cache(maxsize=1)
def _boss_relic_replacement_ids() -> tuple[str, ...]:
    ids: list[str] = []
    for source_path in (BOSS_RELIC_SELECT_SOURCE, ABSTRACT_RELIC_SOURCE):
        for relic_id in RELIC_EQUALS_PATTERN.findall(source_path.read_text(encoding="utf-8")):
            if relic_id not in ids:
                ids.append(relic_id)
    return tuple(ids)


@lru_cache(maxsize=1)
def starter_relic_upgrade_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    paths_by_id = _relic_source_paths_by_id()
    for boss_relic_id in _boss_relic_replacement_ids():
        path = paths_by_id.get(boss_relic_id)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        match = HAS_RELIC_PATTERN.search(text)
        if match:
            mapping[boss_relic_id] = match.group(1)
    return mapping


def initialize_boss_relic_pool(
    randoms: NativeRandomSet,
    *,
    owned_relic_ids: set[str] | None = None,
    character: str = "IRONCLAD",
) -> list[str]:
    owned = {str(relic_id) for relic_id in set(owned_relic_ids or ())}
    pool = list(relic_pools(character).get("BOSS", []))
    java_shuffle_in_place(pool, randoms.stream("relic").random_long())
    if owned:
        pool = [relic_id for relic_id in pool if relic_id not in owned]
    return pool


def draw_boss_relic_choices(
    boss_relic_pool: list[str],
    *,
    count: int = 3,
    act_num: int = 1,
    owned_relic_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    owned = {str(relic_id) for relic_id in set(owned_relic_ids or ())}
    choices: list[dict[str, object]] = []
    while len(choices) < count:
        relic_id = boss_relic_pool.pop(0) if boss_relic_pool else "Red Circlet"
        if relic_id != "Red Circlet" and not _can_spawn_boss_relic(relic_id, act_num=act_num, owned_relic_ids=owned):
            continue
        choices.append(make_relic(relic_id))
    return choices


def _can_spawn_boss_relic(relic_id: str, *, act_num: int, owned_relic_ids: set[str]) -> bool:
    if relic_id == "Ectoplasm":
        return int(act_num) <= 1
    starter_requirements = {
        "Black Blood": "Burning Blood",
        "FrozenCore": "Cracked Core",
        "Ring of the Serpent": "Ring of the Snake",
        "HolyWater": "PureWater",
    }
    required = starter_requirements.get(relic_id)
    if required is not None:
        return required in owned_relic_ids
    return True


def apply_boss_relic_choice(
    relics: list[dict[str, object]],
    chosen_relic: dict[str, object],
) -> list[dict[str, object]]:
    chosen = dict(chosen_relic)
    chosen_id = str(chosen.get("relic_id") or chosen.get("id") or "")
    replaced_id = starter_relic_upgrade_mapping().get(chosen_id)
    if replaced_id is None:
        return [*relics, chosen]
    updated: list[dict[str, object]] = []
    replaced = False
    for relic in relics:
        relic_id = str(relic.get("relic_id") or relic.get("id") or "")
        if not replaced and relic_id == replaced_id:
            updated.append(chosen)
            replaced = True
        else:
            updated.append(dict(relic))
    if not replaced:
        updated.append(chosen)
    return updated


def roll_boss_relics(
    randoms: NativeRandomSet,
    *,
    owned_relic_ids: set[str] | None = None,
    character: str = "IRONCLAD",
    count: int = 3,
) -> list[dict[str, object]]:
    pool = initialize_boss_relic_pool(randoms, owned_relic_ids=owned_relic_ids, character=character)
    return draw_boss_relic_choices(pool, count=count)
