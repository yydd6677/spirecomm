from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

REST_OPTION_SOURCE = sts_source_path("ui/campfire/RestOption.java")
CAMPFIRE_SLEEP_SOURCE = sts_source_path("vfx/campfire/CampfireSleepEffect.java")
REGAL_PILLOW_SOURCE = sts_source_path("relics/RegalPillow.java")

HEAL_AMOUNT_PATTERN = re.compile(r"HEAL_AMOUNT\s*=\s*([0-9.]+)f;")
NIGHT_TERRORS_PATTERN = re.compile(r"maxHealth \* ([0-9.]+)f")
REGAL_PILLOW_HEAL_PATTERN = re.compile(r"HEAL_AMT\s*=\s*(\d+);")


@dataclass(frozen=True, slots=True)
class CampfireRuleDef:
    base_rest_heal_fraction: float
    night_terrors_heal_fraction: float
    regal_pillow_bonus: int
    source_paths: tuple[str, ...]


@lru_cache(maxsize=1)
def campfire_rules() -> CampfireRuleDef:
    rest_option_source = REST_OPTION_SOURCE.read_text(encoding="utf-8")
    sleep_source = CAMPFIRE_SLEEP_SOURCE.read_text(encoding="utf-8")
    regal_pillow_source = REGAL_PILLOW_SOURCE.read_text(encoding="utf-8")

    heal_amount_match = HEAL_AMOUNT_PATTERN.search(sleep_source)
    if heal_amount_match is None:
        raise ValueError(f"could not locate campfire heal amount in {CAMPFIRE_SLEEP_SOURCE}")
    night_terrors_matches = NIGHT_TERRORS_PATTERN.findall(rest_option_source)
    if not night_terrors_matches:
        raise ValueError(f"could not locate Night Terrors heal amount in {REST_OPTION_SOURCE}")
    regal_match = REGAL_PILLOW_HEAL_PATTERN.search(regal_pillow_source)
    if regal_match is None:
        raise ValueError(f"could not locate Regal Pillow heal amount in {REGAL_PILLOW_SOURCE}")

    return CampfireRuleDef(
        base_rest_heal_fraction=float(heal_amount_match.group(1)),
        night_terrors_heal_fraction=max(float(value) for value in night_terrors_matches),
        regal_pillow_bonus=int(regal_match.group(1)),
        source_paths=(
            str(REST_OPTION_SOURCE),
            str(CAMPFIRE_SLEEP_SOURCE),
            str(REGAL_PILLOW_SOURCE),
        ),
    )


def rest_amount(
    max_hp: int,
    *,
    night_terrors: bool = False,
    endless_full_belly: bool = False,
) -> int:
    rules = campfire_rules()
    fraction = rules.night_terrors_heal_fraction if night_terrors else rules.base_rest_heal_fraction
    heal_amount = int(int(max_hp) * float(fraction))
    if endless_full_belly:
        heal_amount //= 2
    return max(1, int(heal_amount))


def regal_pillow_bonus() -> int:
    return int(campfire_rules().regal_pillow_bonus)
