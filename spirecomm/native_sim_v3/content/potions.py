from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import re
import zipfile

from spirecomm.native_sim_v3.content.pricing import potion_price_for_rarity
from spirecomm.native_sim_v3.content.reward_rules import potion_roll_rules
from spirecomm.native_sim_v3.source_paths import GAME_JAR_PATH, sts_localization_path, sts_source_path

_DECOMPILED_ROOT = sts_source_path("")
_POTION_HELPER_PATH = _DECOMPILED_ROOT / "helpers" / "PotionHelper.java"
_POTIONS_DIR = _DECOMPILED_ROOT / "potions"
PLAYER_CLASSES = ("IRONCLAD", "THE_SILENT", "DEFECT", "WATCHER")
OUT_OF_COMBAT_USABLE_POTION_IDS = frozenset({"BloodPotion", "EntropicBrew", "Fruit Juice"})
TRACE_REWARD_POTION_PRIORITY = {
    # Hand-ranked replacement priority. Larger values are kept over smaller
    # values when a full potion belt receives a new reward potion.
    "EntropicBrew": 3000,
    "Fruit Juice": 2990,
    "HeartOfIron": 2980,
    "CultistPotion": 2970,
    "SneckoOil": 2960,
    "Regen Potion": 2900,
    "DistilledChaos": 2890,
    "Ancient Potion": 2880,
    "LiquidMemories": 2870,
    "DuplicationPotion": 2860,
    "EssenceOfSteel": 2850,
    "LiquidBronze": 2840,
    "BloodPotion": 2800,
    "Block Potion": 2790,
    "PowerPotion": 2780,
    "ColorlessPotion": 2770,
    "Strength Potion": 2760,
    "Energy Potion": 2750,
    "FearPotion": 2740,
    "AttackPotion": 2730,
    "SkillPotion": 2720,
    "BlessingOfTheForge": 2710,
    "Fire Potion": 2700,
    "Explosive Potion": 2690,
    "Weak Potion": 2680,
    "SteroidPotion": 2670,
    "Swift Potion": 2660,
    "Dexterity Potion": 2650,
    "SpeedPotion": 2640,
    "GamblersBrew": 2630,
    "ElixirPotion": 2620,
    "SmokeBomb": 2610,
}
IMPUTED_POTION_PRIORITY: dict[str, int] = {}
RARITY_FALLBACK_POTION_PRIORITY = {
    "COMMON": 1100,
    "UNCOMMON": 1200,
    "RARE": 1250,
}


def _extract_case_body(source: str, marker: str) -> str:
    anchor = source.find(marker)
    if anchor == -1:
        return ""
    start = source.find("{", anchor)
    if start == -1:
        return ""
    depth = 1
    index = start + 1
    while index < len(source) and depth > 0:
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    return source[start + 1 : index - 1]


@lru_cache(maxsize=1)
def potion_rarity_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not _POTIONS_DIR.exists():
        return mapping
    for path in sorted(_POTIONS_DIR.glob("*.java")):
        source = path.read_text(errors="ignore")
        potion_id_match = re.search(r'POTION_ID\s*=\s*"([^"]+)"', source)
        rarity_match = re.search(r"PotionRarity\.([A-Z_]+)", source)
        if potion_id_match and rarity_match:
            mapping[potion_id_match.group(1)] = rarity_match.group(1)
    return mapping


@lru_cache(maxsize=1)
def potion_name_map() -> dict[str, str]:
    localization_path = sts_localization_path("potions.json")
    if localization_path.exists():
        data = json.loads(localization_path.read_text(encoding="utf-8"))
    elif GAME_JAR_PATH.exists():
        with zipfile.ZipFile(GAME_JAR_PATH) as archive:
            with archive.open("localization/eng/potions.json") as handle:
                data = json.loads(handle.read().decode("utf-8"))
    else:
        return {}
    names: dict[str, str] = {}
    for potion_id, payload in data.items():
        if isinstance(payload, dict) and payload.get("NAME"):
            names[str(potion_id)] = str(payload["NAME"])
    return names


@lru_cache(maxsize=1)
def potion_requires_target_map() -> dict[str, bool]:
    # Communication Mod exposes this field from AbstractPotion.isThrown, not
    # AbstractPotion.targetRequired.
    mapping: dict[str, bool] = {}
    if not _POTIONS_DIR.exists():
        return mapping
    for path in sorted(_POTIONS_DIR.glob("*.java")):
        source = path.read_text(errors="ignore")
        potion_id_match = re.search(r'POTION_ID\s*=\s*"([^"]+)"', source)
        if potion_id_match:
            mapping[potion_id_match.group(1)] = bool(re.search(r"isThrown\s*=\s*true", source))
    return mapping


def _extract_shared_potions(helper_source: str) -> list[str]:
    common_section_start = helper_source.find("if (!getAll)")
    common_section = helper_source[common_section_start:] if common_section_start != -1 else helper_source
    common_start = common_section.find('retVal.add("Block Potion");')
    common_body = common_section[common_start:] if common_start != -1 else ""
    return [entry for entry in re.findall(r'retVal\.add\("([^"]+)"\);', common_body) if entry != "Potion Slot"]


@lru_cache(maxsize=1)
def potion_scopes() -> dict[str, list[str]]:
    helper_source = _POTION_HELPER_PATH.read_text(errors="ignore")
    scopes: dict[str, list[str]] = {"SHARED": _extract_shared_potions(helper_source)}
    for class_name in PLAYER_CLASSES:
        body = _extract_case_body(helper_source, f"case {class_name}:")
        scopes[class_name] = re.findall(r'retVal\.add\("([^"]+)"\);', body)
    return scopes


@lru_cache(maxsize=None)
def potion_pool(player_class: str = "IRONCLAD") -> list[str]:
    return [*potion_scopes().get(str(player_class), []), *potion_scopes()["SHARED"]]


@lru_cache(maxsize=1)
def ironclad_potion_pool() -> list[str]:
    return list(potion_pool("IRONCLAD"))


def make_potion(potion_id: str) -> dict[str, object]:
    rarity = potion_rarity_map().get(potion_id, "COMMON")
    return {
        "potion_id": potion_id,
        "id": potion_id,
        "name": potion_name_map().get(potion_id, potion_id),
        "rarity": rarity,
        "can_use": True,
        "can_use_out_of_combat": potion_id in OUT_OF_COMBAT_USABLE_POTION_IDS,
        "can_discard": True,
        "requires_target": bool(potion_requires_target_map().get(potion_id, False)),
        "price": potion_price_for_rarity(rarity),
    }


def potion_priority_value(potion: str | dict[str, object]) -> int:
    """Value used when a full potion belt must discard one potion.

    Prefer hand-ranked replacement priority. Potions that are deliberately not
    considered use explicit low values, then unknown potions fall back by rarity.
    Fairy in a Bottle is intentionally pinned above every normal potion.
    """
    if isinstance(potion, dict):
        potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
        rarity = str(potion.get("rarity") or potion_rarity_map().get(potion_id, "COMMON"))
    else:
        potion_id = str(potion)
        rarity = potion_rarity_map().get(potion_id, "COMMON")
    if potion_id == "Potion Slot":
        return -1
    if potion_id == "FairyPotion":
        return 1_000_000
    if potion_id in TRACE_REWARD_POTION_PRIORITY:
        return int(TRACE_REWARD_POTION_PRIORITY[potion_id])
    if potion_id in IMPUTED_POTION_PRIORITY:
        return int(IMPUTED_POTION_PRIORITY[potion_id])
    return int(RARITY_FALLBACK_POTION_PRIORITY.get(rarity, 1100))


def draw_random_potion(
    randoms,
    *,
    player_class: str = "IRONCLAD",
    stream_name: str = "potion",
) -> dict[str, object]:
    pool = list(potion_pool(player_class))
    if not pool:
        raise RuntimeError(f"native_sim_v3 has no potion pool for player class {player_class!r}")
    potion_id = str(pool[int(randoms.stream(stream_name).random(len(pool) - 1))])
    return make_potion(potion_id)


def _roll_potion_rarity(randoms) -> str:
    rules = potion_roll_rules()
    roll = int(randoms.stream("potion").random(0, 99))
    if roll < rules.common_chance:
        return "COMMON"
    if roll < rules.common_chance + rules.uncommon_chance:
        return "UNCOMMON"
    return "RARE"


def roll_random_potion(randoms, *, player_class: str = "IRONCLAD", limited: bool = False) -> dict[str, object]:
    rarity = _roll_potion_rarity(randoms)
    pool = list(potion_pool(player_class))
    if not pool:
        raise RuntimeError(f"native_sim_v3 has no potion pool for player class {player_class!r}")
    rarity_map = potion_rarity_map()
    spam_check = bool(limited)
    potion_id = pool[int(randoms.stream("potion").random(len(pool) - 1))]
    while True:
        potion_rarity = rarity_map.get(potion_id)
        if potion_rarity == rarity and not spam_check:
            return make_potion(potion_id)
        spam_check = bool(limited)
        potion_id = pool[int(randoms.stream("potion").random(len(pool) - 1))]
        if potion_id == "Fruit Juice":
            continue
        spam_check = False
