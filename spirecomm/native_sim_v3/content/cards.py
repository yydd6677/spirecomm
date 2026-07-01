from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from spirecomm.native_sim_v3.content.characters import starting_profile
from spirecomm.native_sim_v3.source_paths import sts_localization_path, sts_source_path

CARD_ROOT = sts_source_path("cards")
CARD_LIBRARY_SOURCE = sts_source_path("helpers/CardLibrary.java")
CARD_STRINGS_SOURCE = sts_localization_path("cards.json")
CARD_DIRS = {
    "red": "RED",
    "green": "GREEN",
    "blue": "BLUE",
    "purple": "PURPLE",
    "colorless": "COLORLESS",
    "curses": "CURSE",
    "status": "STATUS",
}
PLAYER_CLASS_TO_CARD_COLOR = {
    "IRONCLAD": "RED",
    "THE_SILENT": "GREEN",
    "DEFECT": "BLUE",
    "WATCHER": "PURPLE",
}

SUPER_PATTERN = re.compile(
    r"super\(\s*ID\s*,\s*.*?,\s*\"[^\"]+\"\s*,\s*([^,]+)\s*,\s*.*?,\s*"
    r"AbstractCard\.CardType\.([A-Z_]+)\s*,\s*AbstractCard\.CardColor\.([A-Z_]+)\s*,\s*"
    r"AbstractCard\.CardRarity\.([A-Z_]+)\s*,\s*AbstractCard\.CardTarget\.([A-Z_]+)\s*\);",
    re.DOTALL,
)
ID_PATTERN = re.compile(r'public static final String ID = "([^"]+)";')
NAME_PATTERN = re.compile(r'getCardStrings\("([^"]+)"\)')
INT_ASSIGNMENTS = {
    "baseDamage": re.compile(r"this\.baseDamage\s*=\s*(\d+);"),
    "baseBlock": re.compile(r"this\.baseBlock\s*=\s*(\d+);"),
    "baseMagicNumber": re.compile(r"(?:this\.magicNumber\s*=\s*)?this\.baseMagicNumber\s*=\s*(\d+);"),
}
BOOL_ASSIGNMENTS = {
    "exhausts": re.compile(r"this\.exhaust\s*=\s*true;"),
    "ethereal": re.compile(r"this\.isEthereal\s*=\s*true;"),
    "innate": re.compile(r"this\.isInnate\s*=\s*true;"),
    "retain": re.compile(r"this\.selfRetain\s*=\s*true;"),
}
TAG_ASSIGNMENT_PATTERN = re.compile(r"this\.tags\.add\(AbstractCard\.CardTags\.([A-Z_]+)\);")
UPGRADE_COST_PATTERN = re.compile(r"this\.upgradeBaseCost\(([-]?\d+)\);")
UPGRADE_DAMAGE_PATTERN = re.compile(r"this\.upgradeDamage\(([-]?\d+)\);")
UPGRADE_BLOCK_PATTERN = re.compile(r"this\.upgradeBlock\(([-]?\d+)\);")
UPGRADE_MAGIC_PATTERN = re.compile(r"this\.upgradeMagicNumber\(([-]?\d+)\);")
UPGRADE_EXHAUST_PATTERN = re.compile(r"this\.exhaust\s*=\s*false;")
UPGRADE_ETHEREAL_PATTERN = re.compile(r"this\.isEthereal\s*=\s*false;")
TARGETED_TYPES = {"ENEMY", "SELF_AND_ENEMY"}
DEFAULT_UPGRADE_COST = object()
CARD_LIBRARY_METHOD_PATTERN = re.compile(
    r"private static void (addRedCards|addGreenCards|addBlueCards|addPurpleCards|addColorlessCards|addCurseCards)\(\) \{(.*?)^\s*\}",
    re.DOTALL | re.MULTILINE,
)
CARD_LIBRARY_ADD_PATTERN = re.compile(r"CardLibrary\.add\(new ([A-Za-z0-9_]+)\(\)\);")
EXCLUDED_CURSE_SOURCE_IDS = {"Necronomicurse", "AscendersBane", "CurseOfTheBell", "Pride"}
HEALING_CARD_IDS = {"Bandage Up", "Feed", "Reaper", "Self Repair"}
MULTI_UPGRADE_CARD_IDS = {"Searing Blow"}
UPGRADE_REMOVES_EXHAUST_CARD_IDS = {
    "Discovery",
    "Limit Break",
    "Secret Technique",
    "Secret Weapon",
    "Thinking Ahead",
}
UPGRADE_CHANGES_TARGET_TO_ALL_ENEMY_CARD_IDS = {"Blind", "Trip"}


@dataclass(frozen=True, slots=True)
class CardDef:
    card_id: str
    name: str
    color: str
    type: str
    rarity: str
    target: str
    cost: int | None
    upgraded_cost: int | None
    base_damage: int = 0
    base_block: int = 0
    base_magic: int = 0
    upgraded_damage: int = 0
    upgraded_block: int = 0
    upgraded_magic: int = 0
    exhausts: bool = False
    ethereal: bool = False
    innate: bool = False
    upgraded_innate: bool = False
    retain: bool = False
    upgraded_ethereal: bool = False
    tags: tuple[str, ...] = ()


def _parse_cost(raw: str) -> int | None:
    raw = raw.strip()
    if raw in {"-1", "-2"}:
        return int(raw)
    if raw == "X_COST":
        return -1
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_card_file(path: Path) -> CardDef | None:
    text = path.read_text(encoding="utf-8")
    id_match = ID_PATTERN.search(text)
    super_match = SUPER_PATTERN.search(text)
    if not id_match or not super_match:
        return None
    card_id = id_match.group(1)
    name_match = NAME_PATTERN.search(text)
    name = _card_display_names().get(card_id, name_match.group(1) if name_match else card_id)
    cost = _parse_cost(super_match.group(1))
    card_type = super_match.group(2)
    color = super_match.group(3)
    rarity = super_match.group(4)
    target = super_match.group(5)
    upgraded_cost: int | None | object = DEFAULT_UPGRADE_COST
    upgrade_cost_match = UPGRADE_COST_PATTERN.search(text)
    if upgrade_cost_match:
        upgraded_cost = int(upgrade_cost_match.group(1))
    if upgraded_cost is DEFAULT_UPGRADE_COST:
        upgraded_cost = cost
    base_text = text.split("public void upgrade", 1)[0]
    exhausts = BOOL_ASSIGNMENTS["exhausts"].search(text) is not None
    if BOOL_ASSIGNMENTS["exhausts"].search(text) is not None and UPGRADE_EXHAUST_PATTERN.search(text):
        # cards like Seeing Red can change exhaust behavior on upgrade; the
        # current parser keeps base truth only and leaves upgrade-specific
        # behavior to explicit effect implementations later.
        pass
    base_damage = int(INT_ASSIGNMENTS["baseDamage"].search(text).group(1)) if INT_ASSIGNMENTS["baseDamage"].search(text) else 0
    base_block = int(INT_ASSIGNMENTS["baseBlock"].search(text).group(1)) if INT_ASSIGNMENTS["baseBlock"].search(text) else 0
    base_magic = int(INT_ASSIGNMENTS["baseMagicNumber"].search(text).group(1)) if INT_ASSIGNMENTS["baseMagicNumber"].search(text) else 0
    upgraded_damage = base_damage
    upgrade_damage_match = UPGRADE_DAMAGE_PATTERN.search(text)
    if upgrade_damage_match:
        upgraded_damage += int(upgrade_damage_match.group(1))
    upgraded_block = base_block
    upgrade_block_match = UPGRADE_BLOCK_PATTERN.search(text)
    if upgrade_block_match:
        upgraded_block += int(upgrade_block_match.group(1))
    upgraded_magic = base_magic
    upgrade_magic_match = UPGRADE_MAGIC_PATTERN.search(text)
    if upgrade_magic_match:
        upgraded_magic += int(upgrade_magic_match.group(1))
    tags = tuple(TAG_ASSIGNMENT_PATTERN.findall(text))
    innate = BOOL_ASSIGNMENTS["innate"].search(base_text) is not None
    upgraded_innate = innate or BOOL_ASSIGNMENTS["innate"].search(text) is not None
    ethereal = BOOL_ASSIGNMENTS["ethereal"].search(text) is not None
    upgraded_ethereal = ethereal and UPGRADE_ETHEREAL_PATTERN.search(text) is None
    return CardDef(
        card_id=card_id,
        name=name,
        color=color,
        type=card_type,
        rarity=rarity,
        target=target,
        cost=cost,
        upgraded_cost=upgraded_cost if upgraded_cost is None else int(upgraded_cost),
        base_damage=base_damage,
        base_block=base_block,
        base_magic=base_magic,
        upgraded_damage=upgraded_damage,
        upgraded_block=upgraded_block,
        upgraded_magic=upgraded_magic,
        exhausts=exhausts,
        ethereal=ethereal,
        innate=innate,
        upgraded_innate=upgraded_innate,
        retain=BOOL_ASSIGNMENTS["retain"].search(text) is not None,
        upgraded_ethereal=upgraded_ethereal,
        tags=tags,
    )


@lru_cache(maxsize=1)
def _card_display_names() -> dict[str, str]:
    if not CARD_STRINGS_SOURCE.exists():
        return {}
    data = json.loads(CARD_STRINGS_SOURCE.read_text(encoding="utf-8"))
    return {
        str(card_id): str(payload["NAME"])
        for card_id, payload in data.items()
        if isinstance(payload, dict) and payload.get("NAME")
    }


@lru_cache(maxsize=1)
def card_catalog() -> dict[str, CardDef]:
    catalog: dict[str, CardDef] = {}
    for dirname in CARD_DIRS:
        directory = CARD_ROOT / dirname
        for path in sorted(directory.glob("*.java")):
            card = _parse_card_file(path)
            if card is not None:
                catalog[card.card_id] = card
    return catalog


@lru_cache(maxsize=1)
def _card_library_order() -> dict[str, list[str]]:
    text = CARD_LIBRARY_SOURCE.read_text(encoding="utf-8")
    by_stem = {}
    for dirname in CARD_DIRS:
        for path in (CARD_ROOT / dirname).glob("*.java"):
            card = _parse_card_file(path)
            if card is not None:
                by_stem[path.stem] = card.card_id
    result: dict[str, list[str]] = {
        "RED": [],
        "GREEN": [],
        "BLUE": [],
        "PURPLE": [],
        "COLORLESS": [],
        "CURSE": [],
    }
    for method_name, body in CARD_LIBRARY_METHOD_PATTERN.findall(text):
        if method_name == "addRedCards":
            color_key = "RED"
        elif method_name == "addGreenCards":
            color_key = "GREEN"
        elif method_name == "addBlueCards":
            color_key = "BLUE"
        elif method_name == "addPurpleCards":
            color_key = "PURPLE"
        elif method_name == "addColorlessCards":
            color_key = "COLORLESS"
        else:
            color_key = "CURSE"
        order: list[str] = []
        for class_name in CARD_LIBRARY_ADD_PATTERN.findall(body):
            card_id = by_stem.get(class_name)
            if card_id is not None:
                order.append(card_id)
        result[color_key] = order
    return result


@lru_cache(maxsize=1)
def _card_library_insert_order() -> list[str]:
    text = CARD_LIBRARY_SOURCE.read_text(encoding="utf-8")
    by_stem = {}
    for dirname in CARD_DIRS:
        for path in (CARD_ROOT / dirname).glob("*.java"):
            card = _parse_card_file(path)
            if card is not None:
                by_stem[path.stem] = card.card_id
    return [by_stem.get(class_name, class_name) for class_name in CARD_LIBRARY_ADD_PATTERN.findall(text)]


def _java_string_hashcode(value: str) -> int:
    result = 0
    for char in value:
        result = (31 * result + ord(char)) & 0xFFFFFFFF
    return result


def _java_hashmap_spread(hashcode: int) -> int:
    return (hashcode ^ (hashcode >> 16)) & 0xFFFFFFFF


def _java_hashmap_iteration_order(keys: list[str]) -> tuple[str, ...]:
    capacity = 0
    threshold = 0
    table: list[list[tuple[int, str]]] = []
    size = 0

    def resize() -> None:
        nonlocal capacity, threshold, table
        old_table = table
        old_capacity = capacity
        old_threshold = threshold
        if old_capacity > 0:
            new_capacity = old_capacity * 2
            new_threshold = old_threshold * 2
        elif old_threshold > 0:
            new_capacity = old_threshold
            new_threshold = int(new_capacity * 0.75)
        else:
            new_capacity = 16
            new_threshold = 12
        new_table: list[list[tuple[int, str]]] = [[] for _ in range(new_capacity)]
        if old_capacity > 0:
            for bucket in old_table:
                if not bucket:
                    continue
                if len(bucket) == 1:
                    spread_hash, key = bucket[0]
                    new_table[spread_hash & (new_capacity - 1)].append((spread_hash, key))
                    continue
                bucket_index = bucket[0][0] & (old_capacity - 1)
                lo_bucket: list[tuple[int, str]] = []
                hi_bucket: list[tuple[int, str]] = []
                for spread_hash, key in bucket:
                    if spread_hash & old_capacity:
                        hi_bucket.append((spread_hash, key))
                    else:
                        lo_bucket.append((spread_hash, key))
                if lo_bucket:
                    new_table[bucket_index] = lo_bucket
                if hi_bucket:
                    new_table[bucket_index + old_capacity] = hi_bucket
        capacity = new_capacity
        threshold = new_threshold
        table = new_table

    for key in keys:
        spread_hash = _java_hashmap_spread(_java_string_hashcode(key))
        if capacity == 0:
            resize()
        bucket_index = spread_hash & (capacity - 1)
        bucket = table[bucket_index]
        if any(existing_hash == spread_hash and existing_key == key for existing_hash, existing_key in bucket):
            continue
        bucket.append((spread_hash, key))
        size += 1
        if size > threshold:
            resize()
    return tuple(key for bucket in table for _, key in bucket)


@lru_cache(maxsize=1)
def _card_library_runtime_entry_order() -> tuple[str, ...]:
    # CardLibrary.cards is populated via plain HashMap.put(cardID, card). STS
    # later builds reward pools by iterating cards.entrySet() and calling
    # CardGroup.addToTop(), which in Slay the Spire appends to the underlying
    # list. The pool order we want therefore matches raw HashMap bucket
    # iteration order, not the static add order from CardLibrary.java.
    return _java_hashmap_iteration_order(_card_library_insert_order())


@lru_cache(maxsize=1)
def card_library_random_curse_pool() -> list[str]:
    # AbstractDungeon.returnRandomCurse delegates to CardLibrary.getCurse(),
    # which iterates CardLibrary.curses, a separate HashMap from CardLibrary.cards.
    return [
        card_id
        for card_id in _java_hashmap_iteration_order(_card_library_order()["CURSE"])
        if card_id not in EXCLUDED_CURSE_SOURCE_IDS
    ]


@lru_cache(maxsize=None)
def _all_card_pools() -> dict[str, list[str]]:
    catalog = card_catalog()
    runtime_entry_order = _card_library_runtime_entry_order()
    pools = {
        "RED_COMMON": [],
        "RED_UNCOMMON": [],
        "RED_RARE": [],
        "GREEN_COMMON": [],
        "GREEN_UNCOMMON": [],
        "GREEN_RARE": [],
        "BLUE_COMMON": [],
        "BLUE_UNCOMMON": [],
        "BLUE_RARE": [],
        "PURPLE_COMMON": [],
        "PURPLE_UNCOMMON": [],
        "PURPLE_RARE": [],
        "ANY_COLOR_COMMON": [],
        "ANY_COLOR_UNCOMMON": [],
        "ANY_COLOR_RARE": [],
        "COLORLESS_UNCOMMON": [],
        "COLORLESS_RARE": [],
        "COLORLESS_ALL": [],
        "CURSE": [],
        "STATUS": [],
    }
    seen: dict[str, set[str]] = {key: set() for key in pools}

    for color_key in ("RED", "GREEN", "BLUE", "PURPLE"):
        # CardLibrary.add*Cards(ArrayList) iterates CardLibrary.cards.entrySet().
        # The static add*Cards() methods only populate that HashMap.
        for card_id in runtime_entry_order:
            card = catalog.get(card_id)
            if card is None or card.color != color_key or card.rarity not in {"COMMON", "UNCOMMON", "RARE"}:
                continue
            bucket = f"{color_key}_{card.rarity}"
            pools[bucket].append(card_id)
            seen[bucket].add(card_id)

    for card_id in runtime_entry_order:
        card = catalog.get(card_id)
        if card is None:
            continue
        if card.type not in {"CURSE", "STATUS"} and card.rarity in {"COMMON", "UNCOMMON", "RARE"}:
            bucket = f"ANY_COLOR_{card.rarity}"
            pools[bucket].append(card_id)
            seen[bucket].add(card_id)
        if card.type == "STATUS":
            pools["STATUS"].append(card_id)
            seen["STATUS"].add(card_id)
        elif card.color == "COLORLESS" and card.rarity in {"UNCOMMON", "RARE"}:
            bucket = f"COLORLESS_{card.rarity}"
            pools[bucket].append(card_id)
            seen[bucket].add(card_id)
            pools["COLORLESS_ALL"].append(card_id)
            seen["COLORLESS_ALL"].add(card_id)

    for card_id in runtime_entry_order:
        card = catalog.get(card_id)
        if card is not None and card.color == "CURSE":
            pools["CURSE"].append(card_id)
            seen["CURSE"].add(card_id)

    for card_id, card in catalog.items():
        if card.color in {"RED", "GREEN", "BLUE", "PURPLE"} and card.rarity in {"COMMON", "UNCOMMON", "RARE"}:
            bucket = f"{card.color}_{card.rarity}"
            if card_id not in seen[bucket]:
                pools[bucket].append(card_id)
            any_bucket = f"ANY_COLOR_{card.rarity}"
            if card_id not in seen[any_bucket]:
                pools[any_bucket].append(card_id)
        elif card.color == "COLORLESS" and card.rarity in {"UNCOMMON", "RARE"}:
            bucket = f"COLORLESS_{card.rarity}"
            if card_id not in seen[bucket]:
                pools[bucket].append(card_id)
            if card_id not in seen["COLORLESS_ALL"]:
                pools["COLORLESS_ALL"].append(card_id)
            any_bucket = f"ANY_COLOR_{card.rarity}"
            if card_id not in seen[any_bucket]:
                pools[any_bucket].append(card_id)
        elif card.color == "CURSE" and card_id not in seen["CURSE"]:
            pools["CURSE"].append(card_id)
        elif card.type == "STATUS" and card_id not in seen["STATUS"]:
            pools["STATUS"].append(card_id)
    for rarity in ("COMMON", "UNCOMMON", "RARE"):
        pools[f"ANY_COLOR_{rarity}"] = sorted(dict.fromkeys(pools[f"ANY_COLOR_{rarity}"]))
    return pools


@lru_cache(maxsize=None)
def card_pools(character: str = "IRONCLAD") -> dict[str, list[str]]:
    class_color = PLAYER_CLASS_TO_CARD_COLOR.get(str(character))
    if class_color is None:
        raise NotImplementedError(f"native_sim_v3 has no card pool mapping for character {character!r}.")
    all_pools = _all_card_pools()
    pools = {key: list(values) for key, values in all_pools.items()}
    pools["CLASS_COMMON"] = list(all_pools[f"{class_color}_COMMON"])
    pools["CLASS_UNCOMMON"] = list(all_pools[f"{class_color}_UNCOMMON"])
    pools["CLASS_RARE"] = list(all_pools[f"{class_color}_RARE"])
    # Backward-compatible aliases for the current model/runtime surface.
    pools["RED_COMMON"] = list(pools["CLASS_COMMON"])
    pools["RED_UNCOMMON"] = list(pools["CLASS_UNCOMMON"])
    pools["RED_RARE"] = list(pools["CLASS_RARE"])
    return pools


@lru_cache(maxsize=None)
def source_card_pools(character: str = "IRONCLAD") -> dict[str, list[str]]:
    pools = card_pools(character)
    return {
        "SRC_COMMON": list(reversed(pools["CLASS_COMMON"])),
        "SRC_UNCOMMON": list(reversed(pools["CLASS_UNCOMMON"])),
        "SRC_RARE": list(reversed(pools["CLASS_RARE"])),
        "SRC_COLORLESS": list(reversed(pools["COLORLESS_ALL"])),
        "SRC_CURSE": list(reversed([card_id for card_id in pools["CURSE"] if card_id not in EXCLUDED_CURSE_SOURCE_IDS])),
    }


def initialize_runtime_card_pools(character: str = "IRONCLAD") -> dict[str, list[str]]:
    return {key: list(values) for key, values in card_pools(character).items()}


def initialize_source_card_pools(runtime_pools: dict[str, list[str]] | None = None, *, character: str = "IRONCLAD") -> dict[str, list[str]]:
    pools = runtime_pools or card_pools(character)
    return {
        "SRC_COMMON": list(reversed(pools.get("CLASS_COMMON", pools.get("RED_COMMON", [])))),
        "SRC_UNCOMMON": list(reversed(pools.get("CLASS_UNCOMMON", pools.get("RED_UNCOMMON", [])))),
        "SRC_RARE": list(reversed(pools.get("CLASS_RARE", pools.get("RED_RARE", [])))),
        "SRC_COLORLESS": list(
            reversed(
                list(
                    pools.get(
                        "COLORLESS_ALL",
                        [*list(pools.get("COLORLESS_UNCOMMON", [])), *list(pools.get("COLORLESS_RARE", []))],
                    )
                )
            )
        ),
        "SRC_CURSE": list(reversed([card_id for card_id in list(pools.get("CURSE", [])) if card_id not in EXCLUDED_CURSE_SOURCE_IDS])),
    }


def class_reward_pool_key(rarity: str) -> str:
    return f"CLASS_{str(rarity)}"


def truly_random_card_from_source_pools(
    randoms: Any,
    *,
    source_pools: dict[str, list[str]] | None = None,
    include_colorless: bool = False,
    prohibited_id: str | None = None,
    color_mode: str | None = None,
    card_type: str | None = None,
    in_combat: bool = False,
    rng_stream: str = "card_random",
) -> dict[str, Any] | None:
    pools = source_pools or source_card_pools()
    catalog = card_catalog()
    candidates: list[str] = []
    if color_mode == "COLORLESS":
        candidates.extend(pools.get("SRC_COLORLESS", []))
    elif color_mode == "CURSE":
        candidates.extend(pools.get("SRC_CURSE", []))
    else:
        candidates.extend(pools.get("SRC_COMMON", []))
        candidates.extend(pools.get("SRC_UNCOMMON", []))
        candidates.extend(pools.get("SRC_RARE", []))
        if include_colorless:
            candidates.extend(pools.get("SRC_COLORLESS", []))
    if prohibited_id is not None:
        candidates = [card_id for card_id in candidates if card_id != prohibited_id]
    filtered: list[str] = []
    for card_id in candidates:
        card = catalog.get(card_id)
        if card is None:
            continue
        if card_type is not None and str(card.type) != str(card_type):
            continue
        if in_combat and card_id in HEALING_CARD_IDS:
            continue
        filtered.append(card_id)
    if not filtered:
        return None
    pick_index = int(randoms.stream(rng_stream).random(0, len(filtered) - 1))
    return make_card(filtered[pick_index], uuid=f"source-random-{filtered[pick_index]}")


def make_card(card_id: str, *, upgrades: int = 0, uuid: str | None = None) -> dict[str, Any]:
    card = card_catalog()[card_id]
    effective_cost = card.upgraded_cost if upgrades > 0 else card.cost
    effective_damage = card.upgraded_damage if upgrades > 0 else card.base_damage
    effective_block = card.upgraded_block if upgrades > 0 else card.base_block
    effective_magic = card.upgraded_magic if upgrades > 0 else card.base_magic
    effective_exhausts = bool(card.exhausts)
    effective_target = card.target
    if upgrades > 0 and card.card_id in UPGRADE_REMOVES_EXHAUST_CARD_IDS:
        effective_exhausts = False
    if upgrades > 0 and card.card_id in UPGRADE_CHANGES_TARGET_TO_ALL_ENEMY_CARD_IDS:
        effective_target = "ALL_ENEMY"
    return {
        "card_id": card.card_id,
        "name": card.name,
        "type": card.type,
        "rarity": card.rarity,
        "cost": effective_cost,
        "base_cost": effective_cost,
        "cost_for_turn": None,
        "cost_for_combat": None,
        "free_to_play_once": False,
        "upgrades": upgrades,
        "misc": 0,
        "exhausts": effective_exhausts,
        "has_target": effective_target in TARGETED_TYPES,
        "is_playable": False,
        "uuid": uuid or f"{card.card_id}-{uuid4().hex[:12]}",
        "base_damage": effective_damage,
        "base_block": effective_block,
        "base_magic": effective_magic,
        "ethereal": card.upgraded_ethereal if upgrades > 0 else card.ethereal,
        "innate": card.upgraded_innate if upgrades > 0 else card.innate,
        "retain": card.retain,
        "tags": list(card.tags),
        "color": card.color,
        "target": effective_target,
    }


def can_upgrade_card(card: dict[str, Any]) -> bool:
    can_upgrade = card.get("can_upgrade")
    if can_upgrade is False or str(can_upgrade or "").lower() == "false":
        return False
    if str(card.get("type") or "") in {"STATUS", "CURSE"}:
        return False
    if str(card.get("card_id") or card.get("id") or "") in MULTI_UPGRADE_CARD_IDS:
        return True
    return int(card.get("upgrades") or 0) <= 0


def upgrade_card(card: dict[str, Any]) -> dict[str, Any]:
    if not can_upgrade_card(card):
        return dict(card)
    upgraded = make_card(
        str(card["card_id"]),
        upgrades=int(card.get("upgrades") or 0) + 1,
        uuid=str(card.get("uuid") or f"upgrade-{card['card_id']}"),
    )
    upgraded["misc"] = card.get("misc", upgraded.get("misc", 0))
    upgraded["free_to_play_once"] = bool(card.get("free_to_play_once", upgraded.get("free_to_play_once", False)))
    if "_reset_cost_for_turn_after_play" in card:
        upgraded["_reset_cost_for_turn_after_play"] = card.get("_reset_cost_for_turn_after_play")
    if "can_upgrade" in card:
        upgraded["can_upgrade"] = card.get("can_upgrade")
    old_effective_cost = card.get("cost")
    old_base_cost = card.get("base_cost")
    if old_base_cost is None:
        old_base_cost = old_effective_cost
    if str(card.get("card_id") or "") == "Blood for Blood" and isinstance(old_effective_cost, int) and old_effective_cost < 4:
        upgraded_cost = max(0, old_effective_cost - 1)
        upgraded["cost"] = upgraded_cost
        upgraded["base_cost"] = upgraded_cost
    for field in ("cost_for_combat", "cost_for_turn"):
        existing = card.get(field)
        if existing is None:
            continue
        if existing == old_base_cost and old_effective_cost == old_base_cost:
            upgraded[field] = upgraded.get("cost")
        else:
            upgraded[field] = existing
    return upgraded


def starter_deck(character: str = "IRONCLAD") -> list[dict[str, Any]]:
    starter_ids = starting_profile(character).starter_deck_ids
    return [make_card(card_id, uuid=f"starter-{index}-{card_id}") for index, card_id in enumerate(starter_ids)]
