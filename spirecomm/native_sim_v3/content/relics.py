from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.content.characters import starting_profile
from spirecomm.native_sim_v3.content.pricing import relic_price_for_tier
from spirecomm.native_sim_v3.core.randoms import NativeRandomSet, java_shuffle_in_place
from spirecomm.native_sim_v3.source_paths import GAME_JAR_PATH, sts_localization_path, sts_source_path

RELIC_ROOT = sts_source_path("relics")
RELIC_LIBRARY_PATH = sts_source_path("helpers/RelicLibrary.java")
ABSTRACT_DUNGEON_PATH = sts_source_path("dungeons/AbstractDungeon.java")

ID_PATTERN = re.compile(r'public static final String ID = "([^"]+)";')
NAME_FALLBACK_PATTERN = re.compile(r"class\s+([A-Za-z0-9_]+)\s+extends\s+AbstractRelic")
SUPER_PATTERN = re.compile(
    r"super\(\s*ID\s*,\s*\"[^\"]+\"\s*,\s*AbstractRelic\.RelicTier\.([A-Z_]+)\s*,\s*AbstractRelic\.LandingSound\.([A-Z_]+)\s*\);",
    re.DOTALL,
)
RELIC_LIBRARY_ADD_PATTERN = re.compile(r"RelicLibrary\.add(Red|Green|Blue|Purple)?\(new\s+([A-Za-z0-9_]+)\(\)\);")
OBJECT_EQUALS_RELIC_PATTERN = re.compile(r'Objects\.equals\(tmpRelic\.relicId,\s*"([^"]+)"\)')
HAS_RELIC_PATTERN = re.compile(r'return AbstractDungeon\.player\.hasRelic\("([^"]+)"\);')
FLOOR_ONLY_PATTERN = re.compile(r'return Settings\.isEndless \|\| AbstractDungeon\.floorNum <= (\d+);')
FLOOR_AND_SHOP_PATTERN = re.compile(
    r'return \(Settings\.isEndless \|\| AbstractDungeon\.floorNum <= (\d+)\) && !\(AbstractDungeon\.getCurrRoom\(\) instanceof ShopRoom\);'
)
ACT_MAX_PATTERN = re.compile(r'return AbstractDungeon\.actNum <= (\d+);')
CARD_TYPE_NON_BASIC_PATTERN = re.compile(
    r'if \(c\.type != AbstractCard\.CardType\.([A-Z_]+) \|\| c\.rarity == AbstractCard\.CardRarity\.BASIC\) continue;'
)
CARD_TYPE_PATTERN = re.compile(r'CardHelper\.hasCardType\(AbstractCard\.CardType\.([A-Z_]+)\)')
FLOOR_GTE_BLOCK_PATTERN = re.compile(r'if \(AbstractDungeon\.floorNum >= (\d+) && !Settings\.isEndless\)')
CAMPFIRE_CAP_PATTERN = re.compile(r'return campfireRelicCount < (\d+);')

RELIC_POOL_ORDER = ("COMMON", "UNCOMMON", "RARE", "SHOP", "BOSS")
BANNED_RELIC_IDS = frozenset({"PrismaticShard"})
PLAYER_CLASS_SCOPE = {
    None: "SHARED",
    "Red": "IRONCLAD",
    "Green": "THE_SILENT",
    "Blue": "DEFECT",
    "Purple": "WATCHER",
}

JAVA_HASHMAP_DEFAULT_CAPACITY = 16
JAVA_HASHMAP_LOAD_FACTOR = 0.75


def is_banned_relic_id(relic_id: object) -> bool:
    return str(relic_id) in BANNED_RELIC_IDS


@dataclass(frozen=True, slots=True)
class RelicDef:
    relic_id: str
    name: str
    tier: str
    landing_sound: str
    source_path: str


@dataclass(frozen=True, slots=True)
class RelicSpawnRule:
    relic_id: str
    requires_relic_id: str | None = None
    max_floor: int | None = None
    max_act: int | None = None
    exclude_shop_room: bool = False
    campfire_relic_cap: int | None = None
    requires_non_basic_card_type: str | None = None
    requires_card_type: str | None = None
    source_path: str = ""


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


@lru_cache(maxsize=1)
def _localized_relic_names() -> dict[str, str]:
    localization_path = sts_localization_path("relics.json")
    if localization_path.exists():
        data = json.loads(localization_path.read_text(encoding="utf-8"))
    elif GAME_JAR_PATH.exists():
        with zipfile.ZipFile(GAME_JAR_PATH) as archive:
            with archive.open("localization/eng/relics.json") as handle:
                data = json.loads(handle.read().decode("utf-8"))
    else:
        return {}
    names: dict[str, str] = {}
    for relic_id, payload in data.items():
        if isinstance(payload, dict) and payload.get("NAME"):
            names[str(relic_id)] = str(payload["NAME"])
    return names


def _parse_relic_file(path: Path) -> RelicDef | None:
    text = path.read_text(encoding="utf-8")
    id_match = ID_PATTERN.search(text)
    super_match = SUPER_PATTERN.search(text)
    if not id_match or not super_match:
        return None
    relic_id = id_match.group(1)
    name_match = NAME_FALLBACK_PATTERN.search(text)
    return RelicDef(
        relic_id=relic_id,
        name=_localized_relic_names().get(relic_id, relic_id if relic_id else (name_match.group(1) if name_match else path.stem)),
        tier=super_match.group(1),
        landing_sound=super_match.group(2),
        source_path=str(path),
    )


@lru_cache(maxsize=1)
def relic_catalog() -> dict[str, RelicDef]:
    catalog: dict[str, RelicDef] = {}
    for path in sorted(RELIC_ROOT.glob("*.java")):
        relic = _parse_relic_file(path)
        if relic is not None:
            catalog[relic.relic_id] = relic
    return catalog


@lru_cache(maxsize=1)
def screenless_excluded_relic_ids() -> tuple[str, ...]:
    body = _extract_method_body(ABSTRACT_DUNGEON_PATH.read_text(encoding="utf-8"), "returnRandomScreenlessRelic", ABSTRACT_DUNGEON_PATH)
    return tuple(OBJECT_EQUALS_RELIC_PATTERN.findall(body))


@lru_cache(maxsize=1)
def non_campfire_excluded_relic_ids() -> tuple[str, ...]:
    body = _extract_method_body(ABSTRACT_DUNGEON_PATH.read_text(encoding="utf-8"), "returnRandomNonCampfireRelic", ABSTRACT_DUNGEON_PATH)
    return tuple(OBJECT_EQUALS_RELIC_PATTERN.findall(body))


@lru_cache(maxsize=1)
def relic_spawn_rules() -> dict[str, RelicSpawnRule]:
    rules: dict[str, RelicSpawnRule] = {}
    for relic_id, relic in relic_catalog().items():
        source_path = Path(relic.source_path)
        text = source_path.read_text(encoding="utf-8")
        try:
            body = _extract_method_body(text, "canSpawn", source_path)
        except ValueError:
            continue
        requires_relic_match = HAS_RELIC_PATTERN.search(body)
        floor_shop_match = FLOOR_AND_SHOP_PATTERN.search(body)
        floor_only_match = FLOOR_ONLY_PATTERN.search(body)
        act_match = ACT_MAX_PATTERN.search(body)
        non_basic_type_match = CARD_TYPE_NON_BASIC_PATTERN.search(body)
        card_type_match = CARD_TYPE_PATTERN.search(body)
        floor_gte_match = FLOOR_GTE_BLOCK_PATTERN.search(body)
        campfire_cap_match = CAMPFIRE_CAP_PATTERN.search(body)
        if not any(
            (
                requires_relic_match,
                floor_shop_match,
                floor_only_match,
                act_match,
                non_basic_type_match,
                card_type_match,
                campfire_cap_match,
            )
        ):
            continue
        max_floor = None
        exclude_shop_room = False
        campfire_cap = None
        if floor_shop_match is not None:
            max_floor = int(floor_shop_match.group(1))
            exclude_shop_room = True
        elif floor_only_match is not None:
            max_floor = int(floor_only_match.group(1))
        elif floor_gte_match is not None:
            max_floor = int(floor_gte_match.group(1)) - 1
        if campfire_cap_match is not None:
            campfire_cap = int(campfire_cap_match.group(1))
        rules[relic_id] = RelicSpawnRule(
            relic_id=relic_id,
            requires_relic_id=requires_relic_match.group(1) if requires_relic_match is not None else None,
            max_floor=max_floor,
            max_act=int(act_match.group(1)) if act_match is not None else None,
            exclude_shop_room=exclude_shop_room,
            campfire_relic_cap=campfire_cap,
            requires_non_basic_card_type=non_basic_type_match.group(1) if non_basic_type_match is not None else None,
            requires_card_type=card_type_match.group(1) if card_type_match is not None else None,
            source_path=str(source_path),
        )
    return rules


@lru_cache(maxsize=1)
def relic_scope_ids() -> dict[str, set[str]]:
    return {scope: set(relic_ids) for scope, relic_ids in relic_scope_order_ids().items()}


def _java_string_hash(value: str) -> int:
    result = 0
    for char in value:
        result = (31 * result + ord(char)) & 0xFFFFFFFF
    return result


def _java_hashmap_spread(hash_code: int) -> int:
    return (hash_code ^ (hash_code >> 16)) & 0xFFFFFFFF


def _java_hashmap_iteration_order(keys: list[str]) -> list[str]:
    """Return Java 8 HashMap key iteration order for insertion-only string keys."""

    capacity = 0
    threshold = 0
    size = 0
    table: list[list[str]] = []

    def resize() -> None:
        nonlocal capacity, threshold, table
        old_table = table
        if capacity == 0:
            capacity = JAVA_HASHMAP_DEFAULT_CAPACITY
        else:
            capacity *= 2
        threshold = int(capacity * JAVA_HASHMAP_LOAD_FACTOR)
        table = [[] for _ in range(capacity)]
        for bucket in old_table:
            for key in bucket:
                index = _java_hashmap_spread(_java_string_hash(key)) & (capacity - 1)
                table[index].append(key)

    for key in keys:
        if capacity == 0:
            resize()
        index = _java_hashmap_spread(_java_string_hash(key)) & (capacity - 1)
        bucket = table[index]
        if key in bucket:
            continue
        bucket.append(key)
        size += 1
        if size > threshold:
            resize()
    return [key for bucket in table for key in bucket]


@lru_cache(maxsize=1)
def relic_scope_order_ids() -> dict[str, list[str]]:
    text = RELIC_LIBRARY_PATH.read_text(encoding="utf-8")
    class_to_relic_id = {
        Path(relic.source_path).stem: relic_id
        for relic_id, relic in relic_catalog().items()
    }
    scopes: dict[str, list[str]] = {
        "SHARED": [],
        "IRONCLAD": [],
        "THE_SILENT": [],
        "DEFECT": [],
        "WATCHER": [],
    }
    for match in RELIC_LIBRARY_ADD_PATTERN.finditer(text):
        scope = PLAYER_CLASS_SCOPE[match.group(1)]
        class_name = match.group(2)
        relic_id = class_to_relic_id.get(class_name)
        if relic_id is not None:
            scopes[scope].append(relic_id)
    return {scope: _java_hashmap_iteration_order(relic_ids) for scope, relic_ids in scopes.items()}


@lru_cache(maxsize=None)
def relic_pools(character: str = "IRONCLAD") -> dict[str, list[str]]:
    selected_scope = str(character)
    scope_order = relic_scope_order_ids()
    ordered_ids = [*scope_order["SHARED"], *scope_order.get(selected_scope, [])]
    pools: dict[str, list[str]] = {}
    catalog = relic_catalog()
    for relic_id in ordered_ids:
        relic = catalog.get(relic_id)
        if relic is None:
            continue
        pools.setdefault(relic.tier, []).append(relic_id)
    return pools


def make_relic(relic_id: str, **extra: Any) -> dict[str, Any]:
    relic = relic_catalog()[relic_id]
    payload = {
        "relic_id": relic.relic_id,
        "id": relic.relic_id,
        "name": relic.name,
        "tier": relic.tier,
        "landing_sound": relic.landing_sound,
    }
    if relic.relic_id == "WingedGreaves":
        payload["counter"] = 3
    if relic.relic_id == "Matryoshka":
        payload["counter"] = 2
    if relic.relic_id == "NlothsMask":
        payload["counter"] = 1
    if relic.relic_id == "Omamori":
        payload["counter"] = 2
    if relic.relic_id == "NeowsBlessing":
        payload["counter"] = 3
    if relic.relic_id == "Circlet":
        payload["counter"] = 1
    if relic.relic_id == "Girya":
        payload["counter"] = 0
    if relic.relic_id == "Nunchaku":
        payload["counter"] = 0
    if relic.relic_id == "Sundial":
        payload["counter"] = 0
    if relic.relic_id == "Pen Nib":
        payload["counter"] = 0
    if relic.relic_id == "InkBottle":
        payload["counter"] = 0
    payload.update(extra)
    return payload


def starter_relics(character: str = "IRONCLAD") -> list[dict[str, Any]]:
    return [make_relic(relic_id) for relic_id in starting_profile(character).starter_relic_ids]


def initialize_relic_pools(
    randoms: NativeRandomSet,
    *,
    owned_relic_ids: set[str] | None = None,
    character: str = "IRONCLAD",
) -> dict[str, list[str]]:
    owned = {str(relic_id) for relic_id in set(owned_relic_ids or ())}
    pools: dict[str, list[str]] = {}
    for pool_name in RELIC_POOL_ORDER:
        pool = list(relic_pools(character).get(pool_name, []))
        java_shuffle_in_place(pool, randoms.stream("relic").random_long())
        if owned:
            pool = [relic_id for relic_id in pool if relic_id not in owned]
        pools[pool_name] = pool
    return pools


def _deck_has_card_type(
    deck: list[dict[str, Any]] | None,
    card_type: str,
    *,
    allow_basic: bool,
) -> bool:
    for card in deck or []:
        if str(card.get("type") or "") != str(card_type):
            continue
        if not allow_basic and str(card.get("rarity") or "") == "BASIC":
            continue
        return True
    return False


def can_relic_spawn(
    relic_id: str,
    *,
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> bool:
    rule = relic_spawn_rules().get(str(relic_id))
    if rule is None:
        return True
    owned = {str(value) for value in set(owned_relic_ids or ())}
    if rule.requires_relic_id is not None and rule.requires_relic_id not in owned:
        return False
    if rule.max_floor is not None and floor_num is not None and not endless and int(floor_num) > int(rule.max_floor):
        return False
    if rule.max_act is not None and act is not None and int(act) > int(rule.max_act):
        return False
    if rule.exclude_shop_room and str(current_room_type or "") == "ShopRoom":
        return False
    if rule.campfire_relic_cap is not None:
        campfire_count = sum(1 for relic_name in owned if relic_name in {"Peace Pipe", "Shovel", "Girya"})
        if campfire_count >= int(rule.campfire_relic_cap):
            return False
    if rule.requires_non_basic_card_type is not None and not _deck_has_card_type(deck, rule.requires_non_basic_card_type, allow_basic=False):
        return False
    if rule.requires_card_type is not None and not _deck_has_card_type(deck, rule.requires_card_type, allow_basic=True):
        return False
    return True


def _draw_from_pool(
    randoms: NativeRandomSet,
    pool_name: str,
    *,
    exclude: set[str] | None = None,
    character: str = "IRONCLAD",
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> str:
    pool = [
        relic_id
        for relic_id in relic_pools(character).get(pool_name, [])
        if (not exclude or relic_id not in exclude)
        and can_relic_spawn(
            relic_id,
            floor_num=floor_num,
            endless=endless,
            current_room_type=current_room_type,
            owned_relic_ids=owned_relic_ids,
            deck=deck,
            act=act,
        )
    ]
    if not pool:
        raise NotImplementedError(f"native_sim_v3 has no relics available for pool {pool_name!r}.")
    return pool[int(randoms.stream("relic").random(len(pool) - 1))]


def draw_random_relic(
    randoms: NativeRandomSet,
    tier: str,
    *,
    exclude: set[str] | None = None,
    character: str = "IRONCLAD",
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    relic_id = _draw_from_pool(
        randoms,
        tier,
        exclude=exclude,
        character=character,
        floor_num=floor_num,
        endless=endless,
        current_room_type=current_room_type,
        owned_relic_ids=owned_relic_ids,
        deck=deck,
        act=act,
    )
    return make_relic(relic_id)


def draw_random_relic_end(
    randoms: NativeRandomSet,
    tier: str,
    *,
    exclude: set[str] | None = None,
    character: str = "IRONCLAD",
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    combined_owned = {str(relic_id) for relic_id in set(owned_relic_ids or ())}
    combined_owned.update(str(relic_id) for relic_id in set(exclude or ()))
    pools = initialize_relic_pools(
        randoms,
        owned_relic_ids=combined_owned,
        character=character,
    )
    return pop_random_relic_end_from_pools(
        pools,
        tier,
        floor_num=floor_num,
        endless=endless,
        current_room_type=current_room_type,
        owned_relic_ids=combined_owned,
        deck=deck,
        act=act,
    )


def _pop_relic_from_pools_with_direction(
    relic_pools_state: dict[str, list[str]],
    tier: str,
    *,
    from_end: bool,
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    tier_name = str(tier)
    pop_index = -1 if from_end else 0
    if tier_name == "COMMON":
        if not relic_pools_state.get("COMMON"):
            return _pop_relic_from_pools_with_direction(
                relic_pools_state,
                "UNCOMMON",
                from_end=from_end,
                floor_num=floor_num,
                endless=endless,
                current_room_type=current_room_type,
                owned_relic_ids=owned_relic_ids,
                deck=deck,
                act=act,
            )
        relic_id = relic_pools_state["COMMON"].pop(pop_index)
    elif tier_name == "UNCOMMON":
        if not relic_pools_state.get("UNCOMMON"):
            return _pop_relic_from_pools_with_direction(
                relic_pools_state,
                "RARE",
                from_end=from_end,
                floor_num=floor_num,
                endless=endless,
                current_room_type=current_room_type,
                owned_relic_ids=owned_relic_ids,
                deck=deck,
                act=act,
            )
        relic_id = relic_pools_state["UNCOMMON"].pop(pop_index)
    elif tier_name == "RARE":
        if not relic_pools_state.get("RARE"):
            relic_id = "Circlet"
        else:
            relic_id = relic_pools_state["RARE"].pop(pop_index)
    elif tier_name == "SHOP":
        if not relic_pools_state.get("SHOP"):
            return _pop_relic_from_pools_with_direction(
                relic_pools_state,
                "UNCOMMON",
                from_end=from_end,
                floor_num=floor_num,
                endless=endless,
                current_room_type=current_room_type,
                owned_relic_ids=owned_relic_ids,
                deck=deck,
                act=act,
            )
        relic_id = relic_pools_state["SHOP"].pop(pop_index)
    elif tier_name == "BOSS":
        if not relic_pools_state.get("BOSS"):
            relic_id = "Red Circlet"
        else:
            relic_id = relic_pools_state["BOSS"].pop(0)
    else:
        raise NotImplementedError(f"native_sim_v3 does not support relic tier {tier_name!r}.")
    if relic_id not in {"Circlet", "Red Circlet"} and not can_relic_spawn(
        relic_id,
        floor_num=floor_num,
        endless=endless,
        current_room_type=current_room_type,
        owned_relic_ids=owned_relic_ids,
        deck=deck,
        act=act,
    ):
        return _pop_relic_from_pools_with_direction(
            relic_pools_state,
            tier_name,
            from_end=True,
            floor_num=floor_num,
            endless=endless,
            current_room_type=current_room_type,
            owned_relic_ids=owned_relic_ids,
            deck=deck,
            act=act,
        )
    return make_relic(relic_id)


def pop_random_relic_from_pools(
    relic_pools_state: dict[str, list[str]],
    tier: str,
    *,
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    return _pop_relic_from_pools_with_direction(
        relic_pools_state,
        tier,
        from_end=False,
        floor_num=floor_num,
        endless=endless,
        current_room_type=current_room_type,
        owned_relic_ids=owned_relic_ids,
        deck=deck,
        act=act,
    )


def pop_random_relic_end_from_pools(
    relic_pools_state: dict[str, list[str]],
    tier: str,
    *,
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    return _pop_relic_from_pools_with_direction(
        relic_pools_state,
        tier,
        from_end=True,
        floor_num=floor_num,
        endless=endless,
        current_room_type=current_room_type,
        owned_relic_ids=owned_relic_ids,
        deck=deck,
        act=act,
    )


def roll_random_relic_tier(randoms: NativeRandomSet, act: int | str = 1) -> str:
    roll = int(randoms.stream("relic").random(0, 99))
    chances = act_chances(act)
    if roll < chances.common_relic_chance:
        return "COMMON"
    if roll < chances.common_relic_chance + chances.uncommon_relic_chance:
        return "UNCOMMON"
    return "RARE"


def draw_random_screenless_relic(
    randoms: NativeRandomSet,
    *,
    exclude: set[str] | None = None,
    character: str = "IRONCLAD",
) -> dict[str, Any]:
    return draw_random_relic(randoms, roll_random_relic_tier(randoms), exclude=exclude, character=character)


def pop_random_screenless_relic_from_pools(
    relic_pools_state: dict[str, list[str]],
    tier: str,
    *,
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    while True:
        relic = pop_random_relic_from_pools(
            relic_pools_state,
            tier,
            floor_num=floor_num,
            endless=endless,
            current_room_type=current_room_type,
            owned_relic_ids=owned_relic_ids,
            deck=deck,
            act=act,
        )
        if str(relic.get("relic_id")) not in screenless_excluded_relic_ids():
            return relic


def pop_random_non_campfire_relic_from_pools(
    relic_pools_state: dict[str, list[str]],
    tier: str,
    *,
    floor_num: int | None = None,
    endless: bool = False,
    current_room_type: str | None = None,
    owned_relic_ids: set[str] | None = None,
    deck: list[dict[str, Any]] | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    while True:
        relic = pop_random_relic_from_pools(
            relic_pools_state,
            tier,
            floor_num=floor_num,
            endless=endless,
            current_room_type=current_room_type,
            owned_relic_ids=owned_relic_ids,
            deck=deck,
            act=act,
        )
        if str(relic.get("relic_id")) not in non_campfire_excluded_relic_ids():
            return relic


def price_for_relic_tier(tier: str) -> int:
    return relic_price_for_tier(tier)
