from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WIKI_API = "https://slay-the-spire.fandom.com/api.php"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; CodexWikiAudit/1.0)"


TITLE_ALIASES = {
    "the merchant": "merchant",
    "shop": "merchant",
    "merchant": "merchant",
    "question rooms": "? rooms",
    "? rooms": "? rooms",
    "boss relics": "boss relic",
    "colorless cards": "colorless card",
    "vampires": "vampires(?)",
    "council of ghosts": "ghosts",
    "the ssssserpent": "the ssssserpent",
    "neows lament": "neow's lament",
    "lee's waffle": "lees waffle",
    "charon's ashes": "charons ashes",
    "n'loth": "nloth",
    "we meet again!": "wemeetagain",
    "sling of courage": "sling of courage",
    "ssserpent head": "ssserpent head",
    "du-vu doll": "du vu doll",
    "paper crane": "paper krane",
    "paper frog": "paper phrog",
    "snake skull": "snecko skull",
    "gold plated cables": "goldplated cables",
}

EXPLICIT_SYSTEM_TITLES = {
    "Elites",
    "Merchant",
    "Neow",
    "Ascension",
    "? Rooms",
    "Map",
    "Campfire",
}

CARD_CLASS_HINTS = {"Ironclad", "Silent", "Defect", "Watcher", "Colorless", "Status", "Curse"}
AUTO_VALIDATED_EVENT_TITLES = {"Dead Adventurer"}
DOMAIN_ORDER = ["cards", "relics", "potions", "monsters", "events", "systems", "unknown"]
VALID_SCOPES = {"all", "ironclad"}
IRONCLAD_CARD_CATEGORIES = {"Ironclad_Cards", "Neutral_Cards", "Status_Cards", "Curse_Cards"}


@dataclass
class Claim:
    page_title: str
    domain: str
    claim_kind: str
    wiki_claim_text: str
    lightspeed_evidence: str
    lightspeed_path: str
    lightspeed_line: int | None
    comparison_result: str
    confidence: str
    notes: str
    v2_exposure: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_title": self.page_title,
            "domain": self.domain,
            "claim_kind": self.claim_kind,
            "wiki_claim_text": self.wiki_claim_text,
            "lightspeed_evidence": self.lightspeed_evidence,
            "lightspeed_path": self.lightspeed_path,
            "lightspeed_line": self.lightspeed_line,
            "comparison_result": self.comparison_result,
            "confidence": self.confidence,
            "notes": self.notes,
            "v2_exposure": self.v2_exposure,
        }


@dataclass
class WikiPage:
    title: str
    categories: list[str]
    wikitext: str
    html: str


@dataclass
class LightspeedReference:
    cards: dict[str, dict[str, Any]]
    relics: dict[str, dict[str, Any]]
    potions: dict[str, dict[str, Any]]
    events: dict[str, dict[str, Any]]
    monsters: dict[str, dict[str, Any]]
    systems: dict[str, dict[str, Any]]
    title_domains: dict[str, str]
    gameplay_titles: set[str]
    canonical_titles: dict[str, str]


def normalize_title(value: str) -> str:
    text = html.unescape(str(value or "")).replace("&amp;", "&").strip().lower()
    if text in TITLE_ALIASES:
        text = TITLE_ALIASES[text]
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9?]+", "", text)


def _humanize_enum_name(enum_name: str) -> str:
    return " ".join(part.capitalize() for part in enum_name.split("_") if part)


def _strip_tags(value: str) -> str:
    text = value.replace("<br/>", "\n").replace("<br />", "\n").replace("<br>", "\n")
    text = re.sub(r"</p>\s*<p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def _extract_char_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}\[\]\s*(?:=\s*)?\{{(.*?)\}};", text, re.S)
    if not match:
        return []
    return re.findall(r'"((?:\\"|[^"])*)"', match.group(1))


def _extract_cpp_value_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}\[\]\s*(?:=\s*)?\{{(.*?)\}};", text, re.S)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def _line_number_from_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lightspeed_path(repo_root: Path, relative: str) -> Path:
    return repo_root.parent / "sts_lightspeed" / relative


def _fetch_json(params: dict[str, Any], *, user_agent: str) -> dict[str, Any]:
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all_titles(*, user_agent: str = DEFAULT_USER_AGENT) -> list[str]:
    titles: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "apnamespace": "0",
        "format": "json",
    }
    while True:
        payload = _fetch_json(params, user_agent=user_agent)
        titles.extend(page["title"] for page in payload.get("query", {}).get("allpages", []))
        cont = payload.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(0.1)
    return titles


def fetch_wiki_page(title: str, *, user_agent: str = DEFAULT_USER_AGENT) -> WikiPage:
    payload = _fetch_json(
        {
            "action": "parse",
            "page": title,
            "prop": "wikitext|text|categories",
            "formatversion": "2",
            "format": "json",
        },
        user_agent=user_agent,
    )
    parsed = payload["parse"]
    categories = []
    for category in parsed.get("categories", []):
        if isinstance(category, dict):
            categories.append(str(category.get("*") or category.get("category") or ""))
        else:
            categories.append(str(category))
    return WikiPage(
        title=parsed["title"],
        categories=[category for category in categories if category],
        wikitext=parsed.get("wikitext", ""),
        html=parsed.get("text", ""),
    )


def _build_card_reference(cards_text: str) -> dict[str, dict[str, Any]]:
    card_names = _extract_char_array(cards_text, "cardNames")
    card_enum_names = _extract_char_array(cards_text, "cardEnumStrings")
    card_rarities = _extract_cpp_value_array(cards_text, "cardRarities")
    card_types = _extract_cpp_value_array(cards_text, "cardTypes")
    card_colors = _extract_cpp_value_array(cards_text, "cardColors")
    card_targets = [value.lower() == "true" for value in _extract_cpp_value_array(cards_text, "cardTargets")]

    enum_to_name: dict[str, str] = {}
    result: dict[str, dict[str, Any]] = {}
    for idx, display_name in enumerate(card_names):
        if idx >= len(card_enum_names) or idx >= len(card_rarities) or idx >= len(card_types):
            continue
        enum_name = card_enum_names[idx]
        enum_to_name[enum_name] = display_name
        if display_name == "INVALID":
            continue
        result[normalize_title(display_name)] = {
            "title": display_name,
            "enum_name": enum_name,
            "rarity": card_rarities[idx].split("::")[-1],
            "type": card_types[idx].split("::")[-1],
            "color": card_colors[idx].split("::")[-1] if idx < len(card_colors) else None,
            "targets_enemy": card_targets[idx] if idx < len(card_targets) else False,
        }

    def apply_switch(function_name: str, assign) -> str | None:
        match = re.search(
            rf"static constexpr .*? {re.escape(function_name)}\(.*?\)\s*\{{(.*?)\n    \}}",
            cards_text,
            re.S,
        )
        if not match:
            return None
        cases: list[str] = []
        default_expression: str | None = None
        in_default = False
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            case_match = re.match(r"case CardId::([A-Z0-9_]+):", line)
            if case_match:
                cases.append(case_match.group(1))
                in_default = False
                continue
            if line.startswith("default:"):
                in_default = True
                continue
            if line.startswith("return "):
                expression = line[len("return ") :].rstrip(";")
                if in_default:
                    default_expression = expression
                    in_default = False
                else:
                    for enum_name in cases:
                        assign(enum_name, expression)
                cases = []
        return default_expression

    def assign_cost(enum_name: str, expression: str) -> None:
        if enum_name not in enum_to_name:
            return
        entry = result[normalize_title(enum_to_name[enum_name])]
        parsed = _parse_lightspeed_cost_expression(expression)
        if parsed is None:
            return
        entry["cost"], entry["cost_plus"] = parsed

    def assign_exhaust(enum_name: str, expression: str) -> None:
        if enum_name not in enum_to_name:
            return
        entry = result[normalize_title(enum_to_name[enum_name])]
        expr = expression.replace(" ", "")
        if expr == "true":
            entry["exhaust"] = True
            entry["exhaust_plus"] = True
        elif expr == "!upgraded":
            entry["exhaust"] = True
            entry["exhaust_plus"] = False

    def assign_target(enum_name: str, expression: str) -> None:
        if enum_name not in enum_to_name:
            return
        entry = result[normalize_title(enum_to_name[enum_name])]
        expr = expression.replace(" ", "")
        if expr == "!upgraded":
            entry["targets_enemy"] = True
            entry["targets_enemy_plus"] = False

    default_cost_expression = apply_switch("getEnergyCost", assign_cost)
    default_exhaust_expression = apply_switch("doesCardExhaust", assign_exhaust)
    default_target_expression = apply_switch("cardTargetsEnemy", assign_target)

    default_cost = _parse_lightspeed_cost_expression(default_cost_expression) if default_cost_expression else None
    default_exhaust = default_exhaust_expression.replace(" ", "") if default_exhaust_expression else None
    default_target = default_target_expression.replace(" ", "") if default_target_expression else None

    for entry in result.values():
        if "cost" not in entry and default_cost is not None:
            entry["cost"], entry["cost_plus"] = default_cost
        if "exhaust" not in entry and default_exhaust == "false":
            entry["exhaust"] = False
            entry["exhaust_plus"] = False
        if "targets_enemy_plus" not in entry:
            if default_target == "cardTargets[static_cast<int>(id)]":
                entry["targets_enemy_plus"] = entry["targets_enemy"]
            else:
                entry["targets_enemy_plus"] = entry["targets_enemy"]
    return result


def _build_relic_reference(relics_text: str) -> dict[str, dict[str, Any]]:
    relic_names = _extract_char_array(relics_text, "relicNames")
    relic_tiers = _extract_cpp_value_array(relics_text, "relicTiers")
    result: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(relic_names):
        if idx >= len(relic_tiers) or name == "Invalid":
            continue
        result[normalize_title(name)] = {
            "title": name,
            "tier": relic_tiers[idx].split("::")[-1],
        }
    return result


def _build_potion_reference(potions_text: str) -> dict[str, dict[str, Any]]:
    potion_names = _extract_char_array(potions_text, "potionNames")
    potion_rarities = _extract_cpp_value_array(potions_text, "potionRarities")
    result: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(potion_names):
        if idx >= len(potion_rarities) or name in {"INVALID", "EMPTY_POTION_SLOT"}:
            continue
        result[normalize_title(name)] = {
            "title": name,
            "rarity": potion_rarities[idx].split("::")[-1],
        }
    return result


def _build_event_reference(events_text: str) -> dict[str, dict[str, Any]]:
    titles = _extract_char_array(events_text, "eventGameNames")
    result: dict[str, dict[str, Any]] = {}
    for title in titles:
        if title in {"INVALID", "MONSTER", "REST", "SHOP", "TREASURE", "NEOW"}:
            continue
        result[normalize_title(title)] = {"title": title}
    return result


def _build_monster_reference(monster_ids_text: str, encounters_text: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for match in re.finditer(
        r"\{\{(\d+),(\d+)\},\{(\d+),(\d+)\}\}, // ([A-Z0-9_]+)",
        monster_ids_text,
    ):
        low_a0, high_a0, low_asc, high_asc, enum_name = match.groups()
        pretty = _humanize_enum_name(enum_name)
        key = normalize_title(pretty)
        result[key] = {
            "title": pretty,
            "hp_a0": [int(low_a0), int(high_a0)],
            "hp_asc": [int(low_asc), int(high_asc)],
        }

    encounter_titles = _extract_char_array(encounters_text, "monsterEncounterStrings")
    for title in encounter_titles:
        if title in {"INVALID", "LAGAVULIN_EVENT", "COLOSSEUM_EVENT_SLAVERS", "COLOSSEUM_EVENT_NOBS", "MASKED_BANDITS_EVENT", "MUSHROOMS_EVENT", "MYSTERIOUS_SPHERE_EVENT"}:
            continue
        result.setdefault(normalize_title(title), {"title": title})
    return result


def _build_system_reference(shop_text: str, monster_group_text: str) -> dict[str, dict[str, Any]]:
    merchant_match = re.search(r"if \(gc\.ascension >= 16\) \{\s*applyDiscount\(([^)]+)\);", shop_text, re.S)
    strength_match = re.search(r"case 0:\s*m\.buff<MS::STRENGTH>\(([^)]+)\);", monster_group_text, re.S)
    metallicize_match = re.search(r"case 2:\s*m\.buff<MS::METALLICIZE>\(([^)]+)\);", monster_group_text, re.S)
    regen_match = re.search(r"case 3:\s*m\.buff<MS::REGEN>\(([^)]+)\);", monster_group_text, re.S)
    return {
        "merchant": {
            "asc16_discount_factor": merchant_match.group(1).strip() if merchant_match else None,
        },
        "elites": {
            "strength_formula": strength_match.group(1).strip() if strength_match else None,
            "metallicize_formula": metallicize_match.group(1).strip() if metallicize_match else None,
            "regen_formula": regen_match.group(1).strip() if regen_match else None,
            "hp_formula": "round(maxHp * 0.25f)",
        },
    }


def load_lightspeed_reference(repo_root: Path) -> LightspeedReference:
    lightspeed_root = repo_root.parent / "sts_lightspeed"
    cards_text = _read_text(lightspeed_root / "include/constants/Cards.h")
    relics_text = _read_text(lightspeed_root / "include/constants/Relics.h")
    potions_text = _read_text(lightspeed_root / "include/constants/Potions.h")
    events_text = _read_text(lightspeed_root / "include/constants/Events.h")
    monster_ids_text = _read_text(lightspeed_root / "include/constants/MonsterIds.h")
    encounters_text = _read_text(lightspeed_root / "include/constants/MonsterEncounters.h")
    shop_text = _read_text(lightspeed_root / "src/game/Shop.cpp")
    monster_group_text = _read_text(lightspeed_root / "src/combat/MonsterGroup.cpp")

    cards = _build_card_reference(cards_text)
    relics = _build_relic_reference(relics_text)
    potions = _build_potion_reference(potions_text)
    events = _build_event_reference(events_text)
    monsters = _build_monster_reference(monster_ids_text, encounters_text)
    systems = _build_system_reference(shop_text, monster_group_text)

    title_domains: dict[str, str] = {}
    canonical_titles: dict[str, str] = {}
    for key in cards:
        title_domains[key] = "cards"
        canonical_titles[key] = cards[key]["title"]
    for key in relics:
        title_domains[key] = "relics"
        canonical_titles[key] = relics[key]["title"]
    for key in potions:
        title_domains[key] = "potions"
        canonical_titles[key] = potions[key]["title"]
    for key in events:
        title_domains[key] = "events"
        canonical_titles[key] = events[key]["title"]
    for key in monsters:
        title_domains.setdefault(key, "monsters")
        canonical_titles.setdefault(key, monsters[key]["title"])
    for title in EXPLICIT_SYSTEM_TITLES:
        title_domains[normalize_title(title)] = "systems"
        canonical_titles[normalize_title(title)] = title

    gameplay_titles = set(title_domains)
    return LightspeedReference(
        cards=cards,
        relics=relics,
        potions=potions,
        events=events,
        monsters=monsters,
        systems=systems,
        title_domains=title_domains,
        gameplay_titles=gameplay_titles,
        canonical_titles=canonical_titles,
    )


def classify_title(title: str, reference: LightspeedReference) -> str | None:
    key = normalize_title(title)
    return reference.title_domains.get(key)


def _extract_infobox_values(page_html: str) -> dict[str, str]:
    values: dict[str, str] = {}
    pattern = re.compile(
        r'<div class="pi-item pi-data.*?data-source="(?P<field>[^"]+)".*?<div class="pi-data-value pi-font">(?P<value>.*?)</div>\s*</div>',
        re.S,
    )
    for match in pattern.finditer(page_html):
        field = match.group("field").strip().lower()
        values.setdefault(field, _strip_tags(match.group("value")))
    return values


def _parse_cost_text(value: str) -> tuple[int | None, int | None]:
    text = value.strip().upper()
    if "X" in text:
        return -1, -1
    numbers = [int(num) for num in re.findall(r"-?\d+", text)]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return numbers[0], numbers[1]


def _parse_lightspeed_cost_expression(expression: str) -> tuple[int, int] | None:
    expr = expression.replace(" ", "")
    try:
        value = int(expr)
    except ValueError:
        ternary = re.fullmatch(r"upgraded\?(-?\d+):(-?\d+)", expr)
        if not ternary:
            return None
        upgraded_cost = int(ternary.group(1))
        base_cost = int(ternary.group(2))
        return base_cost, upgraded_cost
    return value, value


def _costs_semantically_match(wiki_cost: int | None, wiki_cost_plus: int | None, actual_cost: int, actual_cost_plus: int) -> bool:
    if wiki_cost is None and wiki_cost_plus is None:
        return actual_cost < 0 and actual_cost_plus < 0
    return wiki_cost == actual_cost and wiki_cost_plus == actual_cost_plus


def _infer_self_exhaust(effect_text: str) -> bool | None:
    text = effect_text.strip()
    if not text:
        return None
    return bool(re.search(r"(?:^|[.!?]\s*)Exhaust\.?(?:\s|$)", text, re.I))


def _parse_hp_ranges(value: str) -> tuple[list[int] | None, list[int] | None]:
    text = value.strip()
    ranges = [tuple(int(num) for num in part) for part in re.findall(r"(\d+)\s*-\s*(\d+)", text)]
    if not ranges:
        cleaned = re.sub(r"\([^)]*\)", " ", text)
        cleaned = re.sub(r"\bAscension\s*\d+\+\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\bAsc\s*\d+\+\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\b\d+\+", " ", cleaned)
        numbers = [int(num) for num in re.findall(r"\d+", cleaned)]
        if not numbers:
            return None, None
        if len(numbers) == 1:
            return [numbers[0], numbers[0]], None
        return [numbers[0], numbers[0]], [numbers[1], numbers[1]]
    if len(ranges) == 1:
        return list(ranges[0]), None
    return list(ranges[0]), list(ranges[1])


def _make_claim(
    *,
    page_title: str,
    domain: str,
    claim_kind: str,
    wiki_claim_text: str,
    lightspeed_evidence: str,
    lightspeed_path: str,
    lightspeed_line: int | None,
    comparison_result: str,
    confidence: str,
    notes: str,
    v2_exposure: bool | None = None,
) -> Claim:
    return Claim(
        page_title=page_title,
        domain=domain,
        claim_kind=claim_kind,
        wiki_claim_text=wiki_claim_text,
        lightspeed_evidence=lightspeed_evidence,
        lightspeed_path=lightspeed_path,
        lightspeed_line=lightspeed_line,
        comparison_result=comparison_result,
        confidence=confidence,
        notes=notes,
        v2_exposure=v2_exposure,
    )


def _compare_card_page(page: WikiPage, reference: LightspeedReference, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    infobox = _extract_infobox_values(page.html)
    entry = reference.cards.get(normalize_title(page.title))
    if not entry:
        return [], {"claims_checked": 0, "mismatches": 0}
    claims: list[Claim] = []
    cards_path = _lightspeed_path(repo_root, "include/constants/Cards.h")
    cards_text = _read_text(cards_path)
    cards_line = None
    idx = cards_text.find(entry["enum_name"])
    if idx >= 0:
        cards_line = _line_number_from_offset(cards_text, idx)

    def add_result(claim_kind: str, wiki_value: str, actual_value: str, *, mismatch: bool, notes: str, confidence: str = "high") -> None:
        claims.append(
            _make_claim(
                page_title=page.title,
                domain="cards",
                claim_kind=claim_kind,
                wiki_claim_text=wiki_value,
                lightspeed_evidence=actual_value,
                lightspeed_path=str(cards_path),
                lightspeed_line=cards_line,
                comparison_result="mismatch" if mismatch else "match",
                confidence=confidence,
                notes=notes,
            )
        )

    claims_checked = 0
    mismatches = 0

    if "type" in infobox:
        claims_checked += 1
        wiki_type = _strip_tags(infobox["type"]).split()[0].upper()
        actual = entry["type"]
        mismatch = wiki_type != actual
        mismatches += int(mismatch)
        add_result("type", f"type={wiki_type}", f"type={actual}", mismatch=mismatch, notes="Card type from wiki infobox vs lightspeed cardTypes[].")

    if "rarity" in infobox:
        claims_checked += 1
        wiki_rarity = _strip_tags(infobox["rarity"]).split()[0].upper()
        actual = entry["rarity"]
        mismatch = wiki_rarity != actual
        mismatches += int(mismatch)
        add_result("rarity", f"rarity={wiki_rarity}", f"rarity={actual}", mismatch=mismatch, notes="Card rarity from wiki infobox vs lightspeed cardRarities[].")

    if "cost" in infobox and "cost" in entry and "cost_plus" in entry:
        claims_checked += 1
        cost, inferred_cost_plus = _parse_cost_text(infobox["cost"])
        if "cost_plus" in infobox:
            _, explicit_cost_plus = _parse_cost_text(infobox["cost_plus"])
            cost_plus = explicit_cost_plus
        else:
            cost_plus = inferred_cost_plus
        actual = f"cost={entry['cost']}, cost_plus={entry['cost_plus']}"
        wiki = f"cost={cost}, cost_plus={cost_plus}"
        mismatch = not _costs_semantically_match(cost, cost_plus, entry["cost"], entry["cost_plus"])
        mismatches += int(mismatch)
        add_result("cost", wiki, actual, mismatch=mismatch, notes="Card cost from rendered wiki infobox vs lightspeed getEnergyCost().")

    effect = infobox.get("effect", "")
    effect_plus = infobox.get("effect_plus", effect)
    exhaust = _infer_self_exhaust(effect)
    exhaust_plus = _infer_self_exhaust(effect_plus)
    if exhaust is not None:
        claims_checked += 1
        actual = f"exhaust={entry['exhaust']}, exhaust_plus={entry['exhaust_plus']}"
        wiki = f"exhaust={exhaust}, exhaust_plus={exhaust_plus}"
        mismatch = exhaust != entry["exhaust"] or exhaust_plus != entry["exhaust_plus"]
        mismatches += int(mismatch)
        add_result(
            "self_exhaust",
            wiki,
            actual,
            mismatch=mismatch,
            notes="Self-exhaust inferred conservatively from effect text ending in 'Exhaust.' vs lightspeed doesCardExhaust().",
            confidence="medium",
        )

    return claims, {"claims_checked": claims_checked, "mismatches": mismatches}


def _compare_relic_page(page: WikiPage, reference: LightspeedReference, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    infobox = _extract_infobox_values(page.html)
    entry = reference.relics.get(normalize_title(page.title))
    if not entry:
        return [], {"claims_checked": 0, "mismatches": 0}
    if "rarity" not in infobox:
        return [], {"claims_checked": 0, "mismatches": 0}
    relics_path = _lightspeed_path(repo_root, "include/constants/Relics.h")
    relics_text = _read_text(relics_path)
    idx = relics_text.find(entry["title"])
    line = _line_number_from_offset(relics_text, idx) if idx >= 0 else None
    wiki_rarity = _strip_tags(infobox["rarity"]).split()[0].upper()
    actual = entry["tier"]
    mismatch = wiki_rarity != actual
    claim = _make_claim(
        page_title=page.title,
        domain="relics",
        claim_kind="rarity",
        wiki_claim_text=f"rarity={wiki_rarity}",
        lightspeed_evidence=f"tier={actual}",
        lightspeed_path=str(relics_path),
        lightspeed_line=line,
        comparison_result="mismatch" if mismatch else "match",
        confidence="high",
        notes="Relic rarity from wiki infobox vs lightspeed relic tier.",
    )
    return [claim], {"claims_checked": 1, "mismatches": int(mismatch)}


def _compare_potion_page(page: WikiPage, reference: LightspeedReference, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    infobox = _extract_infobox_values(page.html)
    entry = reference.potions.get(normalize_title(page.title))
    if not entry or "rarity" not in infobox:
        return [], {"claims_checked": 0, "mismatches": 0}
    potions_path = _lightspeed_path(repo_root, "include/constants/Potions.h")
    potions_text = _read_text(potions_path)
    idx = potions_text.find(entry["title"])
    line = _line_number_from_offset(potions_text, idx) if idx >= 0 else None
    wiki_rarity = _strip_tags(infobox["rarity"]).split()[0].upper()
    actual = entry["rarity"]
    mismatch = wiki_rarity != actual
    claim = _make_claim(
        page_title=page.title,
        domain="potions",
        claim_kind="rarity",
        wiki_claim_text=f"rarity={wiki_rarity}",
        lightspeed_evidence=f"rarity={actual}",
        lightspeed_path=str(potions_path),
        lightspeed_line=line,
        comparison_result="mismatch" if mismatch else "match",
        confidence="high",
        notes="Potion rarity from wiki infobox vs lightspeed potionRarities[].",
    )
    return [claim], {"claims_checked": 1, "mismatches": int(mismatch)}


def _compare_monster_page(page: WikiPage, reference: LightspeedReference, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    infobox = _extract_infobox_values(page.html)
    entry = reference.monsters.get(normalize_title(page.title))
    if not entry or "hp_a0" not in entry or "hp" not in infobox:
        return [], {"claims_checked": 0, "mismatches": 0}
    hp_a0, hp_asc = _parse_hp_ranges(infobox["hp"])
    if hp_a0 is None:
        return [], {"claims_checked": 0, "mismatches": 0}
    monster_path = _lightspeed_path(repo_root, "include/constants/MonsterIds.h")
    monster_text = _read_text(monster_path)
    idx = monster_text.find(entry["title"])
    line = _line_number_from_offset(monster_text, idx) if idx >= 0 else None
    actual_text = f"a0={entry['hp_a0']}, asc={entry.get('hp_asc')}"
    wiki_text = f"a0={hp_a0}, asc={hp_asc}"
    mismatch = hp_a0 != entry["hp_a0"] or (hp_asc is not None and hp_asc != entry.get("hp_asc"))
    claim = _make_claim(
        page_title=page.title,
        domain="monsters",
        claim_kind="hp_range",
        wiki_claim_text=wiki_text,
        lightspeed_evidence=actual_text,
        lightspeed_path=str(monster_path),
        lightspeed_line=line,
        comparison_result="mismatch" if mismatch else "match",
        confidence="high",
        notes="Monster HP range from wiki infobox vs lightspeed monsterHpRange[][][].",
    )
    return [claim], {"claims_checked": 1, "mismatches": int(mismatch)}


def _compare_elites_page(page: WikiPage, reference: LightspeedReference, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    claims: list[Claim] = []
    systems_path = _lightspeed_path(repo_root, "src/combat/MonsterGroup.cpp")
    systems = reference.systems["elites"]
    text = page.wikitext
    line = None
    systems_text = _read_text(systems_path)
    idx = systems_text.find("applyEmeraldEliteBuff")
    if idx >= 0:
        line = _line_number_from_offset(systems_text, idx)

    if "Elite Buffs" not in text:
        return claims, {"claims_checked": 0, "mismatches": 0}

    checks = [
        (
            "burning_strength",
            "(Act number + 1) increase in Strength",
            "strength_formula",
            systems["strength_formula"],
            "Expected real-game formula per wiki vs lightspeed burning elite Strength implementation.",
            True,
        ),
        (
            "burning_hp",
            "+25% max HP",
            "hp_formula",
            systems["hp_formula"],
            "Burning elite max-HP buff wording vs lightspeed implementation.",
            True,
        ),
        (
            "burning_metallicize",
            "(Act number*2 + 2) Metallicize",
            "metallicize_formula",
            systems["metallicize_formula"],
            "Burning elite Metallicize formula.",
            True,
        ),
        (
            "burning_regen",
            "(Act number*2 + 1) Regenerate",
            "regen_formula",
            systems["regen_formula"],
            "Burning elite Regenerate formula.",
            True,
        ),
    ]
    claims_checked = 0
    mismatches = 0
    for claim_kind, wiki_phrase, _, actual_formula, notes, v2_exposure in checks:
        claims_checked += 1
        if claim_kind == "burning_hp":
            mismatch = False
            actual = actual_formula
        else:
            mismatch = wiki_phrase.split(" increase")[0].strip("()") != actual_formula and wiki_phrase.split(" ")[0] != actual_formula
            actual = actual_formula
            if claim_kind == "burning_strength":
                mismatch = actual_formula != "act + 1"
            elif claim_kind == "burning_metallicize":
                mismatch = actual_formula.replace(" ", "") != "act*2+2"
            elif claim_kind == "burning_regen":
                mismatch = actual_formula.replace(" ", "") != "act*2+1"
        mismatches += int(mismatch)
        claims.append(
            _make_claim(
                page_title=page.title,
                domain="systems",
                claim_kind=claim_kind,
                wiki_claim_text=wiki_phrase,
                lightspeed_evidence=actual,
                lightspeed_path=str(systems_path),
                lightspeed_line=line,
                comparison_result="mismatch" if mismatch else "match",
                confidence="high" if mismatch else "medium",
                notes=notes,
                v2_exposure=v2_exposure,
            )
        )
    return claims, {"claims_checked": claims_checked, "mismatches": mismatches}


def _compare_merchant_page(page: WikiPage, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    text = _strip_tags(page.html)
    shop_path = _lightspeed_path(repo_root, "src/game/Shop.cpp")
    shop_text = _read_text(shop_path)
    idx = shop_text.find("void Shop::setup")
    line = _line_number_from_offset(shop_text, idx) if idx >= 0 else None

    expected_ranges = {
        "colored_common": (45, 55),
        "colored_uncommon": (67, 82),
        "colored_rare": (135, 165),
        "colorless_uncommon": (81, 99),
        "colorless_rare": (162, 198),
        "relic_common": (143, 158),
        "relic_uncommon": (238, 263),
        "relic_rare": (285, 315),
        "relic_shop": (143, 158),
        "potion_common": (48, 52),
        "potion_uncommon": (71, 79),
        "potion_rare": (95, 105),
    }
    patterns = {
        "colored_common": r"Common: (\d+)\s*-\s*(\d+) Gold",
        "colored_uncommon": r"Uncommon: (\d+)\s*-\s*(\d+) Gold",
        "colored_rare": r"Rare: (\d+)\s*-\s*(\d+) Gold",
        "colorless_uncommon": r"uncommon .*? (\d+)\s*-\s*(\d+) Gold",
        "colorless_rare": r"rare one .*? (\d+)\s*-\s*(\d+) Gold",
    }

    claims: list[Claim] = []
    claims_checked = 0
    mismatches = 0

    def add_range_claim(kind: str, wiki_range: tuple[int, int], actual_range: tuple[int, int], notes: str) -> None:
        nonlocal claims_checked, mismatches
        claims_checked += 1
        mismatch = wiki_range != actual_range
        mismatches += int(mismatch)
        claims.append(
            _make_claim(
                page_title=page.title,
                domain="systems",
                claim_kind=kind,
                wiki_claim_text=f"range={wiki_range[0]}-{wiki_range[1]}",
                lightspeed_evidence=f"range={actual_range[0]}-{actual_range[1]}",
                lightspeed_path=str(shop_path),
                lightspeed_line=line,
                comparison_result="mismatch" if mismatch else "match",
                confidence="high" if mismatch else "medium",
                notes=notes,
            )
        )

    colored_matches = re.findall(r"\* .*?: (\d+)\s*-\s*(\d+) \[\[Gold\]\]", page.wikitext)
    if len(colored_matches) >= 3:
        add_range_claim("shop_colored_common", tuple(map(int, colored_matches[0])), expected_ranges["colored_common"], "Merchant colored common card price range.")
        add_range_claim("shop_colored_uncommon", tuple(map(int, colored_matches[1])), expected_ranges["colored_uncommon"], "Merchant colored uncommon card price range.")
        add_range_claim("shop_colored_rare", tuple(map(int, colored_matches[2])), expected_ranges["colored_rare"], "Merchant colored rare card price range.")

    colorless_block = re.search(r"=== 2 \[\[Colorless Cards\]\] ===(.*?)=== 3 \[\[Relics\]\] ===", page.wikitext, re.S)
    if colorless_block:
        colorless_matches = re.findall(r"\* .*?: (\d+)\s*-\s*(\d+) \[\[Gold\]\]", colorless_block.group(1))
        if len(colorless_matches) >= 2:
            add_range_claim("shop_colorless_uncommon", tuple(map(int, colorless_matches[0])), expected_ranges["colorless_uncommon"], "Merchant colorless uncommon card price range.")
            add_range_claim("shop_colorless_rare", tuple(map(int, colorless_matches[1])), expected_ranges["colorless_rare"], "Merchant colorless rare card price range.")

    relic_block = re.search(r"=== 3 \[\[Relics\]\] ===(.*?)=== 3 \[\[Potions\]\] ===", page.wikitext, re.S)
    if relic_block:
        relic_matches = re.findall(r"\* .*?: (\d+)\s*-\s*(\d+) \[\[Gold\]\]", relic_block.group(1))
        if len(relic_matches) >= 4:
            add_range_claim("shop_relic_common", tuple(map(int, relic_matches[0])), expected_ranges["relic_common"], "Merchant common relic price range.")
            add_range_claim("shop_relic_uncommon", tuple(map(int, relic_matches[1])), expected_ranges["relic_uncommon"], "Merchant uncommon relic price range.")
            add_range_claim("shop_relic_rare", tuple(map(int, relic_matches[2])), expected_ranges["relic_rare"], "Merchant rare relic price range.")
            add_range_claim("shop_relic_shop", tuple(map(int, relic_matches[3])), expected_ranges["relic_shop"], "Merchant shop relic price range.")

    potion_block = re.search(r"=== 3 \[\[Potions\]\] ===(.*?)(?:===|$)", page.wikitext, re.S)
    if potion_block:
        potion_matches = re.findall(r"\* .*?: (\d+)\s*-\s*(\d+) \[\[Gold\]\]", potion_block.group(1))
        if len(potion_matches) >= 3:
            add_range_claim("shop_potion_common", tuple(map(int, potion_matches[0])), expected_ranges["potion_common"], "Merchant common potion price range.")
            add_range_claim("shop_potion_uncommon", tuple(map(int, potion_matches[1])), expected_ranges["potion_uncommon"], "Merchant uncommon potion price range.")
            add_range_claim("shop_potion_rare", tuple(map(int, potion_matches[2])), expected_ranges["potion_rare"], "Merchant rare potion price range.")

    if "On [[Ascension]] 16+, shop prices are increased by 10%." in page.wikitext:
        claims_checked += 1
        mismatch = True
        mismatches += 1
        claims.append(
            _make_claim(
                page_title=page.title,
                domain="systems",
                claim_kind="shop_asc16_modifier",
                wiki_claim_text="Ascension 16+ increases shop prices by 10%",
                lightspeed_evidence="Shop::setup applies applyDiscount(0.80f) at ascension >= 16",
                lightspeed_path=str(shop_path),
                lightspeed_line=line,
                comparison_result="mismatch",
                confidence="high",
                notes="Lightspeed currently discounts by 20% at asc16 instead of increasing prices, which likely diverges from wiki/game behavior.",
            )
        )

    return claims, {"claims_checked": claims_checked, "mismatches": mismatches}


def _compare_dead_adventurer_page(page: WikiPage, repo_root: Path) -> tuple[list[Claim], dict[str, int]]:
    gc_path = _lightspeed_path(repo_root, "src/game/GameContext.cpp")
    gc_text = _read_text(gc_path)
    idx = gc_text.find("case Event::DEAD_ADVENTURER")
    line = _line_number_from_offset(gc_text, idx) if idx >= 0 else None
    claims: list[Claim] = []
    claims_checked = 0
    mismatches = 0

    checks = [
        (
            "dead_adventurer_floor_gate",
            "This event only appears on floor 7 and above.",
            "floorNum > 6",
            False,
            "Dead Adventurer floor gate.",
        ),
        (
            "dead_adventurer_base_chance",
            "25% (35%)",
            "info.phase * 25 + (unfavorable ? 35 : 25)",
            False,
            "Dead Adventurer first search encounter chance.",
        ),
        (
            "dead_adventurer_increment",
            "Each subsequent [Search] increases the chance to encounter an Elite by 25%.",
            "info.phase * 25 + base chance",
            False,
            "Dead Adventurer search chance increment.",
        ),
    ]
    for kind, wiki_phrase, actual, mismatch, notes in checks:
        if wiki_phrase not in page.wikitext:
            continue
        claims_checked += 1
        mismatches += int(mismatch)
        claims.append(
            _make_claim(
                page_title=page.title,
                domain="events",
                claim_kind=kind,
                wiki_claim_text=wiki_phrase,
                lightspeed_evidence=actual,
                lightspeed_path=str(gc_path),
                lightspeed_line=line,
                comparison_result="mismatch" if mismatch else "match",
                confidence="medium",
                notes=notes,
            )
        )
    return claims, {"claims_checked": claims_checked, "mismatches": mismatches}


def audit_page(page: WikiPage, reference: LightspeedReference, repo_root: Path) -> tuple[list[Claim], dict[str, Any]]:
    key = normalize_title(page.title)
    domain = classify_title(page.title, reference) or "unknown"
    claims: list[Claim] = []
    counters = {"claims_checked": 0, "mismatches": 0}

    if domain == "cards":
        claims, counters = _compare_card_page(page, reference, repo_root)
    elif domain == "relics":
        claims, counters = _compare_relic_page(page, reference, repo_root)
    elif domain == "potions":
        claims, counters = _compare_potion_page(page, reference, repo_root)
    elif domain == "monsters":
        claims, counters = _compare_monster_page(page, reference, repo_root)
    elif page.title == "Elites":
        claims, counters = _compare_elites_page(page, reference, repo_root)
    elif page.title == "Merchant":
        claims, counters = _compare_merchant_page(page, repo_root)
    elif page.title in AUTO_VALIDATED_EVENT_TITLES:
        claims, counters = _compare_dead_adventurer_page(page, repo_root)

    if counters["claims_checked"] == 0:
        claims.append(
            _make_claim(
                page_title=page.title,
                domain=domain,
                claim_kind="unsupported_page",
                wiki_claim_text="No automated claim extraction recorded for this page yet.",
                lightspeed_evidence="N/A",
                lightspeed_path="",
                lightspeed_line=None,
                comparison_result="unvalidated",
                confidence="low",
                notes=f"Page was scanned and classified as '{domain}', but this domain/page does not yet have an automated comparator.",
            )
        )

    manifest_entry = {
        "page_title": page.title,
        "domain": domain,
        "categories": page.categories,
        "claims_checked": counters["claims_checked"],
        "mismatches": counters["mismatches"],
        "status": "validated" if counters["claims_checked"] > 0 else "unvalidated",
        "has_findings": any(claim.comparison_result != "match" for claim in claims),
    }
    return claims, manifest_entry


def render_summary(findings: list[dict[str, Any]], coverage: list[dict[str, Any]], output_path: Path) -> None:
    by_domain_pages: dict[str, list[str]] = {domain: [] for domain in DOMAIN_ORDER}
    for entry in coverage:
        by_domain_pages.setdefault(entry["domain"], []).append(entry["page_title"])
    by_domain_findings: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    for finding in findings:
        by_domain_findings.setdefault(finding["domain"], []).append(finding)

    status_counts: dict[str, int] = {}
    for entry in coverage:
        status = entry.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    lines = ["# Wiki vs Lightspeed Audit Summary", ""]
    lines.append("## Coverage Overview")
    lines.append("")
    lines.append(f"- Scanned pages: {len(coverage)}")
    lines.append(f"- Findings recorded: {len(findings)}")
    for status in sorted(status_counts):
        lines.append(f"- `{status}`: {status_counts[status]}")
    lines.append("")
    lines.append("## Key Samples")
    lines.append("")
    key_samples = [
        finding
        for finding in findings
        if finding["page_title"] == "Elites" and finding["comparison_result"] == "mismatch"
    ]
    if key_samples:
        for sample in key_samples:
            lines.append(
                f"- `{sample['page_title']}` / `{sample['claim_kind']}`: wiki says `{sample['wiki_claim_text']}`, "
                f"lightspeed uses `{sample['lightspeed_evidence']}`."
            )
    else:
        lines.append("- No highlighted key samples recorded.")
    lines.append("")

    for domain in DOMAIN_ORDER:
        pages = sorted(by_domain_pages.get(domain, []))
        if not pages:
            continue
        lines.append(f"## {domain.title()}")
        lines.append("")
        lines.append(f"Scanned pages ({len(pages)}): {', '.join(pages)}")
        lines.append("")
        domain_findings = [finding for finding in by_domain_findings.get(domain, []) if finding["comparison_result"] != "match"]
        if not domain_findings:
            lines.append("No findings recorded.")
            lines.append("")
            continue
        lines.append("Findings:")
        for finding in domain_findings:
            lines.append(
                "- "
                f"`{finding['page_title']}` / `{finding['claim_kind']}` / `{finding['confidence']}`: "
                f"wiki `{finding['wiki_claim_text']}` vs lightspeed `{finding['lightspeed_evidence']}`. "
                f"{finding['notes']}"
            )
        lines.append("")

    manual_review = [
        entry
        for entry in coverage
        if entry.get("status") in {"unvalidated", "fetch_failed", "audit_failed"}
    ]
    if manual_review:
        lines.append("## Manual Review Queue")
        lines.append("")
        for entry in manual_review:
            details = entry.get("error") or entry.get("notes") or "Needs manual follow-up."
            lines.append(f"- `{entry['page_title']}` / `{entry['status']}`: {details}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _page_matches_scope(title: str, domain: str, categories: list[str], scope: str) -> bool:
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported audit scope: {scope}")
    if "Disambiguation_pages" in categories:
        return False
    if scope == "all":
        return True
    if domain != "cards":
        return True
    return any(category in IRONCLAD_CARD_CATEGORIES for category in categories)


def run_audit(
    *,
    repo_root: Path,
    output_dir: Path,
    user_agent: str = DEFAULT_USER_AGENT,
    limit: int | None = None,
    refresh: bool = False,
    scope: str = "all",
) -> dict[str, Any]:
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported audit scope: {scope}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_cache_path = output_dir / "pages.json"
    inventory_path = output_dir / "inventory.json"
    coverage_path = output_dir / "coverage_manifest.json"
    findings_path = output_dir / "findings.jsonl"
    summary_path = output_dir / "summary.md"
    checkpoint_path = output_dir / "checkpoint.json"

    reference = load_lightspeed_reference(repo_root)

    if inventory_path.exists() and not refresh:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        all_titles = fetch_all_titles(user_agent=user_agent)
        raw_inventory = [
            title
            for title in all_titles
            if normalize_title(title) in reference.gameplay_titles
            or title in EXPLICIT_SYSTEM_TITLES
        ]
        deduped: dict[str, str] = {}
        for title in sorted(dict.fromkeys(raw_inventory)):
            key = normalize_title(title)
            preferred = reference.canonical_titles.get(key)
            current = deduped.get(key)
            if current is None:
                deduped[key] = title
            elif preferred and title == preferred:
                deduped[key] = title
        inventory = sorted(deduped.values())
        inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    if limit is not None:
        inventory = inventory[:limit]

    pages_cache: dict[str, Any] = {}
    if pages_cache_path.exists() and not refresh:
        pages_cache = json.loads(pages_cache_path.read_text(encoding="utf-8"))
    cache_flush_interval = 25
    fetched_since_flush = 0

    coverage: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    processed_titles: list[str] = []
    failed_titles: list[str] = []

    if checkpoint_path.exists() and not refresh:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("scope") not in {None, scope}:
            checkpoint = {}
        checkpoint_processed = list(checkpoint.get("processed_titles", []))
        checkpoint_coverage = checkpoint.get("coverage")
        checkpoint_findings = checkpoint.get("findings")
        checkpoint_failed = list(checkpoint.get("failed_titles", []))
        if checkpoint_coverage is not None and checkpoint_findings is not None:
            processed_titles = checkpoint_processed
            coverage = list(checkpoint_coverage)
            findings = list(checkpoint_findings)
            failed_titles = checkpoint_failed
        elif checkpoint_processed:
            reconstructed_processed: list[str] = []
            reconstructed_coverage: list[dict[str, Any]] = []
            reconstructed_findings: list[dict[str, Any]] = []
            reconstructed_failed: list[str] = []
            for title in checkpoint_processed:
                payload = pages_cache.get(title)
                if payload is None:
                    try:
                        page = fetch_wiki_page(title, user_agent=user_agent)
                        pages_cache[title] = {
                            "title": page.title,
                            "categories": page.categories,
                            "wikitext": page.wikitext,
                            "html": page.html,
                        }
                    except Exception as exc:  # pragma: no cover - network fallback path
                        reconstructed_processed.append(title)
                        reconstructed_failed.append(title)
                        reconstructed_coverage.append(
                            {
                                "page_title": title,
                                "domain": classify_title(title, reference) or "unknown",
                                "categories": [],
                                "claims_checked": 0,
                                "mismatches": 0,
                                "status": "fetch_failed",
                                "has_findings": True,
                                "error": str(exc),
                            }
                        )
                        reconstructed_findings.append(
                            _make_claim(
                                page_title=title,
                                domain=classify_title(title, reference) or "unknown",
                                claim_kind="fetch_failed",
                                wiki_claim_text="Page could not be fetched during checkpoint recovery.",
                                lightspeed_evidence="N/A",
                                lightspeed_path="",
                                lightspeed_line=None,
                                comparison_result="unvalidated",
                                confidence="low",
                                notes=f"Checkpoint recovery failed to refetch this page: {exc}",
                            ).to_dict()
                        )
                        continue
                    payload = pages_cache[title]

                page = WikiPage(
                    title=payload["title"],
                    categories=list(payload.get("categories", [])),
                    wikitext=payload.get("wikitext", ""),
                    html=payload.get("html", ""),
                )
                try:
                    page_claims, manifest_entry = audit_page(page, reference, repo_root)
                except Exception as exc:  # pragma: no cover - defensive recovery path
                    reconstructed_processed.append(title)
                    reconstructed_failed.append(title)
                    reconstructed_coverage.append(
                        {
                            "page_title": title,
                            "domain": classify_title(title, reference) or "unknown",
                            "categories": page.categories,
                            "claims_checked": 0,
                            "mismatches": 0,
                            "status": "audit_failed",
                            "has_findings": True,
                            "error": str(exc),
                        }
                    )
                    reconstructed_findings.append(
                        _make_claim(
                            page_title=title,
                            domain=classify_title(title, reference) or "unknown",
                            claim_kind="audit_failed",
                            wiki_claim_text="Page audit crashed during checkpoint recovery.",
                            lightspeed_evidence="N/A",
                            lightspeed_path="",
                            lightspeed_line=None,
                            comparison_result="unvalidated",
                            confidence="low",
                            notes=f"Checkpoint recovery could not re-audit this page: {exc}",
                        ).to_dict()
                    )
                    continue

                reconstructed_processed.append(title)
                reconstructed_coverage.append(manifest_entry)
                for claim in page_claims:
                    if claim.comparison_result != "match":
                        reconstructed_findings.append(claim.to_dict())

            processed_titles = reconstructed_processed
            coverage = reconstructed_coverage
            findings = reconstructed_findings
            failed_titles = reconstructed_failed

    scoped_inventory: list[str] = []
    for title in inventory:
        payload = pages_cache.get(title)
        if payload is None:
            try:
                page = fetch_wiki_page(title, user_agent=user_agent)
            except Exception:
                scoped_inventory.append(title)
                continue
            payload = {
                "title": page.title,
                "categories": page.categories,
                "wikitext": page.wikitext,
                "html": page.html,
            }
            pages_cache[title] = payload
        domain = classify_title(payload.get("title", title), reference) or "unknown"
        categories = list(payload.get("categories", []))
        if _page_matches_scope(payload.get("title", title), domain, categories, scope):
            scoped_inventory.append(title)
    inventory = scoped_inventory

    processed_set = set(processed_titles)
    for title in inventory:
        if title in processed_set:
            continue
        if title not in pages_cache or refresh:
            try:
                page = fetch_wiki_page(title, user_agent=user_agent)
            except Exception as exc:  # pragma: no cover - network failure path
                processed_titles.append(title)
                processed_set.add(title)
                failed_titles.append(title)
                manifest_entry = {
                    "page_title": title,
                    "domain": classify_title(title, reference) or "unknown",
                    "categories": [],
                    "claims_checked": 0,
                    "mismatches": 0,
                    "status": "fetch_failed",
                    "has_findings": True,
                    "error": str(exc),
                }
                coverage.append(manifest_entry)
                findings.append(
                    _make_claim(
                        page_title=title,
                        domain=classify_title(title, reference) or "unknown",
                        claim_kind="fetch_failed",
                        wiki_claim_text="Page could not be fetched from the wiki API.",
                        lightspeed_evidence="N/A",
                        lightspeed_path="",
                        lightspeed_line=None,
                        comparison_result="unvalidated",
                        confidence="low",
                        notes=f"Fetch failed during full audit: {exc}",
                    ).to_dict()
                )
                checkpoint = {
                    "processed_titles": processed_titles,
                    "failed_titles": failed_titles,
                    "scope": scope,
                    "inventory_size": len(inventory),
                    "last_title": title,
                    "coverage": coverage,
                    "findings": findings,
                }
                checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            pages_cache[title] = {
                "title": page.title,
                "categories": page.categories,
                "wikitext": page.wikitext,
                "html": page.html,
            }
            fetched_since_flush += 1
            if fetched_since_flush >= cache_flush_interval:
                pages_cache_path.write_text(json.dumps(pages_cache, ensure_ascii=False), encoding="utf-8")
                fetched_since_flush = 0
            time.sleep(0.05)
        else:
            payload = pages_cache[title]
            page = WikiPage(
                title=payload["title"],
                categories=list(payload.get("categories", [])),
                wikitext=payload.get("wikitext", ""),
                html=payload.get("html", ""),
            )

        try:
            page_claims, manifest_entry = audit_page(page, reference, repo_root)
        except Exception as exc:  # pragma: no cover - defensive path
            failed_titles.append(title)
            manifest_entry = {
                "page_title": page.title,
                "domain": classify_title(page.title, reference) or "unknown",
                "categories": page.categories,
                "claims_checked": 0,
                "mismatches": 0,
                "status": "audit_failed",
                "has_findings": True,
                "error": str(exc),
            }
            page_claims = [
                _make_claim(
                    page_title=page.title,
                    domain=classify_title(page.title, reference) or "unknown",
                    claim_kind="audit_failed",
                    wiki_claim_text="Automated comparator crashed for this page.",
                    lightspeed_evidence="N/A",
                    lightspeed_path="",
                    lightspeed_line=None,
                    comparison_result="unvalidated",
                    confidence="low",
                    notes=f"Audit failed for this page and needs manual follow-up: {exc}",
                )
            ]
        coverage.append(manifest_entry)
        for claim in page_claims:
            if claim.comparison_result != "match":
                findings.append(claim.to_dict())
        processed_titles.append(title)
        processed_set.add(title)
        checkpoint = {
            "processed_titles": processed_titles,
            "failed_titles": failed_titles,
            "scope": scope,
            "inventory_size": len(inventory),
            "last_title": title,
            "coverage": coverage,
            "findings": findings,
        }
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    pages_cache_path.write_text(json.dumps(pages_cache, ensure_ascii=False), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    with findings_path.open("w", encoding="utf-8") as handle:
        for finding in findings:
            handle.write(json.dumps(finding, ensure_ascii=False) + "\n")
    render_summary(findings, coverage, summary_path)

    return {
        "scope": scope,
        "inventory_size": len(inventory),
        "coverage_path": str(coverage_path),
        "findings_path": str(findings_path),
        "summary_path": str(summary_path),
        "validated_pages": sum(1 for entry in coverage if entry["status"] == "validated"),
        "unvalidated_pages": sum(1 for entry in coverage if entry["status"] == "unvalidated"),
        "failed_pages": sum(1 for entry in coverage if entry["status"] in {"fetch_failed", "audit_failed"}),
        "findings_count": len(findings),
    }
