from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference" / "localization"
STEAM_RESOURCE_ROOT = Path("/home/yydd/.local/share/Steam/steamapps/common/SlayTheSpire/新建文件夹/localization")


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"#[a-z]|[#@~]", "", text)
    text = text.replace("[", " ").replace("]", " ")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _normalize_option_text(value: Any) -> str:
    return re.sub(r"[0-9]+", "", _normalize_text(value))


def _locale_root(locale: str) -> Path | None:
    locale = str(locale or "").strip()
    if not locale:
        return None
    for root in (REFERENCE_ROOT / locale, STEAM_RESOURCE_ROOT / locale):
        if root.exists():
            return root
    return None


@lru_cache(maxsize=32)
def localized_json(kind: str, locale: str = "zhs") -> dict[str, Any]:
    root = _locale_root(locale)
    if root is None:
        return {}
    path = root / f"{kind}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=8)
def localized_event_name_to_id(locale: str = "zhs") -> dict[str, str]:
    mapping: dict[str, str] = {}
    for event_id, payload in localized_json("events", locale).items():
        if not isinstance(payload, dict):
            continue
        mapping[_normalize_text(event_id)] = str(event_id)
        name = payload.get("NAME")
        if name:
            mapping[_normalize_text(name)] = str(event_id)
    return {key: value for key, value in mapping.items() if key}


def canonical_event_id_from_localized(value: Any, locale: str = "zhs") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return localized_event_name_to_id(locale).get(_normalize_text(raw), raw)


def localized_event_options(event_id: str, locale: str = "zhs") -> list[str]:
    payload = localized_json("events", locale).get(str(event_id or ""))
    if not isinstance(payload, dict):
        return []
    options = payload.get("OPTIONS")
    if not isinstance(options, list):
        return []
    return [str(option or "") for option in options]


def localized_event_option_index(event_id: str, live_label: Any, locale: str = "zhs") -> int | None:
    needle = _normalize_option_text(live_label)
    if not needle:
        return None
    options = localized_event_options(event_id, locale)
    live_bracket = re.search(r"\[([^\]]+)\]", str(live_label or ""))
    if live_bracket is not None:
        live_anchor = _normalize_option_text(live_bracket.group(1))
        if live_anchor:
            for index, option in enumerate(options):
                option_bracket = re.search(r"\[([^\]]+)\]", str(option or ""))
                if option_bracket is None:
                    continue
                if _normalize_option_text(option_bracket.group(1)) == live_anchor:
                    return index
    candidates: list[tuple[int, str]] = []
    for index, option in enumerate(options):
        candidates.append((index, str(option or "")))
    for index, option in enumerate(options):
        if index + 1 < len(options):
            candidates.append((index, f"{option or ''} {options[index + 1] or ''}"))
    best: tuple[int, int] | None = None
    for index, option in candidates:
        haystack = _normalize_option_text(option)
        if not haystack:
            continue
        if needle in haystack or haystack in needle:
            score = len(haystack)
            if best is None or score > best[1]:
                best = (index, score)
    return None if best is None else best[0]
