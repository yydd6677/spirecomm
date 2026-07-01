from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spirecomm.native_sim_v3.source_paths import sts_source_path

MONSTER_ROOM_ELITE_SOURCE = sts_source_path("rooms/MonsterRoomElite.java")

STRENGTH_PATTERN = re.compile(
    r"new StrengthPower\(m,\s*AbstractDungeon\.actNum \+ (\d+)\)"
)
MAX_HP_PATTERN = re.compile(
    r"new IncreaseMaxHpAction\(m,\s*([0-9.]+)f,\s*true\)"
)
METALLICIZE_PATTERN = re.compile(
    r"new MetallicizePower\(m,\s*AbstractDungeon\.actNum \* (\d+) \+ (\d+)\)"
)
REGENERATE_PATTERN = re.compile(
    r"new RegenerateMonsterPower\(m,\s*(\d+) \+ AbstractDungeon\.actNum \* (\d+)\)"
)


@dataclass(frozen=True, slots=True)
class EmeraldEliteRules:
    strength_act_offset: int
    max_hp_bonus_ratio: float
    metallicize_act_multiplier: int
    metallicize_base: int
    regenerate_base: int
    regenerate_act_multiplier: int
    source_path: str

    def strength_amount(self, act: int) -> int:
        return int(act) + self.strength_act_offset

    def metallicize_amount(self, act: int) -> int:
        return int(act) * self.metallicize_act_multiplier + self.metallicize_base

    def regenerate_amount(self, act: int) -> int:
        return self.regenerate_base + int(act) * self.regenerate_act_multiplier


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
def emerald_elite_rules() -> EmeraldEliteRules:
    text = MONSTER_ROOM_ELITE_SOURCE.read_text(encoding="utf-8")
    body = _extract_method_body(text, "applyEmeraldEliteBuff", MONSTER_ROOM_ELITE_SOURCE)
    strength_match = STRENGTH_PATTERN.search(body)
    max_hp_match = MAX_HP_PATTERN.search(body)
    metallicize_match = METALLICIZE_PATTERN.search(body)
    regenerate_match = REGENERATE_PATTERN.search(body)
    if not all((strength_match, max_hp_match, metallicize_match, regenerate_match)):
        raise ValueError("could not parse MonsterRoomElite emerald elite buff rules from source")
    return EmeraldEliteRules(
        strength_act_offset=int(strength_match.group(1)),
        max_hp_bonus_ratio=float(max_hp_match.group(1)),
        metallicize_act_multiplier=int(metallicize_match.group(1)),
        metallicize_base=int(metallicize_match.group(2)),
        regenerate_base=int(regenerate_match.group(1)),
        regenerate_act_multiplier=int(regenerate_match.group(2)),
        source_path=str(MONSTER_ROOM_ELITE_SOURCE),
    )
