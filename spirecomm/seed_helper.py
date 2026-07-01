from __future__ import annotations

from typing import Final


SEED_CHARACTERS: Final[str] = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
_LONG_MASK: Final[int] = (1 << 64) - 1
_LONG_SIGN_BIT: Final[int] = 1 << 63


def _to_signed_long(value: int) -> int:
    value = int(value) & _LONG_MASK
    if value >= _LONG_SIGN_BIT:
        value -= 1 << 64
    return value


def seed_long_to_string(seed: int) -> str:
    """Match com.megacrit.cardcrawl.helpers.SeedHelper.getString(long)."""

    unsigned = int(seed) & ((1 << 64) - 1)
    if unsigned == 0:
        return ""
    base = len(SEED_CHARACTERS)
    chars: list[str] = []
    while unsigned:
        unsigned, remainder = divmod(unsigned, base)
        chars.append(SEED_CHARACTERS[remainder])
    return "".join(reversed(chars))


def sterilize_seed_string(raw: str) -> str:
    cleaned = (raw or "").strip().upper().replace("O", "0")
    return "".join(ch for ch in cleaned if ch in SEED_CHARACTERS)


def seed_string_to_long(seed_str: str) -> int:
    """Match com.megacrit.cardcrawl.helpers.SeedHelper.getLong(String)."""

    total = 0
    for ch in sterilize_seed_string(seed_str):
        total *= len(SEED_CHARACTERS)
        total += SEED_CHARACTERS.index(ch)
        total = _to_signed_long(total)
    return _to_signed_long(total)


def canonical_seed_string(seed_long: int | None, seed_str: str | None = None) -> str | None:
    if seed_long is not None:
        return seed_long_to_string(int(seed_long))
    if seed_str:
        cleaned = sterilize_seed_string(seed_str)
        return cleaned or None
    return None
