from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = PACKAGE_ROOT / "reference"

DEFAULT_LOCALIZATION_ROOT = Path("/home/yydd/sts_instances/align/game/新建文件夹/localization/eng")
DEFAULT_GAME_JAR_PATH = Path("/home/yydd/sts/sts_autodl/sts_instances/autodl/game/desktop-1.0.jar")

BUNDLED_DECOMPILED_ROOT = REFERENCE_ROOT / "decompiled_sts" / "com" / "megacrit" / "cardcrawl"
BUNDLED_LOCALIZATION_ROOT = REFERENCE_ROOT / "localization" / "eng"
REQUIRED_DECOMPILED_SENTINEL = Path("characters/Ironclad.java")


def _root_from_env_or_bundle(env_name: str, bundled: Path, default: Path) -> Path:
    raw_value = os.environ.get(env_name)
    if raw_value:
        return Path(raw_value).expanduser()
    if bundled.exists():
        return bundled
    return default


def _decompiled_root() -> Path:
    raw_value = os.environ.get("SPIRECOMM_STS_DECOMPILED_ROOT")
    if raw_value:
        return Path(raw_value).expanduser()
    return BUNDLED_DECOMPILED_ROOT


DECOMPILED_ROOT = _decompiled_root()
LOCALIZATION_ROOT = _root_from_env_or_bundle(
    "SPIRECOMM_STS_LOCALIZATION_ROOT",
    BUNDLED_LOCALIZATION_ROOT,
    DEFAULT_LOCALIZATION_ROOT,
)
GAME_JAR_PATH = Path(os.environ.get("SPIRECOMM_STS_GAME_JAR", str(DEFAULT_GAME_JAR_PATH))).expanduser()


def sts_source_path(relative_path: str) -> Path:
    if not (DECOMPILED_ROOT / REQUIRED_DECOMPILED_SENTINEL).exists():
        raise FileNotFoundError(
            "Bundled decompiled Slay the Spire sources are missing. "
            f"Expected {DECOMPILED_ROOT / REQUIRED_DECOMPILED_SENTINEL}. "
            "Set SPIRECOMM_STS_DECOMPILED_ROOT only for development overrides."
        )
    return DECOMPILED_ROOT / relative_path


def sts_localization_path(relative_path: str) -> Path:
    return LOCALIZATION_ROOT / relative_path
