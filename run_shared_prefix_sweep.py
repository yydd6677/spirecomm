#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import evaluate_v3_rollout_batch as rollout_eval
from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_model_required_action
from spirecomm.native_sim_v3 import NativeRunEnv


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}

_CONFIG: dict[str, Any] = {}
_CANDIDATES: list[dict[str, Any]] = []
_SELECTORS: dict[str, Any] | None = None
_CANDIDATE_ENV_KEYS: set[str] = set()
_BASE_ENV_VALUES: dict[str, str | None] = {}
_BASE_RUNTIME_GLOBALS: dict[str, Any] = {}
_BASE_CARD_REWARD_GLOBALS: dict[str, Any] = {}
_CANDIDATE_MAP_PARAMS: dict[int, dict[str, int]] = {}


MAP_DP_GLOBALS = {
    "SPIRECOMM_MAP_DP_MONSTER_VALUE": "MAP_DP_MONSTER_VALUE",
    "SPIRECOMM_MAP_DP_REST_VALUE": "MAP_DP_REST_VALUE",
    "SPIRECOMM_MAP_DP_ELITE_BASE": "MAP_DP_ELITE_BASE_VALUE",
    "SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY": "MAP_DP_GREEN_ELITE_PENALTY",
    "SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY": "MAP_DP_WINGED_OFFPATH_PENALTY",
    "SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE": "MAP_DP_SHOP_GOLD_UNIT_VALUE",
    "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS": "MAP_DP_SHOP_PURGEABLE_CURSE_BONUS",
    "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS": "MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS",
    "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON": "MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON",
    "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD": "MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD",
}

CARD_REWARD_GLOBALS = {
    "SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_MAX": "EARLY_CARD_REWARD_ATTACK_BIAS_MAX",
    "SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_POWER": "EARLY_CARD_REWARD_ATTACK_BIAS_POWER",
}

SHOP_ENV_PREFIXES = ("SPIRECOMM_SHOP_",)
CARD_ENV_PREFIXES = ("SPIRECOMM_CARD_", "SPIRECOMM_EARLY_CARD_")
CARD_TARGET_ENV_PREFIXES = ("SPIRECOMM_TRUE_GRIT_", "SPIRECOMM_POSITIVE_CARD_TARGET_")


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("candidates") or payload.get("groups") or []
    if not isinstance(payload, list):
        raise SystemExit(f"candidate JSON must be a list or contain candidates/groups: {path}")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise SystemExit(f"candidate {index} is not an object")
        name = str(raw.get("name") or f"candidate_{index:03d}")
        if name in seen:
            name = f"{name}_{index:03d}"
        seen.add(name)
        env = raw.get("env") or raw.get("env_overrides") or {}
        if not isinstance(env, dict):
            raise SystemExit(f"candidate {name} env/env_overrides is not an object")
        candidates.append(
            {
                "index": int(index),
                "name": name,
                "kind": str(raw.get("kind") or ""),
                "env": {str(key): str(value) for key, value in env.items()},
                "params": raw.get("params") if isinstance(raw.get("params"), dict) else {},
            }
        )
    if not candidates:
        raise SystemExit(f"no candidates in {path}")
    return candidates


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _set_env_if_value(name: str, value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text == "":
        return
    os.environ[name] = text


def _capture_base_runtime_values(env_keys: set[str]) -> None:
    global _BASE_ENV_VALUES, _BASE_RUNTIME_GLOBALS, _BASE_CARD_REWARD_GLOBALS
    _BASE_ENV_VALUES = {key: os.environ.get(key) for key in env_keys}
    import spirecomm.ai.runtime_decision as runtime_decision

    _BASE_RUNTIME_GLOBALS = {
        env_name: getattr(runtime_decision, attr_name)
        for env_name, attr_name in MAP_DP_GLOBALS.items()
        if hasattr(runtime_decision, attr_name)
    }
    try:
        import spirecomm.ai.card_reward_model as card_reward_model
    except Exception:
        card_reward_model = None
    if card_reward_model is not None:
        _BASE_CARD_REWARD_GLOBALS = {
            env_name: getattr(card_reward_model, attr_name)
            for env_name, attr_name in CARD_REWARD_GLOBALS.items()
            if env_name in env_keys and hasattr(card_reward_model, attr_name)
        }
    else:
        _BASE_CARD_REWARD_GLOBALS = {}


def _apply_known_module_globals(effective_values: dict[str, str | None]) -> None:
    import spirecomm.ai.runtime_decision as runtime_decision

    for env_name, attr_name in MAP_DP_GLOBALS.items():
        if env_name not in _CANDIDATE_ENV_KEYS:
            continue
        value = effective_values.get(env_name)
        try:
            if value is None:
                setattr(runtime_decision, attr_name, _BASE_RUNTIME_GLOBALS[env_name])
            else:
                setattr(runtime_decision, attr_name, int(float(value)))
        except (TypeError, ValueError):
            pass

    try:
        import spirecomm.ai.card_reward_model as card_reward_model
    except Exception:
        card_reward_model = None
    if card_reward_model is not None:
        for env_name, attr_name in CARD_REWARD_GLOBALS.items():
            if env_name not in _CANDIDATE_ENV_KEYS:
                continue
            value = effective_values.get(env_name)
            try:
                if value is None:
                    setattr(card_reward_model, attr_name, _BASE_CARD_REWARD_GLOBALS[env_name])
                else:
                    setattr(card_reward_model, attr_name, float(value))
            except (TypeError, ValueError):
                pass


def _apply_candidate(candidate_index: int) -> None:
    env_overrides = _CANDIDATES[int(candidate_index)]["env"]
    effective_values: dict[str, str | None] = {}
    for name in _CANDIDATE_ENV_KEYS:
        if name in env_overrides:
            value = str(env_overrides[name])
        else:
            value = _BASE_ENV_VALUES.get(name)
        effective_values[name] = value
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _apply_known_module_globals(effective_values)


def _default_phase_env_keys(phase: str) -> set[str]:
    normalized = str(phase or "").strip().upper()
    if normalized == "MAP":
        return {key for key in _CANDIDATE_ENV_KEYS if key in MAP_DP_GLOBALS}
    if normalized == "SHOP":
        return {
            key
            for key in _CANDIDATE_ENV_KEYS
            if key.startswith(SHOP_ENV_PREFIXES) or key.startswith(CARD_ENV_PREFIXES)
        }
    if normalized == "CARD_REWARD":
        return {key for key in _CANDIDATE_ENV_KEYS if key.startswith(CARD_ENV_PREFIXES)}
    if normalized == "CARD_SELECT":
        return {
            key
            for key in _CANDIDATE_ENV_KEYS
            if key.startswith(CARD_ENV_PREFIXES) or key.startswith(CARD_TARGET_ENV_PREFIXES)
        }
    if normalized == "CAMPFIRE":
        return {
            key
            for key in _CANDIDATE_ENV_KEYS
            if key.startswith(CARD_ENV_PREFIXES) or key.startswith(CARD_TARGET_ENV_PREFIXES)
        }
    return set(_CANDIDATE_ENV_KEYS)


def _phase_env_keys(phase: str) -> set[str]:
    explicit = _CONFIG.get("phase_env_keys") or {}
    if isinstance(explicit, dict):
        keys = explicit.get(str(phase or "").strip().upper())
        if keys is not None:
            return {str(key) for key in keys}
    return _default_phase_env_keys(phase)


def _candidate_phase_signature(candidate_index: int, phase: str) -> tuple[tuple[str, str | None], ...]:
    env_overrides = _CANDIDATES[int(candidate_index)]["env"]
    keys = _phase_env_keys(phase)
    return tuple(
        (key, str(env_overrides[key]) if key in env_overrides else _BASE_ENV_VALUES.get(key))
        for key in sorted(keys)
    )


def _candidate_env_value(candidate_index: int, key: str) -> str | None:
    env_overrides = _CANDIDATES[int(candidate_index)]["env"]
    if key in env_overrides:
        return str(env_overrides[key])
    value = _BASE_ENV_VALUES.get(key)
    if value is not None:
        return str(value)
    if key in _BASE_RUNTIME_GLOBALS:
        return str(_BASE_RUNTIME_GLOBALS[key])
    return None


def _candidate_int_value(candidate_index: int, key: str, default: int = 0) -> int:
    raw = _candidate_env_value(candidate_index, key)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def _candidate_map_params(candidate_index: int) -> dict[str, int]:
    cached = _CANDIDATE_MAP_PARAMS.get(int(candidate_index))
    if cached is not None:
        return cached
    params = {
        "monster_value": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_MONSTER_VALUE", -10),
        "rest_value": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_REST_VALUE", 50),
        "elite_base": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_ELITE_BASE", 20),
        "green_elite_penalty": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY", 40),
        "winged_offpath_penalty": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY", 20),
        "shop_gold_unit": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE", 20),
        "shop_curse_bonus": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS", 50),
        "shop_curse_urgency_bonus": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS", 50),
        "shop_curse_urgency_horizon": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON", 5),
        "shop_curse_gold_threshold": _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD", 125),
    }
    _CANDIDATE_MAP_PARAMS[int(candidate_index)] = params
    return params


def _map_symbols(env: Any) -> frozenset[str]:
    nodes = getattr(env, "map", None)
    if not nodes:
        return frozenset()
    symbols: set[str] = set()
    try:
        for row in list(nodes or []):
            for node in list(row or []):
                symbol = str(getattr(node, "room_symbol", "") or "")
                if not symbol:
                    continue
                if symbol == "E" and bool(getattr(node, "has_emerald_key", False)):
                    symbols.add("E_GREEN")
                symbols.add(symbol)
    except Exception:
        return frozenset()
    return frozenset(symbols)


def _map_has_purgeable_curse(env: Any) -> bool:
    for card in list(getattr(env, "deck", []) or []):
        if str(card.get("type") or "") != "CURSE":
            continue
        if str(card.get("card_id") or "") in {"AscendersBane", "CurseOfTheBell", "Necronomicurse"}:
            continue
        if bool(card.get("bottled") or card.get("in_bottle_flame") or card.get("in_bottle_lightning") or card.get("in_bottle_tornado")):
            continue
        return True
    return False


def _map_winged_charges(env: Any) -> int:
    for relic in list(getattr(env, "relics", []) or []):
        if str(relic.get("relic_id") or relic.get("id") or "") != "WingedGreaves":
            continue
        return max(0, int(relic.get("counter") or 0))
    return 0


def _phase_signature_context(phase: str, env: Any) -> dict[str, Any]:
    if str(phase or "").strip().upper() != "MAP":
        return {}
    return {
        "symbols": _map_symbols(env),
        "gold": max(0, int(getattr(env, "gold", 0) or 0)),
        "has_purgeable_curse": _map_has_purgeable_curse(env),
        "winged_charges": _map_winged_charges(env),
    }


def _candidate_map_effective_signature(candidate_index: int, env: Any, context: dict[str, Any] | None = None) -> tuple[tuple[str, str | None], ...]:
    keys = _phase_env_keys("MAP")
    if not keys:
        return ()
    context = context or _phase_signature_context("MAP", env)
    symbols = frozenset(context.get("symbols") or ())
    signature: list[tuple[str, str | None]] = []
    if "SPIRECOMM_MAP_DP_MONSTER_VALUE" in keys and "M" in symbols:
        signature.append(("monster", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_MONSTER_VALUE")))
    if "SPIRECOMM_MAP_DP_REST_VALUE" in keys and "R" in symbols:
        signature.append(("rest", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_REST_VALUE")))
    if "SPIRECOMM_MAP_DP_ELITE_BASE" in keys and ({"E", "E_GREEN"} & set(symbols)):
        signature.append(("elite_base", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_ELITE_BASE")))
    if "SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY" in keys and "E_GREEN" in symbols:
        signature.append(("green_elite_penalty", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY")))
    if "SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY" in keys and int(context.get("winged_charges") or 0) > 0:
        signature.append(("winged_offpath_penalty", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY")))
    has_shop = "$" in symbols
    gold = int(context.get("gold") or 0)
    if "SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE" in keys and has_shop and gold // 100 > 0:
        signature.append(("shop_gold_unit", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE")))
    has_curse_shop_bonus = has_shop and bool(context.get("has_purgeable_curse"))
    if has_curse_shop_bonus:
        threshold = _candidate_int_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD", 125)
        active = gold >= threshold
        if "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD" in keys:
            signature.append(("curse_threshold_active", "1" if active else "0"))
        if active:
            if "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS" in keys:
                signature.append(("curse_bonus", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS")))
            if "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS" in keys:
                signature.append(("curse_urgency_bonus", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS")))
            if "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON" in keys:
                signature.append(("curse_urgency_horizon", _candidate_env_value(candidate_index, "SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON")))
    return tuple(signature)


def _candidate_effective_phase_signature(
    candidate_index: int,
    phase: str,
    env: Any,
    context: dict[str, Any] | None = None,
) -> tuple[tuple[str, str | None], ...]:
    if str(phase or "").strip().upper() == "MAP":
        return _candidate_map_effective_signature(candidate_index, env, context)
    return _candidate_phase_signature(candidate_index, phase)


def _map_dp_state_key(env: Any) -> str | None:
    current = getattr(env, "current_map_node", None)
    if current is None:
        return None
    x, y = current
    act = int(getattr(env, "act", 1) or 1)
    return f"a{act}-r{int(y)}-x{int(x)}"


def _map_initial_state(env: Any, params: dict[str, int]) -> tuple[int, int, int]:
    shop_value = (max(0, int(getattr(env, "gold", 0) or 0)) // 100) * int(params["shop_gold_unit"])
    winged_charges = _map_winged_charges(env)
    if not bool(getattr(env, "first_room_chosen", False)):
        return (0, shop_value, winged_charges)
    states = getattr(env, "_map_dp_state_by_node", None)
    key = _map_dp_state_key(env)
    if isinstance(states, dict) and key in states:
        value = states[key]
        return (int(value[0]), shop_value, winged_charges)
    return (int(getattr(env, "_map_dp_elite_count", 0)), shop_value, winged_charges)


def _map_shop_bonus(env: Any, params: dict[str, int]) -> int:
    if int(getattr(env, "gold", 0) or 0) >= int(params["shop_curse_gold_threshold"]) and _map_has_purgeable_curse(env):
        return int(params["shop_curse_bonus"])
    return 0


def _map_shop_urgency_bonus(env: Any, params: dict[str, int], *, distance: int) -> int:
    if int(distance) > int(params["shop_curse_urgency_horizon"]):
        return 0
    if int(getattr(env, "gold", 0) or 0) >= int(params["shop_curse_gold_threshold"]) and _map_has_purgeable_curse(env):
        return int(params["shop_curse_urgency_bonus"])
    return 0


def _map_node_score(
    symbol: str | None,
    elite_count: int,
    shop_value: int,
    params: dict[str, int],
    *,
    shop_bonus: int = 0,
    shop_urgency_bonus: int = 0,
) -> tuple[int, tuple[int, int]]:
    token = str(symbol or "")
    if token == "E_GREEN":
        score = int(params["elite_base"]) - 20 * int(elite_count) - int(params["green_elite_penalty"])
        return score, (int(elite_count) + 1, int(shop_value))
    if token == "E":
        score = int(params["elite_base"]) - 20 * int(elite_count)
        return score, (int(elite_count) + 1, int(shop_value))
    if token == "$":
        return int(shop_value) + int(shop_bonus) + int(shop_urgency_bonus), (int(elite_count), int(shop_value))
    if token == "?":
        return 10, (int(elite_count), int(shop_value))
    if token == "M":
        return int(params["monster_value"]), (int(elite_count), int(shop_value))
    if token == "R":
        return int(params["rest_value"]), (int(elite_count), int(shop_value))
    if token == "T":
        return 100, (int(elite_count), int(shop_value))
    return 0, (int(elite_count), int(shop_value))


def _map_with_winged_state(state: tuple[int, int, int], next_base_state: tuple[int, int], winged_cost: int) -> tuple[int, int, int]:
    return (
        int(next_base_state[0]),
        int(next_base_state[1]),
        max(0, int(state[2]) - max(0, int(winged_cost))),
    )


def _map_winged_offpath_penalty(params: dict[str, int], winged_cost: int) -> int:
    return int(params["winged_offpath_penalty"]) if int(winged_cost) > 0 else 0


def _map_node_symbol(node: Any) -> str:
    symbol = str(getattr(node, "room_symbol", "") or "")
    if symbol == "E" and bool(getattr(node, "has_emerald_key", False)):
        return "E_GREEN"
    return symbol


def _map_best_future_score(
    env: Any,
    node: Any,
    state: tuple[int, int, int],
    params: dict[str, int],
    memo: dict[tuple[int, int, int, int, int, int], int],
    *,
    steps_from_current: int,
) -> int:
    key = (
        int(getattr(node, "x", 0)),
        int(getattr(node, "y", 0)),
        int(state[0]),
        int(state[1]),
        int(state[2]),
        int(steps_from_current),
    )
    if key in memo:
        return memo[key]

    best = 0
    nodes = getattr(env, "map", None)
    edges = list(getattr(node, "edges", []) or [])
    normal_targets = {
        (int(edge.dst_x), int(edge.dst_y))
        for edge in edges
        if int(getattr(edge, "dst_y", 0)) < len(nodes)
    }
    if not normal_targets and any(int(getattr(edge, "dst_y", 0)) >= len(nodes) for edge in edges):
        memo[key] = 0
        return 0

    candidates: list[tuple[Any, int]] = []
    if int(state[2]) > 0 and normal_targets:
        target_y = min(y for _, y in normal_targets)
        for child in list(nodes[target_y]):
            if getattr(child, "room_symbol", None) is None or not bool(child.has_edges()):
                continue
            winged_cost = 0 if (int(getattr(child, "x", 0)), int(getattr(child, "y", 0))) in normal_targets else 1
            candidates.append((child, winged_cost))
    else:
        for x, y in sorted(normal_targets, key=lambda item: (item[1], item[0])):
            candidates.append((nodes[y][x], 0))

    for child, winged_cost in candidates:
        symbol = _map_node_symbol(child)
        immediate, next_base_state = _map_node_score(
            symbol,
            state[0],
            state[1],
            params,
            shop_bonus=_map_shop_bonus(env, params),
            shop_urgency_bonus=_map_shop_urgency_bonus(env, params, distance=int(steps_from_current) + 1),
        )
        next_state = _map_with_winged_state(state, next_base_state, winged_cost)
        candidate_score = immediate - _map_winged_offpath_penalty(params, winged_cost) + _map_best_future_score(
            env,
            child,
            next_state,
            params,
            memo,
            steps_from_current=int(steps_from_current) + 1,
        )
        if candidate_score > best:
            best = candidate_score
    memo[key] = best
    return best


def _map_action_winged_cost(env: Any, action: dict[str, Any]) -> int:
    if int(_map_winged_charges(env)) <= 0 or getattr(env, "current_map_node", None) is None:
        return 0
    if str(action.get("symbol") or "") == "BOSS":
        return 0
    try:
        _, row, x_token = str(action["node_id"]).split("-")
        y = int(row.removeprefix("r"))
        x = int(x_token.removeprefix("x"))
    except Exception:
        return 0
    current_x, current_y = getattr(env, "current_map_node")
    edges = list(getattr(getattr(env, "map", [])[current_y][current_x], "edges", []) or [])
    normal_connection = any(int(edge.dst_x) == x and int(edge.dst_y) == y for edge in edges)
    winged_connection = any(int(edge.dst_y) == y for edge in edges)
    return 1 if winged_connection and not normal_connection else 0


def _choose_map_action_direct(env: Any, candidate_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    actions = env.legal_actions()
    nodes = getattr(env, "map", None)
    if not nodes:
        return dict(actions[0]), {}
    params = _candidate_map_params(candidate_index)
    initial_state = _map_initial_state(env, params)
    shop_bonus = _map_shop_bonus(env, params)
    memo: dict[tuple[int, int, int, int, int, int], int] = {}
    scores: list[float] = []
    selected_states: list[tuple[int, int, int] | None] = []
    winged_costs: list[int] = []
    for action in actions:
        symbol = str(action.get("symbol") or "")
        if symbol == "BOSS":
            scores.append(0.0)
            selected_states.append(initial_state)
            winged_costs.append(0)
            continue
        try:
            _, row, x_token = str(action["node_id"]).split("-")
            y = int(row.removeprefix("r"))
            x = int(x_token.removeprefix("x"))
            node = nodes[y][x]
        except Exception:
            scores.append(float("-inf"))
            selected_states.append(None)
            winged_costs.append(0)
            continue
        winged_cost = _map_action_winged_cost(env, action)
        immediate, next_state = _map_node_score(
            symbol,
            initial_state[0],
            initial_state[1],
            params,
            shop_bonus=shop_bonus,
            shop_urgency_bonus=_map_shop_urgency_bonus(env, params, distance=1),
        )
        next_state = _map_with_winged_state(initial_state, next_state, winged_cost)
        total_score = (
            immediate
            - _map_winged_offpath_penalty(params, winged_cost)
            + _map_best_future_score(env, node, next_state, params, memo, steps_from_current=1)
        )
        scores.append(float(total_score))
        selected_states.append(next_state)
        winged_costs.append(winged_cost)
    best_index = max(
        range(len(actions)),
        key=lambda index: (
            scores[index],
            -int(winged_costs[index]),
            -int(actions[index].get("choice_index", index)),
        ),
    )
    selected_action = dict(actions[best_index])
    selected_state = selected_states[best_index]
    metadata: dict[str, Any] = {}
    if selected_state is not None and selected_action.get("node_id"):
        metadata = {
            "map_node_id": str(selected_action["node_id"]),
            "map_selected_state": tuple(int(value) for value in selected_state),
        }
    return selected_action, metadata


def _action_group_signature(action: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"action": rollout_eval._jsonable_action(dict(action))}
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _apply_action_metadata(env: Any, metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return
    if "map_selected_state" not in metadata or "map_node_id" not in metadata:
        return
    selected_state = tuple(int(value) for value in metadata["map_selected_state"])
    setattr(env, "_map_dp_elite_count", int(selected_state[0]))
    setattr(env, "_map_dp_shop_value", int(selected_state[1]))
    states = getattr(env, "_map_dp_state_by_node", None)
    if not isinstance(states, dict):
        states = {}
        setattr(env, "_map_dp_state_by_node", states)
    states[str(metadata["map_node_id"])] = selected_state


def _choose_map_action_groups_direct(
    env: Any,
    candidate_indices: tuple[int, ...],
    context: dict[str, Any],
) -> tuple[dict[str, tuple[dict[str, Any], list[int], dict[str, Any]]], int]:
    signature_groups: dict[tuple[tuple[str, str | None], ...], list[int]] = {}
    for index in candidate_indices:
        signature_groups.setdefault(
            _candidate_effective_phase_signature(index, "MAP", env, context),
            [],
        ).append(index)
    action_groups: dict[str, tuple[dict[str, Any], list[int], dict[str, Any]]] = {}
    for signature_indices in signature_groups.values():
        index = signature_indices[0]
        action, metadata = _choose_map_action_direct(env, index)
        action_signature = _action_group_signature(action, metadata)
        if action_signature in action_groups:
            action_groups[action_signature][1].extend(signature_indices)
        else:
            action_groups[action_signature] = (action, list(signature_indices), metadata)
    return action_groups, len(signature_groups)


def _choose_map_action_groups_vectorized(
    env: Any,
    candidate_indices: tuple[int, ...],
    context: dict[str, Any],
) -> tuple[dict[str, tuple[dict[str, Any], list[int], dict[str, Any]]], int]:
    signature_groups: dict[tuple[tuple[str, str | None], ...], list[int]] = {}
    for index in candidate_indices:
        signature_groups.setdefault(
            _candidate_effective_phase_signature(index, "MAP", env, context),
            [],
        ).append(index)
    rep_indices = [indices[0] for indices in signature_groups.values()]
    member_lists = [list(indices) for indices in signature_groups.values()]
    if len(rep_indices) <= 1:
        return _choose_map_action_groups_direct(env, candidate_indices, context)

    actions = env.legal_actions()
    nodes = getattr(env, "map", None)
    if not nodes:
        action_groups = {
            _action_group_signature(dict(actions[0]), {}): (dict(actions[0]), list(candidate_indices), {})
        }
        return action_groups, 1

    params_list = [_candidate_map_params(index) for index in rep_indices]
    initial_states = [_map_initial_state(env, params) for params in params_list]
    elite_winged_pairs = {(int(state[0]), int(state[2])) for state in initial_states}
    if len(elite_winged_pairs) != 1:
        # This should not happen for candidates sharing one env branch. Keep an
        # exact fallback rather than over-vectorizing a malformed state.
        return _choose_map_action_groups_direct(env, candidate_indices, context)
    initial_elite_count, initial_winged = next(iter(elite_winged_pairs))
    shop_values = [int(state[1]) for state in initial_states]
    gold = int(context.get("gold") or 0)
    has_purgeable_curse = bool(context.get("has_purgeable_curse"))
    n = len(rep_indices)

    def shop_bonus_vector() -> list[int]:
        if not has_purgeable_curse:
            return [0] * n
        return [
            int(params["shop_curse_bonus"]) if gold >= int(params["shop_curse_gold_threshold"]) else 0
            for params in params_list
        ]

    base_shop_bonus = shop_bonus_vector()

    def shop_urgency_bonus_vector(distance: int) -> list[int]:
        if not has_purgeable_curse:
            return [0] * n
        result: list[int] = []
        for params in params_list:
            if int(distance) > int(params["shop_curse_urgency_horizon"]):
                result.append(0)
            elif gold >= int(params["shop_curse_gold_threshold"]):
                result.append(int(params["shop_curse_urgency_bonus"]))
            else:
                result.append(0)
        return result

    def node_score_vector(symbol: str | None, elite_count: int, *, distance: int) -> tuple[list[int], int]:
        token = str(symbol or "")
        if token == "E_GREEN":
            return (
                [
                    int(params["elite_base"]) - 20 * int(elite_count) - int(params["green_elite_penalty"])
                    for params in params_list
                ],
                int(elite_count) + 1,
            )
        if token == "E":
            return (
                [int(params["elite_base"]) - 20 * int(elite_count) for params in params_list],
                int(elite_count) + 1,
            )
        if token == "$":
            urgency = shop_urgency_bonus_vector(distance)
            return (
                [
                    int(shop_values[index]) + int(base_shop_bonus[index]) + int(urgency[index])
                    for index in range(n)
                ],
                int(elite_count),
            )
        if token == "?":
            return [10] * n, int(elite_count)
        if token == "M":
            return [int(params["monster_value"]) for params in params_list], int(elite_count)
        if token == "R":
            return [int(params["rest_value"]) for params in params_list], int(elite_count)
        if token == "T":
            return [100] * n, int(elite_count)
        return [0] * n, int(elite_count)

    def winged_penalty_vector(winged_cost: int) -> list[int]:
        if int(winged_cost) <= 0:
            return [0] * n
        return [int(params["winged_offpath_penalty"]) for params in params_list]

    memo: dict[tuple[int, int, int, int, int], list[int]] = {}

    def best_future_vector(node: Any, elite_count: int, winged_charges: int, steps_from_current: int) -> list[int]:
        key = (
            int(getattr(node, "x", 0)),
            int(getattr(node, "y", 0)),
            int(elite_count),
            int(winged_charges),
            int(steps_from_current),
        )
        cached = memo.get(key)
        if cached is not None:
            return cached

        best = [0] * n
        edges = list(getattr(node, "edges", []) or [])
        normal_targets = {
            (int(edge.dst_x), int(edge.dst_y))
            for edge in edges
            if int(getattr(edge, "dst_y", 0)) < len(nodes)
        }
        if not normal_targets and any(int(getattr(edge, "dst_y", 0)) >= len(nodes) for edge in edges):
            memo[key] = best
            return best

        child_candidates: list[tuple[Any, int]] = []
        if int(winged_charges) > 0 and normal_targets:
            target_y = min(y for _, y in normal_targets)
            for child in list(nodes[target_y]):
                if getattr(child, "room_symbol", None) is None or not bool(child.has_edges()):
                    continue
                winged_cost = 0 if (int(getattr(child, "x", 0)), int(getattr(child, "y", 0))) in normal_targets else 1
                child_candidates.append((child, winged_cost))
        else:
            for x, y in sorted(normal_targets, key=lambda item: (item[1], item[0])):
                child_candidates.append((nodes[y][x], 0))

        for child, winged_cost in child_candidates:
            immediate, next_elite_count = node_score_vector(
                _map_node_symbol(child),
                elite_count,
                distance=int(steps_from_current) + 1,
            )
            next_winged = max(0, int(winged_charges) - max(0, int(winged_cost)))
            future = best_future_vector(
                child,
                next_elite_count,
                next_winged,
                int(steps_from_current) + 1,
            )
            penalties = winged_penalty_vector(winged_cost)
            for index in range(n):
                score = int(immediate[index]) - int(penalties[index]) + int(future[index])
                if score > best[index]:
                    best[index] = score
        memo[key] = best
        return best

    best_keys: list[tuple[float, int, int]] = [(float("-inf"), 0, 0) for _ in range(n)]
    best_actions: list[dict[str, Any] | None] = [None] * n
    best_metadata: list[dict[str, Any]] = [{} for _ in range(n)]
    for action in actions:
        symbol = str(action.get("symbol") or "")
        winged_cost = 0
        selected_elite_count = int(initial_elite_count)
        selected_winged = int(initial_winged)
        if symbol == "BOSS":
            scores = [0.0] * n
        else:
            try:
                _, row, x_token = str(action["node_id"]).split("-")
                y = int(row.removeprefix("r"))
                x = int(x_token.removeprefix("x"))
                node = nodes[y][x]
            except Exception:
                scores = [float("-inf")] * n
                node = None
            if node is not None:
                winged_cost = _map_action_winged_cost(env, action)
                immediate, selected_elite_count = node_score_vector(
                    symbol,
                    int(initial_elite_count),
                    distance=1,
                )
                selected_winged = max(0, int(initial_winged) - max(0, int(winged_cost)))
                future = best_future_vector(node, selected_elite_count, selected_winged, 1)
                penalties = winged_penalty_vector(winged_cost)
                scores = [
                    float(int(immediate[index]) - int(penalties[index]) + int(future[index]))
                    for index in range(n)
                ]
        choice_index = int(action.get("choice_index", actions.index(action)))
        for index in range(n):
            key = (float(scores[index]), -int(winged_cost), -choice_index)
            if key > best_keys[index]:
                best_keys[index] = key
                best_actions[index] = dict(action)
                if action.get("node_id") and scores[index] != float("-inf"):
                    # Future MAP state only reads elite_count from this tuple;
                    # shop value and winged charges are recomputed from live env.
                    best_metadata[index] = {
                        "map_node_id": str(action["node_id"]),
                        "map_selected_state": (int(selected_elite_count), 0, int(selected_winged)),
                    }
                else:
                    best_metadata[index] = {}

    action_groups: dict[str, tuple[dict[str, Any], list[int], dict[str, Any]]] = {}
    for rep_pos, rep_index in enumerate(rep_indices):
        action = best_actions[rep_pos]
        if action is None:
            action, metadata = _choose_map_action_direct(env, rep_index)
        else:
            metadata = best_metadata[rep_pos]
        action_signature = _action_group_signature(action, metadata)
        if action_signature in action_groups:
            action_groups[action_signature][1].extend(member_lists[rep_pos])
        else:
            action_groups[action_signature] = (action, list(member_lists[rep_pos]), metadata)
    return action_groups, 1


def _init_worker(config: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    global _CONFIG, _CANDIDATES, _SELECTORS, _CANDIDATE_ENV_KEYS, _CANDIDATE_MAP_PARAMS
    _CONFIG = dict(config)
    _CANDIDATES = [dict(candidate) for candidate in candidates]
    _CANDIDATE_MAP_PARAMS = {}
    _CANDIDATE_ENV_KEYS = {
        str(key)
        for candidate in _CANDIDATES
        for key in (candidate.get("env") or {}).keys()
    }
    rollout_eval._disable_gc_for_hot_worker_if_enabled()
    for env_name, config_key in (
        ("SPIRECOMM_SHOP_POLICY", "shop_policy"),
        ("SPIRECOMM_SHOP_VALUE_PRICE_COST", "shop_value_price_cost"),
        ("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "shop_value_reserve_shortfall_cost"),
        ("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "shop_value_future_shop_reserve"),
        ("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "shop_value_future_shop_horizon"),
        ("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "shop_value_card_scale"),
        ("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "shop_value_card_reference_price"),
        ("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "shop_value_card_price_factor_min"),
        ("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "shop_value_card_price_factor_max"),
        ("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "shop_value_potion_scale"),
        ("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "shop_value_relic_scale"),
        ("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "shop_value_item_scale"),
        ("SPIRECOMM_SHOP_VALUE_THRESHOLD", "shop_value_threshold"),
        ("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "shop_prior_weight_override"),
        ("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "v3_normal_room_potion_penalty"),
    ):
        if config_key in _CONFIG and _CONFIG[config_key] is not None:
            _set_env_if_value(env_name, _CONFIG[config_key])
    _capture_base_runtime_values(_CANDIDATE_ENV_KEYS)
    torch_threads = int(_CONFIG.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
            from spirecomm.ai.torch_compat import torch

            if torch is not None:
                torch.set_num_threads(torch_threads)
                torch.set_num_interop_threads(1)
        except Exception:
            pass
    rollout_eval._prewarm_native_content_caches()
    if bool(_CONFIG.get("selectors_preloaded")) and _SELECTORS is not None:
        return
    _SELECTORS = build_runtime_selectors(
        repo_root=Path(_CONFIG["repo_root"]),
        device=str(_CONFIG["device"]),
        combat_device=_CONFIG.get("combat_device"),
        combat_selector=str(_CONFIG["combat_selector"]),
        combat_model=Path(_CONFIG["combat_model"]),
        v3_combat_model=Path(_CONFIG["v3_combat_model"]),
        card_reward_model=Path(_CONFIG["card_reward_model"]),
        shop_model=Path(_CONFIG["shop_choice_model"]),
    )


def _action_signature(action: dict[str, Any]) -> str:
    return json.dumps(rollout_eval._jsonable_action(dict(action)), ensure_ascii=False, sort_keys=True)


def _env_digest(env: NativeRunEnv) -> str:
    return hashlib.blake2b(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL), digest_size=16).hexdigest()


def _result_from_env(
    *,
    seed: int,
    env: NativeRunEnv,
    steps: int,
    started: float,
    error: str | None,
    timeout_reason: str | None,
) -> dict[str, Any]:
    max_floor = int(_CONFIG["max_floor"])
    timed_out = str(env.phase) not in TERMINAL_PHASES and not error and (steps >= int(_CONFIG["max_steps"]) or timeout_reason is not None)
    max_floor_stopped = int(env.floor) > max_floor and str(env.phase) not in TERMINAL_PHASES
    return {
        "seed": int(seed),
        "ascension": int(_CONFIG["ascension"]),
        "phase": str(env.phase),
        "floor": int(env.floor),
        "hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(env.gold),
        "deck_size": len(env.deck),
        "relic_count": len(env.relics),
        "potion_count": len(env.potions),
        "steps": int(steps),
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "timed_out": bool(timed_out),
        "timeout_reason": timeout_reason,
        "max_floor_stopped": bool(max_floor_stopped),
        "error": error,
        "seconds": time.time() - started,
    }


def _candidate_result(candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["candidate_index"] = int(candidate["index"])
    payload["candidate_name"] = str(candidate["name"])
    payload["candidate_kind"] = str(candidate.get("kind") or "")
    params = candidate.get("params") or {}
    if isinstance(params, dict):
        payload.update(params)
    return payload


def _run_seed_shared(seed: int, active_indices: list[int] | tuple[int, ...] | None = None) -> dict[str, Any]:
    assert _SELECTORS is not None
    started = time.time()
    candidate_count = len(_CANDIDATES)
    root_candidate_indices = tuple(int(index) for index in (active_indices if active_indices is not None else range(candidate_count)))
    affected_phases = {str(item).strip().upper() for item in _CONFIG.get("affected_phases", []) if str(item).strip()}
    max_steps = int(_CONFIG["max_steps"])
    max_floor = int(_CONFIG["max_floor"])
    combat_stall_limit = int(_CONFIG.get("combat_stall_limit") or 0)
    state_copy_count = 0
    branch_nodes = 0
    shared_decisions = 0
    affected_decisions = 0
    candidate_action_evals = 0
    unique_action_steps = 0
    rejoined_nodes = 0
    map_vectorized_decisions = 0
    map_vectorized_candidate_width_sum = 0
    max_frontier = 1
    results: dict[int, dict[str, Any]] = {}
    root_env = NativeRunEnv(seed=int(seed), ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    queue: deque[dict[str, Any]] = deque(
        [
            {
                "env": root_env,
                "candidate_indices": root_candidate_indices,
                "steps": 0,
                "combat_stall_count": 0,
                "last_combat_signature": rollout_eval._combat_progress_signature(root_env),
                "error": None,
                "timeout_reason": None,
            }
        ]
    )
    pending_node_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    while queue:
        max_frontier = max(max_frontier, len(queue))
        node = queue.popleft()
        node_key = node.get("merge_key")
        if node_key is not None:
            pending_node_by_key.pop(node_key, None)
        env: NativeRunEnv = node["env"]
        candidate_indices = tuple(int(index) for index in node["candidate_indices"])
        steps = int(node["steps"])
        combat_stall_count = int(node["combat_stall_count"])
        last_combat_signature = node["last_combat_signature"]
        error = node["error"]
        timeout_reason = node["timeout_reason"]
        branch_nodes += 1

        if not candidate_indices:
            continue
        while True:
            if env.phase in TERMINAL_PHASES or int(env.floor) > max_floor or steps >= max_steps or error or timeout_reason:
                result = _result_from_env(
                    seed=seed,
                    env=env,
                    steps=steps,
                    started=started,
                    error=error,
                    timeout_reason=timeout_reason,
                )
                for index in candidate_indices:
                    results[index] = _candidate_result(_CANDIDATES[index], result)
                break

            phase = str(getattr(env, "phase", "")).upper()
            try:
                action_groups: dict[str, tuple[dict[str, Any], list[int], dict[str, Any]]] = {}
                if phase not in affected_phases or len(candidate_indices) == 1:
                    representative = candidate_indices[0]
                    _apply_candidate(representative)
                    action, _scores, _source = choose_model_required_action(env, _SELECTORS, return_scores=False)
                    signature = _action_group_signature(action)
                    action_groups[signature] = (dict(action), list(candidate_indices), {})
                    shared_decisions += 1
                    candidate_action_evals += 1
                elif phase == "MAP" and bool(_CONFIG.get("direct_map_batch", True)):
                    affected_decisions += 1
                    map_vectorized_decisions += 1
                    map_vectorized_candidate_width_sum += len(candidate_indices)
                    phase_signature_context = _phase_signature_context(phase, env)
                    action_groups, map_eval_count = _choose_map_action_groups_vectorized(
                        env,
                        candidate_indices,
                        phase_signature_context,
                    )
                    candidate_action_evals += map_eval_count
                else:
                    affected_decisions += 1
                    phase_signature_context = _phase_signature_context(phase, env)
                    candidate_signature_groups: dict[tuple[tuple[str, str | None], ...], list[int]] = {}
                    for index in candidate_indices:
                        candidate_signature_groups.setdefault(
                            _candidate_effective_phase_signature(index, phase, env, phase_signature_context),
                            [],
                        ).append(index)
                    for signature_indices in candidate_signature_groups.values():
                        index = signature_indices[0]
                        _apply_candidate(index)
                        action, _scores, _source = choose_model_required_action(env, _SELECTORS, return_scores=False)
                        signature = _action_group_signature(action)
                        if signature in action_groups:
                            action_groups[signature][1].extend(signature_indices)
                        else:
                            action_groups[signature] = (dict(action), list(signature_indices), {})
                        candidate_action_evals += 1
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                result = _result_from_env(
                    seed=seed,
                    env=env,
                    steps=steps,
                    started=started,
                    error=error_text,
                    timeout_reason=None,
                )
                result["error_traceback"] = traceback.format_exc()
                for index in candidate_indices:
                    results[index] = _candidate_result(_CANDIDATES[index], result)
                break

            if len(action_groups) == 1:
                action, _indices, metadata = next(iter(action_groups.values()))
                try:
                    _apply_action_metadata(env, metadata)
                    env.step(action)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    continue
                unique_action_steps += 1
                if combat_stall_limit > 0:
                    combat_signature = rollout_eval._combat_progress_signature(env)
                    if combat_signature is None:
                        combat_stall_count = 0
                        last_combat_signature = None
                    elif combat_signature == last_combat_signature:
                        combat_stall_count += 1
                        if combat_stall_count >= combat_stall_limit:
                            timeout_reason = (
                                "combat_stall_no_hp_progress "
                                f"after {combat_stall_count} repeated combat-progress signatures"
                            )
                    else:
                        combat_stall_count = 0
                        last_combat_signature = combat_signature
                if not timeout_reason:
                    steps += 1
                continue

            grouped_actions = list(action_groups.values())
            reuse_position = max(range(len(grouped_actions)), key=lambda pos: len(grouped_actions[pos][1]))
            branch_order = [pos for pos in range(len(grouped_actions)) if pos != reuse_position] + [reuse_position]
            for position in branch_order:
                action, indices, metadata = grouped_actions[position]
                try:
                    if position == reuse_position:
                        branch_env = env
                    else:
                        branch_env = pickle.loads(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
                        state_copy_count += 1
                    _apply_action_metadata(branch_env, metadata)
                    branch_env.step(action)
                    next_combat_stall_count = combat_stall_count
                    next_last_combat_signature = last_combat_signature
                    next_timeout_reason: str | None = None
                    if combat_stall_limit > 0:
                        combat_signature = rollout_eval._combat_progress_signature(branch_env)
                        if combat_signature is None:
                            next_combat_stall_count = 0
                            next_last_combat_signature = None
                        elif combat_signature == next_last_combat_signature:
                            next_combat_stall_count += 1
                            if next_combat_stall_count >= combat_stall_limit:
                                next_timeout_reason = (
                                    "combat_stall_no_hp_progress "
                                    f"after {next_combat_stall_count} repeated combat-progress signatures"
                                )
                        else:
                            next_combat_stall_count = 0
                            next_last_combat_signature = combat_signature
                    next_steps = steps if next_timeout_reason else steps + 1
                    new_node = {
                        "env": branch_env,
                        "candidate_indices": tuple(indices),
                        "steps": next_steps,
                        "combat_stall_count": next_combat_stall_count,
                        "last_combat_signature": next_last_combat_signature,
                        "error": None,
                        "timeout_reason": next_timeout_reason,
                    }
                    if bool(_CONFIG.get("merge_identical_states", True)):
                        merge_key = (
                            _env_digest(branch_env),
                            next_steps,
                            next_combat_stall_count,
                            next_last_combat_signature,
                            next_timeout_reason,
                        )
                        existing_node = pending_node_by_key.get(merge_key)
                        if existing_node is not None:
                            merged = tuple(sorted(set(existing_node["candidate_indices"]) | set(indices)))
                            existing_node["candidate_indices"] = merged
                            rejoined_nodes += 1
                            continue
                        new_node["merge_key"] = merge_key
                        pending_node_by_key[merge_key] = new_node
                    queue.append(new_node)
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    result = _result_from_env(
                        seed=seed,
                        env=env,
                        steps=steps,
                        started=started,
                        error=error_text,
                        timeout_reason=None,
                    )
                    result["error_traceback"] = traceback.format_exc()
                    for index in indices:
                        results[index] = _candidate_result(_CANDIDATES[index], result)
            break

    missing = [index for index in range(candidate_count) if index not in results]
    if missing:
        fallback = _result_from_env(
            seed=seed,
            env=root_env,
            steps=0,
            started=started,
            error="RuntimeError: shared_prefix_missing_result",
            timeout_reason=None,
        )
        for index in missing:
            results[index] = _candidate_result(_CANDIDATES[index], fallback)
    stats = {
        "seed": int(seed),
        "candidate_count": int(len(root_candidate_indices)),
        "seconds": time.time() - started,
        "branch_nodes": int(branch_nodes),
        "max_frontier": int(max_frontier),
        "state_copy_count": int(state_copy_count),
        "shared_decisions": int(shared_decisions),
        "affected_decisions": int(affected_decisions),
        "candidate_action_evals": int(candidate_action_evals),
        "unique_action_steps": int(unique_action_steps),
        "rejoined_nodes": int(rejoined_nodes),
        "map_vectorized_decisions": int(map_vectorized_decisions),
        "map_vectorized_candidate_width_sum": int(map_vectorized_candidate_width_sum),
        "naive_action_evals_upper": int(len(root_candidate_indices) * max(1, unique_action_steps)),
    }
    return {"seed": int(seed), "results": [results[index] for index in root_candidate_indices], "stats": stats}


def _run_seed_batch(tasks: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for task in tasks:
        if isinstance(task, (list, tuple)) and len(task) == 2:
            seed, active_indices = task
            payloads.append(_run_seed_shared(int(seed), tuple(int(index) for index in active_indices)))
        else:
            payloads.append(_run_seed_shared(int(task)))
    return payloads


def _load_existing_seed_results(path: Path, candidate_count: int) -> dict[int, list[dict[str, Any]]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        return by_seed
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = json.loads(line)
                seed = int(result["seed"])
                by_seed.setdefault(seed, []).append(result)
    except Exception:
        return {}
    return {seed: rows for seed, rows in by_seed.items() if len(rows) >= candidate_count}


def _summarize_candidate(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    floors = [int(result["floor"]) for result in rows]
    steps = [int(result["steps"]) for result in rows]
    wins = sum(1 for result in rows if result.get("won"))
    deaths = sum(1 for result in rows if result.get("dead"))
    timeouts = sum(1 for result in rows if result.get("timed_out"))
    errors = [result for result in rows if result.get("error")]
    phases = Counter(str(result["phase"]) for result in rows)
    result = {
        "candidate_index": int(candidate["index"]),
        "name": str(candidate["name"]),
        "kind": str(candidate.get("kind") or ""),
        "eliminated": bool(candidate.get("eliminated", False)),
        "elimination_reason": str(candidate.get("elimination_reason") or ""),
        "count": len(rows),
        "mean_floor": mean(floors) if floors else 0.0,
        "median_floor": median(floors) if floors else 0.0,
        "min_floor": min(floors) if floors else 0,
        "max_floor": max(floors) if floors else 0,
        "win_count": wins,
        "win_rate": wins / max(1, len(rows)),
        "death_count": deaths,
        "death_rate": deaths / max(1, len(rows)),
        "timeout_count": timeouts,
        "error_count": len(errors),
        "mean_steps": mean(steps) if steps else 0.0,
        "phase_counts": dict(phases),
        "env": candidate.get("env") or {},
    }
    params = candidate.get("params") or {}
    if isinstance(params, dict):
        result.update(params)
    return result


def _write_candidate_outputs(output_dir: Path, candidates: list[dict[str, Any]], results: list[dict[str, Any]], stats: list[dict[str, Any]]) -> None:
    by_candidate: dict[int, list[dict[str, Any]]] = {int(candidate["index"]): [] for candidate in candidates}
    for result in results:
        by_candidate[int(result["candidate_index"])].append(result)
    summaries = [
        _summarize_candidate(by_candidate[int(candidate["index"])], candidate)
        for candidate in candidates
    ]
    summaries.sort(
        key=lambda row: (
            not bool(row.get("eliminated")),
            int(row.get("count") or 0),
            float(row.get("mean_floor") or -1.0),
        ),
        reverse=True,
    )
    (output_dir / "candidate_results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames: list[str] = []
    for row in summaries:
        for key in row:
            if key not in fieldnames and key != "env":
                fieldnames.append(key)
    with (output_dir / "candidate_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})
    aggregate = {
        "candidate_count": len(candidates),
        "seed_count": len({int(result["seed"]) for result in results}),
        "result_count": len(results),
        "best": summaries[0] if summaries else None,
        "dag_stats": {
            "seconds_sum": sum(float(item.get("seconds") or 0.0) for item in stats),
            "branch_nodes": sum(int(item.get("branch_nodes") or 0) for item in stats),
            "state_copy_count": sum(int(item.get("state_copy_count") or 0) for item in stats),
            "shared_decisions": sum(int(item.get("shared_decisions") or 0) for item in stats),
            "affected_decisions": sum(int(item.get("affected_decisions") or 0) for item in stats),
            "candidate_action_evals": sum(int(item.get("candidate_action_evals") or 0) for item in stats),
            "unique_action_steps": sum(int(item.get("unique_action_steps") or 0) for item in stats),
            "rejoined_nodes": sum(int(item.get("rejoined_nodes") or 0) for item in stats),
            "map_vectorized_decisions": sum(int(item.get("map_vectorized_decisions") or 0) for item in stats),
            "map_vectorized_candidate_width_sum": sum(int(item.get("map_vectorized_candidate_width_sum") or 0) for item in stats),
            "naive_action_evals_upper": sum(int(item.get("naive_action_evals_upper") or 0) for item in stats),
            "max_frontier": max((int(item.get("max_frontier") or 0) for item in stats), default=0),
        },
    }
    evals = int(aggregate["dag_stats"]["candidate_action_evals"] or 0)
    naive = int(aggregate["dag_stats"]["naive_action_evals_upper"] or 0)
    unique_steps = int(aggregate["dag_stats"]["unique_action_steps"] or 0)
    result_steps = sum(int(result.get("steps") or 0) for result in results)
    map_calls = int(aggregate["dag_stats"].get("map_vectorized_decisions") or 0)
    aggregate["dag_stats"]["result_steps_sum"] = int(result_steps)
    aggregate["dag_stats"]["step_compression"] = (result_steps / unique_steps) if unique_steps > 0 else 0.0
    aggregate["dag_stats"]["action_eval_compression"] = (naive / evals) if evals > 0 else 0.0
    aggregate["dag_stats"]["mean_map_vectorized_width"] = (
        float(aggregate["dag_stats"]["map_vectorized_candidate_width_sum"]) / map_calls
        if map_calls > 0
        else 0.0
    )
    (output_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dag_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_floor_by_seed(results: list[dict[str, Any]]) -> dict[int, dict[int, int]]:
    by_candidate: dict[int, dict[int, int]] = {}
    for row in results:
        try:
            candidate_index = int(row["candidate_index"])
            seed = int(row["seed"])
            floor = int(row["floor"])
        except Exception:
            continue
        by_candidate.setdefault(candidate_index, {})[seed] = floor
    return by_candidate


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = sum(values) / len(values)
    return (sum((value - average) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _update_racing_eliminations(
    *,
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    active_indices: set[int],
    min_seeds: int,
    z_value: float,
    margin: float,
    keep_min: int,
) -> list[dict[str, Any]]:
    if len(active_indices) <= max(1, int(keep_min)):
        return []
    floors_by_candidate = _candidate_floor_by_seed(results)
    candidate_means: dict[int, float] = {}
    for index in active_indices:
        values = list((floors_by_candidate.get(index) or {}).values())
        if values:
            candidate_means[index] = sum(values) / len(values)
    if not candidate_means:
        return []
    best_index = max(candidate_means, key=candidate_means.__getitem__)
    best_floors = floors_by_candidate.get(best_index) or {}
    eliminated: list[dict[str, Any]] = []
    for index in sorted(active_indices):
        if index == best_index or len(active_indices) - len(eliminated) <= max(1, int(keep_min)):
            continue
        candidate_floors = floors_by_candidate.get(index) or {}
        shared_seeds = sorted(set(candidate_floors) & set(best_floors))
        if len(shared_seeds) < int(min_seeds):
            continue
        diffs = [float(candidate_floors[seed] - best_floors[seed]) for seed in shared_seeds]
        mean_diff = sum(diffs) / len(diffs)
        stderr = _sample_std(diffs) / (len(diffs) ** 0.5)
        upper_bound = mean_diff + float(z_value) * stderr
        if upper_bound < -float(margin):
            active_indices.remove(index)
            reason = (
                f"paired_ucb_vs_{candidates[best_index]['name']}={upper_bound:.3f} "
                f"mean_diff={mean_diff:.3f} n={len(shared_seeds)}"
            )
            candidates[index]["eliminated"] = True
            candidates[index]["elimination_reason"] = reason
            eliminated.append(
                {
                    "candidate_index": index,
                    "name": str(candidates[index]["name"]),
                    "best_index": best_index,
                    "best_name": str(candidates[best_index]["name"]),
                    "n": len(shared_seeds),
                    "mean_diff": mean_diff,
                    "stderr": stderr,
                    "upper_bound": upper_bound,
                    "margin": float(margin),
                    "reason": reason,
                }
            )
    return eliminated


def _device_allows_prefork(device: str | None) -> bool:
    token = str(device or "cpu").strip().lower()
    return not (token.startswith("cuda") or token.startswith("mps"))


def _cuda_available() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        return False
    if Path("/proc/driver/nvidia/gpus").exists():
        return True
    if shutil.which("nvidia-smi") is None:
        return False
    return os.system("nvidia-smi -L >/dev/null 2>&1") == 0


def _resolve_combat_device(device: str | None, combat_device: str | None) -> str:
    token = str(combat_device or "auto").strip().lower()
    if token in {"", "auto"}:
        return "cuda" if _cuda_available() else str(device or "cpu")
    return str(combat_device)


def _should_preload_selectors(args: argparse.Namespace, pending_count: int) -> bool:
    mode = str(args.preload_selectors).strip().lower()
    if mode == "never":
        return False
    if pending_count <= 0 or int(args.workers) <= 1:
        return False
    if "fork" not in mp.get_all_start_methods():
        return False
    if mode == "always":
        return True
    return _device_allows_prefork(args.device) and _device_allows_prefork(args.combat_device or args.device)


def _bounded_task_batch_size(raw_batch_size: int, *, pending_count: int, workers: int) -> int:
    if pending_count <= 1:
        return 1
    batch_size = int(raw_batch_size)
    if batch_size <= 0:
        batch_size = 1
    target_batches = max(1, min(int(pending_count), max(1, int(workers)) * 2))
    max_batch_size = max(1, (int(pending_count) + target_batches - 1) // target_batches)
    return max(1, min(batch_size, max_batch_size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sweep candidates with shared per-seed trajectory-prefix DAGs.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--affected-phases",
        required=True,
        help="Comma-separated phases whose decisions can change across candidates, e.g. MAP or SHOP,CARD_REWARD.",
    )
    parser.add_argument(
        "--phase-env-keys-json",
        default="",
        help="Optional JSON object mapping phase names to env keys that affect that phase. Default uses conservative built-in phase groups.",
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--combat-stall-limit", type=int, default=int(os.environ.get("SPIRECOMM_COMBAT_STALL_LIMIT", "250")))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task-batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--preload-selectors", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-interval", type=int, default=10)
    parser.add_argument("--result-flush-interval", type=int, default=8)
    parser.add_argument(
        "--racing",
        action="store_true",
        help="Enable conservative paired anytime racing: clearly losing candidates stop receiving future seeds.",
    )
    parser.add_argument("--racing-min-seeds", type=int, default=60)
    parser.add_argument("--racing-wave-size", type=int, default=20)
    parser.add_argument("--racing-z", type=float, default=2.5)
    parser.add_argument("--racing-margin", type=float, default=0.0)
    parser.add_argument("--racing-keep-min", type=int, default=4)
    parser.add_argument(
        "--merge-identical-states",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SPIRECOMM_SHARED_PREFIX_MERGE_IDENTICAL_STATES", True),
        help="Exact rejoin: merge queued DAG branches whose pickled env state and counters are identical.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"))
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_scorer.pt"))
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument(
        "--shop-choice-model",
        type=Path,
        default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")),
    )
    parser.add_argument("--shop-policy", choices=["model", "value"], default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"))
    parser.add_argument("--shop-value-price-cost", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")))
    parser.add_argument(
        "--shop-value-reserve-shortfall-cost",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")),
    )
    parser.add_argument("--shop-value-future-shop-reserve", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "120")))
    parser.add_argument("--shop-value-future-shop-horizon", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "5")))
    parser.add_argument("--shop-value-card-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "4.6262945279949435")))
    parser.add_argument("--shop-value-card-reference-price", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "60.0")))
    parser.add_argument("--shop-value-card-price-factor-min", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "0.65")))
    parser.add_argument("--shop-value-card-price-factor-max", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "1.35")))
    parser.add_argument("--shop-value-potion-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "0.5084989138155764")))
    parser.add_argument("--shop-value-relic-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "0.8")))
    parser.add_argument("--shop-value-item-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "1.0")))
    parser.add_argument("--shop-value-threshold", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_THRESHOLD", "0.0")))
    parser.add_argument("--shop-prior-weight-override", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "0.8")))
    parser.add_argument("--v3-normal-room-potion-penalty", type=float, default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "1.5")))
    args = parser.parse_args()

    candidates = _load_candidates(args.candidate_json)
    args.combat_device = _resolve_combat_device(args.device, args.combat_device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    else:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))
    affected_phases = [phase.strip().upper() for phase in str(args.affected_phases).split(",") if phase.strip()]
    if not affected_phases:
        raise SystemExit("--affected-phases cannot be empty")
    phase_env_keys: dict[str, list[str]] = {}
    if str(args.phase_env_keys_json or "").strip():
        try:
            parsed_phase_env_keys = json.loads(str(args.phase_env_keys_json))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --phase-env-keys-json: {exc}") from exc
        if not isinstance(parsed_phase_env_keys, dict):
            raise SystemExit("--phase-env-keys-json must be a JSON object")
        phase_env_keys = {
            str(phase).strip().upper(): [str(key) for key in keys]
            for phase, keys in parsed_phase_env_keys.items()
            if isinstance(keys, list)
        }
    config = {
        "repo_root": str(args.repo_root.resolve()),
        "output_dir": str(output_dir),
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "combat_stall_limit": int(args.combat_stall_limit),
        "torch_threads": int(args.torch_threads),
        "device": str(args.device),
        "combat_device": str(args.combat_device),
        "combat_selector": str(args.combat_selector),
        "combat_model": str(args.combat_model),
        "v3_combat_model": str(args.v3_combat_model),
        "card_reward_model": str(args.card_reward_model),
        "shop_choice_model": str(args.shop_choice_model),
        "shop_policy": str(args.shop_policy),
        "shop_value_price_cost": float(args.shop_value_price_cost),
        "shop_value_reserve_shortfall_cost": float(args.shop_value_reserve_shortfall_cost),
        "shop_value_future_shop_reserve": int(args.shop_value_future_shop_reserve),
        "shop_value_future_shop_horizon": int(args.shop_value_future_shop_horizon),
        "shop_value_card_scale": float(args.shop_value_card_scale),
        "shop_value_card_reference_price": float(args.shop_value_card_reference_price),
        "shop_value_card_price_factor_min": float(args.shop_value_card_price_factor_min),
        "shop_value_card_price_factor_max": float(args.shop_value_card_price_factor_max),
        "shop_value_potion_scale": float(args.shop_value_potion_scale),
        "shop_value_relic_scale": float(args.shop_value_relic_scale),
        "shop_value_item_scale": float(args.shop_value_item_scale),
        "shop_value_threshold": float(args.shop_value_threshold),
        "shop_prior_weight_override": None if args.shop_prior_weight_override is None else float(args.shop_prior_weight_override),
        "v3_normal_room_potion_penalty": max(0.0, float(args.v3_normal_room_potion_penalty)),
        "affected_phases": affected_phases,
        "phase_env_keys": phase_env_keys,
        "merge_identical_states": bool(args.merge_identical_states),
        "direct_map_batch": _env_bool("SPIRECOMM_SHARED_PREFIX_DIRECT_MAP_BATCH", True),
    }
    existing = _load_existing_seed_results(output_dir / "seed_results.jsonl", len(candidates)) if args.resume else {}
    results: list[dict[str, Any]] = [row for seed in sorted(existing) for row in existing[seed]]
    pending_seeds = [seed for seed in seeds if seed not in existing]
    config["selectors_preloaded"] = _should_preload_selectors(args, len(pending_seeds))
    (output_dir / "config.json").write_text(
        json.dumps({**config, "seeds": seeds, "candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results_path = output_dir / "seed_results.jsonl"
    stats_path = output_dir / "seed_dag_stats.jsonl"
    if not args.resume:
        for path in (results_path, stats_path):
            if path.exists():
                path.unlink()
    print(
        f"shared-prefix sweep seeds={len(seeds)} pending={len(pending_seeds)} candidates={len(candidates)} "
        f"affected={','.join(affected_phases)} workers={int(args.workers)}",
        flush=True,
    )
    if not pending_seeds:
        stats = []
        if stats_path.exists():
            stats = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        _write_candidate_outputs(output_dir, candidates, results, stats)
        return
    if bool(config["selectors_preloaded"]):
        _init_worker(config, candidates)
    mp_context = mp.get_context("fork") if bool(config["selectors_preloaded"]) else None
    started = time.time()
    completed = 0
    stats_rows: list[dict[str, Any]] = []
    result_buffer: list[dict[str, Any]] = []
    stats_buffer: list[dict[str, Any]] = []
    task_batch_size = _bounded_task_batch_size(int(args.task_batch_size), pending_count=len(pending_seeds), workers=int(args.workers))
    elimination_log_path = output_dir / "racing_eliminations.jsonl"
    if not args.resume and elimination_log_path.exists():
        elimination_log_path.unlink()

    def handle_payloads(batch_payloads: list[dict[str, Any]]) -> None:
        nonlocal completed
        for payload in batch_payloads:
            seed_results = list(payload["results"])
            seed_stats = dict(payload["stats"])
            results.extend(seed_results)
            stats_rows.append(seed_stats)
            result_buffer.extend(seed_results)
            stats_buffer.append(seed_stats)
            completed += 1

    def flush_buffers() -> None:
        if result_buffer:
            _append_jsonl(results_path, result_buffer)
            result_buffer.clear()
        if stats_buffer:
            _append_jsonl(stats_path, stats_buffer)
            stats_buffer.clear()

    def print_progress(target_total: int, active_count: int) -> None:
        _write_candidate_outputs(output_dir, candidates, results, stats_rows)
        best = json.loads((output_dir / "candidate_results.json").read_text(encoding="utf-8"))[0]
        dag_stats = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["dag_stats"]
        rate = completed / max(1e-9, time.time() - started)
        print(
            f"completed {completed}/{target_total} seed_rate={rate:.3f}/s active={active_count} "
            f"best={best['name']} mean={float(best['mean_floor']):.3f} "
            f"compression={float(dag_stats.get('action_eval_compression') or 0.0):.1f}x",
            flush=True,
        )

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_init_worker,
        initargs=(config, candidates),
        mp_context=mp_context,
    ) as executor:
        if bool(args.racing):
            active_indices: set[int] = set(range(len(candidates)))
            pending_queue = list(pending_seeds)
            while pending_queue and active_indices:
                wave_size = max(1, int(args.racing_wave_size))
                wave = pending_queue[:wave_size]
                pending_queue = pending_queue[wave_size:]
                tasks = [(seed, tuple(sorted(active_indices))) for seed in wave]
                batches = [tasks[index : index + task_batch_size] for index in range(0, len(tasks), task_batch_size)]
                futures = [executor.submit(_run_seed_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    handle_payloads(future.result())
                    if len(result_buffer) >= max(1, int(args.result_flush_interval)) * max(1, len(active_indices)):
                        flush_buffers()
                flush_buffers()
                eliminated = _update_racing_eliminations(
                    candidates=candidates,
                    results=results,
                    active_indices=active_indices,
                    min_seeds=int(args.racing_min_seeds),
                    z_value=float(args.racing_z),
                    margin=float(args.racing_margin),
                    keep_min=int(args.racing_keep_min),
                )
                if eliminated:
                    _append_jsonl(elimination_log_path, eliminated)
                    print(
                        "racing eliminated "
                        + ", ".join(f"{item['name']}({item['reason']})" for item in eliminated[:8]),
                        flush=True,
                    )
                if int(args.summary_interval) > 0:
                    print_progress(len(pending_seeds), len(active_indices))
        else:
            batches = [pending_seeds[index : index + task_batch_size] for index in range(0, len(pending_seeds), task_batch_size)]
            futures = [executor.submit(_run_seed_batch, batch) for batch in batches]
            for future in as_completed(futures):
                handle_payloads(future.result())
                if len(result_buffer) >= max(1, int(args.result_flush_interval)) * len(candidates):
                    flush_buffers()
                if int(args.summary_interval) > 0 and completed % int(args.summary_interval) == 0:
                    print_progress(len(pending_seeds), len(candidates))
    flush_buffers()
    _write_candidate_outputs(output_dir, candidates, results, stats_rows)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
