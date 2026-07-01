#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import gc
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from spirecomm.ai.runtime_decision import build_runtime_selectors, choose_modeled_action
from spirecomm.ai.v3_combat_dataset import V3CombatLabeledRoot, load_shard, make_root_sample, save_shard
from spirecomm.ai.v3_combat_features import FEATURE_SCHEMA_VERSION, action_key, root_combat_actions
from spirecomm.ai.v3_combat_teacher import (
    TEACHER_VERSION,
    TeacherConfig,
    label_root_sample,
    teacher_config_from_json_path,
    teacher_config_from_mapping,
)
from spirecomm.native_sim_v3 import NativeCombatEnv, NativeRunEnv
from spirecomm.native_sim_v3.combat.engine import SUPPORTED_COMBAT_POTION_IDS
from spirecomm.native_sim_v3.content.cards import make_card
from spirecomm.native_sim_v3.content.potions import make_potion
from spirecomm.native_sim_v3.content.relics import make_relic


ShardWriteResult = tuple[Path, int]
LabelBatchResult = tuple[list[ShardWriteResult], list[Future[ShardWriteResult]]]


def _disable_gc_for_hot_worker_if_enabled() -> None:
    if str(os.environ.get("SPIRECOMM_FAST_DISABLE_GC", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    gc.disable()


def _git_status(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"git_status_unavailable:{exc}"


def _state_from_env(env: Any) -> dict[str, Any]:
    state_method = getattr(env, "state", None)
    if callable(state_method):
        return state_method()
    return env.serialize()


def _install_combat_state(
    env: NativeCombatEnv,
    *,
    hand: list[str],
    draw: list[str] | None = None,
    discard: list[str] | None = None,
    exhaust: list[str] | None = None,
    relics: list[str] | None = None,
    potions: list[str] | None = None,
    room_type: str | None = None,
    energy: int = 3,
    hp: int = 60,
    max_hp: int | None = None,
    block: int = 0,
) -> NativeCombatEnv:
    max_hp = int(max_hp if max_hp is not None else max(hp, env.player.max_hp))
    env.relics = [make_relic(relic_id) for relic_id in list(relics or [])]
    env.engine.relics = list(env.relics)
    if potions is not None:
        env.potions = [make_potion(potion_id) for potion_id in potions]
        env.engine.potions = list(env.potions)
    if room_type is not None:
        env.room_type = str(room_type)
        env.engine.state.room_type = str(room_type)
    env.player.current_hp = hp
    env.player.max_hp = max_hp
    env.player.block = block
    env.player.energy = energy
    env.engine.player.current_hp = hp
    env.engine.player.max_hp = max_hp
    env.engine.player.block = block
    env.engine.player.energy = energy
    env.state.player.current_hp = hp
    env.state.player.max_hp = max_hp
    env.state.player.block = block
    env.state.player.energy = energy
    env.state.hand = [make_card(card_id) for card_id in hand]
    env.state.draw_pile = [make_card(card_id) for card_id in list(draw or [])]
    env.state.discard_pile = [make_card(card_id) for card_id in list(discard or [])]
    env.state.exhaust_pile = [make_card(card_id) for card_id in list(exhaust or [])]
    env.master_deck = [make_card(card_id) for card_id in hand + list(draw or []) + list(discard or []) + list(exhaust or [])]
    env.engine.master_deck = list(env.master_deck)
    env.engine._refresh_all_card_flags()
    return env


def _set_monster_pressure(env: NativeCombatEnv, *, damage: int = 0, hits: int = 1, all_monsters: bool = False) -> NativeCombatEnv:
    for index, monster in enumerate(env.state.monsters):
        if monster.current_hp <= 0:
            continue
        if index == 0 or all_monsters:
            monster.intent = "ATTACK" if damage > 0 else "UNKNOWN"
            monster.move_adjusted_damage = int(damage)
            monster.move_hits = max(1, int(hits))
    return env


def curated_envs() -> list[tuple[str, NativeCombatEnv]]:
    cases: list[tuple[str, NativeCombatEnv]] = []
    base_kwargs = {"ascension_level": 0, "encounter_name": "Cultist", "master_deck": [make_card("Strike_R") for _ in range(5)] + [make_card("Defend_R") for _ in range(4)] + [make_card("Bash")]}
    cases.append(
        (
            "curated_rage_multi_attack",
            _install_combat_state(
                NativeCombatEnv(seed=1001, **base_kwargs),
                hand=["Rage", "Strike_R", "Strike_R", "Defend_R", "Bash"],
                draw=["Strike_R", "Defend_R", "Pommel Strike"],
            ),
        )
    )
    cases.append(
        (
            "curated_inflame_flex_attacks",
            _install_combat_state(
                NativeCombatEnv(seed=1002, **base_kwargs),
                hand=["Inflame", "Flex", "Strike_R", "Strike_R", "Twin Strike"],
                draw=["Defend_R", "Strike_R"],
            ),
        )
    )
    cases.append(
        (
            "curated_corruption_skills",
            _install_combat_state(
                NativeCombatEnv(seed=1003, **base_kwargs),
                hand=["Corruption", "Defend_R", "Shrug It Off", "True Grit", "Strike_R"],
                draw=["Defend_R", "Armaments", "Pommel Strike"],
            ),
        )
    )
    cases.append(
        (
            "curated_offering_followup",
            _install_combat_state(
                NativeCombatEnv(seed=1004, **base_kwargs),
                hand=["Offering", "Bash", "Strike_R", "Strike_R", "Defend_R"],
                draw=["Inflame", "Pommel Strike", "Twin Strike", "Strike_R"],
            ),
        )
    )
    cases.append(
        (
            "curated_exhume_exhaust",
            _install_combat_state(
                NativeCombatEnv(seed=1005, **base_kwargs),
                hand=["Exhume", "Strike_R", "Defend_R", "Bash"],
                draw=["Strike_R", "Shrug It Off"],
                exhaust=["Offering"],
            ),
        )
    )
    cases.append(
        (
            "curated_fiend_fire_dead_branch",
            _install_combat_state(
                NativeCombatEnv(seed=1006, **base_kwargs),
                hand=["Fiend Fire", "Strike_R", "Defend_R", "Shrug It Off", "True Grit"],
                draw=["Strike_R", "Defend_R"],
                relics=["Dead Branch"],
            ),
        )
    )
    return cases


def collect_curated_roots(limit: int | None = None) -> list[Any]:
    roots = []
    for root_id, env in curated_envs():
        root = make_root_sample(env, root_id=root_id, source="curated")
        if root is not None:
            roots.append(root)
        if limit is not None and len(roots) >= limit:
            break
    return roots


POTION_CURATED_ROOM_TYPES = ("MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss")


POTION_CURATED_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "balanced",
        "encounter": "Cultist",
        "hand": ["Strike_R", "Strike_R", "Defend_R", "Bash", "Pommel Strike"],
        "draw": ["Strike_R", "Defend_R", "Inflame"],
        "hp": 68,
        "energy": 3,
        "incoming": 6,
    },
    {
        "name": "low_hp_high_incoming",
        "encounter": "Cultist",
        "hand": ["Strike_R", "Defend_R", "Defend_R", "Shrug It Off", "Bash"],
        "draw": ["Strike_R", "True Grit", "Pommel Strike"],
        "hp": 18,
        "energy": 3,
        "incoming": 16,
    },
    {
        "name": "multi_attack_incoming",
        "encounter": "Cultist",
        "hand": ["Rage", "Strike_R", "Twin Strike", "Defend_R", "Bash"],
        "draw": ["Strike_R", "Pommel Strike", "Defend_R"],
        "hp": 52,
        "energy": 3,
        "incoming": 5,
        "hits": 3,
    },
    {
        "name": "multi_enemy",
        "encounter": "3 Louse",
        "hand": ["Cleave", "Strike_R", "Defend_R", "Bash", "Pommel Strike"],
        "draw": ["Strike_R", "Defend_R", "Inflame"],
        "hp": 58,
        "energy": 3,
        "incoming": 7,
        "all_monsters": True,
    },
    {
        "name": "draw_followup",
        "encounter": "Cultist",
        "hand": ["Battle Trance", "Strike_R", "Defend_R", "Bash"],
        "draw": ["Carnage", "Pommel Strike", "Twin Strike", "Inflame", "Shrug It Off"],
        "hp": 61,
        "energy": 3,
        "incoming": 6,
    },
    {
        "name": "power_followup",
        "encounter": "Cultist",
        "hand": ["Inflame", "Demon Form", "Strike_R", "Defend_R", "Bash"],
        "draw": ["Strike_R", "Twin Strike", "Pommel Strike"],
        "hp": 60,
        "energy": 3,
        "incoming": 6,
    },
    {
        "name": "energy_starved",
        "encounter": "Cultist",
        "hand": ["Carnage", "Bash", "Strike_R", "Defend_R", "Pommel Strike"],
        "draw": ["Twin Strike", "Strike_R", "Defend_R"],
        "hp": 54,
        "energy": 2,
        "incoming": 6,
    },
    {
        "name": "block_need",
        "encounter": "Cultist",
        "hand": ["Defend_R", "Defend_R", "Shrug It Off", "True Grit", "Strike_R"],
        "draw": ["Bash", "Pommel Strike", "Strike_R"],
        "hp": 35,
        "energy": 3,
        "incoming": 22,
    },
    {
        "name": "low_enemy_hp",
        "encounter": "Cultist",
        "hand": ["Strike_R", "Strike_R", "Bash", "Defend_R", "Pommel Strike"],
        "draw": ["Strike_R", "Defend_R"],
        "hp": 46,
        "energy": 3,
        "incoming": 6,
        "monster_hp": 12,
    },
    {
        "name": "many_cards",
        "encounter": "Cultist",
        "hand": ["Offering", "Strike_R", "Strike_R", "Defend_R", "Bash", "Pommel Strike"],
        "draw": ["Carnage", "Twin Strike", "Inflame", "Shrug It Off", "True Grit"],
        "hp": 42,
        "energy": 3,
        "incoming": 10,
    },
    {
        "name": "exhaust_memory",
        "encounter": "Cultist",
        "hand": ["Exhume", "Strike_R", "Defend_R", "Bash"],
        "draw": ["Pommel Strike", "Defend_R"],
        "exhaust": ["Offering", "Carnage"],
        "hp": 48,
        "energy": 3,
        "incoming": 6,
    },
    {
        "name": "boss_like",
        "encounter": "Lagavulin",
        "hand": ["Demon Form", "Inflame", "Strike_R", "Defend_R", "Bash"],
        "draw": ["Carnage", "Twin Strike", "Shrug It Off"],
        "hp": 70,
        "energy": 3,
        "incoming": 18,
    },
)


def curated_potion_envs() -> list[tuple[str, NativeCombatEnv]]:
    cases: list[tuple[str, NativeCombatEnv]] = []
    deck = [make_card("Strike_R") for _ in range(5)] + [make_card("Defend_R") for _ in range(4)] + [make_card("Bash")]
    potion_ids = sorted(SUPPORTED_COMBAT_POTION_IDS)
    seed = 200000
    for potion_id in potion_ids:
        for room_type in POTION_CURATED_ROOM_TYPES:
            for scenario in POTION_CURATED_SCENARIOS:
                seed += 1
                env = NativeCombatEnv(
                    seed=seed,
                    ascension_level=0,
                    encounter_name=str(scenario["encounter"]),
                    room_type=room_type,
                    master_deck=list(deck),
                    potions=[make_potion(potion_id)],
                )
                _install_combat_state(
                    env,
                    hand=list(scenario["hand"]),
                    draw=list(scenario.get("draw") or []),
                    discard=list(scenario.get("discard") or []),
                    exhaust=list(scenario.get("exhaust") or []),
                    potions=[potion_id],
                    room_type=room_type,
                    energy=int(scenario.get("energy", 3)),
                    hp=int(scenario.get("hp", 60)),
                    max_hp=80,
                    block=int(scenario.get("block", 0)),
                )
                if scenario.get("monster_hp") is not None and env.state.monsters:
                    env.state.monsters[0].current_hp = max(1, int(scenario["monster_hp"]))
                _set_monster_pressure(
                    env,
                    damage=int(scenario.get("incoming", 0)),
                    hits=int(scenario.get("hits", 1)),
                    all_monsters=bool(scenario.get("all_monsters", False)),
                )
                cases.append((f"curated_potion:{potion_id}:{room_type}:{scenario['name']}", env))
    return cases


def collect_curated_potion_roots(limit: int | None = None) -> list[Any]:
    roots: list[Any] = []
    for root_id, env in curated_potion_envs():
        root = make_root_sample(
            env,
            root_id=root_id,
            source="curated_potion",
            metadata={
                "potion_id": str(env.potions[0].get("potion_id") if env.potions else ""),
                "room_type": str(env.serialize().get("room_type") or ""),
            },
        )
        if root is not None:
            roots.append(root)
        if limit is not None and len(roots) >= limit:
            break
    return roots


def _trace_paths(pattern: str, max_files: int | None) -> list[Path]:
    paths = sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern.lstrip("/")))
    if max_files is not None:
        return paths[:max_files]
    return paths


def _resolve_trace_action(env: Any, action: dict[str, Any]) -> dict[str, Any] | None:
    state = _state_from_env(env)
    legal = [dict(candidate) for candidate in env.legal_actions()]
    legal_keys = {action_key(candidate, state): candidate for candidate in legal}
    chosen_key = action_key(action, state)
    if chosen_key in legal_keys:
        return legal_keys[chosen_key]

    kind = str(action.get("kind") or "")
    if kind == "end":
        return next((candidate for candidate in legal if candidate.get("kind") == "end"), None)

    if kind == "card":
        wanted_card_id = str(action.get("card_id") or "")
        wanted_name = str(action.get("name") or "")
        wanted_target = action.get("target_index")
        matches = []
        for candidate in legal:
            if candidate.get("kind") != "card":
                continue
            candidate_card_id = str(candidate.get("card_id") or "")
            candidate_name = str(candidate.get("name") or "")
            if wanted_card_id and candidate_card_id != wanted_card_id:
                continue
            if not wanted_card_id and wanted_name and candidate_name != wanted_name:
                continue
            if wanted_target is not None and candidate.get("target_index") != wanted_target:
                continue
            matches.append(candidate)
        if matches:
            return matches[0]

    wanted_choice = action.get("choice_index")
    wanted_card_id = str(action.get("card_id") or "")
    wanted_name = str(action.get("name") or "")
    matches = []
    for candidate in legal:
        if kind and str(candidate.get("kind") or "") != kind:
            continue
        if wanted_choice is not None and candidate.get("choice_index") != wanted_choice:
            continue
        if wanted_card_id and str(candidate.get("card_id") or "") != wanted_card_id:
            continue
        if not wanted_card_id and wanted_name and str(candidate.get("name") or "") != wanted_name:
            continue
        matches.append(candidate)
    if matches:
        return matches[0]
    return None


def collect_trace_roots(trace_glob: str, *, max_roots: int, max_files: int | None = None) -> list[Any]:
    roots = []
    for trace_path in _trace_paths(trace_glob, max_files):
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        seed = payload.get("seed_long")
        if seed is None:
            seed = payload.get("seed")
        if seed is None:
            continue
        env = NativeRunEnv(seed=int(seed), ascension_level=int(payload.get("ascension") or 0), enable_neow=True)
        for step in payload.get("steps") or []:
            action = dict(step.get("action") or {})
            resolved_action = _resolve_trace_action(env, action)
            if resolved_action is None:
                break
            if getattr(env, "phase", "") == "COMBAT":
                root = make_root_sample(
                    env,
                    root_id=f"trace:{trace_path.name}:{step.get('step')}",
                    source="strict_trace",
                    chosen_action=resolved_action,
                    metadata={"trace_path": str(trace_path), "trace_step": step.get("step")},
                )
                if root is not None:
                    roots.append(root)
                    if len(roots) >= max_roots:
                        return roots
            try:
                env.step(resolved_action)
            except Exception:
                break
    return roots


def _mem_available_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _rss_kb() -> int | None:
    try:
        statm = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
        for line in statm.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def _memory_snapshot(**extra: Any) -> dict[str, Any]:
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "rss_kb": _rss_kb(),
        "mem_available_kb": _mem_available_kb(),
    }
    payload.update(extra)
    return payload


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _labeled_roots_from_payload(payload: dict[str, Any]) -> list[V3CombatLabeledRoot]:
    roots: list[V3CombatLabeledRoot] = []
    for labeled_root in payload.get("roots") or []:
        if getattr(labeled_root, "root", None) is not None:
            roots.append(labeled_root)
    return roots


def _root_id(root: Any) -> str | None:
    value = getattr(root, "root_id", None)
    return str(value) if value is not None else None


def _root_room_type(root: Any) -> str:
    visible = getattr(root, "visible_before", None) or {}
    if isinstance(visible, dict):
        return str(visible.get("room_type") or "")
    return ""


def root_stats(roots: list[Any]) -> dict[str, Any]:
    potion_candidate_roots = 0
    potion_candidate_count_by_id: Counter[str] = Counter()
    room_type_counts: Counter[str] = Counter()
    potion_candidate_by_room_type: dict[str, Counter[str]] = {}
    potion_candidate_roots_by_room_type: Counter[str] = Counter()
    for root in roots:
        room_type = _root_room_type(root) or "UNKNOWN"
        room_type_counts[room_type] += 1
        actions = list(getattr(root, "actions", []) or [])
        potion_actions = [action for action in actions if str(action.get("kind") or "") == "potion"]
        if not potion_actions:
            continue
        potion_candidate_roots += 1
        potion_candidate_roots_by_room_type[room_type] += 1
        room_counter = potion_candidate_by_room_type.setdefault(room_type, Counter())
        for action in potion_actions:
            potion_id = str(action.get("potion_id") or action.get("name") or "UNKNOWN")
            potion_candidate_count_by_id[potion_id] += 1
            room_counter[potion_id] += 1
    return {
        "root_count": len(roots),
        "potion_candidate_roots": potion_candidate_roots,
        "potion_candidate_count_by_id": dict(sorted(potion_candidate_count_by_id.items())),
        "room_type_counts": dict(sorted(room_type_counts.items())),
        "potion_candidate_roots_by_room_type": dict(sorted(potion_candidate_roots_by_room_type.items())),
        "potion_candidate_by_room_type": {
            room_type: dict(sorted(counter.items()))
            for room_type, counter in sorted(potion_candidate_by_room_type.items())
        },
    }


def _merge_root_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["root_count"] = int(target.get("root_count", 0)) + int(source.get("root_count", 0))
    target["potion_candidate_roots"] = int(target.get("potion_candidate_roots", 0)) + int(source.get("potion_candidate_roots", 0))
    for key in ("potion_candidate_count_by_id", "room_type_counts", "potion_candidate_roots_by_room_type"):
        merged = Counter(target.get(key) or {})
        merged.update(source.get(key) or {})
        target[key] = dict(sorted(merged.items()))
    nested = {
        room_type: Counter(counter)
        for room_type, counter in (target.get("potion_candidate_by_room_type") or {}).items()
    }
    for room_type, counter in (source.get("potion_candidate_by_room_type") or {}).items():
        nested.setdefault(room_type, Counter()).update(counter)
    target["potion_candidate_by_room_type"] = {
        room_type: dict(sorted(counter.items()))
        for room_type, counter in sorted(nested.items())
    }


def _load_existing_shard_state(shards: list[Path]) -> tuple[set[str], int, dict[str, Any]]:
    root_ids: set[str] = set()
    roots: list[Any] = []
    for shard_path in shards:
        payload = load_shard(shard_path)
        for labeled_root in _labeled_roots_from_payload(payload):
            root = labeled_root.root
            root_id = _root_id(root)
            if root_id is not None:
                root_ids.add(root_id)
            roots.append(root)
        del payload
    stats = root_stats(roots)
    count = len(roots)
    del roots
    gc.collect()
    return root_ids, count, stats


def _exploratory_seed_from_root_id(root_id: str | None) -> int | None:
    if not root_id:
        return None
    parts = str(root_id).split(":")
    if len(parts) < 3 or parts[0] != "explore":
        return None
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return None


def collect_exploratory_roots(
    *,
    seeds: list[int],
    max_roots: int,
    repo_root: Path,
    device: str,
    random_action_rate: float,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    combat_model: Path | None = None,
    per_seed_root_cap: int | None = None,
    max_steps_per_seed: int = 1000,
) -> list[Any]:
    selectors = build_runtime_selectors(
        repo_root=repo_root,
        device=device,
        combat_model=combat_model,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
    )
    roots = []
    rng = random.Random(7)
    for seed in seeds:
        env = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)
        steps = 0
        seed_roots = 0
        while env.phase not in {"GAME_OVER", "COMPLETE", "VICTORY"} and steps < max_steps_per_seed:
            if env.phase == "COMBAT":
                root = make_root_sample(env, root_id=f"explore:{seed}:{steps}", source="exploratory")
                if root is not None and (per_seed_root_cap is None or seed_roots < per_seed_root_cap):
                    roots.append(root)
                    seed_roots += 1
                    if len(roots) >= max_roots:
                        return roots
            actions = env.legal_actions()
            if not actions:
                break
            if rng.random() < random_action_rate:
                action = rng.choice(actions)
            else:
                action, _, _ = choose_modeled_action(env, selectors)
            env.step(action)
            steps += 1
    return roots


def _collect_exploratory_seed_worker(payload: dict[str, Any]) -> tuple[int, list[Any], dict[str, Any]]:
    _disable_gc_for_hot_worker_if_enabled()
    seed = int(payload["seed"])
    repo_root = Path(payload["repo_root"])
    random_action_rate = float(payload["random_action_rate"])
    per_seed_root_cap = payload.get("per_seed_root_cap")
    max_steps_per_seed = int(payload["max_steps_per_seed"])
    selectors = build_runtime_selectors(
        repo_root=repo_root,
        device=str(payload["device"]),
        combat_model=Path(payload["combat_model"]) if payload.get("combat_model") else None,
        combat_selector=payload.get("combat_selector"),
        v3_combat_model=Path(payload["v3_combat_model"]) if payload.get("v3_combat_model") else None,
    )
    roots: list[Any] = []
    rng = random.Random(7 + seed)
    env = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)
    steps = 0
    while env.phase not in {"GAME_OVER", "COMPLETE", "VICTORY"} and steps < max_steps_per_seed:
        if env.phase == "COMBAT" and (per_seed_root_cap is None or len(roots) < int(per_seed_root_cap)):
            root = make_root_sample(env, root_id=f"explore:{seed}:{steps}", source="exploratory")
            if root is not None:
                roots.append(root)
        actions = env.legal_actions()
        if not actions:
            break
        if rng.random() < random_action_rate:
            action = rng.choice(actions)
        else:
            action, _, _ = choose_modeled_action(env, selectors)
        try:
            env.step(action)
        except Exception as exc:
            return seed, roots, {
                "seed": seed,
                "steps": steps,
                "floor": int(getattr(env, "floor", 0)),
                "phase": str(getattr(env, "phase", "")),
                "error": str(exc),
            }
        steps += 1
    return seed, roots, {
        "seed": seed,
        "steps": steps,
        "floor": int(getattr(env, "floor", 0)),
        "phase": str(getattr(env, "phase", "")),
        "roots": len(roots),
    }


def collect_exploratory_roots_parallel(
    *,
    seeds: list[int],
    max_roots: int,
    repo_root: Path,
    device: str,
    random_action_rate: float,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    combat_model: Path | None = None,
    per_seed_root_cap: int | None = None,
    max_steps_per_seed: int = 1000,
    workers: int = 1,
) -> list[Any]:
    payloads = [
        {
            "seed": int(seed),
            "repo_root": str(repo_root),
            "device": device,
            "random_action_rate": float(random_action_rate),
            "combat_selector": combat_selector,
            "v3_combat_model": str(v3_combat_model) if v3_combat_model is not None else "",
            "combat_model": str(combat_model) if combat_model is not None else "",
            "per_seed_root_cap": per_seed_root_cap,
            "max_steps_per_seed": int(max_steps_per_seed),
        }
        for seed in seeds
    ]
    results: dict[int, tuple[list[Any], dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(_collect_exploratory_seed_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            seed, seed_roots, stats = future.result()
            results[int(seed)] = (seed_roots, stats)
            print(
                "collected "
                f"seed={seed} roots={len(seed_roots)} floor={stats.get('floor')} "
                f"phase={stats.get('phase')} done={len(results)}/{len(payloads)}",
                flush=True,
            )
    roots: list[Any] = []
    for seed in seeds:
        seed_roots, _stats = results.get(int(seed), ([], {}))
        roots.extend(seed_roots)
        if len(roots) >= max_roots:
            return roots[:max_roots]
    return roots[:max_roots]


def _save_labeled_shard(
    *,
    roots: list[Any],
    output_dir: Path,
    shard_index: int,
    config: TeacherConfig,
    workers: int,
    repo_root: Path,
    metadata: dict[str, Any],
    memory_log: Path | None,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[Path, int]:
    _append_jsonl(memory_log, _memory_snapshot(event="label_shard_start", shard_index=shard_index, roots=len(roots)))
    labeled = label_roots(roots, config=config, workers=max(1, int(workers)), executor=executor)
    shard_path = output_dir / f"shard_{shard_index:05d}.pt"
    git_status = metadata.get("git_status")
    if git_status is None:
        git_status = _git_status(repo_root)
    shard_metadata = {
        **metadata,
        "root_count": len(labeled),
        "shard_index": shard_index,
        "git_status": git_status,
        "teacher_version": TEACHER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "teacher_config": config.__dict__,
        "root_stats": root_stats(roots),
    }
    result = _write_labeled_shard(
        shard_path=shard_path,
        labeled_roots=labeled,
        metadata=shard_metadata,
        memory_log=memory_log,
        shard_index=shard_index,
    )
    del labeled
    gc.collect()
    return result


def _write_labeled_shard(
    *,
    shard_path: Path,
    labeled_roots: list[V3CombatLabeledRoot],
    metadata: dict[str, Any],
    memory_log: Path | None,
    shard_index: int,
) -> tuple[Path, int]:
    started = time.time()
    _append_jsonl(
        memory_log,
        _memory_snapshot(event="shard_write_start", shard_index=shard_index, roots=len(labeled_roots), path=str(shard_path)),
    )
    tmp_path = shard_path.with_name(f".{shard_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    save_shard(
        tmp_path,
        labeled_roots,
        metadata=metadata,
    )
    os.replace(tmp_path, shard_path)
    count = len(labeled_roots)
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="label_shard_done",
            shard_index=shard_index,
            roots=count,
            path=str(shard_path),
            write_seconds=round(time.time() - started, 3),
        ),
    )
    return shard_path, count


def _save_labeled_batch(
    *,
    shard_roots: list[list[Any]],
    output_dir: Path,
    shard_index: int,
    config: TeacherConfig,
    workers: int,
    repo_root: Path,
    metadata: dict[str, Any],
    memory_log: Path | None,
    executor: ProcessPoolExecutor | None = None,
    writer_executor: ThreadPoolExecutor | None = None,
) -> LabelBatchResult:
    if not shard_roots:
        return [], []
    flat_roots = [root for roots in shard_roots for root in roots]
    lengths = [len(roots) for roots in shard_roots]
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="label_batch_start",
            shard_index=shard_index,
            shard_count=len(shard_roots),
            roots=len(flat_roots),
        ),
    )
    labeled = label_roots(flat_roots, config=config, workers=max(1, int(workers)), executor=executor)
    chunks: list[list[V3CombatLabeledRoot]] = []
    offset = 0
    for length in lengths:
        chunks.append(labeled[offset : offset + length])
        offset += length
    if offset != len(labeled):
        raise RuntimeError(f"labeled root split mismatch: {offset} != {len(labeled)}")
    scheduled: list[ShardWriteResult] = []
    write_futures: list[Future[ShardWriteResult]] = []
    git_status = metadata.get("git_status")
    if git_status is None:
        git_status = _git_status(repo_root)
    for chunk_offset, (roots, labeled_chunk) in enumerate(zip(shard_roots, chunks, strict=True)):
        current_shard_index = int(shard_index) + int(chunk_offset)
        shard_path = output_dir / f"shard_{current_shard_index:05d}.pt"
        shard_metadata = {
            **metadata,
            "root_count": len(labeled_chunk),
            "shard_index": current_shard_index,
            "git_status": git_status,
            "teacher_version": TEACHER_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "teacher_config": config.__dict__,
            "root_stats": root_stats(roots),
        }
        count = len(labeled_chunk)
        scheduled.append((shard_path, count))
        if writer_executor is None:
            _write_labeled_shard(
                shard_path=shard_path,
                labeled_roots=labeled_chunk,
                metadata=shard_metadata,
                memory_log=memory_log,
                shard_index=current_shard_index,
            )
        else:
            _append_jsonl(
                memory_log,
                _memory_snapshot(
                    event="shard_write_queued",
                    shard_index=current_shard_index,
                    roots=count,
                    path=str(shard_path),
                ),
            )
            write_futures.append(
                writer_executor.submit(
                    _write_labeled_shard,
                    shard_path=shard_path,
                    labeled_roots=labeled_chunk,
                    metadata=shard_metadata,
                    memory_log=memory_log,
                    shard_index=current_shard_index,
                )
            )
    _append_jsonl(
        memory_log,
        _memory_snapshot(
            event="label_batch_write_queued" if write_futures else "label_batch_done",
            shard_index=shard_index,
            shard_count=len(shard_roots),
            roots=len(flat_roots),
        ),
    )
    if not write_futures:
        del labeled
        gc.collect()
    return scheduled, write_futures


def generate_exploratory_shards(
    *,
    output_dir: Path,
    seed_start: int,
    seed_end: int,
    target_roots: int,
    curated_potion_roots: int,
    shard_size: int,
    per_seed_root_cap: int | None,
    max_steps_per_seed: int,
    repo_root: Path,
    device: str,
    random_action_rate: float,
    combat_selector: str | None,
    v3_combat_model: Path | None,
    combat_model: Path | None,
    config: TeacherConfig,
    workers: int,
    collect_workers: int,
    label_batch_shards: int,
    shard_write_workers: int,
    label_pipeline_batches: int,
    memory_log: Path | None,
    append_output: bool = False,
    skip_append_dedupe: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(output_dir.glob("shard_*.pt")) if append_output else []
    existing_root_ids: set[str] = set()
    existing_roots = len(existing_shards) * int(shard_size)
    existing_stats = root_stats([])
    if existing_shards and not skip_append_dedupe:
        existing_root_ids, existing_roots, existing_stats = _load_existing_shard_state(existing_shards)
    if existing_roots >= target_roots:
        raise SystemExit(f"append output already has at least target roots: {existing_roots} >= {target_roots}")
    requested_seed_start = int(seed_start)
    if append_output and existing_root_ids and not skip_append_dedupe:
        existing_seeds = [
            seed
            for root_id in existing_root_ids
            if (seed := _exploratory_seed_from_root_id(root_id)) is not None
        ]
        if existing_seeds:
            # Re-run the latest partially written seed, but skip fully duplicated
            # root ids. This avoids scanning from seed 1 on large resumes.
            seed_start = max(int(seed_start), max(existing_seeds))
    selectors = build_runtime_selectors(
        repo_root=repo_root,
        device=device,
        combat_model=combat_model,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
    )
    rng = random.Random(7)
    buffer: list[Any] = []
    pending_shards: list[list[Any]] = []
    shard_paths: list[str] = [str(path) for path in existing_shards]
    shard_index = len(existing_shards)
    total_roots = int(existing_roots)
    total_exploratory_roots = int(existing_roots)
    total_curated_potion_roots = 0
    processed_seeds = 0
    aggregate_stats = existing_stats
    skipped_duplicate_roots = 0
    metadata = {
        "source": "exploratory_with_curated_potion" if curated_potion_roots > 0 else "exploratory",
        "seed_start": seed_start,
        "requested_seed_start": requested_seed_start,
        "seed_end": seed_end,
        "target_roots": target_roots,
        "append_output": bool(append_output),
        "skip_append_dedupe": bool(skip_append_dedupe),
        "existing_output_roots": int(existing_roots),
        "existing_output_shards": len(existing_shards),
        "curated_potion_roots_requested": curated_potion_roots,
        "shard_size": shard_size,
        "per_seed_root_cap": per_seed_root_cap,
        "max_steps_per_seed": max_steps_per_seed,
        "random_action_rate": random_action_rate,
        "combat_selector": combat_selector,
        "v3_combat_model": str(v3_combat_model) if v3_combat_model is not None else None,
        "combat_model": str(combat_model) if combat_model is not None else None,
        "collect_workers": int(collect_workers),
        "label_batch_shards": int(label_batch_shards),
        "shard_write_workers": int(shard_write_workers),
        "label_pipeline_batches": int(label_pipeline_batches),
        "git_status": _git_status(repo_root),
    }

    write_futures: list[Future[ShardWriteResult]] = []
    label_futures: list[tuple[Future[LabelBatchResult], int, int, int]] = []
    max_write_backlog = max(1, int(label_batch_shards)) * max(1, int(shard_write_workers)) * 2

    def drain_write_futures(*, block: bool) -> None:
        nonlocal write_futures
        if not write_futures:
            return
        remaining: list[Future[tuple[Path, int]]] = []
        for future in write_futures:
            if block or future.done():
                future.result()
            else:
                remaining.append(future)
        write_futures = remaining

    def enforce_write_backlog_limit() -> None:
        nonlocal write_futures
        while len(write_futures) >= max_write_backlog:
            future = write_futures.pop(0)
            future.result()

    def apply_label_batch_result(
        *,
        saved: list[ShardWriteResult],
        futures: list[Future[ShardWriteResult]],
        batch_shard_index: int,
        batch_shard_count: int,
        batch_roots: int,
    ) -> None:
        nonlocal total_roots
        write_futures.extend(futures)
        for shard_path, count in saved:
            shard_paths.append(str(shard_path))
            total_roots += count
        _append_jsonl(
            memory_log,
            _memory_snapshot(
                event="label_batch_result_applied",
                shard_index=batch_shard_index,
                shard_count=batch_shard_count,
                roots=batch_roots,
                write_backlog=len(write_futures),
            ),
        )
        drain_write_futures(block=False)

    def drain_label_futures(*, block: bool) -> None:
        nonlocal label_futures
        if not label_futures:
            return
        remaining: list[tuple[Future[LabelBatchResult], int, int, int]] = []
        for future, batch_shard_index, batch_shard_count, batch_roots in label_futures:
            if block or future.done():
                saved, futures = future.result()
                apply_label_batch_result(
                    saved=saved,
                    futures=futures,
                    batch_shard_index=batch_shard_index,
                    batch_shard_count=batch_shard_count,
                    batch_roots=batch_roots,
                )
            else:
                remaining.append((future, batch_shard_index, batch_shard_count, batch_roots))
        label_futures = remaining

    def enforce_label_backlog_limit() -> None:
        nonlocal label_futures
        max_label_backlog = max(0, int(label_pipeline_batches))
        if max_label_backlog <= 0:
            return
        while len(label_futures) >= max_label_backlog:
            future, batch_shard_index, batch_shard_count, batch_roots = label_futures.pop(0)
            saved, futures = future.result()
            apply_label_batch_result(
                saved=saved,
                futures=futures,
                batch_shard_index=batch_shard_index,
                batch_shard_count=batch_shard_count,
                batch_roots=batch_roots,
            )

    def queued_roots() -> int:
        return len(buffer) + sum(len(roots) for roots in pending_shards) + sum(
            roots for _, _, _, roots in label_futures
        )

    def flush_pending() -> None:
        nonlocal pending_shards, shard_index, total_roots
        if not pending_shards:
            return
        batch_roots = pending_shards
        batch_shard_index = shard_index
        batch_shard_count = len(batch_roots)
        batch_root_count = sum(len(roots) for roots in batch_roots)
        shard_index += batch_shard_count
        pending_shards = []
        drain_label_futures(block=False)
        if label_executor is not None:
            enforce_label_backlog_limit()
            _append_jsonl(
                memory_log,
                _memory_snapshot(
                    event="label_batch_queued",
                    shard_index=batch_shard_index,
                    shard_count=batch_shard_count,
                    roots=batch_root_count,
                    label_backlog=len(label_futures) + 1,
                ),
            )
            label_futures.append(
                (
                    label_executor.submit(
                        _save_labeled_batch,
                        shard_roots=batch_roots,
                        output_dir=output_dir,
                        shard_index=batch_shard_index,
                        config=config,
                        workers=workers,
                        repo_root=repo_root,
                        metadata=metadata,
                        memory_log=memory_log,
                        executor=executor,
                        writer_executor=writer_executor,
                    ),
                    batch_shard_index,
                    batch_shard_count,
                    batch_root_count,
                )
            )
            return
        drain_write_futures(block=False)
        enforce_write_backlog_limit()
        saved, futures = _save_labeled_batch(
            shard_roots=batch_roots,
            output_dir=output_dir,
            shard_index=batch_shard_index,
            config=config,
            workers=workers,
            repo_root=repo_root,
            metadata=metadata,
            memory_log=memory_log,
            executor=executor,
            writer_executor=writer_executor,
        )
        apply_label_batch_result(
            saved=saved,
            futures=futures,
            batch_shard_index=batch_shard_index,
            batch_shard_count=batch_shard_count,
            batch_roots=batch_root_count,
        )
        gc.collect()

    def queue_full_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        pending_shards.append(buffer)
        buffer = []
        if len(pending_shards) >= int(label_batch_shards):
            flush_pending()

    def add_roots(roots: list[Any]) -> int:
        nonlocal buffer, skipped_duplicate_roots
        kept_roots: list[Any] = []
        for root in roots:
            root_id = _root_id(root)
            if not skip_append_dedupe and root_id is not None and root_id in existing_root_ids:
                skipped_duplicate_roots += 1
                continue
            if not skip_append_dedupe and root_id is not None:
                existing_root_ids.add(root_id)
            kept_roots.append(root)
        if not kept_roots:
            return 0
        _merge_root_stats(aggregate_stats, root_stats(kept_roots))
        for root in kept_roots:
            buffer.append(root)
            if len(buffer) >= shard_size:
                queue_full_buffer()
        return len(kept_roots)

    def add_seed_roots(seed: int, seed_roots: list[Any], stats: dict[str, Any]) -> int:
        nonlocal total_exploratory_roots
        added_total = 0
        for root in seed_roots:
            if total_exploratory_roots >= target_roots:
                break
            added = add_roots([root])
            if added:
                total_exploratory_roots += added
                added_total += added
        _append_jsonl(
            memory_log,
            _memory_snapshot(
                event="seed_done",
                seed=seed,
                steps=stats.get("steps"),
                seed_roots=added_total,
                collected_seed_roots=len(seed_roots),
                floor=stats.get("floor"),
                phase=stats.get("phase"),
                error=stats.get("error"),
                total_roots=total_roots,
                exploratory_roots=total_exploratory_roots,
                buffered_roots=len(buffer),
                pending_shards=len(pending_shards),
                queued_roots=queued_roots(),
                skipped_duplicate_roots=skipped_duplicate_roots,
            ),
        )
        return added_total

    worker_pool_context = (
        ProcessPoolExecutor(max_workers=max(1, int(workers))) if int(workers) > 1 else nullcontext(None)
    )
    writer_pool_context = (
        ThreadPoolExecutor(max_workers=max(1, int(shard_write_workers))) if int(shard_write_workers) > 0 else nullcontext(None)
    )
    label_pool_context = (
        ThreadPoolExecutor(max_workers=1) if int(label_pipeline_batches) > 0 else nullcontext(None)
    )
    with worker_pool_context as executor, writer_pool_context as writer_executor, label_pool_context as label_executor:
        if curated_potion_roots > 0:
            curated_roots = collect_curated_potion_roots(limit=curated_potion_roots)
            total_curated_potion_roots = len(curated_roots)
            add_roots(curated_roots)

        if int(collect_workers) > 1:
            max_collect_workers = max(1, int(collect_workers))
            max_pending_collect = max_collect_workers * max(
                1,
                int(os.environ.get("SPIRECOMM_COLLECT_PENDING_MULTIPLIER", "1")),
            )
            seed_iter = iter(range(int(seed_start), int(seed_end) + 1))
            collect_executor = ProcessPoolExecutor(max_workers=max_collect_workers)
            collect_futures: dict[Future[tuple[int, list[Any], dict[str, Any]]], int] = {}

            def submit_next_seed() -> bool:
                try:
                    next_seed = next(seed_iter)
                except StopIteration:
                    return False
                _append_jsonl(
                    memory_log,
                    _memory_snapshot(
                        event="seed_submitted",
                        seed=next_seed,
                        total_roots=total_roots,
                        exploratory_roots=total_exploratory_roots,
                        buffered_roots=len(buffer),
                        pending_shards=len(pending_shards),
                        queued_roots=queued_roots(),
                        skipped_duplicate_roots=skipped_duplicate_roots,
                    ),
                )
                payload = {
                    "seed": int(next_seed),
                    "repo_root": str(repo_root),
                    "device": device,
                    "random_action_rate": float(random_action_rate),
                    "combat_selector": combat_selector,
                    "v3_combat_model": str(v3_combat_model) if v3_combat_model is not None else "",
                    "combat_model": str(combat_model) if combat_model is not None else "",
                    "per_seed_root_cap": per_seed_root_cap,
                    "max_steps_per_seed": int(max_steps_per_seed),
                }
                collect_futures[collect_executor.submit(_collect_exploratory_seed_worker, payload)] = int(next_seed)
                return True

            try:
                while len(collect_futures) < max_pending_collect and submit_next_seed():
                    pass
                while collect_futures and total_exploratory_roots < target_roots:
                    future = next(as_completed(tuple(collect_futures)))
                    submitted_seed = collect_futures.pop(future)
                    seed, seed_roots, stats = future.result()
                    processed_seeds += 1
                    added = add_seed_roots(int(seed), seed_roots, stats)
                    print(
                        "collected "
                        f"seed={seed} added_roots={added} collected_roots={len(seed_roots)} "
                        f"floor={stats.get('floor')} phase={stats.get('phase')} "
                        f"done_seeds={processed_seeds} total_exploratory_roots={total_exploratory_roots}",
                        flush=True,
                    )
                    if int(seed) != int(submitted_seed):
                        _append_jsonl(
                            memory_log,
                            _memory_snapshot(
                                event="seed_result_mismatch",
                                submitted_seed=submitted_seed,
                                result_seed=seed,
                            ),
                        )
                    while (
                        total_exploratory_roots < target_roots
                        and len(collect_futures) < max_pending_collect
                        and submit_next_seed()
                    ):
                        pass
            finally:
                if total_exploratory_roots >= target_roots:
                    for future in collect_futures:
                        future.cancel()
                collect_executor.shutdown(wait=True, cancel_futures=True)
        else:
            for seed in range(int(seed_start), int(seed_end) + 1):
                if total_exploratory_roots >= target_roots:
                    break
                processed_seeds += 1
                _append_jsonl(
                    memory_log,
                    _memory_snapshot(
                        event="seed_start",
                        seed=seed,
                        total_roots=total_roots,
                        exploratory_roots=total_exploratory_roots,
                        buffered_roots=len(buffer),
                        pending_shards=len(pending_shards),
                        queued_roots=queued_roots(),
                        skipped_duplicate_roots=skipped_duplicate_roots,
                    ),
                )
                print(
                    "collected "
                    f"seed={seed} added_roots={seed_roots} collected_roots={seed_roots} "
                    f"floor={int(getattr(env, 'floor', 0))} phase={str(getattr(env, 'phase', ''))} "
                    f"done_seeds={processed_seeds} total_exploratory_roots={total_exploratory_roots}",
                    flush=True,
                )
                env = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=True)
                steps = 0
                seed_roots = 0
                while env.phase not in {"GAME_OVER", "COMPLETE", "VICTORY"} and steps < max_steps_per_seed:
                    if env.phase == "COMBAT" and (per_seed_root_cap is None or seed_roots < per_seed_root_cap):
                        root = make_root_sample(env, root_id=f"explore:{seed}:{steps}", source="exploratory")
                        if root is not None and total_exploratory_roots < target_roots:
                            added_roots = add_roots([root])
                            if added_roots:
                                seed_roots += added_roots
                                total_exploratory_roots += added_roots
                            if total_exploratory_roots >= target_roots:
                                break
                    actions = env.legal_actions()
                    if not actions:
                        break
                    if rng.random() < random_action_rate:
                        action = rng.choice(actions)
                    else:
                        action, _, _ = choose_modeled_action(env, selectors)
                    try:
                        env.step(action)
                    except Exception as exc:
                        _append_jsonl(memory_log, _memory_snapshot(event="seed_step_error", seed=seed, step=steps, error=str(exc)))
                        break
                    steps += 1
                _append_jsonl(
                    memory_log,
                    _memory_snapshot(
                        event="seed_done",
                        seed=seed,
                        steps=steps,
                        seed_roots=seed_roots,
                        collected_seed_roots=seed_roots,
                        floor=int(getattr(env, "floor", 0)),
                        phase=str(getattr(env, "phase", "")),
                        total_roots=total_roots,
                        exploratory_roots=total_exploratory_roots,
                        buffered_roots=len(buffer),
                        pending_shards=len(pending_shards),
                        queued_roots=queued_roots(),
                        skipped_duplicate_roots=skipped_duplicate_roots,
                    ),
                )

        if buffer:
            queue_full_buffer()
        flush_pending()
        drain_label_futures(block=True)
        drain_write_futures(block=True)
    summary = {
        "output_dir": str(output_dir),
        "shards": shard_paths,
        "roots": total_roots,
        "exploratory_roots": total_exploratory_roots,
        "curated_potion_roots": total_curated_potion_roots,
        "processed_seeds": processed_seeds,
        "seed_start": seed_start,
        "requested_seed_start": requested_seed_start,
        "seed_end": seed_end,
        "append_output": bool(append_output),
        "skip_append_dedupe": bool(skip_append_dedupe),
        "existing_output_roots": int(existing_roots),
        "skipped_duplicate_roots": int(skipped_duplicate_roots),
        "label_batch_shards": int(label_batch_shards),
        "collect_workers": int(collect_workers),
        "shard_write_workers": int(shard_write_workers),
        "label_pipeline_batches": int(label_pipeline_batches),
        "root_stats": aggregate_stats,
        "teacher_version": TEACHER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _label_root_batch_worker(
    payload: tuple[list[tuple[int, Any]], TeacherConfig],
) -> list[tuple[int, V3CombatLabeledRoot, str]]:
    _disable_gc_for_hot_worker_if_enabled()
    indexed_roots, config = payload
    results: list[tuple[int, V3CombatLabeledRoot, str]] = []
    for index, root in indexed_roots:
        results.append((index, label_root_sample(root, config=config), str(root.root_id)))
    return results


def _root_label_cost_hint(root: Any) -> tuple[int, int, int, int]:
    actions = list(getattr(root, "actions", []) or [])
    potion_actions = sum(1 for action in actions if str(action.get("kind") or "") == "potion")
    non_potion_actions = max(0, len(actions) - potion_actions)
    blocked_baseline_hint = potion_actions * non_potion_actions
    return (blocked_baseline_hint, potion_actions, len(actions), len(getattr(root, "env_blob", b"") or b""))


def label_roots(
    roots: list[Any],
    *,
    config: TeacherConfig,
    workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> list[V3CombatLabeledRoot]:
    total = len(roots)
    if workers <= 1:
        labeled: list[V3CombatLabeledRoot] = []
        progress_interval = max(1, int(os.environ.get("SPIRECOMM_LABEL_ROOT_PROGRESS_INTERVAL", "100")))
        for root in roots:
            labeled.append(label_root_sample(root, config=config))
            if len(labeled) == 1 or len(labeled) == total or len(labeled) % progress_interval == 0:
                print(f"labeled {len(labeled)}/{total} root={root.root_id}", flush=True)
        return labeled

    results: list[V3CombatLabeledRoot | None] = [None] * total
    completed = 0
    owns_executor = executor is None
    active_executor = executor or ProcessPoolExecutor(max_workers=workers)
    try:
        indexed_roots = sorted(enumerate(roots), key=lambda item: _root_label_cost_hint(item[1]), reverse=True)
        task_batch_size = max(1, int(os.environ.get("SPIRECOMM_LABEL_ROOT_TASK_BATCH_SIZE", "4")))
        if task_batch_size <= 1:
            task_batches = [[item] for item in indexed_roots]
        else:
            batch_count = max(1, math.ceil(len(indexed_roots) / task_batch_size))
            task_batches = [[] for _ in range(batch_count)]
            for position, item in enumerate(indexed_roots):
                task_batches[position % batch_count].append(item)
        futures = [active_executor.submit(_label_root_batch_worker, (batch, config)) for batch in task_batches if batch]
        progress_interval = max(1, int(os.environ.get("SPIRECOMM_LABEL_ROOT_PROGRESS_INTERVAL", "100")))
        for future in as_completed(futures):
            last_root_id = ""
            for index, labeled_root, root_id in future.result():
                results[index] = labeled_root
                completed += 1
                last_root_id = root_id
            if completed == total or completed % progress_interval < task_batch_size:
                print(f"labeled {completed}/{total} root={last_root_id}", flush=True)
    finally:
        if owns_executor:
            active_executor.shutdown()
    return [root for root in results if root is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a v3 combat teacher dataset shard.")
    parser.add_argument("--source", choices=["curated", "trace", "exploratory", "all"], default="curated")
    parser.add_argument("--output", type=Path, default=Path("data/v3_combat_teacher/shards/shard_00000.pt"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Write multiple memory-safe shards into this directory.")
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Continue writing sharded --output-dir after existing shard_*.pt files and skip duplicate root_id values.",
    )
    parser.add_argument(
        "--skip-append-dedupe",
        action="store_true",
        help="With --append-output, trust the requested seed range is non-overlapping and do not load existing shards.",
    )
    parser.add_argument("--max-roots", type=int, default=100)
    parser.add_argument("--target-roots", type=int, default=None, help="Alias for --max-roots in sharded generation.")
    parser.add_argument("--curated-potion-roots", type=int, default=0, help="Add this many curated potion roots to the generated dataset.")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--trace-glob", default="_cache/real_game_first/model_required_seed*_pause1_traces/seed_*_trace.json")
    parser.add_argument("--max-trace-files", type=int, default=20)
    parser.add_argument("--seeds", default="1,12,7133506393411724536")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="legacy-slot")
    parser.add_argument("--combat-model", type=Path, default=None)
    parser.add_argument("--v3-combat-model", type=Path, default=None)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--node-budget", type=int, default=768)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument(
        "--teacher-config-json",
        default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_JSON", ""),
        help="Inline JSON object overriding v3 teacher coefficients.",
    )
    parser.add_argument(
        "--teacher-config-path",
        default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_PATH", ""),
        help="Path to JSON object overriding v3 teacher coefficients.",
    )
    parser.add_argument("--random-action-rate", type=float, default=0.25)
    parser.add_argument("--per-seed-root-cap", type=int, default=None)
    parser.add_argument("--max-steps-per-seed", type=int, default=1000)
    parser.add_argument("--collect-workers", type=int, default=1, help="Parallel seed collectors for non-sharded exploratory generation.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--label-batch-shards",
        type=int,
        default=1,
        help="Label this many output shards in one worker-pool batch to reduce per-shard long-tail idle time.",
    )
    parser.add_argument(
        "--shard-write-workers",
        type=int,
        default=1,
        help="Write labeled shards asynchronously with this many background threads; use 0 for synchronous writes.",
    )
    parser.add_argument(
        "--label-pipeline-batches",
        type=int,
        default=1,
        help="Keep this many label batches in a background pipeline while the main process collects more roots; use 0 for synchronous label batches.",
    )
    parser.add_argument("--memory-log", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    target_roots = int(args.target_roots if args.target_roots is not None else args.max_roots)
    if args.label_batch_shards <= 0:
        raise SystemExit("--label-batch-shards must be positive")
    if args.shard_write_workers < 0:
        raise SystemExit("--shard-write-workers must be non-negative")
    if args.label_pipeline_batches < 0:
        raise SystemExit("--label-pipeline-batches must be non-negative")
    if args.teacher_config_path and args.teacher_config_json:
        raise SystemExit("--teacher-config-path and --teacher-config-json are mutually exclusive")
    if args.teacher_config_path:
        config = teacher_config_from_json_path(args.teacher_config_path)
    elif args.teacher_config_json:
        payload = json.loads(args.teacher_config_json)
        if not isinstance(payload, dict):
            raise SystemExit("--teacher-config-json must decode to a JSON object")
        config = teacher_config_from_mapping(payload)
    else:
        config = TeacherConfig()
    config = TeacherConfig(
        **{
            **config.__dict__,
            "beam_width": int(args.beam_width),
            "node_budget_per_root": int(args.node_budget),
            "max_depth": int(args.max_depth),
        }
    )

    if args.output_dir is not None:
        memory_log = args.memory_log or (args.output_dir / "memory_log.jsonl")
        if args.source != "exploratory":
            raise SystemExit("--output-dir sharded generation currently supports --source exploratory.")
        if args.seed_start is None or args.seed_end is None:
            raise SystemExit("--output-dir exploratory generation requires --seed-start and --seed-end.")
        if args.output_dir.exists() and list(args.output_dir.glob("shard_*.pt")) and not args.append_output:
            raise SystemExit(f"output dir already contains shards; use --append-output to continue: {args.output_dir}")
        summary = generate_exploratory_shards(
            output_dir=args.output_dir,
            seed_start=int(args.seed_start),
            seed_end=int(args.seed_end),
            target_roots=target_roots,
            curated_potion_roots=max(0, int(args.curated_potion_roots)),
            shard_size=max(1, int(args.shard_size)),
            per_seed_root_cap=args.per_seed_root_cap,
            max_steps_per_seed=max(1, int(args.max_steps_per_seed)),
            repo_root=repo_root,
            device=args.device,
            random_action_rate=float(args.random_action_rate),
            combat_selector=args.combat_selector,
            v3_combat_model=args.v3_combat_model,
            combat_model=args.combat_model,
            config=config,
            workers=max(1, int(args.workers)),
            collect_workers=max(1, int(args.collect_workers)),
            label_batch_shards=max(1, int(args.label_batch_shards)),
            shard_write_workers=max(0, int(args.shard_write_workers)),
            label_pipeline_batches=max(0, int(args.label_pipeline_batches)),
            memory_log=memory_log,
            append_output=bool(args.append_output),
            skip_append_dedupe=bool(args.skip_append_dedupe),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    roots = []
    if args.source in {"curated", "all"}:
        roots.extend(collect_curated_roots(limit=target_roots - len(roots)))
        if int(args.curated_potion_roots) > 0 and len(roots) < target_roots:
            roots.extend(collect_curated_potion_roots(limit=min(int(args.curated_potion_roots), target_roots - len(roots))))
    if args.source in {"trace", "all"} and len(roots) < target_roots:
        roots.extend(
            collect_trace_roots(
                args.trace_glob,
                max_roots=target_roots - len(roots),
                max_files=args.max_trace_files,
            )
        )
    if args.source in {"exploratory", "all"} and len(roots) < target_roots:
        if args.seed_start is not None and args.seed_end is not None:
            seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
        else:
            seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
        roots.extend(
            (
                collect_exploratory_roots_parallel
                if int(args.collect_workers) > 1 and len(seeds) > 1
                else collect_exploratory_roots
            )(
                seeds=seeds,
                max_roots=target_roots - len(roots),
                repo_root=repo_root,
                device=args.device,
                random_action_rate=args.random_action_rate,
                combat_selector=args.combat_selector,
                v3_combat_model=args.v3_combat_model,
                combat_model=args.combat_model,
                per_seed_root_cap=args.per_seed_root_cap,
                max_steps_per_seed=max(1, int(args.max_steps_per_seed)),
                **({"workers": max(1, int(args.collect_workers))} if int(args.collect_workers) > 1 and len(seeds) > 1 else {}),
            )
        )

    labeled = label_roots(roots[:target_roots], config=config, workers=max(1, int(args.workers)))

    save_shard(
        args.output,
        labeled,
        metadata={
            "source": args.source,
            "root_count": len(labeled),
            "git_status": _git_status(repo_root),
            "teacher_version": TEACHER_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "teacher_config": config.__dict__,
            "root_stats": root_stats(roots[:target_roots]),
        },
    )
    print(json.dumps({"output": str(args.output), "roots": len(labeled), "root_stats": root_stats(roots[:target_roots])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
