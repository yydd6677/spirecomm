#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from spirecomm.ai.runtime_decision import (
    TRACE_POLICY_MODEL_REQUIRED,
    build_runtime_selectors,
    choose_model_required_action,
    choose_modeled_action,
    model_selector_status,
    normalize_trace_policy,
    validate_model_required_selectors,
)
from spirecomm.ai.strict_trace import (
    LEGACY_TRACE_SCHEMA,
    STRICT_TRACE_SCHEMA,
    build_strict_step_payload,
)
from spirecomm.seed_helper import canonical_seed_string


def _native_env_cls(backend: str):
    if backend == "v2":
        from spirecomm.native_sim_v2 import NativeRunEnv

        return NativeRunEnv
    if backend == "v3":
        from spirecomm.native_sim_v3 import NativeRunEnv

        return NativeRunEnv
    from spirecomm.native_sim import NativeRunEnv

    return NativeRunEnv


def _safe_seed_str(seed: int) -> str:
    return canonical_seed_string(seed) or str(seed)


def _card_label(card: dict[str, Any] | None) -> str:
    if not card:
        return "UNKNOWN_CARD"
    name = str(card.get("name") or card.get("card_id") or "UNKNOWN_CARD")
    upgrades = int(card.get("upgrades", 0) or 0)
    if upgrades > 0:
        name = f"{name}+{upgrades}"
    cost = card.get("cost")
    cost_for_turn = card.get("cost_for_turn")
    cost_bits: list[str] = []
    if cost is not None:
        cost_bits.append(f"cost={cost}")
    if cost_for_turn is not None:
        cost_bits.append(f"turn={cost_for_turn}")
    if card.get("free_to_play_once"):
        cost_bits.append("free_once")
    if cost_bits:
        return f"{name} ({', '.join(cost_bits)})"
    return name


def _monster_label(monster: dict[str, Any] | None) -> str:
    if not monster:
        return "UNKNOWN_MONSTER"
    name = str(monster.get("name") or monster.get("monster_id") or "UNKNOWN_MONSTER")
    hp = monster.get("current_hp")
    block = monster.get("block")
    intent = monster.get("intent")
    move_name = monster.get("move_name") or monster.get("move_id")
    bits: list[str] = []
    if hp is not None:
        bits.append(f"hp={hp}")
    if block is not None:
        bits.append(f"block={block}")
    if intent:
        bits.append(f"intent={intent}")
    if move_name:
        bits.append(f"move={move_name}")
    if bits:
        return f"{name} ({', '.join(bits)})"
    return name


def _target_card_from_deck(state: dict[str, Any], index: int | None) -> str:
    if index is None:
        return "UNKNOWN_TARGET"
    deck = list(state.get("deck") or [])
    if 0 <= index < len(deck):
        return f"{_card_label(deck[index])} [deck_index={index}]"
    return f"UNKNOWN_TARGET [deck_index={index}]"


def _format_choice_item(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "")
    choice_index = item.get("choice_index")
    prefix = f"[{choice_index}] " if choice_index is not None else ""
    if kind == "neow":
        bonus_text = str(item.get("bonus_text") or item.get("bonus") or "")
        drawback_text = str(item.get("drawback_text") or "")
        label = str(item.get("label") or item.get("name") or "NEOW_OPTION")
        if drawback_text:
            return prefix + f"{label}: {bonus_text} | drawback: {drawback_text}"
        return prefix + f"{label}: {bonus_text}"
    if kind in {"card_reward", "card_select", "single_card_select", "multi_card_select"} or item.get("card_id"):
        return prefix + _card_label(item)
    if kind == "boss_relic" or item.get("relic_id"):
        relic_id = str(item.get("relic_id") or item.get("name") or "UNKNOWN_RELIC")
        return prefix + f"Relic {relic_id}"
    if kind == "shop":
        price = item.get("price")
        item_kind = str(item.get("item_kind") or "")
        if item_kind == "card":
            return prefix + f"Shop Card { _card_label(item.get('card')) } | {price}g"
        if item_kind == "relic":
            relic = item.get("relic") or {}
            relic_id = str(relic.get("relic_id") or item.get("name") or "UNKNOWN_RELIC")
            return prefix + f"Shop Relic {relic_id} | {price}g"
        if item_kind == "potion":
            potion_id = str(item.get("potion_id") or item.get("name") or "UNKNOWN_POTION")
            return prefix + f"Shop Potion {potion_id} | {price}g"
        if item_kind == "purge":
            return prefix + f"PURGE | {price}g"
        if item_kind == "leave":
            return prefix + "LEAVE"
    if kind == "event":
        event_id = str(item.get("event_id") or "")
        name = str(item.get("label") or item.get("name") or "UNKNOWN_EVENT_OPTION")
        return prefix + f"{name}" + (f" [event={event_id}]" if event_id else "")
    if kind == "map":
        symbol = str(item.get("symbol") or item.get("name") or "?")
        floor = item.get("floor")
        x = item.get("x")
        node_id = item.get("node_id")
        return prefix + f"{symbol} -> floor {floor}, x={x}, node={node_id}"
    if kind == "campfire":
        return prefix + str(item.get("label") or item.get("name") or "UNKNOWN_CAMPFIRE_OPTION")
    if kind == "treasure":
        return prefix + str(item.get("name") or "UNKNOWN_TREASURE_OPTION")
    if kind == "reward_gold":
        return prefix + f"GOLD +{item.get('amount', 0)}"
    if kind == "reward_key":
        return prefix + f"KEY ({item.get('key')})"
    if kind == "potion":
        return prefix + str(item.get("name") or item.get("potion_id") or "UNKNOWN_POTION")
    return prefix + json.dumps(item, ensure_ascii=True, sort_keys=True)


def _format_action(record: dict[str, Any]) -> str:
    action = dict(record["action"])
    state = dict(record["pre_state"])
    phase = str(record["phase"])
    kind = str(action.get("kind") or "")
    name = str(action.get("name") or "")
    if phase == "COMBAT":
        combat = state.get("combat_state") or {}
        monsters = list(combat.get("monsters") or [])
        if kind == "card":
            target_index = action.get("target_index")
            if target_index is not None and 0 <= int(target_index) < len(monsters):
                return f"{name} -> { _monster_label(monsters[int(target_index)]) } [slot={int(target_index)}]"
            return name
        if kind == "potion":
            target_index = action.get("target_index")
            potion_id = action.get("potion_id") or name
            if target_index is not None and 0 <= int(target_index) < len(monsters):
                return f"Potion {potion_id} -> { _monster_label(monsters[int(target_index)]) } [slot={int(target_index)}]"
            return f"Potion {potion_id}"
        if kind == "end":
            return "END TURN"
        return json.dumps(action, ensure_ascii=True, sort_keys=True)
    if phase == "SHOP":
        item_kind = str(action.get("item_kind") or "")
        price = action.get("price")
        if item_kind == "card":
            return f"BUY card { _card_label(action.get('card')) } | {price}g"
        if item_kind == "relic":
            relic = action.get("relic") or {}
            return f"BUY relic {relic.get('relic_id') or name} | {price}g"
        if item_kind == "potion":
            return f"BUY potion {action.get('potion_id') or name} | {price}g"
        if item_kind == "purge":
            return f"PURGE { _target_card_from_deck(state, action.get('target_index')) } | {price}g"
        if item_kind == "leave":
            return "LEAVE SHOP"
    if phase == "CARD_REWARD":
        if kind == "card_reward":
            return f"TAKE card { _card_label(action.get('card')) }"
        if kind == "reward_relic":
            relic = action.get("relic") or {}
            return f"TAKE relic {relic.get('relic_id') or name}"
        if kind == "reward_potion":
            return f"TAKE potion {action.get('potion_id') or name}"
        if kind == "reward_gold":
            return f"TAKE gold +{action.get('amount', 0)}"
        if kind == "reward_key":
            return f"TAKE key {action.get('key')}"
        if kind == "skip":
            return "LEAVE REWARD SCREEN"
    if phase == "MAP":
        return f"PATH -> {action.get('symbol') or name} [floor={action.get('floor')}, x={action.get('x')}, node={action.get('node_id')}]"
    if phase == "EVENT":
        return f"EVENT OPTION -> {action.get('label') or name} [event={action.get('event_id')}]"
    if phase == "CAMPFIRE":
        if action.get("target_index") is not None:
            return f"{name} -> { _target_card_from_deck(state, action.get('target_index')) }"
        return name
    if phase == "CARD_SELECT":
        options = list(state.get("choice_list") or [])
        choice_index = action.get("choice_index")
        if choice_index is not None and 0 <= int(choice_index) < len(options):
            return f"SELECT { _format_choice_item(options[int(choice_index)]) }"
        if action.get("index") is not None:
            idx = int(action["index"])
            if 0 <= idx < len(options):
                return f"SELECT { _format_choice_item(options[idx]) }"
        return json.dumps(action, ensure_ascii=True, sort_keys=True)
    if phase == "BOSS_RELIC":
        if action.get("relic_id") == "SKIP":
            return "SKIP boss relic"
        return f"TAKE boss relic {action.get('relic_id') or name}"
    if phase in {"TREASURE", "CHEST"}:
        return name
    if phase == "NEOW":
        return f"NEOW -> {action.get('label') or name}"
    return json.dumps(action, ensure_ascii=True, sort_keys=True)


def _format_action_with_source(record: dict[str, Any]) -> str:
    source = str(record.get("action_source") or "unknown")
    return f"[source={source}] {_format_action(record)}"


def _strict_state_has_visible_neow_rewards(state: dict[str, Any]) -> bool:
    if str(state.get("phase") or "").upper() != "NEOW":
        return False
    choices = list((state.get("screen_state") or {}).get("choices") or [])
    if len(choices) < 2:
        return False
    return all(str((choice or {}).get("kind") or "").lower() == "neow" for choice in choices)


def _build_hidden_neow_intro_state(strict_state: dict[str, Any]) -> dict[str, Any]:
    intro_state = copy.deepcopy(strict_state)
    intro_state["phase"] = "NEOW"
    intro_state["room_type"] = "NeowRoom"
    intro_state["screen_type"] = "EVENT"
    intro_screen_state = dict(intro_state.get("screen_state") or {})
    intro_screen_state["choices"] = [
        {
            "choice_index": 0,
            "kind": "neow",
            "bonus": None,
            "drawback": "NONE",
            "label": "Talk",
            "text": "[Talk]",
        }
    ]
    intro_state["screen_state"] = intro_screen_state
    return intro_state


def _inject_strict_neow_intro_step(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steps:
        return steps
    first_step = dict(steps[0])
    if str(first_step.get("phase") or "").upper() != "NEOW":
        return steps
    first_pre_state = dict(first_step.get("strict_pre_state") or {})
    if not _strict_state_has_visible_neow_rewards(first_pre_state):
        return steps

    intro_pre_state = _build_hidden_neow_intro_state(first_pre_state)
    intro_step = {
        "step": 0,
        "floor": first_step.get("floor"),
        "phase": "NEOW",
        "action_source": "strict_neow_intro",
        "scores": [],
        "action": {
            "kind": "neow",
            "name": "TALK",
            "label": "Talk",
            "choice_index": 0,
            "bonus": None,
            "drawback": "NONE",
            "bonus_text": "[Talk]",
            "drawback_text": "",
        },
        "pre_state": copy.deepcopy(first_step.get("pre_state") or {}),
        "post_state": copy.deepcopy(first_step.get("pre_state") or {}),
        "pre_snapshot": copy.deepcopy(first_step.get("pre_snapshot") or {}),
        "post_phase": "NEOW",
        "post_floor": first_step.get("floor"),
        "post_hp": first_step.get("pre_state", {}).get("current_hp"),
        "post_gold": first_step.get("pre_state", {}).get("gold"),
        "strict_action": {
            "phase": "NEOW",
            "kind": "choose_by_index",
            "raw_kind": "neow",
            "choice_index": 0,
            "choice": dict((intro_pre_state.get("screen_state") or {}).get("choices", [{}])[0]),
        },
        "strict_pre_state": intro_pre_state,
        "strict_post_state": first_pre_state,
    }

    adjusted_steps = [intro_step]
    for new_index, step in enumerate(steps, start=1):
        updated = copy.deepcopy(step)
        updated["step"] = new_index
        adjusted_steps.append(updated)
    return adjusted_steps


def _summarize_deck(deck: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter(_card_label(card) for card in deck)
    lines = []
    for label in sorted(counts):
        lines.append(f"{label} x{counts[label]}")
    return lines


def _capture_run(
    *,
    seed: int,
    ascension: int,
    backend: str,
    max_steps: int | None,
    repo_root: Path,
    device: str,
    combat_device: str | None,
    combat_model: Path | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    observation_version: str | None = None,
    trace_schema_mode: str = "strict",
    trace_policy: str = TRACE_POLICY_MODEL_REQUIRED,
) -> dict[str, Any]:
    normalized_trace_policy = normalize_trace_policy(trace_policy)
    selectors = build_runtime_selectors(
        repo_root=repo_root,
        device=device,
        combat_device=combat_device,
        combat_model=combat_model,
        combat_selector=combat_selector,
        v3_combat_model=v3_combat_model,
        observation_version=observation_version,
    )
    selectors["enable_neow"] = True
    selector_status = model_selector_status(selectors)
    if normalized_trace_policy == TRACE_POLICY_MODEL_REQUIRED:
        validate_model_required_selectors(selectors)
    env = _native_env_cls(backend)(seed=seed, ascension_level=ascension, enable_neow=True)
    steps: list[dict[str, Any]] = []
    step_index = 0
    while env.phase not in {"GAME_OVER", "COMPLETE", "VICTORY"} and (
        max_steps is None or step_index < max_steps
    ):
        pre_phase = str(getattr(env, "phase", ""))
        pre_state = copy.deepcopy(env.state())
        pre_snapshot = {
            "state": pre_state,
            "room_symbol": getattr(env, "current_node_symbol", None),
            "current_map_node_id": getattr(env, "current_map_node_id", None),
            "current_event_id": getattr(env, "current_event_id", None),
            "card_select_context": getattr(env, "card_select_context", None),
        }
        if normalized_trace_policy == TRACE_POLICY_MODEL_REQUIRED:
            action, scores, source = choose_model_required_action(env, selectors)
        else:
            action, scores, source = choose_modeled_action(env, selectors)
        env.step(action)
        post_state = copy.deepcopy(env.state())
        strict_payload = None
        if trace_schema_mode == "strict":
            strict_payload = build_strict_step_payload(
                action=copy.deepcopy(action),
                pre_state=pre_state,
                post_state=post_state,
                phase=pre_phase,
                post_phase=str(env.phase),
                pre_snapshot=pre_snapshot,
            )
        steps.append(
            {
                "step": step_index,
                "floor": int(pre_state.get("floor", env.floor)),
                "phase": pre_phase,
                "action_source": source,
                "scores": [float(score) for score in scores],
                "action": copy.deepcopy(action),
                "pre_state": pre_state,
                "post_state": post_state,
                "pre_snapshot": pre_snapshot,
                "post_phase": env.phase,
                "post_floor": env.floor,
                "post_hp": env.player.current_hp,
                "post_gold": env.gold,
                **(
                    {
                        "strict_action": strict_payload.strict_action,
                        "strict_pre_state": strict_payload.strict_pre_state,
                        "strict_post_state": strict_payload.strict_post_state,
                    }
                    if strict_payload is not None
                    else {}
                ),
            }
        )
        step_index += 1

    final_state = env.state()
    return {
        "trace_schema": STRICT_TRACE_SCHEMA if trace_schema_mode == "strict" else LEGACY_TRACE_SCHEMA,
        "trace_policy": normalized_trace_policy,
        "model_required": normalized_trace_policy == TRACE_POLICY_MODEL_REQUIRED,
        "model_status": selector_status,
        "seed_long": seed,
        "seed_str": _safe_seed_str(seed),
        "ascension": ascension,
        "backend": backend,
        "max_steps": max_steps,
        "result": {
            "final_phase": env.phase,
            "final_floor": env.floor,
            "final_hp": env.player.current_hp,
            "final_gold": env.gold,
            "deck_size": len(env.deck),
            "steps": step_index,
        },
        "final_state": copy.deepcopy(final_state),
        "steps": steps,
        "model_paths": {
            "combat": str(combat_model or (repo_root / "models" / "combat.pt")),
            "card_reward": str(repo_root / "models" / "card_reward.pt"),
            "boss_relic": str(repo_root / "models" / "boss_relic.pt"),
            "map": str(repo_root / "models" / "map_choice.pt"),
            "campfire": str(repo_root / "models" / "campfire.pt"),
            "event": str(repo_root / "models" / "event_choice.pt"),
            "shop": str(repo_root / "models" / "shop_choice_prior_delta.pt"),
            "potion": str(repo_root / "models" / "potion_use.pt"),
            "upgrade_target": str(repo_root / "models" / "upgrade_target.pt"),
            "purge_target": str(repo_root / "models" / "purge_target.pt"),
        },
    }


def _cluster_key(step: dict[str, Any]) -> tuple[Any, ...]:
    phase = str(step["phase"])
    context = step["pre_snapshot"].get("card_select_context") if phase == "CARD_SELECT" else None
    room_symbol = step["pre_snapshot"].get("room_symbol") if phase == "COMBAT" else None
    event_id = step["pre_snapshot"].get("current_event_id") if phase == "EVENT" else None
    return (step["floor"], phase, context, room_symbol, event_id)


def _cluster_steps(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    for step in steps:
        key = _cluster_key(step)
        if current_key is None or key == current_key:
            current.append(step)
            current_key = key
            continue
        clusters.append(current)
        current = [step]
        current_key = key
    if current:
        clusters.append(current)
    return clusters


def _render_options(lines: list[str], items: list[dict[str, Any]], *, header: str) -> None:
    if not items:
        return
    lines.append(header)
    for item in items:
        lines.append(f"  - {_format_choice_item(item)}")


def _render_map_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    step = cluster[0]
    floor = int(step["floor"])
    lines.append(f"Floor {floor:02d} | MAP")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Visible paths:")
        lines.append(f"  Chosen path: {_format_action_with_source(record)}")
    lines.append("")


def _render_neow_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    lines.append("Neow")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Options:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
    lines.append("")


def _render_combat_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    first = cluster[0]
    floor = int(first["floor"])
    pre_state = first["pre_state"]
    combat = dict(pre_state.get("combat_state") or {})
    room_symbol = first["pre_snapshot"].get("room_symbol") or "M"
    monsters = list(combat.get("monsters") or [])
    lines.append(f"Floor {floor:02d} | COMBAT ({room_symbol})")
    lines.append(f"  Start HP: {pre_state.get('current_hp')}/{pre_state.get('max_hp')}")
    if monsters:
        lines.append("  Enemies:")
        for index, monster in enumerate(monsters):
            lines.append(f"  - slot {index}: {_monster_label(monster)}")
    turns: dict[int, list[str]] = {}
    for record in cluster:
        turn = int((record["pre_state"].get("combat_state") or {}).get("turn", 0) or 0)
        turns.setdefault(turn, []).append(_format_action_with_source(record))
    for turn in sorted(turns):
        lines.append(f"  Turn {turn}:")
        for action in turns[turn]:
            lines.append(f"  - {action}")
    last = cluster[-1]
    lines.append(
        "  Result: "
        f"HP {pre_state.get('current_hp')} -> {last['post_hp']} | "
        f"phase -> {last['post_phase']} | floor -> {last['post_floor']}"
    )
    lines.append("")


def _render_reward_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    floor = int(cluster[0]["floor"])
    lines.append(f"Floor {floor:02d} | CARD_REWARD")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Visible rewards:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
        pre_hp = int(record["pre_state"].get("current_hp", 0) or 0)
        post_hp = int(record["post_state"].get("current_hp", pre_hp) or pre_hp)
        pre_gold = int(record["pre_state"].get("gold", 0) or 0)
        post_gold = int(record["post_state"].get("gold", pre_gold) or pre_gold)
        if pre_hp != post_hp or pre_gold != post_gold:
            lines.append(f"  State change: HP {pre_hp}->{post_hp} | Gold {pre_gold}->{post_gold}")
    lines.append("")


def _render_shop_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    floor = int(cluster[0]["floor"])
    lines.append(f"Floor {floor:02d} | SHOP ($)")
    for index, record in enumerate(cluster, start=1):
        pre_state = record["pre_state"]
        post_state = record["post_state"]
        lines.append(
            f"  Shop step {index}: HP {pre_state.get('current_hp')}/{pre_state.get('max_hp')} | "
            f"Gold {pre_state.get('gold')}"
        )
        _render_options(lines, list(pre_state.get("choice_list") or []), header="  Visible shop items:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
        lines.append(f"  Gold after action: {pre_state.get('gold')} -> {post_state.get('gold')}")
    lines.append("")


def _render_event_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    first = cluster[0]
    floor = int(first["floor"])
    event_id = first["pre_snapshot"].get("current_event_id") or "UNKNOWN_EVENT"
    lines.append(f"Floor {floor:02d} | EVENT (?) | {event_id}")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Visible options:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
        pre_hp = int(record["pre_state"].get("current_hp", 0) or 0)
        post_hp = int(record["post_state"].get("current_hp", pre_hp) or pre_hp)
        pre_gold = int(record["pre_state"].get("gold", 0) or 0)
        post_gold = int(record["post_state"].get("gold", pre_gold) or pre_gold)
        if pre_hp != post_hp or pre_gold != post_gold:
            lines.append(f"  State change: HP {pre_hp}->{post_hp} | Gold {pre_gold}->{post_gold}")
    lines.append("")


def _render_campfire_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    floor = int(cluster[0]["floor"])
    lines.append(f"Floor {floor:02d} | CAMPFIRE (R)")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Visible options:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
    lines.append("")


def _render_card_select_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    first = cluster[0]
    floor = int(first["floor"])
    context = first["pre_snapshot"].get("card_select_context") or "UNKNOWN_CARD_SELECT"
    lines.append(f"Floor {floor:02d} | CARD_SELECT | {context}")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Select from:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
    lines.append("")


def _render_boss_relic_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    floor = int(cluster[0]["floor"])
    lines.append(f"Floor {floor:02d} | BOSS_RELIC")
    for record in cluster:
        _render_options(lines, list(record["pre_state"].get("choice_list") or []), header="  Visible relics:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
    lines.append("")


def _render_misc_cluster(cluster: list[dict[str, Any]], lines: list[str]) -> None:
    first = cluster[0]
    floor = int(first["floor"])
    phase = str(first["phase"])
    lines.append(f"Floor {floor:02d} | {phase}")
    for record in cluster:
        items = list(record["pre_state"].get("choice_list") or [])
        if items:
            _render_options(lines, items, header="  Visible options:")
        lines.append(f"  Chosen: {_format_action_with_source(record)}")
    lines.append("")


def _render_checklist(trace: dict[str, Any]) -> str:
    lines: list[str] = []
    result = dict(trace["result"])
    final_state = dict(trace["final_state"])
    lines.append("Model-Driven v2 Checklist")
    lines.append(f"seed_str: {trace['seed_str']}")
    lines.append(f"seed_long: {trace['seed_long']}")
    lines.append(f"ascension: {trace['ascension']}")
    lines.append(f"backend: {trace['backend']}")
    lines.append("models:")
    for key, value in trace["model_paths"].items():
        lines.append(f"  {key}: {value}")
    lines.append(
        "result: "
        f"{result['final_phase']} floor={result['final_floor']} hp={result['final_hp']} "
        f"gold={result['final_gold']} deck={result['deck_size']} steps={result['steps']}"
    )
    path_steps = [
        f"{step['action'].get('floor')}:{step['action'].get('symbol')}"
        for step in trace["steps"]
        if step["phase"] == "MAP"
    ]
    if path_steps:
        lines.append(f"path: {' -> '.join(path_steps)}")
    lines.append("")
    lines.append("Replay note:")
    lines.append("  Follow the rooms and choices in order in the real game.")
    lines.append("  If a mismatch happens, record the first floor/phase where the offered choices stop matching.")
    lines.append("")

    for cluster in _cluster_steps(list(trace["steps"])):
        phase = str(cluster[0]["phase"])
        if phase == "NEOW":
            _render_neow_cluster(cluster, lines)
        elif phase == "MAP":
            _render_map_cluster(cluster, lines)
        elif phase == "COMBAT":
            _render_combat_cluster(cluster, lines)
        elif phase == "CARD_REWARD":
            _render_reward_cluster(cluster, lines)
        elif phase == "SHOP":
            _render_shop_cluster(cluster, lines)
        elif phase == "EVENT":
            _render_event_cluster(cluster, lines)
        elif phase == "CAMPFIRE":
            _render_campfire_cluster(cluster, lines)
        elif phase == "CARD_SELECT":
            _render_card_select_cluster(cluster, lines)
        elif phase == "BOSS_RELIC":
            _render_boss_relic_cluster(cluster, lines)
        else:
            _render_misc_cluster(cluster, lines)

    lines.append("Final State")
    lines.append(f"  Floor: {result['final_floor']}")
    lines.append(f"  Phase: {result['final_phase']}")
    lines.append(f"  HP: {final_state.get('current_hp')}/{final_state.get('max_hp')}")
    lines.append(f"  Gold: {final_state.get('gold')}")
    lines.append(f"  Act Boss: {final_state.get('act_boss')}")
    lines.append("  Relics:")
    for relic in list(final_state.get("relics") or []):
        relic_id = relic.get("relic_id") or relic.get("name") or "UNKNOWN_RELIC"
        counter = relic.get("counter")
        if counter is not None:
            lines.append(f"  - {relic_id} (counter={counter})")
        else:
            lines.append(f"  - {relic_id}")
    lines.append("  Deck:")
    for line in _summarize_deck(list(final_state.get("deck") or [])):
        lines.append(f"  - {line}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a model-driven native run as a readable checklist and raw trace (defaults to backend v3).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--backend", choices=["v1", "v2", "v3"], default="v3", help="Native backend to use; defaults to v3, with v2 kept for comparison.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional explicit step cap. By default capture continues until the run reaches a terminal state.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--combat-device", default=None)
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate"], default="legacy-slot")
    parser.add_argument("--combat-model", type=Path, default=None, help="Legacy slot combat checkpoint; defaults to models/combat.pt.")
    parser.add_argument("--v3-combat-model", type=Path, default=Path("/home/yydd/spirecomm/models/v3_combat_scorer.pt"))
    parser.add_argument("--observation-version", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/yydd/spirecomm/eval_runs"))
    parser.add_argument(
        "--trace-schema",
        choices=["strict", "legacy"],
        default="strict",
        help="Trace export schema; defaults to strict, with legacy retained for bridge-only compatibility.",
    )
    parser.add_argument(
        "--trace-policy",
        choices=["model-required", "legacy-fallback"],
        default="model-required",
        help="Decision policy for trace capture; model-required fails instead of silently falling back.",
    )
    args = parser.parse_args()

    repo_root = Path("/home/yydd/spirecomm")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = _capture_run(
        seed=args.seed,
        ascension=args.ascension,
        backend=args.backend,
        max_steps=args.max_steps,
        repo_root=repo_root,
        device=args.device,
        combat_device=args.combat_device,
        combat_model=args.combat_model,
        combat_selector=args.combat_selector,
        v3_combat_model=args.v3_combat_model,
        observation_version=args.observation_version,
        trace_schema_mode=args.trace_schema,
        trace_policy=args.trace_policy,
    )
    checklist_text = _render_checklist(trace)
    seed_prefix = f"seed_{args.seed}"
    checklist_path = args.output_dir / f"{seed_prefix}_checklist.txt"
    trace_path = args.output_dir / f"{seed_prefix}_trace.json"
    checklist_path.write_text(checklist_text, encoding="utf-8")
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checklist": str(checklist_path), "trace": str(trace_path), "result": trace["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
