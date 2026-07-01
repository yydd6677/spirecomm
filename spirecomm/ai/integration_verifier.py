from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random
from typing import Any

from compare_native_to_lightspeed_run import (
    _attach_failure_metadata,
    _battle_signature,
    _choice_list_signature,
    _norm_action,
    _pick_action_by_signature,
    _summarize_results,
)
from spirecomm.ai.learned_policy import CheckpointCombatPolicy
from spirecomm.ai.lightspeed_combat_model import SerializedCombatSelector
from spirecomm.ai.runtime_decision import (
    attach_card_target,
    build_runtime_selectors,
    choose_card_reward,
    choose_combat,
    choose_modeled_action,
    choose_run_choice,
)
from spirecomm.native_sim.cards import make_card
from spirecomm.native_sim.potions import make_potion
from spirecomm.native_sim_v2 import NativeRunEnv
from spirecomm.native_sim_v2.env import NativeCombatEnv
from spirecomm.native_sim_v2.monsters import make_monster


DEFAULT_LIGHTSPEED_BUILD = Path("/home/yydd/sts_lightspeed/build")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _state_fingerprint(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ensure_lightspeed_import(lightspeed_build: Path | None = None):
    build_path = Path(lightspeed_build or DEFAULT_LIGHTSPEED_BUILD)
    if str(build_path) not in sys.path:
        sys.path.insert(0, str(build_path))
    try:
        import slaythespire as sts
    except ModuleNotFoundError as exc:
        if exc.name != "slaythespire":
            raise
        raise ModuleNotFoundError(
            "slaythespire is required for lightspeed-backed model integration checks. "
            "Install the lightspeed Python package/build, or use native-only entrypoints "
            "such as run_native_run.py, run_native_sim.py, or export_model_run_checklist.py."
        ) from exc

    return sts


def default_runtime_model_paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = Path(repo_root or _repo_root())
    return {
        "combat": root / "models" / "combat.pt",
        "combat_bc": root / "models" / "combat_bc.pt",
        "combat_pref": root / "models" / "combat_pref.pt",
        "card_reward": root / "models" / "card_reward.pt",
        "boss_relic": root / "models" / "boss_relic.pt",
        "map": root / "models" / "map_choice.pt",
        "campfire": root / "models" / "campfire.pt",
        "event": root / "models" / "event_choice.pt",
        "shop": root / "models" / "shop_choice_prior_delta.pt",
        "potion": root / "models" / "potion_use.pt",
        "upgrade_target": root / "models" / "upgrade_target.pt",
        "purge_target": root / "models" / "purge_target.pt",
    }


def available_selector_metadata(selectors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for name, selector in selectors.items():
        if name == "enable_neow":
            continue
        checkpoint_path = getattr(selector, "checkpoint_path", None)
        metadata[name] = {
            "available": bool(getattr(selector, "available", False)),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "device": getattr(selector, "device", None),
        }
    return metadata


def validate_combat_model_loads(
    *,
    repo_root: Path | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    observation_version: str | None = None,
    include_alternates: bool = True,
) -> list[dict[str, Any]]:
    paths = default_runtime_model_paths(repo_root)
    model_names = ["combat"]
    if include_alternates:
        model_names.extend(["combat_bc", "combat_pref"])

    env = NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)
    state = env.state()
    actions = env.legal_actions()
    results: list[dict[str, Any]] = []
    for model_name in model_names:
        model_path = paths[model_name]
        result: dict[str, Any] = {
            "model_name": model_name,
            "model_path": str(model_path),
            "selector_available": False,
            "policy_loaded": False,
        }
        try:
            selector = SerializedCombatSelector(
                checkpoint_path=model_path,
                device=combat_device or device,
                observation_version=observation_version,
            )
            result["selector_available"] = selector.available
            if selector.available:
                action, scores = selector.choose(state, actions)
                result["selector_choice"] = {
                    "kind": action.get("kind") if action else None,
                    "name": action.get("name") if action else None,
                    "scores_len": len(scores),
                }
        except Exception as exc:
            result["selector_error"] = repr(exc)
        try:
            policy = CheckpointCombatPolicy(
                checkpoint_path=str(model_path),
                device=combat_device or device,
                observation_version=observation_version,
            )
            scoring = policy.score_state(state)
            result["policy_loaded"] = True
            result["policy_shapes"] = {
                "action_logits": list(scoring["action_logits"].shape),
                "target_logits": list(scoring["target_logits"].shape),
            }
        except Exception as exc:
            result["policy_error"] = repr(exc)
        results.append(result)
    return results


def _combat_env_fixture() -> NativeRunEnv:
    return NativeRunEnv(seed=1, ascension_level=0, enable_neow=False)


def _card_reward_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=1, ascension_level=0, start_on_map=True)
    env.phase = "CARD_REWARD"
    env.reward_context = "COMBAT"
    env.reward_card_bundles = [[
        make_card("Anger", uuid="reward-anger"),
        make_card("Flex", uuid="reward-flex"),
        make_card("Pommel Strike", uuid="reward-pommel"),
    ]]
    env.reward_relics = []
    env.reward_potions = []
    env.reward_gold_piles = []
    env.reward_emerald_key = False
    env._refresh_reward_cards()
    return env


def _boss_relic_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=402, ascension_level=0, start_on_map=True)
    env.phase = "BOSS_RELIC"
    env.floor = 17
    env.act = 1
    env.deck = [
        make_card("Strike_R", uuid="boss-strike"),
        make_card("Defend_R", uuid="boss-defend"),
        make_card("Bash", uuid="boss-bash"),
    ]
    env.boss_relic_options = [
        {"kind": "boss_relic", "name": "Calling Bell", "relic_id": "Calling Bell", "choice_index": 0},
        {"kind": "boss_relic", "name": "Black Blood", "relic_id": "Black Blood", "choice_index": 1},
        {"kind": "boss_relic", "name": "Astrolabe", "relic_id": "Astrolabe", "choice_index": 2},
    ]
    return env


def _map_env_fixture() -> NativeRunEnv:
    return NativeRunEnv(seed=1, ascension_level=0, start_on_map=True)


def _campfire_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=4067, ascension_level=0, start_on_map=True)
    env.phase = "CAMPFIRE"
    env.player.current_hp = 30
    env.player.max_hp = 80
    env.campfire_options = [
        {"kind": "campfire", "name": "REST", "choice_index": 0},
        {"kind": "campfire", "name": "LEAVE", "choice_index": 1},
    ]
    return env


def _event_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=4064, ascension_level=0, start_on_map=True)
    env.phase = "EVENT"
    env.event_id = "Vampires"
    env.event_options = [
        {"kind": "event", "event_id": "Vampires", "name": "Accepted", "label": "Accepted", "choice_index": 1},
        {"kind": "event", "event_id": "Vampires", "name": "Refused", "label": "Refused", "choice_index": 2},
    ]
    return env


def _shop_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=17, ascension_level=0, start_on_map=True)
    env.phase = "SHOP"
    env.gold = 200
    env.shop_items = [
        {
            "kind": "shop",
            "name": "Power Potion",
            "item_kind": "potion",
            "item_id": "Power Potion",
            "potion_id": "Power Potion",
            "price": 51,
            "choice_index": 0,
        },
        {"kind": "shop", "name": "LEAVE", "item_kind": "leave", "price": 0, "choice_index": 1},
    ]
    return env


def _potion_combat_fixture() -> NativeCombatEnv:
    combat = NativeCombatEnv(seed=1, ascension_level=0)
    combat.player.current_hp = 30
    combat.potions = [make_potion("Fire Potion")]
    if combat.monsters:
        combat.monsters[0].current_hp = 8
    else:
        combat.monsters = [make_monster("JawWorm", Random(1), ascension=0)]
        combat.monsters[0].current_hp = 8
    return combat


def _upgrade_target_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=4067, ascension_level=0, start_on_map=True)
    env.phase = "CAMPFIRE"
    env.deck = [
        make_card("Strike_R", uuid="upgrade-strike"),
        make_card("Defend_R", uuid="upgrade-defend"),
        make_card("Bash", uuid="upgrade-bash"),
    ]
    return env


def _purge_target_env_fixture() -> NativeRunEnv:
    env = NativeRunEnv(seed=17, ascension_level=0, start_on_map=True)
    env.phase = "SHOP"
    env.gold = 200
    env.deck = [
        make_card("Strike_R", uuid="purge-strike"),
        make_card("Defend_R", uuid="purge-defend"),
        make_card("Bash", uuid="purge-bash"),
    ]
    env.shop_items = [
        {"kind": "shop", "name": "PURGE", "item_kind": "purge", "price": 75, "choice_index": 0},
        {"kind": "shop", "name": "LEAVE", "item_kind": "leave", "price": 0, "choice_index": 1},
    ]
    return env


def validate_selector_phase_fixtures(selectors: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    env = _combat_env_fixture()
    before = _state_fingerprint(env.state())
    action, scores = choose_combat(env, selectors.get("combat"))
    env.step(action)
    results.append(
        {
            "selector": "combat",
            "action": {"kind": action.get("kind"), "name": action.get("name")},
            "scores_len": len(scores),
            "consumed": before != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    env = _card_reward_env_fixture()
    before_deck = len(env.deck)
    before_state = _state_fingerprint(env.state())
    action, scores = choose_card_reward(env, selectors.get("card_reward"))
    env.step(action)
    results.append(
        {
            "selector": "card_reward",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "card_id": action.get("card_id")},
            "scores_len": len(scores),
            "consumed": before_deck != len(env.deck) or before_state != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    env = _boss_relic_env_fixture()
    before_state = _state_fingerprint(env.state())
    action, scores = choose_run_choice(env, selectors.get("boss_relic"))
    env.step(action)
    results.append(
        {
            "selector": "boss_relic",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "relic_id": action.get("relic_id")},
            "scores_len": len(scores),
            "consumed": before_state != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    env = _map_env_fixture()
    before_state = _state_fingerprint(env.state())
    action, scores = choose_run_choice(env, selectors.get("map"))
    env.step(action)
    results.append(
        {
            "selector": "map",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "symbol": action.get("symbol")},
            "scores_len": len(scores),
            "consumed": before_state != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    env = _campfire_env_fixture()
    before_state = _state_fingerprint(env.state())
    action, scores = choose_run_choice(env, selectors.get("campfire"))
    env.step(action)
    results.append(
        {
            "selector": "campfire",
            "action": {"kind": action.get("kind"), "name": action.get("name")},
            "scores_len": len(scores),
            "consumed": before_state != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    env = _event_env_fixture()
    before_state = _state_fingerprint(env.state())
    action, scores = choose_run_choice(env, selectors.get("event"))
    env.step(action)
    results.append(
        {
            "selector": "event",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "event_id": action.get("event_id")},
            "scores_len": len(scores),
            "consumed": before_state != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    env = _shop_env_fixture()
    before_state = _state_fingerprint(env.state())
    action, scores = choose_run_choice(env, selectors.get("shop"))
    env.step(action)
    results.append(
        {
            "selector": "shop",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "item_kind": action.get("item_kind")},
            "scores_len": len(scores),
            "consumed": before_state != _state_fingerprint(env.state()),
            "phase_after": env.phase,
        }
    )

    combat = _potion_combat_fixture()
    potion_actions = [action for action in combat.legal_actions() if action.get("kind") == "potion"]
    potion_selector = selectors.get("potion")
    state = combat.to_spirecomm_state()
    before_state = _state_fingerprint(state)
    result = potion_selector.choose(state, [dict(action, action="USE") for action in potion_actions]) if potion_selector and potion_selector.available else None
    choice_index = int(result["choice_index"]) if result is not None else 0
    action = potion_actions[min(max(choice_index, 0), len(potion_actions) - 1)]
    combat.step(action)
    results.append(
        {
            "selector": "potion",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "potion_id": action.get("potion_id")},
            "scores_len": len(result.get("scores", [])) if result else 0,
            "consumed": before_state != _state_fingerprint(combat.to_spirecomm_state()),
            "phase_after": "COMBAT",
        }
    )

    env = _upgrade_target_env_fixture()
    base_action = {"kind": "campfire", "name": "SMITH", "choice_index": 0}
    action = attach_card_target(env, base_action, selectors)
    before_upgrades = [card.upgrades for card in env.deck]
    env.step(action)
    after_upgrades = [card.upgrades for card in env.deck]
    results.append(
        {
            "selector": "upgrade_target",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "target_index": action.get("target_index")},
            "consumed": before_upgrades != after_upgrades,
            "phase_after": env.phase,
        }
    )

    env = _purge_target_env_fixture()
    base_action = {"kind": "shop", "name": "PURGE", "item_kind": "purge", "price": 75, "choice_index": 0}
    action = attach_card_target(env, base_action, selectors)
    before_deck = len(env.deck)
    env.step(action)
    results.append(
        {
            "selector": "purge_target",
            "action": {"kind": action.get("kind"), "name": action.get("name"), "target_index": action.get("target_index")},
            "consumed": before_deck != len(env.deck),
            "phase_after": env.phase,
        }
    )

    return results


def run_natural_phase_coverage(
    selectors: dict[str, Any],
    *,
    seed: int = 1,
    count: int = 30,
    ascension: int = 0,
    max_floor: int = 20,
    max_steps: int = 5000,
) -> dict[str, Any]:
    phase_counts: dict[str, int] = {}
    selector_counts: dict[str, int] = {}
    final_phases: dict[str, int] = {}
    final_floors: list[int] = []

    for offset in range(count):
        env = NativeRunEnv(seed=seed + offset, ascension_level=ascension, enable_neow=False)
        steps = 0
        while env.phase not in {"GAME_OVER", "COMPLETE"} and env.floor <= max_floor and steps < max_steps:
            phase_counts[env.phase] = phase_counts.get(env.phase, 0) + 1
            action, _, source = choose_modeled_action(env, selectors)
            selector_counts[source] = selector_counts.get(source, 0) + 1
            env.step(action)
            steps += 1
        final_phases[env.phase] = final_phases.get(env.phase, 0) + 1
        final_floors.append(int(env.floor))

    return {
        "phase_counts": dict(sorted(phase_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
        "final_phases": dict(sorted(final_phases.items())),
        "avg_final_floor": sum(final_floors) / max(1, len(final_floors)),
        "max_final_floor": max(final_floors) if final_floors else 0,
    }


def _compare_frontier(
    *,
    seed: int,
    step: int,
    ls_env: Any,
    native: NativeRunEnv,
    trace: list[dict[str, Any]],
) -> dict[str, Any] | None:
    import slaythespire as sts

    ls_in_battle = bool(ls_env.in_battle)
    native_in_battle = native.phase == "COMBAT"

    if ls_env.outcome != sts.GameOutcome.UNDECIDED or native.phase in {"GAME_OVER", "COMPLETE"}:
        return {
            "seed": seed,
            "match": ls_env.outcome == sts.GameOutcome.PLAYER_VICTORY and native.phase == "COMPLETE"
            or ls_env.outcome == sts.GameOutcome.PLAYER_LOSS and native.phase == "GAME_OVER",
            "reason": "finished",
            "step": step,
            "trace_tail": trace[-5:],
            "lightspeed_outcome": str(ls_env.outcome),
            "native_phase": native.phase,
            "lightspeed_floor": int(ls_env.floor_num),
            "native_floor": int(native.floor),
        }

    if ls_in_battle != native_in_battle:
        return {
            "seed": seed,
            "match": False,
            "reason": "battle_phase_mismatch",
            "step": step,
            "lightspeed_in_battle": ls_in_battle,
            "native_phase": native.phase,
            "lightspeed_floor": int(ls_env.floor_num),
            "native_floor": int(native.floor),
            "trace_tail": trace[-5:],
        }

    if ls_in_battle:
        ls_state = sts.get_battle_state(ls_env)
        native_state = native.state()
        ls_sig = _battle_signature(ls_state)
        native_sig = _battle_signature(native_state)
        if ls_sig != native_sig:
            return {
                "seed": seed,
                "match": False,
                "reason": "battle_state_mismatch",
                "step": step,
                "trace_tail": trace[-5:],
                "lightspeed": ls_sig,
                "native": native_sig,
            }
        ls_actions = [dict(action) for action in sts.get_battle_actions(ls_env)]
        native_actions = [dict(action) for action in native.legal_actions()]
        current_ls_state = ls_state
        current_native_state = native_state
    else:
        ls_actions = [dict(action) for action in sts.get_external_actions(ls_env.game_context)]
        native_actions = [dict(action) for action in native.legal_actions()]
        current_ls_state = None
        current_native_state = None

    ls_norm = _choice_list_signature(ls_actions, current_ls_state)
    native_norm = _choice_list_signature(native_actions, current_native_state)
    if ls_norm != native_norm:
        mismatch_payload: dict[str, Any] = {
            "seed": seed,
            "match": False,
            "reason": "legal_action_mismatch",
            "step": step,
            "trace_tail": trace[-5:],
            "lightspeed_phase": "BATTLE" if ls_in_battle else str(ls_env.screen_state),
            "native_phase": native.phase,
            "lightspeed_floor": int(ls_env.floor_num),
            "native_floor": int(native.floor),
            "lightspeed_actions": ls_norm,
            "native_actions": native_norm,
        }
        if ls_in_battle:
            mismatch_payload["lightspeed"] = _battle_signature(current_ls_state)
            mismatch_payload["native"] = _battle_signature(current_native_state)
        return mismatch_payload

    return None


def compare_seed_with_model(
    *,
    seed: int,
    selectors: dict[str, Any],
    ascension: int = 0,
    max_steps: int = 3000,
    backend_pair: str = "lightspeed,v3",
    lightspeed_build: Path | None = None,
    target_phase: str | None = None,
    target_floor: int | None = None,
    stop_after_model_action: bool = True,
) -> dict[str, Any]:
    if backend_pair not in {"lightspeed,v2", "lightspeed,v3"}:
        raise ValueError(f"unsupported backend pair: {backend_pair}")

    sts = _ensure_lightspeed_import(lightspeed_build)
    ls_env = sts.ModelDrivenEnv(seed, ascension)
    native = NativeRunEnv(seed=seed, ascension_level=ascension, enable_neow=True)
    trace: list[dict[str, Any]] = []
    model_action_executed = False

    for step in range(max_steps):
        mismatch = _compare_frontier(seed=seed, step=step, ls_env=ls_env, native=native, trace=trace)
        if mismatch is not None:
            return _attach_failure_metadata(mismatch, backend="v3")

        use_model = target_phase is None and target_floor is None
        if target_phase is not None and native.phase == target_phase:
            use_model = target_floor is None or int(native.floor) == int(target_floor)

        ls_in_battle = bool(ls_env.in_battle)
        current_ls_state = sts.get_battle_state(ls_env) if ls_in_battle else None
        current_native_state = native.state() if native.phase == "COMBAT" else None

        if use_model:
            native_chosen, scores, source = choose_modeled_action(native, selectors)
            chosen_sig = _norm_action(native_chosen, current_native_state)
            model_action_executed = True
        else:
            ls_actions = [dict(action) for action in (sts.get_battle_actions(ls_env) if ls_in_battle else sts.get_external_actions(ls_env.game_context))]
            chosen_sig = _choice_list_signature(ls_actions, current_ls_state)[0]
            scores = []
            source = "warmup"

        ls_actions = [dict(action) for action in (sts.get_battle_actions(ls_env) if ls_in_battle else sts.get_external_actions(ls_env.game_context))]
        native_actions = [dict(action) for action in native.legal_actions()]
        ls_chosen = _pick_action_by_signature(ls_actions, chosen_sig, current_ls_state)
        native_chosen = _pick_action_by_signature(native_actions, chosen_sig, current_native_state)

        trace.append(
            {
                "step": step,
                "phase": "BATTLE" if ls_in_battle else native.phase,
                "choice": chosen_sig,
                "floor": int(native.floor),
                "source": source,
                "scores_preview": [round(float(value), 4) for value in scores[:6]],
            }
        )
        if ls_in_battle:
            sts.execute_battle_action_bits(ls_env, int(ls_chosen["bits"]))
        else:
            sts.execute_action_bits(ls_env, int(ls_chosen["bits"]))
        native.step(native_chosen)

        if use_model and stop_after_model_action:
            post_mismatch = _compare_frontier(seed=seed, step=step + 1, ls_env=ls_env, native=native, trace=trace)
            if post_mismatch is not None:
                return _attach_failure_metadata(post_mismatch, backend="v3")
            return {
                "seed": seed,
                "match": True,
                "reason": "model_action_matched",
                "step": step + 1,
                "trace_tail": trace[-5:],
                "lightspeed_floor": int(ls_env.floor_num),
                "native_floor": int(native.floor),
                "target_phase": target_phase,
                "target_floor": target_floor,
                "model_action_executed": True,
            }

    return {
        "seed": seed,
        "match": bool(model_action_executed),
        "reason": "max_steps_reached",
        "step": max_steps,
        "trace_tail": trace[-5:],
        "lightspeed_floor": int(ls_env.floor_num),
        "native_floor": int(native.floor),
        "target_phase": target_phase,
        "target_floor": target_floor,
        "model_action_executed": model_action_executed,
    }


def default_parity_cases() -> list[dict[str, Any]]:
    return [
        {"name": "runtime_combat_seed1", "seed": 1, "max_steps": 50},
        {"name": "runtime_card_reward_seed1_floor1", "seed": 1, "target_phase": "CARD_REWARD", "target_floor": 1, "max_steps": 120},
        {"name": "runtime_event_seed1_floor3", "seed": 1, "target_phase": "EVENT", "target_floor": 3, "max_steps": 200},
        {"name": "runtime_campfire_seed1_floor6", "seed": 1, "target_phase": "CAMPFIRE", "target_floor": 6, "max_steps": 400},
        {"name": "runtime_shop_seed8866187513371018371_floor10", "seed": 8866187513371018371, "target_phase": "SHOP", "target_floor": 10, "max_steps": 260},
        {"name": "runtime_boss_relic_seed12_floor17", "seed": 12, "target_phase": "BOSS_RELIC", "target_floor": 17, "max_steps": 600},
    ]


def run_parity_cases(
    selectors: dict[str, Any],
    *,
    cases: list[dict[str, Any]] | None = None,
    ascension: int = 0,
    backend_pair: str = "lightspeed,v3",
    lightspeed_build: Path | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in (cases or default_parity_cases()):
        result = compare_seed_with_model(
            seed=int(case["seed"]),
            selectors=selectors,
            ascension=ascension,
            max_steps=int(case.get("max_steps", 3000)),
            backend_pair=backend_pair,
            lightspeed_build=lightspeed_build,
            target_phase=case.get("target_phase"),
            target_floor=case.get("target_floor"),
            stop_after_model_action=bool(case.get("stop_after_model_action", True)),
        )
        result["case_name"] = case.get("name")
        results.append(_attach_failure_metadata(result, backend="v3"))
    return results


def validate_runtime_integration(
    *,
    repo_root: Path | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    observation_version: str | None = None,
    backend_pair: str = "lightspeed,v3",
    lightspeed_build: Path | None = None,
    include_alternate_combat: bool = True,
    coverage_count: int = 30,
) -> dict[str, Any]:
    root = Path(repo_root or _repo_root())
    selectors = build_runtime_selectors(
        repo_root=root,
        device=device,
        combat_device=combat_device,
        observation_version=observation_version,
    )
    summary: dict[str, Any] = {
        "selector_metadata": available_selector_metadata(selectors),
        "combat_loads": validate_combat_model_loads(
            repo_root=root,
            device=device,
            combat_device=combat_device,
            observation_version=observation_version,
            include_alternates=include_alternate_combat,
        ),
        "fixtures": validate_selector_phase_fixtures(selectors),
        "coverage": run_natural_phase_coverage(selectors, count=coverage_count),
        "parity": run_parity_cases(
            selectors,
            backend_pair=backend_pair,
            lightspeed_build=lightspeed_build,
        ),
    }

    if include_alternate_combat:
        paths = default_runtime_model_paths(root)
        alt_results: list[dict[str, Any]] = []
        for model_name in ("combat_bc", "combat_pref"):
            alt_selectors = build_runtime_selectors(
                repo_root=root,
                device=device,
                combat_device=combat_device,
                combat_model=paths[model_name],
                observation_version=observation_version,
            )
            alt_result = compare_seed_with_model(
                seed=1,
                selectors=alt_selectors,
                ascension=0,
                max_steps=50,
                backend_pair=backend_pair,
                lightspeed_build=lightspeed_build,
            )
            alt_result["case_name"] = f"{model_name}_combat_seed1"
            alt_results.append(_attach_failure_metadata(alt_result, backend="v2"))
        summary["alternate_combat_parity"] = alt_results
    else:
        summary["alternate_combat_parity"] = []

    fixture_failures = [item for item in summary["fixtures"] if not item.get("consumed")]
    combat_failures = [
        item
        for item in summary["combat_loads"]
        if not item.get("selector_available") or not item.get("policy_loaded")
    ]
    parity_failures = [item for item in summary["parity"] if not item.get("match")]
    alt_failures = [item for item in summary["alternate_combat_parity"] if not item.get("match")]
    summary["ok"] = not fixture_failures and not combat_failures and not parity_failures and not alt_failures
    summary["failures"] = {
        "fixture_failures": fixture_failures,
        "combat_failures": combat_failures,
        "parity_failures": parity_failures,
        "alternate_combat_failures": alt_failures,
    }
    summary["parity_summary"] = _summarize_results(summary["parity"] + summary["alternate_combat_parity"])
    return summary
