from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import random
import re
from typing import Any

from spirecomm.ai.card_reward_model import (
    CardRewardSelector,
    PurgeTargetSelector,
    UpgradeTargetSelector,
    early_card_reward_frontload_deficit_multiplier,
    normalize_token,
)
from spirecomm.ai.lightspeed_combat_model import SerializedCombatSelector
from spirecomm.ai.run_choice_model import (
    BossRelicSelector,
    CampfireChoiceSelector,
    EventChoiceSelector,
    PotionUseSelector,
    ShopChoiceSelector,
    canonical_event_model_label,
)
from spirecomm.ai.torch_compat import torch
from spirecomm.ai.v3_combat_features import clone_env_blob, incoming_damage, step_branch_from_blob
from spirecomm.native_sim_v3.content.potions import potion_priority_value

TRACE_POLICY_MODEL_REQUIRED = "model_required"
TRACE_POLICY_LEGACY_FALLBACK = "legacy_fallback"
MAP_DP_INITIAL_SHOP_VALUE = 0
MAP_DP_MONSTER_VALUE = int(os.environ.get("SPIRECOMM_MAP_DP_MONSTER_VALUE", "-10"))
MAP_DP_REST_VALUE = int(os.environ.get("SPIRECOMM_MAP_DP_REST_VALUE", "50"))
MAP_DP_ELITE_BASE_VALUE = int(os.environ.get("SPIRECOMM_MAP_DP_ELITE_BASE", "20"))
MAP_DP_GREEN_ELITE_PENALTY = int(os.environ.get("SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY", "40"))
MAP_DP_WINGED_OFFPATH_PENALTY = int(os.environ.get("SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY", "20"))
MAP_DP_SHOP_GOLD_UNIT_VALUE = int(os.environ.get("SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE", "20"))
MAP_DP_SHOP_PURGEABLE_CURSE_BONUS = int(os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS", "50"))
MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS = int(os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS", "50"))
MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON = int(os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON", "5"))
MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD = int(os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD", "125"))
MAP_DP_EARLY_ELITE_RISK_GUARD = str(os.environ.get("SPIRECOMM_MAP_DP_EARLY_ELITE_RISK_GUARD", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAP_DP_EARLY_ELITE_FLOOR_MAX = int(os.environ.get("SPIRECOMM_MAP_DP_EARLY_ELITE_FLOOR_MAX", "8"))
MAP_DP_EARLY_ELITE_PENALTY = int(os.environ.get("SPIRECOMM_MAP_DP_EARLY_ELITE_PENALTY", "60"))
MAP_DP_EARLY_ELITE_MIN_READINESS = int(os.environ.get("SPIRECOMM_MAP_DP_EARLY_ELITE_MIN_READINESS", "55"))
SHOP_VALUE_POLICY_UNREMOVABLE_CURSES = {"AscendersBane", "CurseOfTheBell", "Necronomicurse"}
SHOP_ACT1_FRONTLOAD_CARD_KEYS = {
    "anger",
    "carnage",
    "cleave",
    "clothesline",
    "headbutt",
    "heavyblade",
    "hemokinesis",
    "immolate",
    "pommelstrike",
    "pummel",
    "swordboomerang",
    "twinstrike",
    "uppercut",
    "whirlwind",
}
POSITIVE_CARD_TARGET_STARTER_BASIC_BIAS = float(os.environ.get("SPIRECOMM_POSITIVE_CARD_TARGET_STARTER_BASIC_BIAS", "-100.0"))
STARTER_BASIC_CARD_KEYS = {"strike", "defend", "bash"}
POSITIVE_CARD_TARGET_MODES = {
    "duplicate",
    "dual_wield",
    "headbutt",
    "exhume",
    "liquid_memories",
    "bottle_flame",
    "bottle_lightning",
    "bottle_tornado",
}
CARD_REWARD_CARD_SELECT_MODES = {*POSITIVE_CARD_TARGET_MODES, "discovery"}
FREE_CARD_SELECT_COST_BIAS_MODES = {"discovery", "liquid_memories"}

MODEL_REQUIRED_SELECTOR_NAMES = (
    "combat",
    "card_reward",
    "boss_relic",
    "campfire",
    "shop",
    "event",
    "potion",
    "upgrade_target",
    "purge_target",
)

COMBAT_ROLLOUT_RERANK_ACTIVE = 0
COMBAT_ROLLOUT_RERANK_TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
COMBAT_ROLLOUT_LIGHT_POLICY_DISABLED_ENVS = (
    "SPIRECOMM_V3_COMBAT_FORCED_TURN_SURVIVAL_GUARD",
    "SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_GUARD",
    "SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SURVIVAL_GUARD",
    "SPIRECOMM_V3_COMBAT_POST_ACTION_SURVIVAL_GUARD",
    "SPIRECOMM_V3_COMBAT_DELAYED_DEATH_GUARD",
    "SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_GUARD",
    "SPIRECOMM_V3_COMBAT_SHORT_WIN_GUARD",
)
CAMPFIRE_ROLLOUT_LIGHT_POLICY_OVERRIDES = {
    "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK": "0",
    "SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK": "0",
    "SPIRECOMM_EVENT_ROLLOUT_RERANK": "0",
    "SPIRECOMM_BOSS_RELIC_ROLLOUT_RERANK": "0",
}
RUNTIME_LATENCY_PROFILE_ENVS = {
    "interactive": {
        "SPIRECOMM_EVENT_FAST_POLICY": "1",
        "SPIRECOMM_EVENT_ROLLOUT_RERANK": "0",
        "SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD": "1",
        "SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RERANK": "0",
        "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER": "1",
        "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DANGER_HP_RATIO_MAX": "0.5",
    },
    "instant": {
        "SPIRECOMM_EVENT_FAST_POLICY": "1",
        "SPIRECOMM_EVENT_ROLLOUT_RERANK": "0",
        "SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD": "1",
        "SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RERANK": "0",
        "SPIRECOMM_BOSS_RELIC_ROLLOUT_RERANK": "0",
        "SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK": "0",
        "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK": "0",
    },
}
MAP_ROLLOUT_RERANK_ACTIVE = 0
DEFAULT_BOSS_RELIC_SCORE_BIAS: dict[str, float] = {
    "emptycage": -1.0,
    "skip": -1.0,
    "sozu": 0.6,
    "fusionhammer": 0.4,
}

_BOSS_RELIC_LOW_HP_RAW_TRUST_HP_MAX = 10
_BOSS_RELIC_LOW_HP_RAW_TRUST_RATIO_MAX = 0.15
_BOSS_RELIC_SKIP_RAW_TRUST_MARGIN = float(os.environ.get("SPIRECOMM_BOSS_RELIC_SKIP_RAW_TRUST_MARGIN", "inf"))
_BOSS_RELIC_ENERGY_ALT_TOKENS = {
    "bustedcrown",
    "coffeedripper",
    "ectoplasm",
    "fusionhammer",
    "markofpain",
    "philosophersstone",
    "runicdome",
    "slaverscollar",
    "sozu",
    "velvetchoker",
}

MODELED_ACTION_SOURCES = {
    "combat",
    "card_reward",
    "card_reward_skip",
    "boss_relic",
    "boss_relic_score_bias",
    "boss_relic_rollout_rerank",
    "boss_relic_skip_raw_trust",
    "map",
    "map_dp",
    "map_dp_rollout_rerank",
    "campfire",
    "campfire_low_hp_rest_guard",
    "campfire_final_boss_rest_fast_guard",
    "campfire_final_boss_rest_rollout",
    "shop",
    "event",
    "event_fast_big_fish",
    "event_rollout_rerank",
    "event_dead_adventurer_macro",
    "event_dead_adventurer_lowhp_cap",
    "event_golden_idol_max_hp_over_damage",
    "event_scrap_ooze_lowhp_cap",
    "event_lowhp_cost_guard",
    "combat_danger_block_progress_guard",
    "combat_monster_block_progress_guard",
    "potion",
    "upgrade_target",
    "purge_target",
    "remove_target",
    "transform_target",
    "duplicate_target",
    "burning_pact_target",
    "dual_wield_target",
    "elixir_target",
    "armaments_target",
    "headbutt_target",
    "exhume_target",
    "liquid_memories_target",
    "prepared_target",
    "true_grit_target",
    "warcry_target",
    "put_on_deck_target",
    "gambling_chip_target",
    "bottle_flame_target",
    "bottle_lightning_target",
    "bottle_tornado_target",
    "secret_technique_target",
    "secret_weapon_target",
    "library_card_select",
    "pandora_confirm",
    "reward_tradeoff",
}

ALLOWED_FORCED_ACTION_SOURCES = {
    "forced_single",
    "neow_fixed",
    "neow_weighted",
    "neow_rollout_rerank",
    "strict_neow_intro",
    "gambling_chip_confirm",
    "treasure_open_chest",
    "event_filtered_policy",
    "event_golden_idol_take",
    "event_match_and_keep_policy",
    "event_untrained_policy",
    "reward_policy_collect_gold",
    "reward_policy_collect_relic",
    "reward_policy_take_key",
    "reward_policy_collect_potion",
    "reward_policy_replace_potion_full",
    "reward_policy_open_card_reward",
    "reward_policy_skip_potion_full",
    "reward_policy_skip",
}

NEOW_BONUS_WEIGHTS = {
    "THREE_ENEMY_KILL": 24922,
    "ONE_RARE_RELIC": 6803,
    "BOSS_RELIC": 5750,
    "RANDOM_COMMON_RELIC": 5206,
    "THREE_RARE_CARDS": 3550,
    "TWO_FIFTY_GOLD": 3386,
    "TEN_PERCENT_HP_BONUS": 2727,
    "TRANSFORM_TWO_CARDS": 2612,
    "ONE_RANDOM_RARE_CARD": 2536,
    "HUNDRED_GOLD": 2261,
    "REMOVE_TWO": 1745,
    "UPGRADE_CARD": 1577,
    "RANDOM_COLORLESS_2": 1222,
    "TRANSFORM_CARD": 1101,
    "THREE_CARDS": 971,
    "REMOVE_CARD": 910,
    "TWENTY_PERCENT_HP_BONUS": 840,
    "RANDOM_COLORLESS": 687,
    "THREE_SMALL_POTIONS": 530,
}

NEOW_COMBO_WEIGHTS = {
    ("THREE_ENEMY_KILL", "NONE"): 24922,
    ("BOSS_RELIC", "NONE"): 5750,
    ("RANDOM_COMMON_RELIC", "NONE"): 5206,
    ("TEN_PERCENT_HP_BONUS", "NONE"): 2727,
    ("ONE_RANDOM_RARE_CARD", "NONE"): 2536,
    ("HUNDRED_GOLD", "NONE"): 2261,
    ("ONE_RARE_RELIC", "NO_GOLD"): 1977,
    ("ONE_RARE_RELIC", "TEN_PERCENT_HP_LOSS"): 1848,
    ("ONE_RARE_RELIC", "PERCENT_DAMAGE"): 1656,
    ("UPGRADE_CARD", "NONE"): 1577,
    ("ONE_RARE_RELIC", "CURSE"): 1322,
    ("TWO_FIFTY_GOLD", "TEN_PERCENT_HP_LOSS"): 1311,
    ("TWO_FIFTY_GOLD", "PERCENT_DAMAGE"): 1183,
    ("THREE_RARE_CARDS", "NO_GOLD"): 1132,
    ("TRANSFORM_CARD", "NONE"): 1101,
    ("THREE_RARE_CARDS", "TEN_PERCENT_HP_LOSS"): 992,
    ("THREE_CARDS", "NONE"): 971,
    ("REMOVE_CARD", "NONE"): 910,
    ("TWO_FIFTY_GOLD", "CURSE"): 892,
    ("TRANSFORM_TWO_CARDS", "NO_GOLD"): 867,
    ("THREE_RARE_CARDS", "PERCENT_DAMAGE"): 866,
    ("TRANSFORM_TWO_CARDS", "TEN_PERCENT_HP_LOSS"): 731,
    ("TRANSFORM_TWO_CARDS", "PERCENT_DAMAGE"): 711,
    ("RANDOM_COLORLESS", "NONE"): 687,
    ("REMOVE_TWO", "NO_GOLD"): 633,
    ("REMOVE_TWO", "TEN_PERCENT_HP_LOSS"): 614,
    ("THREE_RARE_CARDS", "CURSE"): 560,
    ("THREE_SMALL_POTIONS", "NONE"): 530,
    ("REMOVE_TWO", "PERCENT_DAMAGE"): 498,
    ("RANDOM_COLORLESS_2", "NO_GOLD"): 386,
    ("TWENTY_PERCENT_HP_BONUS", "PERCENT_DAMAGE"): 382,
    ("TWENTY_PERCENT_HP_BONUS", "NO_GOLD"): 350,
    ("RANDOM_COLORLESS_2", "PERCENT_DAMAGE"): 338,
    ("RANDOM_COLORLESS_2", "TEN_PERCENT_HP_LOSS"): 327,
    ("TRANSFORM_TWO_CARDS", "CURSE"): 303,
    ("RANDOM_COLORLESS_2", "CURSE"): 171,
    ("TWENTY_PERCENT_HP_BONUS", "CURSE"): 108,
}

NEOW_DEFAULT_CHOICE_INDEXES = {0, 1}

# Empirical v3 rollout performance prior over the choose0/choose1 Neow pools.
# Values are softmax(mean_floor / T) with T=1.0 over seed1-300 fixed-choice
# experiments using the current default model.
NEOW_DEFAULT_BONUS_WEIGHTS = {
    "RANDOM_COMMON_RELIC": 23.94,
    "REMOVE_CARD": 12.95,
    "THREE_SMALL_POTIONS": 12.94,
    "TRANSFORM_CARD": 12.34,
    "THREE_CARDS": 9.76,
    "RANDOM_COLORLESS": 9.66,
    "UPGRADE_CARD": 6.48,
    "HUNDRED_GOLD": 5.03,
    "TEN_PERCENT_HP_BONUS": 3.48,
    "THREE_ENEMY_KILL": 2.07,
    "ONE_RANDOM_RARE_CARD": 1.36,
}


class ModelRequiredDecisionError(RuntimeError):
    """Raised when a model-required trace would otherwise fall back to heuristics."""

    def __init__(
        self,
        *,
        phase: str,
        reason: str,
        legal_action_count: int,
        selector_name: str | None = None,
        checkpoint_path: str | None = None,
        action_kinds: list[str] | None = None,
    ) -> None:
        self.phase = phase
        self.reason = reason
        self.legal_action_count = int(legal_action_count)
        self.selector_name = selector_name
        self.checkpoint_path = checkpoint_path
        self.action_kinds = list(action_kinds or [])
        details = [
            f"phase={phase}",
            f"reason={reason}",
            f"legal_actions={legal_action_count}",
        ]
        if selector_name:
            details.append(f"selector={selector_name}")
        if checkpoint_path:
            details.append(f"checkpoint={checkpoint_path}")
        if self.action_kinds:
            details.append(f"action_kinds={','.join(self.action_kinds)}")
        super().__init__("model-required decision failed: " + " ".join(details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reason": self.reason,
            "legal_action_count": self.legal_action_count,
            "selector_name": self.selector_name,
            "checkpoint_path": self.checkpoint_path,
            "action_kinds": list(self.action_kinds),
        }


def normalize_trace_policy(trace_policy: str | None) -> str:
    token = str(trace_policy or TRACE_POLICY_MODEL_REQUIRED).strip().lower().replace("-", "_")
    if token in {"model", "model_required", "required"}:
        return TRACE_POLICY_MODEL_REQUIRED
    if token in {"legacy", "legacy_fallback", "fallback"}:
        return TRACE_POLICY_LEGACY_FALLBACK
    raise ValueError(f"unsupported trace_policy: {trace_policy}")


def apply_runtime_latency_profile(profile: str | None = None) -> str:
    token = str(profile or os.environ.get("SPIRECOMM_RUNTIME_LATENCY_PROFILE") or "").strip().lower().replace("-", "_")
    if token in {"", "default", "quality", "full"}:
        return ""
    overrides = RUNTIME_LATENCY_PROFILE_ENVS.get(token)
    if overrides is None:
        return token
    for name, value in overrides.items():
        os.environ.setdefault(name, value)
    return token


def source_is_allowed_for_model_required(source: str | None) -> bool:
    token = str(source or "")
    return token in MODELED_ACTION_SOURCES or token in ALLOWED_FORCED_ACTION_SOURCES


def model_selector_status(selectors: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {"torch_available": torch is not None, "selectors": {}}
    for name in MODEL_REQUIRED_SELECTOR_NAMES:
        selector = selectors.get(name)
        status["selectors"][name] = {
            "available": bool(getattr(selector, "available", False)),
            "checkpoint_path": str(getattr(selector, "checkpoint_path", "")) if selector is not None else None,
            "type": type(selector).__name__ if selector is not None else None,
        }
    return status


def validate_model_required_selectors(selectors: dict[str, Any]) -> None:
    if torch is None:
        raise ModelRequiredDecisionError(
            phase="SETUP",
            reason="torch_unavailable_use_spirecomm_rl_python",
            legal_action_count=0,
        )
    for name in MODEL_REQUIRED_SELECTOR_NAMES:
        selector = selectors.get(name)
        if not getattr(selector, "available", False):
            raise ModelRequiredDecisionError(
                phase="SETUP",
                reason="selector_unavailable",
                legal_action_count=0,
                selector_name=name,
                checkpoint_path=str(getattr(selector, "checkpoint_path", "")) if selector is not None else None,
            )


def _reward_actions_by_kind(actions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        grouped.setdefault(str(action.get("kind") or ""), []).append(action)
    return grouped


def _potion_slots_available(env: Any) -> bool:
    max_slots = getattr(env, "max_potion_slots", None)
    if max_slots is not None:
        potions = list(getattr(env, "potions", []) or [])
        if len(potions) < int(max_slots):
            return True
        return any(
            str(
                (potion.get("potion_id") or potion.get("id") or potion.get("name"))
                if isinstance(potion, dict)
                else getattr(potion, "potion_id", "")
            )
            == "Potion Slot"
            for potion in potions
        )
    for potion in list(getattr(env, "potions", []) or []):
        if _potion_id_from_payload(potion) == "Potion Slot":
            return True
        if not isinstance(potion, dict) and not getattr(potion, "can_use", True):
            return True
    return False


def _potion_id_from_payload(potion: Any) -> str:
    if isinstance(potion, dict):
        return str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
    return str(getattr(potion, "potion_id", "") or getattr(potion, "id", "") or getattr(potion, "name", "") or "")


def _potion_priority_from_payload(potion: Any) -> int:
    potion_id = _potion_id_from_payload(potion)
    return potion_priority_value(potion if isinstance(potion, dict) else potion_id)


def _best_full_belt_reward_potion(
    env: Any,
    potion_actions: list[dict[str, Any]],
    discard_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    held = [
        (index, potion)
        for index, potion in enumerate(list(getattr(env, "potions", []) or []))
        if _potion_id_from_payload(potion) != "Potion Slot"
    ]
    if not held:
        return potion_actions[0] if potion_actions else None
    discard_by_index = {
        int(action.get("potion_index")): action
        for action in list(discard_actions or [])
        if action.get("potion_index") is not None
    }
    held_min_index, held_min_potion = min(
        held,
        key=lambda item: (_potion_priority_from_payload(item[1]), item[0]),
    )
    held_min_priority = _potion_priority_from_payload(held_min_potion)
    best_action: dict[str, Any] | None = None
    best_priority = held_min_priority
    for action in potion_actions:
        candidate_priority = potion_priority_value(str(action.get("potion_id") or action.get("name") or ""))
        if candidate_priority > best_priority:
            best_priority = candidate_priority
            best_action = action
    if best_action is not None and discard_by_index:
        discard_action = discard_by_index.get(held_min_index)
        if discard_action is None:
            return None
        discard_action = dict(discard_action)
        discard_action["replace_target_potion_id"] = best_action.get("potion_id") or best_action.get("name")
        return discard_action
    return best_action


def _action_kinds(actions: list[dict[str, Any]]) -> list[str]:
    return sorted({str(action.get("kind") or "") for action in actions})


def _selector_checkpoint(selectors: dict[str, Any], selector_name: str | None) -> str | None:
    if not selector_name:
        return None
    selector = selectors.get(selector_name)
    if selector is None:
        return None
    return str(getattr(selector, "checkpoint_path", "") or "") or None


def _raise_model_required(
    env: Any,
    *,
    reason: str,
    actions: list[dict[str, Any]] | None = None,
    selector_name: str | None = None,
    selectors: dict[str, Any] | None = None,
) -> None:
    legal_actions = list(actions if actions is not None else env.legal_actions())
    raise ModelRequiredDecisionError(
        phase=str(getattr(env, "phase", "")),
        reason=reason,
        legal_action_count=len(legal_actions),
        selector_name=selector_name,
        checkpoint_path=_selector_checkpoint(selectors or {}, selector_name),
        action_kinds=_action_kinds(legal_actions),
    )


def _forced_single_or_raise(
    env: Any,
    actions: list[dict[str, Any]],
    *,
    reason: str,
    selector_name: str | None = None,
    selectors: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[float], str]:
    if len(actions) == 1:
        return actions[0], [], "forced_single"
    _raise_model_required(
        env,
        reason=reason,
        actions=actions,
        selector_name=selector_name,
        selectors=selectors,
    )
    raise AssertionError("unreachable")


def _neow_action_weight(action: dict[str, Any]) -> float:
    bonus = str(action.get("bonus") or "")
    default_weight = NEOW_DEFAULT_BONUS_WEIGHTS.get(bonus)
    if default_weight is not None:
        return float(default_weight)
    if bonus == "BOSS_RELIC":
        return 0.0
    drawback = str(action.get("drawback") or "NONE")
    combo_weight = NEOW_COMBO_WEIGHTS.get((bonus, drawback))
    if combo_weight is not None:
        return float(combo_weight)
    return float(NEOW_BONUS_WEIGHTS.get(bonus, 0.0))


def _stable_neow_policy_random(env: Any, actions: list[dict[str, Any]]) -> float:
    action_key = "|".join(
        f"{action.get('choice_index')}:{action.get('bonus')}:{action.get('drawback')}"
        for action in actions
    )
    material = (
        f"neow_weighted|{getattr(env, 'seed', '')}|"
        f"{getattr(env, 'ascension_level', '')}|{action_key}"
    ).encode("utf-8")
    rng_seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
    return random.Random(rng_seed).random()


def _choose_neow_weighted_action(
    env: Any,
    actions: list[dict[str, Any]] | None = None,
    selectors: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[float], str]:
    legal_actions = list(actions if actions is not None else env.legal_actions())
    if not legal_actions:
        _raise_model_required(env, reason="neow_has_no_legal_actions", actions=legal_actions)
    if len(legal_actions) == 1:
        return legal_actions[0], [], "forced_single"

    fixed_choice_index = os.environ.get("SPIRECOMM_NEOW_FIXED_CHOICE_INDEX")
    if fixed_choice_index is not None:
        try:
            index = int(fixed_choice_index)
        except ValueError:
            index = -1
        for action in legal_actions:
            if int(action.get("choice_index", -1)) == index:
                return action, [], "neow_fixed"
        if 0 <= index < len(legal_actions):
            return legal_actions[index], [], "neow_fixed"

    indexed_actions = [
        action
        for action in legal_actions
        if int(action.get("choice_index", -1)) in NEOW_DEFAULT_CHOICE_INDEXES
    ]
    if indexed_actions:
        legal_actions = indexed_actions

    weights = [_neow_action_weight(action) for action in legal_actions]
    if sum(weights) <= 0.0:
        weights = [1.0 for _ in legal_actions]
    total = float(sum(weights))
    probabilities = [weight / total for weight in weights]

    threshold = _stable_neow_policy_random(env, legal_actions) * total
    cumulative = 0.0
    for action, weight in zip(legal_actions, weights):
        cumulative += weight
        if threshold <= cumulative:
            reranked = _maybe_rerank_neow_rollout(env, legal_actions, probabilities, action, selectors)
            if reranked is not None:
                return reranked
            return action, probabilities, "neow_weighted"
    action = legal_actions[-1]
    reranked = _maybe_rerank_neow_rollout(env, legal_actions, probabilities, action, selectors)
    if reranked is not None:
        return reranked
    return action, probabilities, "neow_weighted"


def _neow_rollout_terminal_score(env: Any, *, max_steps_reached: bool) -> float:
    phase = str(getattr(env, "phase", ""))
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    player = getattr(env, "player", None)
    try:
        hp = int(getattr(player, "current_hp", 0) or 0)
        max_hp = int(getattr(player, "max_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
        max_hp = 0
    if phase in {"COMPLETE", "VICTORY"}:
        return 1_000_000.0 + float(floor) * 1000.0 + float(max(0, hp))
    if phase == "GAME_OVER" or hp <= 0:
        return float(floor) * 1000.0 - 10_000.0
    if max_steps_reached:
        return float(floor) * 1000.0 - 1000.0 + float(max(0, hp)) + float(max(0, max_hp)) * 0.1
    return float(floor) * 1000.0 + float(max(0, hp)) + float(max(0, max_hp)) * 0.1


def _neow_rollout_score_candidate(
    root_blob: bytes,
    action: dict[str, Any],
    selectors: dict[str, Any],
    *,
    max_steps: int,
    max_floor: int,
) -> float:
    previous_flag = os.environ.get("SPIRECOMM_NEOW_IN_ROLLOUT")
    os.environ["SPIRECOMM_NEOW_IN_ROLLOUT"] = "1"
    try:
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        steps = 0
        terminal = {"GAME_OVER", "COMPLETE", "VICTORY"}
        while steps < max_steps:
            phase = str(getattr(branch, "phase", ""))
            try:
                floor = int(getattr(branch, "floor", 0) or 0)
            except (TypeError, ValueError):
                floor = 0
            if phase in terminal or floor > max_floor:
                break
            next_action, _scores, _source = choose_model_required_action(branch, selectors, return_scores=False)
            branch.step(next_action)
            steps += 1
        return _neow_rollout_terminal_score(branch, max_steps_reached=steps >= max_steps)
    except Exception:
        return float("-inf")
    finally:
        if previous_flag is None:
            os.environ.pop("SPIRECOMM_NEOW_IN_ROLLOUT", None)
        else:
            os.environ["SPIRECOMM_NEOW_IN_ROLLOUT"] = previous_flag


def _maybe_rerank_neow_rollout(
    env: Any,
    actions: list[dict[str, Any]],
    scores: list[float],
    chosen_action: dict[str, Any],
    selectors: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_NEOW_ROLLOUT_RERANK", False):
        return None
    if selectors is None or str(os.environ.get("SPIRECOMM_NEOW_IN_ROLLOUT") or "").strip():
        return None
    if len(actions) <= 1:
        return None
    allowed_indexes_raw = str(os.environ.get("SPIRECOMM_NEOW_ROLLOUT_CHOICE_INDEXES", "0,1"))
    allowed_indexes = {
        int(token)
        for token in allowed_indexes_raw.split(",")
        if token.strip().lstrip("-").isdigit()
    }
    candidate_indices = [
        index
        for index, action in enumerate(actions)
        if not allowed_indexes or int(action.get("choice_index", -1)) in allowed_indexes
    ]
    if len(candidate_indices) <= 1:
        return None
    try:
        chosen_index = next(index for index, candidate in enumerate(actions) if candidate is chosen_action or candidate == chosen_action)
    except StopIteration:
        return None
    if chosen_index not in candidate_indices:
        candidate_indices.append(chosen_index)
    try:
        root_blob = clone_env_blob(env, strip_debug_history=True)
    except Exception:
        return None
    max_steps = max(1, _env_int("SPIRECOMM_NEOW_ROLLOUT_MAX_STEPS", 240))
    max_floor_delta = max(1, _env_int("SPIRECOMM_NEOW_ROLLOUT_MAX_FLOOR_DELTA", 6))
    try:
        root_floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        root_floor = 0
    max_floor = root_floor + max_floor_delta
    rollout_scores: dict[int, float] = {}
    for index in sorted(set(candidate_indices)):
        rollout_scores[index] = _neow_rollout_score_candidate(
            root_blob,
            actions[index],
            selectors,
            max_steps=max_steps,
            max_floor=max_floor,
        )
    best_index = max(rollout_scores, key=lambda index: rollout_scores[index])
    best_score = float(rollout_scores[best_index])
    chosen_score = float(rollout_scores.get(chosen_index, float("-inf")))
    min_advantage = _env_float("SPIRECOMM_NEOW_ROLLOUT_MIN_ADVANTAGE", 1000.0)
    if math.isfinite(best_score) and best_index != chosen_index and best_score - chosen_score >= min_advantage:
        return actions[best_index], scores, "neow_rollout_rerank"
    return None


def _choose_run_choice_required(
    env: Any,
    selectors: dict[str, Any],
    selector_name: str,
    source: str,
    *,
    return_scores: bool = True,
):
    actions = env.legal_actions()
    if len(actions) == 1:
        return actions[0], [], "forced_single"
    selector = selectors.get(selector_name)
    if not getattr(selector, "available", False):
        _raise_model_required(
            env,
            reason="selector_unavailable",
            actions=actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    result = selector.choose(env.state(), actions, return_scores=return_scores)
    if result is None:
        _raise_model_required(
            env,
            reason="selector_returned_no_choice",
            actions=actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    index = int(result["choice_index"])
    if not 0 <= index < len(actions):
        _raise_model_required(
            env,
            reason="selector_choice_out_of_range",
            actions=actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    return actions[index], result["scores"], source


def _action_name(action: dict[str, Any]) -> str:
    return str(action.get("name") or action.get("label") or "").strip().lower()


def _campfire_next_map_actions(env: Any) -> list[dict[str, Any]]:
    nodes = getattr(env, "map", None)
    if not nodes:
        return []
    current_node = getattr(env, "current_map_node", None)
    if current_node is None:
        return []
    try:
        current_x, current_y = current_node
        node = nodes[int(current_y)][int(current_x)]
    except Exception:
        return []
    edges = list(getattr(node, "edges", []) or [])
    normal_targets = [
        (int(edge.dst_x), int(edge.dst_y))
        for edge in edges
        if int(getattr(edge, "dst_y", 0)) < len(nodes)
    ]
    if not normal_targets:
        if any(int(getattr(edge, "dst_y", 0)) >= len(nodes) for edge in edges):
            return [{"symbol": "BOSS", "next_symbols": []}]
        return []
    if int(_map_dp_winged_charges(env)) > 0:
        target_y = min(y for _x, y in normal_targets)
        candidates = [
            child
            for child in list(nodes[target_y])
            if getattr(child, "room_symbol", None) is not None and bool(child.has_edges())
        ]
    else:
        candidates = [nodes[y][x] for x, y in sorted(normal_targets, key=lambda item: (item[1], item[0]))]
    actions: list[dict[str, Any]] = []
    for child in candidates:
        next_symbols: list[str] = []
        for edge in list(getattr(child, "edges", []) or []):
            try:
                if int(edge.dst_y) >= len(nodes):
                    next_symbols.append("BOSS")
                else:
                    next_symbols.append(_map_dp_node_symbol(nodes[int(edge.dst_y)][int(edge.dst_x)]))
            except Exception:
                continue
        actions.append({"symbol": _map_dp_node_symbol(child), "next_symbols": next_symbols})
    return actions


def _campfire_path_has_safe_route(actions: list[dict[str, Any]], *, horizon: int, danger_symbols: set[str]) -> bool:
    if not actions:
        return _env_bool("SPIRECOMM_CAMPFIRE_LOW_HP_REST_UNKNOWN_PATH_SAFE", False)
    if int(horizon) <= 1:
        return any(str(action.get("symbol") or "") not in danger_symbols for action in actions)
    for action in actions:
        symbol = str(action.get("symbol") or "")
        if symbol in danger_symbols:
            continue
        next_symbols = [str(token) for token in (action.get("next_symbols") or [])]
        if not next_symbols or any(token not in danger_symbols for token in next_symbols):
            return True
    return False


def _maybe_apply_campfire_low_hp_rest_guard(
    env: Any,
    action: dict[str, Any],
    scores: list[float],
    source: str,
    selectors: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[float], str]:
    reranked = _maybe_rerank_final_boss_campfire_rest(env, action, scores, selectors)
    if reranked is not None:
        return reranked
    if not _env_bool("SPIRECOMM_CAMPFIRE_LOW_HP_REST_GUARD", False):
        return action, scores, source
    if str(action.get("kind") or "") != "campfire":
        return action, scores, source
    guarded_actions = {
        token.strip().lower()
        for token in str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_GUARD_ACTIONS", "smith,dig,lift")).split(",")
        if token.strip()
    }
    if _action_name(action) not in guarded_actions:
        return action, scores, source
    legal_actions = env.legal_actions()
    rest_action = next(
        (
            candidate
            for candidate in legal_actions
            if str(candidate.get("kind") or "") == "campfire" and _action_name(candidate) == "rest"
        ),
        None,
    )
    if rest_action is None:
        return action, scores, source
    player = getattr(env, "player", None)
    try:
        hp = int(getattr(player, "current_hp", 0) or 0)
        max_hp = int(getattr(player, "max_hp", 0) or 0)
    except (TypeError, ValueError):
        return action, scores, source
    if hp <= 0 or max_hp <= 0 or hp >= max_hp:
        return action, scores, source
    ratio_max = _env_float("SPIRECOMM_CAMPFIRE_LOW_HP_REST_RATIO_MAX", 0.60)
    hp_max = _env_int("SPIRECOMM_CAMPFIRE_LOW_HP_REST_HP_MAX", 50)
    if ratio_max > 0.0 and (float(hp) / float(max_hp)) > ratio_max:
        return action, scores, source
    if hp_max > 0 and hp > hp_max:
        return action, scores, source
    floor_max = _env_int("SPIRECOMM_CAMPFIRE_LOW_HP_REST_FLOOR_MAX", 16)
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    if floor_max > 0 and floor > floor_max:
        return action, scores, source
    danger_symbols = {
        token.strip()
        for token in str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_DANGER_SYMBOLS", "E,E_GREEN,BOSS")).split(",")
        if token.strip()
    }
    next_actions = _campfire_next_map_actions(env)
    horizon = max(1, _env_int("SPIRECOMM_CAMPFIRE_LOW_HP_REST_DANGER_HORIZON", 2))
    if _campfire_path_has_safe_route(next_actions, horizon=horizon, danger_symbols=danger_symbols):
        return action, scores, source
    return rest_action, scores, "campfire_low_hp_rest_guard"


def _campfire_rollout_terminal_score(env: Any, *, root_floor: int, max_steps_reached: bool) -> float:
    phase = str(getattr(env, "phase", ""))
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
    if phase in {"COMPLETE", "VICTORY"}:
        return 1_000_000.0 + float(floor) * 1000.0 + float(max(0, hp))
    if phase == "GAME_OVER" or hp <= 0:
        return float(floor) * 1000.0 - 10_000.0
    if floor > root_floor:
        return 100_000.0 + float(floor) * 1000.0 + float(max(0, hp))
    if max_steps_reached:
        return float(floor) * 1000.0 - 1000.0 + float(max(0, hp))
    return float(floor) * 1000.0 + float(max(0, hp))


def _apply_env_overrides(overrides: dict[str, str]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for name, value in overrides.items():
        snapshot[name] = os.environ.get(name)
        os.environ[name] = value
    return snapshot


def _restore_env_overrides(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _campfire_rollout_score_candidate(
    root_blob: bytes,
    action: dict[str, Any],
    selectors: dict[str, Any],
    *,
    root_floor: int,
    max_steps: int,
    max_floor: int,
) -> float:
    previous_flag = os.environ.get("SPIRECOMM_CAMPFIRE_IN_ROLLOUT")
    os.environ["SPIRECOMM_CAMPFIRE_IN_ROLLOUT"] = "1"
    light_policy = _env_bool("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_LIGHT_POLICY", False)
    env_snapshot: dict[str, str | None] = {}
    try:
        if light_policy:
            env_snapshot = _apply_env_overrides(CAMPFIRE_ROLLOUT_LIGHT_POLICY_OVERRIDES)
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        steps = 0
        terminal = {"GAME_OVER", "COMPLETE", "VICTORY"}
        while steps < max_steps:
            phase = str(getattr(branch, "phase", ""))
            try:
                floor = int(getattr(branch, "floor", 0) or 0)
            except (TypeError, ValueError):
                floor = 0
            if phase in terminal or floor > max_floor:
                break
            next_action, _scores, _source = choose_model_required_action(branch, selectors, return_scores=False)
            branch.step(next_action)
            steps += 1
        return _campfire_rollout_terminal_score(
            branch,
            root_floor=root_floor,
            max_steps_reached=steps >= max_steps,
        )
    except Exception:
        return float("-inf")
    finally:
        if light_policy:
            _restore_env_overrides(env_snapshot)
        if previous_flag is None:
            os.environ.pop("SPIRECOMM_CAMPFIRE_IN_ROLLOUT", None)
        else:
            os.environ["SPIRECOMM_CAMPFIRE_IN_ROLLOUT"] = previous_flag


def _maybe_rerank_final_boss_campfire_rest(
    env: Any,
    action: dict[str, Any],
    scores: list[float],
    selectors: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[float], str] | None:
    if selectors is None or str(os.environ.get("SPIRECOMM_CAMPFIRE_IN_ROLLOUT") or "").strip():
        return None
    if str(action.get("kind") or "") != "campfire" or _action_name(action) == "rest":
        return None
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    floor_min = _env_int("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_FLOOR_MIN", 15)
    floor_max = _env_int("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_FLOOR_MAX", 15)
    if floor_min > 0 and floor < floor_min:
        return None
    if floor_max > 0 and floor > floor_max:
        return None
    next_actions = _campfire_next_map_actions(env)
    if not next_actions or any(str(candidate.get("symbol") or "") != "BOSS" for candidate in next_actions):
        return None
    rest_action = next(
        (
            candidate
            for candidate in env.legal_actions()
            if str(candidate.get("kind") or "") == "campfire" and _action_name(candidate) == "rest"
        ),
        None,
    )
    if rest_action is None:
        return None
    player = getattr(env, "player", None)
    try:
        hp = int(getattr(player, "current_hp", 0) or 0)
        max_hp = int(getattr(player, "max_hp", 0) or 0)
    except (TypeError, ValueError):
        return None
    hp_max = _env_int("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_HP_MAX", 50)
    ratio_max = _env_float("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RATIO_MAX", 0.65)
    if hp_max > 0 and hp > hp_max:
        return None
    if max_hp > 0 and ratio_max > 0.0 and float(hp) / float(max_hp) > ratio_max:
        return None
    if _env_bool("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD", True):
        legal_actions = env.legal_actions()
        try:
            chosen_index = next(
                index for index, candidate in enumerate(legal_actions) if candidate is action or candidate == action
            )
            rest_index = next(
                index
                for index, candidate in enumerate(legal_actions)
                if str(candidate.get("kind") or "") == "campfire" and _action_name(candidate) == "rest"
            )
        except StopIteration:
            chosen_index = -1
            rest_index = -1
        if (
            chosen_index >= 0
            and rest_index >= 0
            and scores
            and len(scores) == len(legal_actions)
        ):
            try:
                score_gap = float(scores[chosen_index]) - float(scores[rest_index])
            except (TypeError, ValueError):
                score_gap = float("inf")
            max_gap = _env_float("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD_MAX_SCORE_GAP", 2.5)
            if score_gap <= max_gap:
                return legal_actions[rest_index], scores, "campfire_final_boss_rest_fast_guard"
    if not _env_bool("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RERANK", False):
        return None
    try:
        root_blob = clone_env_blob(env, strip_debug_history=True)
    except Exception:
        return None
    max_steps = max(1, _env_int("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MAX_STEPS", 260))
    max_floor = max(floor + 1, _env_int("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MAX_FLOOR", floor + 1))
    chosen_score = _campfire_rollout_score_candidate(
        root_blob,
        action,
        selectors,
        root_floor=floor,
        max_steps=max_steps,
        max_floor=max_floor,
    )
    rest_score = _campfire_rollout_score_candidate(
        root_blob,
        rest_action,
        selectors,
        root_floor=floor,
        max_steps=max_steps,
        max_floor=max_floor,
    )
    min_advantage = _env_float("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MIN_ADVANTAGE", 50_000.0)
    if math.isfinite(rest_score) and rest_score - chosen_score >= min_advantage:
        return rest_action, scores, "campfire_final_boss_rest_rollout"
    return None


def _score_bias_payload(env_name: str) -> dict[str, float]:
    raw = os.environ.get(env_name)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        normalized = normalize_token(key)
        if not normalized:
            continue
        try:
            result[normalized] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _maybe_apply_boss_relic_score_bias(
    actions: list[dict[str, Any]],
    scores: list[float],
    source: str,
    env: Any | None = None,
) -> tuple[dict[str, Any] | None, list[float], str]:
    if not scores or len(scores) != len(actions):
        return None, scores, source
    raw_bias = os.environ.get("SPIRECOMM_BOSS_RELIC_SCORE_BIAS_JSON")
    biases = dict(DEFAULT_BOSS_RELIC_SCORE_BIAS) if raw_bias is None else _score_bias_payload(
        "SPIRECOMM_BOSS_RELIC_SCORE_BIAS_JSON"
    )
    if not biases:
        return None, scores, source
    raw_best_index = max(range(len(actions)), key=lambda index: float(scores[index]))
    raw_best_token = normalize_token(actions[raw_best_index].get("name") or actions[raw_best_index].get("label") or "")
    skip_raw_trusted = False
    if raw_best_token == "skip" and _BOSS_RELIC_SKIP_RAW_TRUST_MARGIN < float("inf"):
        raw_best_score = float(scores[raw_best_index])
        raw_second_score = max(
            (float(score) for index, score in enumerate(scores) if index != raw_best_index),
            default=float("-inf"),
        )
        if raw_best_score - raw_second_score >= _BOSS_RELIC_SKIP_RAW_TRUST_MARGIN:
            biases = dict(biases)
            biases[raw_best_token] = 0.0
            skip_raw_trusted = True
    if raw_best_token in {"emptycage", "skip"} and env is not None:
        player = getattr(env, "player", None)
        current_hp = int(getattr(player, "current_hp", 0) or 0)
        max_hp = max(1, int(getattr(player, "max_hp", 1) or 1))
        low_hp = (
            current_hp <= _BOSS_RELIC_LOW_HP_RAW_TRUST_HP_MAX
            or (current_hp / max_hp) <= _BOSS_RELIC_LOW_HP_RAW_TRUST_RATIO_MAX
        )
        if low_hp:
            has_energy_alt = any(
                index != raw_best_index
                and normalize_token(action.get("name") or action.get("label") or "") in _BOSS_RELIC_ENERGY_ALT_TOKENS
                for index, action in enumerate(actions)
            )
            if raw_best_token == "emptycage" or (raw_best_token == "skip" and not has_energy_alt):
                biases = dict(biases)
                biases[raw_best_token] = 0.0
    biased_scores = [
        float(score) + float(biases.get(normalize_token(action.get("name") or action.get("label") or ""), 0.0))
        for action, score in zip(actions, scores, strict=False)
    ]
    best_index = max(range(len(actions)), key=lambda index: biased_scores[index])
    if best_index == raw_best_index:
        return None, biased_scores, "boss_relic_skip_raw_trust" if skip_raw_trusted else source
    if skip_raw_trusted and normalize_token(actions[best_index].get("name") or actions[best_index].get("label") or "") == "skip":
        return actions[best_index], biased_scores, "boss_relic_skip_raw_trust"
    return actions[best_index], biased_scores, "boss_relic_score_bias"


def _boss_relic_rollout_terminal_score(env: Any, *, max_steps_reached: bool) -> float:
    phase = str(getattr(env, "phase", ""))
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
    if phase in {"COMPLETE", "VICTORY"}:
        return 1_000_000.0 + float(floor) * 1000.0 + float(max(0, hp))
    if phase == "GAME_OVER" or hp <= 0:
        return float(floor) * 1000.0 - 10_000.0
    if max_steps_reached:
        return float(floor) * 1000.0 - 1000.0 + float(max(0, hp))
    return float(floor) * 1000.0 + float(max(0, hp))


def _boss_relic_rollout_score_candidate(
    root_blob: bytes,
    action: dict[str, Any],
    selectors: dict[str, Any],
    *,
    max_steps: int,
    max_floor: int,
) -> float:
    previous_flag = os.environ.get("SPIRECOMM_BOSS_RELIC_IN_ROLLOUT")
    previous_event_ids = os.environ.get("SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS")
    previous_event_names = os.environ.get("SPIRECOMM_EVENT_ROLLOUT_NAMES")
    event_ids_override = os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_EVENT_IDS_OVERRIDE")
    os.environ["SPIRECOMM_BOSS_RELIC_IN_ROLLOUT"] = "1"
    if event_ids_override is not None:
        os.environ["SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS"] = event_ids_override
        os.environ.pop("SPIRECOMM_EVENT_ROLLOUT_NAMES", None)
    try:
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        steps = 0
        terminal = {"GAME_OVER", "COMPLETE", "VICTORY"}
        while steps < max_steps:
            phase = str(getattr(branch, "phase", ""))
            try:
                floor = int(getattr(branch, "floor", 0) or 0)
            except (TypeError, ValueError):
                floor = 0
            if phase in terminal or floor > max_floor:
                break
            next_action, _scores, _source = choose_model_required_action(branch, selectors, return_scores=False)
            branch.step(next_action)
            steps += 1
        return _boss_relic_rollout_terminal_score(branch, max_steps_reached=steps >= max_steps)
    except Exception:
        return float("-inf")
    finally:
        if previous_flag is None:
            os.environ.pop("SPIRECOMM_BOSS_RELIC_IN_ROLLOUT", None)
        else:
            os.environ["SPIRECOMM_BOSS_RELIC_IN_ROLLOUT"] = previous_flag
        if previous_event_ids is None:
            os.environ.pop("SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS", None)
        else:
            os.environ["SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS"] = previous_event_ids
        if previous_event_names is None:
            os.environ.pop("SPIRECOMM_EVENT_ROLLOUT_NAMES", None)
        else:
            os.environ["SPIRECOMM_EVENT_ROLLOUT_NAMES"] = previous_event_names


def _maybe_rerank_boss_relic_rollout(
    env: Any,
    actions: list[dict[str, Any]],
    scores: list[float],
    chosen_action: dict[str, Any] | None,
    selectors: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_BOSS_RELIC_ROLLOUT_RERANK", True):
        return None
    if selectors is None or str(os.environ.get("SPIRECOMM_BOSS_RELIC_IN_ROLLOUT") or "").strip():
        return None
    if not actions or not scores or len(actions) != len(scores) or chosen_action is None:
        return None
    try:
        root_floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        root_floor = 0
    floor_min = _env_int("SPIRECOMM_BOSS_RELIC_ROLLOUT_FLOOR_MIN", 0)
    floor_max_limit = _env_int("SPIRECOMM_BOSS_RELIC_ROLLOUT_FLOOR_MAX", 0)
    if floor_min > 0 and root_floor < floor_min:
        return None
    if floor_max_limit > 0 and root_floor > floor_max_limit:
        return None
    try:
        chosen_index = next(index for index, candidate in enumerate(actions) if candidate is chosen_action or candidate == chosen_action)
    except StopIteration:
        return None
    topk = max(1, _env_int("SPIRECOMM_BOSS_RELIC_ROLLOUT_TOPK", 3))
    ranked = sorted(range(len(actions)), key=lambda index: float(scores[index]), reverse=True)
    candidate_indices = {chosen_index, *ranked[:topk]}
    if _env_bool("SPIRECOMM_BOSS_RELIC_ROLLOUT_ALL", True):
        candidate_indices.update(range(len(actions)))
    max_model_score_drop = _env_float("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_MODEL_SCORE_DROP", float("inf"))
    if math.isfinite(max_model_score_drop):
        try:
            chosen_model_score = float(scores[chosen_index])
        except (TypeError, ValueError, IndexError):
            chosen_model_score = float("-inf")
        candidate_indices = {
            index
            for index in candidate_indices
            if index == chosen_index or float(scores[index]) >= chosen_model_score - max_model_score_drop
        }
    if len(candidate_indices) <= 1:
        return None
    try:
        root_blob = clone_env_blob(env, strip_debug_history=True)
    except Exception:
        return None
    max_steps = max(1, _env_int("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_STEPS", 320))
    max_floor_delta = max(1, _env_int("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_FLOOR_DELTA", 5))
    max_floor = root_floor + max_floor_delta
    rollout_scores: dict[int, float] = {}
    for index in sorted(candidate_indices):
        rollout_scores[index] = _boss_relic_rollout_score_candidate(
            root_blob,
            actions[index],
            selectors,
            max_steps=max_steps,
            max_floor=max_floor,
        )
    best_index = max(rollout_scores, key=lambda index: rollout_scores[index])
    snecko_energy_gap = _env_float("SPIRECOMM_BOSS_RELIC_ROLLOUT_SNECKO_ENERGY_MODEL_GAP", float("inf"))
    if math.isfinite(snecko_energy_gap) and len(rollout_scores) > 1:
        best_action = actions[best_index]
        best_token = normalize_token(str(best_action.get("relic_id") or best_action.get("name") or ""))
        if best_token == "sneckoeye":
            snecko_energy_min_score_lead = _env_float(
                "SPIRECOMM_BOSS_RELIC_ROLLOUT_SNECKO_ENERGY_MIN_SCORE_LEAD",
                0.0,
            )
            try:
                snecko_model_score = float(scores[best_index])
            except (TypeError, ValueError, IndexError):
                snecko_model_score = float("inf")
            try:
                top_model_score = max(float(score) for score in scores)
            except (TypeError, ValueError):
                top_model_score = float("inf")
            energy_model_close = False
            for index in candidate_indices:
                action = actions[index]
                token = normalize_token(str(action.get("relic_id") or action.get("name") or ""))
                if token not in _BOSS_RELIC_ENERGY_ALT_TOKENS:
                    continue
                try:
                    model_score = float(scores[index])
                except (TypeError, ValueError, IndexError):
                    continue
                if (
                    model_score >= top_model_score - snecko_energy_gap
                    and model_score >= snecko_model_score + snecko_energy_min_score_lead
                ):
                    energy_model_close = True
                    break
            if energy_model_close:
                filtered_rollout_scores = {
                    index: score for index, score in rollout_scores.items() if index != best_index
                }
                if filtered_rollout_scores:
                    best_index = max(filtered_rollout_scores, key=lambda index: filtered_rollout_scores[index])
                    rollout_scores = filtered_rollout_scores
    best_score = float(rollout_scores[best_index])
    chosen_score = float(rollout_scores.get(chosen_index, float("-inf")))
    min_advantage = _env_float("SPIRECOMM_BOSS_RELIC_ROLLOUT_MIN_ADVANTAGE", 1000.0)
    if math.isfinite(best_score) and best_index != chosen_index and best_score - chosen_score >= min_advantage:
        return actions[best_index], scores, "boss_relic_rollout_rerank"
    return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _shop_value_policy_enabled() -> bool:
    token = str(os.environ.get("SPIRECOMM_SHOP_POLICY", "value")).strip().lower().replace("_", "-")
    return token in {"value", "value-policy", "gold-reserve", "heuristic"}


def _shop_action_price(action: dict[str, Any]) -> int:
    try:
        return max(0, int(action.get("price") or 0))
    except (TypeError, ValueError):
        return 0


def _shop_leave_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        (action for action in actions if str(action.get("item_kind") or "").lower() == "leave"),
        actions[-1] if actions else {"kind": "shop", "item_kind": "leave", "item_id": "leave", "name": "Leave", "price": 0},
    )


def _shop_card_for_action(env: Any, action: dict[str, Any]) -> dict[str, Any] | None:
    shop = getattr(env, "current_shop", None)
    if shop is None:
        return None
    try:
        index = int(action.get("shop_index"))
    except (TypeError, ValueError):
        return None
    cards = list(getattr(shop, "cards", []) or [])
    if 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _shop_card_key(card: dict[str, Any]) -> str:
    return normalize_token(card.get("card_id") or card.get("id") or card.get("name"))


def _shop_act1_frontload_multiplier(env: Any) -> float:
    if not _env_bool("SPIRECOMM_SHOP_ACT1_FRONTLOAD_BIAS_ENABLED", True):
        return 0.0
    try:
        act = int(getattr(env, "act", 0) or 0)
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    floor_min = _env_int("SPIRECOMM_SHOP_ACT1_FRONTLOAD_FLOOR_MIN", 3)
    floor_max = _env_int("SPIRECOMM_SHOP_ACT1_FRONTLOAD_FLOOR_MAX", 6)
    if act != 1 or (floor_min > 0 and floor < floor_min) or (floor_max > 0 and floor > floor_max):
        return 0.0
    if _env_bool("SPIRECOMM_SHOP_ACT1_FRONTLOAD_REQUIRE_NO_ATTACK", True):
        for card in list(getattr(env, "deck", []) or []):
            if isinstance(card, dict) and _shop_card_key(card) in SHOP_ACT1_FRONTLOAD_CARD_KEYS:
                return 0.0
    try:
        return float(early_card_reward_frontload_deficit_multiplier(env))
    except Exception:
        return 0.0


def _shop_act1_frontload_card_bonus(env: Any, card: dict[str, Any]) -> float:
    multiplier = _shop_act1_frontload_multiplier(env)
    if multiplier <= 0.0:
        return 0.0
    card_key = _shop_card_key(card)
    if card_key not in SHOP_ACT1_FRONTLOAD_CARD_KEYS:
        return 0.0
    return _env_float("SPIRECOMM_SHOP_ACT1_FRONTLOAD_CARD_BIAS", 1.0) * multiplier


def _shop_is_removable_card(card: dict[str, Any]) -> bool:
    if bool(card.get("bottled") or card.get("in_bottle_flame") or card.get("in_bottle_lightning") or card.get("in_bottle_tornado")):
        return False
    if str(card.get("card_id") or "") in SHOP_VALUE_POLICY_UNREMOVABLE_CURSES:
        return False
    return True


def _shop_purge_target_values(env: Any) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for index, card in enumerate(list(getattr(env, "deck", []) or [])):
        if not isinstance(card, dict) or not _shop_is_removable_card(card):
            continue
        card_id = str(card.get("card_id") or card.get("id") or "")
        card_type = str(card.get("type") or "")
        value = 0.0
        if card_type == "CURSE":
            value = _env_float("SPIRECOMM_SHOP_VALUE_CURSE_PURGE_VALUE", 35.0)
        elif card_id == "Strike_R":
            value = _env_float("SPIRECOMM_SHOP_VALUE_STRIKE_PURGE_VALUE", 12.0)
        elif card_id == "Defend_R":
            value = _env_float("SPIRECOMM_SHOP_VALUE_DEFEND_PURGE_VALUE", 10.0)
        if value > 0.0:
            values.append((index, value))
    return values


def _shop_has_removable_curse(env: Any) -> bool:
    for _, value in _shop_purge_target_values(env):
        if value >= _env_float("SPIRECOMM_SHOP_VALUE_CURSE_PURGE_VALUE", 35.0):
            return True
    return False


def _shop_removable_starter_basic_count(env: Any) -> int:
    count = 0
    for card in list(getattr(env, "deck", []) or []):
        if not isinstance(card, dict) or not _shop_is_removable_card(card):
            continue
        if str(card.get("card_id") or card.get("id") or "") in {"Strike_R", "Defend_R"}:
            count += 1
    return count


def _shop_current_purge_cost(env: Any, actions: list[dict[str, Any]]) -> int:
    purge_action = next((action for action in actions if str(action.get("item_kind") or "").lower() == "purge"), None)
    if purge_action is not None:
        return _shop_action_price(purge_action)
    shop = getattr(env, "current_shop", None)
    try:
        return max(0, int(getattr(shop, "purge_cost", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _shop_future_shop_within(env: Any, horizon: int) -> bool:
    nodes = getattr(env, "map", None)
    current = getattr(env, "current_map_node", None)
    if not nodes or current is None or int(horizon) <= 0:
        return False
    try:
        start_x, start_y = int(current[0]), int(current[1])
        start = nodes[start_y][start_x]
    except Exception:
        return False
    queue: list[tuple[Any, int]] = [(start, 0)]
    seen = {(start_x, start_y)}
    while queue:
        node, distance = queue.pop(0)
        if distance >= int(horizon):
            continue
        for edge in list(getattr(node, "edges", []) or []):
            try:
                dst_y = int(edge.dst_y)
                dst_x = int(edge.dst_x)
                child = nodes[dst_y][dst_x]
            except Exception:
                continue
            key = (dst_x, dst_y)
            if key in seen:
                continue
            seen.add(key)
            next_distance = distance + 1
            if str(getattr(child, "room_symbol", "") or "") == "$":
                return True
            queue.append((child, next_distance))
    return False


def _shop_future_shop_reserve_danger_multiplier(env: Any, horizon: int) -> float:
    if not _env_bool("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_GATE", False):
        return 1.0
    danger_horizon = _env_int("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_HORIZON", int(horizon))
    if danger_horizon <= 0:
        danger_horizon = int(horizon)
    if not _map_forced_symbol_within(env, {"E", "E_GREEN", "BOSS"}, horizon=danger_horizon, safe_stop_symbols={"R", "$"}):
        return 1.0
    return max(0.0, _env_float("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_MULTIPLIER", 0.0))


def _shop_reserve_gold(env: Any, actions: list[dict[str, Any]]) -> int:
    purge_cost = _shop_current_purge_cost(env, actions)
    purge_reserve = 0
    if purge_cost > 0 and _shop_has_removable_curse(env):
        purge_reserve = purge_cost
    else:
        starter_count = _shop_removable_starter_basic_count(env)
        if starter_count >= 5:
            purge_reserve = round(0.75 * purge_cost)
        elif starter_count >= 3:
            purge_reserve = round(0.50 * purge_cost)
    horizon = _env_int("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", 5)
    future_reserve = _env_int("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", 120) if _shop_future_shop_within(env, horizon) else 0
    if future_reserve > 0:
        future_reserve = int(round(float(future_reserve) * _shop_future_shop_reserve_danger_multiplier(env, horizon)))
    return max(int(purge_reserve), int(future_reserve))


def _shop_spend_cost(env: Any, action: dict[str, Any], reserve_gold: int) -> float:
    price = _shop_action_price(action)
    remaining_gold = int(getattr(env, "gold", 0) or 0) - price
    reserve_shortfall = max(0, int(reserve_gold) - remaining_gold)
    return (
        price * _env_float("SPIRECOMM_SHOP_VALUE_PRICE_COST", 0.044348003822393976)
        + reserve_shortfall * _env_float("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", 0.043490245962190935)
    )


def _shop_model_advantages(env: Any, selector: Any, actions: list[dict[str, Any]]) -> dict[int, float]:
    if not getattr(selector, "available", False):
        return {}
    result = selector.choose(env.state(), actions, return_scores=True)
    if result is None:
        return {}
    scores = [float(value) for value in list(result.get("scores") or [])]
    if len(scores) != len(actions):
        return {}
    leave_index = next((index for index, action in enumerate(actions) if str(action.get("item_kind") or "").lower() == "leave"), None)
    leave_score = scores[leave_index] if leave_index is not None else 0.0
    return {index: score - leave_score for index, score in enumerate(scores)}


def _shop_card_values(env: Any, selector: CardRewardSelector | None, actions: list[dict[str, Any]]) -> dict[int, float]:
    card_items: list[tuple[int, dict[str, Any]]] = []
    for index, action in enumerate(actions):
        if str(action.get("item_kind") or "").lower() != "card":
            continue
        card = _shop_card_for_action(env, action)
        if card is not None:
            card_items.append((index, card))
    if not card_items or not getattr(selector, "available", False):
        return {}
    result = selector.choose(
        env.state(),
        [card for _, card in card_items],
        can_skip=True,
        return_scores=True,
    )
    if result is None:
        return {}
    scores = [float(value) for value in list(result.get("scores") or [])]
    if len(scores) < len(card_items):
        return {}
    baseline = scores[-1] if len(scores) == len(card_items) + 1 and _env_bool("SPIRECOMM_SHOP_VALUE_CARD_SKIP_BASELINE", True) else 0.0
    scale = _env_float("SPIRECOMM_SHOP_VALUE_CARD_SCALE", 4.6262945279949435)
    reference_price = max(1.0, _env_float("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", 60.0))
    min_factor = _env_float("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", 0.65)
    max_factor = _env_float("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", 1.35)
    values: dict[int, float] = {}
    for position, (index, card) in enumerate(card_items):
        price = max(1, _shop_action_price(actions[index]))
        price_factor = math.sqrt(reference_price / float(price))
        price_factor = max(min_factor, min(max_factor, price_factor))
        values[index] = (scores[position] - baseline) * scale * price_factor
    return values


def _shop_model_item_scale(item_kind: str) -> float:
    normalized = str(item_kind or "").strip().lower()
    if normalized == "potion":
        return _env_float("SPIRECOMM_SHOP_VALUE_POTION_SCALE", 0.5084989138155764)
    if normalized == "relic":
        return _env_float("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", 0.8)
    return _env_float("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", 1.0)


def _shop_value_item_bias(action: dict[str, Any]) -> float:
    raw = os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_BIAS_JSON")
    if raw is None or not str(raw).strip():
        return 0.0
    try:
        payload = json.loads(str(raw))
    except Exception:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    item_kind = str(action.get("item_kind") or "").strip()
    keys = {
        str(action.get("item_id") or ""),
        str(action.get("potion_id") or ""),
        str(action.get("card_id") or ""),
        str(action.get("name") or ""),
        str(action.get("label") or ""),
    }
    keys.update({normalize_token(key) for key in list(keys) if key})
    if item_kind:
        keys.update({f"{item_kind}:{key}" for key in list(keys) if key})
        keys.update({f"{item_kind.lower()}:{key}" for key in list(keys) if key})
    for key in keys:
        if not key:
            continue
        if key not in payload:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _shop_value_forced_danger_item_bias(env: Any, action: dict[str, Any]) -> float:
    raw = os.environ.get("SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_JSON")
    if raw is None or not str(raw).strip():
        return 0.0
    horizon = max(1, _env_int("SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_HORIZON", 2))
    if not _map_forced_symbol_within(env, {"E", "E_GREEN", "BOSS"}, horizon=horizon, safe_stop_symbols={"R"}):
        return 0.0
    try:
        payload = json.loads(str(raw))
    except Exception:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    item_kind = str(action.get("item_kind") or "").strip()
    keys = {
        str(action.get("item_id") or ""),
        str(action.get("potion_id") or ""),
        str(action.get("card_id") or ""),
        str(action.get("name") or ""),
        str(action.get("label") or ""),
    }
    keys.update({normalize_token(key) for key in list(keys) if key})
    if item_kind:
        keys.update({f"{item_kind}:{key}" for key in list(keys) if key})
        keys.update({f"{item_kind.lower()}:{key}" for key in list(keys) if key})
    for key in keys:
        if not key or key not in payload:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _shop_value_conditional_item_bias(env: Any, action: dict[str, Any]) -> float:
    forced_danger_bias = _shop_value_forced_danger_item_bias(env, action)
    item_kind = str(action.get("item_kind") or "").strip().lower()
    if item_kind != "potion":
        return forced_danger_bias
    item_ids = {
        str(action.get("item_id") or ""),
        str(action.get("potion_id") or ""),
        str(action.get("name") or ""),
        str(action.get("label") or ""),
    }
    normalized_ids = {normalize_token(value) for value in item_ids if value}
    is_fairy = bool({"fairypotion", "fairyinabottle"}.intersection(normalized_ids))
    if not is_fairy:
        return forced_danger_bias
    bias = _env_float("SPIRECOMM_SHOP_VALUE_FAIRY_FORCED_ELITE_BIAS", 6.0)
    if bias == 0.0:
        return forced_danger_bias
    horizon = max(1, _env_int("SPIRECOMM_SHOP_VALUE_FAIRY_FORCED_ELITE_HORIZON", 2))
    if not _map_forced_symbol_within(env, {"E", "E_GREEN", "BOSS"}, horizon=horizon, safe_stop_symbols={"R"}):
        return forced_danger_bias
    return forced_danger_bias + float(bias)


def _choose_shop_value_policy_required(
    env: Any,
    selectors: dict[str, Any],
    *,
    return_scores: bool = True,
) -> tuple[dict[str, Any], list[float], str]:
    actions = env.legal_actions()
    if len(actions) == 1:
        return actions[0], [], "forced_single"
    leave_action = _shop_leave_action(actions)
    reserve_gold = _shop_reserve_gold(env, actions)
    model_advantages = _shop_model_advantages(env, selectors.get("shop"), actions)
    card_values = _shop_card_values(env, selectors.get("card_reward"), actions)
    purge_targets = _shop_purge_target_values(env)
    best_purge_target = max(purge_targets, key=lambda item: item[1]) if purge_targets else None
    threshold = _env_float("SPIRECOMM_SHOP_VALUE_THRESHOLD", 0.0)

    base_policy_scores: list[float] = []
    for index, action in enumerate(actions):
        item_kind = str(action.get("item_kind") or "").lower()
        if item_kind == "leave":
            base_policy_scores.append(0.0)
            continue
        item_value = 0.0
        if item_kind == "card":
            item_value = card_values.get(
                index,
                model_advantages.get(index, 0.0) * _shop_model_item_scale(item_kind),
            )
        elif item_kind == "purge":
            item_value = best_purge_target[1] if best_purge_target is not None else 0.0
            if best_purge_target is not None and not _shop_has_removable_curse(env):
                item_value -= _env_float("SPIRECOMM_SHOP_ACT1_FRONTLOAD_PURGE_PENALTY", 0.0) * _shop_act1_frontload_multiplier(env)
        else:
            item_value = model_advantages.get(index, 0.0) * _shop_model_item_scale(item_kind)
        score = float(item_value) - _shop_spend_cost(env, action, reserve_gold)
        score += _shop_value_item_bias(action)
        score += _shop_value_conditional_item_bias(env, action)
        base_policy_scores.append(score)

    best_base_score = max(base_policy_scores) if base_policy_scores else float("-inf")
    best_action: dict[str, Any] | None = None
    best_score = float("-inf")
    policy_scores: list[float] = []
    for action, base_score in zip(actions, base_policy_scores):
        item_kind = str(action.get("item_kind") or "").lower()
        score = float(base_score)
        if item_kind == "card":
            card = _shop_card_for_action(env, action)
            if card is not None:
                frontload_bonus = _shop_act1_frontload_card_bonus(env, card)
                if frontload_bonus != 0.0:
                    min_pre_bonus_score = _env_float("SPIRECOMM_SHOP_ACT1_FRONTLOAD_MIN_PRE_BONUS_SCORE", 0.0)
                    max_pre_bonus_gap = _env_float("SPIRECOMM_SHOP_ACT1_FRONTLOAD_MAX_PRE_BONUS_GAP", 0.4)
                    if score > min_pre_bonus_score and score >= best_base_score - max_pre_bonus_gap:
                        score += frontload_bonus
        policy_scores.append(score)
        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None or best_score <= threshold:
        return leave_action, policy_scores if return_scores else [], "shop_value_leave"
    chosen = dict(best_action)
    if str(chosen.get("item_kind") or "").lower() == "purge" and best_purge_target is not None:
        chosen["target_index"] = int(best_purge_target[0])
    return chosen, policy_scores if return_scores else [], "shop_value"


def _event_key_from_actions(actions: list[dict[str, Any]]) -> str:
    for action in actions:
        event_id = str(action.get("event_id") or action.get("event_name") or "")
        if event_id:
            return normalize_token(event_id)
    return ""


def _event_label_key(action: dict[str, Any]) -> str:
    event_id = str(action.get("event_id") or action.get("event_name") or "")
    label = action.get("model_label") or action.get("label") or action.get("key") or action.get("text") or action.get("choice") or action.get("name") or ""
    return normalize_token(canonical_event_model_label(event_id, label))


def _match_and_keep_card_scores(env: Any, selector: Any, cards: list[dict[str, Any]]) -> dict[str, float]:
    if not getattr(selector, "available", False) or not cards:
        return {}
    try:
        result = selector.choose(env.state(), cards, can_skip=True, return_scores=True)
    except Exception:
        return {}
    if result is None:
        return {}
    scores = list(result.get("scores") or [])
    if len(scores) != len(cards) + 1:
        return {}
    skip_score = float(scores[-1])
    margins: dict[str, float] = {}
    for card, score in zip(cards, scores[:-1]):
        key = normalize_token(card.get("card_id") or card.get("id") or card.get("name"))
        if key:
            margins[key] = float(score) - skip_score
    return margins


def _choose_match_and_keep_action(env: Any, actions: list[dict[str, Any]], selector: Any | None = None) -> dict[str, Any]:
    event = getattr(env, "current_event", None)
    data = getattr(event, "data", {}) if event is not None else {}
    cards = list(data.get("cards") or [])
    revealed = {int(index) for index in list(data.get("revealed_card_indexes") or [])}
    removed = {int(index) for index in list(data.get("removed_card_indexes") or [])}
    first_index = data.get("first_card_index")
    use_card_scores = _env_bool("SPIRECOMM_MATCH_KEEP_CARD_SCORE_POLICY", False)
    min_margin = _env_float("SPIRECOMM_MATCH_KEEP_MIN_CARD_SKIP_MARGIN", 1.5)

    def _card_id(index: int) -> str:
        if 0 <= int(index) < len(cards):
            card = cards[int(index)]
            if isinstance(card, dict):
                return str(card.get("card_id") or card.get("id") or card.get("name") or "")
        return ""

    def _is_gold_card(index: int) -> bool:
        if 0 <= int(index) < len(cards):
            card = cards[int(index)]
            if isinstance(card, dict):
                return str(card.get("rarity") or "").upper() == "RARE"
        return False

    def _card_payload(index: int) -> dict[str, Any]:
        if 0 <= int(index) < len(cards) and isinstance(cards[int(index)], dict):
            return dict(cards[int(index)])
        return {}

    action_by_index = {
        int(action.get("match_card_index")): action
        for action in actions
        if action.get("match_card_index") is not None
    }
    representative_cards = []
    seen_card_ids: set[str] = set()
    for index in sorted(set(revealed) | ({int(first_index)} if first_index is not None else set())):
        card_id = _card_id(index)
        key = normalize_token(card_id)
        if not key or key in seen_card_ids:
            continue
        payload = _card_payload(index)
        if payload:
            representative_cards.append(payload)
            seen_card_ids.add(key)
    card_margins = (
        _match_and_keep_card_scores(env, selector, representative_cards)
        if use_card_scores and representative_cards
        else {}
    )

    def _is_worth_matching(index: int) -> bool:
        card_id = _card_id(index)
        key = normalize_token(card_id)
        if not key:
            return False
        card_type = str(_card_payload(index).get("type") or "").upper()
        if card_type in {"CURSE", "STATUS"}:
            return False
        if card_margins:
            return float(card_margins.get(key, float("-inf"))) >= min_margin
        return _is_gold_card(index)

    if first_index is not None:
        first = int(first_index)
        target_id = _card_id(first)
        for index in sorted(revealed):
            if (
                index != first
                and index not in removed
                and _card_id(index) == target_id
                and _is_worth_matching(first)
                and index in action_by_index
            ):
                return action_by_index[index]
        for index, action in sorted(action_by_index.items()):
            if index not in revealed:
                return action
        return actions[0]

    revealed_by_card: dict[str, list[int]] = {}
    for index in sorted(revealed):
        if index in removed:
            continue
        card_id = _card_id(index)
        if card_id and index in action_by_index:
            revealed_by_card.setdefault(card_id, []).append(index)
    scored_pairs: list[tuple[float, int]] = []
    for indexes in revealed_by_card.values():
        if len(indexes) >= 2 and _is_worth_matching(indexes[0]):
            key = normalize_token(_card_id(indexes[0]))
            score = float(card_margins.get(key, 1.0 if _is_gold_card(indexes[0]) else 0.0))
            scored_pairs.append((score, indexes[0]))
    if scored_pairs:
        _, index = max(scored_pairs, key=lambda item: (item[0], -item[1]))
        return action_by_index[index]

    for index, action in sorted(action_by_index.items()):
        if index not in revealed:
            return action
    return actions[0]


def _choose_note_for_yourself_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
    # The A20 SlayTheData event model has no NoteForYourself labels. Prefer the
    # non-mutating option rather than asking the model to score unseen tokens.
    for action in actions:
        if _event_label_key(action) in {"leave", "ignored"}:
            return action
    return actions[-1]


def _choose_golden_idol_boulder_guard(
    env: Any,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_EVENT_GOLDEN_IDOL_MAX_HP_OVER_DAMAGE", False):
        return None
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    floor_max = _env_int("SPIRECOMM_EVENT_GOLDEN_IDOL_MAX_HP_OVER_DAMAGE_FLOOR_MAX", 16)
    if floor_max > 0 and floor > floor_max:
        return None
    damage_action = next((action for action in actions if "damage" in _event_label_key(action)), None)
    max_hp_action = next((action for action in actions if "maxhp" in _event_label_key(action)), None)
    if damage_action is None or max_hp_action is None:
        return None
    return max_hp_action, [], "event_golden_idol_max_hp_over_damage"


def _map_forced_symbol_within(
    env: Any,
    target_symbols: set[str],
    *,
    horizon: int,
    safe_stop_symbols: set[str] | None = None,
) -> bool:
    nodes = getattr(env, "map", None)
    current = getattr(env, "current_map_node", None)
    if not nodes or current is None or horizon <= 0:
        return False
    safe_symbols = safe_stop_symbols or set()
    try:
        current_x, current_y = current
        current_node = nodes[int(current_y)][int(current_x)]
    except Exception:
        return False

    memo: dict[tuple[int, int, int], bool] = {}

    def _forced_from(node: Any, depth: int) -> bool:
        if depth <= 0:
            return False
        key = (int(getattr(node, "x", -1)), int(getattr(node, "y", -1)), int(depth))
        if key in memo:
            return memo[key]
        branches: list[bool] = []
        for edge in list(getattr(node, "edges", []) or []):
            try:
                child = nodes[int(edge.dst_y)][int(edge.dst_x)]
            except Exception:
                continue
            symbol = _map_dp_node_symbol(child)
            if symbol in safe_symbols:
                branches.append(False)
            elif symbol in target_symbols:
                branches.append(True)
            else:
                branches.append(_forced_from(child, depth - 1))
        result = bool(branches) and all(branches)
        memo[key] = result
        return result

    return _forced_from(current_node, int(horizon))


def _choose_golden_idol_intro_forced_elite_guard(
    env: Any,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_GUARD", False):
        return None
    take_action = next((action for action in actions if _event_label_key(action) == "takegoldenidol"), None)
    leave_action = next((action for action in actions if _event_label_key(action) in {"ignore", "ignored", "leave"}), None)
    if take_action is None or leave_action is None:
        return None
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    floor_max = _env_int("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_FLOOR_MAX", 6)
    if floor_max > 0 and floor > floor_max:
        return None
    horizon = max(1, _env_int("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_HORIZON", 4))
    if not _map_forced_symbol_within(env, {"E", "E_GREEN"}, horizon=horizon, safe_stop_symbols={"R"}):
        return None
    try:
        current_hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
        max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
    except (TypeError, ValueError):
        return None
    if current_hp <= 0 or max_hp <= 0:
        return None
    try:
        ascension = int(getattr(env, "ascension_level", 0) or 0)
    except (TypeError, ValueError):
        ascension = 0
    projected_hp = max(0, current_hp - int(max_hp * (0.35 if ascension >= 15 else 0.25)))
    min_projected_hp = _env_int("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_PROJECTED_HP_MIN", 56)
    min_projected_ratio = _env_float("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_PROJECTED_HP_RATIO_MIN", 0.70)
    if projected_hp >= min_projected_hp and float(projected_hp) / float(max_hp) >= min_projected_ratio:
        return None
    return leave_action, [], "event_golden_idol_skip_forced_elite"


def _scrap_ooze_next_open_damage(action: dict[str, Any]) -> int:
    match = re.search(r"(\d+)\s*hp", str(action.get("label") or action.get("name") or ""), re.IGNORECASE)
    if match is None:
        return 0
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return 0


def _choose_scrap_ooze_lowhp_cap(
    env: Any,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_CAP", False):
        return None
    open_action = next((action for action in actions if _event_label_key(action).startswith("open")), None)
    leave_action = next((action for action in actions if _event_label_key(action) in {"leave", "ignored", "ignore"}), None)
    if open_action is None or leave_action is None:
        return None
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
        max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
    except (TypeError, ValueError):
        return None
    if hp <= 0 or max_hp <= 0:
        return None
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    floor_max = _env_int("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_FLOOR_MAX", 16)
    if floor_max > 0 and floor > floor_max:
        return None
    hp_max = _env_int("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_MAX", 50)
    ratio_max = _env_float("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_RATIO_MAX", 0.65)
    damage = _scrap_ooze_next_open_damage(open_action)
    after_hp = hp - damage
    flat_match = hp_max > 0 and (hp <= hp_max or after_hp <= hp_max)
    ratio_match = ratio_max > 0.0 and (
        (float(hp) / float(max_hp)) <= ratio_max or (float(after_hp) / float(max_hp)) <= ratio_max
    )
    if flat_match or ratio_match:
        return leave_action, [], "event_scrap_ooze_lowhp_cap"
    return None


def _event_action_hp_cost(action: dict[str, Any]) -> int:
    label = str(action.get("label") or action.get("name") or "")
    if "max hp" in label.lower():
        return 0
    match = re.search(r"(\d+)\s*(?:hp|damage)", label, re.IGNORECASE)
    if match is None:
        return 0
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return 0


def _choose_event_lowhp_cost_guard(
    env: Any,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_EVENT_LOW_HP_COST_GUARD", False):
        return None
    safe_action = next(
        (
            action
            for action in actions
            if _event_label_key(action)
            in {"leave", "ignore", "ignored", "skip", "refuse", "stop", "stopped", "fled"}
        ),
        None,
    )
    if safe_action is None:
        return None
    max_cost = max((_event_action_hp_cost(action) for action in actions), default=0)
    if max_cost <= 0:
        return None
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        return None
    if hp <= 0:
        return None
    min_after = _env_int("SPIRECOMM_EVENT_LOW_HP_COST_MIN_AFTER", 25)
    if hp - max_cost <= min_after:
        return safe_action, [], "event_lowhp_cost_guard"
    return None


def _choose_knowing_skull_leave_guard(
    env: Any,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[float], str] | None:
    leave_action = next((action for action in actions if _event_label_key(action) in {"leave", "ignore", "ignored"}), None)
    if leave_action is None:
        return None
    non_leave_actions = [action for action in actions if action is not leave_action]
    if not non_leave_actions:
        return None
    current_event = getattr(env, "current_event", None)
    data = getattr(current_event, "data", {}) if current_event is not None else {}
    already_paid = False
    if isinstance(data, dict):
        already_paid = any(
            int(data.get(key) or 6) > 6
            for key in ("potion_cost", "gold_cost", "card_cost")
        )
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
        max_hp = max(1, int(getattr(getattr(env, "player", None), "max_hp", 1) or 1))
    except (TypeError, ValueError):
        hp, max_hp = 0, 1
    hp_max = _env_int("SPIRECOMM_KNOWING_SKULL_LEAVE_HP_MAX", 24)
    ratio_max = _env_float("SPIRECOMM_KNOWING_SKULL_LEAVE_HP_RATIO_MAX", 0.35)
    low_hp = hp_max > 0 and hp <= hp_max
    low_ratio = ratio_max > 0.0 and (float(hp) / float(max_hp)) <= ratio_max
    if already_paid or low_hp or low_ratio:
        return leave_action, [], "event_knowing_skull_leave_guard"
    return None


def _event_action_by_label(actions: list[dict[str, Any]], *labels: str) -> dict[str, Any] | None:
    wanted = {normalize_token(label) for label in labels if normalize_token(label)}
    if not wanted:
        return None
    for action in actions:
        if _event_label_key(action) in wanted:
            return action
    return None


def _choose_event_fast_policy(
    env: Any,
    actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_EVENT_FAST_POLICY", True):
        return None
    event_key = _event_key_from_actions(actions)
    if event_key == "bigfish" and _env_bool("SPIRECOMM_EVENT_FAST_BIG_FISH", True):
        banana = _event_action_by_label(actions, "Banana")
        donut = _event_action_by_label(actions, "Donut")
        if banana is None or donut is None:
            return None
        try:
            hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
            max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
        except (TypeError, ValueError):
            return None
        if hp <= 0 or max_hp <= 0:
            return None
        missing_hp = max(0, max_hp - hp)
        banana_missing_min = max(0, _env_int("SPIRECOMM_EVENT_FAST_BIG_FISH_BANANA_MISSING_MIN", 10))
        banana_ratio_max = _env_float("SPIRECOMM_EVENT_FAST_BIG_FISH_BANANA_RATIO_MAX", 0.70)
        low_ratio = banana_ratio_max > 0.0 and (float(hp) / float(max_hp)) <= banana_ratio_max
        if missing_hp >= banana_missing_min or low_ratio:
            return banana, [], "event_fast_big_fish"
        return None
    if event_key == "ghosts" and _env_bool("SPIRECOMM_EVENT_FAST_GHOSTS_ACCEPT", True):
        accept = next(
            (
                action
                for action in actions
                if _event_label_key(action) in {"becameaghost", "accept", "accepted"}
            ),
            None,
        )
        leave = next(
            (
                action
                for action in actions
                if _event_label_key(action) in {"ignored", "leave", "refuse", "refused"}
            ),
            None,
        )
        if accept is None or leave is None:
            return None
        try:
            hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
            max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
        except (TypeError, ValueError):
            return None
        if hp <= 0 or max_hp <= 1:
            return None
        max_hp_loss = max(1, min(max_hp - 1, int(math.ceil(max_hp * 0.5))))
        hp_after = min(max_hp - max_hp_loss, hp)
        min_after = _env_int("SPIRECOMM_EVENT_FAST_GHOSTS_ACCEPT_MIN_HP_AFTER", 20)
        if hp_after >= min_after:
            return accept, [], "event_fast_ghosts_accept"
        return None
    return None


def _maybe_apply_event_chosen_lowhp_cost_guard(
    env: Any,
    chosen_action: dict[str, Any],
    actions: list[dict[str, Any]],
    scores: list[float],
    source: str,
) -> tuple[dict[str, Any], list[float], str]:
    if not _env_bool("SPIRECOMM_EVENT_CHOSEN_LOW_HP_COST_GUARD", False):
        return chosen_action, scores, source
    safe_action = next(
        (
            action
            for action in actions
            if _event_label_key(action)
            in {"leave", "ignore", "ignored", "skip", "refuse", "stop", "stopped", "fled"}
        ),
        None,
    )
    if safe_action is None or safe_action is chosen_action:
        return chosen_action, scores, source
    chosen_cost = _event_action_hp_cost(chosen_action)
    if chosen_cost <= 0:
        return chosen_action, scores, source
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        return chosen_action, scores, source
    if hp <= 0:
        return chosen_action, scores, source
    min_after = _env_int("SPIRECOMM_EVENT_CHOSEN_LOW_HP_COST_MIN_AFTER", 25)
    if hp - chosen_cost <= min_after:
        return safe_action, scores, "event_chosen_lowhp_cost_guard"
    return chosen_action, scores, source


def _dead_adventurer_search_count_from_label(label: str) -> int:
    match = re.search(r"(\d+)", str(label or ""))
    if match is None:
        return 0
    return max(0, min(3, int(match.group(1))))


def _choose_dead_adventurer_macro(
    env: Any,
    actions: list[dict[str, Any]],
    selector: Any,
    *,
    require_model: bool = False,
    selectors: dict[str, Any] | None = None,
    return_scores: bool = True,
) -> tuple[dict[str, Any], list[float], str]:
    event = getattr(env, "current_event", None)
    data = getattr(event, "data", {}) if event is not None else {}
    search_action = next((action for action in actions if _event_label_key(action) == "searched1times"), None)
    leave_action = next((action for action in actions if _event_label_key(action) == "searched0times"), None)
    if search_action is None or leave_action is None:
        return actions[0], [], "event_untrained_policy"

    target_key = "model_target_search_count"
    target = data.get(target_key)
    capped_by_low_hp = False
    scores: list[float] = []
    if target is None:
        if not getattr(selector, "available", False):
            if require_model:
                _raise_model_required(
                    env,
                    reason="selector_unavailable",
                    actions=actions,
                    selector_name="event",
                    selectors=selectors,
                )
            return search_action, [], "event_untrained_policy"
        macro_candidates = [
            {
                "kind": "event",
                "event_id": "Dead Adventurer",
                "name": f"Searched '{count}' times",
                "label": f"Searched '{count}' times",
                "choice_index": count,
            }
            for count in range(4)
        ]
        result = selector.choose(env.state(), macro_candidates, return_scores=return_scores)
        if result is None:
            if require_model:
                _raise_model_required(
                    env,
                    reason="selector_returned_no_choice",
                    actions=macro_candidates,
                    selector_name="event",
                    selectors=selectors,
                )
            return search_action, [], "event_untrained_policy"
        index = int(result["choice_index"])
        if not 0 <= index < len(macro_candidates):
            if require_model:
                _raise_model_required(
                    env,
                    reason="selector_choice_out_of_range",
                    actions=macro_candidates,
                    selector_name="event",
                    selectors=selectors,
                )
            return search_action, [], "event_untrained_policy"
        target = _dead_adventurer_search_count_from_label(str(macro_candidates[index]["label"]))
        data[target_key] = target
        scores = list(result.get("scores") or [])

    if _env_bool("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_CAP", True):
        try:
            hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
            max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
        except (TypeError, ValueError):
            hp = 0
            max_hp = 0
        ratio_max = _env_float("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_RATIO_MAX", 0.65)
        hp_max = _env_int("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_MAX", 50)
        floor_max = _env_int("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_FLOOR_MAX", 16)
        try:
            floor = int(getattr(env, "floor", 0) or 0)
        except (TypeError, ValueError):
            floor = 0
        flat_match = hp_max > 0 and hp > 0 and hp <= hp_max
        ratio_match = max_hp > 0 and ratio_max > 0.0 and (float(hp) / float(max_hp)) <= ratio_max
        floor_match = floor_max <= 0 or floor <= floor_max
        if floor_match and (flat_match or ratio_match):
            capped_target = max(0, _env_int("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_MAX_SEARCHES", 1))
            if int(target) > capped_target:
                target = capped_target
                capped_by_low_hp = True

    successful_searches = int(data.get("num_rewards") or 0)
    source = "event_dead_adventurer_lowhp_cap" if capped_by_low_hp else "event_dead_adventurer_macro"
    if successful_searches < int(target):
        return search_action, scores, source
    return leave_action, scores, source


def _event_actions_supported_by_training(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_key = _event_key_from_actions(actions)
    if event_key == "addict":
        # SlayTheData A20 victories contain only Obtained Relic / Stole Relic
        # for this event, not the Leave option.
        return [
            action
            for action in actions
            if _event_label_key(action) in {"obtainedrelic", "stolerelic"}
        ]
    return actions


def _choose_event_choice(
    env: Any,
    selector: Any,
    *,
    require_model: bool = False,
    selectors: dict[str, Any] | None = None,
    return_scores: bool = True,
) -> tuple[dict[str, Any], list[float], str]:
    actions = list(env.legal_actions())
    if len(actions) == 1:
        return actions[0], [], "forced_single"
    if not actions:
        _raise_model_required(
            env,
            reason="event_has_no_legal_actions",
            actions=actions,
            selector_name="event",
            selectors=selectors,
        )

    event_key = _event_key_from_actions(actions)
    if event_key == "matchandkeep":
        card_selector = selectors.get("card_reward") if selectors is not None else None
        return _choose_match_and_keep_action(env, actions, card_selector), [], "event_match_and_keep_policy"
    if event_key == "noteforyourself":
        return _choose_note_for_yourself_action(actions), [], "event_untrained_policy"
    if event_key == "deadadventurer":
        return _choose_dead_adventurer_macro(
            env,
            actions,
            selector,
            require_model=require_model,
            selectors=selectors,
            return_scores=return_scores,
        )
    if event_key == "goldenidol" and any(_event_label_key(action) == "takegoldenidol" for action in actions):
        guarded = _choose_golden_idol_intro_forced_elite_guard(env, actions)
        if guarded is not None:
            return guarded
        for action in actions:
            if _event_label_key(action) == "takegoldenidol":
                return action, [], "event_golden_idol_take"
    if event_key == "goldenidol":
        guarded = _choose_golden_idol_boulder_guard(env, actions)
        if guarded is not None:
            return guarded
    if event_key == "scrapooze":
        guarded = _choose_scrap_ooze_lowhp_cap(env, actions)
        if guarded is not None:
            return guarded
    if event_key == "knowingskull":
        guarded = _choose_knowing_skull_leave_guard(env, actions)
        if guarded is not None:
            return guarded
    guarded = _choose_event_lowhp_cost_guard(env, actions)
    if guarded is not None:
        return guarded
    fast_policy = _choose_event_fast_policy(env, actions)
    if fast_policy is not None:
        return fast_policy

    model_actions = _event_actions_supported_by_training(actions)
    if not model_actions:
        if require_model:
            _raise_model_required(
                env,
                reason="event_has_no_training_aligned_actions",
                actions=actions,
                selector_name="event",
                selectors=selectors,
            )
        return actions[0], [], "event_untrained_policy"
    if len(model_actions) == 1:
        return model_actions[0], [], "event_filtered_policy" if len(actions) > 1 else "forced_single"

    if not getattr(selector, "available", False):
        if require_model:
            _raise_model_required(
                env,
                reason="selector_unavailable",
                actions=model_actions,
                selector_name="event",
                selectors=selectors,
            )
        return model_actions[0], [], "event_untrained_policy"
    needs_rollout_scores = _env_bool("SPIRECOMM_EVENT_ROLLOUT_RERANK", False)
    result = selector.choose(env.state(), model_actions, return_scores=return_scores or needs_rollout_scores)
    if result is None:
        if require_model:
            _raise_model_required(
                env,
                reason="selector_returned_no_choice",
                actions=model_actions,
                selector_name="event",
                selectors=selectors,
            )
        return model_actions[0], [], "event_untrained_policy"
    index = int(result["choice_index"])
    if not 0 <= index < len(model_actions):
        if require_model:
            _raise_model_required(
                env,
                reason="selector_choice_out_of_range",
                actions=model_actions,
                selector_name="event",
                selectors=selectors,
            )
        return model_actions[0], [], "event_untrained_policy"
    chosen_action = model_actions[index]
    scores = list(result.get("scores") or [])
    guarded_action, guarded_scores, guarded_source = _maybe_apply_event_chosen_lowhp_cost_guard(
        env,
        chosen_action,
        actions,
        scores,
        "event",
    )
    if guarded_source != "event":
        return guarded_action, guarded_scores, guarded_source
    reranked = _maybe_rerank_event_rollout(env, model_actions, scores, chosen_action, selectors)
    if reranked is not None:
        return reranked
    return guarded_action, guarded_scores, guarded_source


def _map_dp_state_key(env: Any) -> str | None:
    current = getattr(env, "current_map_node", None)
    if current is None:
        return None
    x, y = current
    act = int(getattr(env, "act", 1) or 1)
    return f"a{act}-r{int(y)}-x{int(x)}"


def _map_dp_winged_charges(env: Any) -> int:
    for relic in list(getattr(env, "relics", []) or []):
        if str(relic.get("relic_id") or relic.get("id") or "") != "WingedGreaves":
            continue
        return max(0, int(relic.get("counter") or 0))
    return 0


def _map_dp_initial_state(env: Any) -> tuple[int, int, int]:
    shop_value = _map_dp_current_shop_value(env)
    winged_charges = _map_dp_winged_charges(env)
    if not bool(getattr(env, "first_room_chosen", False)):
        return (0, shop_value, winged_charges)
    states = getattr(env, "_map_dp_state_by_node", None)
    key = _map_dp_state_key(env)
    if isinstance(states, dict) and key in states:
        value = states[key]
        return (int(value[0]), shop_value, winged_charges)
    return (
        int(getattr(env, "_map_dp_elite_count", 0)),
        shop_value,
        winged_charges,
    )


def _map_dp_current_shop_value(env: Any) -> int:
    gold = max(0, int(getattr(env, "gold", 0) or 0))
    return (gold // 100) * MAP_DP_SHOP_GOLD_UNIT_VALUE


def _map_dp_shop_increment(env: Any) -> int:
    del env
    return 0


def _map_dp_has_purgeable_curse(env: Any) -> bool:
    for card in list(getattr(env, "deck", []) or []):
        if str(card.get("type") or "") != "CURSE":
            continue
        if str(card.get("card_id") or "") in {"AscendersBane", "CurseOfTheBell", "Necronomicurse"}:
            continue
        if bool(card.get("bottled") or card.get("in_bottle_flame") or card.get("in_bottle_lightning") or card.get("in_bottle_tornado")):
            continue
        return True
    return False


def _map_dp_shop_bonus(env: Any) -> int:
    gold = int(getattr(env, "gold", 0) or 0)
    if gold >= MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD and _map_dp_has_purgeable_curse(env):
        return MAP_DP_SHOP_PURGEABLE_CURSE_BONUS
    return 0


def _map_dp_shop_urgency_bonus(env: Any, *, distance: int) -> int:
    if int(distance) > MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_HORIZON:
        return 0
    gold = int(getattr(env, "gold", 0) or 0)
    if gold >= MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD and _map_dp_has_purgeable_curse(env):
        return MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS
    return 0


def _map_dp_card_id(card: Any) -> str:
    if isinstance(card, dict):
        return str(card.get("card_id") or card.get("id") or card.get("name") or "")
    return str(getattr(card, "card_id", "") or getattr(card, "id", "") or getattr(card, "name", "") or "")


def _map_dp_card_type(card: Any) -> str:
    if isinstance(card, dict):
        return str(card.get("type") or "").upper()
    card_def = getattr(card, "card_def", None)
    return str(getattr(card_def, "card_type", "") or getattr(card, "type", "") or "").upper()


def _map_dp_card_upgrades(card: Any) -> int:
    if isinstance(card, dict):
        raw_value = card.get("upgrades")
    else:
        raw_value = getattr(card, "upgrades", 0)
    try:
        return int(raw_value or 0)
    except (TypeError, ValueError):
        return 0


def _map_dp_early_elite_readiness(env: Any) -> int:
    deck = list(getattr(env, "deck", []) or [])
    relic_count = len(list(getattr(env, "relics", []) or []))
    potion_count = len(list(getattr(env, "potions", []) or []))
    starter_ids = {"Strike_R", "Defend_R", "Bash"}
    nonstarter_cards = 0
    upgraded_cards = 0
    useful_attacks = 0
    block_cards = 0
    for card in deck:
        card_id = _map_dp_card_id(card)
        card_type = _map_dp_card_type(card)
        if card_type in {"STATUS", "CURSE"}:
            continue
        if card_id not in starter_ids:
            nonstarter_cards += 1
        if _map_dp_card_upgrades(card) > 0:
            upgraded_cards += 1
        if card_type == "ATTACK" and card_id not in {"Strike_R"}:
            useful_attacks += 1
        if card_type == "SKILL" and card_id not in {"Defend_R"}:
            block_cards += 1

    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
        max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
        max_hp = 0
    hp_ratio = float(hp) / float(max_hp) if max_hp > 0 else 0.0

    readiness = 0
    readiness += min(30, nonstarter_cards * 10)
    readiness += min(20, useful_attacks * 10)
    readiness += min(15, block_cards * 5)
    readiness += min(15, upgraded_cards * 8)
    readiness += min(20, relic_count * 10)
    readiness += min(15, potion_count * 5)
    if hp_ratio >= 0.80:
        readiness += 15
    elif hp_ratio >= 0.65:
        readiness += 8
    return int(readiness)


def _map_dp_early_elite_risk_penalty(env: Any, symbol: str | None, *, node_floor: int | None = None) -> int:
    if not MAP_DP_EARLY_ELITE_RISK_GUARD:
        return 0
    token = str(symbol or "")
    if token not in {"E", "E_GREEN"}:
        return 0
    if token == "E_GREEN" and not _env_bool("SPIRECOMM_MAP_DP_EARLY_ELITE_INCLUDE_GREEN", False):
        return 0
    try:
        act = int(getattr(env, "act", 1) or 1)
        floor = int(node_floor) if node_floor is not None else int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        return 0
    if act != 1 or floor > MAP_DP_EARLY_ELITE_FLOOR_MAX:
        return 0
    readiness = _map_dp_early_elite_readiness(env)
    shortfall = max(0, MAP_DP_EARLY_ELITE_MIN_READINESS - readiness)
    if shortfall <= 0:
        return 0
    scale = shortfall / max(1.0, float(MAP_DP_EARLY_ELITE_MIN_READINESS))
    return int(round(float(MAP_DP_EARLY_ELITE_PENALTY) * scale))


def _map_dp_hp_ratio(env: Any) -> float | None:
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
        max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
    except (TypeError, ValueError):
        return None
    if max_hp <= 0:
        return None
    return max(0.0, min(1.0, float(hp) / float(max_hp)))


def _map_dp_hp_aware_score_adjustment(env: Any, symbol: str | None, *, node_floor: int | None = None) -> int:
    if not _env_bool("SPIRECOMM_MAP_DP_HP_AWARE_RISK_GUARD", True):
        return 0
    token = str(symbol or "")
    if token not in {"E", "E_GREEN", "R"}:
        return 0
    try:
        act = int(getattr(env, "act", 1) or 1)
        floor = int(node_floor) if node_floor is not None else int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        return 0
    floor_max = _env_int("SPIRECOMM_MAP_DP_HP_AWARE_FLOOR_MAX", 6)
    if act != 1 or (floor_max > 0 and floor > floor_max):
        return 0
    hp_ratio = _map_dp_hp_ratio(env)
    if hp_ratio is None:
        return 0
    if token in {"E", "E_GREEN"}:
        safe_ratio = _env_float("SPIRECOMM_MAP_DP_HP_AWARE_ELITE_SAFE_RATIO", 0.78)
        penalty = _env_int("SPIRECOMM_MAP_DP_HP_AWARE_ELITE_PENALTY", 120)
        if safe_ratio <= 0.0 or hp_ratio >= safe_ratio:
            return 0
        scale = (safe_ratio - hp_ratio) / safe_ratio
        return -int(round(float(penalty) * scale))
    rest_ratio = _env_float("SPIRECOMM_MAP_DP_HP_AWARE_REST_TRIGGER_RATIO", 0.72)
    bonus = _env_int("SPIRECOMM_MAP_DP_HP_AWARE_REST_BONUS", 90)
    if rest_ratio <= 0.0 or hp_ratio >= rest_ratio:
        return 0
    scale = (rest_ratio - hp_ratio) / rest_ratio
    return int(round(float(bonus) * scale))


def _map_dp_node_score(
    env: Any,
    symbol: str | None,
    elite_count: int,
    shop_value: int,
    *,
    shop_increment: int,
    shop_bonus: int = 0,
    shop_urgency_bonus: int = 0,
    node_floor: int | None = None,
) -> tuple[int, tuple[int, int]]:
    token = str(symbol or "")
    if token == "E_GREEN":
        score = MAP_DP_ELITE_BASE_VALUE - 20 * int(elite_count) - MAP_DP_GREEN_ELITE_PENALTY
        score -= _map_dp_early_elite_risk_penalty(env, token, node_floor=node_floor)
        score += _map_dp_hp_aware_score_adjustment(env, token, node_floor=node_floor)
        return score, (int(elite_count) + 1, int(shop_value))
    if token == "E":
        score = MAP_DP_ELITE_BASE_VALUE - 20 * int(elite_count)
        score -= _map_dp_early_elite_risk_penalty(env, token, node_floor=node_floor)
        score += _map_dp_hp_aware_score_adjustment(env, token, node_floor=node_floor)
        return score, (int(elite_count) + 1, int(shop_value))
    if token == "$":
        return int(shop_value) + int(shop_bonus) + int(shop_urgency_bonus), (int(elite_count), int(shop_value))
    if token == "?":
        return 10, (int(elite_count), int(shop_value))

    score_by_symbol = {
        "M": MAP_DP_MONSTER_VALUE,
        "R": MAP_DP_REST_VALUE + _map_dp_hp_aware_score_adjustment(env, token, node_floor=node_floor),
        "T": 100,
        "BOSS": 0,
        "VICTORY": 0,
    }
    return int(score_by_symbol.get(token, 0)), (int(elite_count), int(shop_value))


def _map_dp_with_winged_state(state: tuple[int, int, int], next_base_state: tuple[int, int], winged_cost: int) -> tuple[int, int, int]:
    return (
        int(next_base_state[0]),
        int(next_base_state[1]),
        max(0, int(state[2]) - max(0, int(winged_cost))),
    )


def _map_dp_winged_offpath_penalty(winged_cost: int) -> int:
    return MAP_DP_WINGED_OFFPATH_PENALTY if int(winged_cost) > 0 else 0


def _map_dp_node_symbol(node: Any) -> str:
    symbol = str(getattr(node, "room_symbol", "") or "")
    if symbol == "E" and bool(getattr(node, "has_emerald_key", False)):
        return "E_GREEN"
    return symbol


def _map_dp_best_future_score(
    env: Any,
    node: Any,
    state: tuple[int, int, int],
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
        symbol = _map_dp_node_symbol(child)
        immediate, next_base_state = _map_dp_node_score(
            env,
            symbol,
            state[0],
            state[1],
            shop_increment=_map_dp_shop_increment(env),
            shop_bonus=_map_dp_shop_bonus(env),
            shop_urgency_bonus=_map_dp_shop_urgency_bonus(env, distance=int(steps_from_current) + 1),
            node_floor=int(getattr(child, "y", 0)) + 1,
        )
        next_state = _map_dp_with_winged_state(state, next_base_state, winged_cost)
        candidate = immediate - _map_dp_winged_offpath_penalty(winged_cost) + _map_dp_best_future_score(
            env,
            child,
            next_state,
            memo,
            steps_from_current=int(steps_from_current) + 1,
        )
        if candidate > best:
            best = candidate
    memo[key] = best
    return best


def _map_dp_action_winged_cost(env: Any, action: dict[str, Any]) -> int:
    if int(_map_dp_winged_charges(env)) <= 0 or getattr(env, "current_map_node", None) is None:
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
    nodes = getattr(env, "map", []) or []
    if current_y < 0 or current_y >= len(nodes):
        return 0
    row_nodes = nodes[current_y] or []
    if current_x < 0 or current_x >= len(row_nodes):
        return 0
    edges = list(getattr(row_nodes[current_x], "edges", []) or [])
    normal_connection = any(int(edge.dst_x) == x and int(edge.dst_y) == y for edge in edges)
    winged_connection = any(int(edge.dst_y) == y for edge in edges)
    return 1 if winged_connection and not normal_connection else 0


def _map_rollout_rerank_topk() -> int:
    if MAP_ROLLOUT_RERANK_ACTIVE > 0:
        return 0
    return max(0, _env_int("SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK", 2))


def _map_rollout_score(env: Any, *, root_floor: int, steps: int, max_steps_reached: bool) -> float:
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = root_floor
    phase = str(getattr(env, "phase", "") or "")
    try:
        hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
    try:
        gold = int(getattr(env, "gold", 0) or 0)
    except (TypeError, ValueError):
        gold = 0
    if phase == "GAME_OVER" or hp <= 0:
        return -100000.0 + float(floor) * 100.0 - float(steps) * 0.05
    if phase in {"COMPLETE", "VICTORY"}:
        return 100000.0 + float(floor) * 1000.0 + float(hp) * 10.0 + float(gold) * 0.05
    floor_gain = max(0, floor - int(root_floor))
    score = float(floor_gain) * 1000.0 + float(hp) * 12.0 + float(gold) * 0.05 - float(steps) * 0.02
    if max_steps_reached:
        score -= 5.0
    return score


def _map_rollout_score_candidate(
    root_blob: bytes,
    action: dict[str, Any],
    selectors: dict[str, Any],
    *,
    root_floor: int,
    max_steps: int,
    max_floor_delta: int,
) -> float:
    global MAP_ROLLOUT_RERANK_ACTIVE
    steps = 1
    MAP_ROLLOUT_RERANK_ACTIVE += 1
    previous_map_rollout_flag = os.environ.get("SPIRECOMM_V3_COMBAT_IN_MAP_ROLLOUT")
    os.environ["SPIRECOMM_V3_COMBAT_IN_MAP_ROLLOUT"] = "1"
    try:
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        target_floor = int(root_floor) + max(1, int(max_floor_delta))
        while steps < max_steps:
            phase = str(getattr(branch, "phase", "") or "")
            if phase in {"GAME_OVER", "COMPLETE", "VICTORY"}:
                break
            try:
                if int(getattr(branch, "floor", 0) or 0) >= target_floor:
                    break
            except (TypeError, ValueError):
                pass
            try:
                hp = int(getattr(getattr(branch, "player", None), "current_hp", 0) or 0)
            except (TypeError, ValueError):
                hp = 0
            if hp <= 0:
                break
            next_action, _scores, _source = choose_model_required_action(branch, selectors, return_scores=False)
            branch.step(next_action)
            steps += 1
        max_steps_reached = steps >= max_steps
        return _map_rollout_score(
            branch,
            root_floor=root_floor,
            steps=steps,
            max_steps_reached=max_steps_reached,
        )
    except Exception:
        return float("-inf")
    finally:
        if previous_map_rollout_flag is None:
            os.environ.pop("SPIRECOMM_V3_COMBAT_IN_MAP_ROLLOUT", None)
        else:
            os.environ["SPIRECOMM_V3_COMBAT_IN_MAP_ROLLOUT"] = previous_map_rollout_flag
        MAP_ROLLOUT_RERANK_ACTIVE = max(0, MAP_ROLLOUT_RERANK_ACTIVE - 1)


def _maybe_rerank_map_with_rollout(
    env: Any,
    actions: list[dict[str, Any]],
    scores: list[float],
    selected_index: int,
    selectors: dict[str, Any] | None,
) -> tuple[int, bool]:
    topk = _map_rollout_rerank_topk()
    if selectors is None or topk <= 1 or len(actions) <= 1 or len(scores) != len(actions):
        return selected_index, False
    avoid_elite_gate = _env_bool("SPIRECOMM_MAP_ROLLOUT_RERANK_AVOID_ELITE_GATE", True)
    selected_symbol = str(actions[selected_index].get("symbol") or "")
    if avoid_elite_gate and selected_symbol not in {"E", "E_GREEN"}:
        return selected_index, False
    try:
        root_floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        root_floor = 0
    floor_max = _env_int("SPIRECOMM_MAP_ROLLOUT_RERANK_FLOOR_MAX", 8)
    if floor_max > 0 and root_floor > floor_max:
        return selected_index, False
    try:
        numeric_scores = [float(score) for score in scores]
    except (TypeError, ValueError):
        return selected_index, False
    selected_score = numeric_scores[selected_index]
    if not math.isfinite(selected_score):
        return selected_index, False
    margin_max = max(0.0, _env_float("SPIRECOMM_MAP_ROLLOUT_RERANK_MARGIN_MAX", 120.0))
    ranked_indices = sorted(range(len(actions)), key=lambda index: numeric_scores[index], reverse=True)
    candidate_indices: list[int] = []
    for index in ranked_indices:
        if len(candidate_indices) >= topk:
            break
        if selected_score - numeric_scores[index] <= margin_max:
            if avoid_elite_gate:
                candidate_symbol = str(actions[index].get("symbol") or "")
                if candidate_symbol in {"E", "E_GREEN"}:
                    continue
                # Rest-before-elite helped against burning elites, but regressed
                # ordinary elite routes by merely delaying the same risk.
                if candidate_symbol == "R" and selected_symbol != "E_GREEN":
                    continue
            candidate_indices.append(index)
    if selected_index not in candidate_indices:
        candidate_indices.append(selected_index)
    if len(candidate_indices) <= 1:
        return selected_index, False
    try:
        root_blob = clone_env_blob(env, strip_debug_history=True)
    except Exception:
        return selected_index, False
    max_steps = max(1, _env_int("SPIRECOMM_MAP_ROLLOUT_RERANK_MAX_STEPS", 60))
    max_floor_delta = max(1, _env_int("SPIRECOMM_MAP_ROLLOUT_RERANK_MAX_FLOOR_DELTA", 2))
    rollout_scores: dict[int, float] = {}
    for index in candidate_indices:
        rollout_score = _map_rollout_score_candidate(
            root_blob,
            actions[index],
            selectors,
            root_floor=root_floor,
            max_steps=max_steps,
            max_floor_delta=max_floor_delta,
        )
        if math.isfinite(rollout_score):
            rollout_scores[index] = rollout_score
    if selected_index not in rollout_scores or len(rollout_scores) <= 1:
        return selected_index, False
    best_index = max(rollout_scores, key=lambda index: (rollout_scores[index], numeric_scores[index], -index))
    if best_index == selected_index:
        return selected_index, False
    min_advantage = max(0.0, _env_float("SPIRECOMM_MAP_ROLLOUT_RERANK_MIN_ADVANTAGE", 500.0))
    if rollout_scores[best_index] - rollout_scores[selected_index] < min_advantage:
        return selected_index, False
    return best_index, True


def _choose_map_dynamic_programming(env: Any, selectors: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[float], str]:
    actions = env.legal_actions()
    nodes = getattr(env, "map", None)
    if not nodes:
        return actions[0], [0.0 for _ in actions], "map_dp"

    initial_state = _map_dp_initial_state(env)
    shop_bonus = _map_dp_shop_bonus(env)
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
        winged_cost = _map_dp_action_winged_cost(env, action)
        immediate, next_state = _map_dp_node_score(
            env,
            symbol,
            initial_state[0],
            initial_state[1],
            shop_increment=_map_dp_shop_increment(env),
            shop_bonus=shop_bonus,
            shop_urgency_bonus=_map_dp_shop_urgency_bonus(env, distance=1),
            node_floor=int(y) + 1,
        )
        next_state = _map_dp_with_winged_state(initial_state, next_state, winged_cost)
        total_score = (
            immediate
            - _map_dp_winged_offpath_penalty(winged_cost)
            + _map_dp_best_future_score(env, node, next_state, memo, steps_from_current=1)
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
    best_index, rollout_reranked = _maybe_rerank_map_with_rollout(env, actions, scores, best_index, selectors)
    selected_action = actions[best_index]
    selected_state = selected_states[best_index]
    if selected_state is not None and selected_action.get("node_id"):
        setattr(env, "_map_dp_elite_count", int(selected_state[0]))
        setattr(env, "_map_dp_shop_value", int(selected_state[1]))
        states = getattr(env, "_map_dp_state_by_node", None)
        if not isinstance(states, dict):
            states = {}
            setattr(env, "_map_dp_state_by_node", states)
        states[str(selected_action["node_id"])] = selected_state
    return selected_action, scores, "map_dp_rollout_rerank" if rollout_reranked else "map_dp"


def _choose_candidates_with_selector_required(
    env: Any,
    selectors: dict[str, Any],
    *,
    selector_name: str,
    source: str,
    candidates: list[dict[str, Any]],
    all_actions: list[dict[str, Any]],
    reason_prefix: str,
    return_scores: bool = True,
) -> tuple[dict[str, Any], list[float], str]:
    if len(candidates) == 1:
        return candidates[0], [], "forced_single"
    selector = selectors.get(selector_name)
    if not getattr(selector, "available", False):
        _raise_model_required(
            env,
            reason=f"{reason_prefix}_selector_unavailable",
            actions=all_actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    result = selector.choose(env.state(), candidates, return_scores=return_scores)
    if result is None:
        _raise_model_required(
            env,
            reason=f"{reason_prefix}_selector_returned_no_choice",
            actions=all_actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    index = int(result["choice_index"])
    if not 0 <= index < len(candidates):
        _raise_model_required(
            env,
            reason=f"{reason_prefix}_selector_choice_out_of_range",
            actions=all_actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    return candidates[index], result["scores"], source


def build_runtime_selectors(
    *,
    repo_root: Path | None = None,
    device: str = "cpu",
    combat_device: str | None = None,
    combat_model: Path | None = None,
    combat_selector: str | None = None,
    v3_combat_model: Path | None = None,
    card_reward_model: Path | None = None,
    shop_model: Path | None = None,
    observation_version: str | None = None,
) -> dict[str, Any]:
    apply_runtime_latency_profile()
    resolved_root = Path(repo_root or Path(__file__).resolve().parents[2])
    if torch is None:
        return {
            "combat": None,
            "card_reward": None,
            "boss_relic": None,
            "map": None,
            "campfire": None,
            "shop": None,
            "event": None,
            "potion": None,
            "upgrade_target": None,
            "purge_target": None,
        }
    selector_kind = str(combat_selector or os.environ.get("SPIRECOMM_COMBAT_SELECTOR") or "legacy-slot").strip().lower().replace("_", "-")
    if selector_kind in {"v3-teacher", "teacher"}:
        from spirecomm.ai.v3_combat_selector import V3TeacherCombatSelector

        combat = V3TeacherCombatSelector()
    elif selector_kind in {"v3-candidate", "v3", "candidate"}:
        from spirecomm.ai.v3_combat_selector import V3CandidateCombatSelector

        combat = V3CandidateCombatSelector(
            checkpoint_path=v3_combat_model
            or os.environ.get("SPIRECOMM_V3_COMBAT_MODEL")
            or (resolved_root / "models" / "v3_combat_scorer.pt"),
            device=combat_device or device,
        )
    else:
        combat = SerializedCombatSelector(
            checkpoint_path=combat_model or (resolved_root / "models" / "combat.pt"),
            device=combat_device or device,
            observation_version=observation_version,
        )
    return {
        "combat": combat,
        "card_reward": CardRewardSelector(
            checkpoint_path=card_reward_model or (resolved_root / "models" / "card_reward.pt"),
            device=device,
        ),
        "boss_relic": BossRelicSelector(device=device),
        "campfire": CampfireChoiceSelector(device=device),
        "shop": ShopChoiceSelector(checkpoint_path=shop_model, device=device),
        "event": EventChoiceSelector(device=device),
        "potion": PotionUseSelector(device=device),
        "upgrade_target": UpgradeTargetSelector(device=device),
        "purge_target": PurgeTargetSelector(device=device),
    }


def choose_combat(env: Any, selector: SerializedCombatSelector | None, *, return_scores: bool = True):
    state = env.state()
    actions = env.legal_actions()
    if selector and selector.available:
        if hasattr(selector, "choose_env"):
            chosen, scores = selector.choose_env(env, return_scores=return_scores)
        else:
            chosen, scores = selector.choose(state, actions)
        if chosen is not None:
            return chosen, scores
    return actions[0], []


def choose_potion(env: Any, selector: PotionUseSelector | None, *, return_scores: bool = True):
    actions = [action for action in env.legal_actions() if action.get("kind") == "potion"]
    if not actions:
        return None, []
    candidates = [dict(action, action="USE") for action in actions]
    candidates.append({"kind": "potion", "name": "HOLD", "action": "HOLD", "potion_id": "HOLD"})
    if selector and selector.available:
        result = selector.choose(env.state(), candidates, return_scores=return_scores)
        if result is not None:
            index = int(result["choice_index"])
            if 0 <= index < len(actions):
                return actions[index], result["scores"]
            return None, result["scores"]
    return None, []


def _combat_selector_handles_potions(selectors: dict[str, Any]) -> bool:
    selector = selectors.get("combat")
    return bool(selector is not None and hasattr(selector, "choose_env") and getattr(selector, "handles_potions", False))


def choose_card_reward(env: Any, selector: CardRewardSelector | None):
    state = env.state()
    actions = env.legal_actions()
    reward_actions = [action for action in actions if action.get("kind") == "card_reward"]
    skip_action = next((action for action in actions if action.get("kind") in {"skip", "proceed"}), None)
    if selector and selector.available and reward_actions:
        reward_cards = [action["card"] for action in reward_actions]
        result = selector.choose(state, reward_cards, can_skip=skip_action is not None)
        if result is not None:
            index = int(result["choice_index"])
            if 0 <= index < len(reward_actions):
                return reward_actions[index], result["scores"]
            return (skip_action or actions[-1]), result["scores"]
    return actions[-1], []


def choose_reward_screen_action(
    env: Any,
    card_selector: CardRewardSelector | None,
    *,
    require_model: bool = False,
    selectors: dict[str, Any] | None = None,
    return_scores: bool = True,
):
    actions = env.legal_actions()
    grouped = _reward_actions_by_kind(actions)

    gold_actions = grouped.get("reward_gold") or []
    if gold_actions:
        action = max(gold_actions, key=lambda candidate: int(candidate.get("amount") or 0))
        return action, [], "reward_policy_collect_gold"

    relic_actions = grouped.get("reward_relic") or []
    key_actions = grouped.get("reward_key") or []
    if relic_actions and key_actions:
        # Temporarily disable sapphire-key tradeoff decisions. The boss relic
        # selector is trained for boss relic choices, not relic-vs-key rewards.
        return relic_actions[0], [], "reward_policy_relic_over_key"
    if relic_actions:
        if require_model and len(relic_actions) > 1:
            return _choose_candidates_with_selector_required(
                env,
                selectors or {},
                selector_name="boss_relic",
                source="reward_tradeoff",
                candidates=relic_actions,
                all_actions=actions,
                reason_prefix="reward_relic",
                return_scores=return_scores,
            )
        return relic_actions[0], [], "reward_policy_collect_relic"
    potion_actions = grouped.get("reward_potion") or []
    replacement_target_id = str(getattr(env, "reward_potion_replacement_target_id", "") or "")
    if potion_actions and replacement_target_id and _potion_slots_available(env):
        for action in potion_actions:
            if str(action.get("potion_id") or action.get("name") or "") == replacement_target_id:
                return action, [], "reward_policy_collect_replacement_potion"

    if potion_actions and _potion_slots_available(env):
        if require_model and len(potion_actions) > 1:
            candidates = []
            for action in potion_actions:
                candidate = dict(action)
                candidate["action"] = "ACQUIRE"
                candidate.setdefault("item_id", candidate.get("potion_id") or candidate.get("name"))
                candidates.append(candidate)
            return _choose_candidates_with_selector_required(
                env,
                selectors or {},
                selector_name="potion",
                source="reward_tradeoff",
                candidates=candidates,
                all_actions=actions,
                reason_prefix="reward_potion",
                return_scores=return_scores,
            )
        return potion_actions[0], [], "reward_policy_collect_potion"

    if key_actions:
        if require_model and len(key_actions) > 1:
            _raise_model_required(
                env,
                reason="multiple_key_rewards_without_selector",
                actions=actions,
                selector_name="reward_tradeoff",
                selectors=selectors,
            )
        return key_actions[0], [], "reward_policy_take_key"

    card_reward_open_actions = [
        action
        for action in actions
        if str(action.get("kind") or "").lower() == "raw"
        and str(action.get("label") or action.get("name") or "").upper() == "CARD"
    ]
    if card_reward_open_actions and not bool(getattr(env, "reward_card_reward_declined", False)):
        return card_reward_open_actions[0], [], "reward_policy_open_card_reward"

    reward_actions = grouped.get("card_reward") or []
    if reward_actions:
        skip_action = next((action for action in actions if action.get("kind") in {"skip", "proceed"}), None)
        if card_selector and card_selector.available:
            reward_cards = [action["card"] for action in reward_actions]
            result = card_selector.choose(
                env.state(),
                reward_cards,
                can_skip=skip_action is not None,
                return_scores=return_scores,
            )
            if result is not None:
                index = int(result["choice_index"])
                branch_used = bool(getattr(card_selector, "last_branch_used", False))
                if 0 <= index < len(reward_actions):
                    return reward_actions[index], result["scores"], "card_reward_branch_gate" if branch_used else "card_reward"
                if require_model and skip_action is None:
                    _raise_model_required(
                        env,
                        reason="card_reward_selector_chose_skip_but_skip_missing",
                        actions=actions,
                        selector_name="card_reward",
                        selectors=selectors,
                    )
                skip_action = skip_action or actions[-1]
                return skip_action, result["scores"], "card_reward_branch_gate_skip" if branch_used else "card_reward_skip"
        if require_model:
            _raise_model_required(
                env,
                reason="card_reward_selector_unavailable_or_no_choice",
                actions=actions,
                selector_name="card_reward",
                selectors=selectors,
            )
        return reward_actions[0], [], "card_reward_fallback_first"

    if potion_actions:
        if _env_bool("SPIRECOMM_REWARD_POTION_FULL_REPLACE", True):
            replacement_action = _best_full_belt_reward_potion(env, potion_actions, grouped.get("discard_potion") or [])
            if replacement_action is not None:
                return replacement_action, [], "reward_policy_replace_potion_full"
        skip_action = next((action for action in actions if action.get("kind") in {"skip", "proceed"}), actions[-1])
        return skip_action, [], "reward_policy_skip_potion_full"

    skip_action = next(
        (action for action in actions if action.get("kind") in {"skip", "proceed"}),
        actions[-1] if actions else {"kind": "noop", "name": "NOOP"},
    )
    return skip_action, [], "reward_policy_skip"


def choose_run_choice(env: Any, selector):
    actions = env.legal_actions()
    if selector and selector.available and actions:
        result = selector.choose(env.state(), actions)
        if result is not None:
            index = int(result["choice_index"])
            if 0 <= index < len(actions):
                return actions[index], result["scores"]
    return actions[0], []


def attach_card_target(env: Any, action: dict, selectors: dict, *, require_model: bool = False) -> dict:
    def _card_candidate(card):
        if isinstance(card, dict):
            return {
                "card_id": card.get("card_id"),
                "name": card.get("name"),
                "type": card.get("type"),
                "rarity": card.get("rarity"),
                "upgrades": int(card.get("upgrades") or 0),
            }
        return {
            "card_id": card.card_id,
            "name": card.name,
            "type": card.card_def.card_type,
            "rarity": card.card_def.rarity,
            "upgrades": card.upgrades,
        }

    action_name = str(action.get("name") or "").upper()

    if (action.get("kind") == "shop" and action.get("item_kind") == "purge") or (action.get("kind") == "campfire" and action_name == "PURGE"):
        if action.get("target_index") is not None:
            return action
        candidates = [
            _card_candidate(card)
            for card in env.deck
        ]
        selector = selectors.get("purge_target")
        chosen_target_index = None
        if selector and selector.available and candidates:
            result = selector.choose(env.state(), candidates, return_scores=False)
            if result is not None and 0 <= int(result["choice_index"]) < len(candidates):
                chosen_target_index = int(result["choice_index"])
        if require_model and chosen_target_index is None and len(candidates) > 1:
            _raise_model_required(
                env,
                reason="purge_target_selector_unavailable_or_no_choice",
                actions=env.legal_actions(),
                selector_name="purge_target",
                selectors=selectors,
            )
        if chosen_target_index is None and candidates:
            chosen_target_index = 0
        if chosen_target_index is not None:
            action = dict(action)
            action["target_index"] = chosen_target_index
    return action


def choose_modeled_action(env: Any, selectors: dict[str, Any]) -> tuple[dict[str, Any], list[float], str]:
    if env.phase == "COMBAT":
        if not _combat_selector_handles_potions(selectors):
            action, scores = choose_potion(env, selectors.get("potion"))
            if action is not None:
                return action, scores, "potion"
        action, scores = choose_combat(env, selectors.get("combat"))
        return action, scores, "combat"
    if env.phase == "CARD_REWARD":
        action, scores, source = choose_reward_screen_action(env, selectors.get("card_reward"))
        return action, scores, source
    if env.phase == "MAP":
        return _choose_map_dynamic_programming(env, selectors)
    if env.phase == "BOSS_RELIC":
        actions = env.legal_actions()
        action, scores = choose_run_choice(env, selectors.get("boss_relic"))
        biased_action, scores, source = _maybe_apply_boss_relic_score_bias(actions, scores, "boss_relic", env)
        if biased_action is not None:
            action = biased_action
        reranked = _maybe_rerank_boss_relic_rollout(env, actions, scores, action, selectors)
        if reranked is not None:
            return reranked
        return action, scores, source
    if env.phase == "CAMPFIRE":
        action, scores = choose_run_choice(env, selectors.get("campfire"))
        action, scores, source = _maybe_apply_campfire_low_hp_rest_guard(env, action, scores, "campfire", selectors)
        return attach_card_target(env, action, selectors), scores, source
    if env.phase == "SHOP":
        if _shop_value_policy_enabled():
            action, scores, source = _choose_shop_value_policy_required(env, selectors)
        else:
            action, scores = choose_run_choice(env, selectors.get("shop"))
            source = "shop"
        return attach_card_target(env, action, selectors), scores, source
    if env.phase == "CARD_SELECT":
        actions = env.legal_actions()
        current_card_select = getattr(env, "current_card_select", None) or {}
        mode = str(current_card_select.get("mode") or "")
        if mode in {"purge", "remove", "transform", "duplicate"}:
            selector = selectors.get("purge_target")
            candidates = [
                {
                    "card_id": action.get("card_id"),
                    "name": action.get("name"),
                    "type": action.get("type"),
                    "rarity": action.get("rarity"),
                    "upgrades": int(action.get("upgrades") or 0),
                }
                for action in actions
            ]
            if selector and selector.available and candidates:
                result = selector.choose(env.state(), candidates)
                if result is not None and 0 <= int(result["choice_index"]) < len(actions):
                    return actions[int(result["choice_index"])], result["scores"], f"{mode}_target"
        if mode == "upgrade":
            selector = selectors.get("upgrade_target")
            candidates = [
                {
                    "card_id": action.get("card_id"),
                    "name": action.get("name"),
                    "type": action.get("type"),
                    "rarity": action.get("rarity"),
                    "upgrades": int(action.get("upgrades") or 0),
                }
                for action in actions
            ]
            if selector and selector.available and candidates:
                result = selector.choose(env.state(), candidates)
                if result is not None and 0 <= int(result["choice_index"]) < len(actions):
                    return actions[int(result["choice_index"])], result["scores"], "upgrade_target"
        return actions[0], [], "fallback"
    if env.phase == "EVENT":
        return _choose_event_choice(env, selectors.get("event"), selectors=selectors)
    if env.phase == "NEOW":
        return _choose_neow_weighted_action(env, selectors=selectors)
    if env.phase == "TREASURE":
        actions = env.legal_actions()
        action = next((candidate for candidate in actions if candidate.get("name") == "OPEN_CHEST"), actions[0] if actions else {"kind": "noop", "name": "NOOP"})
        return action, [], "fallback"
    if env.phase == "CHEST":
        actions = env.legal_actions()
        action = next((candidate for candidate in actions if candidate.get("item_kind") == "sapphire_key"), actions[0] if actions else {"kind": "noop", "name": "NOOP"})
        return action, [], "fallback"
    actions = env.legal_actions()
    fallback = actions[0] if actions else {"kind": "noop", "name": "NOOP"}
    return fallback, [], "fallback"


def _choose_potion_required(env: Any, selectors: dict[str, Any], *, return_scores: bool = True):
    actions = [action for action in env.legal_actions() if action.get("kind") == "potion"]
    if not actions:
        return None, []
    selector = selectors.get("potion")
    if not getattr(selector, "available", False):
        _raise_model_required(
            env,
            reason="potion_selector_unavailable",
            actions=env.legal_actions(),
            selector_name="potion",
            selectors=selectors,
        )
    candidates = [dict(action, action="USE") for action in actions]
    candidates.append({"kind": "potion", "name": "HOLD", "action": "HOLD", "potion_id": "HOLD"})
    result = selector.choose(env.state(), candidates, return_scores=return_scores)
    if result is None:
        _raise_model_required(
            env,
            reason="potion_selector_returned_no_choice",
            actions=env.legal_actions(),
            selector_name="potion",
            selectors=selectors,
        )
    index = int(result["choice_index"])
    if 0 <= index < len(actions):
        return actions[index], result["scores"]
    if index == len(actions):
        return None, result["scores"]
    _raise_model_required(
        env,
        reason="potion_selector_choice_out_of_range",
        actions=env.legal_actions(),
        selector_name="potion",
        selectors=selectors,
    )
    raise AssertionError("unreachable")


def _combat_rollout_rerank_topk() -> int:
    if COMBAT_ROLLOUT_RERANK_ACTIVE > 0:
        return 0
    return max(0, _env_int("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", 3))


def _combat_rollout_state(env: Any) -> dict[str, Any]:
    try:
        state = env.state()
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


def _combat_rollout_combat_state(state: dict[str, Any]) -> dict[str, Any]:
    combat = state.get("combat_state")
    return combat if isinstance(combat, dict) else {}


def _combat_rollout_phase(env: Any, state: dict[str, Any] | None = None) -> str:
    raw_phase = getattr(env, "phase", None)
    if raw_phase:
        return str(raw_phase)
    if state is None:
        state = _combat_rollout_state(env)
    return str(state.get("phase") or state.get("screen") or "")


def _combat_rollout_turn(state: dict[str, Any]) -> int | None:
    combat = _combat_rollout_combat_state(state)
    try:
        return int(combat.get("turn"))
    except (TypeError, ValueError):
        return None


def _combat_rollout_has_combat_context(env: Any, state: dict[str, Any] | None = None) -> bool:
    phase = _combat_rollout_phase(env, state)
    if phase == "COMBAT":
        return True
    if phase not in {"CARD_SELECT", "CARD_REWARD"}:
        return False
    if state is None:
        state = _combat_rollout_state(env)
    combat = _combat_rollout_combat_state(state)
    return bool(combat.get("monsters"))


def _combat_rollout_player_hp(env: Any, state: dict[str, Any] | None = None) -> int:
    if state is None:
        state = _combat_rollout_state(env)
    for key in ("current_hp", "hp"):
        raw_value = state.get(key)
        if raw_value is not None:
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                pass
    player = _combat_rollout_combat_state(state).get("player")
    if isinstance(player, dict):
        try:
            return int(player.get("current_hp") or 0)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _combat_rollout_player_block(state: dict[str, Any]) -> int:
    player = _combat_rollout_combat_state(state).get("player")
    if not isinstance(player, dict):
        return 0
    try:
        return int(player.get("block") or 0)
    except (TypeError, ValueError):
        return 0


def _combat_rollout_live_monster_stats(state: dict[str, Any]) -> tuple[int, int]:
    monsters = _combat_rollout_combat_state(state).get("monsters")
    if not isinstance(monsters, list):
        return 0, 0
    live_count = 0
    live_hp = 0
    for monster in monsters:
        if not isinstance(monster, dict):
            continue
        if bool(monster.get("is_gone") or monster.get("half_dead")):
            continue
        try:
            hp = int(monster.get("current_hp") or 0)
        except (TypeError, ValueError):
            hp = 0
        if hp > 0:
            live_count += 1
            live_hp += hp
    return live_count, live_hp


def _combat_rollout_action_card_type(before_state: dict[str, Any], action: dict[str, Any]) -> str:
    action_card = action.get("card")
    if isinstance(action_card, dict):
        return str(action_card.get("type") or action.get("type") or "").strip().upper()
    index = action.get("card_index")
    if index is None:
        index = action.get("source_index")
    try:
        card_index = int(index)
    except (TypeError, ValueError):
        return str(action.get("type") or "").strip().upper()
    hand = _combat_rollout_combat_state(before_state).get("hand")
    if isinstance(hand, list) and 0 <= card_index < len(hand) and isinstance(hand[card_index], dict):
        return str(hand[card_index].get("type") or action.get("type") or "").strip().upper()
    return str(action.get("type") or "").strip().upper()


def _combat_rollout_env_set(name: str) -> set[str]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _combat_rollout_dangerous_state(state: dict[str, Any]) -> bool:
    hp = _combat_rollout_player_hp(None, state)
    if hp <= 0:
        return True
    player = _combat_rollout_combat_state(state).get("player")
    max_hp = 0
    if isinstance(player, dict):
        try:
            max_hp = int(player.get("max_hp") or 0)
        except (TypeError, ValueError):
            max_hp = 0
    try:
        incoming = int(incoming_damage(state))
    except Exception:
        incoming = 0
    block = _combat_rollout_player_block(state)
    uncovered = max(0, incoming - max(0, block))
    if uncovered >= hp:
        return True
    hp_ratio_max = _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DANGER_HP_RATIO_MAX", 0.5)
    return bool(uncovered > 0 and max_hp > 0 and float(hp) / float(max_hp) <= hp_ratio_max)


def _combat_rollout_should_stop(env: Any, *, root_turn: int | None) -> bool:
    state = _combat_rollout_state(env)
    phase = _combat_rollout_phase(env, state)
    if phase in COMBAT_ROLLOUT_RERANK_TERMINAL_PHASES:
        return True
    if _combat_rollout_player_hp(env, state) <= 0:
        return True
    if not _combat_rollout_has_combat_context(env, state):
        return True
    turn = _combat_rollout_turn(state)
    if (
        _env_bool("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_SAME_TURN_ONLY", False)
        and root_turn is not None
        and turn is not None
        and turn > root_turn
    ):
        return True
    return False


def _combat_rollout_terminal_score(
    env: Any,
    *,
    root_live_hp: int,
    steps: int,
    max_steps_reached: bool,
) -> float:
    state = _combat_rollout_state(env)
    phase = _combat_rollout_phase(env, state)
    hp = _combat_rollout_player_hp(env, state)
    try:
        floor = int(getattr(env, "floor", state.get("floor", 0)) or 0)
    except (TypeError, ValueError):
        floor = 0
    if phase == "GAME_OVER" or hp <= 0:
        death_step_weight = _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_DEATH_STEP_WEIGHT", -1.0)
        return -100000.0 + float(death_step_weight) * float(steps)
    if phase in {"COMPLETE", "VICTORY"}:
        return 100000.0 + float(floor) * 100.0 + float(hp) * 10.0 - float(steps) * 0.01
    if not _combat_rollout_has_combat_context(env, state):
        # Combat has ended and reward/run decisions can take over. Prefer clean
        # combat wins, then HP preservation; later run policy is intentionally
        # not simulated here to avoid turning this into a seed-specific planner.
        return 50000.0 + float(floor) * 25.0 + float(hp) * 10.0 - float(steps) * 0.01

    live_count, live_hp = _combat_rollout_live_monster_stats(state)
    progress = max(0.0, float(root_live_hp - live_hp))
    block = _combat_rollout_player_block(state)
    try:
        incoming = int(incoming_damage(state))
    except Exception:
        incoming = 0
    uncovered = max(0, incoming - max(0, block))
    score = (
        float(hp) * 18.0
        + progress * 2.5
        - float(live_hp) * 1.2
        - float(live_count) * 3.0
        + float(min(block, incoming)) * 1.0
        - float(uncovered) * 12.0
        - float(steps) * 0.05
    )
    if max_steps_reached:
        score -= 5.0
    return float(score)


def _combat_rollout_outcome(env: Any) -> str:
    state = _combat_rollout_state(env)
    phase = _combat_rollout_phase(env, state)
    hp = _combat_rollout_player_hp(env, state)
    if phase == "GAME_OVER" or hp <= 0:
        return "death"
    if phase in {"COMPLETE", "VICTORY"}:
        return "run_complete"
    if not _combat_rollout_has_combat_context(env, state):
        return "combat_clear"
    return "combat_active"


def _combat_rollout_apply_light_policy_env() -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for name in COMBAT_ROLLOUT_LIGHT_POLICY_DISABLED_ENVS:
        snapshot[name] = os.environ.get(name)
        os.environ[name] = "0"
    return snapshot


def _combat_rollout_restore_light_policy_env(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _combat_rollout_score_candidate(
    root_blob: bytes,
    action: dict[str, Any],
    selectors: dict[str, Any],
    *,
    root_turn: int | None,
    root_live_hp: int,
    max_steps: int,
    light_policy_override: bool | None = None,
) -> tuple[float, int, str]:
    global COMBAT_ROLLOUT_RERANK_ACTIVE
    steps = 1
    COMBAT_ROLLOUT_RERANK_ACTIVE += 1
    light_policy = (
        _env_bool("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LIGHT_POLICY", True)
        if light_policy_override is None
        else bool(light_policy_override)
    )
    env_snapshot: dict[str, str | None] = {}
    try:
        if light_policy:
            env_snapshot = _combat_rollout_apply_light_policy_env()
        # root_blob already has debug history stripped. The rollout branch is
        # not re-pickled, so clearing debug history again after the first step
        # only adds overhead and does not affect gameplay state.
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=False)
        while steps < max_steps and not _combat_rollout_should_stop(branch, root_turn=root_turn):
            next_action = _choose_combat_rollout_policy_action(branch, selectors)
            branch.step(next_action)
            steps += 1
        score = _combat_rollout_terminal_score(
            branch,
            root_live_hp=root_live_hp,
            steps=steps,
            max_steps_reached=steps >= max_steps and not _combat_rollout_should_stop(branch, root_turn=root_turn),
        )
        return score, steps, _combat_rollout_outcome(branch)
    except Exception:
        return float("-inf"), steps, "error"
    finally:
        if light_policy:
            _combat_rollout_restore_light_policy_env(env_snapshot)
        COMBAT_ROLLOUT_RERANK_ACTIVE = max(0, COMBAT_ROLLOUT_RERANK_ACTIVE - 1)


def _choose_combat_rollout_policy_action(env: Any, selectors: dict[str, Any]) -> dict[str, Any]:
    """Fast path for combat-rerank continuations.

    During combat rollout rerank, recursive rollout rerank is disabled by
    COMBAT_ROLLOUT_RERANK_ACTIVE. For ordinary COMBAT states, calling the combat
    selector directly is therefore equivalent to choose_model_required_action()
    but avoids full phase dispatch and source bookkeeping on every simulated
    continuation step. Non-combat choice screens still use the generic required
    policy because they may need card reward/card select/shop/event selectors.
    """

    if str(getattr(env, "phase", "")) == "COMBAT":
        selector = selectors.get("combat")
        if _env_bool("SPIRECOMM_V3_COMBAT_ROLLOUT_CPU_CONTINUATION", False):
            cpu_selector_fn = getattr(selector, "rollout_cpu_selector", None)
            if callable(cpu_selector_fn):
                try:
                    selector = cpu_selector_fn()
                except Exception:
                    pass
        legal_actions_env = getattr(selector, "legal_actions_env", None)
        actions = legal_actions_env(env) if callable(legal_actions_env) else env.legal_actions()
        if len(actions) == 1:
            return actions[0]
        if not getattr(selector, "available", False):
            raise RuntimeError("combat_selector_unavailable")
        if hasattr(selector, "choose_env"):
            chosen, _scores = selector.choose_env(env, return_scores=False, legal_actions=actions)
        else:
            chosen, _scores = selector.choose(env.state(), actions)
        if chosen is None:
            selector_error = str(getattr(selector, "last_error", "") or "")
            raise RuntimeError(f"combat_selector_returned_no_choice:{selector_error}")
        return chosen
    next_action, _scores, _source = choose_model_required_action(env, selectors, return_scores=False)
    return next_action


def _combat_action_index(actions: list[dict[str, Any]], chosen: dict[str, Any]) -> int | None:
    for index, action in enumerate(actions):
        if action is chosen:
            return index
    for index, action in enumerate(actions):
        if action == chosen:
            return index
    return None


def _event_rollout_terminal_score(env: Any, *, max_steps_reached: bool) -> float:
    phase = str(getattr(env, "phase", ""))
    try:
        floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        floor = 0
    player = getattr(env, "player", None)
    try:
        hp = int(getattr(player, "current_hp", 0) or 0)
        max_hp = int(getattr(player, "max_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
        max_hp = 0
    resource_bonus = 0.0
    if _env_bool("SPIRECOMM_EVENT_ROLLOUT_RESOURCE_BONUS", False) and phase != "GAME_OVER" and hp > 0:
        try:
            gold = int(getattr(env, "gold", 0) or 0)
        except (TypeError, ValueError):
            gold = 0
        try:
            relic_count = len(list(getattr(env, "relics", []) or []))
        except Exception:
            relic_count = 0
        potions = list(getattr(env, "potions", []) or [])
        potion_count = 0
        for potion in potions:
            if not isinstance(potion, dict):
                continue
            potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
            if potion_id and potion_id.lower() not in {"potion slot", "empty", "none"}:
                potion_count += 1
        curse_count = 0
        for card in list(getattr(env, "deck", []) or []):
            if not isinstance(card, dict):
                continue
            if str(card.get("type") or "").upper() == "CURSE":
                curse_count += 1
        resource_bonus = (
            float(max(0, gold)) * _env_float("SPIRECOMM_EVENT_ROLLOUT_RESOURCE_GOLD_WEIGHT", 0.25)
            + float(max(0, relic_count)) * _env_float("SPIRECOMM_EVENT_ROLLOUT_RESOURCE_RELIC_WEIGHT", 80.0)
            + float(max(0, potion_count)) * _env_float("SPIRECOMM_EVENT_ROLLOUT_RESOURCE_POTION_WEIGHT", 12.0)
            - float(max(0, curse_count)) * _env_float("SPIRECOMM_EVENT_ROLLOUT_RESOURCE_CURSE_PENALTY", 60.0)
        )
    if phase in {"COMPLETE", "VICTORY"}:
        return 1_000_000.0 + float(floor) * 1000.0 + float(max(0, hp)) + resource_bonus
    if phase == "GAME_OVER" or hp <= 0:
        return float(floor) * 1000.0 - 10_000.0
    if max_steps_reached:
        return float(floor) * 1000.0 - 1000.0 + float(max(0, hp)) + float(max(0, max_hp)) * 0.1 + resource_bonus
    return float(floor) * 1000.0 + float(max(0, hp)) + float(max(0, max_hp)) * 0.1 + resource_bonus


def _event_rollout_potion_count(env: Any) -> int:
    count = 0
    for potion in list(getattr(env, "potions", []) or []):
        if not isinstance(potion, dict):
            continue
        potion_id = str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "")
        if potion_id and potion_id.lower() not in {"potion slot", "empty", "none"}:
            count += 1
    return count


def _event_rollout_curse_count(env: Any) -> int:
    count = 0
    for card in list(getattr(env, "deck", []) or []):
        if not isinstance(card, dict):
            continue
        if str(card.get("type") or "").upper() == "CURSE":
            count += 1
    return count


def _event_rollout_resource_state(env: Any) -> dict[str, int]:
    player = getattr(env, "player", None)
    try:
        hp = int(getattr(player, "current_hp", 0) or 0)
    except (TypeError, ValueError):
        hp = 0
    try:
        max_hp = int(getattr(player, "max_hp", 0) or 0)
    except (TypeError, ValueError):
        max_hp = 0
    try:
        gold = int(getattr(env, "gold", 0) or 0)
    except (TypeError, ValueError):
        gold = 0
    try:
        relic_count = len(list(getattr(env, "relics", []) or []))
    except Exception:
        relic_count = 0
    return {
        "hp": hp,
        "max_hp": max_hp,
        "gold": gold,
        "relic_count": relic_count,
        "potion_count": _event_rollout_potion_count(env),
        "curse_count": _event_rollout_curse_count(env),
        **_event_rollout_deck_metrics(env),
    }


def _event_rollout_deck_metrics(env: Any) -> dict[str, int]:
    cards = list(getattr(env, "deck", []) or [])
    card_count = 0
    upgraded_count = 0
    total_upgrades = 0
    curse_count = 0
    basic_count = 0
    strike_defend_tokens = {"strike", "striker", "defend", "defendr"}
    for card in cards:
        if not isinstance(card, dict):
            card_id = str(getattr(card, "card_id", "") or getattr(card, "id", "") or "")
            name = str(getattr(card, "name", "") or "")
            card_type = str(getattr(getattr(card, "card_def", None), "card_type", "") or getattr(card, "type", "") or "")
            try:
                upgrades = int(getattr(card, "upgrades", 0) or 0)
            except (TypeError, ValueError):
                upgrades = 0
        else:
            card_id = str(card.get("card_id") or card.get("id") or "")
            name = str(card.get("name") or "")
            card_type = str(card.get("type") or "")
            try:
                upgrades = int(card.get("upgrades") or 0)
            except (TypeError, ValueError):
                upgrades = 0
        card_count += 1
        if upgrades > 0:
            upgraded_count += 1
            total_upgrades += upgrades
        if card_type.upper() == "CURSE":
            curse_count += 1
        token = normalize_token(card_id or name)
        if token in strike_defend_tokens:
            basic_count += 1
    return {
        "deck_card_count": card_count,
        "deck_upgraded_count": upgraded_count,
        "deck_total_upgrades": total_upgrades,
        "deck_curse_count": curse_count,
        "deck_basic_count": basic_count,
    }


def _event_rollout_immediate_resource_bonus(root_state: dict[str, int], branch: Any, action: dict[str, Any]) -> float:
    event_token = normalize_token(action.get("event_id") or action.get("event_name") or "")
    phase = str(getattr(branch, "phase", ""))
    if phase == "GAME_OVER":
        return 0.0
    branch_state = _event_rollout_resource_state(branch)
    bonus = 0.0
    if _env_bool("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_RESOURCE_BONUS", False):
        excluded_events = {
            normalize_token(token)
            for token in str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_RESOURCE_EXCLUDE_EVENTS", "MindBloom")).split(",")
            if normalize_token(token)
        }
        if not event_token or event_token not in excluded_events:
            bonus += float(branch_state["gold"] - root_state["gold"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_GOLD_WEIGHT",
                0.20,
            )
            bonus += float(branch_state["relic_count"] - root_state["relic_count"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_RELIC_WEIGHT",
                60.0,
            )
            bonus += float(branch_state["potion_count"] - root_state["potion_count"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_POTION_WEIGHT",
                8.0,
            )
            bonus -= float(branch_state["curse_count"] - root_state["curse_count"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_CURSE_PENALTY",
                45.0,
            )
            bonus += float(branch_state["max_hp"] - root_state["max_hp"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_MAX_HP_WEIGHT",
                1.5,
            )
    if _env_bool("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_DECK_BONUS", False):
        included_events = {
            normalize_token(token)
            for token in str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_DECK_EVENTS", "")).split(",")
            if normalize_token(token)
        }
        excluded_events = {
            normalize_token(token)
            for token in str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_DECK_EXCLUDE_EVENTS", "")).split(",")
            if normalize_token(token)
        }
        if (not included_events or event_token in included_events) and event_token not in excluded_events:
            bonus += float(branch_state["deck_total_upgrades"] - root_state["deck_total_upgrades"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_UPGRADE_WEIGHT",
                25.0,
            )
            bonus -= float(branch_state["deck_curse_count"] - root_state["deck_curse_count"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_DECK_CURSE_PENALTY",
                60.0,
            )
            bonus -= float(branch_state["deck_basic_count"] - root_state["deck_basic_count"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_BASIC_CARD_PENALTY",
                8.0,
            )
            bonus -= float(branch_state["deck_card_count"] - root_state["deck_card_count"]) * _env_float(
                "SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_CARD_COUNT_PENALTY",
                0.0,
            )
    return bonus


def _event_rollout_score_candidate(
    root_blob: bytes,
    action: dict[str, Any],
    selectors: dict[str, Any],
    *,
    max_steps: int,
    max_floor: int,
    root_resource_state: dict[str, int] | None = None,
) -> tuple[float, float]:
    previous_flag = os.environ.get("SPIRECOMM_EVENT_IN_ROLLOUT")
    os.environ["SPIRECOMM_EVENT_IN_ROLLOUT"] = "1"
    try:
        branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
        immediate_bonus = (
            _event_rollout_immediate_resource_bonus(root_resource_state, branch, action)
            if root_resource_state is not None
            else 0.0
        )
        steps = 0
        terminal = {"GAME_OVER", "COMPLETE", "VICTORY"}
        while steps < max_steps:
            phase = str(getattr(branch, "phase", ""))
            try:
                floor = int(getattr(branch, "floor", 0) or 0)
            except (TypeError, ValueError):
                floor = 0
            if phase in terminal or floor > max_floor:
                break
            next_action, _scores, _source = choose_model_required_action(branch, selectors, return_scores=False)
            branch.step(next_action)
            steps += 1
        return _event_rollout_terminal_score(branch, max_steps_reached=steps >= max_steps) + immediate_bonus, immediate_bonus
    except Exception:
        return float("-inf"), 0.0
    finally:
        if previous_flag is None:
            os.environ.pop("SPIRECOMM_EVENT_IN_ROLLOUT", None)
        else:
            os.environ["SPIRECOMM_EVENT_IN_ROLLOUT"] = previous_flag


def _event_rollout_model_score_drop_limit(event_tokens: set[str]) -> float:
    raw_by_event = str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_BY_EVENT", "") or "")
    for item in raw_by_event.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        if normalize_token(key) not in event_tokens:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    max_model_score_drop = _env_float("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP", float("inf"))
    if not math.isfinite(max_model_score_drop):
        return float("inf")
    scoped_events = {
        normalize_token(token)
        for token in str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_EVENTS", "")).split(",")
        if normalize_token(token)
    }
    if scoped_events and not event_tokens.intersection(scoped_events):
        return float("inf")
    return max_model_score_drop


def _event_rollout_allows_candidate_action(chosen_action: dict[str, Any], candidate_action: dict[str, Any]) -> bool:
    event_token = normalize_token(candidate_action.get("event_id") or candidate_action.get("event_name") or "")
    if event_token != "wemeetagain":
        return True
    candidate_token = normalize_token(candidate_action.get("name") or candidate_action.get("label") or "")
    if candidate_token != "givecard":
        return True
    chosen_token = normalize_token(chosen_action.get("name") or chosen_action.get("label") or "")
    if _env_bool("SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_PAY_GOLD_TO_GIVE_CARD_OVERRIDE", False):
        return not chosen_token.startswith("pay") or "gold" not in chosen_token
    if _env_bool("SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_GIVE_CARD_OVERRIDE", False):
        return chosen_token == "givecard"
    return True


def _maybe_rerank_event_rollout(
    env: Any,
    actions: list[dict[str, Any]],
    scores: list[float],
    chosen_action: dict[str, Any],
    selectors: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_EVENT_ROLLOUT_RERANK", False):
        return None
    if selectors is None or str(os.environ.get("SPIRECOMM_EVENT_IN_ROLLOUT") or "").strip():
        return None
    if len(actions) <= 1 or not scores or len(scores) != len(actions):
        return None
    try:
        root_floor = int(getattr(env, "floor", 0) or 0)
    except (TypeError, ValueError):
        root_floor = 0
    event_id_env = os.environ.get("SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS")
    if event_id_env is None:
        event_id_env = os.environ.get(
            "SPIRECOMM_EVENT_ROLLOUT_NAMES",
            "Big Fish,Falling,The Library,MindBloom,Mysterious Sphere,The Woman in Blue",
        )
    allowed_events = {
        normalize_token(token)
        for token in str(event_id_env).split(",")
        if normalize_token(token)
    }
    event_tokens = {
        normalize_token(action.get("event_id") or "")
        for action in actions
        if normalize_token(action.get("event_id") or "")
    }
    if allowed_events:
        if not event_tokens.intersection(allowed_events):
            return None
    if "mysterioussphere" in event_tokens:
        try:
            current_hp = int(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
            max_hp = int(getattr(getattr(env, "player", None), "max_hp", 0) or 0)
        except (TypeError, ValueError):
            current_hp = 0
            max_hp = 0
        sphere_floor_min = _env_int("SPIRECOMM_EVENT_ROLLOUT_MYSTERIOUS_SPHERE_FLOOR_MIN", 45)
        sphere_hp_ratio_max = _env_float("SPIRECOMM_EVENT_ROLLOUT_MYSTERIOUS_SPHERE_HP_RATIO_MAX", 0.40)
        low_hp = max_hp > 0 and sphere_hp_ratio_max > 0.0 and float(current_hp) / float(max_hp) <= sphere_hp_ratio_max
        if root_floor < sphere_floor_min and not low_hp:
            return None
    try:
        chosen_index = next(index for index, candidate in enumerate(actions) if candidate is chosen_action or candidate == chosen_action)
    except StopIteration:
        return None
    floor_min = _env_int("SPIRECOMM_EVENT_ROLLOUT_FLOOR_MIN", 0)
    floor_max_limit = _env_int("SPIRECOMM_EVENT_ROLLOUT_FLOOR_MAX", 0)
    if floor_min > 0 and root_floor < floor_min:
        return None
    if floor_max_limit > 0 and root_floor > floor_max_limit:
        return None
    topk = max(1, _env_int("SPIRECOMM_EVENT_ROLLOUT_TOPK", 4))
    ranked = sorted(range(len(actions)), key=lambda index: float(scores[index]), reverse=True)
    candidate_indices = {chosen_index, *ranked[:topk]}
    if _env_bool("SPIRECOMM_EVENT_ROLLOUT_ALL", True):
        candidate_indices.update(range(len(actions)))
    max_model_score_drop = _event_rollout_model_score_drop_limit(event_tokens)
    if math.isfinite(max_model_score_drop):
        try:
            chosen_model_score = float(scores[chosen_index])
        except (TypeError, ValueError, IndexError):
            chosen_model_score = float("-inf")
        candidate_indices = {
            index
            for index in candidate_indices
            if index == chosen_index or float(scores[index]) >= chosen_model_score - max_model_score_drop
        }
    candidate_indices = {
        index
        for index in candidate_indices
        if index == chosen_index or _event_rollout_allows_candidate_action(chosen_action, actions[index])
    }
    if len(candidate_indices) <= 1:
        return None
    try:
        root_blob = clone_env_blob(env, strip_debug_history=True)
    except Exception:
        return None
    root_resource_state = (
        _event_rollout_resource_state(env)
        if (
            _env_bool("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_RESOURCE_BONUS", False)
            or _env_bool("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_DECK_BONUS", False)
        )
        else None
    )
    max_steps = max(1, _env_int("SPIRECOMM_EVENT_ROLLOUT_MAX_STEPS", 260))
    max_floor_delta = max(1, _env_int("SPIRECOMM_EVENT_ROLLOUT_MAX_FLOOR_DELTA", 3))
    max_floor = root_floor + max_floor_delta
    rollout_scores: dict[int, float] = {}
    rollout_immediate_bonuses: dict[int, float] = {}
    for index in sorted(candidate_indices):
        action = actions[index]
        action_name_token = normalize_token(action.get("name") or action.get("label") or "")
        if (
            normalize_token(action.get("event_id") or "") == "thelibrary"
            and (action_name_token == "heal" or action_name_token.startswith("sleep"))
        ):
            library_sleep_floor_min = _env_int("SPIRECOMM_EVENT_ROLLOUT_LIBRARY_SLEEP_FLOOR_MIN", 0)
            if library_sleep_floor_min > 0 and root_floor < library_sleep_floor_min:
                continue
        if normalize_token(action.get("event_id") or "") == "thelibrary" and action_name_token == "read":
            library_read_floor_min = _env_int("SPIRECOMM_EVENT_ROLLOUT_LIBRARY_READ_FLOOR_MIN", 0)
            if library_read_floor_min > 0 and root_floor < library_read_floor_min:
                continue
        rollout_score, immediate_bonus = _event_rollout_score_candidate(
            root_blob,
            action,
            selectors,
            max_steps=max_steps,
            max_floor=max_floor,
            root_resource_state=root_resource_state,
        )
        rollout_scores[index] = rollout_score
        rollout_immediate_bonuses[index] = immediate_bonus
    if not rollout_scores:
        return None
    best_index = max(rollout_scores, key=lambda index: rollout_scores[index])
    best_score = float(rollout_scores[best_index])
    chosen_score = float(rollout_scores.get(chosen_index, float("-inf")))
    min_advantage = _env_float("SPIRECOMM_EVENT_ROLLOUT_MIN_ADVANTAGE", 1000.0)
    immediate_min_advantage_raw = os.environ.get("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_MIN_ADVANTAGE")
    if (
        immediate_min_advantage_raw is not None
        and abs(float(rollout_immediate_bonuses.get(best_index, 0.0))) > 1.0e-9
    ):
        min_advantage = _env_float("SPIRECOMM_EVENT_ROLLOUT_IMMEDIATE_MIN_ADVANTAGE", min_advantage)
    if "mindbloom" in event_tokens:
        allowed_mindbloom_actions = {
            normalize_token(token)
            for token in str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MINDBLOOM_ALLOWED_ACTIONS", "I Am Rich")).split(",")
            if normalize_token(token)
        }
        if allowed_mindbloom_actions:
            best_action_token = normalize_token(actions[best_index].get("name") or actions[best_index].get("label") or "")
            if best_action_token not in allowed_mindbloom_actions:
                return None
    if math.isfinite(best_score) and best_index != chosen_index and best_score - chosen_score >= min_advantage:
        return actions[best_index], scores, "event_rollout_rerank"
    return None


def _combat_rollout_may_consider(env: Any, actions: list[dict[str, Any]]) -> bool:
    if _combat_rollout_rerank_topk() <= 1 or len(actions) <= 1:
        return False
    phase = str(getattr(env, "phase", ""))
    if phase != "COMBAT":
        return False
    allowed_rooms_raw = os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES")
    if allowed_rooms_raw is None:
        allowed_rooms = {"MonsterRoomBoss"}
    else:
        allowed_rooms = {part.strip() for part in allowed_rooms_raw.split(",") if part.strip()}
    if allowed_rooms:
        room_type = str(getattr(env, "current_room_type", "") or "")
        if room_type not in allowed_rooms:
            return False
    floor_max = _env_int("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", 16)
    if floor_max > 0:
        try:
            root_floor = int(getattr(env, "floor", 0) or 0)
        except (TypeError, ValueError):
            root_floor = 0
        if root_floor > floor_max:
            return False
    if _env_bool("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", False) and not _combat_rollout_dangerous_state(
        _combat_rollout_state(env)
    ):
        return False
    return True


def _maybe_rerank_combat_with_rollout(
    env: Any,
    actions: list[dict[str, Any]],
    scores: list[float],
    chosen: dict[str, Any],
    selectors: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    debug_enabled = _env_bool("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DEBUG", False)
    combat_selector = selectors.get("combat") if debug_enabled else None

    def _set_debug(reason: str, **payload: Any) -> None:
        if combat_selector is None:
            return
        info = {"reason": reason}
        info.update(payload)
        try:
            setattr(combat_selector, "last_rollout_rerank_debug", info)
        except Exception:
            pass

    if combat_selector is not None:
        _set_debug("not_evaluated")
    topk = _combat_rollout_rerank_topk()
    if topk <= 1 or len(actions) <= 1 or len(scores) != len(actions):
        _set_debug("disabled_or_invalid", topk=topk, action_count=len(actions), score_count=len(scores))
        return chosen, False
    phase = str(getattr(env, "phase", ""))
    if phase != "COMBAT":
        _set_debug("phase", phase=phase)
        return chosen, False
    allowed_rooms_raw = os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES")
    root_state = _combat_rollout_state(env)
    if allowed_rooms_raw is None:
        allowed_rooms = {"MonsterRoomBoss"}
    else:
        allowed_rooms = {part.strip() for part in allowed_rooms_raw.split(",") if part.strip()}
    if allowed_rooms:
        room_type = str(root_state.get("room_type") or getattr(env, "current_room_type", "") or "")
        if room_type not in allowed_rooms:
            _set_debug("room_type", room_type=room_type, allowed_rooms=sorted(allowed_rooms))
            return chosen, False
    floor_max = _env_int("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", 16)
    if floor_max > 0:
        try:
            root_floor = int(root_state.get("floor") or getattr(env, "floor", 0) or 0)
        except (TypeError, ValueError):
            root_floor = 0
        if root_floor > floor_max:
            _set_debug("floor_max", floor=root_floor, floor_max=floor_max)
            return chosen, False
    if _env_bool("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", False) and not _combat_rollout_dangerous_state(root_state):
        _set_debug("not_dangerous")
        return chosen, False
    chosen_index = _combat_action_index(actions, chosen)
    if chosen_index is None:
        _set_debug("chosen_index_missing")
        return chosen, False
    top_kind_filter = _combat_rollout_env_set("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS")
    if not top_kind_filter and os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS") is None:
        top_kind_filter = {"card"}
    if top_kind_filter:
        chosen_kind = str(actions[chosen_index].get("kind") or "").strip()
        if chosen_kind not in top_kind_filter:
            _set_debug("top_kind", chosen_index=chosen_index, chosen_kind=chosen_kind, allowed=sorted(top_kind_filter))
            return chosen, False
    top_card_type_filter = {token.upper() for token in _combat_rollout_env_set("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES")}
    if not top_card_type_filter and os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES") is None:
        top_card_type_filter = {"ATTACK"}
    if top_card_type_filter and str(actions[chosen_index].get("kind") or "").strip() == "card":
        chosen_card_type = _combat_rollout_action_card_type(root_state, actions[chosen_index])
        if chosen_card_type not in top_card_type_filter:
            _set_debug("top_card_type", chosen_index=chosen_index, card_type=chosen_card_type, allowed=sorted(top_card_type_filter))
            return chosen, False
    try:
        numeric_scores = [float(score) for score in scores]
    except (TypeError, ValueError):
        _set_debug("non_numeric_scores")
        return chosen, False
    chosen_score = numeric_scores[chosen_index]
    if not math.isfinite(chosen_score):
        _set_debug("nonfinite_chosen_score", chosen_index=chosen_index, chosen_score=chosen_score)
        return chosen, False
    top_score_max = _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_SCORE_MAX", 0.0)
    if chosen_score > top_score_max:
        _set_debug("top_score", chosen_index=chosen_index, chosen_score=chosen_score, top_score_max=top_score_max)
        return chosen, False
    margin_max = max(0.0, _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", 8.0))
    min_advantage = max(0.0, _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", 1000.0))
    max_steps = max(1, _env_int("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", 80))
    require_chosen_not_clear = _env_bool(
        "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_NOT_CLEAR",
        True,
    )
    chosen_clear_full_policy = _env_bool(
        "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CHOSEN_CLEAR_FULL_POLICY",
        True,
    )
    candidate_kind_filter = _combat_rollout_env_set("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS")
    if not candidate_kind_filter and os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS") is None:
        candidate_kind_filter = {"card", "potion"}
    ranked_indices = sorted(range(len(actions)), key=lambda index: numeric_scores[index], reverse=True)
    low_margin_threshold = _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LOW_MARGIN_THRESHOLD", 0.35)
    low_margin_topk = max(topk, _env_int("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LOW_MARGIN_TOPK", 5))
    if topk < low_margin_topk and low_margin_threshold >= 0.0 and ranked_indices and ranked_indices[0] == chosen_index:
        for other_index in ranked_indices[1:]:
            other_score = numeric_scores[other_index]
            if math.isfinite(other_score):
                if chosen_score - other_score <= low_margin_threshold:
                    topk = low_margin_topk
                break
    high_score_margin_threshold = _env_float(
        "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_HIGH_SCORE_MARGIN_THRESHOLD",
        1.0,
    )
    high_score_min = _env_float("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_HIGH_SCORE_MIN", -1.0)
    if (
        topk < low_margin_topk
        and high_score_margin_threshold >= 0.0
        and chosen_score >= high_score_min
        and ranked_indices
        and ranked_indices[0] == chosen_index
    ):
        for other_index in ranked_indices[1:]:
            other_score = numeric_scores[other_index]
            if math.isfinite(other_score):
                if chosen_score - other_score <= high_score_margin_threshold:
                    topk = low_margin_topk
                break
    candidate_indices: list[int] = []
    for index in ranked_indices:
        if len(candidate_indices) >= topk:
            break
        if candidate_kind_filter and index != chosen_index:
            candidate_kind = str(actions[index].get("kind") or "").strip()
            if candidate_kind not in candidate_kind_filter:
                continue
        if chosen_score - numeric_scores[index] <= margin_max:
            candidate_indices.append(index)
    if chosen_index not in candidate_indices:
        candidate_indices.append(chosen_index)
    if len(candidate_indices) <= 1:
        _set_debug(
            "not_enough_candidates",
            chosen_index=chosen_index,
            chosen_score=chosen_score,
            topk=topk,
            candidate_indices=list(candidate_indices),
        )
        return chosen, False
    root_turn = _combat_rollout_turn(root_state)
    _root_live_count, root_live_hp = _combat_rollout_live_monster_stats(root_state)
    try:
        root_blob = clone_env_blob(env, strip_debug_history=True)
    except Exception:
        _set_debug("clone_failed")
        return chosen, False

    required_chosen_outcomes = _combat_rollout_env_set("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_OUTCOMES")
    rollout_scores: dict[int, float] = {}
    rollout_outcomes: dict[int, str] = {}
    if required_chosen_outcomes:
        rollout_score, _steps, outcome = _combat_rollout_score_candidate(
            root_blob,
            actions[chosen_index],
            selectors,
            root_turn=root_turn,
            root_live_hp=root_live_hp,
            max_steps=max_steps,
        )
        if math.isfinite(rollout_score):
            rollout_scores[chosen_index] = rollout_score
            rollout_outcomes[chosen_index] = outcome
        if rollout_outcomes.get(chosen_index) not in required_chosen_outcomes:
            _set_debug(
                "chosen_outcome_filter",
                chosen_index=chosen_index,
                chosen_outcome=rollout_outcomes.get(chosen_index),
                required_chosen_outcomes=sorted(required_chosen_outcomes),
                candidate_indices=list(candidate_indices),
                rollout_scores={str(index): score for index, score in rollout_scores.items()},
                rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
            )
            return chosen, False
    for index in candidate_indices:
        if index in rollout_scores:
            continue
        rollout_score, _steps, outcome = _combat_rollout_score_candidate(
            root_blob,
            actions[index],
            selectors,
            root_turn=root_turn,
            root_live_hp=root_live_hp,
            max_steps=max_steps,
        )
        if math.isfinite(rollout_score):
            rollout_scores[index] = rollout_score
            rollout_outcomes[index] = outcome
    if chosen_index not in rollout_scores or len(rollout_scores) <= 1:
        _set_debug(
            "rollout_failed",
            chosen_index=chosen_index,
            candidate_indices=list(candidate_indices),
            rollout_scores={str(index): score for index, score in rollout_scores.items()},
            rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
        )
        return chosen, False
    if required_chosen_outcomes and rollout_outcomes.get(chosen_index) not in required_chosen_outcomes:
        _set_debug(
            "chosen_outcome_filter",
            chosen_index=chosen_index,
            chosen_outcome=rollout_outcomes.get(chosen_index),
            required_chosen_outcomes=sorted(required_chosen_outcomes),
            candidate_indices=list(candidate_indices),
            rollout_scores={str(index): score for index, score in rollout_scores.items()},
            rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
        )
        return chosen, False
    required_best_outcomes = _combat_rollout_env_set("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_BEST_OUTCOMES")
    selectable_indices = list(rollout_scores)
    if required_best_outcomes:
        selectable_indices = [
            index
            for index in selectable_indices
            if index != chosen_index and rollout_outcomes.get(index) in required_best_outcomes
        ]
        if not selectable_indices:
            _set_debug(
                "best_outcome_filter",
                chosen_index=chosen_index,
                required_best_outcomes=sorted(required_best_outcomes),
                candidate_indices=list(candidate_indices),
                rollout_scores={str(index): score for index, score in rollout_scores.items()},
                rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
            )
            return chosen, False
    best_index = max(selectable_indices, key=lambda index: (rollout_scores[index], numeric_scores[index], -index))
    if best_index == chosen_index:
        _set_debug(
            "chosen_best",
            chosen_index=chosen_index,
            chosen_score=chosen_score,
            candidate_indices=list(candidate_indices),
            rollout_scores={str(index): score for index, score in rollout_scores.items()},
            rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
        )
        return chosen, False
    rollout_delta = rollout_scores[best_index] - rollout_scores[chosen_index]
    if rollout_delta < min_advantage:
        _set_debug(
            "min_advantage",
            chosen_index=chosen_index,
            best_index=best_index,
            delta=rollout_delta,
            min_advantage=min_advantage,
            candidate_indices=list(candidate_indices),
            rollout_scores={str(index): score for index, score in rollout_scores.items()},
            rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
        )
        return chosen, False
    if require_chosen_not_clear and chosen_clear_full_policy:
        _score, _steps, chosen_full_outcome = _combat_rollout_score_candidate(
            root_blob,
            actions[chosen_index],
            selectors,
            root_turn=root_turn,
            root_live_hp=root_live_hp,
            max_steps=max_steps,
            light_policy_override=False,
        )
        if chosen_full_outcome in {"combat_clear", "run_complete"}:
            _set_debug(
                "chosen_full_policy_clears",
                chosen_index=chosen_index,
                best_index=best_index,
                delta=rollout_delta,
                chosen_full_outcome=chosen_full_outcome,
                rollout_scores={str(index): score for index, score in rollout_scores.items()},
                rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
            )
            return chosen, False
    if require_chosen_not_clear and rollout_outcomes.get(chosen_index) in {"combat_clear", "run_complete"}:
        _set_debug(
            "chosen_light_policy_clears",
            chosen_index=chosen_index,
            best_index=best_index,
            delta=rollout_delta,
            rollout_scores={str(index): score for index, score in rollout_scores.items()},
            rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
        )
        return chosen, False
    _set_debug(
        "reranked",
        chosen_index=chosen_index,
        best_index=best_index,
        delta=rollout_delta,
        rollout_scores={str(index): score for index, score in rollout_scores.items()},
        rollout_outcomes={str(index): outcome for index, outcome in rollout_outcomes.items()},
    )
    return actions[best_index], True


def _choose_combat_required(env: Any, selectors: dict[str, Any], *, return_scores: bool = True):
    selector = selectors.get("combat")
    legal_actions_env = getattr(selector, "legal_actions_env", None)
    if callable(legal_actions_env):
        actions = legal_actions_env(env)
    else:
        actions = env.legal_actions()
    if len(actions) == 1:
        return actions[0], [], "forced_single"
    if not getattr(selector, "available", False):
        _raise_model_required(
            env,
            reason="combat_selector_unavailable",
            actions=actions,
            selector_name="combat",
            selectors=selectors,
        )
    selector_return_scores = bool(return_scores or _combat_rollout_may_consider(env, actions))
    if hasattr(selector, "choose_env"):
        chosen, scores = selector.choose_env(env, return_scores=selector_return_scores, legal_actions=actions)
    else:
        chosen, scores = selector.choose(env.state(), actions)
    if chosen is None:
        reason = "combat_selector_returned_no_choice"
        selector_error = str(getattr(selector, "last_error", "") or "")
        if selector_error:
            reason = f"{reason}:{selector_error}"
        _raise_model_required(
            env,
            reason=reason,
            actions=actions,
            selector_name="combat",
            selectors=selectors,
        )
    source_parts = ["combat"]
    if bool(getattr(selector, "last_rescue_used", False)):
        source_parts.append("rescue")
    if bool(getattr(selector, "last_teacher_blend_used", False)):
        source_parts.append("teacher_blend")
    elif bool(getattr(selector, "last_teacher_fallback_used", False)):
        source_parts.append("teacher_fallback")
    if bool(getattr(selector, "last_dangerous_end_bias_used", False)):
        source_parts.append("dangerous_end_bias")
    if bool(getattr(selector, "last_potion_over_end_used", False)):
        source_parts.append("potion_over_end")
    if bool(getattr(selector, "last_block_over_end_used", False)):
        source_parts.append("block_over_end")
    if bool(getattr(selector, "last_sharp_hide_danger_guard_used", False)):
        source_parts.append("sharp_hide_danger_guard")
    if bool(getattr(selector, "last_lethal_card_over_setup_used", False)):
        source_parts.append("lethal_card_over_setup")
    if bool(getattr(selector, "last_lethal_sequence_preserve_used", False)):
        source_parts.append("lethal_sequence_preserve")
    if bool(getattr(selector, "last_setup_power_over_basic_attack_used", False)):
        source_parts.append("setup_power_over_basic_attack")
    if bool(getattr(selector, "last_high_block_progress_guard_used", False)):
        source_parts.append("high_block_progress_guard")
    if bool(getattr(selector, "last_monster_block_progress_guard_used", False)):
        source_parts.append("monster_block_progress_guard")
    if bool(getattr(selector, "last_danger_block_progress_guard_used", False)):
        source_parts.append("danger_block_progress_guard")
    if bool(getattr(selector, "last_gremlin_nob_skill_bias_used", False)):
        source_parts.append("gremlin_nob_skill_bias")
    if bool(getattr(selector, "last_short_win_guard_used", False)):
        source_parts.append("short_win_guard")
    if bool(getattr(selector, "last_branch_advisor_used", False)):
        source_parts.append("branch_advisor")
    if bool(getattr(selector, "last_suicidal_action_guard_used", False)):
        source_parts.append("suicidal_action_guard")
    if bool(getattr(selector, "last_forced_turn_survival_guard_used", False)):
        source_parts.append("forced_turn_survival_guard")
    if bool(getattr(selector, "last_policy_survival_guard_used", False)):
        source_parts.append("policy_survival_guard")
    if bool(getattr(selector, "last_post_forced_turn_survival_guard_used", False)):
        source_parts.append("post_forced_turn_survival_guard")
    if bool(getattr(selector, "last_post_action_survival_guard_used", False)):
        source_parts.append("post_action_survival_guard")
    if bool(getattr(selector, "last_delayed_death_guard_used", False)):
        source_parts.append("delayed_death_guard")
    if bool(getattr(selector, "last_survival_potion_rescue_used", False)):
        source_parts.append("survival_potion_rescue")
    if bool(getattr(selector, "last_suicidal_end_guard_used", False)):
        source_parts.append("suicidal_end_guard")
    chosen, rollout_reranked = _maybe_rerank_combat_with_rollout(env, actions, scores, chosen, selectors)
    if _env_bool("SPIRECOMM_FAST_FLOOR_SKIP_COMBAT_SOURCE_FLAGS", False):
        return chosen, scores if return_scores else [], "combat_rollout_rerank" if rollout_reranked else "combat"
    if rollout_reranked:
        source_parts.append("rollout_rerank")
    source = "_".join(source_parts)
    return chosen, scores if return_scores else [], source


def _card_select_selector_for_mode(mode: str) -> tuple[str | None, str | None]:
    normalized = mode.strip().lower()
    if normalized in {"upgrade", "armaments"}:
        return "upgrade_target", "upgrade_target" if normalized == "upgrade" else "armaments_target"
    if normalized in {"secret_technique", "secret_weapon"}:
        # These effects fetch a useful card from the draw pile. Until a
        # dedicated draw-pile-to-hand selector exists, use the positive card
        # target model rather than the purge/removal model.
        return "upgrade_target", f"{normalized}_target"
    if normalized in CARD_REWARD_CARD_SELECT_MODES:
        return "card_reward", f"{normalized}_target"
    if normalized in {
        "purge",
        "remove",
        "transform",
        "burning_pact",
        "elixir",
        "prepared",
        "true_grit",
        "warcry",
        "put_on_deck",
        "gambling_chip",
    }:
        source = f"{normalized}_target"
        return "purge_target", source
    return None, None


def _positive_card_target_bias(candidate: dict[str, Any]) -> float:
    key = normalize_token(candidate.get("card_id") or candidate.get("name"))
    if key in STARTER_BASIC_CARD_KEYS:
        return POSITIVE_CARD_TARGET_STARTER_BASIC_BIAS
    return 0.0


def _card_select_cost(candidate: dict[str, Any]) -> int:
    cost = candidate.get("cost_for_turn")
    if cost is None:
        cost = candidate.get("cost")
    if cost is None:
        cost = candidate.get("base_cost")
    try:
        return max(0, int(cost))
    except (TypeError, ValueError):
        return 0


def _free_card_select_cost_bias(mode: str, candidate: dict[str, Any]) -> float:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in FREE_CARD_SELECT_COST_BIAS_MODES:
        return 0.0
    return math.sqrt(float(_card_select_cost(candidate)))


def _true_grit_target_search_score(env: Any, steps: int) -> float:
    phase = str(getattr(env, "phase", ""))
    hp = float(getattr(getattr(env, "player", None), "current_hp", 0) or 0)
    if phase == "GAME_OVER" or hp <= 0.0:
        return -100000.0 - float(steps)
    # This search only resolves an in-combat exhaust target. Prefer combat
    # survival first, then preserve HP; step count is just a deterministic tie-breaker.
    return 100000.0 + hp * 10.0 - float(steps) * 0.01


def _true_grit_target_priority(action: dict[str, Any]) -> float:
    card_key = normalize_token(action.get("name") or action.get("card_id"))
    card_type = str(action.get("type") or "").strip().upper()
    if card_type in {"CURSE", "STATUS"}:
        return 100.0
    if card_key == "strike":
        return 60.0
    if card_key == "defend":
        return 25.0
    if card_type == "ATTACK":
        return 15.0
    if card_key in {
        "impervious",
        "powerthrough",
        "secondwind",
        "truegrit",
        "armaments",
        "entrench",
        "shrugitoff",
        "flamebarrier",
        "ghostlyarmor",
    }:
        return -30.0
    if card_type == "POWER":
        return -50.0
    return 0.0


def _true_grit_target_runtime_bias(action: dict[str, Any]) -> float:
    if not _env_bool("SPIRECOMM_TRUE_GRIT_TARGET_BIAS", False):
        return 0.0
    card_key = normalize_token(action.get("name") or action.get("card_id"))
    card_type = str(action.get("type") or "").strip().upper()
    if card_type in {"CURSE", "STATUS"}:
        return _env_float("SPIRECOMM_TRUE_GRIT_TARGET_CURSE_STATUS_BIAS", 1.0)
    if card_key == "strike":
        return _env_float("SPIRECOMM_TRUE_GRIT_TARGET_STRIKE_BIAS", 2.0)
    if card_key == "defend":
        return _env_float("SPIRECOMM_TRUE_GRIT_TARGET_DEFEND_BIAS", -2.0)
    if card_key in {
        "impervious",
        "powerthrough",
        "secondwind",
        "truegrit",
        "armaments",
        "entrench",
        "shrugitoff",
        "flamebarrier",
        "ghostlyarmor",
    }:
        return _env_float("SPIRECOMM_TRUE_GRIT_TARGET_BLOCK_ENGINE_BIAS", -2.0)
    if card_type == "POWER":
        return _env_float("SPIRECOMM_TRUE_GRIT_TARGET_POWER_BIAS", -4.0)
    if card_type == "ATTACK":
        return _env_float("SPIRECOMM_TRUE_GRIT_TARGET_ATTACK_BIAS", 0.25)
    return 0.0


def _choose_true_grit_target_by_combat_search(
    env: Any,
    actions: list[dict[str, Any]],
    selectors: dict[str, Any],
    *,
    return_scores: bool,
) -> tuple[dict[str, Any], list[float], str] | None:
    if not _env_bool("SPIRECOMM_TRUE_GRIT_TARGET_COMBAT_SEARCH", False):
        return None
    if _env_bool("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_ACTIVE", False):
        return None
    if len(actions) <= 1:
        return None
    max_steps = max(1, _env_int("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_MAX_STEPS", 80))
    max_targets = max(1, _env_int("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_MAX_TARGETS", 10))
    min_hp_gain = max(0.0, _env_float("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_MIN_HP_GAIN", 0.0))
    tiebreak_hp_window = max(0.0, _env_float("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_TIEBREAK_HP_WINDOW", 0.0))
    limited_actions = list(actions[:max_targets])
    active_previous = os.environ.get("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_ACTIVE")
    os.environ["SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_ACTIVE"] = "1"
    scores: list[float] = []
    try:
        for action in limited_actions:
            try:
                branch = pickle.loads(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
                branch.step(dict(action))
                steps = 0
                while str(getattr(branch, "phase", "")) == "COMBAT" and steps < max_steps:
                    next_action, _, _source = choose_model_required_action(branch, selectors, return_scores=False)
                    branch.step(next_action)
                    steps += 1
                scores.append(_true_grit_target_search_score(branch, steps))
            except Exception:
                scores.append(float("-inf"))
    finally:
        if active_previous is None:
            os.environ.pop("SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_ACTIVE", None)
        else:
            os.environ["SPIRECOMM_TRUE_GRIT_TARGET_SEARCH_ACTIVE"] = active_previous
    if not scores or all(math.isinf(score) and score < 0.0 for score in scores):
        return None
    if len(limited_actions) < len(actions):
        scores.extend([float("-inf")] * (len(actions) - len(limited_actions)))
    index = max(range(len(scores)), key=lambda idx: scores[idx])
    finite_scores = sorted((score for score in scores if math.isfinite(score)), reverse=True)
    if tiebreak_hp_window > 0.0 and math.isfinite(scores[index]):
        floor_score = scores[index] - tiebreak_hp_window * 10.0
        near_indices = [idx for idx, score in enumerate(scores) if math.isfinite(score) and score >= floor_score]
        if len(near_indices) >= 2:
            index = max(
                near_indices,
                key=lambda idx: (_true_grit_target_priority(actions[idx]), scores[idx], -idx),
            )
    elif min_hp_gain > 0.0:
        if len(finite_scores) >= 2 and finite_scores[0] - finite_scores[1] < min_hp_gain * 10.0:
            return None
    return actions[index], scores if return_scores else [], "true_grit_target_search"


def _choose_card_select_required(env: Any, selectors: dict[str, Any], *, return_scores: bool = True):
    actions = env.legal_actions()
    if len(actions) == 1:
        return actions[0], [], "forced_single"
    current_card_select = getattr(env, "current_card_select", None) or {}
    mode = str(current_card_select.get("mode") or "")
    if not mode and actions:
        mode = str(actions[0].get("mode") or "")
    if not mode:
        engine = getattr(getattr(env, "combat", None), "engine", None)
        pending = getattr(engine, "pending_card_select", None) or {}
        mode = str(pending.get("mode") or "")
    if mode.strip().lower() == "pandora_confirm":
        return actions[0], [], "pandora_confirm"
    candidates = [
        {
            "card_id": action.get("card_id"),
            "name": action.get("name"),
            "type": action.get("type"),
            "rarity": action.get("rarity"),
            "upgrades": int(action.get("upgrades") or 0),
            "cost": action.get("cost"),
            "base_cost": action.get("base_cost"),
            "cost_for_turn": action.get("cost_for_turn"),
        }
        for action in actions
    ]
    if mode.strip().lower() == "library":
        selector = selectors.get("card_reward")
        if not getattr(selector, "available", False):
            _raise_model_required(
                env,
                reason="card_select_selector_unavailable",
                actions=actions,
                selector_name="card_reward",
                selectors=selectors,
            )
        result = selector.choose(env.state(), candidates, can_skip=False, return_scores=return_scores)
        if result is None:
            _raise_model_required(
                env,
                reason="card_select_selector_returned_no_choice",
                actions=actions,
                selector_name="card_reward",
                selectors=selectors,
            )
        index = int(result["choice_index"])
        if not 0 <= index < len(actions):
            _raise_model_required(
                env,
                reason="card_select_selector_choice_out_of_range",
                actions=actions,
                selector_name="card_reward",
                selectors=selectors,
            )
        return actions[index], result["scores"], "library_card_select"
    normalized_mode = mode.strip().lower()
    selector_name, source = _card_select_selector_for_mode(mode)
    if selector_name is None or source is None:
        _raise_model_required(
            env,
            reason=f"unsupported_card_select_mode:{mode or 'unknown'}",
            actions=actions,
            selectors=selectors,
        )
    if normalized_mode == "true_grit":
        searched = _choose_true_grit_target_by_combat_search(env, actions, selectors, return_scores=return_scores)
        if searched is not None:
            return searched
    selector = selectors.get(selector_name)
    if not getattr(selector, "available", False):
        _raise_model_required(
            env,
            reason="card_select_selector_unavailable",
            actions=actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    if selector_name == "card_reward" and normalized_mode in CARD_REWARD_CARD_SELECT_MODES:
        result = selector.choose(env.state(), candidates, can_skip=False, return_scores=True)
        if result is not None:
            biased_scores = [
                float(score)
                + (_positive_card_target_bias(candidate) if normalized_mode in POSITIVE_CARD_TARGET_MODES else 0.0)
                + _free_card_select_cost_bias(mode, candidate)
                for score, candidate in zip(list(result.get("scores") or []), candidates)
            ]
            if len(biased_scores) == len(actions):
                result = dict(result)
                result["scores"] = biased_scores if return_scores else []
                result["choice_index"] = max(range(len(biased_scores)), key=lambda idx: biased_scores[idx])
    else:
        selector_return_scores = return_scores or normalized_mode == "true_grit"
        result = selector.choose(env.state(), candidates, return_scores=selector_return_scores)
        if normalized_mode == "true_grit" and result is not None:
            raw_scores = list(result.get("scores") or [])
            if len(raw_scores) == len(actions):
                biased_scores = [
                    float(score) + _true_grit_target_runtime_bias(action)
                    for score, action in zip(raw_scores, actions)
                ]
                result = dict(result)
                result["scores"] = biased_scores if return_scores else []
                result["choice_index"] = max(range(len(biased_scores)), key=lambda idx: biased_scores[idx])
    if result is None:
        _raise_model_required(
            env,
            reason="card_select_selector_returned_no_choice",
            actions=actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    index = int(result["choice_index"])
    if not 0 <= index < len(actions):
        _raise_model_required(
            env,
            reason="card_select_selector_choice_out_of_range",
            actions=actions,
            selector_name=selector_name,
            selectors=selectors,
        )
    return actions[index], result["scores"], source


def choose_model_required_action(
    env: Any,
    selectors: dict[str, Any],
    *,
    return_scores: bool = True,
) -> tuple[dict[str, Any], list[float], str]:
    """Choose an action for trace generation without silent heuristic fallbacks."""

    phase = str(getattr(env, "phase", ""))
    if phase == "COMBAT":
        if not _combat_selector_handles_potions(selectors):
            action, scores = _choose_potion_required(env, selectors, return_scores=return_scores)
            if action is not None:
                return action, scores, "potion"
        return _choose_combat_required(env, selectors, return_scores=return_scores)
    if phase == "CARD_REWARD":
        action, scores, source = choose_reward_screen_action(
            env,
            selectors.get("card_reward"),
            require_model=True,
            selectors=selectors,
            return_scores=return_scores,
        )
        return action, scores, source
    if phase == "MAP":
        return _choose_map_dynamic_programming(env, selectors)
    if phase == "BOSS_RELIC":
        needs_bias_scores = os.environ.get("SPIRECOMM_BOSS_RELIC_SCORE_BIAS_JSON") is None or bool(
            os.environ.get("SPIRECOMM_BOSS_RELIC_SCORE_BIAS_JSON")
        )
        needs_rollout_scores = _env_bool("SPIRECOMM_BOSS_RELIC_ROLLOUT_RERANK", True)
        action, scores, source = _choose_run_choice_required(
            env,
            selectors,
            "boss_relic",
            "boss_relic",
            return_scores=return_scores or needs_bias_scores or needs_rollout_scores,
        )
        actions = env.legal_actions()
        biased_action, scores, source = _maybe_apply_boss_relic_score_bias(actions, scores, source, env)
        if biased_action is not None:
            action = biased_action
        reranked = _maybe_rerank_boss_relic_rollout(env, actions, scores, action, selectors)
        if reranked is not None:
            return reranked
        return action, scores, source
    if phase == "CAMPFIRE":
        needs_fast_guard_scores = _env_bool("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD", True)
        action, scores, source = _choose_run_choice_required(
            env,
            selectors,
            "campfire",
            "campfire",
            return_scores=return_scores or needs_fast_guard_scores,
        )
        action, scores, source = _maybe_apply_campfire_low_hp_rest_guard(env, action, scores, source, selectors)
        return attach_card_target(env, action, selectors, require_model=True), scores, source
    if phase == "SHOP":
        if _shop_value_policy_enabled():
            action, scores, source = _choose_shop_value_policy_required(env, selectors, return_scores=return_scores)
        else:
            action, scores, source = _choose_run_choice_required(env, selectors, "shop", "shop", return_scores=return_scores)
        return attach_card_target(env, action, selectors, require_model=True), scores, source
    if phase == "CARD_SELECT":
        return _choose_card_select_required(env, selectors, return_scores=return_scores)
    if phase == "EVENT":
        return _choose_event_choice(
            env,
            selectors.get("event"),
            require_model=True,
            selectors=selectors,
            return_scores=return_scores,
        )
    if phase == "NEOW":
        return _choose_neow_weighted_action(env, selectors=selectors)
    if phase == "TREASURE":
        actions = env.legal_actions()
        open_actions = [action for action in actions if action.get("name") == "OPEN_CHEST"]
        if len(open_actions) == 1:
            return open_actions[0], [], "treasure_open_chest"
        return _forced_single_or_raise(
            env,
            actions,
            reason="treasure_has_branching_actions",
            selectors=selectors,
        )
    if phase == "CHEST":
        actions = env.legal_actions()
        return _forced_single_or_raise(
            env,
            actions,
            reason="chest_has_branching_actions_without_selector",
            selectors=selectors,
        )
    actions = env.legal_actions()
    return _forced_single_or_raise(
        env,
        actions,
        reason="unsupported_branching_phase_without_selector",
        selectors=selectors,
    )
