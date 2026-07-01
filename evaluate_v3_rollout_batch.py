#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from spirecomm.ai.runtime_decision import (
    DEFAULT_BOSS_RELIC_SCORE_BIAS,
    apply_runtime_latency_profile,
    build_runtime_selectors,
    choose_model_required_action,
)
from spirecomm.native_sim_v3 import NativeRunEnv


_SELECTORS: dict[str, Any] | None = None
_CONFIG: dict[str, Any] = {}
_START_SNAPSHOT_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_NATIVE_CONTENT_PREWARMED = False
_RUN_PROGRESS_HOOK: Any | None = None


TERMINAL_PHASES = {"GAME_OVER", "COMPLETE", "VICTORY"}
ACTION_METRIC_KEYS = (
    "sources",
    "potion_action_count",
    "potion_actions_by_id",
    "potion_actions_by_room_type",
    "shop_action_count",
    "shop_actions_by_item_kind",
    "shop_spend_by_item_kind",
    "shop_spend_total",
)


def _disable_gc_for_hot_worker_if_enabled() -> None:
    if str(os.environ.get("SPIRECOMM_FAST_DISABLE_GC", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    gc.disable()


def _prewarm_native_content_caches() -> None:
    global _NATIVE_CONTENT_PREWARMED
    if _NATIVE_CONTENT_PREWARMED:
        return
    try:
        from spirecomm.native_sim_v3.content.cards import (
            card_catalog,
            card_pools,
            initialize_runtime_card_pools,
            initialize_source_card_pools,
            source_card_pools,
            starter_deck,
        )

        card_catalog()
        card_pools("IRONCLAD")
        source_card_pools("IRONCLAD")
        initialize_runtime_card_pools("IRONCLAD")
        initialize_source_card_pools(character="IRONCLAD")
        starter_deck("IRONCLAD")
    except Exception:
        pass
    try:
        from spirecomm.native_sim_v3.content.relics import (
            relic_catalog,
            relic_pools,
            relic_scope_ids,
            relic_scope_order_ids,
            relic_spawn_rules,
            starter_relics,
        )

        relic_catalog()
        relic_scope_order_ids()
        relic_scope_ids()
        relic_pools("IRONCLAD")
        relic_spawn_rules()
        starter_relics("IRONCLAD")
    except Exception:
        pass
    try:
        from spirecomm.native_sim_v3.content.potions import (
            ironclad_potion_pool,
            potion_name_map,
            potion_pool,
            potion_rarity_map,
            potion_requires_target_map,
            potion_scopes,
        )

        potion_rarity_map()
        potion_name_map()
        potion_requires_target_map()
        potion_scopes()
        potion_pool("IRONCLAD")
        ironclad_potion_pool()
    except Exception:
        pass
    try:
        from spirecomm.native_sim_v3.content.encounters import encounter_catalog
        from spirecomm.native_sim_v3.content.events import (
            abstract_dungeon_event_gate_rules,
            abstract_dungeon_shrine_gate_rules,
            dungeon_event_ids,
            dungeon_shrine_ids,
            event_catalog,
            special_one_time_event_ids,
        )
        from spirecomm.native_sim_v3.content.map_rules import map_rules
        from spirecomm.native_sim_v3.content.reward_rules import (
            card_blizz_rules,
            post_combat_potion_rules,
            potion_roll_rules,
        )
        from spirecomm.native_sim_v3.content.room_reward_rules import room_reward_rules
        from spirecomm.native_sim_v3.content.shop import shop_rules

        encounter_catalog()
        event_catalog()
        for dungeon_id in ("Exordium", "TheCity", "TheBeyond"):
            dungeon_event_ids(dungeon_id)
            dungeon_shrine_ids(dungeon_id)
        special_one_time_event_ids(True)
        abstract_dungeon_event_gate_rules()
        abstract_dungeon_shrine_gate_rules()
        map_rules()
        potion_roll_rules()
        card_blizz_rules()
        post_combat_potion_rules()
        room_reward_rules()
        shop_rules()
    except Exception:
        pass
    _NATIVE_CONTENT_PREWARMED = True


def _jsonable_action(action: dict[str, Any]) -> dict[str, Any]:
    payload = dict(action)
    card = payload.get("card")
    if isinstance(card, dict):
        payload["card"] = {
            key: card.get(key)
            for key in (
                "card_id",
                "name",
                "type",
                "rarity",
                "cost",
                "upgrades",
                "base_damage",
                "base_block",
                "base_magic",
            )
            if key in card
        }
    return payload


def _progress_signature(env: NativeRunEnv) -> tuple[Any, ...]:
    shop = getattr(env, "current_shop", None)
    shop_signature: tuple[Any, ...] = ()
    if shop is not None:
        shop_signature = (
            tuple(str(card.get("card_id") or card.get("id") or card.get("name") or "") for card in list(shop.cards or [])),
            tuple(str(relic.get("relic_id") or relic.get("id") or relic.get("name") or "") for relic in list(shop.relics or [])),
            tuple(str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "") for potion in list(shop.potions or [])),
            bool(getattr(shop, "purge_available", False)),
        )
    event = getattr(env, "current_event", None)
    return (
        str(getattr(env, "phase", "")),
        int(getattr(env, "floor", 0)),
        int(getattr(getattr(env, "player", None), "current_hp", 0)),
        int(getattr(env, "gold", 0)),
        str(getattr(env, "current_room_type", "")),
        tuple(str(potion.get("potion_id") or potion.get("id") or potion.get("name") or "") for potion in list(getattr(env, "potions", []) or [])),
        len(list(getattr(env, "deck", []) or [])),
        len(list(getattr(env, "reward_potions", []) or [])),
        len(list(getattr(env, "reward_relics", []) or [])),
        len(list(getattr(env, "reward_cards", []) or [])),
        bool(getattr(env, "reward_card_screen_open", False)),
        getattr(event, "event_id", None),
        getattr(event, "screen", None),
        shop_signature,
    )


def _combat_progress_signature(env: NativeRunEnv) -> tuple[Any, ...] | None:
    if str(getattr(env, "phase", "")) != "COMBAT":
        return None
    combat = getattr(env, "combat", None)
    state = getattr(combat, "state", None)
    if state is None:
        return None
    player = getattr(state, "player", None)
    monsters = []
    for monster in list(getattr(state, "monsters", []) or []):
        meta = getattr(monster, "meta", {}) or {}
        escaped = bool(meta.get("escaped", False)) if isinstance(meta, dict) else False
        half_dead = bool(getattr(monster, "half_dead", False))
        monsters.append(
            (
                str(getattr(monster, "id", "") or getattr(monster, "name", "")),
                int(getattr(monster, "current_hp", 0)),
                bool(escaped),
                bool(half_dead),
            )
        )
    return (
        str(getattr(env, "phase", "")),
        int(getattr(env, "floor", 0)),
        int(getattr(player, "current_hp", getattr(getattr(env, "player", None), "current_hp", 0))),
        tuple(monsters),
    )


def _init_worker(config: dict[str, Any]) -> None:
    global _CONFIG, _SELECTORS
    _CONFIG = dict(config)
    _disable_gc_for_hot_worker_if_enabled()
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
        ("SPIRECOMM_SHOP_VALUE_ITEM_BIAS_JSON", "shop_value_item_bias_json"),
        (
            "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_GATE",
            "shop_value_future_shop_reserve_danger_gate",
        ),
        (
            "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_HORIZON",
            "shop_value_future_shop_reserve_danger_horizon",
        ),
        (
            "SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_MULTIPLIER",
            "shop_value_future_shop_reserve_danger_multiplier",
        ),
        (
            "SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_JSON",
            "shop_value_forced_danger_item_bias_json",
        ),
        (
            "SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_HORIZON",
            "shop_value_forced_danger_item_bias_horizon",
        ),
        ("SPIRECOMM_SHOP_VALUE_FAIRY_FORCED_ELITE_BIAS", "shop_value_fairy_forced_elite_bias"),
        ("SPIRECOMM_SHOP_VALUE_FAIRY_FORCED_ELITE_HORIZON", "shop_value_fairy_forced_elite_horizon"),
        ("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "shop_prior_weight_override"),
        ("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "v3_normal_room_potion_penalty"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", "v3_combat_rollout_rerank_topk"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", "v3_combat_rollout_rerank_max_steps"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", "v3_combat_rollout_rerank_margin_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", "v3_combat_rollout_rerank_min_advantage"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES", "v3_combat_rollout_rerank_room_types"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", "v3_combat_rollout_rerank_require_danger"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DANGER_HP_RATIO_MAX", "v3_combat_rollout_rerank_danger_hp_ratio_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS", "v3_combat_rollout_rerank_top_kinds"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES", "v3_combat_rollout_rerank_top_card_types"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_SCORE_MAX", "v3_combat_rollout_rerank_top_score_max"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS", "v3_combat_rollout_rerank_candidate_kinds"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_SAME_TURN_ONLY", "v3_combat_rollout_rerank_same_turn_only"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LIGHT_POLICY", "v3_combat_rollout_rerank_light_policy"),
        ("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", "v3_combat_rollout_rerank_floor_max"),
        (
            "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_NOT_CLEAR",
            "v3_combat_rollout_rerank_require_chosen_not_clear",
        ),
        (
            "SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CHOSEN_CLEAR_FULL_POLICY",
            "v3_combat_rollout_rerank_chosen_clear_full_policy",
        ),
        ("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_GUARD", "v3_combat_high_block_progress_guard"),
        ("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_ROOM_TYPES", "v3_combat_high_block_progress_room_types"),
        ("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MIN_BLOCK", "v3_combat_high_block_progress_min_block"),
        ("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_SURPLUS_MIN", "v3_combat_high_block_progress_surplus_min"),
        ("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MIN_DAMAGE", "v3_combat_high_block_progress_min_damage"),
        ("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MARGIN_MAX", "v3_combat_high_block_progress_margin_max"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_GUARD", "v3_combat_monster_block_progress_guard"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_ROOM_TYPES", "v3_combat_monster_block_progress_room_types"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MIN_BLOCK", "v3_combat_monster_block_progress_min_block"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_STALL_COUNT_MIN", "v3_combat_monster_block_progress_stall_count_min"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MIN_PROGRESS", "v3_combat_monster_block_progress_min_progress"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MARGIN_MAX", "v3_combat_monster_block_progress_margin_max"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_HP_WEIGHT", "v3_combat_monster_block_progress_hp_weight"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_BEST_SCORE_MAX", "v3_combat_monster_block_progress_best_score_max"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_TOP_CARD_TYPES", "v3_combat_monster_block_progress_top_card_types"),
        ("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_EXCLUDE_CARDS", "v3_combat_monster_block_progress_exclude_cards"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_GUARD", "v3_combat_danger_block_progress_guard"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_ROOM_TYPES", "v3_combat_danger_block_progress_room_types"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MIN_UNCOVERED", "v3_combat_danger_block_progress_min_uncovered"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_HP_RATIO_MIN", "v3_combat_danger_block_progress_hp_ratio_min"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_KINDS", "v3_combat_danger_block_progress_top_kinds"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_TYPES", "v3_combat_danger_block_progress_top_types"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_EXCLUDE_TOP_CARDS", "v3_combat_danger_block_progress_exclude_top_cards"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_REDUCTION_SKIP", "v3_combat_danger_block_progress_top_reduction_skip"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MIN_REDUCTION", "v3_combat_danger_block_progress_min_reduction"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MIN_EXTRA_REDUCTION", "v3_combat_danger_block_progress_min_extra_reduction"),
        ("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MARGIN_MAX", "v3_combat_danger_block_progress_margin_max"),
        (
            "SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_EXCLUDE_CARD_TYPES",
            "v3_combat_danger_block_progress_exclude_card_types",
        ),
        ("SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS", "v3_combat_survival_include_potions"),
        (
            "SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_INCLUDE_POTIONS",
            "v3_combat_lethal_card_over_setup_include_potions",
        ),
        ("SPIRECOMM_V3_COMBAT_SHORT_WIN_INCLUDE_POTIONS", "v3_combat_short_win_include_potions"),
        ("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_INCLUDE_POTIONS", "v3_combat_delayed_death_include_potions"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK", "v3_combat_teacher_fallback"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_BLEND_WEIGHT", "v3_combat_teacher_blend_weight"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_MARGIN_MAX", "v3_combat_teacher_fallback_margin_max"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_REQUIRE_POTION_CANDIDATE", "v3_combat_teacher_fallback_require_potion_candidate"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_REQUIRE_SUICIDAL_END_GUARD", "v3_combat_teacher_fallback_require_suicidal_end_guard"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_ROOM_TYPES", "v3_combat_teacher_fallback_room_types"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_FLOOR_MIN", "v3_combat_teacher_fallback_floor_min"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_FLOOR_MAX", "v3_combat_teacher_fallback_floor_max"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_TOP_KINDS", "v3_combat_teacher_fallback_top_kinds"),
        ("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_MODEL_TOP_KINDS", "v3_combat_teacher_fallback_model_top_kinds"),
        ("SPIRECOMM_V3_TEACHER_CONFIG_JSON", "teacher_config_json"),
        ("SPIRECOMM_V3_TEACHER_CONFIG_PATH", "teacher_config_path"),
        ("SPIRECOMM_BOSS_RELIC_MODEL_PATH", "boss_relic_model_path"),
        ("SPIRECOMM_EVENT_CHOICE_MODEL_PATH", "event_choice_model_path"),
        ("SPIRECOMM_CAMPFIRE_MODEL_PATH", "campfire_model_path"),
        ("SPIRECOMM_MAP_CHOICE_MODEL_PATH", "map_choice_model_path"),
        ("SPIRECOMM_V3_TEACHER_SAFE_SINGLE_ACTION_INPLACE", "v3_teacher_safe_single_action_inplace"),
        ("SPIRECOMM_V3_TEACHER_COMBAT_BRANCH_ONLY", "v3_teacher_combat_branch_only"),
        ("SPIRECOMM_V3_TEACHER_DEDUPE_EQUIVALENT_CARD_ACTIONS", "v3_teacher_dedupe_equivalent_card_actions"),
        ("SPIRECOMM_FAST_FLOOR_SKIP_COMBAT_SOURCE_FLAGS", "fast_floor_skip_combat_source_flags"),
        ("SPIRECOMM_CAMPFIRE_LOW_HP_REST_GUARD", "campfire_low_hp_rest_guard"),
        ("SPIRECOMM_CAMPFIRE_LOW_HP_REST_RATIO_MAX", "campfire_low_hp_rest_ratio_max"),
        ("SPIRECOMM_CAMPFIRE_LOW_HP_REST_HP_MAX", "campfire_low_hp_rest_hp_max"),
        ("SPIRECOMM_CAMPFIRE_LOW_HP_REST_FLOOR_MAX", "campfire_low_hp_rest_floor_max"),
        ("SPIRECOMM_CAMPFIRE_LOW_HP_REST_DANGER_HORIZON", "campfire_low_hp_rest_danger_horizon"),
        ("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RERANK", "campfire_final_boss_rest_rollout_rerank"),
        ("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_HP_MAX", "campfire_final_boss_rest_rollout_hp_max"),
        ("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RATIO_MAX", "campfire_final_boss_rest_rollout_ratio_max"),
        ("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MAX_STEPS", "campfire_final_boss_rest_rollout_max_steps"),
        ("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MIN_ADVANTAGE", "campfire_final_boss_rest_rollout_min_advantage"),
        ("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_LIGHT_POLICY", "campfire_final_boss_rest_rollout_light_policy"),
        ("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_GUARD", "v3_combat_sharp_hide_danger_guard"),
        ("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_MARGIN_MAX", "v3_combat_sharp_hide_danger_margin_max"),
        ("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_CAP", "event_dead_adventurer_low_hp_cap"),
        ("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_RATIO_MAX", "event_dead_adventurer_low_hp_ratio_max"),
        ("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_MAX", "event_dead_adventurer_low_hp_max"),
        ("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_MAX_SEARCHES", "event_dead_adventurer_low_hp_max_searches"),
        ("SPIRECOMM_EVENT_GOLDEN_IDOL_MAX_HP_OVER_DAMAGE", "event_golden_idol_max_hp_over_damage"),
        (
            "SPIRECOMM_EVENT_GOLDEN_IDOL_MAX_HP_OVER_DAMAGE_FLOOR_MAX",
            "event_golden_idol_max_hp_over_damage_floor_max",
        ),
        ("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_GUARD", "event_golden_idol_skip_forced_elite_guard"),
        ("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_HORIZON", "event_golden_idol_skip_forced_elite_horizon"),
        (
            "SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_PROJECTED_HP_MIN",
            "event_golden_idol_skip_forced_elite_projected_hp_min",
        ),
        (
            "SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_PROJECTED_HP_RATIO_MIN",
            "event_golden_idol_skip_forced_elite_projected_hp_ratio_min",
        ),
        ("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_CAP", "event_scrap_ooze_low_hp_cap"),
        ("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_MAX", "event_scrap_ooze_low_hp_max"),
        ("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_RATIO_MAX", "event_scrap_ooze_low_hp_ratio_max"),
        ("SPIRECOMM_BOSS_RELIC_SCORE_BIAS_JSON", "boss_relic_score_bias_json"),
        ("SPIRECOMM_BOSS_RELIC_SKIP_RAW_TRUST_MARGIN", "boss_relic_skip_raw_trust_margin"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_RERANK", "boss_relic_rollout_rerank"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_TOPK", "boss_relic_rollout_topk"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_STEPS", "boss_relic_rollout_max_steps"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_FLOOR_DELTA", "boss_relic_rollout_max_floor_delta"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_MIN_ADVANTAGE", "boss_relic_rollout_min_advantage"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_MODEL_SCORE_DROP", "boss_relic_rollout_max_model_score_drop"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_EVENT_IDS_OVERRIDE", "boss_relic_rollout_event_ids_override"),
        ("SPIRECOMM_BOSS_RELIC_ROLLOUT_SNECKO_ENERGY_MODEL_GAP", "boss_relic_rollout_snecko_energy_model_gap"),
        (
            "SPIRECOMM_BOSS_RELIC_ROLLOUT_SNECKO_ENERGY_MIN_SCORE_LEAD",
            "boss_relic_rollout_snecko_energy_min_score_lead",
        ),
        ("SPIRECOMM_EVENT_LOW_HP_COST_GUARD", "event_low_hp_cost_guard"),
        ("SPIRECOMM_EVENT_LOW_HP_COST_MIN_AFTER", "event_low_hp_cost_min_after"),
        ("SPIRECOMM_EVENT_CHOSEN_LOW_HP_COST_GUARD", "event_chosen_low_hp_cost_guard"),
        ("SPIRECOMM_EVENT_CHOSEN_LOW_HP_COST_MIN_AFTER", "event_chosen_low_hp_cost_min_after"),
        ("SPIRECOMM_EVENT_ROLLOUT_RERANK", "event_rollout_rerank"),
        ("SPIRECOMM_EVENT_ROLLOUT_TOPK", "event_rollout_topk"),
        ("SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS", "event_rollout_event_ids"),
        ("SPIRECOMM_EVENT_ROLLOUT_MYSTERIOUS_SPHERE_FLOOR_MIN", "event_rollout_mysterious_sphere_floor_min"),
        ("SPIRECOMM_EVENT_ROLLOUT_MYSTERIOUS_SPHERE_HP_RATIO_MAX", "event_rollout_mysterious_sphere_hp_ratio_max"),
        ("SPIRECOMM_EVENT_ROLLOUT_MINDBLOOM_ALLOWED_ACTIONS", "event_rollout_mindbloom_allowed_actions"),
        ("SPIRECOMM_EVENT_ROLLOUT_LIBRARY_SLEEP_FLOOR_MIN", "event_rollout_library_sleep_floor_min"),
        ("SPIRECOMM_EVENT_ROLLOUT_LIBRARY_READ_FLOOR_MIN", "event_rollout_library_read_floor_min"),
        ("SPIRECOMM_EVENT_ROLLOUT_FLOOR_MIN", "event_rollout_floor_min"),
        ("SPIRECOMM_EVENT_ROLLOUT_FLOOR_MAX", "event_rollout_floor_max"),
        ("SPIRECOMM_EVENT_ROLLOUT_MAX_STEPS", "event_rollout_max_steps"),
        ("SPIRECOMM_EVENT_ROLLOUT_MAX_FLOOR_DELTA", "event_rollout_max_floor_delta"),
        ("SPIRECOMM_EVENT_ROLLOUT_MIN_ADVANTAGE", "event_rollout_min_advantage"),
        ("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP", "event_rollout_max_model_score_drop"),
        (
            "SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_EVENTS",
            "event_rollout_max_model_score_drop_events",
        ),
        (
            "SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_BY_EVENT",
            "event_rollout_max_model_score_drop_by_event",
        ),
        (
            "SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_GIVE_CARD_OVERRIDE",
            "event_rollout_wemeetagain_block_give_card_override",
        ),
        (
            "SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_PAY_GOLD_TO_GIVE_CARD_OVERRIDE",
            "event_rollout_wemeetagain_block_pay_gold_to_give_card_override",
        ),
        ("SPIRECOMM_NEOW_ROLLOUT_RERANK", "neow_rollout_rerank"),
        ("SPIRECOMM_NEOW_ROLLOUT_CHOICE_INDEXES", "neow_rollout_choice_indexes"),
        ("SPIRECOMM_NEOW_ROLLOUT_MAX_STEPS", "neow_rollout_max_steps"),
        ("SPIRECOMM_NEOW_ROLLOUT_MAX_FLOOR_DELTA", "neow_rollout_max_floor_delta"),
        ("SPIRECOMM_NEOW_ROLLOUT_MIN_ADVANTAGE", "neow_rollout_min_advantage"),
    ):
        if config_key in _CONFIG and _CONFIG[config_key] is not None:
            os.environ[env_name] = str(_CONFIG[config_key])
    torch_threads = int(_CONFIG.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
            from spirecomm.ai.torch_compat import torch

            if torch is not None:
                torch.set_num_threads(torch_threads)
                torch.set_num_interop_threads(1)
        except Exception:
            pass
    _prewarm_native_content_caches()
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


def _load_start_snapshot(snapshot_dir: str, seed: int) -> dict[str, Any] | None:
    cache_key = (str(snapshot_dir), int(seed))
    cached = _START_SNAPSHOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    snapshot_path = Path(snapshot_dir) / f"seed_{int(seed)}.pkl"
    if not snapshot_path.exists():
        return None
    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    max_entries = int(_CONFIG.get("start_snapshot_cache_entries") or 512)
    if max_entries > 0:
        if len(_START_SNAPSHOT_CACHE) >= max_entries:
            _START_SNAPSHOT_CACHE.clear()
        _START_SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


def _preload_start_snapshot_cache(
    snapshot_dir: str,
    seeds: list[int] | None = None,
    *,
    max_entries: int | None = None,
) -> int:
    if not snapshot_dir:
        return 0
    root = Path(snapshot_dir)
    if not root.exists():
        return 0
    if seeds is None:
        paths = sorted(root.glob("seed_*.pkl"))
    else:
        paths = [root / f"seed_{int(seed)}.pkl" for seed in seeds]
    if max_entries is None:
        max_entries = int(_CONFIG.get("start_snapshot_cache_entries") or 512)
    loaded = 0
    for path in paths:
        if max_entries is not None and int(max_entries) > 0 and len(_START_SNAPSHOT_CACHE) >= int(max_entries):
            break
        if not path.exists():
            continue
        try:
            seed_text = path.stem.removeprefix("seed_")
            seed = int(seed_text)
        except (TypeError, ValueError):
            continue
        cache_key = (str(snapshot_dir), int(seed))
        if cache_key in _START_SNAPSHOT_CACHE:
            continue
        try:
            with path.open("rb") as handle:
                _START_SNAPSHOT_CACHE[cache_key] = pickle.load(handle)
            loaded += 1
        except Exception:
            continue
    return loaded


def _run_seed(seed: int) -> dict[str, Any]:
    assert _SELECTORS is not None
    started = time.time()
    output_dir = Path(_CONFIG["output_dir"])
    trace_mode = str(_CONFIG.get("trace_mode") or "compact")
    metrics_mode = str(_CONFIG.get("metrics_mode") or "full")
    collect_action_metrics = metrics_mode != "floor"
    trace_path: Path | None = None
    if trace_mode != "none":
        trace_dir = output_dir / ("compact_traces" if trace_mode == "compact" else "summary_traces")
        trace_dir.mkdir(parents=True, exist_ok=True)
        suffix = "compact_trace" if trace_mode == "compact" else "summary_trace"
        trace_path = trace_dir / f"seed_{seed}_{suffix}.json"

    prefix_steps = 0
    prefix_metrics: dict[str, Any] = {}
    start_snapshot_dir = str(_CONFIG.get("start_snapshot_dir") or "")
    if start_snapshot_dir:
        snapshot = _load_start_snapshot(start_snapshot_dir, seed)
        if snapshot is not None:
            raw_prefix_metrics = snapshot.get("prefix_metrics")
            if isinstance(raw_prefix_metrics, dict):
                prefix_metrics = raw_prefix_metrics
            terminal_result = snapshot.get("terminal_result")
            if isinstance(terminal_result, dict):
                result = dict(terminal_result)
                if collect_action_metrics and prefix_metrics:
                    for key in ACTION_METRIC_KEYS:
                        if key in prefix_metrics and key not in result:
                            result[key] = prefix_metrics[key]
                if not collect_action_metrics:
                    for key in ACTION_METRIC_KEYS:
                        result.pop(key, None)
                result["seconds"] = time.time() - started
                return result
            env_blob = snapshot.get("env_blob")
            if not isinstance(env_blob, bytes):
                raise RuntimeError(f"start snapshot {start_snapshot_dir}/seed_{seed}.pkl does not contain env_blob")
            env = pickle.loads(env_blob)
            prefix_steps = int(snapshot.get("steps") or 0)
        else:
            env = NativeRunEnv(seed=seed, ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    else:
        env = NativeRunEnv(seed=seed, ascension_level=int(_CONFIG["ascension"]), enable_neow=True)
    steps: list[dict[str, Any]] = []
    if collect_action_metrics:
        sources: Counter[str] = Counter(prefix_metrics.get("sources") or {})
        potion_actions_by_id: Counter[str] = Counter(prefix_metrics.get("potion_actions_by_id") or {})
        potion_actions_by_room_type: Counter[str] = Counter(prefix_metrics.get("potion_actions_by_room_type") or {})
        shop_actions_by_item_kind: Counter[str] = Counter(prefix_metrics.get("shop_actions_by_item_kind") or {})
        shop_spend_by_item_kind: Counter[str] = Counter(prefix_metrics.get("shop_spend_by_item_kind") or {})
    error: str | None = None
    error_traceback: str | None = None
    max_steps = int(_CONFIG["max_steps"])
    max_floor = int(_CONFIG["max_floor"])
    no_progress_limit = int(_CONFIG.get("no_progress_limit") or 0)
    no_progress_count = 0
    last_no_progress_action = ""
    combat_stall_limit = int(_CONFIG.get("combat_stall_limit") or 0)
    combat_stall_count = 0
    last_combat_progress_signature: tuple[Any, ...] | None = _combat_progress_signature(env)
    timeout_reason: str | None = None
    record_steps = trace_mode == "compact"
    record_action_scores = record_steps and str(os.environ.get("SPIRECOMM_COMPACT_TRACE_ACTION_SCORES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    record_debug_state = record_steps and str(os.environ.get("SPIRECOMM_COMPACT_TRACE_STATE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    step_count = 0
    fast_teacher_combat_direct = (
        bool(_CONFIG.get("fast_teacher_combat_direct"))
        and not record_steps
        and not collect_action_metrics
        and no_progress_limit <= 0
        and str(_CONFIG.get("combat_selector") or "") in {"v3-teacher", "teacher"}
    )
    fast_combat_selector = (_SELECTORS or {}).get("combat") if fast_teacher_combat_direct else None

    remaining_steps = max(0, max_steps - prefix_steps)
    progress_hook = _RUN_PROGRESS_HOOK
    progress_seconds = max(
        0.25,
        float(os.environ.get("SPIRECOMM_EVAL_RUN_PROGRESS_SECONDS", "1") or "1"),
    )
    last_progress_time = time.time()
    last_progress_step_count = 0
    for step_index in range(remaining_steps):
        if env.phase in TERMINAL_PHASES or int(env.floor) > max_floor:
            break
        if record_steps:
            pre_phase = str(env.phase)
            pre_floor = int(env.floor)
            pre_hp = int(env.player.current_hp)
            pre_max_hp = int(env.player.max_hp)
            pre_gold = int(env.gold)
            pre_state = env.state()
            pre_room_type = str(pre_state.get("room_type") or getattr(env, "current_room_type", "") or "")
            pre_legal_actions = None
            if record_action_scores:
                try:
                    pre_legal_actions = [_jsonable_action(dict(action)) for action in env.legal_actions()]
                except Exception:
                    pre_legal_actions = []
        elif collect_action_metrics:
            pre_room_type = str(getattr(env, "current_room_type", "") or getattr(env, "phase", "") or "")
        else:
            pre_room_type = ""
        pre_signature = _progress_signature(env) if no_progress_limit > 0 else None
        try:
            if fast_teacher_combat_direct and str(getattr(env, "phase", "")) == "COMBAT" and fast_combat_selector is not None:
                legal_actions_env = getattr(fast_combat_selector, "legal_actions_env", None)
                actions = legal_actions_env(env) if callable(legal_actions_env) else env.legal_actions()
                if len(actions) == 1:
                    action = actions[0]
                else:
                    chosen, _scores = fast_combat_selector.choose_env(
                        env,
                        return_scores=False,
                        legal_actions=actions,
                    )
                    if chosen is None:
                        selector_error = str(getattr(fast_combat_selector, "last_error", "") or "")
                        raise RuntimeError(f"combat_selector_returned_no_choice:{selector_error}")
                    action = chosen
                scores = []
                source = "combat"
            else:
                action, scores, source = choose_model_required_action(env, _SELECTORS, return_scores=record_steps)
            if collect_action_metrics and str(action.get("kind") or "") == "potion":
                potion_id = str(action.get("potion_id") or action.get("name") or "UNKNOWN")
                potion_actions_by_id[potion_id] += 1
                potion_actions_by_room_type[pre_room_type or "UNKNOWN"] += 1
            if collect_action_metrics and str(action.get("kind") or "") == "shop":
                item_kind = str(action.get("item_kind") or "UNKNOWN")
                shop_actions_by_item_kind[item_kind] += 1
                try:
                    price = max(0, int(action.get("price") or 0))
                except (TypeError, ValueError):
                    price = 0
                shop_spend_by_item_kind[item_kind] += price
            env.step(action)
            if no_progress_limit > 0 and _progress_signature(env) == pre_signature:
                action_signature = json.dumps(_jsonable_action(action), ensure_ascii=False, sort_keys=True)
                no_progress_count = no_progress_count + 1 if action_signature == last_no_progress_action else 1
                last_no_progress_action = action_signature
                if no_progress_count >= no_progress_limit:
                    error = f"RuntimeError: no_progress_loop after {no_progress_count} repeated no-op actions"
                    break
            else:
                no_progress_count = 0
                last_no_progress_action = ""
            if combat_stall_limit > 0:
                combat_signature = _combat_progress_signature(env)
                if combat_signature is None:
                    combat_stall_count = 0
                    last_combat_progress_signature = None
                elif combat_signature == last_combat_progress_signature:
                    combat_stall_count += 1
                    if combat_stall_count >= combat_stall_limit:
                        timeout_reason = (
                            "combat_stall_no_hp_progress "
                            f"after {combat_stall_count} repeated combat-progress signatures"
                        )
                        break
                else:
                    combat_stall_count = 0
                    last_combat_progress_signature = combat_signature
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            break
        if collect_action_metrics:
            sources[str(source)] += 1
        step_count += 1
        if progress_hook is not None:
            now = time.time()
            if now - last_progress_time >= progress_seconds:
                step_delta = max(1, step_count - last_progress_step_count)
                try:
                    progress_hook(
                        seed=int(seed),
                        step_delta=int(step_delta),
                        total_steps=int(prefix_steps + step_count),
                        floor=int(getattr(env, "floor", 0) or 0),
                        phase=str(getattr(env, "phase", "")),
                        elapsed=float(now - started),
                    )
                except Exception:
                    pass
                last_progress_time = now
                last_progress_step_count = step_count
        if record_steps:
            post_state_for_record: dict[str, Any] | None = None
            if record_debug_state:
                try:
                    post_state_for_record = env.state()
                except Exception:
                    post_state_for_record = {}
            post_hp = int(env.player.current_hp)
            if isinstance(post_state_for_record, dict):
                try:
                    post_hp = int(post_state_for_record.get("current_hp", post_hp))
                except (TypeError, ValueError):
                    pass
            step_record = {
                "step": step_index,
                "phase": pre_phase,
                "floor": pre_floor,
                "hp": pre_hp,
                "max_hp": pre_max_hp,
                "gold": pre_gold,
                "room_type": pre_room_type,
                "source": str(source),
                "action": _jsonable_action(action),
                "score_count": len(scores),
                "post_phase": str(env.phase),
                "post_floor": int(env.floor),
                "post_hp": post_hp,
            }
            if record_action_scores:
                step_record["legal_actions"] = pre_legal_actions or []
                step_record["scores"] = [float(score) for score in scores]
                if str(pre_phase) == "COMBAT":
                    combat_selector = _SELECTORS.get("combat")
                    rollout_debug = getattr(combat_selector, "last_rollout_rerank_debug", None)
                    if isinstance(rollout_debug, dict):
                        step_record["rollout_rerank_debug"] = rollout_debug
            if record_debug_state:
                step_record["pre_state"] = pre_state
                step_record["post_state"] = post_state_for_record or {}
            steps.append(step_record)

    total_steps = prefix_steps + step_count
    timed_out = env.phase not in TERMINAL_PHASES and not error and (total_steps >= max_steps or timeout_reason is not None)
    max_floor_stopped = int(env.floor) > max_floor and env.phase not in TERMINAL_PHASES
    result = {
        "seed": seed,
        "ascension": int(_CONFIG["ascension"]),
        "phase": str(env.phase),
        "floor": int(env.floor),
        "hp": int(env.player.current_hp),
        "max_hp": int(env.player.max_hp),
        "gold": int(env.gold),
        "deck_size": len(env.deck),
        "relic_count": len(env.relics),
        "potion_count": len(env.potions),
        "steps": total_steps,
        "won": str(env.phase) in {"COMPLETE", "VICTORY"},
        "dead": str(env.phase) == "GAME_OVER",
        "timed_out": timed_out,
        "timeout_reason": timeout_reason,
        "max_floor_stopped": max_floor_stopped,
        "error": error,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "seconds": time.time() - started,
    }
    if bool(_CONFIG.get("compact_floor_results")) and not collect_action_metrics and trace_path is None:
        result = {
            key: result[key]
            for key in (
                "seed",
                "ascension",
                "phase",
                "floor",
                "hp",
                "max_hp",
                "gold",
                "steps",
                "won",
                "dead",
                "timed_out",
                "timeout_reason",
                "max_floor_stopped",
                "error",
            )
        }
    if collect_action_metrics:
        result.update(
            {
                "sources": dict(sources),
                "potion_action_count": sum(potion_actions_by_id.values()),
                "potion_actions_by_id": dict(potion_actions_by_id),
                "potion_actions_by_room_type": dict(potion_actions_by_room_type),
                "shop_action_count": sum(shop_actions_by_item_kind.values()),
                "shop_actions_by_item_kind": dict(shop_actions_by_item_kind),
                "shop_spend_by_item_kind": dict(shop_spend_by_item_kind),
                "shop_spend_total": sum(shop_spend_by_item_kind.values()),
            }
        )
    if trace_path is not None:
        trace_payload = {
            "seed": seed,
            "ascension": int(_CONFIG["ascension"]),
            "backend": "v3",
            "trace_policy": "model_required",
            "trace_mode": trace_mode,
            "combat_selector": str(_CONFIG["combat_selector"]),
            "combat_model": str(_CONFIG["combat_model"]),
            "v3_combat_model": str(_CONFIG["v3_combat_model"]),
            "result": result,
            "error_traceback": error_traceback,
        }
        if record_steps:
            trace_payload["steps"] = steps
        trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False), encoding="utf-8")
    return result


def _run_seed_batch(seeds: list[int]) -> list[dict[str, Any]]:
    return [_run_seed(int(seed)) for seed in seeds]


def _load_existing_results(output_dir: Path) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for path in (output_dir / "results.json", output_dir / "results.jsonl"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        result = json.loads(line)
                        by_seed[int(result["seed"])] = result
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    for result in payload:
                        by_seed[int(result["seed"])] = result
        except Exception:
            continue
    return by_seed


def _paired_floor_delta(results: list[dict[str, Any]], baseline_by_seed: dict[int, dict[str, Any]]) -> dict[str, int | float]:
    common = [
        result
        for result in results
        if int(result.get("seed", -1)) in baseline_by_seed
        and result.get("error") is None
        and baseline_by_seed[int(result.get("seed", -1))].get("error") is None
    ]
    if not common:
        return {"count": 0, "delta": 0, "up": 0, "down": 0, "same": 0, "mean_delta": 0.0}
    delta = 0
    up = down = same = 0
    for result in common:
        seed = int(result["seed"])
        candidate_floor = int(result.get("floor") or 0)
        baseline_floor = int(baseline_by_seed[seed].get("floor") or 0)
        item_delta = candidate_floor - baseline_floor
        delta += item_delta
        if item_delta > 0:
            up += 1
        elif item_delta < 0:
            down += 1
        else:
            same += 1
    return {
        "count": len(common),
        "delta": delta,
        "up": up,
        "down": down,
        "same": same,
        "mean_delta": float(delta) / max(1, len(common)),
    }


def _append_result_jsonl(path: Path, result: dict[str, Any]) -> None:
    _append_results_jsonl(path, [result])


def _append_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


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
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resolve_combat_device(device: str | None, combat_device: str | None) -> str:
    token = str(combat_device or "auto").strip().lower()
    if token in {"", "auto"}:
        return "cuda" if _cuda_available() else str(device or "cpu")
    return str(combat_device)


def _bounded_task_batch_size(raw_batch_size: int, *, pending_count: int, workers: int, fast_like: bool) -> int:
    if pending_count <= 1:
        return 1
    batch_size = int(raw_batch_size)
    if batch_size <= 0:
        env_default = os.environ.get("SPIRECOMM_EVAL_AUTO_TASK_BATCH_SIZE")
        if env_default is not None and str(env_default).strip():
            try:
                batch_size = int(env_default)
            except ValueError:
                batch_size = 1
        else:
            # Seed runtime variance is high; smaller futures improve load balancing
            # more than the saved scheduling overhead for normal floor evals.
            batch_size = 1 if fast_like else 1
    if batch_size <= 1:
        return 1
    target_batches = max(1, min(int(pending_count), max(1, int(workers)) * 2))
    max_batch_size = max(1, (int(pending_count) + target_batches - 1) // target_batches)
    return max(1, min(batch_size, max_batch_size))


def _should_preload_selectors(args: argparse.Namespace, pending_seeds: list[int]) -> bool:
    mode = str(args.preload_selectors).strip().lower()
    if mode == "never":
        return False
    if not pending_seeds or int(args.workers) <= 1:
        return False
    if "fork" not in mp.get_all_start_methods():
        return False
    if mode == "always":
        return True
    return _device_allows_prefork(args.device) and _device_allows_prefork(args.combat_device or args.device)


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _summarize(
    results: list[dict[str, Any]],
    *,
    started: float,
    output_dir: Path,
    paired_baseline_by_seed: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    floors = [int(result["floor"]) for result in results]
    steps = [int(result["steps"]) for result in results]
    hp_alive = [int(result["hp"]) for result in results if not result.get("dead")]
    phases = Counter(str(result["phase"]) for result in results)
    ordered_floors = sorted(floors)

    def floor_quantile(q: float) -> float:
        if not ordered_floors:
            return 0.0
        index = min(len(ordered_floors) - 1, max(0, round((len(ordered_floors) - 1) * q)))
        return float(ordered_floors[index])

    has_action_metrics = any("sources" in result or "potion_action_count" in result for result in results)
    sources: Counter[str] = Counter()
    potion_actions_by_id: Counter[str] = Counter()
    potion_actions_by_room_type: Counter[str] = Counter()
    shop_actions_by_item_kind: Counter[str] = Counter()
    shop_spend_by_item_kind: Counter[str] = Counter()
    potion_action_count = 0
    shop_action_count = 0
    shop_spend_total = 0
    if has_action_metrics:
        for result in results:
            sources.update(result.get("sources") or {})
            potion_actions_by_id.update(result.get("potion_actions_by_id") or {})
            potion_actions_by_room_type.update(result.get("potion_actions_by_room_type") or {})
            potion_action_count += int(result.get("potion_action_count") or 0)
            shop_actions_by_item_kind.update(result.get("shop_actions_by_item_kind") or {})
            shop_spend_by_item_kind.update(result.get("shop_spend_by_item_kind") or {})
            shop_action_count += int(result.get("shop_action_count") or 0)
            shop_spend_total += int(result.get("shop_spend_total") or 0)
    errors = [result for result in results if result.get("error")]
    wins = sum(1 for result in results if result.get("won"))
    deaths = sum(1 for result in results if result.get("dead"))
    timeouts = sum(1 for result in results if result.get("timed_out"))
    timeout_reasons = Counter(
        str(result.get("timeout_reason") or "max_steps") for result in results if result.get("timed_out")
    )
    elapsed = max(1e-9, time.time() - started)
    total_steps = sum(steps)
    summary = {
        "count": len(results),
        "seconds": elapsed,
        "seconds_per_seed": elapsed / max(1, len(results)),
        "seeds_per_second": len(results) / elapsed,
        "steps_per_second": total_steps / elapsed,
        "output_dir": str(output_dir),
        "mean_floor": mean(floors) if floors else 0.0,
        "median_floor": median(floors) if floors else 0.0,
        "p10_floor": floor_quantile(0.10),
        "p25_floor": floor_quantile(0.25),
        "p75_floor": floor_quantile(0.75),
        "p90_floor": floor_quantile(0.90),
        "max_floor": max(floors) if floors else 0,
        "min_floor": min(floors) if floors else 0,
        "win_count": wins,
        "win_rate": wins / max(1, len(results)),
        "death_count": deaths,
        "death_rate": deaths / max(1, len(results)),
        "timeout_count": timeouts,
        "timeout_reasons": dict(timeout_reasons.most_common()),
        "error_count": len(errors),
        "mean_steps": mean(steps) if steps else 0.0,
        "mean_hp_if_not_dead": mean(hp_alive) if hp_alive else 0.0,
        "phase_counts": dict(phases),
        "floor_counts": dict(sorted(Counter(floors).items())),
        "source_counts": dict(sources.most_common()),
        "potion_action_count": potion_action_count,
        "potion_actions_by_id": dict(potion_actions_by_id.most_common()),
        "potion_actions_by_room_type": dict(potion_actions_by_room_type.most_common()),
        "potion_actions_per_run": potion_action_count / max(1, len(results)),
        "shop_action_count": shop_action_count,
        "shop_actions_by_item_kind": dict(shop_actions_by_item_kind.most_common()),
        "shop_spend_by_item_kind": dict(shop_spend_by_item_kind.most_common()),
        "shop_spend_total": shop_spend_total,
        "shop_spend_per_run": shop_spend_total / max(1, len(results)),
        "errors": [
            {"seed": result["seed"], "floor": result["floor"], "phase": result["phase"], "error": result["error"]}
            for result in errors[:20]
        ],
    }
    if paired_baseline_by_seed:
        summary["paired_floor_delta"] = _paired_floor_delta(results, paired_baseline_by_seed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run many v3 model-required rollouts with the v3 combat candidate selector.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--no-progress-limit", type=int, default=50)
    parser.add_argument(
        "--combat-stall-limit",
        type=int,
        default=int(os.environ.get("SPIRECOMM_COMBAT_STALL_LIMIT", "250")),
        help=(
            "Stop a combat as timeout after this many consecutive combat decisions "
            "with unchanged player/monster HP. Use 0 to disable."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--trace-mode",
        choices=["compact", "summary", "none"],
        default="compact",
        help="Per-seed trace output. Use 'none' for fast aggregate metric sweeps.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip seeds already present in results.json/results.jsonl.")
    parser.add_argument(
        "--rerun-timeouts",
        action="store_true",
        help="With --resume, treat previously timed-out seeds as pending so they can be retried with a larger --max-steps.",
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=0,
        help="Write summary_partial.json every N completed seeds. Use 0 to only write the first resume summary and final summary.",
    )
    parser.add_argument(
        "--write-first-partial-summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write summary_partial.json after the first newly completed seed.",
    )
    parser.add_argument(
        "--result-flush-interval",
        type=int,
        default=64,
        help="Buffer this many seed results before appending results.jsonl.",
    )
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=0,
        help="Run this many seeds per process-pool future. 0 auto-batches fast/floor-only evals.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast metric sweep shortcut: --trace-mode none --no-progress-limit 0.",
    )
    parser.add_argument(
        "--metrics-mode",
        choices=["full", "floor"],
        default="full",
        help="Use 'floor' when only aggregate floor metrics are needed.",
    )
    parser.add_argument(
        "--mean-floor-only",
        action="store_true",
        help="Shortcut for --fast --metrics-mode floor.",
    )
    parser.add_argument(
        "--fast-disable-runtime-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Disable expensive runtime rollout rerankers during fast/floor-only model comparisons. "
            "Default: on for --fast/--mean-floor-only, off otherwise. Use "
            "--no-fast-disable-runtime-search for full-policy performance evals."
        ),
    )
    parser.add_argument(
        "--runtime-latency-profile",
        default=os.environ.get("SPIRECOMM_RUNTIME_LATENCY_PROFILE", ""),
        help=(
            "Optional runtime latency profile. 'interactive' disables long event/campfire rollouts; "
            "'instant' also disables combat/map/boss-relic rollout rerankers. Explicit env values still win."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Per-worker torch intra-op threads. Keep at 1 for multi-process seed sweeps.",
    )
    parser.add_argument(
        "--preload-selectors",
        choices=["auto", "always", "never"],
        default="auto",
        help="Preload selectors before forking CPU workers so model pages can be shared copy-on-write.",
    )
    parser.add_argument(
        "--write-results-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also write a pretty results.json. Default is on for full traced evals and off for fast/floor-only sweeps.",
    )
    parser.add_argument("--start-snapshot-dir", default="")
    parser.add_argument("--start-snapshot-cache-entries", type=int, default=512)
    parser.add_argument(
        "--paired-baseline-dir",
        type=Path,
        default=None,
        help="Optional result directory/file used for live paired floor-delta reporting.",
    )
    parser.add_argument(
        "--early-stop-paired-delta-below",
        type=int,
        default=None,
        help="If paired baseline is provided, stop scheduling more seeds once paired total delta is <= this value.",
    )
    parser.add_argument(
        "--early-stop-min-paired",
        type=int,
        default=0,
        help="Minimum paired seed count before --early-stop-paired-delta-below can stop the run.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--combat-device",
        default=os.environ.get("SPIRECOMM_EVAL_COMBAT_DEVICE", "auto"),
        help="Device for combat model scoring. Default 'auto' uses CUDA when available, otherwise --device.",
    )
    parser.add_argument("--combat-selector", choices=["legacy-slot", "v3-candidate", "v3-teacher"], default="v3-candidate")
    parser.add_argument("--combat-model", type=Path, default=Path("models/combat.pt"))
    parser.add_argument("--v3-combat-model", type=Path, default=Path("models/v3_combat_scorer.pt"))
    parser.add_argument(
        "--teacher-config-json",
        default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_JSON", ""),
        help="Inline JSON object overriding v3 teacher coefficients for --combat-selector v3-teacher.",
    )
    parser.add_argument(
        "--teacher-config-path",
        default=os.environ.get("SPIRECOMM_V3_TEACHER_CONFIG_PATH", ""),
        help="Path to JSON object overriding v3 teacher coefficients for --combat-selector v3-teacher.",
    )
    parser.add_argument("--card-reward-model", type=Path, default=Path("models/card_reward.pt"))
    parser.add_argument(
        "--shop-choice-model",
        type=Path,
        default=Path(os.environ.get("SPIRECOMM_SHOP_CHOICE_MODEL_PATH", "models/shop_choice_prior_delta.pt")),
    )
    parser.add_argument(
        "--shop-policy",
        choices=["model", "value"],
        default=os.environ.get("SPIRECOMM_SHOP_POLICY", "value"),
        help="Use the legacy learned shop selector or the explicit value-minus-gold-cost policy.",
    )
    parser.add_argument(
        "--shop-value-price-cost",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_PRICE_COST", "0.044348003822393976")),
    )
    parser.add_argument(
        "--shop-value-reserve-shortfall-cost",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RESERVE_SHORTFALL_COST", "0.043490245962190935")),
    )
    parser.add_argument("--shop-value-future-shop-reserve", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE", "120")))
    parser.add_argument("--shop-value-future-shop-horizon", type=int, default=int(os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_HORIZON", "5")))
    parser.add_argument(
        "--shop-value-card-scale",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_SCALE", "4.6262945279949435")),
    )
    parser.add_argument("--shop-value-card-reference-price", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_REFERENCE_PRICE", "60.0")))
    parser.add_argument("--shop-value-card-price-factor-min", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MIN", "0.65")))
    parser.add_argument("--shop-value-card-price-factor-max", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_CARD_PRICE_FACTOR_MAX", "1.35")))
    parser.add_argument(
        "--shop-value-potion-scale",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_POTION_SCALE", "0.5084989138155764")),
    )
    parser.add_argument(
        "--shop-value-relic-scale",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_RELIC_SCALE", "0.8")),
    )
    parser.add_argument("--shop-value-item-scale", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_SCALE", "1.0")))
    parser.add_argument("--shop-value-threshold", type=float, default=float(os.environ.get("SPIRECOMM_SHOP_VALUE_THRESHOLD", "0.0")))
    parser.add_argument(
        "--shop-prior-weight-override",
        type=float,
        default=float(os.environ.get("SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE", "0.8")),
    )
    parser.add_argument(
        "--v3-normal-room-potion-penalty",
        type=float,
        default=float(os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY", "5.0")),
        help="Subtract this score from combat potion actions in ordinary MonsterRoom only. Default 5.0 reserves potions for high-leverage fights.",
    )
    args = parser.parse_args()
    if args.runtime_latency_profile:
        os.environ["SPIRECOMM_RUNTIME_LATENCY_PROFILE"] = str(args.runtime_latency_profile)
        apply_runtime_latency_profile(args.runtime_latency_profile)

    if args.mean_floor_only:
        args.fast = True
        args.metrics_mode = "floor"
    if args.fast:
        args.trace_mode = "none"
        args.no_progress_limit = 0
    if args.fast_disable_runtime_search is None:
        args.fast_disable_runtime_search = bool(args.fast or args.mean_floor_only)
    if args.fast_disable_runtime_search:
        # Fast candidate-vs-baseline comparisons should compare the scorer itself,
        # not the expensive runtime rollout search wrapped around it.
        os.environ["SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK"] = "0"
        os.environ["SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK"] = "0"
    if args.metrics_mode == "floor" and args.trace_mode == "none":
        os.environ.setdefault("SPIRECOMM_FAST_FLOOR_SKIP_COMBAT_SOURCE_FLAGS", "1")
    args.combat_device = _resolve_combat_device(args.device, args.combat_device)
    write_results_json = bool(args.write_results_json)
    if args.write_results_json is None:
        write_results_json = not (args.fast or args.metrics_mode == "floor" or args.trace_mode == "none")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    else:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.count)))

    config = {
        "repo_root": str(args.repo_root.resolve()),
        "output_dir": str(output_dir),
        "ascension": int(args.ascension),
        "max_floor": int(args.max_floor),
        "max_steps": int(args.max_steps),
        "no_progress_limit": int(args.no_progress_limit),
        "combat_stall_limit": int(args.combat_stall_limit),
        "trace_mode": str(args.trace_mode),
        "metrics_mode": str(args.metrics_mode),
        "fast_disable_runtime_search": bool(args.fast_disable_runtime_search),
        "runtime_latency_profile": str(args.runtime_latency_profile or ""),
        "torch_threads": int(args.torch_threads),
        "device": str(args.device),
        "combat_device": args.combat_device,
        "combat_selector": str(args.combat_selector),
        "combat_model": str(args.combat_model),
        "v3_combat_model": str(args.v3_combat_model),
        "teacher_config_json": str(args.teacher_config_json or ""),
        "teacher_config_path": str(args.teacher_config_path or ""),
        "boss_relic_model_path": str(os.environ.get("SPIRECOMM_BOSS_RELIC_MODEL_PATH", "")),
        "event_choice_model_path": str(os.environ.get("SPIRECOMM_EVENT_CHOICE_MODEL_PATH", "")),
        "campfire_model_path": str(os.environ.get("SPIRECOMM_CAMPFIRE_MODEL_PATH", "")),
        "map_choice_model_path": str(os.environ.get("SPIRECOMM_MAP_CHOICE_MODEL_PATH", "")),
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
        "shop_value_item_bias_json": str(os.environ.get("SPIRECOMM_SHOP_VALUE_ITEM_BIAS_JSON", "")),
        "shop_value_future_shop_reserve_danger_gate": str(
            os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_GATE", "")
        ),
        "shop_value_future_shop_reserve_danger_horizon": str(
            os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_HORIZON", "")
        ),
        "shop_value_future_shop_reserve_danger_multiplier": str(
            os.environ.get("SPIRECOMM_SHOP_VALUE_FUTURE_SHOP_RESERVE_DANGER_MULTIPLIER", "")
        ),
        "shop_value_forced_danger_item_bias_json": str(
            os.environ.get("SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_JSON", "")
        ),
        "shop_value_forced_danger_item_bias_horizon": str(
            os.environ.get("SPIRECOMM_SHOP_VALUE_FORCED_DANGER_ITEM_BIAS_HORIZON", "")
        ),
        "shop_value_fairy_forced_elite_bias": str(os.environ.get("SPIRECOMM_SHOP_VALUE_FAIRY_FORCED_ELITE_BIAS", "")),
        "shop_value_fairy_forced_elite_horizon": str(os.environ.get("SPIRECOMM_SHOP_VALUE_FAIRY_FORCED_ELITE_HORIZON", "")),
        "shop_prior_weight_override": None if args.shop_prior_weight_override is None else float(args.shop_prior_weight_override),
        "map_dp_monster_value": str(os.environ.get("SPIRECOMM_MAP_DP_MONSTER_VALUE", "")),
        "map_dp_rest_value": str(os.environ.get("SPIRECOMM_MAP_DP_REST_VALUE", "")),
        "map_dp_elite_base": str(os.environ.get("SPIRECOMM_MAP_DP_ELITE_BASE", "")),
        "map_dp_green_elite_penalty": str(os.environ.get("SPIRECOMM_MAP_DP_GREEN_ELITE_PENALTY", "")),
        "map_dp_winged_offpath_penalty": str(os.environ.get("SPIRECOMM_MAP_DP_WINGED_OFFPATH_PENALTY", "")),
        "map_dp_shop_gold_unit_value": str(os.environ.get("SPIRECOMM_MAP_DP_SHOP_GOLD_UNIT_VALUE", "")),
        "map_dp_shop_purgeable_curse_bonus": str(
            os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_BONUS", "")
        ),
        "map_dp_shop_purgeable_curse_urgency_bonus": str(
            os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_URGENCY_BONUS", "")
        ),
        "map_dp_shop_purgeable_curse_gold_threshold": str(
            os.environ.get("SPIRECOMM_MAP_DP_SHOP_PURGEABLE_CURSE_GOLD_THRESHOLD", "")
        ),
        "map_dp_hp_aware_risk_guard": str(os.environ.get("SPIRECOMM_MAP_DP_HP_AWARE_RISK_GUARD", "")),
        "map_dp_hp_aware_floor_max": str(os.environ.get("SPIRECOMM_MAP_DP_HP_AWARE_FLOOR_MAX", "")),
        "map_dp_hp_aware_elite_safe_ratio": str(os.environ.get("SPIRECOMM_MAP_DP_HP_AWARE_ELITE_SAFE_RATIO", "")),
        "map_dp_hp_aware_elite_penalty": str(os.environ.get("SPIRECOMM_MAP_DP_HP_AWARE_ELITE_PENALTY", "")),
        "map_dp_hp_aware_rest_trigger_ratio": str(os.environ.get("SPIRECOMM_MAP_DP_HP_AWARE_REST_TRIGGER_RATIO", "")),
        "map_dp_hp_aware_rest_bonus": str(os.environ.get("SPIRECOMM_MAP_DP_HP_AWARE_REST_BONUS", "")),
        "v3_normal_room_potion_penalty": max(0.0, float(args.v3_normal_room_potion_penalty)),
        "v3_combat_card_bias": str(os.environ.get("SPIRECOMM_V3_COMBAT_CARD_BIAS", "")),
        "v3_combat_potion_bias": str(os.environ.get("SPIRECOMM_V3_COMBAT_POTION_BIAS", "")),
        "v3_combat_end_bias": str(os.environ.get("SPIRECOMM_V3_COMBAT_END_BIAS", "")),
        "v3_combat_end_bias_room_types": str(os.environ.get("SPIRECOMM_V3_COMBAT_END_BIAS_ROOM_TYPES", "")),
        "early_card_reward_attack_bias_max": str(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_MAX", "")),
        "map_rollout_rerank_topk": str(os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_TOPK", "2")),
        "map_rollout_rerank_margin_max": str(os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_MARGIN_MAX", "120")),
        "map_rollout_rerank_min_advantage": str(os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_MIN_ADVANTAGE", "500")),
        "map_rollout_rerank_max_steps": str(os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_MAX_STEPS", "60")),
        "map_rollout_rerank_max_floor_delta": str(os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_MAX_FLOOR_DELTA", "2")),
        "map_rollout_rerank_floor_max": str(os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_FLOOR_MAX", "8")),
        "map_rollout_rerank_avoid_elite_gate": str(
            os.environ.get("SPIRECOMM_MAP_ROLLOUT_RERANK_AVOID_ELITE_GATE", "1")
        ),
        "v3_combat_dangerous_end_bias": str(os.environ.get("SPIRECOMM_V3_COMBAT_DANGEROUS_END_BIAS", "")),
        "v3_combat_dangerous_end_hp_ratio_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_DANGEROUS_END_HP_RATIO_MAX", "")),
        "v3_combat_dangerous_end_hp_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_DANGEROUS_END_HP_MAX", "")),
        "v3_combat_potion_over_end_margin_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_POTION_OVER_END_MARGIN_MAX", "")),
        "v3_combat_potion_over_end_room_types": str(os.environ.get("SPIRECOMM_V3_COMBAT_POTION_OVER_END_ROOM_TYPES", "")),
        "v3_combat_block_over_end_margin_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_MARGIN_MAX", "")),
        "v3_combat_block_over_end_room_types": str(os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_ROOM_TYPES", "")),
        "v3_combat_block_over_end_effective_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_MARGIN_MAX", "0.03")
        ),
        "v3_combat_block_over_end_effective_room_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_ROOM_TYPES", "MonsterRoomElite,MonsterRoomBoss")
        ),
        "v3_combat_block_over_end_end_hp_ratio_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_END_HP_RATIO_MAX", "")
        ),
        "v3_combat_block_over_end_end_hp_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_END_HP_MAX", "")),
        "v3_combat_lethal_card_over_setup_include_potions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_INCLUDE_POTIONS", "")
        ),
        "v3_combat_rollout_rerank_topk": str(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOPK", "3")),
        "v3_combat_rollout_rerank_max_steps": str(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MAX_STEPS", "80")),
        "v3_combat_rollout_rerank_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MARGIN_MAX", "8.0")
        ),
        "v3_combat_rollout_rerank_low_margin_threshold": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LOW_MARGIN_THRESHOLD", "0.35")
        ),
        "v3_combat_rollout_rerank_low_margin_topk": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LOW_MARGIN_TOPK", "5")
        ),
        "v3_combat_rollout_rerank_high_score_margin_threshold": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_HIGH_SCORE_MARGIN_THRESHOLD", "1.0")
        ),
        "v3_combat_rollout_rerank_high_score_min": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_HIGH_SCORE_MIN", "-1.0")
        ),
        "v3_combat_rollout_rerank_min_advantage": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_MIN_ADVANTAGE", "1000.0")
        ),
        "v3_combat_rollout_rerank_room_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_ROOM_TYPES", "MonsterRoomBoss")
        ),
        "v3_combat_rollout_rerank_require_danger": str(os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_DANGER", "")),
        "v3_combat_rollout_rerank_danger_hp_ratio_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_DANGER_HP_RATIO_MAX", "")
        ),
        "v3_combat_rollout_rerank_top_kinds": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_KINDS", "card")
        ),
        "v3_combat_rollout_rerank_top_card_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_CARD_TYPES", "ATTACK")
        ),
        "v3_combat_rollout_rerank_top_score_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_TOP_SCORE_MAX", "0.0")
        ),
        "v3_combat_rollout_rerank_candidate_kinds": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CANDIDATE_KINDS", "card,potion")
        ),
        "v3_combat_rollout_rerank_same_turn_only": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_SAME_TURN_ONLY", "0")
        ),
        "v3_combat_rollout_rerank_light_policy": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_LIGHT_POLICY", "1")
        ),
        "v3_combat_rollout_rerank_floor_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_FLOOR_MAX", "16")
        ),
        "v3_combat_rollout_rerank_require_chosen_not_clear": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_NOT_CLEAR", "1")
        ),
        "v3_combat_rollout_rerank_chosen_clear_full_policy": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_CHOSEN_CLEAR_FULL_POLICY", "1")
        ),
        "v3_combat_rollout_rerank_require_chosen_outcomes": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_CHOSEN_OUTCOMES", "")
        ),
        "v3_combat_rollout_rerank_require_best_outcomes": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_ROLLOUT_RERANK_REQUIRE_BEST_OUTCOMES", "")
        ),
        "v3_combat_short_win_guard": str(os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_GUARD", "")),
        "v3_combat_short_win_require_top_death": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_REQUIRE_TOP_DEATH", "")
        ),
        "v3_combat_short_win_max_decisions": str(os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_MAX_DECISIONS", "")),
        "v3_combat_short_win_potion_max_decisions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_POTION_MAX_DECISIONS", "")
        ),
        "v3_combat_short_win_topk": str(os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_TOPK", "")),
        "v3_combat_short_win_margin_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_MARGIN_MAX", "")),
        "v3_combat_short_win_optional_room_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_OPTIONAL_ROOM_TYPES", "")
        ),
        "v3_combat_short_win_include_potions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SHORT_WIN_INCLUDE_POTIONS", "")
        ),
        "v3_combat_policy_survival_guard": str(os.environ.get("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_GUARD", "")),
        "v3_combat_policy_survival_max_decisions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_MAX_DECISIONS", "")
        ),
        "v3_combat_policy_survival_topk": str(os.environ.get("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_TOPK", "")),
        "v3_combat_forced_turn_survival_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_FORCED_TURN_SURVIVAL_GUARD", "")
        ),
        "v3_combat_survival_guard_restrict_safe": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_RESTRICT_SAFE", "")
        ),
        "v3_combat_survival_guard_allow_safe_end": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_ALLOW_SAFE_END", "")
        ),
        "v3_combat_survival_include_potions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS", "")
        ),
        "v3_combat_suppress_suicidal_action": str(os.environ.get("SPIRECOMM_V3_SUPPRESS_SUICIDAL_ACTION", "")),
        "v3_combat_post_forced_turn_survival_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SURVIVAL_GUARD", "")
        ),
        "v3_combat_post_forced_turn_survival_topk": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SURVIVAL_TOPK", "")
        ),
        "v3_combat_post_forced_turn_allow_safe_end": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_ALLOW_SAFE_END", "")
        ),
        "v3_combat_post_forced_turn_prefer_win": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_PREFER_WIN", "")
        ),
        "v3_combat_post_forced_turn_include_potions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_INCLUDE_POTIONS", "")
        ),
        "v3_combat_post_forced_turn_skip_uncovered_gate": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SKIP_UNCOVERED_GATE", "")
        ),
        "v3_combat_post_action_survival_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_POST_ACTION_SURVIVAL_GUARD", "")
        ),
        "v3_combat_delayed_death_guard": str(os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_GUARD", "")),
        "v3_combat_delayed_death_max_decisions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_MAX_DECISIONS", "")
        ),
        "v3_combat_delayed_death_topk": str(os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_TOPK", "")),
        "v3_combat_delayed_death_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_MARGIN_MAX", "")
        ),
        "v3_combat_delayed_death_room_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_ROOM_TYPES", "")
        ),
        "v3_combat_delayed_death_top_kinds": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_TOP_KINDS", "")
        ),
        "v3_combat_delayed_death_allow_unknown": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_ALLOW_UNKNOWN", "")
        ),
        "v3_combat_delayed_death_include_potions": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_INCLUDE_POTIONS", "")
        ),
        "v3_combat_delayed_death_exclude_card_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_EXCLUDE_CARD_TYPES", "")
        ),
        "v3_combat_delayed_death_require_danger": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_REQUIRE_DANGER", "")
        ),
        "v3_combat_delayed_death_hp_ratio_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_HP_RATIO_MAX", "")
        ),
        "v3_combat_lethal_card_over_setup_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_MARGIN_MAX", "")
        ),
        "v3_combat_lethal_card_over_setup_top_kinds": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_TOP_KINDS", "")
        ),
        "v3_combat_lethal_card_over_setup_top_card_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_TOP_CARD_TYPES", "")
        ),
        "v3_combat_lethal_card_over_setup_skip_block_top_hp_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_SKIP_BLOCK_TOP_HP_MAX", "")
        ),
        "v3_combat_high_block_progress_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_GUARD", "")
        ),
        "v3_combat_high_block_progress_room_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_ROOM_TYPES", "")
        ),
        "v3_combat_high_block_progress_min_block": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MIN_BLOCK", "")
        ),
        "v3_combat_high_block_progress_surplus_min": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_SURPLUS_MIN", "")
        ),
        "v3_combat_high_block_progress_min_damage": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MIN_DAMAGE", "")
        ),
        "v3_combat_high_block_progress_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MARGIN_MAX", "")
        ),
        "v3_combat_monster_block_progress_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_GUARD", "")
        ),
        "v3_combat_monster_block_progress_room_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_ROOM_TYPES", "")
        ),
        "v3_combat_monster_block_progress_min_block": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MIN_BLOCK", "")
        ),
        "v3_combat_monster_block_progress_stall_count_min": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_STALL_COUNT_MIN", "")
        ),
        "v3_combat_monster_block_progress_min_progress": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MIN_PROGRESS", "")
        ),
        "v3_combat_monster_block_progress_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MARGIN_MAX", "")
        ),
        "v3_combat_monster_block_progress_hp_weight": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_HP_WEIGHT", "")
        ),
        "v3_combat_monster_block_progress_best_score_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_BEST_SCORE_MAX", "")
        ),
        "v3_combat_monster_block_progress_top_card_types": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_TOP_CARD_TYPES", "")
        ),
        "v3_combat_monster_block_progress_exclude_cards": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_EXCLUDE_CARDS", "")
        ),
        "v3_combat_gremlin_nob_skill_bias": str(os.environ.get("SPIRECOMM_V3_COMBAT_GREMLIN_NOB_SKILL_BIAS", "")),
        "v3_combat_gremlin_nob_skill_bias_exclude_cards": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_GREMLIN_NOB_SKILL_BIAS_EXCLUDE_CARDS", "")
        ),
        "v3_combat_gremlin_nob_skill_bias_disable_in_map_rollout": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_GREMLIN_NOB_SKILL_BIAS_DISABLE_IN_MAP_ROLLOUT", "")
        ),
        "campfire_low_hp_rest_guard": str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_GUARD", "")),
        "campfire_low_hp_rest_ratio_max": str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_RATIO_MAX", "")),
        "campfire_low_hp_rest_hp_max": str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_HP_MAX", "")),
        "campfire_low_hp_rest_floor_max": str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_FLOOR_MAX", "")),
        "campfire_low_hp_rest_danger_horizon": str(os.environ.get("SPIRECOMM_CAMPFIRE_LOW_HP_REST_DANGER_HORIZON", "")),
        "campfire_final_boss_rest_rollout_rerank": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RERANK", "0")
        ),
        "campfire_final_boss_rest_fast_guard": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD", "1")
        ),
        "campfire_final_boss_rest_fast_guard_max_score_gap": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_FAST_GUARD_MAX_SCORE_GAP", "2.5")
        ),
        "campfire_final_boss_rest_rollout_hp_max": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_HP_MAX", "")
        ),
        "campfire_final_boss_rest_rollout_ratio_max": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_RATIO_MAX", "")
        ),
        "campfire_final_boss_rest_rollout_max_steps": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MAX_STEPS", "")
        ),
        "campfire_final_boss_rest_rollout_min_advantage": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_MIN_ADVANTAGE", "")
        ),
        "campfire_final_boss_rest_rollout_light_policy": str(
            os.environ.get("SPIRECOMM_CAMPFIRE_FINAL_BOSS_REST_ROLLOUT_LIGHT_POLICY", "0")
        ),
        "v3_combat_sharp_hide_danger_guard": str(os.environ.get("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_GUARD", "")),
        "v3_combat_sharp_hide_danger_margin_max": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_MARGIN_MAX", "")
        ),
        "event_dead_adventurer_low_hp_cap": str(os.environ.get("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_CAP", "1")),
        "event_dead_adventurer_low_hp_ratio_max": str(
            os.environ.get("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_RATIO_MAX", "")
        ),
        "event_dead_adventurer_low_hp_max": str(os.environ.get("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_MAX", "")),
        "event_dead_adventurer_low_hp_max_searches": str(
            os.environ.get("SPIRECOMM_EVENT_DEAD_ADVENTURER_LOW_HP_MAX_SEARCHES", "")
        ),
        "event_golden_idol_max_hp_over_damage": str(
            os.environ.get("SPIRECOMM_EVENT_GOLDEN_IDOL_MAX_HP_OVER_DAMAGE", "")
        ),
        "event_golden_idol_max_hp_over_damage_floor_max": str(
            os.environ.get("SPIRECOMM_EVENT_GOLDEN_IDOL_MAX_HP_OVER_DAMAGE_FLOOR_MAX", "")
        ),
        "event_golden_idol_skip_forced_elite_guard": str(
            os.environ.get("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_GUARD", "")
        ),
        "event_golden_idol_skip_forced_elite_horizon": str(
            os.environ.get("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_HORIZON", "")
        ),
        "event_golden_idol_skip_forced_elite_projected_hp_min": str(
            os.environ.get("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_PROJECTED_HP_MIN", "")
        ),
        "event_golden_idol_skip_forced_elite_projected_hp_ratio_min": str(
            os.environ.get("SPIRECOMM_EVENT_GOLDEN_IDOL_SKIP_FORCED_ELITE_PROJECTED_HP_RATIO_MIN", "")
        ),
        "event_scrap_ooze_low_hp_cap": str(os.environ.get("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_CAP", "")),
        "event_scrap_ooze_low_hp_max": str(os.environ.get("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_MAX", "")),
        "event_scrap_ooze_low_hp_ratio_max": str(os.environ.get("SPIRECOMM_EVENT_SCRAP_OOZE_LOW_HP_RATIO_MAX", "")),
        "boss_relic_score_bias_json": str(
            os.environ.get(
                "SPIRECOMM_BOSS_RELIC_SCORE_BIAS_JSON",
                json.dumps(DEFAULT_BOSS_RELIC_SCORE_BIAS, sort_keys=True),
            )
        ),
        "boss_relic_skip_raw_trust_margin": str(os.environ.get("SPIRECOMM_BOSS_RELIC_SKIP_RAW_TRUST_MARGIN", "")),
        "boss_relic_rollout_rerank": str(os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_RERANK", "1")),
        "boss_relic_rollout_topk": str(os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_TOPK", "")),
        "boss_relic_rollout_max_steps": str(os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_STEPS", "")),
        "boss_relic_rollout_max_floor_delta": str(
            os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_FLOOR_DELTA", "")
        ),
        "boss_relic_rollout_min_advantage": str(os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_MIN_ADVANTAGE", "")),
        "boss_relic_rollout_max_model_score_drop": str(
            os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_MAX_MODEL_SCORE_DROP", "")
        ),
        "boss_relic_rollout_snecko_energy_model_gap": str(
            os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_SNECKO_ENERGY_MODEL_GAP", "")
        ),
        "boss_relic_rollout_snecko_energy_min_score_lead": str(
            os.environ.get("SPIRECOMM_BOSS_RELIC_ROLLOUT_SNECKO_ENERGY_MIN_SCORE_LEAD", "")
        ),
        "event_low_hp_cost_guard": str(os.environ.get("SPIRECOMM_EVENT_LOW_HP_COST_GUARD", "")),
        "event_low_hp_cost_min_after": str(os.environ.get("SPIRECOMM_EVENT_LOW_HP_COST_MIN_AFTER", "")),
        "event_chosen_low_hp_cost_guard": str(os.environ.get("SPIRECOMM_EVENT_CHOSEN_LOW_HP_COST_GUARD", "")),
        "event_chosen_low_hp_cost_min_after": str(
            os.environ.get("SPIRECOMM_EVENT_CHOSEN_LOW_HP_COST_MIN_AFTER", "")
        ),
        "event_rollout_rerank": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_RERANK", "0")),
        "event_rollout_topk": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_TOPK", "")),
        "event_rollout_event_ids": str(
            os.environ.get(
                "SPIRECOMM_EVENT_ROLLOUT_EVENT_IDS",
                "Big Fish,Falling,The Library,MindBloom,Mysterious Sphere,The Woman in Blue",
            )
        ),
        "event_rollout_mysterious_sphere_floor_min": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MYSTERIOUS_SPHERE_FLOOR_MIN", "")
        ),
        "event_rollout_mysterious_sphere_hp_ratio_max": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MYSTERIOUS_SPHERE_HP_RATIO_MAX", "")
        ),
        "event_rollout_mindbloom_allowed_actions": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MINDBLOOM_ALLOWED_ACTIONS", "I Am Rich")
        ),
        "event_rollout_library_sleep_floor_min": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_LIBRARY_SLEEP_FLOOR_MIN", "")
        ),
        "event_rollout_library_read_floor_min": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_LIBRARY_READ_FLOOR_MIN", "")
        ),
        "event_rollout_floor_min": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_FLOOR_MIN", "")),
        "event_rollout_floor_max": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_FLOOR_MAX", "")),
        "event_rollout_max_steps": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_STEPS", "")),
        "event_rollout_max_floor_delta": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_FLOOR_DELTA", "")),
        "event_rollout_min_advantage": str(os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MIN_ADVANTAGE", "")),
        "event_rollout_max_model_score_drop": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP", "")
        ),
        "event_rollout_max_model_score_drop_events": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_EVENTS", "")
        ),
        "event_rollout_max_model_score_drop_by_event": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_MAX_MODEL_SCORE_DROP_BY_EVENT", "")
        ),
        "event_rollout_wemeetagain_block_give_card_override": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_GIVE_CARD_OVERRIDE", "")
        ),
        "event_rollout_wemeetagain_block_pay_gold_to_give_card_override": str(
            os.environ.get("SPIRECOMM_EVENT_ROLLOUT_WEMEETAGAIN_BLOCK_PAY_GOLD_TO_GIVE_CARD_OVERRIDE", "")
        ),
        "neow_rollout_rerank": str(os.environ.get("SPIRECOMM_NEOW_ROLLOUT_RERANK", "")),
        "neow_rollout_choice_indexes": str(os.environ.get("SPIRECOMM_NEOW_ROLLOUT_CHOICE_INDEXES", "")),
        "neow_rollout_max_steps": str(os.environ.get("SPIRECOMM_NEOW_ROLLOUT_MAX_STEPS", "")),
        "neow_rollout_max_floor_delta": str(os.environ.get("SPIRECOMM_NEOW_ROLLOUT_MAX_FLOOR_DELTA", "")),
        "neow_rollout_min_advantage": str(os.environ.get("SPIRECOMM_NEOW_ROLLOUT_MIN_ADVANTAGE", "")),
        "v3_combat_ensemble_models": str(os.environ.get("SPIRECOMM_V3_COMBAT_ENSEMBLE_MODELS", "")),
        "v3_combat_ensemble_weights": str(os.environ.get("SPIRECOMM_V3_COMBAT_ENSEMBLE_WEIGHTS", "")),
        "v3_combat_rescue_model": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_MODEL", "")),
        "v3_combat_rescue_margin_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_MARGIN_MAX", "")),
        "v3_combat_rescue_min_rescue_margin": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_MIN_RESCUE_MARGIN", "")),
        "v3_combat_rescue_require_disagree": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_REQUIRE_DISAGREE", "")),
        "v3_combat_rescue_require_suicidal_end_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_REQUIRE_SUICIDAL_END_GUARD", "")
        ),
        "v3_combat_rescue_primary_top_kinds": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_PRIMARY_TOP_KINDS", "")),
        "v3_combat_rescue_top_kinds": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_TOP_KINDS", "")),
        "v3_combat_rescue_room_types": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_ROOM_TYPES", "")),
        "v3_combat_rescue_floor_min": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_FLOOR_MIN", "")),
        "v3_combat_rescue_floor_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_FLOOR_MAX", "")),
        "v3_combat_rescue_hp_ratio_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_HP_RATIO_MAX", "")),
        "v3_combat_teacher_fallback": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK", "")),
        "v3_combat_teacher_blend_weight": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_BLEND_WEIGHT", "")),
        "v3_combat_teacher_fallback_margin_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_MARGIN_MAX", "")),
        "v3_combat_teacher_fallback_require_potion_candidate": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_REQUIRE_POTION_CANDIDATE", "")
        ),
        "v3_combat_teacher_fallback_require_suicidal_end_guard": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_REQUIRE_SUICIDAL_END_GUARD", "")
        ),
        "v3_combat_teacher_fallback_room_types": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_ROOM_TYPES", "")),
        "v3_combat_teacher_fallback_floor_min": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_FLOOR_MIN", "")),
        "v3_combat_teacher_fallback_floor_max": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_FLOOR_MAX", "")),
        "v3_combat_teacher_fallback_top_kinds": str(os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_TOP_KINDS", "")),
        "v3_combat_teacher_fallback_model_top_kinds": str(
            os.environ.get("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_MODEL_TOP_KINDS", "")
        ),
        "fast_floor_skip_combat_source_flags": str(os.environ.get("SPIRECOMM_FAST_FLOOR_SKIP_COMBAT_SOURCE_FLAGS", "")),
        "card_score_bias_enabled": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_ENABLED", "")),
        "card_score_bias_tier1_value": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER1_VALUE", "")),
        "card_score_bias_tier2_value": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER2_VALUE", "")),
        "card_score_bias_boost_value": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_BOOST_VALUE", "")),
        "card_score_bias_tier1_cards": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER1_CARDS", "")),
        "card_score_bias_tier2_cards": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_TIER2_CARDS", "")),
        "card_score_bias_boost_cards": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_BOOST_CARDS", "")),
        "card_score_bias_json": str(os.environ.get("SPIRECOMM_CARD_SCORE_BIAS_JSON", "")),
        "early_card_reward_extra_threshold": str(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_EXTRA_THRESHOLD", "")),
        "early_card_reward_skip_power": str(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_SKIP_POWER", "")),
        "early_card_reward_skip_min_weight": str(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_SKIP_MIN_WEIGHT", "")),
        "early_card_reward_attack_bias_max": str(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_MAX", "")),
        "early_card_reward_attack_bias_power": str(os.environ.get("SPIRECOMM_EARLY_CARD_REWARD_ATTACK_BIAS_POWER", "")),
        "upgrade_rate_prior_weight_override": str(os.environ.get("SPIRECOMM_UPGRADE_RATE_PRIOR_WEIGHT_OVERRIDE", "")),
        "upgrade_card_score_bias_scale": str(os.environ.get("SPIRECOMM_UPGRADE_CARD_SCORE_BIAS_SCALE", "")),
        "card_archetype_bias_enabled": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_BIAS_ENABLED", "")),
        "card_archetype_strength_bias": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_STRENGTH_BIAS", "")),
        "card_archetype_block_bias": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_BLOCK_BIAS", "")),
        "card_archetype_aoe_bias": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_AOE_BIAS", "")),
        "card_archetype_exhaust_bias": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_EXHAUST_BIAS", "")),
        "card_archetype_strength_half_count": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_STRENGTH_HALF_COUNT", "")),
        "card_archetype_strength_full_count": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_STRENGTH_FULL_COUNT", "")),
        "card_archetype_aoe_max_count": str(os.environ.get("SPIRECOMM_CARD_ARCHETYPE_AOE_MAX_COUNT", "")),
        "write_results_json": bool(write_results_json),
        "write_first_partial_summary": bool(args.write_first_partial_summary),
        "summary_interval": int(args.summary_interval),
        "result_flush_interval": int(args.result_flush_interval),
        "task_batch_size": int(args.task_batch_size),
        "compact_floor_results": bool(args.metrics_mode == "floor" and args.trace_mode == "none"),
        "start_snapshot_dir": str(args.start_snapshot_dir),
        "start_snapshot_cache_entries": int(args.start_snapshot_cache_entries),
        "paired_baseline_dir": str(args.paired_baseline_dir or ""),
        "early_stop_paired_delta_below": args.early_stop_paired_delta_below,
        "early_stop_min_paired": int(args.early_stop_min_paired),
        "rerun_timeouts": bool(args.rerun_timeouts),
    }
    started = time.time()
    existing_by_seed = _load_existing_results(output_dir) if args.resume else {}
    if args.rerun_timeouts and existing_by_seed:
        timeout_seeds = sorted(
            seed
            for seed in seeds
            if bool(existing_by_seed.get(seed, {}).get("timed_out"))
        )
        if timeout_seeds:
            timeout_seed_set = set(timeout_seeds)
            existing_by_seed = {
                seed: result
                for seed, result in existing_by_seed.items()
                if seed not in timeout_seed_set
            }
            preview = ",".join(str(seed) for seed in timeout_seeds[:20])
            if len(timeout_seeds) > 20:
                preview += f",...(+{len(timeout_seeds) - 20})"
            print(f"rerunning {len(timeout_seeds)} timed-out seeds: {preview}", flush=True)
    results: list[dict[str, Any]] = [existing_by_seed[seed] for seed in seeds if seed in existing_by_seed]
    pending_seeds = [seed for seed in seeds if seed not in existing_by_seed]
    paired_baseline_by_seed = _load_existing_results(args.paired_baseline_dir) if args.paired_baseline_dir else {}
    preload_selectors = _should_preload_selectors(args, pending_seeds)
    config["selectors_preloaded"] = bool(preload_selectors)
    (output_dir / "config.json").write_text(json.dumps(config | {"seeds": seeds}, ensure_ascii=False, indent=2), encoding="utf-8")
    results_jsonl = output_dir / "results.jsonl"
    if not args.resume and results_jsonl.exists():
        results_jsonl.unlink()
    if results:
        partial = _summarize(results, started=started, output_dir=output_dir)
        (output_dir / "summary_partial.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"resumed {len(results)}/{len(seeds)} mean_floor={partial['mean_floor']:.2f} "
            f"wins={partial['win_count']} deaths={partial['death_count']} errors={partial['error_count']}",
            flush=True,
        )
    mp_context = mp.get_context("fork") if preload_selectors else None
    if preload_selectors:
        _init_worker(config)
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_init_worker,
        initargs=(config,),
        mp_context=mp_context,
    ) as executor:
        result_buffer: list[dict[str, Any]] = []
        result_flush_interval = max(1, int(args.result_flush_interval))

        def flush_result_buffer() -> None:
            if not result_buffer:
                return
            _append_results_jsonl(results_jsonl, result_buffer)
            result_buffer.clear()

        task_batch_size = _bounded_task_batch_size(
            int(args.task_batch_size),
            pending_count=len(pending_seeds),
            workers=max(1, int(args.workers)),
            fast_like=bool(args.fast or args.metrics_mode == "floor" or args.trace_mode == "none"),
        )
        seed_batches = [
            pending_seeds[index : index + task_batch_size]
            for index in range(0, len(pending_seeds), task_batch_size)
        ]
        if pending_seeds:
            print(
                f"scheduler seeds={len(pending_seeds)} batches={len(seed_batches)} "
                f"task_batch_size={task_batch_size} workers={args.workers}",
                flush=True,
            )
        futures = {executor.submit(_run_seed_batch, seed_batch): seed_batch for seed_batch in seed_batches}
        stop_requested = False
        try:
            for future in as_completed(futures):
                for result in future.result():
                    results.append(result)
                    result_buffer.append(result)
                    if len(result_buffer) >= result_flush_interval:
                        flush_result_buffer()
                    completed = len(results)
                    pending_completed = completed - len(existing_by_seed)
                    summary_interval = int(args.summary_interval)
                    write_first_partial = bool(args.write_first_partial_summary) and pending_completed == 1
                    if write_first_partial or (summary_interval > 0 and completed % summary_interval == 0) or completed == len(seeds):
                        flush_result_buffer()
                        partial = _summarize(
                            results,
                            started=started,
                            output_dir=output_dir,
                            paired_baseline_by_seed=paired_baseline_by_seed,
                        )
                        (output_dir / "summary_partial.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(
                            f"completed {completed}/{len(seeds)} mean_floor={partial['mean_floor']:.2f} "
                            f"wins={partial['win_count']} deaths={partial['death_count']} errors={partial['error_count']}",
                            flush=True,
                        )
                        if paired_baseline_by_seed:
                            paired = _paired_floor_delta(results, paired_baseline_by_seed)
                            if int(paired["count"]) > 0:
                                print(
                                    "paired "
                                    f"n={paired['count']} delta={paired['delta']} "
                                    f"mean_delta={float(paired['mean_delta']):+.4f} "
                                    f"up/down/same={paired['up']}/{paired['down']}/{paired['same']}",
                                    flush=True,
                                )
                            if (
                                args.early_stop_paired_delta_below is not None
                                and int(paired["count"]) >= int(args.early_stop_min_paired)
                                and int(paired["delta"]) <= int(args.early_stop_paired_delta_below)
                            ):
                                print(
                                    "early_stop_paired_delta "
                                    f"n={paired['count']} delta={paired['delta']} "
                                    f"threshold={args.early_stop_paired_delta_below}",
                                    flush=True,
                                )
                                for pending in futures:
                                    if pending is not future:
                                        pending.cancel()
                                stop_requested = True
                                break
                if stop_requested:
                    break
        finally:
            flush_result_buffer()

    results.sort(key=lambda item: int(item["seed"]))
    summary = _summarize(
        results,
        started=started,
        output_dir=output_dir,
        paired_baseline_by_seed=paired_baseline_by_seed,
    )
    if write_results_json:
        (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
