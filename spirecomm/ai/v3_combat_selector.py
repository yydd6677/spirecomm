from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from spirecomm.ai.torch_compat import require_torch, torch
from spirecomm.ai.v3_combat_features import (
    action_key,
    action_keys_are_unique,
    clone_env_blob,
    encode_candidate_with_before_summary,
    encode_state_summary,
    incoming_damage,
    root_combat_actions,
    step_branch_from_blob,
)
from spirecomm.ai.v3_combat_model import V3CombatCandidateScorer, load_v3_combat_checkpoint
from spirecomm.ai.v3_combat_transformer import (
    collate_root_transformer_records,
    collate_transformer_records,
    collate_transformer_candidates_shared_before,
    entity_index_from_vocab,
    encode_root_transformer_actions,
    load_v3_combat_transformer_checkpoint,
    root_token_spec_from_payload,
    token_spec_from_payload,
)
from spirecomm.native_sim_v3.serialize import combat_state as serialize_v3_combat_state


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _configure_strict_cuda_inference(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    if not _env_bool("SPIRECOMM_TORCH_STRICT_CUDA_DETERMINISM", False):
        return
    if torch is None:
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class V3CandidateCombatSelector:
    """Score v3 combat legal actions by one-step successor features."""

    handles_potions = True

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path("/home/yydd/spirecomm/models/v3_combat_scorer.pt")
        self.device = device
        _configure_strict_cuda_inference(device)
        self.model = model
        self.model_kind = str(getattr(model, "model_kind", "mlp")) if model is not None else "mlp"
        self.transformer_entity_index: dict[str, int] | None = None
        self.transformer_token_spec: Any | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.ensemble_models: list[Any] = []
        self.ensemble_weights: list[float] = [1.0]
        self.rescue_model: Any | None = None
        self.last_rescue_used: bool = False
        self.last_rescue_primary_margin: float | None = None
        self.last_rescue_rescue_margin: float | None = None
        self.last_rescue_primary_action: dict[str, Any] | None = None
        self.last_rescue_action: dict[str, Any] | None = None
        self.last_suicidal_end_guard_used: bool = False
        self.last_dangerous_end_bias_used: bool = False
        self.last_potion_over_end_used: bool = False
        self.last_block_over_end_used: bool = False
        self.last_sharp_hide_danger_guard_used: bool = False
        self.last_lethal_card_over_setup_used: bool = False
        self.last_lethal_sequence_preserve_used: bool = False
        self.last_setup_power_over_basic_attack_used: bool = False
        self.last_high_block_progress_guard_used: bool = False
        self.last_monster_block_progress_guard_used: bool = False
        self.last_danger_block_progress_guard_used: bool = False
        self.last_gremlin_nob_skill_bias_used: bool = False
        self.last_short_win_guard_used: bool = False
        self.last_teacher_fallback_used: bool = False
        self.last_teacher_blend_used: bool = False
        self.last_branch_advisor_used: bool = False
        self.last_suicidal_action_guard_used: bool = False
        self.last_forced_turn_survival_guard_used: bool = False
        self.last_policy_survival_guard_used: bool = False
        self.last_post_forced_turn_survival_guard_used: bool = False
        self.last_post_action_survival_guard_used: bool = False
        self.last_delayed_death_guard_used: bool = False
        self.last_survival_potion_rescue_used: bool = False
        self.last_pre_guard_scores: list[float] = []
        self.last_final_scores: list[float] = []
        self.last_pre_guard_top_index: int | None = None
        self.last_final_top_index: int | None = None
        self.last_pre_guard_top_action: dict[str, Any] | None = None
        self.last_final_top_action: dict[str, Any] | None = None
        self.last_guard_names: list[str] = []
        self.last_root_actions: list[dict[str, Any]] = []
        self.last_before_state: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.branch_advisor_model: Any | None = None
        self._rollout_cpu_selector: Any | None = None
        self._survival_policy_probe_depth: int = 0
        self._monster_block_progress_signature: tuple[Any, ...] | None = None
        self._monster_block_progress_stall_count: int = 0
        if self.model is None and self.checkpoint_path.exists():
            try:
                loaded_model, checkpoint = load_v3_combat_checkpoint(self.checkpoint_path, device=device)
                self.model = loaded_model
                self.checkpoint = checkpoint
                self.model_kind = "mlp"
            except Exception as exc:
                try:
                    loaded_model, checkpoint = load_v3_combat_transformer_checkpoint(self.checkpoint_path, device=device)
                    self.model = loaded_model
                    self.checkpoint = checkpoint
                    self.model_kind = "transformer"
                    self.transformer_entity_index = entity_index_from_vocab(checkpoint.get("entity_vocab"))
                    if bool(getattr(loaded_model, "expects_root_batch", False)):
                        self.transformer_token_spec = root_token_spec_from_payload(checkpoint.get("token_schema"))
                    else:
                        self.transformer_token_spec = token_spec_from_payload(checkpoint.get("token_schema"))
                except Exception as transformer_exc:
                    self.last_error = f"v3_candidate_checkpoint_load_failed:{exc}; transformer_load_failed:{transformer_exc}"
        if self.model is not None:
            self.model.to(device)
            self.model.eval()
            self.model_kind = str(getattr(self.model, "model_kind", self.model_kind))
            self._load_env_ensemble_models()
            self._load_env_rescue_model()
            self._load_env_branch_advisor_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def rollout_cpu_selector(self) -> Any:
        """Lazily load a CPU mirror for deterministic rollout continuations."""

        if str(self.device).lower().startswith("cpu"):
            return self
        if self._rollout_cpu_selector is not None:
            return self._rollout_cpu_selector
        try:
            selector = V3CandidateCombatSelector(self.checkpoint_path, device="cpu")
            if selector.available:
                self._rollout_cpu_selector = selector
                return selector
        except Exception as exc:
            self.last_error = f"v3_candidate_rollout_cpu_selector_failed:{exc}"
        self._rollout_cpu_selector = self
        return self

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return default
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _env_ensemble_model_paths() -> list[Path]:
        raw = os.environ.get("SPIRECOMM_V3_COMBAT_ENSEMBLE_MODELS")
        if raw is None or not str(raw).strip():
            return []
        return [Path(part.strip()) for part in str(raw).split(",") if part.strip()]

    @staticmethod
    def _env_ensemble_weights(count: int) -> list[float]:
        raw = os.environ.get("SPIRECOMM_V3_COMBAT_ENSEMBLE_WEIGHTS")
        if raw is None or not str(raw).strip():
            return [1.0] * count
        try:
            weights = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
        except (TypeError, ValueError):
            return [1.0] * count
        if len(weights) != count or sum(max(0.0, weight) for weight in weights) <= 0.0:
            return [1.0] * count
        return [max(0.0, weight) for weight in weights]

    def _load_env_ensemble_models(self) -> None:
        extra_paths = self._env_ensemble_model_paths()
        if not extra_paths:
            return
        primary_checkpoint = self.checkpoint or {}
        primary_vocab = list(primary_checkpoint.get("entity_vocab") or [])
        primary_expects_root = bool(getattr(self.model, "expects_root_batch", False))
        loaded: list[Any] = []
        for path in extra_paths:
            if path.resolve() == self.checkpoint_path.resolve():
                continue
            try:
                if self.model_kind == "transformer":
                    model, checkpoint = load_v3_combat_transformer_checkpoint(path, device=self.device)
                    if bool(getattr(model, "expects_root_batch", False)) != primary_expects_root:
                        raise ValueError("ensemble transformer root-batch mode mismatch")
                    if primary_vocab and list(checkpoint.get("entity_vocab") or []) != primary_vocab:
                        raise ValueError("ensemble transformer entity vocab mismatch")
                else:
                    model, checkpoint = load_v3_combat_checkpoint(path, device=self.device)
                if str(getattr(model, "model_kind", self.model_kind)) != self.model_kind:
                    raise ValueError("ensemble model kind mismatch")
                model.to(self.device)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                loaded.append(model)
            except Exception as exc:
                self.last_error = f"v3_candidate_ensemble_load_failed:{path}:{exc}"
                self.ensemble_models = []
                self.ensemble_weights = [1.0]
                return
        self.ensemble_models = loaded
        self.ensemble_weights = self._env_ensemble_weights(1 + len(loaded))

    @staticmethod
    def _env_rescue_model_path() -> Path | None:
        raw = os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_MODEL")
        if raw is None or not str(raw).strip():
            return None
        return Path(str(raw).strip())

    def _load_env_rescue_model(self) -> None:
        path = self._env_rescue_model_path()
        if path is None:
            return
        try:
            if path.resolve() == self.checkpoint_path.resolve():
                return
        except Exception:
            pass
        primary_checkpoint = self.checkpoint or {}
        primary_vocab = list(primary_checkpoint.get("entity_vocab") or [])
        primary_expects_root = bool(getattr(self.model, "expects_root_batch", False))
        try:
            if self.model_kind == "transformer":
                model, checkpoint = load_v3_combat_transformer_checkpoint(path, device=self.device)
                if bool(getattr(model, "expects_root_batch", False)) != primary_expects_root:
                    raise ValueError("rescue transformer root-batch mode mismatch")
                if primary_vocab and list(checkpoint.get("entity_vocab") or []) != primary_vocab:
                    raise ValueError("rescue transformer entity vocab mismatch")
            else:
                model, checkpoint = load_v3_combat_checkpoint(path, device=self.device)
            if str(getattr(model, "model_kind", self.model_kind)) != self.model_kind:
                raise ValueError("rescue model kind mismatch")
            model.to(self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.rescue_model = model
        except Exception as exc:
            self.last_error = f"v3_candidate_rescue_load_failed:{path}:{exc}"
            self.rescue_model = None

    @staticmethod
    def _env_branch_advisor_model_path() -> Path | None:
        raw = os.environ.get("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_MODEL")
        if raw is None or not str(raw).strip():
            return None
        return Path(str(raw).strip())

    def _load_env_branch_advisor_model(self) -> None:
        path = self._env_branch_advisor_model_path()
        if path is None:
            return
        try:
            model, _checkpoint = load_v3_combat_checkpoint(path, device=self.device)
            model.to(self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.branch_advisor_model = model
        except Exception as exc:
            self.last_error = f"v3_candidate_branch_advisor_load_failed:{path}:{exc}"
            self.branch_advisor_model = None

    @staticmethod
    def _normal_room_potion_penalty() -> float:
        raw = os.environ.get("SPIRECOMM_V3_NORMAL_ROOM_POTION_PENALTY")
        if raw is None or not str(raw).strip():
            return 5.0
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 5.0

    @staticmethod
    def _action_kind_bias(kind: str) -> float:
        env_name = {
            "card": "SPIRECOMM_V3_COMBAT_CARD_BIAS",
            "potion": "SPIRECOMM_V3_COMBAT_POTION_BIAS",
            "end": "SPIRECOMM_V3_COMBAT_END_BIAS",
        }.get(kind)
        if not env_name:
            return 0.0
        raw = os.environ.get(env_name)
        if raw is None or not str(raw).strip():
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _apply_runtime_score_adjustments(scores_tensor: Any, before_state: dict[str, Any], actions: list[dict[str, Any]]) -> Any:
        penalty = V3CandidateCombatSelector._normal_room_potion_penalty()
        room_type = str(before_state.get("room_type") or "")
        card_bias = V3CandidateCombatSelector._action_kind_bias("card")
        potion_bias = V3CandidateCombatSelector._action_kind_bias("potion")
        end_bias = V3CandidateCombatSelector._action_kind_bias("end")
        end_bias_room_types = V3CandidateCombatSelector._env_set("SPIRECOMM_V3_COMBAT_END_BIAS_ROOM_TYPES")
        if end_bias_room_types and room_type not in end_bias_room_types:
            end_bias = 0.0
        has_kind_bias = card_bias != 0.0 or potion_bias != 0.0 or end_bias != 0.0
        has_normal_potion_penalty = penalty > 0.0 and room_type == "MonsterRoom"
        if not has_kind_bias and not has_normal_potion_penalty:
            return scores_tensor
        adjusted = scores_tensor.clone()
        for index, action in enumerate(actions):
            kind = str(action.get("kind") or "")
            if kind == "potion" and has_normal_potion_penalty:
                adjusted[index] = adjusted[index] - penalty
            if kind == "card" and card_bias != 0.0:
                adjusted[index] = adjusted[index] + card_bias
            elif kind == "potion" and potion_bias != 0.0:
                adjusted[index] = adjusted[index] + potion_bias
            elif kind == "end" and end_bias != 0.0:
                adjusted[index] = adjusted[index] + end_bias
        return adjusted

    def _apply_gremlin_nob_skill_bias(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
    ) -> Any:
        skill_bias = self._env_float("SPIRECOMM_V3_COMBAT_GREMLIN_NOB_SKILL_BIAS", 0.0)
        if skill_bias == 0.0 or int(scores_tensor.numel()) <= 0:
            return scores_tensor
        if self._env_bool("SPIRECOMM_V3_COMBAT_GREMLIN_NOB_SKILL_BIAS_DISABLE_IN_MAP_ROLLOUT", True):
            if str(os.environ.get("SPIRECOMM_V3_COMBAT_IN_MAP_ROLLOUT") or "").strip():
                return scores_tensor
        if len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        if not self._state_has_live_monster_id(before_state, {"GremlinNob", "Gremlin Nob"}):
            return scores_tensor
        excluded_cards = self._env_set("SPIRECOMM_V3_COMBAT_GREMLIN_NOB_SKILL_BIAS_EXCLUDE_CARDS")
        adjusted = scores_tensor.clone()
        changed = False
        for index, action in enumerate(actions):
            if str(action.get("kind") or "") != "card":
                continue
            if self._action_card_type(before_state, action) != "SKILL":
                continue
            if excluded_cards:
                card = self._selected_card_from_state(before_state, action) or {}
                card_ids = {
                    str(action.get("card_id") or ""),
                    str(action.get("name") or ""),
                    str(card.get("card_id") or ""),
                    str(card.get("name") or ""),
                }
                if any(card_id in excluded_cards for card_id in card_ids if card_id):
                    continue
            adjusted[index] = adjusted[index] + float(skill_bias)
            changed = True
        if changed:
            self.last_gremlin_nob_skill_bias_used = True
        return adjusted

    @staticmethod
    def _suppress_suicidal_end_enabled() -> bool:
        raw = os.environ.get("SPIRECOMM_V3_SUPPRESS_SUICIDAL_END")
        if raw is None or not str(raw).strip():
            return True
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _state_player_current_hp(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        raw_hp = state.get("current_hp")
        if raw_hp is not None:
            try:
                return int(raw_hp)
            except (TypeError, ValueError):
                return None
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return None
        player = combat.get("player")
        if not isinstance(player, dict):
            return None
        raw_hp = player.get("current_hp")
        if raw_hp is None:
            return None
        try:
            return int(raw_hp)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _state_player_max_hp(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        raw_hp = state.get("max_hp")
        if raw_hp is not None:
            try:
                return int(raw_hp)
            except (TypeError, ValueError):
                return None
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return None
        player = combat.get("player")
        if not isinstance(player, dict):
            return None
        raw_hp = player.get("max_hp")
        if raw_hp is None:
            return None
        try:
            return int(raw_hp)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _state_player_block(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        player = state.get("player")
        if not isinstance(player, dict):
            combat = state.get("combat_state")
            if isinstance(combat, dict):
                player = combat.get("player")
        if not isinstance(player, dict):
            return None
        raw_block = player.get("block")
        if raw_block is None:
            return None
        try:
            return int(raw_block)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _state_phase(state: dict[str, Any] | None) -> str:
        if not isinstance(state, dict):
            return ""
        return str(state.get("phase") or state.get("screen") or state.get("screen_type") or "")

    @staticmethod
    def _state_combat(state: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        combat = state.get("combat_state")
        if isinstance(combat, dict):
            return combat
        return {}

    @staticmethod
    def _selected_card_from_state(before_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
        action_card = action.get("card")
        if isinstance(action_card, dict):
            return action_card
        index = action.get("card_index")
        if index is None:
            index = action.get("source_index")
        try:
            card_index = int(index)
        except (TypeError, ValueError):
            return None
        hand = V3CandidateCombatSelector._state_combat(before_state).get("hand")
        if not isinstance(hand, list):
            return None
        if 0 <= card_index < len(hand) and isinstance(hand[card_index], dict):
            return hand[card_index]
        return None

    @staticmethod
    def _action_card_type(before_state: dict[str, Any], action: dict[str, Any]) -> str:
        card = V3CandidateCombatSelector._selected_card_from_state(before_state, action)
        return str((card or {}).get("type") or action.get("type") or "").upper()

    @staticmethod
    def _state_uncovered_incoming(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        block = V3CandidateCombatSelector._state_player_block(state)
        if block is None:
            return None
        try:
            incoming = int(incoming_damage(state))
        except Exception:
            return None
        return max(0, incoming - max(0, int(block)))

    @staticmethod
    def _is_post_combat_state(before_state: dict[str, Any], after_state: dict[str, Any] | None) -> bool:
        if not isinstance(after_state, dict):
            return False
        before_phase = V3CandidateCombatSelector._state_phase(before_state)
        after_phase = V3CandidateCombatSelector._state_phase(after_state)
        if before_phase != "COMBAT":
            return False
        if after_phase not in {"CARD_REWARD", "BOSS_RELIC", "MAP", "EVENT", "SHOP", "CAMPFIRE", "TREASURE", "COMPLETE", "VICTORY"}:
            return False
        return not bool(V3CandidateCombatSelector._state_combat(after_state))

    @staticmethod
    def _state_floor(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        for key in ("floor", "floor_num", "floor_number", "current_floor"):
            raw_value = state.get(key)
            if raw_value is None:
                continue
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _state_live_monster_count(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return None
        monsters = combat.get("monsters")
        if not isinstance(monsters, list):
            return None
        live = 0
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
                live += 1
        return live

    @staticmethod
    def _state_has_live_monster_id(state: dict[str, Any] | None, monster_ids: set[str]) -> bool:
        if not isinstance(state, dict):
            return False
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return False
        monsters = combat.get("monsters")
        if not isinstance(monsters, list):
            return False
        normalized_ids = {str(monster_id).strip() for monster_id in monster_ids if str(monster_id).strip()}
        for monster in monsters:
            if not isinstance(monster, dict):
                continue
            if bool(monster.get("is_gone") or monster.get("half_dead")):
                continue
            try:
                hp = int(monster.get("current_hp") or 0)
            except (TypeError, ValueError):
                hp = 0
            if hp <= 0:
                continue
            raw_ids = {
                str(monster.get("monster_id") or ""),
                str(monster.get("id") or ""),
                str(monster.get("name") or ""),
            }
            if any(raw_id in normalized_ids for raw_id in raw_ids if raw_id):
                return True
        return False

    @staticmethod
    def _state_live_monster_power_amount(state: dict[str, Any] | None, power_ids: set[str]) -> int:
        if not isinstance(state, dict):
            return 0
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return 0
        monsters = combat.get("monsters")
        if not isinstance(monsters, list):
            return 0
        normalized_ids = {str(power_id).strip() for power_id in power_ids if str(power_id).strip()}
        total = 0
        for monster in monsters:
            if not isinstance(monster, dict):
                continue
            if bool(monster.get("is_gone") or monster.get("half_dead")):
                continue
            try:
                hp = int(monster.get("current_hp") or 0)
            except (TypeError, ValueError):
                hp = 0
            if hp <= 0:
                continue
            powers = monster.get("powers")
            if not isinstance(powers, list):
                continue
            for power in powers:
                if not isinstance(power, dict):
                    continue
                raw_ids = {
                    str(power.get("power_id") or ""),
                    str(power.get("id") or ""),
                    str(power.get("name") or ""),
                }
                if not any(raw_id in normalized_ids for raw_id in raw_ids if raw_id):
                    continue
                try:
                    total += max(0, int(power.get("amount") or 0))
                except (TypeError, ValueError):
                    total += 1
        return total

    @staticmethod
    def _state_live_monster_hp(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return None
        monsters = combat.get("monsters")
        if not isinstance(monsters, list):
            return None
        total = 0
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
                total += hp
        return total

    @staticmethod
    def _state_live_monster_block(state: dict[str, Any] | None) -> int | None:
        if not isinstance(state, dict):
            return None
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return None
        monsters = combat.get("monsters")
        if not isinstance(monsters, list):
            return None
        total = 0
        for monster in monsters:
            if not isinstance(monster, dict):
                continue
            if bool(monster.get("is_gone") or monster.get("half_dead")):
                continue
            try:
                hp = int(monster.get("current_hp") or 0)
            except (TypeError, ValueError):
                hp = 0
            if hp <= 0:
                continue
            try:
                total += max(0, int(monster.get("block") or 0))
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _monster_hp_progress(before_state: dict[str, Any], after_state: dict[str, Any] | None) -> int | None:
        before_hp = V3CandidateCombatSelector._state_live_monster_hp(before_state)
        after_hp = V3CandidateCombatSelector._state_live_monster_hp(after_state)
        if before_hp is None or after_hp is None:
            return None
        return max(0, int(before_hp) - int(after_hp))

    @staticmethod
    def _monster_block_progress(before_state: dict[str, Any], after_state: dict[str, Any] | None) -> int | None:
        before_block = V3CandidateCombatSelector._state_live_monster_block(before_state)
        after_block = V3CandidateCombatSelector._state_live_monster_block(after_state)
        if before_block is None or after_block is None:
            return None
        return max(0, int(before_block) - int(after_block))

    @staticmethod
    def _monster_hp_stall_signature(state: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if not isinstance(state, dict):
            return None
        combat = state.get("combat_state")
        if not isinstance(combat, dict):
            return None
        monsters = combat.get("monsters")
        if not isinstance(monsters, list):
            return None
        live: list[tuple[str, int, bool, bool]] = []
        for monster in monsters:
            if not isinstance(monster, dict):
                continue
            monster_id = str(monster.get("monster_id") or monster.get("id") or monster.get("name") or "")
            try:
                hp = int(monster.get("current_hp") or 0)
            except (TypeError, ValueError):
                hp = 0
            half_dead = bool(monster.get("half_dead"))
            gone = bool(monster.get("is_gone"))
            if hp > 0 and not half_dead and not gone:
                live.append((monster_id, hp, half_dead, gone))
        if not live:
            return None
        return (
            str(state.get("room_type") or ""),
            int(state.get("floor") or 0),
            tuple(live),
        )

    def _apply_suicidal_end_guard(
        self,
        scores_tensor: Any,
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
        env: Any | None = None,
    ) -> Any:
        if not V3CandidateCombatSelector._suppress_suicidal_end_enabled():
            return scores_tensor
        if int(scores_tensor.numel()) <= 0 or len(actions) != len(after_states):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        if str(actions[best_index].get("kind") or "") != "end":
            return scores_tensor
        end_after_state = after_states[best_index]
        end_hp = V3CandidateCombatSelector._state_player_current_hp(end_after_state)
        if end_hp is None:
            return scores_tensor
        live_after_end = V3CandidateCombatSelector._state_live_monster_count(end_after_state)
        if live_after_end is not None and live_after_end <= 0:
            return scores_tensor
        if end_hp > 0:
            dangerous_bias = V3CandidateCombatSelector._env_float("SPIRECOMM_V3_COMBAT_DANGEROUS_END_BIAS", 0.0)
            if dangerous_bias == 0.0:
                return scores_tensor
            max_hp = V3CandidateCombatSelector._state_player_max_hp(end_after_state)
            hp_ratio_max = V3CandidateCombatSelector._env_float("SPIRECOMM_V3_COMBAT_DANGEROUS_END_HP_RATIO_MAX", 0.0)
            hp_max = V3CandidateCombatSelector._env_int("SPIRECOMM_V3_COMBAT_DANGEROUS_END_HP_MAX", 0)
            ratio_match = max_hp is not None and max_hp > 0 and hp_ratio_max > 0.0 and (float(end_hp) / float(max_hp)) <= hp_ratio_max
            flat_match = hp_max > 0 and int(end_hp) <= hp_max
            if not ratio_match and not flat_match:
                return scores_tensor
            adjusted = scores_tensor.clone()
            adjusted[best_index] = adjusted[best_index] + float(dangerous_bias)
            self.last_dangerous_end_bias_used = True
            return adjusted
        has_survivable_non_end = False
        for action, after_state in zip(actions, after_states, strict=False):
            if str(action.get("kind") or "") == "end":
                continue
            hp = V3CandidateCombatSelector._state_player_current_hp(after_state)
            if hp is not None and hp > 0:
                has_survivable_non_end = True
                break
        if not has_survivable_non_end:
            return scores_tensor
        adjusted = scores_tensor.clone()
        safe_indices = self._suicidal_end_safe_non_end_indices(env, actions)
        if safe_indices:
            safe_set = set(safe_indices)
            for index, action in enumerate(actions):
                if str(action.get("kind") or "") == "end" or index not in safe_set:
                    adjusted[index] = float("-inf")
            if not bool(torch.isfinite(adjusted).any().item()):
                adjusted = scores_tensor.clone()
                adjusted[best_index] = float("-inf")
        else:
            adjusted[best_index] = float("-inf")
        self.last_suicidal_end_guard_used = True
        return adjusted

    def _apply_suicidal_action_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_SUPPRESS_SUICIDAL_ACTION", True):
            return scores_tensor
        if int(scores_tensor.numel()) <= 1 or len(actions) != len(after_states):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        if str(actions[best_index].get("kind") or "") == "end":
            return scores_tensor
        best_after = after_states[best_index]
        best_hp = self._state_player_current_hp(best_after)
        if best_hp is None or best_hp > 0 or self._is_post_combat_state(before_state, best_after):
            return scores_tensor
        safe_indices: list[int] = []
        allow_safe_end = self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_ALLOW_SAFE_END", False)
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if index == best_index:
                continue
            if str(action.get("kind") or "") == "end" and not allow_safe_end:
                continue
            hp = self._state_player_current_hp(after_state)
            if self._is_post_combat_state(before_state, after_state) or (hp is not None and hp > 0):
                safe_indices.append(index)
        if not safe_indices:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[best_index] = float("-inf")
        if not bool(torch.isfinite(adjusted).any().item()):
            return scores_tensor
        self.last_suicidal_action_guard_used = True
        return adjusted

    def _apply_forced_turn_survival_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        env: Any | None,
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_FORCED_TURN_SURVIVAL_GUARD", True):
            return scores_tensor
        if env is None or int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        player_hp = self._state_player_current_hp(before_state)
        uncovered = self._state_uncovered_incoming(before_state)
        if player_hp is None or uncovered is None or int(uncovered) < int(player_hp):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_survives = self._action_survives_forced_turn(env, before_state, actions[best_index])
        if best_survives is not False:
            return scores_tensor
        safe_indices: list[int] = []
        allow_safe_end = self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_ALLOW_SAFE_END", False)
        for index, action in enumerate(actions):
            if index == best_index:
                continue
            if str(action.get("kind") or "") == "end" and not allow_safe_end:
                continue
            survives = self._action_survives_forced_turn(env, before_state, action)
            if survives is True:
                safe_indices.append(index)
        if not safe_indices:
            return scores_tensor
        adjusted = scores_tensor.clone()
        if self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_RESTRICT_SAFE", True):
            safe_set = set(safe_indices)
            for index in range(len(actions)):
                if index not in safe_set:
                    adjusted[index] = float("-inf")
        else:
            adjusted[best_index] = float("-inf")
        if not bool(torch.isfinite(adjusted).any().item()):
            return scores_tensor
        self.last_forced_turn_survival_guard_used = True
        return adjusted

    def _apply_policy_survival_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        env: Any | None,
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_GUARD", True):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if env is None or int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        player_hp = self._state_player_current_hp(before_state)
        uncovered = self._state_uncovered_incoming(before_state)
        if player_hp is None or uncovered is None or int(uncovered) < int(player_hp):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_forced = self._action_survives_forced_turn(env, before_state, actions[best_index])
        if best_forced is not None:
            return scores_tensor
        max_decisions = max(1, self._env_int("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_MAX_DECISIONS", 6))
        best_policy = self._action_survives_policy_turn(env, before_state, actions[best_index], max_decisions=max_decisions)
        if best_policy is not False:
            return scores_tensor
        topk = max(1, self._env_int("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_TOPK", 8))
        ranked_indices = [
            int(index)
            for index in torch.argsort(scores_tensor, descending=True).detach().cpu().tolist()
            if int(index) != best_index
        ]
        candidate_indices = ranked_indices[:topk]
        if self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS", False):
            seen_indices = set(candidate_indices)
            for index, action in enumerate(actions):
                if index == best_index or index in seen_indices:
                    continue
                if str(action.get("kind") or "") == "potion":
                    candidate_indices.append(index)
                    seen_indices.add(index)
        safe_indices: list[int] = []
        allow_safe_end = self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_ALLOW_SAFE_END", False)
        for index in candidate_indices:
            if str(actions[index].get("kind") or "") == "end" and not allow_safe_end:
                continue
            forced = self._action_survives_forced_turn(env, before_state, actions[index])
            if forced is True:
                safe_indices.append(index)
                break
            if forced is False:
                continue
            policy = self._action_survives_policy_turn(env, before_state, actions[index], max_decisions=max_decisions)
            if policy is True:
                safe_indices.append(index)
                break
        if not safe_indices:
            return scores_tensor
        if any(str(actions[index].get("kind") or "") == "potion" for index in safe_indices):
            self.last_survival_potion_rescue_used = True
        adjusted = scores_tensor.clone()
        if self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_RESTRICT_SAFE", True):
            safe_set = set(safe_indices)
            for index in range(len(actions)):
                if index not in safe_set:
                    adjusted[index] = float("-inf")
        else:
            adjusted[best_index] = float("-inf")
        if not bool(torch.isfinite(adjusted).any().item()):
            return scores_tensor
        self.last_policy_survival_guard_used = True
        return adjusted

    def _apply_post_forced_turn_survival_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
        env: Any | None,
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SURVIVAL_GUARD", True):
            return scores_tensor
        if env is None or int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        if len(actions) != len(after_states):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_after_state = after_states[best_index]
        if self._is_post_combat_state(before_state, best_after_state):
            return scores_tensor
        after_hp = self._state_player_current_hp(best_after_state)
        after_uncovered = self._state_uncovered_incoming(best_after_state)
        if not self._env_bool("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SKIP_UNCOVERED_GATE", True) and (
            after_hp is None or after_uncovered is None or int(after_uncovered) < int(after_hp)
        ):
            return scores_tensor
        best_forced = self._action_survives_forced_turn(env, before_state, actions[best_index])
        if best_forced is not False:
            return scores_tensor
        topk = max(1, self._env_int("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_SURVIVAL_TOPK", 8))
        allow_safe_end = self._env_bool(
            "SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_ALLOW_SAFE_END",
            self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_ALLOW_SAFE_END", False),
        )
        ranked_indices = [
            int(index)
            for index in torch.argsort(scores_tensor, descending=True).detach().cpu().tolist()
            if int(index) != best_index
        ]
        candidate_indices = ranked_indices[:topk]
        if self._env_bool("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_INCLUDE_POTIONS", False) or self._env_bool(
            "SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS",
            False,
        ):
            seen_indices = set(candidate_indices)
            for index, action in enumerate(actions):
                if index != best_index and index not in seen_indices and str(action.get("kind") or "") == "potion":
                    candidate_indices.append(index)
                    seen_indices.add(index)
        prefer_win = self._env_bool("SPIRECOMM_V3_COMBAT_POST_FORCED_TURN_PREFER_WIN", False)
        first_safe_index: int | None = None
        first_win_index: int | None = None
        for index in candidate_indices:
            if str(actions[index].get("kind") or "") == "end" and not allow_safe_end:
                continue
            if self._is_post_combat_state(before_state, after_states[index]):
                first_win_index = index
                if prefer_win:
                    break
                first_safe_index = index
                break
            if self._action_survives_forced_turn(env, before_state, actions[index]) is True:
                if first_safe_index is None:
                    first_safe_index = index
                if not prefer_win:
                    break
        chosen_index = first_win_index if first_win_index is not None else first_safe_index
        if chosen_index is None:
            return scores_tensor
        if str(actions[chosen_index].get("kind") or "") == "potion":
            self.last_survival_potion_rescue_used = True
        adjusted = scores_tensor.clone()
        if self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_RESTRICT_SAFE", True):
            for index in range(len(actions)):
                if index != chosen_index:
                    adjusted[index] = float("-inf")
        else:
            adjusted[best_index] = float("-inf")
        if not bool(torch.isfinite(adjusted).any().item()):
            return scores_tensor
        self.last_post_forced_turn_survival_guard_used = True
        return adjusted

    def _apply_post_action_survival_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
        env: Any | None,
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_POST_ACTION_SURVIVAL_GUARD", False):
            return scores_tensor
        if env is None or int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        if len(actions) != len(after_states):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_after_state = after_states[best_index]
        if self._is_post_combat_state(before_state, best_after_state):
            return scores_tensor
        after_hp = self._state_player_current_hp(best_after_state)
        after_uncovered = self._state_uncovered_incoming(best_after_state)
        if after_hp is None or after_uncovered is None or int(after_uncovered) < int(after_hp):
            return scores_tensor
        best_forced = self._action_survives_forced_turn(env, before_state, actions[best_index])
        if best_forced is True:
            return scores_tensor
        if best_forced is None:
            max_decisions = max(1, self._env_int("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_MAX_DECISIONS", 6))
            best_policy = self._action_survives_policy_turn(
                env,
                before_state,
                actions[best_index],
                max_decisions=max_decisions,
            )
            if best_policy is not False:
                return scores_tensor
        topk = max(1, self._env_int("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_TOPK", 8))
        allow_safe_end = self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_ALLOW_SAFE_END", False)
        ranked_indices = [
            int(index)
            for index in torch.argsort(scores_tensor, descending=True).detach().cpu().tolist()
            if int(index) != best_index
        ]
        candidate_indices = ranked_indices[:topk]
        if self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS", False):
            seen_indices = set(candidate_indices)
            for index, action in enumerate(actions):
                if index == best_index or index in seen_indices:
                    continue
                if str(action.get("kind") or "") == "potion":
                    candidate_indices.append(index)
                    seen_indices.add(index)
        safe_indices: list[int] = []
        for index in candidate_indices:
            if str(actions[index].get("kind") or "") == "end" and not allow_safe_end:
                continue
            if self._is_post_combat_state(before_state, after_states[index]):
                safe_indices.append(index)
                break
            forced = self._action_survives_forced_turn(env, before_state, actions[index])
            if forced is True:
                safe_indices.append(index)
                break
            if forced is False:
                continue
            policy = self._action_survives_policy_turn(
                env,
                before_state,
                actions[index],
                max_decisions=max(1, self._env_int("SPIRECOMM_V3_COMBAT_POLICY_SURVIVAL_MAX_DECISIONS", 6)),
            )
            if policy is True:
                safe_indices.append(index)
                break
        if not safe_indices:
            return scores_tensor
        if any(str(actions[index].get("kind") or "") == "potion" for index in safe_indices):
            self.last_survival_potion_rescue_used = True
        adjusted = scores_tensor.clone()
        if self._env_bool("SPIRECOMM_V3_COMBAT_SURVIVAL_GUARD_RESTRICT_SAFE", True):
            safe_set = set(safe_indices)
            for index in range(len(actions)):
                if index not in safe_set:
                    adjusted[index] = float("-inf")
        else:
            adjusted[best_index] = float("-inf")
        if not bool(torch.isfinite(adjusted).any().item()):
            return scores_tensor
        self.last_post_action_survival_guard_used = True
        return adjusted

    def _action_survives_forced_turn(
        self,
        env: Any,
        before_state: dict[str, Any],
        action: dict[str, Any],
    ) -> bool | None:
        try:
            root_blob = clone_env_blob(env, strip_debug_history=True)
            branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
            branch_state = self._state_from_env(branch)
            if self._is_post_combat_state(before_state, branch_state):
                return True
            hp = self._state_player_current_hp(branch_state)
            if hp is not None and hp <= 0:
                return False
            if str(getattr(branch, "phase", "")) != "COMBAT":
                return True
            followup_actions = root_combat_actions(branch)
            if len(followup_actions) != 1 or str(followup_actions[0].get("kind") or "") != "end":
                return None
            branch.step(dict(followup_actions[0]))
            ended = branch
            ended_state = self._state_from_env(ended)
            if self._is_post_combat_state(branch_state, ended_state):
                return True
            ended_hp = self._state_player_current_hp(ended_state)
            if ended_hp is None:
                return None
            return bool(ended_hp > 0)
        except Exception:
            return None

    def _action_survives_policy_turn(
        self,
        env: Any,
        before_state: dict[str, Any],
        action: dict[str, Any],
        *,
        max_decisions: int,
    ) -> bool | None:
        flag_names = (
            "last_rescue_used",
            "last_suicidal_end_guard_used",
            "last_dangerous_end_bias_used",
            "last_potion_over_end_used",
            "last_block_over_end_used",
            "last_sharp_hide_danger_guard_used",
            "last_lethal_card_over_setup_used",
            "last_lethal_sequence_preserve_used",
            "last_setup_power_over_basic_attack_used",
            "last_high_block_progress_guard_used",
            "last_monster_block_progress_guard_used",
            "last_danger_block_progress_guard_used",
            "last_gremlin_nob_skill_bias_used",
            "last_short_win_guard_used",
            "last_teacher_fallback_used",
            "last_teacher_blend_used",
            "last_branch_advisor_used",
            "last_suicidal_action_guard_used",
            "last_forced_turn_survival_guard_used",
            "last_policy_survival_guard_used",
            "last_post_forced_turn_survival_guard_used",
            "last_post_action_survival_guard_used",
            "last_delayed_death_guard_used",
            "last_survival_potion_rescue_used",
        )
        flag_snapshot = {name: getattr(self, name, None) for name in flag_names}
        error_snapshot = self.last_error
        try:
            root_blob = clone_env_blob(env, strip_debug_history=True)
            branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
            previous_state = before_state
            self._survival_policy_probe_depth += 1
            for _ in range(max(0, int(max_decisions))):
                branch_state = self._state_from_env(branch)
                if self._is_post_combat_state(previous_state, branch_state):
                    return True
                hp = self._state_player_current_hp(branch_state)
                if hp is not None and hp <= 0:
                    return False
                if str(getattr(branch, "phase", "")) != "COMBAT":
                    return True
                legal_actions = root_combat_actions(branch)
                if not legal_actions:
                    return None
                if len(legal_actions) == 1:
                    chosen = legal_actions[0]
                else:
                    chosen, _scores = self.choose_env(branch, legal_actions=legal_actions, return_scores=False)
                    if chosen is None:
                        return None
                branch_blob = clone_env_blob(branch, strip_debug_history=True)
                previous_state = branch_state
                branch = step_branch_from_blob(branch_blob, chosen, strip_debug_history=True)
            branch_state = self._state_from_env(branch)
            if self._is_post_combat_state(previous_state, branch_state):
                return True
            hp = self._state_player_current_hp(branch_state)
            if hp is not None and hp <= 0:
                return False
            return None
        except Exception:
            return None
        finally:
            self._survival_policy_probe_depth = max(0, self._survival_policy_probe_depth - 1)
            for name, value in flag_snapshot.items():
                setattr(self, name, value)
            self.last_error = error_snapshot

    def _action_policy_terminal_outcome(
        self,
        env: Any,
        before_state: dict[str, Any],
        action: dict[str, Any],
        *,
        max_decisions: int,
    ) -> str:
        flag_names = (
            "last_rescue_used",
            "last_suicidal_end_guard_used",
            "last_dangerous_end_bias_used",
            "last_potion_over_end_used",
            "last_block_over_end_used",
            "last_sharp_hide_danger_guard_used",
            "last_lethal_card_over_setup_used",
            "last_lethal_sequence_preserve_used",
            "last_setup_power_over_basic_attack_used",
            "last_high_block_progress_guard_used",
            "last_monster_block_progress_guard_used",
            "last_danger_block_progress_guard_used",
            "last_gremlin_nob_skill_bias_used",
            "last_short_win_guard_used",
            "last_teacher_fallback_used",
            "last_teacher_blend_used",
            "last_branch_advisor_used",
            "last_suicidal_action_guard_used",
            "last_forced_turn_survival_guard_used",
            "last_policy_survival_guard_used",
            "last_post_forced_turn_survival_guard_used",
            "last_post_action_survival_guard_used",
            "last_delayed_death_guard_used",
            "last_survival_potion_rescue_used",
        )
        flag_snapshot = {name: getattr(self, name, None) for name in flag_names}
        error_snapshot = self.last_error
        try:
            root_blob = clone_env_blob(env, strip_debug_history=True)
            branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
            previous_state = before_state
            self._survival_policy_probe_depth += 1
            for _ in range(max(0, int(max_decisions))):
                branch_state = self._state_from_env(branch)
                if self._is_post_combat_state(previous_state, branch_state):
                    return "win"
                hp = self._state_player_current_hp(branch_state)
                if hp is not None and hp <= 0:
                    return "death"
                if str(getattr(branch, "phase", "")) != "COMBAT":
                    return "safe"
                legal_actions = root_combat_actions(branch)
                if not legal_actions:
                    return "unknown"
                if len(legal_actions) == 1:
                    chosen = legal_actions[0]
                else:
                    chosen, _scores = self.choose_env(branch, legal_actions=legal_actions, return_scores=False)
                    if chosen is None:
                        return "unknown"
                branch_blob = clone_env_blob(branch, strip_debug_history=True)
                previous_state = branch_state
                branch = step_branch_from_blob(branch_blob, chosen, strip_debug_history=True)
            branch_state = self._state_from_env(branch)
            if self._is_post_combat_state(previous_state, branch_state):
                return "win"
            hp = self._state_player_current_hp(branch_state)
            if hp is not None and hp <= 0:
                return "death"
            return "unknown"
        except Exception:
            return "unknown"
        finally:
            self._survival_policy_probe_depth = max(0, self._survival_policy_probe_depth - 1)
            for name, value in flag_snapshot.items():
                setattr(self, name, value)
            self.last_error = error_snapshot

    def _apply_short_win_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        env: Any | None,
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_SHORT_WIN_GUARD", True):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if env is None or int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        player_hp = self._state_player_current_hp(before_state)
        uncovered = self._state_uncovered_incoming(before_state)
        if player_hp is None or uncovered is None or int(uncovered) < int(player_hp):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        max_decisions = max(1, self._env_int("SPIRECOMM_V3_COMBAT_SHORT_WIN_MAX_DECISIONS", 10))
        require_top_death = self._env_bool("SPIRECOMM_V3_COMBAT_SHORT_WIN_REQUIRE_TOP_DEATH", True)
        topk = max(1, self._env_int("SPIRECOMM_V3_COMBAT_SHORT_WIN_TOPK", 8))
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_SHORT_WIN_MARGIN_MAX", 0.50)
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        ranked_indices = [
            int(index)
            for index in torch.argsort(scores_tensor, descending=True).detach().cpu().tolist()
            if int(index) != best_index
        ]
        candidate_indices = ranked_indices[:topk]
        if self._env_bool("SPIRECOMM_V3_COMBAT_SHORT_WIN_INCLUDE_POTIONS", False) or self._env_bool(
            "SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS",
            False,
        ):
            seen_indices = set(candidate_indices)
            for index, action in enumerate(actions):
                if index == best_index or index in seen_indices:
                    continue
                if str(action.get("kind") or "") == "potion":
                    candidate_indices.append(index)
                    seen_indices.add(index)

        def _find_short_win(max_policy_decisions: int, *, potion_only: bool) -> int | None:
            best_outcome = self._action_policy_terminal_outcome(
                env,
                before_state,
                actions[best_index],
                max_decisions=max_policy_decisions,
            )
            if best_outcome == "win":
                return None
            if require_top_death and best_outcome != "death":
                return None
            if not require_top_death and best_outcome != "death":
                optional_room_types = self._env_set("SPIRECOMM_V3_COMBAT_SHORT_WIN_OPTIONAL_ROOM_TYPES")
                if optional_room_types:
                    room_type = str(before_state.get("room_type") or getattr(env, "current_room_type", "") or "")
                    if room_type not in optional_room_types:
                        return None
            for index in candidate_indices:
                action = actions[index]
                action_kind = str(action.get("kind") or "")
                if potion_only and action_kind != "potion":
                    continue
                score = float(scores_tensor[index].detach().cpu().item())
                if action_kind != "potion" and margin_max >= 0.0 and best_score - score > margin_max:
                    continue
                outcome = self._action_policy_terminal_outcome(
                    env,
                    before_state,
                    action,
                    max_decisions=max_policy_decisions,
                )
                if outcome == "win":
                    return index
            return None

        chosen_index = _find_short_win(max_decisions, potion_only=False)
        if chosen_index is None:
            potion_max_decisions = max(1, self._env_int("SPIRECOMM_V3_COMBAT_SHORT_WIN_POTION_MAX_DECISIONS", 14))
            if potion_max_decisions > max_decisions:
                chosen_index = _find_short_win(potion_max_decisions, potion_only=True)
        if chosen_index is None:
            return scores_tensor
        if str(actions[chosen_index].get("kind") or "") == "potion":
            self.last_survival_potion_rescue_used = True
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_short_win_guard_used = True
        return adjusted

    def _apply_danger_block_progress_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_GUARD", False):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_ROOM_TYPES") is None:
            room_types = {"MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        player_hp = self._state_player_current_hp(before_state)
        before_uncovered = self._state_uncovered_incoming(before_state)
        if player_hp is None or before_uncovered is None or int(before_uncovered) <= 0:
            return scores_tensor
        min_uncovered = self._env_int("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MIN_UNCOVERED", 12)
        hp_ratio_min = self._env_float("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_HP_RATIO_MIN", 0.65)
        if int(before_uncovered) < min_uncovered or float(before_uncovered) < float(player_hp) * hp_ratio_min:
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_after_state = after_states[best_index]
        if self._is_post_combat_state(before_state, best_after_state):
            return scores_tensor
        best_after_uncovered = self._state_uncovered_incoming(best_after_state)
        if best_after_uncovered is None:
            return scores_tensor
        best_reduction = max(0, int(before_uncovered) - int(best_after_uncovered))
        top_kinds = self._env_set("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_KINDS")
        if not top_kinds and os.environ.get("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_KINDS") is None:
            top_kinds = {"card", "end"}
        best_action = actions[best_index]
        best_kind = str(best_action.get("kind") or "")
        if top_kinds and best_kind not in top_kinds:
            return scores_tensor
        if best_kind == "card":
            excluded_top_cards = self._env_set("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_EXCLUDE_TOP_CARDS")
            if excluded_top_cards:
                top_card = self._selected_card_from_state(before_state, best_action) or {}
                top_card_ids = {
                    str(best_action.get("card_id") or ""),
                    str(best_action.get("name") or ""),
                    str(top_card.get("card_id") or ""),
                    str(top_card.get("name") or ""),
                }
                if any(card_id in excluded_top_cards for card_id in top_card_ids if card_id):
                    return scores_tensor
            top_types = {value.upper() for value in self._env_set("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_TYPES")}
            if not top_types and os.environ.get("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_TYPES") is None:
                top_types = {"POWER", "SKILL"}
            if self._action_card_type(before_state, best_action) not in top_types:
                return scores_tensor
            top_reduction_skip = self._env_int("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_TOP_REDUCTION_SKIP", 4)
            if best_reduction >= top_reduction_skip:
                return scores_tensor
        min_reduction = self._env_int("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MIN_REDUCTION", 5)
        min_extra_reduction = self._env_int("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MIN_EXTRA_REDUCTION", 4)
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_MARGIN_MAX", 2.0)
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        excluded_types = {value.upper() for value in self._env_set("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_EXCLUDE_CARD_TYPES")}
        if not excluded_types and os.environ.get("SPIRECOMM_V3_COMBAT_DANGER_BLOCK_PROGRESS_EXCLUDE_CARD_TYPES") is None:
            excluded_types = {"STATUS", "CURSE"}
        chosen_index: int | None = None
        chosen_reduction = -1
        chosen_score = float("-inf")
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if index == best_index or str(action.get("kind") or "") != "card":
                continue
            if self._action_card_type(before_state, action) in excluded_types:
                continue
            if self._is_post_combat_state(before_state, after_state):
                continue
            hp = self._state_player_current_hp(after_state)
            if hp is not None and hp <= 0:
                continue
            after_uncovered = self._state_uncovered_incoming(after_state)
            if after_uncovered is None:
                continue
            reduction = max(0, int(before_uncovered) - int(after_uncovered))
            if reduction < min_reduction or reduction - best_reduction < min_extra_reduction:
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if margin_max >= 0.0 and best_score - score > margin_max:
                continue
            if reduction > chosen_reduction or (reduction == chosen_reduction and score > chosen_score):
                chosen_index = index
                chosen_reduction = reduction
                chosen_score = score
        if chosen_index is None:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_short_win_guard_used = True
        return adjusted

    def _apply_delayed_death_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        env: Any | None,
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_GUARD", True):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if env is None or int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_ROOM_TYPES") is None:
            room_types = {"MonsterRoomElite", "MonsterRoomBoss"}
        if room_types:
            room_type = str(before_state.get("room_type") or getattr(env, "current_room_type", "") or "")
            if room_type not in room_types:
                return scores_tensor
        if self._env_bool("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_REQUIRE_DANGER", True):
            player_hp = self._state_player_current_hp(before_state)
            uncovered = self._state_uncovered_incoming(before_state)
            if player_hp is None or uncovered is None or int(uncovered) <= 0:
                return scores_tensor
            max_hp = self._state_player_max_hp(before_state)
            hp_ratio_max = self._env_float("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_HP_RATIO_MAX", 0.55)
            low_hp = max_hp is not None and max_hp > 0 and (float(player_hp) / float(max_hp)) <= hp_ratio_max
            if int(uncovered) < int(player_hp) and not low_hp:
                return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        top_kinds = self._env_set("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_TOP_KINDS")
        if top_kinds and str(actions[best_index].get("kind") or "") not in top_kinds:
            return scores_tensor
        max_decisions = max(1, self._env_int("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_MAX_DECISIONS", 10))
        best_outcome = self._action_policy_terminal_outcome(
            env,
            before_state,
            actions[best_index],
            max_decisions=max_decisions,
        )
        if best_outcome != "death":
            return scores_tensor
        topk = max(1, self._env_int("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_TOPK", 8))
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_MARGIN_MAX", 0.50)
        allow_unknown = self._env_bool("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_ALLOW_UNKNOWN", False)
        excluded_card_types = {value.upper() for value in self._env_set("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_EXCLUDE_CARD_TYPES")}
        if not excluded_card_types and os.environ.get("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_EXCLUDE_CARD_TYPES") is None:
            excluded_card_types = {"STATUS", "CURSE"}
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        ranked_indices = [
            int(index)
            for index in torch.argsort(scores_tensor, descending=True).detach().cpu().tolist()
            if int(index) != best_index
        ]
        candidate_indices = ranked_indices[:topk]
        if self._env_bool("SPIRECOMM_V3_COMBAT_DELAYED_DEATH_INCLUDE_POTIONS", False) or self._env_bool(
            "SPIRECOMM_V3_COMBAT_SURVIVAL_INCLUDE_POTIONS",
            False,
        ):
            seen_indices = set(candidate_indices)
            for index, action in enumerate(actions):
                if index == best_index or index in seen_indices:
                    continue
                if str(action.get("kind") or "") == "potion":
                    candidate_indices.append(index)
                    seen_indices.add(index)
        chosen_index: int | None = None
        chosen_outcome_rank = 99
        for index in candidate_indices:
            action = actions[index]
            if str(action.get("kind") or "") == "card":
                card_type = self._action_card_type(before_state, action)
                if card_type in excluded_card_types:
                    continue
            score = float(scores_tensor[index].detach().cpu().item())
            if str(action.get("kind") or "") != "potion" and margin_max >= 0.0 and best_score - score > margin_max:
                continue
            outcome = self._action_policy_terminal_outcome(
                env,
                before_state,
                actions[index],
                max_decisions=max_decisions,
            )
            if outcome == "win":
                chosen_index = index
                chosen_outcome_rank = 0
                break
            if outcome == "safe" and chosen_outcome_rank > 1:
                chosen_index = index
                chosen_outcome_rank = 1
            elif allow_unknown and outcome == "unknown" and chosen_outcome_rank > 2:
                chosen_index = index
                chosen_outcome_rank = 2
        if chosen_index is None:
            return scores_tensor
        if str(actions[chosen_index].get("kind") or "") == "potion":
            self.last_survival_potion_rescue_used = True
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_delayed_death_guard_used = True
        return adjusted

    def _suicidal_end_safe_non_end_indices(self, env: Any | None, actions: list[dict[str, Any]]) -> list[int]:
        if env is None:
            return []
        try:
            root_blob = clone_env_blob(env, strip_debug_history=True)
        except Exception:
            return []
        safe: list[int] = []
        for index, action in enumerate(actions):
            if str(action.get("kind") or "") == "end":
                continue
            try:
                branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
                branch_state = self._state_from_env(branch)
                if self._state_player_current_hp(branch_state) is not None and int(self._state_player_current_hp(branch_state) or 0) <= 0:
                    continue
                if self._is_post_combat_state(self._state_from_env(env), branch_state):
                    safe.append(index)
                    continue
                if str(getattr(branch, "phase", "")) != "COMBAT":
                    safe.append(index)
                    continue
                followup_actions = root_combat_actions(branch)
                if len(followup_actions) != 1 or str(followup_actions[0].get("kind") or "") != "end":
                    # More decisions remain; the normal policy/guards can still react.
                    safe.append(index)
                    continue
                branch.step(dict(followup_actions[0]))
                ended = branch
                ended_state = self._state_from_env(ended)
                ended_hp = self._state_player_current_hp(ended_state)
                if self._is_post_combat_state(branch_state, ended_state):
                    safe.append(index)
                elif ended_hp is not None and ended_hp > 0:
                    safe.append(index)
            except Exception:
                continue
        return safe

    def _apply_potion_over_end_gate(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
    ) -> Any:
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_POTION_OVER_END_MARGIN_MAX", -1.0)
        if margin_max < 0.0 or int(scores_tensor.numel()) <= 0:
            return scores_tensor
        if len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_POTION_OVER_END_ROOM_TYPES")
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        if str(actions[best_index].get("kind") or "") != "end":
            return scores_tensor
        best_potion_index: int | None = None
        best_potion_score: float | None = None
        for index, action in enumerate(actions):
            if str(action.get("kind") or "") != "potion":
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if best_potion_score is None or score > best_potion_score:
                best_potion_score = score
                best_potion_index = index
        if best_potion_index is None or best_potion_score is None:
            return scores_tensor
        end_score = float(scores_tensor[best_index].detach().cpu().item())
        if end_score - best_potion_score > margin_max:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[best_potion_index] = adjusted[best_index] + 1e-4
        self.last_potion_over_end_used = True
        return adjusted

    def _apply_block_card_over_end_gate(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_MARGIN_MAX", 0.03)
        if margin_max < 0.0 or int(scores_tensor.numel()) <= 0:
            return scores_tensor
        if len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_ROOM_TYPES") is None:
            room_types = {"MonsterRoomElite", "MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        if str(actions[best_index].get("kind") or "") != "end":
            return scores_tensor
        end_hp_ratio_max = self._env_float("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_END_HP_RATIO_MAX", 0.0)
        end_hp_max = self._env_int("SPIRECOMM_V3_COMBAT_BLOCK_OVER_END_END_HP_MAX", 0)
        if end_hp_ratio_max > 0.0 or end_hp_max > 0:
            end_after_state = after_states[best_index]
            end_hp = self._state_player_current_hp(end_after_state)
            if end_hp is None:
                return scores_tensor
            end_max_hp = self._state_player_max_hp(end_after_state)
            ratio_match = (
                end_max_hp is not None
                and end_max_hp > 0
                and end_hp_ratio_max > 0.0
                and (float(end_hp) / float(end_max_hp)) <= end_hp_ratio_max
            )
            flat_match = end_hp_max > 0 and int(end_hp) <= end_hp_max
            if not ratio_match and not flat_match:
                return scores_tensor
        before_block = self._state_player_block(before_state)
        if before_block is None:
            return scores_tensor
        best_block_index: int | None = None
        best_block_score: float | None = None
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if str(action.get("kind") or "") != "card":
                continue
            after_block = self._state_player_block(after_state)
            if after_block is None or after_block <= before_block:
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if best_block_score is None or score > best_block_score:
                best_block_score = score
                best_block_index = index
        if best_block_index is None or best_block_score is None:
            return scores_tensor
        end_score = float(scores_tensor[best_index].detach().cpu().item())
        if end_score - best_block_score > margin_max:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[best_block_index] = adjusted[best_index] + 1e-4
        self.last_block_over_end_used = True
        return adjusted

    def _apply_sharp_hide_post_action_danger_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_GUARD", False):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_ROOM_TYPES") is None:
            room_types = {"MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        sharp_hide = self._state_live_monster_power_amount(before_state, {"Sharp Hide"})
        if int(sharp_hide) <= 0:
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_action = actions[best_index]
        if str(best_action.get("kind") or "") != "card" or self._action_card_type(before_state, best_action) != "ATTACK":
            return scores_tensor
        best_after_state = after_states[best_index]
        if self._is_post_combat_state(before_state, best_after_state):
            return scores_tensor
        best_after_hp = self._state_player_current_hp(best_after_state)
        best_after_uncovered = self._state_uncovered_incoming(best_after_state)
        if best_after_hp is None or best_after_uncovered is None:
            return scores_tensor
        hp_buffer = self._env_int("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_HP_BUFFER", 0)
        if int(best_after_uncovered) + int(hp_buffer) < int(best_after_hp):
            return scores_tensor
        min_after_hp = self._env_int("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_MIN_AFTER_HP", 1)
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_MARGIN_MAX", 8.0)
        excluded_types = {value.upper() for value in self._env_set("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_EXCLUDE_CARD_TYPES")}
        if not excluded_types and os.environ.get("SPIRECOMM_V3_COMBAT_SHARP_HIDE_DANGER_EXCLUDE_CARD_TYPES") is None:
            excluded_types = {"ATTACK", "STATUS", "CURSE"}
        chosen_index: int | None = None
        chosen_score = float("-inf")
        chosen_safety = -10**9
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if index == best_index:
                continue
            if self._is_post_combat_state(before_state, after_state):
                chosen_index = index
                chosen_score = float(scores_tensor[index].detach().cpu().item())
                chosen_safety = 10**9
                break
            hp = self._state_player_current_hp(after_state)
            uncovered = self._state_uncovered_incoming(after_state)
            if hp is None or uncovered is None or int(hp) < min_after_hp:
                continue
            if int(uncovered) + int(hp_buffer) >= int(hp):
                continue
            kind = str(action.get("kind") or "")
            if kind == "card" and self._action_card_type(before_state, action) in excluded_types:
                continue
            if kind == "potion":
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if margin_max >= 0.0 and best_score - score > margin_max:
                continue
            safety = int(hp) - int(uncovered)
            if safety > chosen_safety or (safety == chosen_safety and score > chosen_score):
                chosen_index = index
                chosen_score = score
                chosen_safety = safety
        if chosen_index is None:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_sharp_hide_danger_guard_used = True
        return adjusted

    def _apply_lethal_card_over_setup_gate(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
        env: Any | None = None,
    ) -> Any:
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_MARGIN_MAX", 4.0)
        if margin_max < 0.0 or int(scores_tensor.numel()) <= 0:
            return scores_tensor
        if len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_ROOM_TYPES") is None:
            room_types = {"MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        player_hp = self._state_player_current_hp(before_state)
        uncovered = self._state_uncovered_incoming(before_state)
        if player_hp is None or uncovered is None or int(uncovered) < int(player_hp):
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_action = actions[best_index]
        if self._is_post_combat_state(before_state, after_states[best_index]):
            return scores_tensor
        top_kinds = self._env_set("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_TOP_KINDS")
        if not top_kinds and os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_TOP_KINDS") is None:
            top_kinds = {"card", "potion", "end"}
        best_kind = str(best_action.get("kind") or "")
        if best_kind not in top_kinds:
            return scores_tensor
        if best_kind == "card":
            if self._env_bool("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_SKIP_BLOCK_TOP", True):
                skip_block_allowed = True
                hp_max = self._env_int("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_SKIP_BLOCK_TOP_HP_MAX", 8)
                hp_ratio_max = self._env_float("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_SKIP_BLOCK_TOP_HP_RATIO_MAX", 0.0)
                if hp_max > 0 or hp_ratio_max > 0.0:
                    skip_block_allowed = False
                    max_hp = self._state_player_max_hp(before_state)
                    flat_match = hp_max > 0 and int(player_hp) <= hp_max
                    ratio_match = (
                        max_hp is not None
                        and max_hp > 0
                        and hp_ratio_max > 0.0
                        and (float(player_hp) / float(max_hp)) <= hp_ratio_max
                    )
                    if flat_match or ratio_match:
                        skip_block_allowed = True
                if skip_block_allowed:
                    before_block = self._state_player_block(before_state)
                    after_block = self._state_player_block(after_states[best_index])
                    if before_block is not None and after_block is not None and int(after_block) > int(before_block):
                        return scores_tensor
            top_types = {value.upper() for value in self._env_set("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_TOP_TYPES")}
            if not top_types and os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_TOP_TYPES") is None:
                top_types = {"POWER", "SKILL"}
            if self._action_card_type(before_state, best_action) not in top_types:
                return scores_tensor
            excluded_top_cards = self._env_set("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_EXCLUDE_TOP_CARDS")
            if excluded_top_cards:
                top_card = self._selected_card_from_state(before_state, best_action) or {}
                top_card_ids = {
                    str(best_action.get("card_id") or ""),
                    str(best_action.get("name") or ""),
                    str(top_card.get("card_id") or ""),
                    str(top_card.get("name") or ""),
                }
                if any(card_id in excluded_top_cards for card_id in top_card_ids if card_id):
                    return scores_tensor
        best_lethal_index: int | None = None
        best_lethal_score: float | None = None
        include_potions = self._env_bool("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_INCLUDE_POTIONS", False)
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            action_kind = str(action.get("kind") or "")
            if action_kind != "card" and not (include_potions and action_kind == "potion"):
                continue
            if not self._is_post_combat_state(before_state, after_state):
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if best_lethal_score is None or score > best_lethal_score:
                best_lethal_score = score
                best_lethal_index = index
        if best_lethal_index is None or best_lethal_score is None:
            return scores_tensor
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        if best_score - best_lethal_score > margin_max:
            return scores_tensor
        if self._should_preserve_sequence_before_lethal(
            env,
            before_state,
            best_action,
            after_states[best_index],
        ):
            self.last_lethal_sequence_preserve_used = True
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[best_lethal_index] = adjusted[best_index] + 1e-4
        self.last_lethal_card_over_setup_used = True
        return adjusted

    def _apply_setup_power_over_basic_attack_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_GUARD", True):
            return scores_tensor
        if int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_ROOM_TYPES") is None:
            room_types = {"MonsterRoomElite", "MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        floor_max = self._env_int("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_FLOOR_MAX", 16)
        if floor_max > 0:
            try:
                if int(before_state.get("floor", 0) or 0) > floor_max:
                    return scores_tensor
            except (TypeError, ValueError):
                return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        best_action = actions[best_index]
        if str(best_action.get("kind") or "") != "card":
            return scores_tensor
        top_cards = self._env_set("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_TOP_CARDS")
        if not top_cards and os.environ.get("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_TOP_CARDS") is None:
            top_cards = {"Strike_R"}
        top_card = self._selected_card_from_state(before_state, best_action) or {}
        top_ids = {
            str(best_action.get("card_id") or ""),
            str(best_action.get("name") or ""),
            str(top_card.get("card_id") or ""),
            str(top_card.get("name") or ""),
        }
        if top_cards and not any(card_id in top_cards for card_id in top_ids if card_id):
            return scores_tensor
        if self._is_post_combat_state(before_state, after_states[best_index]):
            return scores_tensor
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_MARGIN_MAX", 0.15)
        allowed_powers = self._env_set("SPIRECOMM_V3_COMBAT_SETUP_POWER_OVER_BASIC_ATTACK_POWER_CARDS")
        chosen_index: int | None = None
        chosen_score = float("-inf")
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if index == best_index or str(action.get("kind") or "") != "card":
                continue
            if self._action_card_type(before_state, action) != "POWER":
                continue
            card = self._selected_card_from_state(before_state, action) or {}
            if allowed_powers:
                ids = {
                    str(action.get("card_id") or ""),
                    str(action.get("name") or ""),
                    str(card.get("card_id") or ""),
                    str(card.get("name") or ""),
                }
                if not any(card_id in allowed_powers for card_id in ids if card_id):
                    continue
            if self._is_post_combat_state(before_state, after_state):
                continue
            hp = self._state_player_current_hp(after_state)
            if hp is not None and int(hp) <= 0:
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if margin_max >= 0.0 and best_score - score > margin_max:
                continue
            if score > chosen_score:
                chosen_index = index
                chosen_score = score
        if chosen_index is None:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_setup_power_over_basic_attack_used = True
        return adjusted

    def _apply_high_block_progress_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_GUARD", False):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_ROOM_TYPES") is None:
            room_types = {"MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        player_block = self._state_player_block(before_state)
        if player_block is None:
            return scores_tensor
        min_block = self._env_int("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MIN_BLOCK", 80)
        if int(player_block) < min_block:
            return scores_tensor
        try:
            incoming = int(incoming_damage(before_state))
        except Exception:
            return scores_tensor
        surplus_min = self._env_int("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_SURPLUS_MIN", 40)
        if int(player_block) - max(0, incoming) < surplus_min:
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        if self._is_post_combat_state(before_state, after_states[best_index]):
            return scores_tensor
        best_progress = self._monster_hp_progress(before_state, after_states[best_index])
        if best_progress is None or int(best_progress) > 0:
            return scores_tensor
        min_damage = self._env_int("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MIN_DAMAGE", 25)
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_HIGH_BLOCK_PROGRESS_MARGIN_MAX", 0.75)
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        chosen_index: int | None = None
        chosen_progress = 0
        chosen_score = float("-inf")
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if index == best_index or str(action.get("kind") or "") != "card":
                continue
            if self._action_card_type(before_state, action) != "ATTACK":
                continue
            if self._is_post_combat_state(before_state, after_state):
                continue
            hp = self._state_player_current_hp(after_state)
            if hp is not None and hp <= 0:
                continue
            progress = self._monster_hp_progress(before_state, after_state)
            if progress is None or int(progress) < min_damage:
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if margin_max >= 0.0 and best_score - score > margin_max:
                continue
            if int(progress) > chosen_progress or (int(progress) == chosen_progress and score > chosen_score):
                chosen_index = index
                chosen_progress = int(progress)
                chosen_score = score
        if chosen_index is None:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_high_block_progress_guard_used = True
        return adjusted

    def _apply_monster_block_progress_guard(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        after_states: list[dict[str, Any] | None],
    ) -> Any:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_GUARD", True):
            return scores_tensor
        if self._survival_policy_probe_depth > 0:
            return scores_tensor
        if int(scores_tensor.numel()) <= 1 or len(actions) != int(scores_tensor.numel()) or len(actions) != len(after_states):
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_ROOM_TYPES")
        if not room_types and os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_ROOM_TYPES") is None:
            room_types = {"MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss"}
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        monster_block = self._state_live_monster_block(before_state)
        if monster_block is None:
            return scores_tensor
        min_monster_block = self._env_int("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MIN_BLOCK", 25)
        if int(monster_block) < min_monster_block:
            self._monster_block_progress_signature = None
            self._monster_block_progress_stall_count = 0
            return scores_tensor
        stall_signature = self._monster_hp_stall_signature(before_state)
        if stall_signature is None:
            self._monster_block_progress_signature = None
            self._monster_block_progress_stall_count = 0
            return scores_tensor
        if stall_signature == self._monster_block_progress_signature:
            self._monster_block_progress_stall_count += 1
        else:
            self._monster_block_progress_signature = stall_signature
            self._monster_block_progress_stall_count = 0
        stall_count_min = self._env_int("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_STALL_COUNT_MIN", 20)
        if stall_count_min > 0 and self._monster_block_progress_stall_count < stall_count_min:
            return scores_tensor
        best_index = int(torch.argmax(scores_tensor).item())
        if self._is_post_combat_state(before_state, after_states[best_index]):
            return scores_tensor
        best_hp_progress = self._monster_hp_progress(before_state, after_states[best_index])
        best_block_progress = self._monster_block_progress(before_state, after_states[best_index])
        if best_hp_progress is None or best_block_progress is None:
            return scores_tensor
        if int(best_hp_progress) > 0 or int(best_block_progress) > 0:
            return scores_tensor
        best_action = actions[best_index]
        best_kind = str(best_action.get("kind") or "")
        if best_kind == "card":
            top_types_raw = os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_TOP_CARD_TYPES")
            allowed_top_types = self._env_set("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_TOP_CARD_TYPES")
            if not allowed_top_types and not str(top_types_raw or "").strip():
                allowed_top_types = {"SKILL"}
            if allowed_top_types and self._action_card_type(before_state, best_action) not in allowed_top_types:
                return scores_tensor
        elif best_kind != "end":
            return scores_tensor
        min_progress = self._env_int("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MIN_PROGRESS", 6)
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_MARGIN_MAX", 3.0)
        hp_weight = self._env_float("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_HP_WEIGHT", 4.0)
        best_score_max = self._env_float("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_BEST_SCORE_MAX", 0.0)
        excluded_raw = os.environ.get("SPIRECOMM_V3_COMBAT_MONSTER_BLOCK_PROGRESS_EXCLUDE_CARDS")
        if not str(excluded_raw or "").strip():
            excluded_raw = "Fiend Fire"
        excluded_cards = {
            token.strip()
            for token in str(excluded_raw).split(",")
            if token.strip()
        }
        best_score = float(scores_tensor[best_index].detach().cpu().item())
        if best_score_max >= 0.0 and best_score > best_score_max:
            return scores_tensor
        chosen_index: int | None = None
        chosen_progress_score = float("-inf")
        chosen_score = float("-inf")
        for index, (action, after_state) in enumerate(zip(actions, after_states, strict=False)):
            if index == best_index or str(action.get("kind") or "") != "card":
                continue
            if self._action_card_type(before_state, action) != "ATTACK":
                continue
            card = self._selected_card_from_state(before_state, action) or {}
            card_names = {
                str(action.get("card_id") or ""),
                str(action.get("name") or ""),
                str(card.get("card_id") or ""),
                str(card.get("name") or ""),
            }
            if excluded_cards and any(name in excluded_cards for name in card_names if name):
                continue
            if self._is_post_combat_state(before_state, after_state):
                continue
            hp = self._state_player_current_hp(after_state)
            if hp is not None and hp <= 0:
                continue
            hp_progress = self._monster_hp_progress(before_state, after_state)
            block_progress = self._monster_block_progress(before_state, after_state)
            if hp_progress is None or block_progress is None:
                continue
            progress_score = float(block_progress) + hp_weight * float(hp_progress)
            if progress_score < float(min_progress):
                continue
            score = float(scores_tensor[index].detach().cpu().item())
            if margin_max >= 0.0 and best_score - score > margin_max:
                continue
            if progress_score > chosen_progress_score or (
                progress_score == chosen_progress_score and score > chosen_score
            ):
                chosen_index = index
                chosen_progress_score = progress_score
                chosen_score = score
        if chosen_index is None:
            return scores_tensor
        adjusted = scores_tensor.clone()
        adjusted[chosen_index] = adjusted[best_index] + 1e-4
        self.last_monster_block_progress_guard_used = True
        return adjusted

    def _should_preserve_sequence_before_lethal(
        self,
        env: Any | None,
        before_state: dict[str, Any],
        top_action: dict[str, Any],
        top_after_state: dict[str, Any] | None,
    ) -> bool:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_PRESERVE_SEQUENCE", False):
            return False
        if env is None or str(top_action.get("kind") or "") != "card":
            return False
        if self._is_post_combat_state(before_state, top_after_state):
            return False
        preserve_types = {value.upper() for value in self._env_set("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_PRESERVE_TYPES")}
        if not preserve_types and os.environ.get("SPIRECOMM_V3_COMBAT_LETHAL_CARD_OVER_SETUP_PRESERVE_TYPES") is None:
            preserve_types = {"POWER", "SKILL"}
        before_block = self._state_player_block(before_state)
        after_block = self._state_player_block(top_after_state)
        block_gain = before_block is not None and after_block is not None and int(after_block) > int(before_block)
        if self._action_card_type(before_state, top_action) not in preserve_types and not block_gain:
            return False
        return self._action_allows_followup_card_lethal(env, before_state, top_action)

    def _action_allows_followup_card_lethal(
        self,
        env: Any,
        before_state: dict[str, Any],
        action: dict[str, Any],
    ) -> bool:
        try:
            root_blob = clone_env_blob(env, strip_debug_history=True)
            branch = step_branch_from_blob(root_blob, action, strip_debug_history=True)
            branch_state = self._state_from_env(branch)
            if self._is_post_combat_state(before_state, branch_state):
                return False
            hp = self._state_player_current_hp(branch_state)
            if hp is not None and hp <= 0:
                return False
            if str(getattr(branch, "phase", "")) != "COMBAT":
                return False
            followup_actions = root_combat_actions(branch)
            if not followup_actions:
                return False
            branch_blob = clone_env_blob(branch, strip_debug_history=True)
            for followup in followup_actions:
                if str(followup.get("kind") or "") != "card":
                    continue
                try:
                    ended = step_branch_from_blob(branch_blob, followup, strip_debug_history=True)
                    ended_state = self._state_from_env(ended)
                except Exception:
                    continue
                if self._is_post_combat_state(branch_state, ended_state):
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_set(name: str) -> set[str]:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return set()
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _quantize_decision_scores(self, scores_tensor: Any) -> Any:
        quantum = self._env_float("SPIRECOMM_V3_COMBAT_DECISION_SCORE_QUANTUM", 0.0)
        if quantum <= 0.0:
            return scores_tensor
        try:
            finite_mask = torch.isfinite(scores_tensor)
            quantized = torch.round(scores_tensor / float(quantum)) * float(quantum)
            return torch.where(finite_mask, quantized, scores_tensor)
        except Exception:
            return scores_tensor

    @staticmethod
    def _score_top_margin(scores_tensor: Any) -> float:
        if int(scores_tensor.numel()) < 2:
            return float("inf")
        top2 = torch.topk(scores_tensor, k=2).values
        return float((top2[0] - top2[1]).detach().cpu().item())

    def _rescue_gate_enabled_for_state(
        self,
        before_state: dict[str, Any],
        primary_scores: Any,
        rescue_scores: Any,
        actions: list[dict[str, Any]],
    ) -> bool:
        if int(primary_scores.numel()) <= 0 or int(rescue_scores.numel()) != int(primary_scores.numel()):
            return False
        margin_max = V3CandidateCombatSelector._env_float("SPIRECOMM_V3_COMBAT_RESCUE_MARGIN_MAX", -1.0)
        if margin_max < 0.0:
            return False
        primary_margin = V3CandidateCombatSelector._score_top_margin(primary_scores)
        if primary_margin > margin_max:
            return False
        min_rescue_margin = V3CandidateCombatSelector._env_float("SPIRECOMM_V3_COMBAT_RESCUE_MIN_RESCUE_MARGIN", 0.0)
        if min_rescue_margin > 0.0 and V3CandidateCombatSelector._score_top_margin(rescue_scores) < min_rescue_margin:
            return False
        room_types = V3CandidateCombatSelector._env_set("SPIRECOMM_V3_COMBAT_RESCUE_ROOM_TYPES")
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return False
        floor_min = V3CandidateCombatSelector._env_int("SPIRECOMM_V3_COMBAT_RESCUE_FLOOR_MIN", 0)
        if floor_min > 0:
            floor = V3CandidateCombatSelector._state_floor(before_state)
            if floor is None or floor < floor_min:
                return False
        floor_max = V3CandidateCombatSelector._env_int("SPIRECOMM_V3_COMBAT_RESCUE_FLOOR_MAX", 0)
        if floor_max > 0:
            floor = V3CandidateCombatSelector._state_floor(before_state)
            if floor is None or floor > floor_max:
                return False
        hp_ratio_max = V3CandidateCombatSelector._env_float("SPIRECOMM_V3_COMBAT_RESCUE_HP_RATIO_MAX", 0.0)
        if hp_ratio_max > 0.0:
            hp = V3CandidateCombatSelector._state_player_current_hp(before_state)
            max_hp = V3CandidateCombatSelector._state_player_max_hp(before_state)
            if hp is None or max_hp is None or max_hp <= 0 or (float(hp) / float(max_hp)) > hp_ratio_max:
                return False
        primary_top = int(torch.argmax(primary_scores).item())
        rescue_top = int(torch.argmax(rescue_scores).item())
        if str(os.environ.get("SPIRECOMM_V3_COMBAT_RESCUE_REQUIRE_DISAGREE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            if primary_top == rescue_top:
                return False
        primary_kinds = V3CandidateCombatSelector._env_set("SPIRECOMM_V3_COMBAT_RESCUE_PRIMARY_TOP_KINDS")
        if primary_kinds and str(actions[primary_top].get("kind") or "") not in primary_kinds:
            return False
        rescue_kinds = V3CandidateCombatSelector._env_set("SPIRECOMM_V3_COMBAT_RESCUE_TOP_KINDS")
        if rescue_kinds and str(actions[rescue_top].get("kind") or "") not in rescue_kinds:
            return False
        if V3CandidateCombatSelector._env_bool("SPIRECOMM_V3_COMBAT_RESCUE_REQUIRE_SUICIDAL_END_GUARD", False):
            if not bool(getattr(self, "last_suicidal_end_guard_used", False)):
                return False
        return True

    def _teacher_fallback_scores(
        self,
        env: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        scores_tensor: Any,
    ) -> Any | None:
        if not self._env_bool("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK", False):
            return None
        if int(scores_tensor.numel()) <= 0 or len(actions) != int(scores_tensor.numel()):
            return None
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_MARGIN_MAX", -1.0)
        if margin_max >= 0.0 and self._score_top_margin(scores_tensor) > margin_max:
            return None
        model_top_index = int(torch.argmax(scores_tensor).item())
        model_top_kinds = self._env_set("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_MODEL_TOP_KINDS")
        if model_top_kinds and str(actions[model_top_index].get("kind") or "") not in model_top_kinds:
            return None
        if self._env_bool("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_REQUIRE_POTION_CANDIDATE", False):
            if not any(str(action.get("kind") or "") == "potion" for action in actions):
                return None
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_ROOM_TYPES")
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return None
        if self._env_bool("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_REQUIRE_SUICIDAL_END_GUARD", False):
            if not bool(getattr(self, "last_suicidal_end_guard_used", False)):
                return None
        floor_min = self._env_int("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_FLOOR_MIN", 0)
        if floor_min > 0:
            floor = self._state_floor(before_state)
            if floor is None or floor < floor_min:
                return None
        floor_max = self._env_int("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_FLOOR_MAX", 0)
        if floor_max > 0:
            floor = self._state_floor(before_state)
            if floor is None or floor > floor_max:
                return None
        try:
            from spirecomm.ai.v3_combat_teacher import label_env, teacher_config_from_env

            labeled = label_env(
                env,
                root_id="runtime:teacher_fallback",
                source="runtime_teacher_fallback",
                config=teacher_config_from_env(),
                legal_actions=actions,
            )
        except Exception as exc:
            self.last_error = f"teacher_fallback_label_failed:{exc}"
            return None
        if labeled is None or not getattr(labeled, "candidates", None):
            return None
        by_key = {tuple(candidate.action_key): candidate for candidate in labeled.candidates}
        values: list[float] = []
        for action in actions:
            candidate = by_key.get(action_key(action, before_state))
            if candidate is None:
                return None
            values.append(float(candidate.teacher_q))
        if not values:
            return None
        teacher_top_index = max(range(len(values)), key=values.__getitem__)
        top_kinds = self._env_set("SPIRECOMM_V3_COMBAT_TEACHER_FALLBACK_TOP_KINDS")
        if top_kinds and str(actions[teacher_top_index].get("kind") or "") not in top_kinds:
            return None
        teacher_tensor = torch.tensor(values, dtype=scores_tensor.dtype, device=scores_tensor.device)
        blend_weight = self._env_float("SPIRECOMM_V3_COMBAT_TEACHER_BLEND_WEIGHT", 0.0)
        if blend_weight > 0.0:
            model_std = torch.std(scores_tensor.float(), unbiased=False).clamp_min(1.0e-6)
            teacher_std = torch.std(teacher_tensor.float(), unbiased=False).clamp_min(1.0e-6)
            model_z = (scores_tensor.float() - scores_tensor.float().mean()) / model_std
            teacher_z = (teacher_tensor.float() - teacher_tensor.float().mean()) / teacher_std
            self.last_teacher_blend_used = True
            return (model_z + float(blend_weight) * teacher_z).to(dtype=scores_tensor.dtype, device=scores_tensor.device)
        return teacher_tensor

    def _apply_branch_advisor_scores(
        self,
        scores_tensor: Any,
        before_state: dict[str, Any],
        actions: list[dict[str, Any]],
        candidate_features: list[list[float]],
    ) -> Any:
        advisor = self.branch_advisor_model
        weight = self._env_float("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_WEIGHT", 0.0)
        if advisor is None or weight <= 0.0 or int(scores_tensor.numel()) <= 1:
            return scores_tensor
        if len(candidate_features) != len(actions) or len(actions) != int(scores_tensor.numel()):
            return scores_tensor
        margin_max = self._env_float("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_MARGIN_MAX", -1.0)
        if margin_max >= 0.0 and self._score_top_margin(scores_tensor) > margin_max:
            return scores_tensor
        room_types = self._env_set("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_ROOM_TYPES")
        if room_types and str(before_state.get("room_type") or "") not in room_types:
            return scores_tensor
        model_top_index = int(torch.argmax(scores_tensor).item())
        model_top_kinds = self._env_set("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_MODEL_TOP_KINDS")
        if model_top_kinds and str(actions[model_top_index].get("kind") or "") not in model_top_kinds:
            return scores_tensor
        floor_min = self._env_int("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_FLOOR_MIN", 0)
        if floor_min > 0:
            floor = self._state_floor(before_state)
            if floor is None or floor < floor_min:
                return scores_tensor
        floor_max = self._env_int("SPIRECOMM_V3_COMBAT_BRANCH_ADVISOR_FLOOR_MAX", 0)
        if floor_max > 0:
            floor = self._state_floor(before_state)
            if floor is None or floor > floor_max:
                return scores_tensor
        try:
            features_tensor = torch.tensor(candidate_features, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                advisor_scores = advisor(features_tensor).to(device=scores_tensor.device, dtype=torch.float32)
        except Exception as exc:
            self.last_error = f"branch_advisor_score_failed:{exc}"
            return scores_tensor
        if int(advisor_scores.numel()) != int(scores_tensor.numel()):
            return scores_tensor
        advisor_std = torch.std(advisor_scores.float(), unbiased=False)
        if not bool(torch.isfinite(advisor_std).item()) or float(advisor_std.detach().cpu().item()) < 1.0e-6:
            return scores_tensor
        advisor_z = (advisor_scores.float() - advisor_scores.float().mean()) / advisor_std.clamp_min(1.0e-6)
        adjusted = scores_tensor.float() + float(weight) * advisor_z
        self.last_branch_advisor_used = True
        return adjusted.to(dtype=scores_tensor.dtype, device=scores_tensor.device)

    @staticmethod
    def _state_from_env(env: Any) -> dict[str, Any]:
        fast_state = V3CandidateCombatSelector._fast_v3_combat_state_from_env(env)
        if fast_state is not None:
            return fast_state
        state_method = getattr(env, "state", None)
        state = state_method() if callable(state_method) else env.serialize()
        if isinstance(state, dict):
            state.pop("rng_trace", None)
            state.pop("commands", None)
        return state

    @staticmethod
    def _fast_v3_combat_state_from_env(env: Any) -> dict[str, Any] | None:
        combat_env = V3CandidateCombatSelector._combat_env_from_root(env)
        if combat_env is None:
            return None
        try:
            return serialize_v3_combat_state(combat_env, include_debug_trace=False, include_commands=False)
        except Exception:
            return None

    @staticmethod
    def _combat_env_from_root(env: Any) -> Any | None:
        phase = str(getattr(env, "phase", "COMBAT") or "COMBAT")
        if getattr(env, "sim_backend", None) == "v3" and getattr(env, "combat", None) is not None:
            if phase in {"COMBAT", "CARD_SELECT", "CARD_REWARD"}:
                return getattr(env, "combat", None)
        if getattr(env, "sim_backend", None) == "v3" and getattr(env, "engine", None) is not None:
            return env
        return None

    def choose_env(
        self,
        env: Any,
        *,
        return_scores: bool = True,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, list[float]]:
        self.last_error = None
        self.last_rescue_used = False
        self.last_rescue_primary_margin = None
        self.last_rescue_rescue_margin = None
        self.last_rescue_primary_action = None
        self.last_rescue_action = None
        self.last_suicidal_end_guard_used = False
        self.last_dangerous_end_bias_used = False
        self.last_potion_over_end_used = False
        self.last_block_over_end_used = False
        self.last_sharp_hide_danger_guard_used = False
        self.last_lethal_card_over_setup_used = False
        self.last_lethal_sequence_preserve_used = False
        self.last_setup_power_over_basic_attack_used = False
        self.last_high_block_progress_guard_used = False
        self.last_monster_block_progress_guard_used = False
        self.last_danger_block_progress_guard_used = False
        self.last_gremlin_nob_skill_bias_used = False
        self.last_short_win_guard_used = False
        self.last_teacher_fallback_used = False
        self.last_teacher_blend_used = False
        self.last_branch_advisor_used = False
        self.last_suicidal_action_guard_used = False
        self.last_forced_turn_survival_guard_used = False
        self.last_policy_survival_guard_used = False
        self.last_post_forced_turn_survival_guard_used = False
        self.last_post_action_survival_guard_used = False
        self.last_delayed_death_guard_used = False
        self.last_survival_potion_rescue_used = False
        self.last_pre_guard_scores = []
        self.last_final_scores = []
        self.last_pre_guard_top_index = None
        self.last_final_top_index = None
        self.last_pre_guard_top_action = None
        self.last_final_top_action = None
        self.last_guard_names = []
        self.last_root_actions = []
        self.last_before_state = None
        if not self.available:
            self.last_error = "v3_candidate_checkpoint_unavailable"
            return None, []
        try:
            require_torch()
            actions = root_combat_actions(env, legal_actions=legal_actions)
            if not actions:
                self.last_error = "no_root_combat_actions"
                return None, []
            self.last_root_actions = [dict(action) for action in actions]
            before_state = self._state_from_env(env)
            self.last_before_state = before_state
            use_transformer = str(getattr(self.model, "model_kind", self.model_kind)) == "transformer"
            use_root_transformer = use_transformer and bool(getattr(self.model, "expects_root_batch", False))
            feature_rows: list[list[float]] = []
            transformer_records: list[dict[str, Any]] = []
            root_feature_rows: list[list[float]] = []
            advisor_feature_rows: list[list[float]] = []
            after_states: list[dict[str, Any] | None] = []
            transformer_batch: dict[str, Any] | None = None
            before_summary = encode_state_summary(before_state)
            combat_env = self._combat_env_from_root(env)
            slim_combat_clone = self._env_bool("SPIRECOMM_V3_COMBAT_SELECTOR_SLIM_COMBAT_CLONE", False)
            drop_unsimulatable = self._env_bool("SPIRECOMM_V3_COMBAT_SELECTOR_DROP_UNSIMULATABLE_ACTIONS", False)
            combat_env_blob = (
                clone_env_blob(
                    combat_env,
                    strip_debug_history=True,
                    teacher_branch_slim=slim_combat_clone,
                )
                if combat_env is not None
                else None
            )
            env_blob: bytes | None = None
            filtered_actions: list[dict[str, Any]] = []
            dropped_actions: list[str] = []
            for action in actions:
                after_state = None
                try:
                    if combat_env_blob is not None:
                        combat_branch = step_branch_from_blob(combat_env_blob, action, strip_debug_history=True)
                        outcome = str(getattr(combat_branch, "outcome", "") or "")
                        if not outcome or outcome == "UNDECIDED":
                            after_state = self._state_from_env(combat_branch)
                    if after_state is None:
                        if env_blob is None:
                            env_blob = clone_env_blob(env, strip_debug_history=True)
                        branch = step_branch_from_blob(env_blob, action, strip_debug_history=True)
                        after_state = self._state_from_env(branch)
                except Exception as exc:
                    if not drop_unsimulatable:
                        raise
                    dropped_actions.append(
                        "{}:{}:{}".format(
                            action.get("kind"),
                            action.get("card_id") or action.get("potion_id") or action.get("name"),
                            exc,
                        )
                    )
                    continue
                filtered_actions.append(action)
                after_states.append(after_state)
                features = encode_candidate_with_before_summary(before_state, before_summary, action, after_state)
                if use_root_transformer:
                    root_feature_rows.append(features)
                elif use_transformer:
                    transformer_records.append({})
                else:
                    feature_rows.append(features)
                advisor_feature_rows.append(features)
            actions = filtered_actions
            self.last_root_actions = [dict(action) for action in actions]
            if dropped_actions:
                self.last_error = "dropped_unsimulatable_actions:" + ";".join(dropped_actions[:8])
            if not actions:
                self.last_error = self.last_error or "no_simulatable_root_combat_actions"
                return None, []
            if return_scores and not action_keys_are_unique(actions, before_state):
                self.last_error = "ambiguous_action_key"
                return None, []
            if use_transformer and not use_root_transformer:
                transformer_batch = collate_transformer_candidates_shared_before(
                    before_state,
                    actions,
                    after_states,
                    candidate_features=advisor_feature_rows,
                    entity_index=self.transformer_entity_index,
                    spec=self.transformer_token_spec,
                    device=self.device,
                )
            with torch.inference_mode():
                rescue_scores_tensor = None
                if use_root_transformer:
                    root_record = encode_root_transformer_actions(
                        before_state,
                        actions,
                        candidate_features=root_feature_rows,
                        entity_index=self.transformer_entity_index,
                        spec=self.transformer_token_spec,
                        trim_padding=True,
                    )
                    batch = collate_root_transformer_records([root_record], device=self.device)
                    scores_tensor = self.model(batch)
                    if self.ensemble_models:
                        scores_tensor = self._average_ensemble_scores(scores_tensor, [model(batch) for model in self.ensemble_models])
                    if self.rescue_model is not None:
                        rescue_scores_tensor = self.rescue_model(batch)
                elif use_transformer:
                    batch = transformer_batch if transformer_batch is not None else collate_transformer_records(transformer_records, device=self.device)
                    scores_tensor = self.model(batch)
                    if self.ensemble_models:
                        scores_tensor = self._average_ensemble_scores(scores_tensor, [model(batch) for model in self.ensemble_models])
                    if self.rescue_model is not None:
                        rescue_scores_tensor = self.rescue_model(batch)
                else:
                    features_tensor = torch.tensor(feature_rows, dtype=torch.float32, device=self.device)
                    scores_tensor = self.model(features_tensor)
                    if self.ensemble_models:
                        scores_tensor = self._average_ensemble_scores(scores_tensor, [model(features_tensor) for model in self.ensemble_models])
                    if self.rescue_model is not None:
                        rescue_scores_tensor = self.rescue_model(features_tensor)
            scores_tensor = self._apply_runtime_score_adjustments(scores_tensor, before_state, actions)
            scores_tensor = self._apply_gremlin_nob_skill_bias(scores_tensor, before_state, actions)
            scores_tensor = self._quantize_decision_scores(scores_tensor)
            if int(scores_tensor.numel()) > 0:
                pre_guard_top = int(torch.argmax(scores_tensor).item())
                self.last_pre_guard_top_index = pre_guard_top
                self.last_pre_guard_top_action = dict(actions[pre_guard_top])
                self.last_pre_guard_scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
            scores_tensor = self._apply_suicidal_end_guard(scores_tensor, actions, after_states, env=env)
            scores_tensor = self._apply_suicidal_action_guard(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_forced_turn_survival_guard(scores_tensor, before_state, actions, env)
            scores_tensor = self._apply_policy_survival_guard(scores_tensor, before_state, actions, env)
            scores_tensor = self._apply_post_forced_turn_survival_guard(scores_tensor, before_state, actions, after_states, env)
            scores_tensor = self._apply_post_action_survival_guard(scores_tensor, before_state, actions, after_states, env)
            scores_tensor = self._apply_danger_block_progress_guard(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_delayed_death_guard(scores_tensor, before_state, actions, env)
            scores_tensor = self._apply_potion_over_end_gate(scores_tensor, before_state, actions)
            scores_tensor = self._apply_block_card_over_end_gate(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_sharp_hide_post_action_danger_guard(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_lethal_card_over_setup_gate(scores_tensor, before_state, actions, after_states, env)
            scores_tensor = self._apply_setup_power_over_basic_attack_guard(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_high_block_progress_guard(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_monster_block_progress_guard(scores_tensor, before_state, actions, after_states)
            scores_tensor = self._apply_short_win_guard(scores_tensor, before_state, actions, env)
            if rescue_scores_tensor is not None:
                rescue_scores_tensor = self._apply_runtime_score_adjustments(rescue_scores_tensor, before_state, actions)
                rescue_scores_tensor = self._apply_gremlin_nob_skill_bias(rescue_scores_tensor, before_state, actions)
                rescue_scores_tensor = self._quantize_decision_scores(rescue_scores_tensor)
                rescue_scores_tensor = self._apply_suicidal_end_guard(rescue_scores_tensor, actions, after_states, env=env)
                rescue_scores_tensor = self._apply_suicidal_action_guard(rescue_scores_tensor, before_state, actions, after_states)
                rescue_scores_tensor = self._apply_forced_turn_survival_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    env,
                )
                rescue_scores_tensor = self._apply_policy_survival_guard(rescue_scores_tensor, before_state, actions, env)
                rescue_scores_tensor = self._apply_post_forced_turn_survival_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                    env,
                )
                rescue_scores_tensor = self._apply_post_action_survival_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                    env,
                )
                rescue_scores_tensor = self._apply_danger_block_progress_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                )
                rescue_scores_tensor = self._apply_delayed_death_guard(rescue_scores_tensor, before_state, actions, env)
                rescue_scores_tensor = self._apply_potion_over_end_gate(rescue_scores_tensor, before_state, actions)
                rescue_scores_tensor = self._apply_block_card_over_end_gate(rescue_scores_tensor, before_state, actions, after_states)
                rescue_scores_tensor = self._apply_sharp_hide_post_action_danger_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                )
                rescue_scores_tensor = self._apply_lethal_card_over_setup_gate(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                    env,
                )
                rescue_scores_tensor = self._apply_setup_power_over_basic_attack_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                )
                rescue_scores_tensor = self._apply_high_block_progress_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                )
                rescue_scores_tensor = self._apply_monster_block_progress_guard(
                    rescue_scores_tensor,
                    before_state,
                    actions,
                    after_states,
                )
                rescue_scores_tensor = self._apply_short_win_guard(rescue_scores_tensor, before_state, actions, env)
                self.last_rescue_primary_margin = self._score_top_margin(scores_tensor)
                self.last_rescue_rescue_margin = self._score_top_margin(rescue_scores_tensor)
                self.last_rescue_primary_action = actions[int(torch.argmax(scores_tensor).item())]
                self.last_rescue_action = actions[int(torch.argmax(rescue_scores_tensor).item())]
                if self._rescue_gate_enabled_for_state(before_state, scores_tensor, rescue_scores_tensor, actions):
                    scores_tensor = rescue_scores_tensor
                    self.last_rescue_used = True
            scores_tensor = self._apply_branch_advisor_scores(scores_tensor, before_state, actions, advisor_feature_rows)
            scores_tensor = self._quantize_decision_scores(scores_tensor)
            teacher_fallback_scores = self._teacher_fallback_scores(env, before_state, actions, scores_tensor)
            if teacher_fallback_scores is not None:
                scores_tensor = teacher_fallback_scores
                scores_tensor = self._apply_gremlin_nob_skill_bias(scores_tensor, before_state, actions)
                scores_tensor = self._quantize_decision_scores(scores_tensor)
                scores_tensor = self._apply_suicidal_end_guard(scores_tensor, actions, after_states, env=env)
                scores_tensor = self._apply_suicidal_action_guard(scores_tensor, before_state, actions, after_states)
                scores_tensor = self._apply_forced_turn_survival_guard(scores_tensor, before_state, actions, env)
                scores_tensor = self._apply_policy_survival_guard(scores_tensor, before_state, actions, env)
                scores_tensor = self._apply_post_forced_turn_survival_guard(scores_tensor, before_state, actions, after_states, env)
                scores_tensor = self._apply_post_action_survival_guard(scores_tensor, before_state, actions, after_states, env)
                scores_tensor = self._apply_danger_block_progress_guard(scores_tensor, before_state, actions, after_states)
                scores_tensor = self._apply_delayed_death_guard(scores_tensor, before_state, actions, env)
                scores_tensor = self._apply_sharp_hide_post_action_danger_guard(scores_tensor, before_state, actions, after_states)
                scores_tensor = self._apply_lethal_card_over_setup_gate(scores_tensor, before_state, actions, after_states, env)
                scores_tensor = self._apply_setup_power_over_basic_attack_guard(scores_tensor, before_state, actions, after_states)
                scores_tensor = self._apply_high_block_progress_guard(scores_tensor, before_state, actions, after_states)
                scores_tensor = self._apply_monster_block_progress_guard(scores_tensor, before_state, actions, after_states)
                scores_tensor = self._apply_short_win_guard(scores_tensor, before_state, actions, env)
                self.last_teacher_fallback_used = True
            if int(scores_tensor.numel()) <= 0:
                self.last_error = "empty_scores"
                return None, []
            best_index = int(torch.argmax(scores_tensor).item())
            self.last_final_top_index = best_index
            self.last_final_top_action = dict(actions[best_index])
            self.last_final_scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
            guard_names: list[str] = []
            for attr_name, guard_name in (
                ("last_suicidal_end_guard_used", "suicidal_end_guard"),
                ("last_suicidal_action_guard_used", "suicidal_action_guard"),
                ("last_forced_turn_survival_guard_used", "forced_turn_survival_guard"),
                ("last_policy_survival_guard_used", "policy_survival_guard"),
                ("last_post_forced_turn_survival_guard_used", "post_forced_turn_survival_guard"),
                ("last_post_action_survival_guard_used", "post_action_survival_guard"),
                ("last_delayed_death_guard_used", "delayed_death_guard"),
                ("last_potion_over_end_used", "potion_over_end"),
                ("last_block_over_end_used", "block_over_end"),
                ("last_sharp_hide_danger_guard_used", "sharp_hide_danger_guard"),
                ("last_lethal_card_over_setup_used", "lethal_card_over_setup"),
                ("last_setup_power_over_basic_attack_used", "setup_power_over_basic_attack"),
                ("last_high_block_progress_guard_used", "high_block_progress_guard"),
                ("last_monster_block_progress_guard_used", "monster_block_progress_guard"),
                ("last_danger_block_progress_guard_used", "danger_block_progress_guard"),
                ("last_short_win_guard_used", "short_win_guard"),
                ("last_rescue_used", "rescue"),
                ("last_branch_advisor_used", "branch_advisor"),
                ("last_teacher_blend_used", "teacher_blend"),
                ("last_teacher_fallback_used", "teacher_fallback"),
            ):
                if bool(getattr(self, attr_name, False)):
                    guard_names.append(guard_name)
            self.last_guard_names = guard_names
            scores = [float(value) for value in scores_tensor.detach().cpu().tolist()] if return_scores else []
            return actions[best_index], scores
        except Exception as exc:
            self.last_error = f"v3_candidate_scoring_failed:{exc}"
            return None, []

    def _average_ensemble_scores(self, primary_scores: Any, extra_scores: list[Any]) -> Any:
        weights = self.ensemble_weights
        if len(weights) != 1 + len(extra_scores):
            weights = [1.0] * (1 + len(extra_scores))
        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            return primary_scores
        combined = primary_scores * float(weights[0])
        for weight, scores in zip(weights[1:], extra_scores, strict=False):
            combined = combined + scores.to(device=primary_scores.device, dtype=primary_scores.dtype) * float(weight)
        return combined / total_weight

    def choose(self, _serialized_state: dict[str, Any], _legal_actions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[float]]:
        self.last_error = "v3_candidate_selector_requires_env"
        return None, []


class V3TeacherCombatSelector:
    """Choose the highest-ranked action from the v3 combat teacher directly."""

    handles_potions = True

    def __init__(self, config: Any | None = None) -> None:
        if config is None:
            from spirecomm.ai.v3_combat_teacher import teacher_config_from_env

            config = teacher_config_from_env()
        self.config = config
        self.last_error: str | None = None
        self._fast_actions_fn: Any | None = None
        self._best_teacher_action_env_fn: Any | None = None
        self._label_env_fn: Any | None = None
        self._cache_teacher_functions()

    def _cache_teacher_functions(self) -> None:
        try:
            from spirecomm.ai.v3_combat_teacher import (
                _fast_combat_teacher_actions,
                best_teacher_action_env,
                label_env,
            )
        except Exception as exc:
            self.last_error = f"teacher_function_cache_failed:{exc}"
            return
        self._fast_actions_fn = _fast_combat_teacher_actions
        self._best_teacher_action_env_fn = best_teacher_action_env
        self._label_env_fn = label_env

    @property
    def available(self) -> bool:
        return True

    def legal_actions_env(self, env: Any) -> list[dict[str, Any]]:
        try:
            phase = str(getattr(env, "phase", "COMBAT") or "COMBAT")
            fast_actions_fn = self._fast_actions_fn
            if fast_actions_fn is None:
                from spirecomm.ai.v3_combat_teacher import _fast_combat_teacher_actions

                fast_actions_fn = _fast_combat_teacher_actions
                self._fast_actions_fn = fast_actions_fn
            actions = fast_actions_fn(env, phase)
            if actions is not None:
                return actions
        except Exception:
            pass
        return env.legal_actions()

    def choose_env(
        self,
        env: Any,
        *,
        return_scores: bool = True,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, list[float]]:
        self.last_error = None
        try:
            if not return_scores:
                best_teacher_action_env = self._best_teacher_action_env_fn
                if best_teacher_action_env is None:
                    from spirecomm.ai.v3_combat_teacher import best_teacher_action_env as imported_best_teacher_action_env

                    best_teacher_action_env = imported_best_teacher_action_env
                    self._best_teacher_action_env_fn = best_teacher_action_env
                best = best_teacher_action_env(env, config=self.config, legal_actions=legal_actions)
                if best is None:
                    self.last_error = "teacher_label_unavailable"
                    return None, []
                return best, []

            label_env = self._label_env_fn
            if label_env is None:
                from spirecomm.ai.v3_combat_teacher import label_env as imported_label_env

                label_env = imported_label_env
                self._label_env_fn = label_env
            labeled = label_env(
                env,
                root_id="runtime_teacher",
                source="runtime_teacher",
                config=self.config,
                legal_actions=legal_actions,
                validate_action_keys=False,
            )
            if labeled is None or not labeled.candidates:
                self.last_error = "teacher_label_unavailable"
                return None, []
            best = min(labeled.candidates, key=lambda candidate: candidate.teacher_rank)
            scores = [float(candidate.teacher_q) for candidate in labeled.candidates] if return_scores else []
            return dict(best.action), scores
        except Exception as exc:
            self.last_error = f"teacher_scoring_failed:{exc}"
            return None, []

    def choose(self, _serialized_state: dict[str, Any], _legal_actions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[float]]:
        self.last_error = "v3_teacher_selector_requires_env"
        return None, []
