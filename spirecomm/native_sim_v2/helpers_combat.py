from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from spirecomm.native_sim_v2.helpers_cards import CARD_LIBRARY, card_to_spirecomm, clone_card, ironclad_card_pool, ironclad_type_rarity_card_pool, make_card, roll_colorless_card
from spirecomm.native_sim.potions import get_random_potion, make_potion, potions_to_spirecomm, roll_potion
from spirecomm.native_sim.randoms import java_collections_shuffle
from spirecomm.native_sim_v2.helpers_relics import draw_relic_from_pool, make_relic
from spirecomm.native_sim.schema import CardInstance, MonsterState, PlayerState, PotionInstance
from spirecomm.native_sim_v2.helpers_common import *
from spirecomm.native_sim_v2.monster_support import _set_move
from spirecomm.native_sim_v2.monsters import choose_next_move, make_monster


class CombatHelpersMixin:
    def _card_is_ethereal(self, card: CardInstance) -> bool:
            if not card.card_def.ethereal:
                return False
            if card.card_id == "Apparition" and card.upgrades > 0:
                return False
            return True

    def _strike_dummy_bonus(self, card: CardInstance) -> int:
            if not self._has_relic("Strike Dummy"):
                return 0
            if card.card_id in STRIKE_CARD_IDS:
                return 3
            return 0

    def _card_is_opening_innate(self, card: CardInstance) -> bool:
            if card.card_id in {"AscendersBane", "Writhe", "Dramatic Entrance", "Mind Blast"}:
                return True
            # Some cards gain Innate only after being upgraded. Model the
            # battle-start opening hand based on the current combat card state,
            # not just the base card id.
            if card.card_id == "Brutality" and card.upgrades > 0:
                return True
            return False

    def _discard_end_turn_drawn_cards(self, hand_len_before: int) -> None:
            if len(self.hand) <= hand_len_before:
                return
            original_end_turn_drawn = list(self.hand[hand_len_before:])
            del self.hand[hand_len_before:]
            drawn_cards: list[CardInstance] = []
            for card in reversed(original_end_turn_drawn):
                if self._card_is_ethereal(card):
                    self._exhaust_card(card)
                    continue
                drawn_cards.append(card)
            insert_index = max(0, len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0))
            self.discard_pile[insert_index:insert_index] = drawn_cards

    def _draw_player_damage_relic_cards(self, count: int, *, end_turn_self_damage: bool) -> None:
            hand_len_before = len(self.hand)
            draw_kwargs: dict[str, Any] = {
                "exclude_end_turn_regular_discards": end_turn_self_damage,
            }
            if end_turn_self_damage:
                draw_kwargs["suppress_status_draw_triggers"] = True
                draw_kwargs["allow_fire_breathing_when_suppressed"] = True
                draw_kwargs["defer_status_draws_to_start_turn"] = True
                draw_kwargs["deferred_status_damage_all"] = self.pending_start_turn_fire_breathing_damage
            self.draw_cards(count, **draw_kwargs)
            if end_turn_self_damage:
                self._discard_end_turn_drawn_cards(hand_len_before)

    def _add_random_card_to_hand(self, *, card_type: str | None = None, colorless: bool = False, cost_for_turn: int | None = None) -> None:
            self._add_to_hand(self._random_card_id(card_type=card_type, colorless=colorless), cost_for_turn=cost_for_turn)

    def _add_to_discard(self, card_id: str, upgrades: int = 0) -> None:
            self.discard_pile.append(make_card(card_id, upgrades=upgrades, uuid=self._new_uuid(card_id)))

    def _add_to_hand(
            self,
            card_id: str,
            upgrades: int = 0,
            cost_for_turn: int | None = None,
            cost_for_combat: int | None = None,
        ) -> None:
            card = make_card(card_id, upgrades=upgrades, uuid=self._new_uuid(card_id))
            card.cost_for_combat = cost_for_combat
            card.cost_for_turn = cost_for_turn
            if (
                self.player.power("Corruption") > 0
                and card.card_def.card_type == "SKILL"
                and card.cost_for_combat is None
                and card.cost > 0
            ):
                card.cost_for_combat = 0
                card.cost_for_turn = -9
            if len(self.hand) >= 10:
                self.discard_pile.append(card)
            else:
                self.hand.append(card)

    def _advance_relic_counter(self, relic_id: str, threshold: int) -> bool:
            relic = self._relic(relic_id)
            if relic is None:
                return False
            counter = int(relic.get("counter", -1))
            if counter < 0:
                counter = 0
            counter += 1
            if counter >= threshold:
                relic["counter"] = 0
                return True
            relic["counter"] = counter
            return False

    def _alive_monsters(self) -> list[MonsterState]:
            return [monster for monster in self.monsters if monster.alive]

    def _defeat_reward_triggers(self, monster: MonsterState | None) -> bool:
            if monster is None or monster.alive or monster.half_dead:
                return False
            # Gremlin Leader's adds are minions in lightspeed, so Feed/Hand of
            # Greed/Ritual Dagger rewards should not fire when they die.
            if bool(getattr(monster, "ai_state", {}).get("leader_minion", 0)):
                return False
            if (
                monster.monster_id == "Darkling"
                and any(
                    ally is not monster
                    and ally.monster_id == "Darkling"
                    and ally.alive
                    for ally in self.monsters
                )
            ):
                return False
            return True

    def _apply_ascension_monster_scaling(self, *, elite: bool) -> None:
            is_boss = self.floor in {16, 33, 50, 54}
            is_act4_elite = self.floor == 53
            hp_scale = 1.0
            damage_bonus = 0
            strength_bonus = 0
            if self.ascension_level >= 2 and not elite and not is_boss and not is_act4_elite:
                hp_scale += 0.05
                damage_bonus += 1
            if self.ascension_level >= 3 and (elite or is_act4_elite):
                hp_scale += 0.08
                damage_bonus += 1
            if self.ascension_level >= 4 and is_boss:
                hp_scale += 0.08
                damage_bonus += 1
            if self.ascension_level >= 17 and not elite and not is_boss and not is_act4_elite:
                strength_bonus += 1
            if self.ascension_level >= 18 and (elite or is_act4_elite):
                strength_bonus += 1
            if self.ascension_level >= 19 and is_boss:
                strength_bonus += 1
            for monster in self.monsters:
                if monster.monster_id == "INVALID = 0":
                    continue
                if hp_scale != 1.0:
                    old_max = monster.max_hp
                    monster.max_hp = max(1, int(monster.max_hp * hp_scale))
                    monster.current_hp = max(1, int(monster.current_hp * monster.max_hp / max(1, old_max)))
                if damage_bonus and monster.move_base_damage > 0:
                    monster.move_base_damage += damage_bonus
                if strength_bonus:
                    monster.add_power("Strength", strength_bonus)

    def _apply_damage_to_monster(self, amount: int, monster: MonsterState) -> int:
            amount = max(0, int(amount))
            if amount <= 0:
                return 0
            if monster.monster_id == "CorruptHeart":
                cap = 300
                already = self.monster_damage_this_turn.get(id(monster), 0)
                remaining = max(0, cap - already)
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + min(hp_damage, remaining)
            if monster.power("Intangible") > 0:
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + (1 if hp_damage > 0 else 0)
            dealt = _apply_damage(amount, monster)
            self._wake_lagavulin_if_damaged(monster, dealt)
            if monster.monster_id == "CorruptHeart" and dealt > 0:
                self.monster_damage_this_turn[id(monster)] = self.monster_damage_this_turn.get(id(monster), 0) + dealt
            return dealt

    def _wake_lagavulin_if_damaged(self, monster: MonsterState | None, dealt: int) -> None:
            if monster is None or dealt <= 0:
                return
            if monster.monster_id != "Lagavulin" or not monster.ai_state.get("asleep", 0):
                return
            monster.ai_state["asleep"] = 0
            monster.ai_state["latent_awake"] = 0
            monster.ai_state["awoken"] = 1
            monster.move = "LAGAVULIN_STUN"
            monster.intent = "STUN"
            monster.move_base_damage = 0
            monster.move_hits = 0
            if monster.power("Metallicize") > 0:
                monster.add_power("Metallicize", -8)
                if monster.power("Metallicize") <= 0:
                    monster.powers.pop("Metallicize", None)

    def _apply_guardian_mode_shift(self, monster: MonsterState, dealt: int) -> None:
            if monster.monster_id != "TheGuardian" or dealt <= 0 or monster.power("Mode Shift") <= 0:
                return
            monster.add_power("Mode Shift", -dealt)
            if monster.power("Mode Shift") > 0:
                return
            monster.powers.pop("Mode Shift", None)
            if getattr(self, "processing_monster_turns", False):
                monster.ai_state["guardian_shifted_this_round"] = 1
            monster.move = "THE_GUARDIAN_DEFENSIVE_MODE"
            monster.intent = "BUFF"
            monster.move_base_damage = 0
            monster.move_hits = 0
            self.pending_monster_block_gains.append((monster, 20))

    def _apply_happy_flower_start_of_turn(self) -> None:
            relic = self._relic("Happy Flower")
            if relic is None:
                return
            counter = int(relic.get("counter", 0))
            if counter < 0:
                counter = 0
            counter += 1
            if counter >= 3:
                self.player.energy += 1
                counter = 0
            relic["counter"] = counter

    def _apply_monster_end_of_turn_triggers(self, monster: MonsterState) -> None:
            if not monster.alive:
                return
            if monster.power("Metallicize") > 0:
                monster.block += monster.power("Metallicize")
            if monster.power("Malleable") > 0:
                monster.powers["Malleable"] = 3
            if monster.power("Plated Armor") > 0:
                monster.block += monster.power("Plated Armor")
            if monster.power("Intangible") > 0:
                monster.add_power("Intangible", -1)
                if monster.power("Intangible") <= 0:
                    monster.powers.pop("Intangible", None)
            if monster.power("Regenerate") > 0:
                monster.current_hp = min(monster.max_hp, monster.current_hp + monster.power("Regenerate"))

    def _apply_monster_power(self, monster: MonsterState | None, power_id: str, amount: int) -> bool:
            if monster is None or amount == 0:
                return False
            is_debuff = self._is_debuff_power(power_id, amount)
            if is_debuff and monster.power("Artifact") > 0:
                monster.add_power("Artifact", -1)
                return False
            monster.add_power(power_id, amount)
            if power_id == "Vulnerable" and amount > 0 and self._has_relic("Champion Belt"):
                self._apply_monster_power(monster, "Weakened", 1)
            if is_debuff and self.player.power("Sadistic Nature") > 0 and monster.alive:
                self.pending_monster_direct_damage.append((monster, self.player.power("Sadistic Nature")))
            return True

    def _apply_monster_prebattle_actions(self) -> None:
            for monster in self.monsters:
                pending = monster.ai_state.pop("pending_prebattle", None)
                if pending == "LOUSE":
                    if self.ascension_level >= 17:
                        monster.add_power("Curl Up", self.monster_hp_rng.randint(9, 12))
                    elif self.ascension_level >= 7:
                        monster.add_power("Curl Up", self.monster_hp_rng.randint(4, 8))
                    else:
                        monster.add_power("Curl Up", self.monster_hp_rng.randint(3, 7))
                if monster.monster_id == "BronzeAutomaton":
                    monster.add_power("Artifact", 3)

    def _apply_opening_pre_draw_relics(self) -> None:
            miracle_count = 0
            if self._has_relic("Holy Water"):
                miracle_count += 3
            if self._has_relic("Pure Water"):
                miracle_count += 1
            for _ in range(miracle_count):
                if len(self.hand) < 10:
                    self.hand.append(make_card("Miracle", uuid=self._new_uuid("Miracle")))
            if self._has_relic("Ninja Scroll"):
                for _ in range(3):
                    if len(self.hand) < 10:
                        self.hand.append(make_card("Shiv", uuid=self._new_uuid("Shiv")))
            if self._has_relic("Toolbox"):
                self._add_random_card_to_hand(colorless=True, cost_for_turn=0)

    def _apply_player_power(
            self,
            power_id: str,
            amount: int,
            *,
            just_applied: bool | None = None,
            reroll_confusion_current_hand: bool = True,
    ) -> None:
            if amount == 0:
                return
            if power_id == "Weakened" and self._has_relic("Ginger"):
                return
            if power_id == "Frail" and self._has_relic("Turnip"):
                return
            is_debuff = self._is_debuff_power(power_id, amount)
            existing_amount = self.player.power(power_id)
            if is_debuff and self.player.power("Artifact") > 0:
                self.player.add_power("Artifact", -1)
                return
            self.player.add_power(power_id, amount)
            if is_debuff and amount > 0:
                if just_applied is None:
                    if power_id in {"Weakened", "Frail", "Vulnerable", "No Draw"}:
                        just_applied = existing_amount <= 0
                    else:
                        just_applied = True
                if just_applied:
                    self.player_powers_just_applied.add(power_id)
                    if power_id == "Confusion" and reroll_confusion_current_hand:
                        for card in self.hand:
                            self._roll_confusion_cost(card)

    def _apply_start_combat_relics(self) -> None:
            if self._has_relic("Vajra"):
                self.player.add_power("Strength", 1)
            if self._has_relic("Oddly Smooth Stone"):
                self.player.add_power("Dexterity", 1)
            if self._has_relic("Akabeko"):
                self.player.add_power("Akabeko", 8)
            if self._has_relic("Mutagenic Strength"):
                self.player.add_power("Strength", 3)
                self._apply_player_power("Flex Strength Down", 3)
            if self._has_relic("Bronze Scales"):
                self.player.add_power("Thorns", 3)
            if self._has_relic("Thread and Needle"):
                self.player.add_power("Plated Armor", 4)
            if self._has_relic("Fossilized Helix"):
                self.player.add_power("Buffer", 1)
            if self._has_relic("Pantograph") and self.floor in {16, 33, 50, 54}:
                self._heal(25)
            if self._has_relic("Philosopher's Stone"):
                for monster in self.monsters:
                    monster.add_power("Strength", 1)
            if self._has_relic("Sling of Courage") and self.elite:
                self.player.add_power("Strength", 2)
            if (girya := self._relic("Girya")) is not None and int(girya.get("counter", 0)) > 0:
                self.player.add_power("Strength", int(girya.get("counter", 0)))
            if self._has_relic("Snecko Eye"):
                self._apply_player_power("Confusion", 1)
            if self._has_relic("Enchiridion"):
                power_pool = [
                    card_id
                    for card_id in COMBAT_POWER_CARD_POOL_IRONCLAD
                    if card_id not in self.locked_card_ids
                ]
                if power_pool:
                    card_id = power_pool[int(self.card_random_rng.random(len(power_pool) - 1))]
                    self._add_to_hand(card_id, cost_for_turn=0)
            if self._has_relic("Brimstone"):
                self.player.add_power("Strength", 2)
                for monster in self.monsters:
                    if monster.alive:
                        monster.add_power("Strength", 1)
            if self._has_relic("Incense Burner") and self._advance_relic_counter("Incense Burner", 6):
                self.player.add_power("Intangible", 1)
            if self._has_relic("Mercury Hourglass"):
                self._deal_direct_damage_all(3)

    def _base_energy(self) -> int:
            extra_energy_relics = {
                "Busted Crown", "Coffee Dripper", "Cursed Key", "Ectoplasm", "Fusion Hammer",
                "Mark of Pain", "Philosopher's Stone", "Runic Dome", "Sozu", "Velvet Choker",
            }
            energy = 3 + sum(1 for relic_id in extra_energy_relics if self._has_relic(relic_id))
            if self._has_relic("Slaver's Collar") and (self.elite or self.floor in {16, 33, 50, 53, 54}):
                energy += 1
            energy += self.player.power("Berserk Energy")
            return energy

    def _bottled_card_refs(self) -> list[dict[str, str]]:
            bottled_types = {
                "Bottled Flame": "ATTACK",
                "Bottled Lightning": "SKILL",
                "Bottled Tornado": "POWER",
            }
            card_refs: list[dict[str, str]] = []
            for relic_id, card_type in bottled_types.items():
                relic = self._relic(relic_id)
                if relic is None:
                    continue
                card_id = relic.get("card_id")
                card_uuid = relic.get("card_uuid")
                if not card_id:
                    for card in self.deck:
                        if card.card_def.card_type == card_type:
                            card_id = card.card_id
                            card_uuid = card.uuid
                            relic["card_id"] = card_id
                            relic["card_uuid"] = card_uuid
                            break
                if not card_id:
                    continue
                card_refs.append({
                    "card_id": str(card_id),
                    "card_uuid": str(card_uuid) if card_uuid else "",
                })
            return card_refs

    def _bottled_deck_indexes(self) -> set[int]:
            bottled_uuids = {
                str(relic.get("card_uuid"))
                for relic in self.relics
                if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_uuid")
            }
            if bottled_uuids:
                return {index for index, card in enumerate(self.deck) if card.uuid in bottled_uuids}
            bottled_ids = {
                str(relic.get("card_id"))
                for relic in self.relics
                if relic.get("relic_id") in {"Bottled Flame", "Bottled Lightning", "Bottled Tornado"} and relic.get("card_id")
            }
            if not bottled_ids:
                return set()
            return {index for index, card in enumerate(self.deck) if card.card_id in bottled_ids}

    def _card_draw_per_turn(self) -> int:
            count = 5
            if self._has_relic("Snecko Eye"):
                count += 2
            if self._has_relic("Ring of the Serpent"):
                count += 1
            return count

    def _card_energy_cost(self, card: CardInstance) -> int:
            if self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL":
                return 0
            if card.card_id == "Blood for Blood":
                base_cost = 3 if card.upgrades > 0 else 4
                cost = card.cost_for_turn if card.cost_for_turn is not None else base_cost - card.misc
                return max(0, cost)
            cost = card.cost if card.cost_for_turn is None else card.cost_for_turn
            return max(0, cost)

    def _card_display_cost(self, card: CardInstance) -> int:
            if card.card_def.card_type == "SKILL" and card.cost_for_turn == -9:
                return 0
            if card.card_id == "Blood for Blood":
                base_cost = 3 if card.upgrades > 0 else 4
                cost = card.cost_for_turn if card.cost_for_turn is not None else base_cost - card.misc
                return max(0, cost)
            return card.cost

    def _roll_confusion_cost(self, card: CardInstance) -> None:
            if card.card_def.card_type not in {"ATTACK", "SKILL", "POWER"} or card.card_def.x_cost:
                return
            if card.cost < 0:
                return
            if card.card_id == "Blood for Blood":
                new_cost = self.card_random_rng.randint(0, 3)
                card.cost_for_turn = new_cost
                card.free_to_play_once = False
                return
            new_cost = self.card_random_rng.randint(0, 3)
            card.cost_for_combat = new_cost
            card.cost_for_turn = new_cost
            card.free_to_play_once = False
            if self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL":
                card.cost_for_turn = -9

    def _apply_corruption_to_existing_cards(self) -> None:
            for pile in (self.hand, self.draw_pile, self.discard_pile, self.exhaust_pile):
                for card in pile:
                    if card.card_def.card_type != "SKILL" or card.cost <= 0:
                        continue
                    card.cost_for_combat = 0
                    card.cost_for_turn = 0

    def _check_outcome(self) -> None:
            if self.player.current_hp <= 0:
                self.outcome = "PLAYER_LOSS"
            elif not any(monster.alive for monster in self.monsters):
                self.outcome = "PLAYER_VICTORY"
            else:
                can_lose_from_empty_piles = not any(
                    (
                        getattr(self, "processing_player_action", False),
                        getattr(self, "processing_start_turn", False),
                        getattr(self, "processing_end_turn_cleanup", False),
                        getattr(self, "processing_monster_turns", False),
                        bool(getattr(self, "pending_end_turn_resume", False)),
                        bool(getattr(self, "pending_start_turn_resume", False)),
                        bool(getattr(self, "pending_start_turn_autoplay", False)),
                        bool(getattr(self, "pending_start_turn_post_draw_resume", False)),
                        bool(getattr(self, "pending_autoplay_cards", [])),
                        getattr(self, "card_select_context", None) is not None,
                        bool(getattr(self, "pending_resolve_card", None)),
                    )
                )
                has_cards_remaining = bool(self.hand or self.draw_pile or self.discard_pile)
                has_damage_without_cards = (
                    self.player.power("Omega") > 0
                    or self.player.power("Thorns") > 0
                    or self.player.power("The Bomb") > 0
                )
                has_transient_alive = any(
                    monster.alive and monster.monster_id == "Transient"
                    for monster in self.monsters
                )
                if (
                    can_lose_from_empty_piles
                    and not has_cards_remaining
                    and not has_damage_without_cards
                    and not has_transient_alive
                ):
                    self.outcome = "PLAYER_LOSS"

    def _consume_match_and_keep_rng(self) -> list[CardInstance]:
            cards = [
                self._random_class_card_of_rarity("RARE"),
                self._random_class_card_of_rarity("UNCOMMON"),
                self._random_class_card_of_rarity("COMMON"),
            ]
            # returnColorlessCard(UNCOMMON) shuffles the colorless pool using the
            # shuffle RNG, not the card RNG.
            self.randoms.shuffle.random_long()
            colorless_pool = [CARD_LIBRARY[card_id] for card_id in COLORLESS_CARD_ID_ORDER if CARD_LIBRARY[card_id].rarity == "UNCOMMON"]
            if colorless_pool:
                cards.append(make_card(colorless_pool[0].card_id, uuid=f"match-colorless-{self.floor}"))
            curse_pool = [
                card_id for card_id, card_def in CARD_LIBRARY.items()
                if card_def.card_type == "CURSE" and card_id not in {"AscendersBane", "CurseOfTheBell"}
            ]
            if curse_pool:
                self.randoms.card.random(len(curse_pool) - 1)
            cards.append(make_card("Strike_R", uuid=f"match-starter-{self.floor}"))
            self.randoms.misc.random_long()
            return cards

    def _calculate_player_attack_damage_amount(
            self,
            base: int,
            monster: MonsterState | None,
            *,
            strength_multiplier: int = 1,
            strength_override: int | None = None,
            vigor_bonus: int | None = None,
            include_the_boot: bool = False,
        ) -> int:
            if monster is None:
                return 0
            if vigor_bonus is None:
                vigor_bonus = self._consume_attack_vigor_bonus()
            strength_amount = self.player.power("Strength") if strength_override is None else int(strength_override)
            damage = float(base + vigor_bonus + strength_amount * max(0, strength_multiplier))
            if self.player.power("Weakened") > 0:
                damage *= 0.75
            if monster.power("Vulnerable") > 0:
                damage *= 1.75 if self._has_relic("Paper Frog") else 1.5
            if monster.power("Slow") > 0:
                damage *= 1.0 + 0.1 * monster.power("Slow")
            if "Flight" in monster.powers:
                damage *= 0.5
            if getattr(self, "_attack_pen_nib_active", False):
                damage *= 2
            total_damage = max(0, math.floor(damage))
            if include_the_boot:
                unblocked_damage = max(0, total_damage - monster.block)
                if self._has_relic("The Boot") and 0 < unblocked_damage < 5:
                    total_damage = monster.block + 5
            return total_damage

    def _deal_attack_damage(
            self,
            base: int,
            monster: MonsterState | None,
            hits: int = 1,
            *,
            strength_multiplier: int = 1,
            strength_override: int | None = None,
            defer_counter_damage: bool = False,
            defer_attack_relic_proc: bool = False,
            suppress_deferred_attack_relic_resolution: bool = False,
            defer_monster_block_gains: bool = False,
            deferred_monster_block_requires_alive: bool = False,
            vigor_bonus: int | None = None,
        ) -> int:
            if monster is None:
                return 0
            if int(hits) <= 0:
                return 0
            total = 0
            curl_up_amount = monster.power("Curl Up")
            sharp_hide_damage = monster.power("Sharp Hide")
            pending_malleable_block = 0
            if vigor_bonus is None:
                vigor_bonus = self._consume_attack_vigor_bonus()
            for _ in range(int(hits)):
                flight_present = "Flight" in monster.powers
                flight_value = monster.power("Flight")
                was_alive = monster.alive
                if was_alive and monster.power("Angry") > 0:
                    monster.add_power("Strength", monster.power("Angry"))
                block_before = monster.block
                total_damage = self._calculate_player_attack_damage_amount(
                    base,
                    monster,
                    strength_multiplier=strength_multiplier,
                    strength_override=strength_override,
                    vigor_bonus=vigor_bonus,
                    include_the_boot=True,
                )
                dealt = self._apply_damage_to_monster(total_damage, monster)
                total += dealt
                if dealt > 0 and monster.power("Plated Armor") > 0:
                    monster.add_power("Plated Armor", -1)
                    if monster.power("Plated Armor") <= 0:
                        monster.powers.pop("Plated Armor", None)
                        if monster.monster_id == "ShelledParasite" and monster.alive:
                            monster.move = "SHELLED_STUNNED"
                            monster.intent = "STUN"
                            monster.move_base_damage = 0
                            monster.move_hits = 0
                if flight_present and dealt > 0:
                    next_flight = flight_value - 1
                    if monster.monster_id == "Byrd" and flight_value == 1:
                        monster.move = "BYRD_STUNNED"
                        monster.intent = "STUN"
                        monster.move_base_damage = 0
                        monster.move_hits = 0
                    monster.powers["Flight"] = next_flight
                if self._has_relic("Hand Drill") and block_before > 0 and monster.block <= 0 and monster.alive:
                    self._apply_monster_power(monster, "Vulnerable", 2)
                self._apply_guardian_mode_shift(monster, dealt)
                self._wake_lagavulin_if_damaged(monster, dealt)
                if dealt > 0 and monster.alive and monster.power("Malleable") > 0:
                    pending_malleable_block += monster.power("Malleable")
                    monster.add_power("Malleable", 1)
                    if (
                        pending_malleable_block > 0
                        and self.pending_attack_relic_proc
                        and not defer_counter_damage
                        and not defer_attack_relic_proc
                    ):
                        # Lightspeed queues the current hit's Malleable block
                        # before resolving 3-attack relic procs like
                        # Ornamental Fan, so Fan's Juggernaut damage sees the
                        # pending monster block gain ahead of it.
                        self.pending_monster_block_gains.append((monster, pending_malleable_block, True))
                        pending_malleable_block = 0
                if self.pending_attack_relic_proc and not defer_counter_damage and not defer_attack_relic_proc:
                    self._resolve_pending_attack_relic_proc()
                counter_damage = monster.power("Thorns")
                if counter_damage > 0:
                    if defer_counter_damage:
                        self.pending_counter_damage += counter_damage
                    else:
                        self._take_counter_damage(counter_damage)
                if was_alive and not monster.alive:
                    self._on_monster_defeated(monster)
                elif was_alive:
                    self._maybe_split_slime_boss(monster, dealt=total)
                if not monster.alive:
                    break
            if pending_malleable_block > 0:
                # Lightspeed queues Malleable's block gain after the current
                # attack action, so later on-exhaust damage like Charon's Ashes
                # can resolve before that block is awarded.
                self.pending_monster_block_gains.append((monster, pending_malleable_block, True))
            if (
                self.pending_attack_relic_proc
                and defer_attack_relic_proc
                and not defer_counter_damage
                and not suppress_deferred_attack_relic_resolution
            ):
                self._resolve_pending_attack_relic_proc()
            if sharp_hide_damage > 0:
                if defer_counter_damage:
                    self.pending_counter_damage += sharp_hide_damage
                else:
                    self._take_counter_damage(sharp_hide_damage)
            if total > 0 and curl_up_amount > 0:
                if defer_monster_block_gains:
                    self.pending_monster_block_gains.append((monster, curl_up_amount, deferred_monster_block_requires_alive))
                else:
                    self.pending_monster_block_gains.append((monster, curl_up_amount))
                monster.powers.pop("Curl Up", None)
            return total

    def _consume_attack_vigor_bonus(self) -> int:
            bonus = 0
            for key in ("Vigor", "Akabeko"):
                amount = self.player.power(key)
                if amount > 0:
                    bonus += amount
                    self.player.powers.pop(key, None)
            return bonus

    def _deal_combust_damage_all(self, amount: int) -> int:
            total = 0
            for monster in list(self._alive_monsters()):
                if monster.monster_id == "Lagavulin" and monster.ai_state.get("asleep", 0):
                    continue
                total += self._deal_direct_damage_to_monster(amount, monster)
            return total

    def _deal_damage_all(
            self,
            base: int,
            hits: int = 1,
            *,
            vigor_bonus: int | None = None,
            defer_attack_relic_proc: bool = False,
        ) -> int:
            if vigor_bonus is None:
                vigor_bonus = self._consume_attack_vigor_bonus()
            total = sum(
                self._deal_attack_damage(
                    base,
                    monster,
                    hits=hits,
                    vigor_bonus=vigor_bonus,
                    defer_attack_relic_proc=defer_attack_relic_proc,
                    suppress_deferred_attack_relic_resolution=defer_attack_relic_proc,
                )
                for monster in list(self._alive_monsters())
            )
            if self.pending_attack_relic_proc and defer_attack_relic_proc:
                self._resolve_pending_attack_relic_proc()
            return total

    def _deal_direct_damage_all(self, amount: int) -> int:
            return sum(self._deal_direct_damage_to_monster(amount, monster) for monster in list(self._alive_monsters()))

    def _deal_damage_to_monster(self, amount: int, monster: MonsterState | None) -> int:
            if monster is None:
                return 0
            was_alive = monster.alive
            amount = max(0, int(amount))
            if amount <= 0:
                return 0
            if monster.monster_id == "CorruptHeart":
                cap = 300
                already = self.monster_damage_this_turn.get(id(monster), 0)
                remaining = max(0, cap - already)
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + min(hp_damage, remaining)
            had_block = monster.block > 0
            if monster.power("Intangible") > 0 and amount > 0:
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + (1 if hp_damage > 0 else 0)
            dealt = _apply_damage(amount, monster)
            if had_block and monster.block <= 0 and self._has_relic("Hand Drill"):
                self._apply_monster_power(monster, "Vulnerable", 2)
            self._apply_guardian_mode_shift(monster, dealt)
            self._wake_lagavulin_if_damaged(monster, dealt)
            if was_alive and not monster.alive:
                self._on_monster_defeated(monster)
            elif was_alive:
                self._maybe_split_slime_boss(monster, dealt=dealt)
            return dealt

    def _deal_direct_damage_to_monster(self, amount: int, monster: MonsterState | None) -> int:
            return self._deal_damage_to_monster(amount, monster)

    def _deal_retaliatory_damage_to_monster(self, amount: int, monster: MonsterState | None) -> int:
            if monster is None:
                return 0
            was_alive = monster.alive
            amount = max(0, int(amount))
            if amount <= 0:
                return 0
            if monster.monster_id == "CorruptHeart":
                cap = 300
                already = self.monster_damage_this_turn.get(id(monster), 0)
                remaining = max(0, cap - already)
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + min(hp_damage, remaining)
            if monster.power("Intangible") > 0 and amount > 0:
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + (1 if hp_damage > 0 else 0)
            # Lightspeed resolves player Thorns through the normal monster
            # damage pipeline, so existing monster block still absorbs it.
            dealt = _apply_damage(amount, monster)
            if monster.monster_id == "CorruptHeart" and dealt > 0:
                self.monster_damage_this_turn[id(monster)] = self.monster_damage_this_turn.get(id(monster), 0) + dealt
            self._apply_guardian_mode_shift(monster, dealt)
            self._wake_lagavulin_if_damaged(monster, dealt)
            if was_alive and not monster.alive:
                self._on_monster_defeated(monster)
            elif was_alive:
                self._maybe_split_slime_boss(monster, dealt=dealt)
            return dealt

    def _deal_flame_barrier_damage_to_monster(self, amount: int, monster: MonsterState | None) -> int:
            if monster is None:
                return 0
            was_alive = monster.alive
            amount = max(0, int(amount))
            if amount <= 0:
                return 0
            if monster.monster_id == "CorruptHeart":
                cap = 300
                already = self.monster_damage_this_turn.get(id(monster), 0)
                remaining = max(0, cap - already)
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + min(hp_damage, remaining)
            if monster.power("Intangible") > 0 and amount > 0:
                blocked = min(monster.block, amount)
                hp_damage = max(0, amount - blocked)
                amount = blocked + (1 if hp_damage > 0 else 0)
            dealt = _apply_damage(amount, monster)
            if monster.monster_id == "CorruptHeart" and dealt > 0:
                self.monster_damage_this_turn[id(monster)] = self.monster_damage_this_turn.get(id(monster), 0) + dealt
            self._apply_guardian_mode_shift(monster, dealt)
            self._wake_lagavulin_if_damaged(monster, dealt)
            if was_alive and not monster.alive:
                self._on_monster_defeated(monster)
            elif was_alive:
                self._maybe_split_slime_boss(monster, dealt=dealt)
            return dealt

    def _discovery_card_options(self, *, card_type: str | None = None, colorless: bool = False) -> list[CardInstance]:
            options: list[CardInstance] = []
            seen: set[str] = set()
            if colorless:
                candidates = [
                    card_id for card_id in COMBAT_COLORLESS_CARD_POOL
                    if card_type is None or CARD_LIBRARY[card_id].card_type == card_type
                ]
            elif card_type == "ATTACK":
                candidates = list(COMBAT_ATTACK_CARD_POOL_IRONCLAD)
            elif card_type == "SKILL":
                candidates = list(COMBAT_SKILL_CARD_POOL_IRONCLAD)
            elif card_type == "POWER":
                candidates = list(COMBAT_POWER_CARD_POOL_IRONCLAD)
            else:
                candidates = list(COMBAT_CARD_POOL_IRONCLAD)
            candidates = [card_id for card_id in candidates if card_id not in self.locked_card_ids]
            while len(options) < 3 and candidates:
                card_id = candidates[int(self.card_random_rng.random(len(candidates) - 1))]
                if card_id in seen:
                    continue
                seen.add(card_id)
                options.append(make_card(card_id, uuid=self._new_uuid(f"Discovery-{card_id}")))
            return options

    def _drain_pending_autoplay_cards(self) -> None:
            if self._processing_autoplay_cards:
                return
            self._processing_autoplay_cards = True
            try:
                while self.pending_autoplay_cards and self.card_select_context is None and self.outcome == "UNDECIDED":
                    top, target_index, force_exhaust, energy_on_use = self.pending_autoplay_cards.pop(0)
                    target = self.monsters[target_index] if 0 <= target_index < len(self.monsters) else None
                    if top.card_def.has_target and (target is None or not target.alive):
                        continue
                    self.hand.append(top)
                    index = len(self.hand) - 1
                    if not self.playable(top, free_to_play=True):
                        self.hand.pop(index)
                        continue
                    self.play_card(
                        index,
                        target_index,
                        force_exhaust=force_exhaust,
                        free_to_play=True,
                        energy_on_use=energy_on_use,
                    )
            finally:
                self._processing_autoplay_cards = False

    def _exhaust_card(
            self,
            card: CardInstance,
            *,
            defer_dark_embrace_draws: list[int] | None = None,
            deferred_dark_embrace_status_damage_all: list[int] | None = None,
            deferred_after_exhaust_actions: list[tuple[str, int | str]] | None = None,
            defer_feel_no_pain_blocks: list[int] | None = None,
        ) -> None:
            if self._has_relic("Strange Spoon") and card.card_def.card_type in {"ATTACK", "SKILL", "POWER"} and self.rng.random() < 0.5:
                self.discard_pile.append(card)
                return
            self.exhaust_pile.append(card)
            if card.card_id == "Sentinel":
                self.player.energy += 3 if card.upgrades else 2
            if self._has_relic("Charon's Ashes"):
                self._deal_direct_damage_all(3)
            if self._has_relic("Dead Branch"):
                chosen = self._random_combat_card_id()
                if deferred_after_exhaust_actions is not None:
                    deferred_after_exhaust_actions.append(("add_to_hand", chosen))
                else:
                    self._add_to_hand(chosen)
            if self.player.power("Dark Embrace") > 0:
                if deferred_after_exhaust_actions is not None:
                    deferred_after_exhaust_actions.append(("draw", self.player.power("Dark Embrace")))
                elif defer_dark_embrace_draws is not None:
                    defer_dark_embrace_draws.append(self.player.power("Dark Embrace"))
                else:
                    self.draw_cards(
                        self.player.power("Dark Embrace"),
                        deferred_status_damage_all=deferred_dark_embrace_status_damage_all,
                        exclude_end_turn_regular_discards=getattr(self, "processing_end_turn_cleanup", False),
                    )
            if self.player.power("Feel No Pain") > 0:
                if deferred_after_exhaust_actions is not None:
                    deferred_after_exhaust_actions.append(("gain_block_no_mod", self.player.power("Feel No Pain")))
                elif defer_feel_no_pain_blocks is not None:
                    defer_feel_no_pain_blocks.append(self.player.power("Feel No Pain"))
                else:
                    self._gain_block(
                        self.player.power("Feel No Pain"),
                        defer_juggernaut=getattr(self, "processing_end_turn_cleanup", False),
                        apply_block_modifiers=False,
                    )
            if card.card_id == "Necronomicurse":
                if card in self.exhaust_pile:
                    self.exhaust_pile.remove(card)
                if len(self.hand) < 10:
                    self.hand.append(card)
                else:
                    self.discard_pile.append(card)

    def _resolve_deferred_after_exhaust_actions(self, actions: list[tuple[str, int | str]]) -> None:
            for action_kind, payload in actions:
                if action_kind == "add_to_hand":
                    self._add_to_hand(str(payload))
                elif action_kind == "add_to_draw_top":
                    self.draw_pile.append(make_card(str(payload), uuid=self._new_uuid(str(payload))))
                elif action_kind == "draw":
                    self.draw_cards(int(payload))
                elif action_kind == "gain_block_no_mod":
                    self._gain_block(int(payload), apply_block_modifiers=False)

    def _exhaust_non_attacks_from_hand(
            self,
            *,
            deferred_after_exhaust_actions: list[tuple[str, int | str]] | None = None,
        ) -> int:
            exhausted = 0
            for hand_index in range(len(self.hand) - 1, -1, -1):
                other = self.hand[hand_index]
                if other.card_def.card_type != "ATTACK":
                    self.hand.pop(hand_index)
                    self._exhaust_card(other, deferred_after_exhaust_actions=deferred_after_exhaust_actions)
                    exhausted += 1
            return exhausted

    def _pop_non_attacks_from_hand(self) -> list[CardInstance]:
            exhausted_cards: list[CardInstance] = []
            for hand_index in range(len(self.hand) - 1, -1, -1):
                other = self.hand[hand_index]
                if other.card_def.card_type != "ATTACK":
                    exhausted_cards.append(self.hand.pop(hand_index))
            return exhausted_cards

    def _fetch_from_draw_to_hand(self, card_type: str) -> None:
            for card in list(self.draw_pile):
                if card.card_def.card_type == card_type:
                    self.draw_pile.remove(card)
                    if len(self.hand) >= 10:
                        self.discard_pile.append(card)
                    else:
                        self.hand.append(card)
                    return

    def _draw_to_hand_candidate_indexes(self, card_type: str) -> list[int]:
            matching_indexes: list[int] = []
            for draw_index, card in enumerate(self.draw_pile):
                if card.card_def.card_type != card_type:
                    continue
                if matching_indexes:
                    # Lightspeed consumes cardRandomRng while building the temporary
                    # selectable list for Secret Technique / Secret Weapon.
                    self.card_random_rng.random(len(matching_indexes) - 1)
                matching_indexes.append(draw_index)
            return matching_indexes

    def _choose_draw_pile_card_to_hand(self, draw_index: int) -> None:
            if draw_index < 0 or draw_index >= len(self.draw_pile):
                return
            selected = self.draw_pile.pop(draw_index)
            if len(self.hand) >= 10:
                self.discard_pile.append(selected)
            else:
                self.hand.append(selected)

    def _gain_block(self, amount: int, *, defer_juggernaut: bool = False, apply_block_modifiers: bool = True) -> None:
            gained = _player_block_amount(amount, self.player) if apply_block_modifiers else max(0, int(amount))
            if gained <= 0:
                return
            self.player.block += gained
            if self.player.power("Juggernaut") > 0:
                defer_juggernaut = defer_juggernaut or bool(getattr(self, "defer_juggernaut_for_current_card", False))
                if defer_juggernaut:
                    self.pending_juggernaut_damage += self.player.power("Juggernaut")
                else:
                    self._trigger_juggernaut(self.player.power("Juggernaut"))

    def _has_relic(self, relic_id: str) -> bool:
            return any(relic.get("relic_id") == relic_id for relic in self.relics)

    def _sync_red_skull_strength(self) -> None:
            if not self._has_relic("Red Skull"):
                self._red_skull_active = False
                return
            is_bloodied = self.player.current_hp <= self.player.max_hp // 2
            if is_bloodied and not self._red_skull_active:
                self.player.add_power("Strength", 3)
                self._red_skull_active = True
            elif not is_bloodied:
                self._red_skull_active = False

    def _heal(self, amount: int) -> None:
            amount = max(0, int(amount))
            if amount > 0 and self._has_relic("Magic Flower"):
                amount = int(amount * 1.5)
            was_bloodied = self.player.current_hp <= self.player.max_hp // 2
            self.player.current_hp = min(self.player.max_hp, self.player.current_hp + amount)
            # Lightspeed's current Red Skull handling adds Strength again when
            # healing from bloodied to above-half, rather than removing it.
            if self._has_relic("Red Skull") and was_bloodied and self.player.current_hp > self.player.max_hp // 2:
                self.player.add_power("Strength", 3)
                self._red_skull_active = False

    def _try_auto_revive_with_fairy_potion(self) -> bool:
            if self.player.current_hp > 0:
                return False
            for index, potion in enumerate(self.potions):
                if potion.potion_id != "FairyPotion":
                    continue
                potion_multiplier = 2 if self._has_relic("Sacred Bark") else 1
                self.potions[index] = PotionInstance()
                self.player.current_hp = 0
                self._heal(max(1, int(self.player.max_hp * 0.3 * potion_multiplier)))
                return True
            return False

    def _init_opening_draw_pile(self) -> None:
            idxs = list(range(len(self.deck)))
            java_collections_shuffle(idxs, self.shuffle_rng.random_long())
            bottled_indexes = self._bottled_deck_indexes()
            normal_cards: list[CardInstance] = []
            innate_cards: list[CardInstance] = []
            for deck_idx in idxs:
                card = self.deck[deck_idx]
                if deck_idx in bottled_indexes or self._card_is_opening_innate(card):
                    innate_cards.append(card)
                else:
                    normal_cards.append(card)
            self.opening_innate_count = len(innate_cards)
            self.draw_pile = normal_cards + innate_cards

    def _insert_temp_card_into_draw_pile(self, card_id: str, *, upgrades: int = 0) -> None:
            card = make_card(card_id, upgrades=upgrades, uuid=self._new_uuid(card_id))
            insert_index = 0 if not self.draw_pile else int(self.card_random_rng.random(len(self.draw_pile) - 1))
            self.draw_pile.insert(insert_index, card)

    def _ironclad_card_pool(self, *, card_type: str | None = None, rarity: str | None = None):
            return ironclad_card_pool(card_type=card_type, rarity=rarity, exclude_ids=self.locked_card_ids)

    def _is_debuff_power(self, power_id: str, amount: int) -> bool:
            return power_id in {
                "Vulnerable",
                "Weakened",
                "Frail",
                "Shackled",
                "Dexterity Down",
                "Flex Strength Down",
                "Confusion",
                "Hex",
                "No Draw",
                "No Block",
                "No Attack",
            } or (power_id in {"Strength", "Dexterity"} and amount < 0)

    def _lose_hp(self, amount: int, *, from_attack: bool = False, self_damage: bool = False) -> None:
            amount = max(0, int(amount))
            if amount <= 0:
                return
            if self.player.power("Intangible") > 0:
                amount = min(amount, 1)
            if from_attack and self.player.power("Buffer") > 0:
                self.player.add_power("Buffer", -1)
                return
            if from_attack and self._has_relic("Torii") and 1 < amount <= 5:
                amount = 1
            if self._has_relic("Tungsten Rod"):
                amount = max(0, amount - 1)
                if amount <= 0:
                    return
            self.player.current_hp = max(0, self.player.current_hp - amount)
            self._sync_red_skull_strength()
            self.hp_lost_this_combat += amount
            self.hp_loss_events_this_combat += 1
            self._on_player_took_damage_cards()
            if amount > 0 and (relic := self._relic("Centennial Puzzle")) is not None and int(relic.get("counter", 0)) == 0:
                relic["counter"] = 1
                end_turn_self_damage = (
                    self_damage
                    and getattr(self, "processing_end_turn_cleanup", False)
                    and getattr(self, "monster_turn_damage_draws_are_end_turn", False)
                )
                if not end_turn_self_damage or self.draw_pile or (
                    len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0)
                ) > 0:
                    self._draw_player_damage_relic_cards(3, end_turn_self_damage=end_turn_self_damage)
            if amount > 0 and self._has_relic("Runic Cube"):
                end_turn_self_damage = (
                    self_damage
                    and getattr(self, "processing_end_turn_cleanup", False)
                    and getattr(self, "monster_turn_damage_draws_are_end_turn", False)
                )
                if not end_turn_self_damage or self.draw_pile or (
                    len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0)
                ) > 0:
                    self._draw_player_damage_relic_cards(1, end_turn_self_damage=end_turn_self_damage)
            self._try_auto_revive_with_fairy_potion()
            if self.player.power("Rupture") > 0 and self_damage:
                self.player.add_power("Strength", self.player.power("Rupture"))
            if self._has_relic("Self-Forming Clay"):
                self.player.add_power("Self-Forming Clay Block", 3)
            self._check_outcome()

    def _maybe_split_slime_boss(self, monster: MonsterState, *, dealt: int = 1) -> None:
            if dealt <= 0:
                return
            if monster.current_hp <= 0 or monster.current_hp > monster.max_hp // 2:
                return
            if monster.monster_id == "AcidSlime_L":
                monster.move = "ACID_SLIME_L_SPLIT"
                if monster.move_history:
                    monster.move_history[0] = "ACID_SLIME_L_SPLIT"
                else:
                    monster.move_history = ["ACID_SLIME_L_SPLIT"]
                monster.intent = "UNKNOWN"
                monster.move_base_damage = 0
                monster.move_hits = 0
                return
            if monster.monster_id == "SpikeSlime_L":
                monster.move = "SPIKE_SLIME_L_SPLIT"
                if monster.move_history:
                    monster.move_history[0] = "SPIKE_SLIME_L_SPLIT"
                else:
                    monster.move_history = ["SPIKE_SLIME_L_SPLIT"]
                monster.intent = "UNKNOWN"
                monster.move_base_damage = 0
                monster.move_hits = 0
                return
            if monster.monster_id != "SlimeBoss":
                return
            monster.move = "SLIME_BOSS_SPLIT"
            if monster.move_history:
                monster.move_history[0] = "SLIME_BOSS_SPLIT"
            else:
                monster.move_history = ["SLIME_BOSS_SPLIT"]
            monster.intent = "MAGIC"
            monster.move_base_damage = 0
            monster.move_hits = 0

    def _monster_vulnerable_multiplier(self) -> float:
            return 1.25 if self._has_relic("Odd Mushroom") else 1.5

    def _move_bottled_cards_to_hand(self) -> None:
            for bottle in self._bottled_card_refs():
                for hand_index, card in enumerate(list(self.hand)):
                    if (
                        bottle.get("card_uuid") and card.uuid == bottle.get("card_uuid")
                    ) or card.card_id == bottle.get("card_id"):
                        self.hand.pop(hand_index)
                        self.hand.insert(0, card)
                        break

    def _move_bottled_cards_to_top(self) -> None:
            for bottle in self._bottled_card_refs():
                for pile in (self.draw_pile, self.discard_pile):
                    for card in list(pile):
                        if (
                            bottle.get("card_uuid") and card.uuid == bottle.get("card_uuid")
                        ) or card.card_id == bottle.get("card_id"):
                            pile.remove(card)
                            self.draw_pile.append(card)
                            break
                    else:
                        continue
                    break

    def _move_card_to_discard(self, card: CardInstance) -> None:
            self.discard_pile.append(card)

    def _clear_temporary_cost_state(self) -> None:
            for pile in (self.hand, self.draw_pile, self.discard_pile):
                for card in pile:
                    if card.card_id == "Blood for Blood":
                        # Blood for Blood keeps real combat-wide cost reduction,
                        # but turn-only freebies such as Infernal Blade should
                        # fall off at the turn boundary. Madness-style base-cost
                        # rewrites are preserved because they mutate the card def.
                        if card.card_def.cost == 0 and (card.card_def.upgraded_cost or 0) == 0:
                            card.cost_for_turn = 0
                            continue
                        if card.cost_for_combat is not None:
                            card.cost_for_turn = card.cost_for_combat
                            continue
                        base_cost = 3 if card.upgrades > 0 else 4
                        card.cost_for_turn = max(0, base_cost - int(card.misc or 0))
                        continue
                    card.cost_for_turn = None

    def _move_innate_cards_to_top(self) -> None:
            innate_cards: list[CardInstance] = []
            for pile in (self.draw_pile, self.discard_pile):
                for card in list(pile):
                    if self._card_is_opening_innate(card):
                        pile.remove(card)
                        innate_cards.append(card)
            self.draw_pile.extend(innate_cards)

    def _new_uuid(self, prefix: str) -> str:
            self._next_uuid += 1
            return f"{prefix}-{self.seed}-{self.turn}-{self._next_uuid}"

    def _on_monster_defeated(self, monster: MonsterState) -> None:
            if monster.monster_id == "Darkling":
                monster.current_hp = 0
                monster.is_gone = False
                monster.half_dead = True
                return
            if monster.monster_id == "AwakenedOne" and monster.power("Awakened Reborn") <= 0:
                monster.current_hp = 300
                monster.max_hp = max(monster.max_hp, 300)
                monster.is_gone = False
                monster.half_dead = False
                monster.powers = {"Awakened Reborn": 1}
                choose_next_move(monster, self.ai_rng)
                return
            if monster.monster_id in {"Looter", "Mugger"}:
                stolen_gold = int(monster.ai_state.pop("stolen_gold", 0) or 0)
                if stolen_gold > 0:
                    self.reward_gold_bonus += stolen_gold
            battle_continues = any(other.alive for other in self.monsters if other is not monster)
            if not battle_continues:
                return
            if monster.power("Spore Cloud") > 0:
                if getattr(self, "processing_monster_turns", False):
                    self.pending_spore_cloud_monster_turn_triggers += 1
                else:
                    self.pending_spore_cloud_player_turn_triggers += 1
            if monster.monster_id == "BronzeOrb":
                stasis_card = monster.ai_state.pop("stasis_card", None)
                if isinstance(stasis_card, CardInstance):
                    if len(self.hand) < 10:
                        self.hand.append(stasis_card)
                    else:
                        self.discard_pile.append(stasis_card)
            self._on_monster_killed()

    def _on_monster_killed(self) -> None:
            if self._has_relic("Gremlin Horn"):
                self.pending_monster_kill_triggers += 1

    def _resolve_pending_monster_kill_triggers(self) -> None:
            pending_spore_cloud_player_turn = max(0, int(getattr(self, "pending_spore_cloud_player_turn_triggers", 0) or 0))
            pending_spore_cloud_monster_turn = max(0, int(getattr(self, "pending_spore_cloud_monster_turn_triggers", 0) or 0))
            pending = max(0, int(getattr(self, "pending_monster_kill_triggers", 0) or 0))
            if pending_spore_cloud_player_turn <= 0 and pending_spore_cloud_monster_turn <= 0 and pending <= 0:
                return
            self.pending_spore_cloud_player_turn_triggers = 0
            self.pending_spore_cloud_monster_turn_triggers = 0
            self.pending_monster_kill_triggers = 0
            for _ in range(pending_spore_cloud_player_turn):
                self._apply_player_power("Vulnerable", 2, just_applied=False)
            for _ in range(pending_spore_cloud_monster_turn):
                self._apply_player_power("Vulnerable", 2)
            for _ in range(pending):
                if self._has_relic("Gremlin Horn"):
                    self.player.energy += 1
                    self.draw_cards(1)

    def _resolve_pending_spore_cloud_triggers(self) -> None:
            pending_spore_cloud_player_turn = max(0, int(getattr(self, "pending_spore_cloud_player_turn_triggers", 0) or 0))
            pending_spore_cloud_monster_turn = max(0, int(getattr(self, "pending_spore_cloud_monster_turn_triggers", 0) or 0))
            if pending_spore_cloud_player_turn <= 0 and pending_spore_cloud_monster_turn <= 0:
                return
            self.pending_spore_cloud_player_turn_triggers = 0
            self.pending_spore_cloud_monster_turn_triggers = 0
            for _ in range(pending_spore_cloud_player_turn):
                self._apply_player_power("Vulnerable", 2, just_applied=False)
            for _ in range(pending_spore_cloud_monster_turn):
                self._apply_player_power("Vulnerable", 2)

    def _on_player_took_damage_cards(self) -> None:
            for pile in (self.hand, self.draw_pile, self.discard_pile):
                for card in pile:
                    if card.card_id != "Blood for Blood":
                        continue
                    card.misc += 1
                    if card.cost_for_combat is not None:
                        card.cost_for_combat = max(0, card.cost_for_combat - 1)
                    if card.cost_for_turn is not None:
                        card.cost_for_turn = max(0, card.cost_for_turn - 1)

    def _open_combat_card_select(
            self,
            context: str,
            selectable_indexes: list[int],
            *,
            pending_card: CardInstance | None = None,
            pending_force_exhaust: bool = False,
            pending_hex_dazed: int = 0,
        ) -> None:
            self.card_select_context = context
            self.pending_resolve_card = pending_card
            self.pending_resolve_force_exhaust = pending_force_exhaust
            self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
            self.pending_resolve_used_free_to_play_once = False
            self.card_select_generated_cards = []
            self.card_select_source_indexes = []
            self.card_select_options = []
            for hand_index in selectable_indexes:
                if 0 <= hand_index < len(self.hand):
                    card = self.hand[hand_index]
                    self.card_select_options.append({
                        "kind": "card_select",
                        "name": context,
                        "select_type": context,
                        "choice_index": hand_index,
                        "select_index": hand_index,
                        "deck_index": hand_index,
                        "card_id": card.card_id,
                        "bits": 2000 + hand_index,
                    })

    def _open_confirm_combat_card_select(
            self,
            context: str,
            *,
            pending_card: CardInstance | None = None,
            pending_force_exhaust: bool = False,
            pending_hex_dazed: int = 0,
        ) -> None:
            self.card_select_context = context
            self.pending_resolve_card = pending_card
            self.pending_resolve_force_exhaust = pending_force_exhaust
            self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
            self.pending_resolve_used_free_to_play_once = False
            self.card_select_generated_cards = []
            self.card_select_source_indexes = []
            self.card_select_options = [
                {
                    "kind": "multi_card_select",
                    "name": context,
                    "select_type": context,
                    "choice_index": 0,
                    "select_index": 0,
                    "deck_index": 0,
                    "bits": 3500,
                }
            ]

    def _open_discard_card_select(
            self,
            context: str,
            selectable_indexes: list[int],
            *,
            pending_card: CardInstance | None = None,
            pending_force_exhaust: bool = False,
            pending_hex_dazed: int = 0,
        ) -> None:
            self.card_select_context = context
            self.pending_resolve_card = pending_card
            self.pending_resolve_force_exhaust = pending_force_exhaust
            self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
            self.pending_resolve_used_free_to_play_once = False
            self.card_select_generated_cards = []
            self.card_select_source_indexes = []
            self.card_select_options = []
            for discard_index in selectable_indexes:
                if 0 <= discard_index < len(self.discard_pile):
                    card = self.discard_pile[discard_index]
                    self.card_select_options.append({
                        "kind": "card_select",
                        "name": context,
                        "select_type": context,
                        "choice_index": discard_index,
                        "select_index": discard_index,
                        "deck_index": discard_index,
                        "card_id": card.card_id,
                        "bits": 4000 + discard_index,
                    })

    def _open_exhaust_card_select(
            self,
            context: str,
            selectable_indexes: list[int],
            *,
            pending_card: CardInstance | None = None,
            pending_force_exhaust: bool = False,
            pending_hex_dazed: int = 0,
        ) -> None:
            self.card_select_context = context
            self.pending_resolve_card = pending_card
            self.pending_resolve_force_exhaust = pending_force_exhaust
            self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
            self.pending_resolve_used_free_to_play_once = False
            self.card_select_generated_cards = []
            self.card_select_source_indexes = list(selectable_indexes)
            self.card_select_options = []
            for display_index, exhaust_index in enumerate(self.card_select_source_indexes):
                if 0 <= exhaust_index < len(self.exhaust_pile):
                    card = self.exhaust_pile[exhaust_index]
                    action_index = exhaust_index if context == "EXHUME" else display_index
                    self.card_select_options.append({
                        "kind": "card_select",
                        "name": context,
                        "select_type": context,
                        "choice_index": display_index,
                        "select_index": action_index,
                        "deck_index": action_index,
                        "card_id": card.card_id,
                        "bits": 5000 + action_index,
                    })

    def _open_generated_combat_card_select(
            self,
            context: str,
            cards: list[CardInstance],
            *,
            pending_card: CardInstance | None = None,
            pending_force_exhaust: bool = False,
            pending_hex_dazed: int = 0,
        ) -> None:
            self.card_select_context = context
            self.pending_resolve_card = pending_card
            self.pending_resolve_force_exhaust = pending_force_exhaust
            self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
            self.pending_resolve_used_free_to_play_once = False
            self.card_select_generated_cards = cards
            self.card_select_source_indexes = []
            self.card_select_options = [
                {
                    "kind": "card_select",
                    "name": context,
                    "select_type": context,
                    "choice_index": index,
                    "select_index": index,
                    "deck_index": index,
                    "card_id": card.card_id,
                    "card": card_to_spirecomm(card),
                    "bits": 3000 + index,
                }
                for index, card in enumerate(cards)
            ]

    def _open_draw_pile_card_select(
            self,
            context: str,
            selectable_indexes: list[int],
            *,
            pending_card: CardInstance | None = None,
            pending_force_exhaust: bool = False,
            pending_hex_dazed: int = 0,
        ) -> None:
            self.card_select_context = context
            self.pending_resolve_card = pending_card
            self.pending_resolve_force_exhaust = pending_force_exhaust
            self.pending_resolve_hex_dazed = max(0, int(pending_hex_dazed))
            self.pending_resolve_used_free_to_play_once = False
            self.card_select_generated_cards = []
            self.card_select_source_indexes = []
            self.card_select_options = []
            for draw_index in selectable_indexes:
                if 0 <= draw_index < len(self.draw_pile):
                    card = self.draw_pile[draw_index]
                    self.card_select_options.append({
                        "kind": "card_select",
                        "name": context,
                        "select_type": context,
                        "choice_index": draw_index,
                        "select_index": draw_index,
                        "deck_index": draw_index,
                        "card_id": card.card_id,
                        "card": card_to_spirecomm(card),
                        "bits": 6000 + draw_index,
                    })

    def _play_random_top_card(self, *, force_exhaust: bool = False) -> None:
            top = self._top_card_from_draw_pile()
            if top is None:
                return
            # Lightspeed currently burns one card-random target roll for any
            # autoplayed top card while monsters are alive, even if that card
            # ends up not using the target directly (for example, Havoc ->
            # Defend_R / Inflame after a reshuffle). The chosen target index is
            # then only consumed if the played card actually has a target.
            target = self._random_alive_monster(burn_if_single=True)
            energy_on_use = self.player.energy if top.card_def.x_cost else None
            self.pending_autoplay_cards.append(
                (
                    top,
                    self.monsters.index(target) if target in self.monsters else 0,
                    bool(force_exhaust and top.card_def.card_type != "POWER"),
                    energy_on_use,
                )
            )

    def _put_random_cards_in_draw_pile(self, *, card_type: str, count: int, cost_for_turn: int = 0) -> None:
            card_ids = [self._random_combat_card_id(card_type=card_type) for _ in range(count)]
            for card_id in card_ids:
                card = make_card(card_id, uuid=self._new_uuid(card_id))
                card.cost_for_combat = cost_for_turn
                card.cost_for_turn = cost_for_turn
                insert_index = 0 if not self.draw_pile else int(self.card_random_rng.random(len(self.draw_pile) - 1))
                self.draw_pile.insert(insert_index, card)

    def _random_alive_monster(self, *, burn_if_single: bool = True) -> MonsterState | None:
            alive = self._alive_monsters()
            if not alive:
                return None
            if len(alive) == 1 and not burn_if_single:
                return alive[0]
            return alive[int(self.card_random_rng.random(len(alive) - 1))]

    def _random_card_id(self, *, card_type: str | None = None, colorless: bool = False) -> str:
            if colorless:
                candidates = [
                    card_id for card_id in COMBAT_COLORLESS_CARD_POOL
                    if card_type is None or CARD_LIBRARY[card_id].card_type == card_type
                ]
            elif card_type == "ATTACK":
                candidates = list(COMBAT_ATTACK_CARD_POOL_IRONCLAD)
            elif card_type == "SKILL":
                candidates = list(COMBAT_SKILL_CARD_POOL_IRONCLAD)
            elif card_type == "POWER":
                candidates = list(COMBAT_POWER_CARD_POOL_IRONCLAD)
            else:
                candidates = list(COMBAT_CARD_POOL_IRONCLAD)
            candidates = [card_id for card_id in candidates if card_id not in self.locked_card_ids]
            return candidates[int(self.card_random_rng.random(len(candidates) - 1))]

    def _random_class_card_of_rarity(self, rarity: str) -> CardInstance:
            pool = self._ironclad_card_pool(rarity=rarity)
            if not pool:
                pool = self._ironclad_card_pool()
            chosen = pool[int(self.randoms.card.random(len(pool) - 1))]
            return make_card(chosen.card_id, uuid=f"rarity-{self.floor}-{chosen.card_id}")

    def _random_class_card_of_rarity_with_rng(self, rarity: str, rng: StsRandom) -> CardInstance:
            pool = self._ironclad_card_pool(rarity=rarity)
            if not pool:
                pool = self._ironclad_card_pool()
            chosen = pool[int(rng.random(len(pool) - 1))]
            return self._make_deck_card(chosen.card_id, uuid=f"rarity-{self.floor}-{chosen.card_id}")

    def _random_combat_card_id(self, *, card_type: str | None = None) -> str:
            if card_type == "ATTACK":
                candidates = COMBAT_ATTACK_CARD_POOL_IRONCLAD
            elif card_type == "SKILL":
                candidates = COMBAT_SKILL_CARD_POOL_IRONCLAD
            elif card_type == "POWER":
                candidates = COMBAT_POWER_CARD_POOL_IRONCLAD
            else:
                candidates = COMBAT_CARD_POOL_IRONCLAD
            return candidates[int(self.card_random_rng.random(len(candidates) - 1))]

    def _relic(self, relic_id: str) -> dict[str, Any] | None:
            return next((relic for relic in self.relics if relic.get("relic_id") == relic_id), None)

    def _resolve_headbutt_replay(self, card: CardInstance, target: MonsterState | None) -> bool:
            if card.card_def.has_target and target is not None and not target.alive:
                return False
            if self.card_select_context is None:
                self._resolve_pending_monster_block_gains()
                self._resolve_pending_monster_direct_damage()
            pending_after_use_status_effects: list[tuple[str, int]] = []
            pending_after_use_energy_gain = 0
            self.cards_played_this_turn += 1
            self.attack_played_this_turn += 1
            if self.attack_played_this_turn % 3 == 0:
                self.pending_attack_relic_proc = True
            if self._advance_relic_counter("Nunchaku", 10):
                self.player.energy += 1
                pending_after_use_energy_gain += 1
            if self._advance_relic_counter("Pen Nib", 10):
                self.player.add_power("Pen Nib", 1)
            self._attack_pen_nib_active = self.player.power("Pen Nib") > 0
            if self._attack_pen_nib_active:
                self.player.add_power("Pen Nib", -1)
            if self.player.power("Rage") > 0:
                pending_after_use_status_effects.append(("gain_block_no_mod", self.player.power("Rage")))
            if self._advance_relic_counter("Ink Bottle", 10):
                pending_after_use_status_effects.append(("draw", 1))
            target_was_alive = bool(target is not None and target.alive)
            self.pending_counter_damage = 0
            self._deal_attack_damage(
                12 if card.upgrades else 9,
                target,
                defer_counter_damage=True,
                defer_monster_block_gains=True,
                deferred_monster_block_requires_alive=True,
            )
            counter_damage = self.pending_counter_damage
            self.pending_counter_damage = 0
            self._check_outcome()
            self._resolve_pending_spore_cloud_triggers()
            if self.outcome != "UNDECIDED":
                if counter_damage > 0:
                    self._take_counter_damage(counter_damage)
                if pending_after_use_status_effects:
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
                self._resolve_pending_attack_relic_proc()
                self._attack_pen_nib_active = False
                return False
            if not target_was_alive and target is not None and not target.alive:
                if counter_damage > 0:
                    self._take_counter_damage(counter_damage)
                if pending_after_use_status_effects:
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
                self._resolve_pending_attack_relic_proc()
                self._attack_pen_nib_active = False
                return False
            if len(self.discard_pile) == 1:
                self.draw_pile.append(self.discard_pile.pop(0))
                if counter_damage > 0:
                    self._take_counter_damage(counter_damage)
                if pending_after_use_status_effects:
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
                self._resolve_pending_attack_relic_proc()
                self._attack_pen_nib_active = False
                return False
            if len(self.discard_pile) > 1:
                if pending_after_use_energy_gain > 0:
                    self.player.energy = max(0, self.player.energy - pending_after_use_energy_gain)
                    self.pending_after_use_energy_gain += pending_after_use_energy_gain
                if pending_after_use_status_effects:
                    self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
                self.pending_counter_damage = counter_damage
                self.pending_attack_target_index = self.monsters.index(target) if target in self.monsters else None
                self._open_discard_card_select("HEADBUTT", list(range(len(self.discard_pile))), pending_card=None)
                self._attack_pen_nib_active = False
                return True
            if counter_damage > 0:
                self._take_counter_damage(counter_damage)
            if pending_after_use_status_effects:
                self._resolve_deferred_status_draw_effects(pending_after_use_status_effects)
            self._resolve_pending_attack_relic_proc()
            self._attack_pen_nib_active = False
            return False

    def _resolve_pending_monster_block_gains(self) -> None:
            if not self.pending_monster_block_gains:
                return
            pending = self.pending_monster_block_gains
            self.pending_monster_block_gains = []
            for entry in pending:
                require_alive = False
                if len(entry) == 3:
                    monster, amount, require_alive = entry
                else:
                    monster, amount = entry
                if amount > 0 and (monster.alive or not require_alive):
                    monster.block += amount

    def _resolve_pending_monster_direct_damage(self) -> None:
            if not self.pending_monster_direct_damage:
                return
            pending = self.pending_monster_direct_damage
            self.pending_monster_direct_damage = []
            for monster, amount in pending:
                if amount > 0:
                    self._deal_damage_to_monster(amount, monster)

    def _shuffle_cards(self, cards: list[CardInstance]) -> None:
            java_collections_shuffle(cards, self.shuffle_rng.random_long())

    def _spawn_split_child(self, monster_id: str, hp: int) -> MonsterState:
            if monster_id == "AcidSlime_M":
                slime = MonsterState(
                    "AcidSlime_M",
                    "Acid Slime (M)",
                    hp,
                    hp,
                    "ACID_SLIME_M_TACKLE",
                    "ATTACK",
                    move_base_damage=12 if self.ascension_level >= 2 else 10,
                    move_hits=1,
                )
            elif monster_id == "SpikeSlime_M":
                slime = MonsterState(
                    "SpikeSlime_M",
                    "Spike Slime (M)",
                    hp,
                    hp,
                    "SPIKE_SLIME_M_FLAME_TACKLE",
                    "ATTACK_DEBUFF",
                    move_base_damage=10 if self.ascension_level >= 2 else 8,
                    move_hits=1,
                )
            elif monster_id == "AcidSlime_L":
                slime = MonsterState(
                    "AcidSlime_L",
                    "Acid Slime (L)",
                    hp,
                    hp,
                    "ACID_SLIME_L_CORROSIVE_SPIT",
                    "ATTACK_DEBUFF",
                    move_base_damage=12 if self.ascension_level >= 2 else 11,
                    move_hits=1,
                )
            elif monster_id == "SpikeSlime_L":
                slime = MonsterState(
                    "SpikeSlime_L",
                    "Spike Slime (L)",
                    hp,
                    hp,
                    "SPIKE_SLIME_L_FLAME_TACKLE",
                    "ATTACK_DEBUFF",
                    move_base_damage=12 if self.ascension_level >= 2 else 10,
                    move_hits=1,
                )
            else:
                slime = make_monster(monster_id, self.monster_hp_rng, self.ascension_level)
                slime.max_hp = hp
                slime.current_hp = hp
            slime.ai_state["ascension_level"] = self.ascension_level
            choose_next_move(slime, self.ai_rng)
            return slime

    def _start_opening_turn(self) -> None:
            self.turn = 0
            self.monster_damage_this_turn = {}
            self.cards_discarded_this_turn = 0
            self.attack_played_this_turn = 0
            self.skills_played_this_turn = 0
            self.cards_played_this_turn = 0
            self.card_types_played_this_turn = set()
            self.player.energy = self._base_energy()
            self._apply_happy_flower_start_of_turn()
            self._apply_opening_pre_draw_relics()
            self.draw_cards(self._card_draw_per_turn())
            if self.opening_innate_count > self._card_draw_per_turn():
                self.draw_cards(
                    self.opening_innate_count - self._card_draw_per_turn(),
                )
            self._apply_opening_post_draw_relics()
            if self._has_relic("Anchor"):
                self.player.block += 10
            if self._has_relic("Lantern"):
                self.player.energy += 1
            if self._has_relic("Blood Vial"):
                self._heal(2)
            if self._has_relic("Ancient Tea Set"):
                relic = self._relic("Ancient Tea Set")
                if relic is not None and int(relic.get("counter", 0)) > 0:
                    self.player.energy += 2
                    relic["counter"] = 0
            if self._has_relic("Warped Tongs"):
                candidates = [card for card in self.hand if card.card_def.card_type not in {"STATUS", "CURSE"} and card.upgrades <= 0]
                if candidates:
                    self._upgrade_combat_card(self.card_random_rng.choice(candidates))
            if self._has_relic("Gambling Chip") and self.hand:
                self._open_combat_card_select("GAMBLE", [0])

    def _summon_monster(self, monster_id: str, *, max_alive: int = 5) -> bool:
            if len(self._alive_monsters()) >= max_alive:
                return False
            # lightspeed matches a game quirk here: spawned Torch Heads roll
            # monster HP inside construct(), then immediately roll HP again in
            # SpawnTorchHeads(). The second roll is the HP that sticks.
            if monster_id == "TorchHead":
                make_monster(monster_id, self.monster_hp_rng, self.ascension_level)
            summoned = make_monster(monster_id, self.monster_hp_rng, self.ascension_level)
            if self._has_relic("Philosopher's Stone"):
                summoned.add_power("Strength", 1)
            choose_next_move(summoned, self.ai_rng)
            for index, monster in enumerate(self.monsters):
                if monster.monster_id == "INVALID = 0" or monster.is_gone or not monster.alive:
                    self.monsters[index] = summoned
                    break
            else:
                self.monsters.append(summoned)
            for monster in self.monsters:
                setattr(monster, "_group_ref", self.monsters)
            return True

    def _replace_monster_with_invalid_slot(self, monster: MonsterState) -> None:
            try:
                index = self.monsters.index(monster)
            except ValueError:
                return
            placeholder = make_monster("INVALID = 0", self.monster_hp_rng, self.ascension_level)
            placeholder.block = monster.block
            placeholder.powers = dict(monster.powers)
            placeholder.ai_state["ascension_level"] = self.ascension_level
            placeholder.ai_state["spawn_index"] = int(monster.ai_state.get("spawn_index", index))
            self.monsters[index] = placeholder
            for slot_index, current in enumerate(self.monsters):
                current.ai_state["spawn_index"] = slot_index
                setattr(current, "_group_ref", self.monsters)

    def _take_counter_damage(self, amount: int, *, self_damage: bool = False) -> None:
            amount = max(0, int(amount))
            if amount <= 0:
                return
            if self.player.power("Intangible") > 0:
                amount = min(amount, 1)
            blocked = min(self.player.block, amount)
            self.player.block -= blocked
            amount -= blocked
            if amount <= 0:
                return
            if self.player.power("Buffer") > 0:
                self.player.add_power("Buffer", -1)
                return
            if self._has_relic("Tungsten Rod"):
                amount = max(0, amount - 1)
                if amount <= 0:
                    return
            self.player.current_hp = max(0, self.player.current_hp - amount)
            self._sync_red_skull_strength()
            self.hp_lost_this_combat += amount
            self.hp_loss_events_this_combat += 1
            self._on_player_took_damage_cards()
            if amount > 0 and (relic := self._relic("Centennial Puzzle")) is not None and int(relic.get("counter", 0)) == 0:
                relic["counter"] = 1
                end_turn_self_damage = (
                    self_damage
                    and getattr(self, "processing_end_turn_cleanup", False)
                    and getattr(self, "monster_turn_damage_draws_are_end_turn", False)
                )
                if not end_turn_self_damage or self.draw_pile or (
                    len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0)
                ) > 0:
                    self._draw_player_damage_relic_cards(3, end_turn_self_damage=end_turn_self_damage)
            if amount > 0 and self._has_relic("Runic Cube"):
                end_turn_self_damage = (
                    self_damage
                    and getattr(self, "processing_end_turn_cleanup", False)
                    and getattr(self, "monster_turn_damage_draws_are_end_turn", False)
                )
                if not end_turn_self_damage or self.draw_pile or (
                    len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0)
                ) > 0:
                    self._draw_player_damage_relic_cards(1, end_turn_self_damage=end_turn_self_damage)
            self._try_auto_revive_with_fairy_potion()
            if self.player.power("Rupture") > 0 and self_damage:
                self.player.add_power("Strength", self.player.power("Rupture"))
            if self._has_relic("Self-Forming Clay"):
                self.player.add_power("Self-Forming Clay Block", 3)
            self._check_outcome()

    def _take_non_attack_damage(self, amount: int, *, self_damage: bool = False) -> None:
            amount = max(0, int(amount))
            if amount <= 0:
                return
            if self.player.power("Intangible") > 0:
                amount = min(amount, 1)
            if self.player.power("Buffer") > 0:
                self.player.add_power("Buffer", -1)
                return
            if self._has_relic("Tungsten Rod"):
                amount = max(0, amount - 1)
                if amount <= 0:
                    return
            self.player.current_hp = max(0, self.player.current_hp - amount)
            self._sync_red_skull_strength()
            if amount > 0 and (relic := self._relic("Centennial Puzzle")) is not None and int(relic.get("counter", 0)) == 0:
                relic["counter"] = 1
                end_turn_self_damage = (
                    self_damage
                    and getattr(self, "processing_end_turn_cleanup", False)
                    and getattr(self, "monster_turn_damage_draws_are_end_turn", False)
                )
                if not end_turn_self_damage or self.draw_pile or (
                    len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0)
                ) > 0:
                    self._draw_player_damage_relic_cards(3, end_turn_self_damage=end_turn_self_damage)
            if amount > 0 and self._has_relic("Runic Cube"):
                end_turn_self_damage = (
                    self_damage
                    and getattr(self, "processing_end_turn_cleanup", False)
                    and getattr(self, "monster_turn_damage_draws_are_end_turn", False)
                )
                if not end_turn_self_damage or self.draw_pile or (
                    len(self.discard_pile) - int(getattr(self, "end_turn_regular_discard_count", 0) or 0)
                ) > 0:
                    self._draw_player_damage_relic_cards(1, end_turn_self_damage=end_turn_self_damage)
            self._try_auto_revive_with_fairy_potion()
            self._check_outcome()
            if self.player.power("Rupture") > 0 and self_damage:
                self.player.add_power("Strength", self.player.power("Rupture"))
            if self._has_relic("Self-Forming Clay"):
                self.player.add_power("Self-Forming Clay Block", 3)
            self._check_outcome()

    def _top_card_from_draw_pile(self) -> CardInstance | None:
            if not self.draw_pile:
                if not self.discard_pile:
                    return None
                self.draw_pile = list(self.discard_pile)
                self.discard_pile = []
                self._shuffle_cards(self.draw_pile)
            return self.draw_pile.pop()

    def _top_discard_card(self) -> CardInstance | None:
            return self.discard_pile[-1] if self.discard_pile else None

    def _trigger_juggernaut(self, amount: int) -> None:
            if amount <= 0:
                return
            target = self._random_alive_monster()
            if target is None:
                return
            was_alive = target.alive
            dealt = self._apply_damage_to_monster(amount, target)
            self._apply_guardian_mode_shift(target, dealt)
            if was_alive and not target.alive:
                self._on_monster_defeated(target)
            elif was_alive:
                self._maybe_split_slime_boss(target, dealt=dealt)

    def _resolve_pending_juggernaut_damage(self) -> None:
            if self.pending_juggernaut_damage <= 0:
                return
            while self.pending_juggernaut_damage > 0:
                damage = min(self.player.power("Juggernaut"), self.pending_juggernaut_damage)
                self.pending_juggernaut_damage -= damage
                if damage <= 0:
                    self.pending_juggernaut_damage = 0
                    return
                self._trigger_juggernaut(damage)

    def _upgrade_combat_card(self, card: CardInstance) -> None:
            if card.card_id != "Searing Blow" and card.upgrades > 0:
                return
            old_effective_cost = 0 if card.cost_for_turn is not None and card.cost_for_turn < 0 else card.cost
            _increment_card_upgrade(card)
            card._temporary_upgrade = True
            if card.card_id == "Blood for Blood":
                card.misc = 0
            upgraded_cost = card.card_def.upgraded_cost
            base_upgraded_cost = CARD_LIBRARY[card.card_id].upgraded_cost if card.card_id in CARD_LIBRARY else None
            if base_upgraded_cost is not None and (upgraded_cost is None or upgraded_cost == card.card_def.cost):
                upgraded_cost = base_upgraded_cost
            if upgraded_cost is None:
                return
            if card.cost_for_turn is not None:
                if card.cost_for_turn == old_effective_cost:
                    card.cost_for_turn = upgraded_cost
                return
            if card.cost_for_combat is not None and card.cost_for_combat == old_effective_cost:
                card.cost_for_combat = upgraded_cost
                return
            if card.cost == old_effective_cost and card.cost != upgraded_cost:
                card.cost_for_turn = upgraded_cost

    def _resolve_deferred_status_draw_effects(self, effects: list[tuple[str, int | str]]) -> None:
            for effect_kind, amount in effects:
                if effect_kind == "add_to_hand":
                    self._add_to_hand(str(amount))
                    continue
                if amount <= 0:
                    continue
                if effect_kind == "draw":
                    self.draw_cards(int(amount))
                elif effect_kind == "damage_all":
                    self._deal_direct_damage_all(int(amount))
                    self._resolve_pending_monster_kill_triggers()
                    self._check_outcome()
                    if self.outcome != "UNDECIDED":
                        return
                elif effect_kind == "gain_block_no_mod":
                    self._gain_block(amount, apply_block_modifiers=False)

    def draw_cards(
            self,
            count: int,
            *,
            deferred_status_effects: list[tuple[str, int]] | None = None,
            deferred_status_damage_all: list[int] | None = None,
            exclude_end_turn_regular_discards: bool = False,
            suppress_status_draw_triggers: bool = False,
            allow_fire_breathing_when_suppressed: bool = False,
            defer_status_draws_to_start_turn: bool = False,
            after_current_batch_actions: list[tuple[str, int | str]] | None = None,
        ) -> None:
            draw_queue = [count]
            queue_index = 0
            after_current_batch_flushed = False
            while queue_index < len(draw_queue):
                amount = draw_queue[queue_index]
                queue_index += 1
                if amount <= 0 or self.player.power("No Draw") > 0:
                    continue
                drew_any = False
                for _ in range(amount):
                    if len(self.hand) >= 10:
                        break
                    if not self.draw_pile:
                        reserved_regular_discards: list[CardInstance] = []
                        discard_source = self.discard_pile
                        if exclude_end_turn_regular_discards:
                            reserved_count = min(
                                len(self.discard_pile),
                                int(getattr(self, "end_turn_regular_discard_count", 0) or 0),
                            )
                            if reserved_count > 0:
                                split_index = len(self.discard_pile) - reserved_count
                                reserved_regular_discards = list(self.discard_pile[split_index:])
                                discard_source = self.discard_pile[:split_index]
                        if not discard_source:
                            if drew_any:
                                self.shuffle_rng.random_long()
                            break
                        self._draw_triggered_shuffle = True
                        shuffled_source = list(discard_source)
                        java_collections_shuffle(shuffled_source, self.shuffle_rng.random_long())
                        self.draw_pile = shuffled_source
                        self.discard_pile = reserved_regular_discards
                        if self._advance_relic_counter("Sundial", 3):
                            self.player.energy += 2
                    card = self.draw_pile.pop()
                    drew_any = True
                    if self.player.power("Confusion") > 0 and card.card_def.card_type in {"ATTACK", "SKILL", "POWER"} and not card.card_def.x_cost:
                        self._roll_confusion_cost(card)
                    elif self.player.power("Corruption") > 0 and card.card_def.card_type == "SKILL" and card.cost > 0:
                        if card.cost_for_combat is None:
                            card.cost_for_combat = 0
                        card.cost_for_turn = -9
                    elif card.card_id == "Blood for Blood" and card.cost_for_turn is None:
                        base_cost = 3 if card.upgrades > 0 else 4
                        card.cost_for_turn = max(0, base_cost - card.misc)
                    self.hand.append(card)
                    if card.card_id == "Void":
                        self.player.energy = max(0, self.player.energy - 1)
                    if card.card_def.card_type == "STATUS":
                        evolve_draws = self.player.power("Evolve")
                        pending_finesse_hex_generated = int(getattr(self, "pending_finesse_hex_generated_dazed", 0) or 0)
                        if pending_finesse_hex_generated > 0 and evolve_draws > 1:
                            evolve_draws = max(0, evolve_draws - 1)
                            self.pending_finesse_hex_generated_dazed = pending_finesse_hex_generated - 1
                        if not suppress_status_draw_triggers:
                            if evolve_draws > 0:
                                if deferred_status_effects is None:
                                    draw_queue.append(evolve_draws)
                                else:
                                    deferred_status_effects.append(("draw", evolve_draws))
                        elif defer_status_draws_to_start_turn and evolve_draws > 0:
                            self.pending_start_turn_evolve_draws += evolve_draws
                        if (
                            (not suppress_status_draw_triggers or allow_fire_breathing_when_suppressed)
                            and self.player.power("Fire Breathing") > 0
                        ):
                            if deferred_status_damage_all is not None:
                                deferred_status_damage_all.append(self.player.power("Fire Breathing"))
                            elif deferred_status_effects is None:
                                self._deal_direct_damage_all(self.player.power("Fire Breathing"))
                                self._resolve_pending_monster_kill_triggers()
                                self._check_outcome()
                                if self.outcome != "UNDECIDED":
                                    return
                            else:
                                deferred_status_effects.append(("damage_all", self.player.power("Fire Breathing")))
                    elif (
                        card.card_def.card_type == "CURSE"
                        and (not suppress_status_draw_triggers or allow_fire_breathing_when_suppressed)
                    ):
                        if self.player.power("Fire Breathing") > 0:
                            if deferred_status_damage_all is not None:
                                deferred_status_damage_all.append(self.player.power("Fire Breathing"))
                            elif deferred_status_effects is None:
                                self._deal_direct_damage_all(self.player.power("Fire Breathing"))
                                self._resolve_pending_monster_kill_triggers()
                                self._check_outcome()
                                if self.outcome != "UNDECIDED":
                                    return
                            else:
                                deferred_status_effects.append(("damage_all", self.player.power("Fire Breathing")))
                if after_current_batch_actions and not after_current_batch_flushed:
                    self._resolve_deferred_after_exhaust_actions(after_current_batch_actions)
                    after_current_batch_flushed = True

    def playable(self, card: CardInstance, *, free_to_play: bool = False) -> bool:
            if card.card_def.card_type == "STATUS" and card.card_id != "Slimed" and not self._has_relic("Medical Kit"):
                return False
            if card.card_def.card_type == "CURSE" and not self._has_relic("Blue Candle"):
                return False
            if any(other.card_id == "Normality" for other in self.hand) and self.cards_played_this_turn >= 3:
                return False
            if self.player.power("No Attack") > 0 and card.card_def.card_type == "ATTACK":
                return False
            if card.card_id == "Clash" and any(other is not card and other.card_def.card_type != "ATTACK" for other in self.hand):
                return False
            if card.card_id == "Secret Weapon" and not any(draw.card_def.card_type == "ATTACK" for draw in self.draw_pile):
                return False
            if card.card_id == "Secret Technique" and not any(draw.card_def.card_type == "SKILL" for draw in self.draw_pile):
                return False
            if self._has_relic("Velvet Choker") and self.cards_played_this_turn >= 6:
                return False
            if free_to_play or card.free_to_play_once:
                return True
            if card.card_def.x_cost:
                return True
            return self._card_energy_cost(card) <= self.player.energy

    def use_potion(self, potion_index: int, target_index: int = 0) -> None:
            if potion_index < 0 or potion_index >= len(self.potions):
                raise IndexError(potion_index)
            potion = self.potions[potion_index]
            if not potion.can_use:
                raise ValueError(f"potion slot is empty: {potion_index}")
            target = self.monsters[target_index] if self.monsters and target_index < len(self.monsters) else None
            potion_id = potion.potion_id
            potion_multiplier = 2 if self._has_relic("Sacred Bark") else 1
            if potion_id == "Fire Potion":
                if target:
                    was_alive = target.alive
                    dealt = self._apply_damage_to_monster(20 * potion_multiplier, target)
                    if was_alive and not target.alive:
                        self._on_monster_defeated(target)
                    elif was_alive:
                        self._maybe_split_slime_boss(target, dealt=dealt)
            elif potion_id == "Explosive Potion":
                for monster in list(self._alive_monsters()):
                    was_alive = monster.alive
                    dealt = self._apply_damage_to_monster(10 * potion_multiplier, monster)
                    if was_alive and not monster.alive:
                        self._on_monster_defeated(monster)
                    elif was_alive:
                        self._maybe_split_slime_boss(monster, dealt=dealt)
            elif potion_id == "Weak Potion":
                if target:
                    self._apply_monster_power(target, "Weakened", 3 * potion_multiplier)
            elif potion_id == "Fear Potion":
                if target:
                    self._apply_monster_power(target, "Vulnerable", 3 * potion_multiplier)
            elif potion_id == "Strength Potion":
                self.player.add_power("Strength", 2 * potion_multiplier)
            elif potion_id == "Dexterity Potion":
                self.player.add_power("Dexterity", 2 * potion_multiplier)
            elif potion_id == "Block Potion":
                self._gain_block(12 * potion_multiplier)
            elif potion_id == "Energy Potion":
                self.player.energy += 2 * potion_multiplier
            elif potion_id == "Swift Potion":
                self.draw_cards(3 * potion_multiplier)
            elif potion_id == "Blood Potion":
                self._heal(max(1, int(self.player.max_hp * 0.2 * potion_multiplier)))
            elif potion_id == "Flex Potion":
                self.player.add_power("Strength", 5 * potion_multiplier)
                self.player.add_power("Flex Strength Down", 5 * potion_multiplier)
            elif potion_id == "Attack Potion":
                attack_ids = [card.card_id for card in self._ironclad_card_pool(card_type="ATTACK")]
                self._add_to_hand(self.rng.choice(attack_ids), cost_for_turn=0)
            elif potion_id == "Skill Potion":
                skill_ids = [card.card_id for card in self._ironclad_card_pool(card_type="SKILL")]
                self._add_to_hand(self.rng.choice(skill_ids), cost_for_turn=0)
            elif potion_id == "Power Potion":
                power_ids = [card.card_id for card in self._ironclad_card_pool(card_type="POWER")]
                self._add_to_hand(self.rng.choice(power_ids), cost_for_turn=0)
            elif potion_id == "Ancient Potion":
                self.player.add_power("Artifact", 1)
            elif potion_id == "Blessing of the Forge":
                for card in self.hand:
                    if card.card_def.card_type not in {"STATUS", "CURSE"}:
                        _ensure_card_upgraded(card)
            elif potion_id == "Colorless Potion":
                for _ in range(potion_multiplier):
                    self._add_random_card_to_hand(colorless=True, cost_for_turn=0)
            elif potion_id == "Duplication Potion":
                self.player.add_power("Double Tap", potion_multiplier)
            elif potion_id == "ElixirPotion":
                self.exhaust_pile.extend(card for card in self.hand if card.card_def.card_type == "STATUS")
                self.hand = [card for card in self.hand if card.card_def.card_type != "STATUS"]
            elif potion_id == "Essence of Steel":
                self.player.add_power("Plated Armor", 4 * potion_multiplier)
            elif potion_id == "Speed Potion":
                self.player.add_power("Dexterity", 5 * potion_multiplier)
                self._apply_player_power("Dexterity Down", 5 * potion_multiplier)
            elif potion_id == "Steroid Potion":
                self.player.add_power("Strength", 5 * potion_multiplier)
                self._apply_player_power("Flex Strength Down", 5 * potion_multiplier)
            elif potion_id == "Regen Potion":
                self.player.add_power("Regen", 5 * potion_multiplier)
            elif potion_id == "Liquid Bronze":
                self.player.add_power("Thorns", 3 * potion_multiplier)
            elif potion_id == "Liquid Memories":
                if self.discard_pile:
                    card = self.discard_pile.pop()
                    card.cost_for_turn = 0
                    self.hand.append(card) if len(self.hand) < 10 else self.discard_pile.append(card)
            elif potion_id == "Gambler's Brew":
                count = len(self.hand)
                self.discard_pile.extend(self.hand)
                self.hand = []
                self.draw_cards(count)
            elif potion_id == "Entropic Brew":
                self.potions[potion_index] = PotionInstance()
                for index, current in enumerate(self.potions):
                    if not current.can_use:
                        self.potions[index] = roll_potion(self.potion_rng, limited=True)
                if self._has_relic("Toy Ornithopter"):
                    self._heal(5)
                self._check_outcome()
                return
            elif potion_id == "Fruit Juice":
                self.player.max_hp += 5 * potion_multiplier
                self.player.current_hp += 5 * potion_multiplier
            elif potion_id == "Heart of Iron":
                self.player.add_power("Metallicize", 6 * potion_multiplier)
            elif potion_id == "Distilled Chaos":
                for _ in range(3 * potion_multiplier):
                    self._play_random_top_card(force_exhaust=False)
            elif potion_id == "Cultist Potion":
                self.player.add_power("Ritual", 1 * potion_multiplier)
            elif potion_id == "Snecko Oil":
                self.draw_cards(5 * potion_multiplier)
                for card in self.hand:
                    self._roll_confusion_cost(card)
            elif potion_id == "Smoke Bomb":
                if self.floor not in {16, 33, 50, 53, 54}:
                    for monster in self.monsters:
                        monster.current_hp = 0
                        monster.is_gone = True
                    self.outcome = "PLAYER_VICTORY"
            elif potion_id == "FairyPotion":
                self._heal(max(1, int(self.player.max_hp * 0.3 * potion_multiplier)))
            else:
                raise ValueError(f"unsupported potion effect: {potion_id}")
            if self._has_relic("Toy Ornithopter"):
                self._heal(5)
            self.potions[potion_index] = PotionInstance()
            self._check_outcome()
