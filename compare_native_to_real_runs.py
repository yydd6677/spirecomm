#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spirecomm.native_sim import NativeRunEnv


ALIASES = {
    "Bonfire Elementals": "Bonfire Spirits",
    "Hypnotizing Colored Mushrooms": "Mushrooms",
    "Boot": "The Boot",
    "MawBank": "Maw Bank",
    "NeowsBlessing": "Neow's Lament",
    "Addict": "Drug Dealer",
    "WeMeetAgain": "We Meet Again!",
}


def norm(value: Any) -> str:
    text = ALIASES.get(str(value or ""), str(value or ""))
    text = text.replace("_R", "").replace("_G", "").replace("_B", "")
    text = text.replace("Mindbloom", "MindBloom")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def relic_norm(value: Any) -> str:
    text = ALIASES.get(str(value or ""), str(value or ""))
    # Run history can serialize relic counters into names, e.g. "Toxic Egg 2".
    text = re.sub(r"\s+\d+$", "", text)
    return norm(text)


def map_norm(value: Any) -> str:
    text = str(value or "")
    if text == "E_GREEN":
        return "e"
    if text in {"B", "BOSS"}:
        return "boss"
    return norm(text)


def card_label(card: dict[str, Any] | None) -> str:
    if not card:
        return ""
    name = str(card.get("name") or card.get("card_id") or "")
    upgrades = int(card.get("upgrades") or 0)
    if upgrades > 0:
        name = name.rstrip("+")
        return f"{name}+{upgrades}"
    return name


def real_by_floor(items: list[dict[str, Any]], key: str = "floor") -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items or []:
        if key in item:
            result[int(item[key])].append(item)
    return result


def real_card_choice_sets(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for choice in run.get("card_choices") or []:
        floor = int(choice.get("floor", -1))
        picked = str(choice.get("picked") or "")
        offered = list(choice.get("not_picked") or [])
        if picked and picked != "SKIP":
            offered.append(picked)
        result[floor] = {"picked": picked, "offered": offered}
    return result


def real_purchase_by_floor(run: dict[str, Any]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = defaultdict(list)
    for floor, item in zip(run.get("item_purchase_floors") or [], run.get("items_purchased") or []):
        result[int(floor)].append(str(item))
    for floor, item in zip(run.get("items_purged_floors") or [], run.get("items_purged") or []):
        result[int(floor)].append(f"PURGE:{item}")
    return result


def real_relics_by_floor(run: dict[str, Any]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = defaultdict(list)
    for item in run.get("relics_obtained") or []:
        floor = int(item.get("floor", -1))
        relic = str(item.get("key") or "")
        if floor >= 0 and relic:
            result[floor].append(relic)
    for item in run.get("event_choices") or []:
        floor = int(item.get("floor", -1))
        for relic in item.get("relics_obtained") or []:
            relic_name = str(relic or "")
            if floor >= 0 and relic_name:
                result[floor].append(relic_name)
    return result


def choose_matching(actions: list[dict[str, Any]], target: str, *fields: str) -> dict[str, Any] | None:
    target_norm = norm(target)
    for action in actions:
        for field in fields:
            action_norm = norm(action.get(field))
            if action_norm == target_norm:
                return action
    for action in actions:
        for field in fields:
            action_norm = norm(action.get(field))
            if action_norm and (target_norm.startswith(action_norm) or action_norm in target_norm):
                return action
    return None


def real_neow_relic(run: dict[str, Any]) -> str | None:
    bonus = str(run.get("neow_bonus") or "NONE")
    if bonus not in {"RANDOM_COMMON_RELIC", "ONE_RARE_RELIC", "BOSS_RELIC"}:
        return None
    for relic in run.get("relics") or []:
        if relic_norm(relic) == relic_norm("Burning Blood"):
            continue
        return str(relic)
    return None


def compare_neow_relic(
    env: NativeRunEnv,
    run: dict[str, Any],
    native_relic: dict[str, Any],
    *,
    seed: int,
    bonus: str,
    stats: Counter,
    add_example,
) -> None:
    real_relic = real_neow_relic(run)
    if not real_relic:
        return
    native_name = str(native_relic.get("name") or native_relic.get("relic_id") or "")
    if relic_norm(real_relic) == relic_norm(native_name):
        stats["neow_relic_match"] += 1
    else:
        stats["neow_relic_mismatch"] += 1
        add_example("neow_relic", {
            "floor": 0,
            "seed": seed,
            "bonus": bonus,
            "real": real_relic,
            "native": native_name,
            "native_pool_front": {
                tier: list(pool[:8]) for tier, pool in getattr(env, "relic_pools", {}).items()
            },
        })


def apply_real_neow(
    env: NativeRunEnv,
    run: dict[str, Any],
    *,
    seed: int,
    stats: Counter,
    add_example,
) -> None:
    bonus = str(run.get("neow_bonus") or "NONE")
    cost = str(run.get("neow_cost") or "NONE")

    if cost == "TEN_PERCENT_HP_LOSS":
        env.player.max_hp = int(0.9 * env.player.max_hp)
        env.player.current_hp = min(env.player.current_hp, env.player.max_hp)
    elif cost == "NO_GOLD":
        env.gold = 0
    elif cost == "PERCENT_DAMAGE":
        env.player.current_hp = env.player.current_hp // 10 * 7
    elif cost == "LOSE_STARTER_RELIC":
        env.relics = [relic for relic in env.relics if relic.get("relic_id") != "Burning Blood"]
    elif cost == "CURSE":
        env._add_curse_to_deck(uuid="neow-curse-0")

    floor_zero_choice = real_card_choice_sets(run).get(0)

    def take_neow_reward(cards) -> None:
        native_offered = [card.name for card in cards]
        if floor_zero_choice:
            real_offered = floor_zero_choice["offered"]
            if {norm(x) for x in native_offered} == {norm(x) for x in real_offered}:
                stats["neow_card_reward_match"] += 1
            else:
                stats["neow_card_reward_mismatch"] += 1
                add_example("neow", {
                    "floor": 0,
                    "seed": seed,
                    "bonus": bonus,
                    "real": real_offered,
                    "native": native_offered,
                })
            picked = str(floor_zero_choice["picked"])
            chosen = next((card for card in cards if norm(card.name) == norm(picked) or norm(card.card_id) == norm(picked)), None)
            if chosen is not None:
                env._add_card_to_deck(chosen.card_id, upgrades=chosen.upgrades, uuid=f"neow-picked-{chosen.card_id}")
        elif cards:
            env._add_card_to_deck(cards[0].card_id, upgrades=cards[0].upgrades, uuid=f"neow-default-{cards[0].card_id}")

    if bonus == "THREE_CARDS":
        take_neow_reward(env._neow_card_reward(rare_only=False))
    elif bonus == "THREE_RARE_CARDS":
        take_neow_reward(env._neow_card_reward(rare_only=True))
    elif bonus in {"RANDOM_COLORLESS", "RANDOM_COLORLESS_2"}:
        take_neow_reward(env._neow_colorless_card_reward(rare_only=bonus == "RANDOM_COLORLESS_2"))
    elif bonus == "ONE_RANDOM_RARE_CARD":
        card = env._random_class_card_of_rarity_from_rng(env.randoms.neow, "RARE")
        env._add_card_to_deck(card.card_id, uuid=f"neow-rare-{card.card_id}")
    elif bonus == "RANDOM_COMMON_RELIC":
        relic = env._roll_relic_of_tier("COMMON")
        compare_neow_relic(env, run, relic, seed=seed, bonus=bonus, stats=stats, add_example=add_example)
        env._obtain_relic(relic)
    elif bonus == "ONE_RARE_RELIC":
        relic = env._roll_relic_of_tier("RARE")
        compare_neow_relic(env, run, relic, seed=seed, bonus=bonus, stats=stats, add_example=add_example)
        env._obtain_relic(relic)
    elif bonus == "TEN_PERCENT_HP_BONUS":
        gain = int(env.player.max_hp * 0.1)
        env.player.max_hp += gain
        env.player.current_hp += gain
    elif bonus == "TWENTY_PERCENT_HP_BONUS":
        gain = int(env.player.max_hp * 0.2)
        env.player.max_hp += gain
        env.player.current_hp += gain
    elif bonus == "HUNDRED_GOLD":
        env._gain_gold(100)
    elif bonus == "TWO_FIFTY_GOLD":
        env._gain_gold(250)
    elif bonus == "THREE_ENEMY_KILL":
        env._obtain_relic({"relic_id": "Neow's Lament", "name": "Neow's Lament", "tier": "SPECIAL"})
    elif bonus == "REMOVE_CARD":
        index = env._first_purge_index()
        if index is not None:
            env.deck.pop(index)
    elif bonus == "UPGRADE_CARD":
        index = env._first_upgradable_index()
        if index is not None:
            env.deck[index].upgrades = max(1, env.deck[index].upgrades)
    elif bonus == "TRANSFORM_CARD":
        index = env._first_purge_index()
        if index is not None:
            env.deck.pop(index)
            card = env._random_class_card_of_rarity_from_rng(env.randoms.neow, "COMMON")
            env._add_card_to_deck(card.card_id, uuid=f"neow-transform-{card.card_id}")
    elif bonus == "BOSS_RELIC":
        env.relics = [relic for relic in env.relics if relic.get("relic_id") != "Burning Blood"]
        relic = env._roll_boss_relics(count=1)[0]
        compare_neow_relic(env, run, relic, seed=seed, bonus=bonus, stats=stats, add_example=add_example)
        env._obtain_relic(relic)


def remove_relic_from_pools(env: NativeRunEnv, relic_name: str) -> None:
    target = relic_norm(relic_name)
    for pool in getattr(env, "relic_pools", {}).values():
        for index, relic_id in enumerate(list(pool)):
            if relic_norm(relic_id) == target:
                del pool[index]
                return


def force_last_gained_relics(env: NativeRunEnv, real_relics: list[str], previous_relic_count: int) -> None:
    current = list(getattr(env, "relics", []))[:previous_relic_count]
    env.relics = current
    for relic_name in real_relics:
        remove_relic_from_pools(env, relic_name)
        env._obtain_relic({"relic_id": relic_name, "name": relic_name, "tier": "FORCED"})


def split_card_name_and_upgrades(card_name: str) -> tuple[str, int]:
    text = str(card_name or "").strip()
    match = re.match(r"^(.*?)(?:\+(\d+))?$", text)
    if not match:
        return text, 0
    name = match.group(1).strip()
    upgrades = int(match.group(2) or 0)
    return name, upgrades


def force_last_reward_card(env: NativeRunEnv, picked: str, previous_deck_size: int) -> None:
    env.deck = list(getattr(env, "deck", []))[:previous_deck_size]
    picked = str(picked or "")
    if not picked or picked == "SKIP":
        return
    card_id, upgrades = split_card_name_and_upgrades(picked)
    env._add_card_to_deck(card_id, upgrades=upgrades, uuid=f"forced-reward-{env.floor}-{card_id}")


def compare_relic_gain(
    *,
    env: NativeRunEnv,
    seed: int,
    real_by_floor: dict[int, list[str]],
    floor: int,
    before_relic_count: int,
    stats: Counter,
    add_example,
) -> None:
    real_relics = list(real_by_floor.get(floor) or [])
    native_relics = [
        str(relic.get("name") or relic.get("relic_id") or "")
        for relic in list(getattr(env, "relics", []))[before_relic_count:]
    ]
    if not real_relics and not native_relics:
        return
    if {relic_norm(item) for item in real_relics} == {relic_norm(item) for item in native_relics}:
        stats["relic_obtained_match"] += 1
        return
    stats["relic_obtained_mismatch"] += 1
    add_example("relic_obtained", {
        "floor": floor,
        "seed": seed,
        "real": real_relics,
        "native": native_relics,
    })
    if real_relics:
        force_last_gained_relics(env, real_relics, before_relic_count)


def choose_map_action_with_lookahead(
    env: NativeRunEnv,
    actions: list[dict[str, Any]],
    path: list[Any],
    next_floor: int,
    target: str,
) -> dict[str, Any] | None:
    matching = [action for action in actions if map_norm(action.get("symbol")) == map_norm(target)]
    if not matching:
        return None

    def score(action: dict[str, Any]) -> int:
        node_id = str(action.get("node_id") or "")
        score_value = 1
        floor = next_floor
        while True:
            floor += 1
            expected = path[floor - 1] if 0 <= floor - 1 < len(path) else None
            if expected is None:
                return score_value
            children = list(env.map_graph.get(node_id, {}).get("children", []))
            if not children:
                return score_value
            child = next(
                (
                    child_id for child_id in children
                    if map_norm(env.map_graph.get(child_id, {}).get("symbol")) == map_norm(expected)
                ),
                None,
            )
            if child is None:
                return score_value
            score_value += 1
            node_id = child

    return max(matching, key=score)


def replay_distribution(
    run: dict[str, Any],
    *,
    max_steps: int = 2000,
    ironclad_unlock_level: int = 5,
    ironclad_relic_unlock_level: int = 5,
) -> dict[str, Any]:
    seed = int(run["seed_played"])
    ascension = int(run.get("ascension_level") or 0)
    enable_act4_keys = bool(
        run.get("green_key_taken_log")
        or run.get("blue_key_relic_skipped_log")
        or run.get("red_key")
        or run.get("has_emerald_key")
        or run.get("has_sapphire_key")
        or run.get("has_ruby_key")
    )
    env = NativeRunEnv(
        seed=seed,
        ascension_level=ascension,
        ironclad_unlock_level=ironclad_unlock_level,
        ironclad_relic_unlock_level=ironclad_relic_unlock_level,
        enable_act4_keys=enable_act4_keys,
        start_on_map=True,
    )
    card_choices = real_card_choice_sets(run)
    events = real_by_floor(run.get("event_choices") or [])
    campfires = real_by_floor(run.get("campfire_choices") or [])
    purchases = real_purchase_by_floor(run)
    relics_by_floor = real_relics_by_floor(run)
    boss_relics = list(run.get("boss_relics") or [])
    # ``path_taken`` is the actual map node symbol. ``path_per_floor`` is the
    # resolved room result, so a question mark that becomes a monster is stored
    # as ``M`` there. Map replay must follow the former or it will branch early.
    path = list(run.get("path_taken") or run.get("path_per_floor") or [])
    stats = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shop_purchase_cursor: dict[int, int] = defaultdict(int)
    compared_relic_floors: set[tuple[str, int]] = set()

    def add_example(kind: str, payload: dict[str, Any]) -> None:
        if len(examples[kind]) < 5:
            examples[kind].append(payload)

    def advance(action: dict[str, Any]) -> None:
        phase = env.phase
        floor = env.floor
        before_relic_count = len(getattr(env, "relics", []))
        env.step(action)
        compare_key = (phase, floor)
        should_compare = False
        if phase == "COMBAT":
            should_compare = env.phase != "COMBAT"
        elif phase == "CHEST":
            should_compare = env.phase != "CHEST"
        elif phase == "EVENT":
            should_compare = env.phase != "EVENT" or len(getattr(env, "relics", [])) != before_relic_count
        if (
            phase in {"COMBAT", "CHEST", "EVENT"}
            and compare_key not in compared_relic_floors
            and should_compare
            and (relics_by_floor.get(floor) or len(getattr(env, "relics", [])) != before_relic_count)
        ):
            compared_relic_floors.add(compare_key)
            compare_relic_gain(
                env=env,
                seed=seed,
                real_by_floor=relics_by_floor,
                floor=floor,
                before_relic_count=before_relic_count,
                stats=stats,
                add_example=add_example,
            )

    apply_real_neow(env, run, seed=seed, stats=stats, add_example=add_example)

    for _step in range(max_steps):
        if env.phase in {"GAME_OVER", "COMPLETE"}:
            break
        if env.floor >= int(run.get("floor_reached") or 0):
            break
        actions = env.legal_actions()
        if not actions:
            break

        if env.phase == "COMBAT":
            if env.combat.outcome == "UNDECIDED":
                env.combat.outcome = "PLAYER_VICTORY"
            advance({"kind": "end", "name": "RESOLVE_COMBAT", "action_index": 0, "bits": 0})
            continue

        if env.phase == "CARD_REWARD":
            real = card_choices.get(env.floor)
            before_deck_size = len(getattr(env, "deck", []))
            native_offered = [
                card_label(action.get("card") or action)
                for action in actions
                if action.get("kind") == "card_reward"
            ]
            if real:
                real_offered = real["offered"]
                if {norm(x) for x in native_offered} == {norm(x) for x in real_offered}:
                    stats["card_reward_match"] += 1
                else:
                    stats["card_reward_mismatch"] += 1
                    add_example("card_reward", {
                        "floor": env.floor,
                        "seed": seed,
                        "real": real_offered,
                        "native": native_offered,
                    })
                picked = str(real["picked"])
                action = choose_matching(actions, picked, "name", "card_id") if picked != "SKIP" else None
                advance(action or actions[-1])
                force_last_reward_card(env, picked, before_deck_size)
            else:
                advance(actions[-1])
            continue

        if env.phase == "MAP":
            next_floor = env.floor + 1
            target = path[next_floor - 1] if 0 <= next_floor - 1 < len(path) else None
            action = choose_map_action_with_lookahead(env, actions, path, next_floor, target) if target else None
            if target and action is None:
                stats["map_symbol_mismatch"] += 1
                add_example("map", {
                    "floor": next_floor,
                    "seed": seed,
                    "real": target,
                    "native_options": [a.get("symbol") for a in actions],
                })
            elif target:
                stats["map_symbol_match"] += 1
            advance(action or actions[0])
            continue

        if env.phase == "EVENT":
            real_list = events.get(env.floor) or []
            real_event = real_list[0] if real_list else None
            native_event = actions[0].get("event_id") if actions else None
            if real_event:
                if norm(real_event.get("event_name")) == norm(native_event):
                    stats["event_match"] += 1
                else:
                    stats["event_mismatch"] += 1
                    add_example("event", {
                        "floor": env.floor,
                        "seed": seed,
                        "real": real_event.get("event_name"),
                        "native": native_event,
                    })
                action = choose_matching(actions, real_event.get("player_choice"), "name", "label")
                advance(action or actions[0])
            else:
                advance(actions[0])
            continue

        if env.phase == "CAMPFIRE":
            real_list = campfires.get(env.floor) or []
            real = real_list[0] if real_list else None
            action = choose_matching(actions, real.get("key") if real else None, "name") if real else None
            if real and action is None:
                stats["campfire_mismatch"] += 1
                add_example("campfire", {
                    "floor": env.floor,
                    "seed": seed,
                    "real": real.get("key"),
                    "native_options": [a.get("name") for a in actions],
                })
            elif real:
                stats["campfire_match"] += 1
            advance(action or actions[0])
            continue

        if env.phase == "BOSS_RELIC":
            real = boss_relics.pop(0) if boss_relics else None
            before_relic_count = len(getattr(env, "relics", []))
            native = [action.get("name") for action in actions]
            if real:
                real_offered = list(real.get("not_picked") or [])
                if real.get("picked"):
                    real_offered.append(real["picked"])
                if {norm(x) for x in native} == {norm(x) for x in real_offered}:
                    stats["boss_relic_match"] += 1
                else:
                    stats["boss_relic_mismatch"] += 1
                    add_example("boss_relic", {
                        "floor": env.floor,
                        "seed": seed,
                        "real": real_offered,
                        "native": native,
                    })
                picked = str(real.get("picked") or "")
                action = choose_matching(actions, picked, "name", "item_id", "relic_id")
                advance(action or actions[0])
                if picked:
                    force_last_gained_relics(env, [picked], before_relic_count)
            else:
                advance(actions[0])
            continue

        if env.phase == "SHOP":
            shop_actions = list(getattr(env, "shop_items", None) or actions)
            floor_purchases = purchases.get(env.floor) or []
            cursor = shop_purchase_cursor[env.floor]
            target = floor_purchases[cursor] if cursor < len(floor_purchases) else None
            action = None
            if target:
                if target.startswith("PURGE:"):
                    action = next((a for a in shop_actions if a.get("item_kind") == "purge"), None)
                else:
                    action = choose_matching(shop_actions, target, "name", "item_id")
                if action:
                    shop_purchase_cursor[env.floor] += 1
                    env.gold = max(env.gold, int(action.get("price", 0) or 0))
                else:
                    stats["shop_purchase_mismatch"] += 1
                    add_example("shop", {
                        "floor": env.floor,
                        "seed": seed,
                        "real": target,
                        "native_options": [a.get("name") for a in shop_actions],
                    })
            if action is None:
                action = next((a for a in actions if a.get("item_kind") == "leave"), actions[0])
            advance(action)
            continue

        if env.phase == "CHEST":
            advance(actions[0])
            continue

        advance(actions[0])

    return {
        "seed": seed,
        "floor_reached": run.get("floor_reached"),
        "native_floor": env.floor,
        "stats": dict(stats),
        "examples": dict(examples),
    }


def iter_runs(run_dir: Path):
    for path in sorted(run_dir.glob("*.run"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("character_chosen") == "IRONCLAD" and data.get("seed_played"):
            yield path, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native simulator distribution against real StS run history.")
    parser.add_argument("--run-dir", type=Path, default=Path("/home/yydd/sts_instances/sts1/game/runs/IRONCLAD"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--ironclad-unlock-level",
        type=int,
        default=5,
        help="Ironclad unlock bundle count to use for replay. 0 excludes all unlock cards; 5 means fully unlocked.",
    )
    parser.add_argument(
        "--ironclad-relic-unlock-level",
        type=int,
        default=5,
        help="Ironclad/global relic unlock bundle count to use for replay. 0 excludes locked relic bundles observed in early save data.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    aggregate = Counter()
    reports = []
    for index, (path, run) in enumerate(iter_runs(args.run_dir)):
        if index >= args.limit:
            break
        report = replay_distribution(
            run,
            max_steps=args.max_steps,
            ironclad_unlock_level=args.ironclad_unlock_level,
            ironclad_relic_unlock_level=args.ironclad_relic_unlock_level,
        )
        report["path"] = str(path)
        reports.append(report)
        aggregate.update(report["stats"])
        print(
            f"run {index + 1}/{args.limit} seed={report['seed']} "
            f"real_floor={report['floor_reached']} native_floor={report['native_floor']} "
            f"stats={report['stats']}",
            flush=True,
        )
    print("summary", dict(aggregate))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": dict(aggregate), "runs": reports}, ensure_ascii=False, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
