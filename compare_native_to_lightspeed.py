#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

from spirecomm.native_sim.env import NativeRunEnv


def compare_seed(seed: int):
    import slaythespire as sts

    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    native = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=False, start_on_map=True)

    ls_map_env = sts.ModelDrivenEnv(seed, 0)
    neow_to_map = None
    for candidate in sts.get_external_actions(ls_map_env.game_context):
        trial = sts.ModelDrivenEnv(seed, 0)
        sts.execute_action_bits(trial, candidate["bits"])
        if str(trial.screen_state).endswith("MAP_SCREEN"):
            neow_to_map = candidate
            ls_map_env = trial
            break
    if neow_to_map is None:
        raise RuntimeError(f"Could not find a direct-to-map Neow option for seed {seed}")
    ls_map_actions = [action for action in sts.get_external_actions(ls_map_env.game_context) if action["kind"] == "map"]
    native_map_actions = [action for action in native.legal_actions() if action["kind"] == "map"]

    ls_first_reward = [card["card_id"] for card in gc.debug_create_card_reward(sts.Room.MONSTER)]
    native.floor = 1
    native.current_node_symbol = "M"
    native_first_reward = [card.card_id for card in native._roll_card_reward(count=3, room="M")]

    first_ls_map = ls_map_actions[0]
    sts.execute_action_bits(ls_map_env, first_ls_map["bits"])
    ls_battle_state = sts.get_battle_state(ls_map_env)
    ls_first_monsters = [monster["monster_id"] for monster in ls_battle_state["combat_state"]["monsters"]]

    native = NativeRunEnv(seed=seed, ascension_level=0, enable_neow=False, start_on_map=True)
    native.step(native_map_actions[0])
    native_first_monsters = [
        monster["monster_id"] for monster in native.combat.to_spirecomm_state()["combat_state"]["monsters"]
    ]

    return {
        "seed": seed,
        "lightspeed_boss": str(gc.boss).split(".")[-1],
        "native_boss": native._act_bosses[1],
        "boss_match": native._act_bosses[1] in {"Hexaghost", "Slime Boss", "The Guardian"} and (
            {
                "Hexaghost": "HEXAGHOST",
                "Slime Boss": "SLIME_BOSS",
                "The Guardian": "THE_GUARDIAN",
            }[native._act_bosses[1]]
            == str(gc.boss).split(".")[-1]
        ),
        "lightspeed_monster_list": str(gc).split("monsterList: offset(0) {", 1)[1].split("\n", 1)[0].strip(),
        "native_monster_list": native.monster_list[:16],
        "lightspeed_elite_list": str(gc).split("eliteMonsterList: offset(0) {", 1)[1].split("\n", 1)[0].strip(),
        "native_elite_list": native.elite_monster_list[:10],
        "map_actions_match": [(a["symbol"], a["x"]) for a in ls_map_actions] == [(a["symbol"], a["x"]) for a in native_map_actions],
        "lightspeed_map_actions": [(a["symbol"], a["x"]) for a in ls_map_actions],
        "native_map_actions": [(a["symbol"], a["x"]) for a in native_map_actions],
        "first_reward_match": ls_first_reward == native_first_reward,
        "lightspeed_first_reward": ls_first_reward,
        "native_first_reward": native_first_reward,
        "first_monsters_match": ls_first_monsters == native_first_monsters,
        "lightspeed_first_monsters": ls_first_monsters,
        "native_first_monsters": native_first_monsters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native simulator against lightspeed for selected seeds.")
    parser.add_argument("--seed", type=str, default=None, help="Game seed string (e.g. 17N212GSYK7DR)")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    import slaythespire as sts

    seeds: list[int] = []
    if args.seed:
        seeds = [sts.get_seed_long(args.seed)]
    else:
        rng = random.Random(0)
        seeds = [rng.randint(0, (1 << 63) - 1) for _ in range(max(1, args.count))]

    for seed in seeds:
        result = compare_seed(seed)
        print(result)


if __name__ == "__main__":
    main()
