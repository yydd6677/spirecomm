from __future__ import annotations

import math
import random

from spirecomm.native_sim.schema import MonsterState, PlayerState
from spirecomm.native_sim.randoms import java_collections_shuffle

ACT_BOSSES = {
    1: ["TheGuardian", "Hexaghost", "SlimeBoss"],
    2: ["BronzeAutomaton", "TheCollector", "TheChamp"],
    3: ["AwakenedOne", "TimeEater", "DonuDeca"],
}

WEAK_ENCOUNTERS = {
    1: ["Cultist", "JawWorm", "Louses", "SmallSlimes"],
    2: ["SphericGuardian", "Chosen", "ShelledParasite", "ByrdTriple", "Thieves"],
    3: ["Darklings", "OrbWalker", "ShapesTriple"],
}

WEAK_WEIGHTS = {
    1: [1.0 / 4] * 4,
    2: [1.0 / 5] * 5,
    3: [1.0 / 3] * 3,
}

STRONG_ENCOUNTERS = {
    1: [
        "GremlinGang",
        "LotsOfSlimes",
        "SlaverRed",
        "ExordiumThugs",
        "ExordiumWildlife",
        "SlaverBlue",
        "Looter",
        "LargeSlime",
        "ThreeLouses",
        "FungiPair",
    ],
    2: [
        "ChosenAndByrds",
        "SentryAndSphere",
        "CultistAndChosen",
        "ThreeCultists",
        "ShelledParasiteAndFungi",
        "Snecko",
        "SnakePlant",
        "CenturionMystic",
    ],
    3: [
        "SpireGrowth",
        "Transient",
        "ShapesQuad",
        "TheMaw",
        "SphereAndShapes",
        "JawWormHorde",
        "Darklings",
        "WrithingMass",
    ],
}

STRONG_WEIGHTS = {
    1: [1.0 / 16, 1.0 / 16, 1.0 / 16, 1.5 / 16, 1.5 / 16, 2.0 / 16, 2.0 / 16, 2.0 / 16, 2.0 / 16, 2.0 / 16],
    2: [2.0 / 29, 2.0 / 29, 3.0 / 29, 3.0 / 29, 3.0 / 29, 4.0 / 29, 6.0 / 29, 6.0 / 29],
    3: [1.0 / 8] * 8,
}

ELITE_ENCOUNTERS = {
    1: ["GremlinNob", "Lagavulin", "ThreeSentries"],
    2: ["GremlinLeader", "Slavers", "BookOfStabbing"],
    3: ["GiantHead", "Nemesis", "Reptomancer"],
}

def _roll_weighted_idx(roll: float, weights: list[float]) -> int:
    total = 0.0
    for index, weight in enumerate(weights):
        total += weight
        if roll < total:
            return index
    return max(0, len(weights) - 1)

def _rng_ascension_level(rng: random.Random) -> int:
    return int(getattr(rng, "ascension_level", 0))

def _roll_louse(rng: random.Random) -> str:
    return "RedLouse" if rng.random_boolean() else "GreenLouse"

def _populate_monster_list(rng: random.Random, encounters: list[str], weights: list[float], count: int, result: list[str] | None = None) -> list[str]:
    result = [] if result is None else result
    starting_size = len(result)
    for _ in range(count):
        while True:
            idx = _roll_weighted_idx(rng.random(), weights)
            encounter = encounters[idx]
            if len(result) == starting_size and starting_size == 0:
                result.append(encounter)
                break
            if encounter != result[-1] and (len(result) < 2 or encounter != result[-2]):
                result.append(encounter)
                break
    return result

def _populate_first_strong_enemy(rng: random.Random, act: int, current: list[str], previous_tail: str | None = None) -> None:
    encounters = STRONG_ENCOUNTERS[act]
    weights = STRONG_WEIGHTS[act]
    last_monster = previous_tail if previous_tail is not None else (current[-1] if current else None)
    while True:
        idx = _roll_weighted_idx(rng.random(), weights)
        encounter = encounters[idx]
        if act == 1:
            if encounter in {"LargeSlime", "LotsOfSlimes"} and last_monster == "SmallSlimes":
                continue
            if encounter == "ThreeLouses" and last_monster == "Louses":
                continue
        current.append(encounter)
        return

def generate_monster_schedules(rng: random.Random, act: int, ascension_level: int = 0) -> tuple[list[str], list[str], str, str | None]:
    weak = _populate_monster_list(rng, WEAK_ENCOUNTERS[act], WEAK_WEIGHTS[act], 3 if act == 1 else 2)
    strong = weak + generate_strong_monster_schedule(rng, act, previous_tail=weak[-1] if weak else None)

    elites: list[str] = []
    elites = generate_elite_schedule(rng, act)

    bosses = list(ACT_BOSSES[act])
    java_collections_shuffle(bosses, rng.random_long())
    second_boss = bosses[1] if act == 3 and ascension_level >= 20 else None
    return strong, elites, bosses[0], second_boss

def generate_strong_monster_schedule(rng: random.Random, act: int, previous_tail: str | None = None) -> list[str]:
    strong: list[str] = []
    _populate_first_strong_enemy(rng, act, strong, previous_tail=previous_tail)
    _populate_monster_list(rng, STRONG_ENCOUNTERS[act], STRONG_WEIGHTS[act], 12, strong)
    return strong

def generate_elite_schedule(rng: random.Random, act: int) -> list[str]:
    elites: list[str] = []
    for _ in range(10):
        while True:
            roll = rng.random()
            if roll < 1.0 / 3.0:
                encounter = ELITE_ENCOUNTERS[act][0]
            elif roll < 2.0 / 3.0:
                encounter = ELITE_ENCOUNTERS[act][1]
            else:
                encounter = ELITE_ENCOUNTERS[act][2]
            if not elites or encounter != elites[-1]:
                elites.append(encounter)
                break
    return elites

def monster_adjusted_damage(monster: MonsterState, player: PlayerState, *, vulnerable_multiplier: float = 1.5) -> int:
    if monster.move_base_damage <= 0 or monster.move_hits <= 0:
        return 0
    damage = float(monster.move_base_damage + monster.power("Strength"))
    if monster.power("Weakened") > 0:
        damage *= 0.75
    if player.power("Vulnerable") > 0:
        damage *= vulnerable_multiplier
    return max(0, math.floor(damage))

def _advance_ai_counter(monster: MonsterState, key: str) -> int:
    monster.ai_state[key] = int(monster.ai_state.get(key, 0)) + 1
    return monster.ai_state[key]

def _monster_ascension(monster: MonsterState) -> int:
    return int(monster.ai_state.get("ascension_level", 0))

def _last_move(monster: MonsterState, move: str) -> bool:
    return bool(monster.move_history) and monster.move_history[0] == move

def _last_move_before(monster: MonsterState, move: str) -> bool:
    return len(monster.move_history) >= 2 and monster.move_history[1] == move

def _last_two_moves(monster: MonsterState, move: str) -> bool:
    return len(monster.move_history) >= 2 and monster.move_history[0] == move and monster.move_history[1] == move

def _cycle_move(monster: MonsterState, cycle: list[str]) -> str:
    index = int(monster.ai_state.get("cycle_index", 0))
    monster.ai_state["cycle_index"] = index + 1
    return cycle[index % len(cycle)]

def _random_move(monster: MonsterState, rng: random.Random, moves: list[str]) -> str:
    if not moves:
        raise ValueError("moves must not be empty")
    if len(moves) == 1:
        return moves[0]
    last_move = monster.move_history[0] if monster.move_history else monster.move
    candidates = [move for move in moves if move != last_move]
    return rng.choice(candidates or moves)

def _set_move(monster: MonsterState, move: str) -> None:
    previous = monster.move_history[0] if monster.move_history else None
    if previous is None:
        monster.move_history = [move]
    else:
        monster.move_history = [move, previous]
    monster.move = move
    monster.move_hits = 1
    if move == "JAW_WORM_CHOMP":
        monster.intent = "ATTACK"
        monster.move_base_damage = 12 if _monster_ascension(monster) >= 2 else 11
        monster.move_hits = 1
        return
    if move == "JAW_WORM_BELLOW":
        monster.intent = "DEFEND_BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "JAW_WORM_THRASH":
        monster.intent = "ATTACK_DEFEND"
        monster.move_base_damage = 7
        monster.move_hits = 1
        return
    if move == "SPHERIC_GUARDIAN_HARDEN":
        monster.intent = "ATTACK_DEFEND"
        monster.move_base_damage = 11 if _monster_ascension(monster) >= 2 else 10
        monster.move_hits = 1
        return
    if move == "BYRD_STUNNED":
        monster.intent = "STUN"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "SHELLED_STUNNED":
        monster.intent = "STUN"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move in {"ACID_SLIME_L_SPLIT", "SPIKE_SLIME_L_SPLIT", "SLIME_BOSS_SPLIT"}:
        monster.intent = "MAGIC" if move == "SLIME_BOSS_SPLIT" else "UNKNOWN"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "LAGAVULIN_SLEEP":
        monster.intent = "SLEEP"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "LAGAVULIN_STUN":
        monster.intent = "STUN"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "LOOTER_SMOKE_BOMB":
        monster.intent = "DEFEND"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "MUGGER_SMOKE_BOMB":
        monster.intent = "DEFEND"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move in {"LOOTER_ESCAPE", "MUGGER_ESCAPE"}:
        monster.intent = "ESCAPE"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move in {
        "BYRD_CAW", "BYRD_FLY", "MYSTIC_BUFF",
        "GREMLIN_LEADER_RALLY", "CHAMP_DEFENSIVE_STANCE", "CHAMP_GLOAT", "CHAMP_ANGER", "COLLECTOR_BUFF", "THE_COLLECTOR_SPAWN", "BRONZE_AUTOMATON_SPAWN_ORBS",
        "DARKLING_REINCARNATE", "ORB_WALKER_CHARGE", "MAW_ROAR", "SPIKER_GROW",
        "GIANT_HEAD_COUNT", "REPTOMANCER_SUMMON", "AWAKENED_REBIRTH", "DONU_CIRCLE_OF_POWER",
        "RED_LOUSE_GROW",
        "CULTIST_INCANTATION", "GREMLIN_NOB_BELLOW", "FUNGI_BEAST_GROW",
    }:
        if move in {"GREMLIN_LEADER_RALLY", "THE_COLLECTOR_SPAWN", "BRONZE_AUTOMATON_SPAWN_ORBS"}:
            monster.intent = "MAGIC"
        elif move in {"CHAMP_DEFENSIVE_STANCE", "COLLECTOR_BUFF"}:
            monster.intent = "DEFEND_BUFF"
        else:
            monster.intent = "BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "GREMLIN_LEADER_ENCOURAGE":
        monster.intent = "DEFEND_BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "BRONZE_AUTOMATON_BOOST":
        monster.intent = "DEFEND_BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "CULTIST_DARK_STRIKE":
        monster.intent = "ATTACK"
        monster.move_base_damage = 6
        monster.move_hits = 1
        return
    if move == "GREMLIN_WIZARD_CHARGING":
        monster.intent = "MAGIC"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "MYSTIC_HEAL":
        monster.intent = "MAGIC"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move in {
        "SPHERIC_GUARDIAN_ACTIVATE", "CENTURION_DEFEND", "DECA_SQUARE_OF_PROTECTION",
        "SHIELD_FORTIFY", "SHIELD_GREMLIN_PROTECT", "BRONZE_ORB_SUPPORT_BEAM",
    }:
        monster.intent = "DEFEND"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move == "BRONZE_AUTOMATON_STUNNED":
        monster.intent = "STUN"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move in {
        "CHOSEN_DRAIN", "SNAKE_PLANT_ENFEEBLING_SPORES", "SNECKO_PERPLEXING_GLARE",
        "COLLECTOR_MEGA_DEBUFF", "MAW_DROOL", "WRITHING_IMPLANT", "WRITHING_WITHER",
        "REPULSOR_REPULSE", "NEMESIS_DEBUFF", "TIME_EATER_RIPPLE",
        "HEART_DEBILITATE", "LAGAVULIN_SIPHON_SOUL", "BEAR_BEAR_HUG", "GREEN_LOUSE_SPIT_WEB", "CHAMP_TAUNT",
    }:
        monster.intent = "STRONG_DEBUFF" if move in {"CHOSEN_DRAIN", "COLLECTOR_MEGA_DEBUFF", "LAGAVULIN_SIPHON_SOUL", "HEART_DEBILITATE"} else "DEBUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move in {
        "TASKMASTER_RAKE", "CHAMP_FACE_SLAP", "MYSTIC_ATTACK",
        "WRITHING_FLAIL", "SPEAR_BURN_STRIKE", "RED_SLAVER_SCRAPE",
        "ROMEO_AGONIZING_SLASH", "ROMEO_MOCK", "SPHERIC_GUARDIAN_ATTACK_DEBUFF", "CHOSEN_DEBILITATE",
        "SNECKO_TAIL_WHIP",
    }:
        monster.intent = "ATTACK_DEBUFF"
        monster.move_base_damage = {
            "TASKMASTER_RAKE": 12,
            "CHAMP_FACE_SLAP": 14 if _monster_ascension(monster) >= 4 else 12,
            "MYSTIC_ATTACK": 9 if _monster_ascension(monster) >= 2 else 8,
            "WRITHING_FLAIL": 15,
            "SPEAR_BURN_STRIKE": 6,
            "RED_SLAVER_SCRAPE": 8,
            "ROMEO_AGONIZING_SLASH": 12,
            "ROMEO_MOCK": 6,
            "SPHERIC_GUARDIAN_ATTACK_DEBUFF": 11 if _monster_ascension(monster) >= 2 else 10,
            "CHOSEN_DEBILITATE": 12 if _monster_ascension(monster) >= 2 else 10,
            "SNECKO_TAIL_WHIP": 10 if _monster_ascension(monster) >= 2 else 8,
        }[move]
        if move == "SPEAR_BURN_STRIKE":
            monster.move_hits = 2
        else:
            monster.move_hits = 1
        return
    elif move in {"BLUE_SLAVER_RAKE"}:
        monster.intent = "ATTACK_DEBUFF"
        monster.move_base_damage = 7
        monster.move_hits = 1
        return
    elif move in {"BLUE_SLAVER_STAB", "RED_LOUSE_BITE", "GREEN_LOUSE_BITE", "FUNGI_BEAST_BITE", "SHELLED_FELL"}:
        monster.intent = "ATTACK"
        if move == "BLUE_SLAVER_STAB":
            monster.move_base_damage = 12
        elif move == "SHELLED_FELL":
            monster.move_base_damage = 21 if _monster_ascension(monster) >= 2 else 18
        elif move == "FUNGI_BEAST_BITE":
            monster.move_base_damage = 6
        else:
            monster.move_base_damage = int(monster.ai_state.get("bite_damage", 0))
        monster.move_hits = 1
        return
    elif move in {
        "BYRD_PECK", "BYRD_HEADBUTT", "BYRD_SWOOP", "CHOSEN_POKE", "SPHERIC_GUARDIAN_SLAM",
        "SPHERIC_GUARDIAN_HARDEN", "SPHERIC_GUARDIAN_ATTACK_DEBUFF", "SNAKE_PLANT_CHOMP",
        "SNECKO_BITE", "SHELLED_DOUBLE_STRIKE", "CENTURION_SLASH",
        "CENTURION_FURY", "BOOK_MULTI_STAB", "BOOK_OF_STABBING_MULTI_STAB", "BOOK_BIG_STAB", "BOOK_OF_STABBING_SINGLE_STAB",
        "GREMLIN_LEADER_STAB", "TASKMASTER_SCOURING_WHIP", "CHAMP_HEAVY_SLASH", "CHAMP_EXECUTE",
        "COLLECTOR_FIREBALL", "BRONZE_AUTOMATON_FLAIL", "BRONZE_AUTOMATON_HYPER_BEAM", "DARKLING_NIP",
        "DARKLING_CHOMP", "ORB_WALKER_LASER", "ORB_WALKER_CLAW", "MAW_SLAM", "MAW_NOM",
        "TRANSIENT_ATTACK", "SPIKER_CUT", "EXPLODER_SLAM", "EXPLODER_EXPLODE", "REPULSOR_ATTACK",
        "NEMESIS_ATTACK", "NEMESIS_SCYTHE", "GIANT_HEAD_IT_IS_TIME", "GIANT_HEAD_ATTACK",
        "REPTOMANCER_STAB", "REPTOMANCER_BIG_STAB", "AWAKENED_SLASH", "AWAKENED_SOUL_STRIKE",
        "TIME_EATER_REVERBERATE", "TIME_EATER_HEAD_SLAM", "DONU_BEAM", "DECA_BEAM",
        "SHIELD_BASH", "SHIELD_SMASH", "SPEAR_SKEWER", "SPEAR_PIERCER", "HEART_BLOOD_SHOTS", "HEART_ECHO",
        "GREMLIN_WIZARD_ULTIMATE_BLAST", "SNEAKY_GREMLIN_PUNCTURE", "MAD_GREMLIN_SCRATCH", "SHIELD_GREMLIN_SHIELD_BASH",
        "FAT_GREMLIN_SMASH",
        "TORCH_HEAD_TACKLE", "DAGGER_STAB", "DAGGER_EXPLODE",
        "GREMLIN_NOB_RUSH", "LAGAVULIN_ATTACK", "SENTRY_BEAM", "RED_SLAVER_STAB", "BRONZE_ORB_BEAM",
        "LOOTER_LUNGE", "LOOTER_MUG", "MUGGER_LUNGE", "MUGGER_MUG",
        "BEAR_MAUL", "BEAR_LUNGE", "POINTY_ATTACK", "ROMEO_CROSS_SLASH",
        "HEXAGHOST_TACKLE", "CHOSEN_ZAP",
        "SHELLED_SUCK",
    }:
        monster.intent = "ATTACK_DEBUFF" if move == "TASKMASTER_SCOURING_WHIP" else "ATTACK"
        base_hits = {
            "BYRD_PECK": (5, 1),
            "BYRD_HEADBUTT": (1, 3),
            "BYRD_SWOOP": (1, 14 if _monster_ascension(monster) >= 2 else 12),
            "CHOSEN_POKE": (2, 6 if _monster_ascension(monster) >= 2 else 5),
            "CHOSEN_ZAP": (1, 21 if _monster_ascension(monster) >= 2 else 18),
            "SPHERIC_GUARDIAN_SLAM": (2, 11 if _monster_ascension(monster) >= 2 else 10),
            "SPHERIC_GUARDIAN_HARDEN": (1, 11 if _monster_ascension(monster) >= 2 else 10),
            "SPHERIC_GUARDIAN_ATTACK_DEBUFF": (1, 11 if _monster_ascension(monster) >= 2 else 10),
            "SNAKE_PLANT_CHOMP": (3, 8 if _monster_ascension(monster) >= 2 else 7),
            "SNECKO_TAIL_WHIP": (1, 10 if _monster_ascension(monster) >= 2 else 8),
            "SNECKO_BITE": (1, 18 if _monster_ascension(monster) >= 2 else 15),
            "SHELLED_DOUBLE_STRIKE": (2, 7 if _monster_ascension(monster) >= 2 else 6),
            "SHELLED_SUCK": (1, 12 if _monster_ascension(monster) >= 2 else 10),
            "CENTURION_SLASH": (1, 14 if _monster_ascension(monster) >= 2 else 12),
            "CENTURION_FURY": (3, 7 if _monster_ascension(monster) >= 2 else 6),
            "BOOK_MULTI_STAB": (max(2, int(monster.ai_state.get("stab_count", 2))), 7 if _monster_ascension(monster) >= 3 else 6),
            "BOOK_OF_STABBING_MULTI_STAB": (max(2, int(monster.ai_state.get("stab_count", 2))), 7 if _monster_ascension(monster) >= 3 else 6),
            "BOOK_BIG_STAB": (1, 24 if _monster_ascension(monster) >= 3 else 21),
            "BOOK_OF_STABBING_SINGLE_STAB": (1, 24 if _monster_ascension(monster) >= 3 else 21),
            "GREMLIN_LEADER_STAB": (3, 6),
            "TASKMASTER_SCOURING_WHIP": (1, 7),
            "CHAMP_HEAVY_SLASH": (1, 18 if _monster_ascension(monster) >= 4 else 16),
            "CHAMP_EXECUTE": (2, 10),
            "COLLECTOR_FIREBALL": (1, 18),
            "BRONZE_AUTOMATON_FLAIL": (2, 8 if _monster_ascension(monster) >= 4 else 7),
            "BRONZE_AUTOMATON_HYPER_BEAM": (1, 50 if _monster_ascension(monster) >= 4 else 45),
            "DARKLING_NIP": (1, 9),
            "DARKLING_CHOMP": (1, 13),
            "ORB_WALKER_LASER": (1, 11),
            "ORB_WALKER_CLAW": (2, 5),
            "MAW_SLAM": (1, 25),
            "MAW_NOM": (3, 5),
            "TRANSIENT_ATTACK": (1, 30),
            "SPIKER_CUT": (1, 7),
            "EXPLODER_SLAM": (1, 11),
            "EXPLODER_EXPLODE": (1, 30),
            "REPULSOR_ATTACK": (1, 11),
            "NEMESIS_ATTACK": (1, 18),
            "NEMESIS_SCYTHE": (3, 6),
            "GIANT_HEAD_IT_IS_TIME": (1, 30),
            "GIANT_HEAD_ATTACK": (1, 13),
            "REPTOMANCER_STAB": (2, 13),
            "REPTOMANCER_BIG_STAB": (1, 34),
            "AWAKENED_SLASH": (1, 20),
            "AWAKENED_SOUL_STRIKE": (4, 6),
            "TIME_EATER_REVERBERATE": (3, 7),
            "TIME_EATER_HEAD_SLAM": (1, 26),
            "DONU_BEAM": (2, 10),
            "DECA_BEAM": (2, 10),
            "SHIELD_BASH": (1, 12),
            "SHIELD_SMASH": (1, 34),
            "SPEAR_SKEWER": (4, 10),
            "SPEAR_PIERCER": (1, 18),
            "HEART_BLOOD_SHOTS": (15, 2),
            "HEART_ECHO": (1, 45),
            "GREMLIN_WIZARD_ULTIMATE_BLAST": (1, 25),
            "SNEAKY_GREMLIN_PUNCTURE": (1, 9),
            "MAD_GREMLIN_SCRATCH": (1, 4),
            "SHIELD_GREMLIN_SHIELD_BASH": (1, 8 if _monster_ascension(monster) >= 2 else 6),
            "FAT_GREMLIN_SMASH": (1, 4),
            "TORCH_HEAD_TACKLE": (1, 7),
            "DAGGER_STAB": (1, 9),
            "DAGGER_EXPLODE": (1, 25),
            "GREMLIN_NOB_RUSH": (1, 14),
            "LAGAVULIN_ATTACK": (1, 18),
            "SENTRY_BEAM": (1, 9),
            "RED_SLAVER_STAB": (1, 13),
            "BRONZE_ORB_BEAM": (1, 8),
            "LOOTER_LUNGE": (1, 14 if _monster_ascension(monster) >= 2 else 12),
            "LOOTER_MUG": (1, 11 if _monster_ascension(monster) >= 2 else 10),
            "MUGGER_LUNGE": (1, 18 if _monster_ascension(monster) >= 2 else 16),
            "MUGGER_MUG": (1, 11 if _monster_ascension(monster) >= 2 else 10),
            "BEAR_MAUL": (1, 10),
            "BEAR_LUNGE": (1, 20),
            "POINTY_ATTACK": (2, 5),
            "ROMEO_CROSS_SLASH": (1, 15),
            "HEXAGHOST_TACKLE": (2, 6 if _monster_ascension(monster) >= 4 else 5),
        }[move]
        monster.move_hits, monster.move_base_damage = base_hits
        return
    if move == "CHOSEN_HEX":
        monster.intent = "DEBUFF"
        monster.move_hits = 0
        monster.move_base_damage = 0
        return
    if move in {
        "BYRD_CAW", "BYRD_PECK", "BYRD_HEADBUTT", "CHOSEN_POKE", "CHOSEN_ZAP", "CHOSEN_DRAIN", "CHOSEN_HEX", "CHOSEN_DEBILITATE",
        "SPHERIC_GUARDIAN_SLAM", "SPHERIC_GUARDIAN_ACTIVATE", "SPHERIC_GUARDIAN_HARDEN",
        "SPHERIC_GUARDIAN_ATTACK_DEBUFF", "SNAKE_PLANT_CHOMP",
        "SNAKE_PLANT_ENFEEBLING_SPORES", "SNECKO_TAIL_WHIP", "SNECKO_BITE",
        "SNECKO_PERPLEXING_GLARE", "SHELLED_DOUBLE_STRIKE", "SHELLED_FELL", "SHELLED_SUCK",
        "CENTURION_SLASH", "CENTURION_FURY", "CENTURION_DEFEND", "MYSTIC_HEAL",
        "MYSTIC_ATTACK", "MYSTIC_BUFF", "BOOK_MULTI_STAB", "BOOK_OF_STABBING_MULTI_STAB", "BOOK_BIG_STAB", "BOOK_OF_STABBING_SINGLE_STAB",
        "GREMLIN_LEADER_ENCOURAGE", "GREMLIN_LEADER_STAB", "GREMLIN_LEADER_RALLY",
        "TASKMASTER_SCOURING_WHIP", "TASKMASTER_RAKE", "CHAMP_HEAVY_SLASH", "CHAMP_DEFENSIVE_STANCE",
        "CHAMP_FACE_SLAP", "CHAMP_GLOAT", "CHAMP_TAUNT", "CHAMP_ANGER", "CHAMP_EXECUTE", "COLLECTOR_BUFF",
        "COLLECTOR_FIREBALL", "COLLECTOR_MEGA_DEBUFF", "BRONZE_AUTOMATON_BOOST", "BRONZE_AUTOMATON_FLAIL",
        "BRONZE_AUTOMATON_HYPER_BEAM", "BRONZE_AUTOMATON_SPAWN_ORBS", "DARKLING_NIP", "DARKLING_CHOMP", "DARKLING_REINCARNATE",
        "ORB_WALKER_LASER", "ORB_WALKER_CLAW", "ORB_WALKER_CHARGE", "MAW_ROAR", "MAW_DROOL",
        "MAW_SLAM", "MAW_NOM", "TRANSIENT_ATTACK", "WRITHING_FLAIL", "WRITHING_IMPLANT",
        "WRITHING_WITHER", "SPIKER_CUT", "SPIKER_GROW", "EXPLODER_SLAM", "EXPLODER_EXPLODE",
        "REPULSOR_REPULSE", "REPULSOR_ATTACK", "NEMESIS_ATTACK", "NEMESIS_SCYTHE",
        "NEMESIS_DEBUFF", "GIANT_HEAD_COUNT", "GIANT_HEAD_IT_IS_TIME", "GIANT_HEAD_ATTACK",
        "REPTOMANCER_SUMMON", "REPTOMANCER_STAB", "REPTOMANCER_BIG_STAB", "AWAKENED_SLASH",
        "AWAKENED_SOUL_STRIKE", "AWAKENED_REBIRTH", "TIME_EATER_REVERBERATE",
        "TIME_EATER_HEAD_SLAM", "TIME_EATER_RIPPLE", "DONU_CIRCLE_OF_POWER", "DONU_BEAM",
        "DECA_BEAM", "DECA_SQUARE_OF_PROTECTION",
        "SHIELD_BASH", "SHIELD_FORTIFY", "SHIELD_SMASH", "SPEAR_BURN_STRIKE",
        "SPEAR_SKEWER", "SPEAR_PIERCER", "HEART_DEBILITATE", "HEART_BLOOD_SHOTS", "HEART_ECHO",
        "FAT_GREMLIN_SMASH", "FAT_GREMLIN_WEAKEN", "GREMLIN_WIZARD_CHARGING", "GREMLIN_WIZARD_ULTIMATE_BLAST",
        "SNEAKY_GREMLIN_PUNCTURE", "SHIELD_GREMLIN_PROTECT", "SHIELD_GREMLIN_SHIELD_BASH", "MAD_GREMLIN_SCRATCH",
        "TORCH_HEAD_TACKLE", "DAGGER_STAB", "DAGGER_EXPLODE", "RED_SLAVER_STAB",
        "RED_SLAVER_SCRAPE", "BRONZE_ORB_BEAM", "LOOTER_MUG", "LOOTER_LUNGE", "LOOTER_SMOKE_BOMB",
        "MUGGER_MUG", "MUGGER_LUNGE",
        "BEAR_MAUL", "BEAR_LUNGE", "BEAR_BEAR_HUG", "POINTY_ATTACK",
        "ROMEO_AGONIZING_SLASH", "ROMEO_CROSS_SLASH", "ROMEO_MOCK",
        "GREMLIN_NOB_RUSH", "LAGAVULIN_ATTACK", "SENTRY_BEAM", "LAGAVULIN_SLEEP", "LAGAVULIN_STUN",
        "BLUE_SLAVER_STAB", "BLUE_SLAVER_RAKE", "RED_LOUSE_BITE", "RED_LOUSE_GROW",
        "GREEN_LOUSE_BITE", "GREEN_LOUSE_SPIT_WEB", "MUGGER_SMOKE_BOMB",
    }:
        return
    if move in {"RED_SLAVER_ENTANGLE", "BRONZE_ORB_STASIS"}:
        monster.intent = "DEBUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    if move.endswith("LICK") or move in {"LOUSE_SPIT_WEB", "SLIME_BOSS_GOOP_SPRAY", "THE_GUARDIAN_VENT_STEAM", "FAT_GREMLIN_WEAKEN"}:
        monster.intent = "DEBUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "HEXAGHOST_SEAR":
        monster.intent = "ATTACK"
        monster.move_base_damage = 6
        monster.move_hits = 1
        return
    elif move in {"SPIKE_SLIME_M_FLAME_TACKLE", "SPIKE_SLIME_L_FLAME_TACKLE", "ACID_SLIME_M_CORROSIVE_SPIT", "ACID_SLIME_L_CORROSIVE_SPIT", "SLAVER_RAKE"}:
        monster.intent = "ATTACK_DEBUFF"
        if move == "ACID_SLIME_L_CORROSIVE_SPIT":
            monster.move_base_damage = 11
        elif move == "ACID_SLIME_M_CORROSIVE_SPIT":
            monster.move_base_damage = 7
        elif move == "SPIKE_SLIME_L_FLAME_TACKLE":
            monster.move_base_damage = 16
        elif move == "SPIKE_SLIME_M_FLAME_TACKLE":
            monster.move_base_damage = 8
        elif move == "SLAVER_RAKE":
            monster.move_base_damage = 7
        else:
            monster.move_base_damage = 7
        return
    elif move == "GREMLIN_NOB_SKULL_BASH":
        monster.intent = "ATTACK"
        monster.move_base_damage = 6
        monster.move_hits = 1
        return
    elif move in {"ACID_SLIME_S_TACKLE", "SPIKE_SLIME_S_TACKLE"}:
        monster.intent = "ATTACK"
        monster.move_base_damage = 3 if move == "ACID_SLIME_S_TACKLE" else 5
        return
    elif move == "ACID_SLIME_L_TACKLE":
        monster.intent = "ATTACK"
        monster.move_base_damage = 16
        return
    elif move in {"SENTRY_BOLT"}:
        monster.intent = "DEBUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "HEXAGHOST_ACTIVATE":
        monster.intent = "MAGIC"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "SLIME_BOSS_PREPARING":
        monster.intent = "UNKNOWN"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "HEXAGHOST_INFERNO":
        monster.intent = "ATTACK"
        monster.move_base_damage = 2
        monster.move_hits = 6
        return
    elif move == "HEXAGHOST_INFLAME":
        monster.intent = "DEFEND_BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "SLIME_BOSS_SLAM":
        monster.intent = "ATTACK"
        monster.move_base_damage = 35
        return
    elif move == "HEXAGHOST_DIVIDER":
        monster.intent = "ATTACK"
        monster.move_base_damage = 6
        monster.move_hits = 6
        return
    elif move == "THE_GUARDIAN_CHARGING_UP":
        monster.intent = "DEFEND"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "THE_GUARDIAN_FIERCE_BASH":
        monster.intent = "ATTACK"
        monster.move_base_damage = 32
        return
    elif move == "THE_GUARDIAN_DEFENSIVE_MODE":
        monster.intent = "BUFF"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move == "THE_GUARDIAN_ROLL_ATTACK":
        monster.intent = "ATTACK"
        monster.move_base_damage = 9
        monster.move_hits = 1
        return
    elif move == "THE_GUARDIAN_TWIN_SLAM":
        monster.intent = "ATTACK"
        monster.move_base_damage = 8
        monster.move_hits = 2
        return
    elif move == "THE_GUARDIAN_WHIRLWIND":
        monster.intent = "ATTACK"
        monster.move_base_damage = 5
        monster.move_hits = 4
        return
    else:
        monster.intent = "ATTACK"
        monster.move_base_damage = 10
