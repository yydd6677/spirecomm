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


def encounter_to_monster_ids(encounter: str, rng: random.Random) -> list[str]:
    if encounter == "Louses":
        return [_roll_louse(rng), _roll_louse(rng)]
    if encounter == "SmallSlimes":
        if rng.random_boolean():
            return ["SpikeSlime_S", "AcidSlime_M"]
        return ["AcidSlime_S", "SpikeSlime_M"]
    if encounter == "LotsOfSlimes":
        slime_pool = ["SpikeSlime_S", "SpikeSlime_S", "SpikeSlime_S", "AcidSlime_S", "AcidSlime_S"]
        monsters: list[str] = []
        for index in range(4, -1, -1):
            chosen = int(rng.random(index))
            monster_id = slime_pool[chosen]
            del slime_pool[chosen]
            monsters.append(monster_id)
        return monsters
    if encounter == "ExordiumThugs":
        return ["Looter", "SlaverBlue"]
    if encounter == "ExordiumWildlife":
        return ["FungiBeast", "JawWorm"]
    if encounter == "LargeSlime":
        return ["AcidSlime_L" if rng.random_boolean() else "SpikeSlime_L"]
    if encounter == "ThreeLouses":
        return [_roll_louse(rng) for _ in range(3)]
    if encounter == "FungiPair":
        return ["FungiBeast", "FungiBeast"]
    if encounter == "GremlinGang":
        pool = [
            "GremlinWarrior",
            "GremlinWarrior",
            "GremlinThief",
            "GremlinThief",
            "GremlinFat",
            "GremlinFat",
            "GremlinTsundere",
            "GremlinWizard",
        ]
        result: list[str] = []
        last_idx = len(pool) - 1
        for _ in range(4):
            idx = int(rng.random(last_idx))
            result.append(pool[idx])
            del pool[idx]
            last_idx -= 1
        return result
    if encounter == "ThreeSentries":
        return ["Sentry", "Sentry", "Sentry"]
    if encounter == "Slavers":
        return ["SlaverBlue", "Taskmaster", "SlaverRed"]
    if encounter == "ByrdTriple":
        return ["Byrd", "Byrd", "Byrd"]
    if encounter == "Thieves":
        return ["Looter", "Mugger"]
    if encounter == "ChosenAndByrds":
        return ["Chosen", "Byrd", "Byrd"]
    if encounter == "SentryAndSphere":
        return ["Sentry", "SphericGuardian"]
    if encounter == "CultistAndChosen":
        return ["Cultist", "Chosen"]
    if encounter == "ThreeCultists":
        return ["Cultist", "Cultist", "Cultist"]
    if encounter == "ShelledParasiteAndFungi":
        return ["ShelledParasite", "FungiBeast"]
    if encounter == "CenturionMystic":
        return ["Centurion", "Mystic"]
    if encounter == "Darklings":
        return ["Darkling", "Darkling", "Darkling"]
    if encounter == "ShapesTriple":
        pool = ["Repulsor", "Repulsor", "Exploder", "Exploder", "Spiker", "Spiker"]
        result: list[str] = []
        last_idx = len(pool) - 1
        for _ in range(3):
            idx = int(rng.random(last_idx))
            result.append(pool[idx])
            del pool[idx]
            last_idx -= 1
        return result
    if encounter == "ShapesQuad":
        pool = ["Repulsor", "Repulsor", "Exploder", "Exploder", "Spiker", "Spiker"]
        result: list[str] = []
        last_idx = len(pool) - 1
        for _ in range(4):
            idx = int(rng.random(last_idx))
            result.append(pool[idx])
            del pool[idx]
            last_idx -= 1
        return result
    if encounter == "SphereAndShapes":
        return ["SphericGuardian", rng.choice(["Spiker", "Exploder", "Repulsor"]), rng.choice(["Spiker", "Exploder", "Repulsor"])]
    if encounter == "JawWormHorde":
        return ["JawWorm", "JawWorm", "JawWorm"]
    if encounter == "DonuDeca":
        return ["Donu", "Deca"]
    if encounter == "Reptomancer":
        return ["Reptomancer", "SnakeDagger", "SnakeDagger"]
    if encounter == "BronzeAutomaton":
        return ["BronzeAutomaton", "BronzeOrb", "BronzeOrb"]
    if encounter == "TheCollector":
        return ["TheCollector", "TorchHead", "TorchHead"]
    if encounter in {"Cultist", "JawWorm", "AcidSlime_M", "SpikeSlime_M", "FungiBeast", "SlaverBlue", "Looter", "SlaverRed", "SphericGuardian", "Chosen", "ShelledParasite", "SnakePlant", "Snecko", "OrbWalker", "TheMaw", "Transient", "WrithingMass", "GremlinNob", "Lagavulin", "BookOfStabbing", "GremlinLeader", "GiantHead", "Nemesis", "Hexaghost", "SlimeBoss", "TheGuardian", "TheChamp", "AwakenedOne", "TimeEater", "SpireGrowth"}:
        return [encounter]
    return ["Cultist"]


def encounter_to_monsters(
    encounter: str,
    hp_rng: random.Random,
    misc_rng: random.Random | None = None,
    ascension: int = 0,
) -> list[MonsterState]:
    misc_rng = hp_rng if misc_rng is None else misc_rng

    if encounter == "ExordiumThugs":
        weak_candidates = [
            make_monster(_roll_louse(misc_rng), hp_rng, ascension),
            make_monster("SpikeSlime_M", hp_rng, ascension),
            make_monster("AcidSlime_M", hp_rng, ascension),
        ]
        weak_choice = int(misc_rng.random(2))
        strong_candidates = [
            make_monster("Cultist", hp_rng, ascension),
            make_monster("SlaverRed" if misc_rng.random_boolean() else "SlaverBlue", hp_rng, ascension),
            make_monster("Looter", hp_rng, ascension),
        ]
        strong_choice = int(misc_rng.random(2))
        return [
            weak_candidates[weak_choice],
            strong_candidates[strong_choice],
        ]

    if encounter == "ExordiumWildlife":
        strong_candidates = [
            make_monster("FungiBeast", hp_rng, ascension),
            make_monster("JawWorm", hp_rng, ascension),
        ]
        strong_choice = int(misc_rng.random(1))
        weak_candidates = [
            make_monster(_roll_louse(misc_rng), hp_rng, ascension),
            make_monster("SpikeSlime_M", hp_rng, ascension),
            make_monster("AcidSlime_M", hp_rng, ascension),
        ]
        weak_choice = int(misc_rng.random(2))
        return [
            strong_candidates[strong_choice],
            weak_candidates[weak_choice],
        ]

    return [make_monster(monster_id, hp_rng, ascension) for monster_id in encounter_to_monster_ids(encounter, misc_rng)]


def make_monster(monster_id: str, rng: random.Random, ascension: int = 0) -> MonsterState:
    if monster_id == "Cultist":
        hp = rng.randint(48, 54)
        return MonsterState("Cultist", "Cultist", hp, hp, "CULTIST_INCANTATION", "BUFF")
    if monster_id == "JawWorm":
        hp = rng.randint(40, 44)
        return MonsterState("JawWorm", "Jaw Worm", hp, hp, "JAW_WORM_CHOMP", "ATTACK", move_base_damage=11, move_hits=1)
    if monster_id == "AcidSlime_M":
        hp = rng.randint(28, 32)
        return MonsterState("AcidSlime_M", "Acid Slime (M)", hp, hp, "ACID_SLIME_M_TACKLE", "ATTACK", move_base_damage=10, move_hits=1)
    if monster_id == "AcidSlime_S":
        hp = rng.randint(8, 12)
        return MonsterState("AcidSlime_S", "Acid Slime (S)", hp, hp, "ACID_SLIME_S_TACKLE", "ATTACK", move_base_damage=3, move_hits=1)
    if monster_id == "SpikeSlime_M":
        hp = rng.randint(28, 32)
        return MonsterState("SpikeSlime_M", "Spike Slime (M)", hp, hp, "SPIKE_SLIME_M_FLAME_TACKLE", "ATTACK_DEBUFF", move_base_damage=8, move_hits=1)
    if monster_id == "SpikeSlime_S":
        hp = rng.randint(10, 14)
        return MonsterState("SpikeSlime_S", "Spike Slime (S)", hp, hp, "SPIKE_SLIME_S_TACKLE", "ATTACK", move_base_damage=5, move_hits=1)
    if monster_id == "RedLouse":
        hp = rng.randint(10, 15)
        bite_damage = rng.randint(6, 8) if _rng_ascension_level(rng) >= 2 else rng.randint(5, 7)
        monster = MonsterState("RedLouse", "Red Louse", hp, hp, "RED_LOUSE_BITE", "ATTACK", move_base_damage=bite_damage, move_hits=1)
        monster.ai_state["bite_damage"] = bite_damage
        monster.ai_state["pending_prebattle"] = "LOUSE"
        return monster
    if monster_id == "GreenLouse":
        hp = rng.randint(11, 17)
        bite_damage = rng.randint(6, 8) if _rng_ascension_level(rng) >= 2 else rng.randint(5, 7)
        monster = MonsterState("GreenLouse", "Green Louse", hp, hp, "GREEN_LOUSE_SPIT_WEB", "DEBUFF")
        monster.ai_state["bite_damage"] = bite_damage
        monster.ai_state["pending_prebattle"] = "LOUSE"
        return monster
    if monster_id == "FungiBeast":
        hp = rng.randint(22, 28)
        monster = MonsterState("FungiBeast", "Fungi Beast", hp, hp, "FUNGI_BEAST_BITE", "ATTACK", move_base_damage=6, move_hits=1)
        monster.add_power("Spore Cloud", 2)
        return monster
    if monster_id == "SlaverBlue":
        rng.randint(52, 58)
        hp = rng.randint(48, 52) if ascension >= 9 else rng.randint(46, 50)
        return MonsterState("SlaverBlue", "Blue Slaver", hp, hp, "BLUE_SLAVER_RAKE", "ATTACK_DEBUFF", move_base_damage=7, move_hits=1)
    if monster_id == "SlaverRed":
        rng.randint(52, 58)
        hp = rng.randint(48, 52) if ascension >= 9 else rng.randint(46, 50)
        return MonsterState("SlaverRed", "Red Slaver", hp, hp, "RED_SLAVER_STAB", "ATTACK", move_base_damage=13, move_hits=1)
    if monster_id == "Looter":
        hp = rng.randint(44, 48)
        return MonsterState("Looter", "Looter", hp, hp, "LOOTER_MUG", "ATTACK_DEBUFF", move_base_damage=10, move_hits=1)
    if monster_id == "Mugger":
        hp = rng.randint(48, 52)
        return MonsterState("Mugger", "Mugger", hp, hp, "MUGGER_MUG", "ATTACK_DEBUFF", move_base_damage=10, move_hits=1)
    if monster_id == "Bear":
        hp = rng.randint(38, 42)
        return MonsterState("Bear", "Bear", hp, hp, "BEAR_MAUL", "ATTACK", move_base_damage=10, move_hits=1)
    if monster_id == "Pointy":
        hp = rng.randint(28, 32)
        return MonsterState("Pointy", "Pointy", hp, hp, "POINTY_ATTACK", "ATTACK", move_base_damage=5, move_hits=2)
    if monster_id == "Romeo":
        hp = rng.randint(35, 39)
        return MonsterState("Romeo", "Romeo", hp, hp, "ROMEO_MOCK", "ATTACK_DEBUFF", move_base_damage=6, move_hits=1)
    if monster_id == "GremlinFat":
        hp = rng.randint(13, 17)
        return MonsterState("GremlinFat", "Fat Gremlin", hp, hp, "FAT_GREMLIN_SMASH", "ATTACK_DEBUFF", move_base_damage=4, move_hits=1)
    if monster_id == "GremlinWizard":
        hp = rng.randint(21, 25)
        return MonsterState("GremlinWizard", "Gremlin Wizard", hp, hp, "GREMLIN_WIZARD_CHARGING", "UNKNOWN")
    if monster_id == "GremlinThief":
        hp = rng.randint(10, 14)
        return MonsterState("GremlinThief", "Sneaky Gremlin", hp, hp, "SNEAKY_GREMLIN_PUNCTURE", "ATTACK", move_base_damage=9, move_hits=1)
    if monster_id == "GremlinTsundere":
        hp = rng.randint(12, 15)
        return MonsterState("GremlinTsundere", "Shield Gremlin", hp, hp, "SHIELD_GREMLIN_PROTECT", "DEFEND")
    if monster_id == "GremlinWarrior":
        hp = rng.randint(20, 24)
        monster = MonsterState("GremlinWarrior", "Mad Gremlin", hp, hp, "MAD_GREMLIN_SCRATCH", "ATTACK", move_base_damage=4, move_hits=1)
        monster.add_power("Angry", 2 if ascension >= 17 else 1)
        return monster
    if monster_id == "GremlinNob":
        hp = rng.randint(82, 86)
        return MonsterState("GremlinNob", "Gremlin Nob", hp, hp, "GREMLIN_NOB_BELLOW", "BUFF")
    if monster_id == "Lagavulin":
        hp = rng.randint(109, 111)
        monster = MonsterState("Lagavulin", "Lagavulin", hp, hp, "LAGAVULIN_SLEEP", "SLEEP")
        monster.add_power("Metallicize", 8)
        monster.block = 8
        monster.ai_state["asleep"] = 1
        monster.ai_state["latent_awake"] = 0
        return monster
    if monster_id == "LagavulinEvent":
        hp = rng.randint(109, 111)
        return MonsterState("Lagavulin", "Lagavulin", hp, hp, "LAGAVULIN_SIPHON_SOUL", "STRONG_DEBUFF")
    if monster_id == "Sentry":
        hp = rng.randint(38, 42)
        monster = MonsterState("Sentry", "Sentry", hp, hp, "SENTRY_BEAM", "ATTACK", move_base_damage=9, move_hits=1)
        monster.add_power("Artifact", 1)
        return monster
    if monster_id == "SlimeBoss":
        hp = 140
        return MonsterState("SlimeBoss", "Slime Boss", hp, hp, "SLIME_BOSS_GOOP_SPRAY", "STRONG_DEBUFF")
    if monster_id == "AcidSlime_L":
        hp = rng.randint(65, 69)
        return MonsterState("AcidSlime_L", "Acid Slime (L)", hp, hp, "ACID_SLIME_L_CORROSIVE_SPIT", "ATTACK_DEBUFF", move_base_damage=11, move_hits=1)
    if monster_id == "SpikeSlime_L":
        hp = rng.randint(64, 70)
        return MonsterState("SpikeSlime_L", "Spike Slime (L)", hp, hp, "SPIKE_SLIME_L_FLAME_TACKLE", "ATTACK_DEBUFF", move_base_damage=16, move_hits=1)
    if monster_id == "Hexaghost":
        hp = 264 if ascension >= 9 else 250
        return MonsterState("Hexaghost", "Hexaghost", hp, hp, "HEXAGHOST_ACTIVATE", "MAGIC")
    if monster_id == "TheGuardian":
        hp = 240
        monster = MonsterState("TheGuardian", "The Guardian", hp, hp, "THE_GUARDIAN_CHARGING_UP", "DEFEND")
        monster.add_power("Mode Shift", 30)
        return monster
    if monster_id == "Byrd":
        hp = rng.randint(25, 31)
        monster = MonsterState("Byrd", "Byrd", hp, hp, "BYRD_CAW", "BUFF")
        monster.add_power("Flight", 3)
        return monster
    if monster_id == "Chosen":
        hp = rng.randint(95, 99)
        return MonsterState(
            "Chosen",
            "Chosen",
            hp,
            hp,
            "CHOSEN_POKE",
            "ATTACK",
            move_base_damage=6 if ascension >= 2 else 5,
            move_hits=2,
        )
    if monster_id == "SphericGuardian":
        hp = rng.randint(20, 20)
        monster = MonsterState("SphericGuardian", "Spheric Guardian", hp, hp, "SPHERIC_GUARDIAN_ACTIVATE", "DEFEND", move_base_damage=0, move_hits=0)
        monster.block = 40
        monster.add_power("Barricade", 1)
        monster.add_power("Artifact", 3)
        return monster
    if monster_id == "SnakePlant":
        hp = rng.randint(78, 82)
        monster = MonsterState(
            "SnakePlant",
            "Snake Plant",
            hp,
            hp,
            "SNAKE_PLANT_CHOMP",
            "ATTACK",
            move_base_damage=8 if ascension >= 2 else 7,
            move_hits=3,
        )
        monster.add_power("Malleable", 3)
        return monster
    if monster_id == "Snecko":
        hp = rng.randint(112, 118)
        return MonsterState("Snecko", "Snecko", hp, hp, "SNECKO_PERPLEXING_GLARE", "DEBUFF")
    if monster_id == "ShelledParasite":
        hp = rng.randint(68, 72)
        monster = MonsterState("ShelledParasite", "Shelled Parasite", hp, hp, "SHELLED_DOUBLE_STRIKE", "ATTACK", move_base_damage=6, move_hits=2)
        monster.add_power("Plated Armor", 14)
        monster.block = 14
        return monster
    if monster_id == "Centurion":
        hp = rng.randint(76, 83 if ascension >= 7 else 80)
        return MonsterState("Centurion", "Centurion", hp, hp, "CENTURION_SLASH", "ATTACK", move_base_damage=12, move_hits=1)
    if monster_id == "Mystic":
        hp = rng.randint(48, 56)
        return MonsterState("Mystic", "Mystic", hp, hp, "MYSTIC_HEAL", "MAGIC")
    if monster_id == "BookOfStabbing":
        hp = rng.randint(160, 164)
        return MonsterState("BookOfStabbing", "Book of Stabbing", hp, hp, "BOOK_MULTI_STAB", "ATTACK", move_base_damage=6, move_hits=2)
    if monster_id == "GremlinLeader":
        hp = rng.randint(148, 155)
        return MonsterState("GremlinLeader", "Gremlin Leader", hp, hp, "GREMLIN_LEADER_ENCOURAGE", "BUFF")
    if monster_id == "Taskmaster":
        rng.randint(54, 60)
        hp = rng.randint(57, 64) if ascension >= 8 else rng.randint(54, 60)
        return MonsterState("Taskmaster", "Taskmaster", hp, hp, "TASKMASTER_SCOURING_WHIP", "ATTACK_DEBUFF", move_base_damage=7, move_hits=1)
    if monster_id == "TheChamp":
        hp = 440
        return MonsterState("TheChamp", "The Champ", hp, hp, "CHAMP_DEFENSIVE_STANCE", "DEFEND_BUFF")
    if monster_id == "TheCollector":
        hp = 300
        return MonsterState("TheCollector", "The Collector", hp, hp, "COLLECTOR_BUFF", "BUFF")
    if monster_id == "TorchHead":
        hp = rng.randint(38, 40)
        return MonsterState("TorchHead", "Torch Head", hp, hp, "TORCH_HEAD_TACKLE", "ATTACK", move_base_damage=7, move_hits=1)
    if monster_id == "BronzeOrb":
        hp = rng.randint(52, 58)
        return MonsterState("BronzeOrb", "Bronze Orb", hp, hp, "BRONZE_ORB_BEAM", "ATTACK", move_base_damage=8, move_hits=1)
    if monster_id == "BronzeAutomaton":
        hp = 300
        return MonsterState("BronzeAutomaton", "Bronze Automaton", hp, hp, "AUTOMATON_BOOST", "BUFF")
    if monster_id == "Darkling":
        hp = rng.randint(48, 56)
        return MonsterState("Darkling", "Darkling", hp, hp, "DARKLING_NIP", "ATTACK", move_base_damage=9, move_hits=1)
    if monster_id == "OrbWalker":
        hp = rng.randint(90, 96)
        return MonsterState("OrbWalker", "Orb Walker", hp, hp, "ORB_WALKER_LASER", "ATTACK", move_base_damage=11, move_hits=1)
    if monster_id == "TheMaw":
        hp = rng.randint(290, 300)
        return MonsterState("TheMaw", "The Maw", hp, hp, "MAW_ROAR", "BUFF")
    if monster_id == "Transient":
        hp = 999
        monster = MonsterState("Transient", "Transient", hp, hp, "TRANSIENT_ATTACK", "ATTACK", move_base_damage=30, move_hits=1)
        monster.add_power("Shifting", 1)
        return monster
    if monster_id == "WrithingMass":
        hp = rng.randint(160, 175)
        return MonsterState("WrithingMass", "Writhing Mass", hp, hp, "WRITHING_FLAIL", "ATTACK_DEBUFF", move_base_damage=15, move_hits=1)
    if monster_id == "Spiker":
        hp = rng.randint(42, 56)
        monster = MonsterState("Spiker", "Spiker", hp, hp, "SPIKER_CUT", "ATTACK", move_base_damage=7, move_hits=1)
        monster.add_power("Thorns", 3)
        return monster
    if monster_id == "Exploder":
        hp = rng.randint(28, 35)
        return MonsterState("Exploder", "Exploder", hp, hp, "EXPLODER_SLAM", "ATTACK", move_base_damage=11, move_hits=1)
    if monster_id == "Repulsor":
        hp = rng.randint(29, 38)
        return MonsterState("Repulsor", "Repulsor", hp, hp, "REPULSOR_REPULSE", "DEBUFF")
    if monster_id == "Nemesis":
        hp = rng.randint(185, 200)
        return MonsterState("Nemesis", "Nemesis", hp, hp, "NEMESIS_DEBUFF", "DEBUFF")
    if monster_id == "GiantHead":
        hp = rng.randint(500, 520)
        monster = MonsterState("GiantHead", "Giant Head", hp, hp, "GIANT_HEAD_COUNT", "BUFF")
        monster.add_power("Slow", 0)
        return monster
    if monster_id == "Reptomancer":
        hp = rng.randint(180, 190)
        return MonsterState("Reptomancer", "Reptomancer", hp, hp, "REPTOMANCER_SUMMON", "UNKNOWN")
    if monster_id == "SnakeDagger":
        hp = rng.randint(20, 25)
        return MonsterState("SnakeDagger", "Dagger", hp, hp, "DAGGER_STAB", "ATTACK", move_base_damage=9, move_hits=1)
    if monster_id == "AwakenedOne":
        hp = 300
        monster = MonsterState("AwakenedOne", "Awakened One", hp, hp, "AWAKENED_SLASH", "ATTACK", move_base_damage=20, move_hits=1)
        monster.add_power("Curiosity", 1)
        return monster
    if monster_id == "TimeEater":
        hp = 456
        return MonsterState("TimeEater", "Time Eater", hp, hp, "TIME_EATER_REVERBERATE", "ATTACK", move_base_damage=7, move_hits=3)
    if monster_id == "Donu":
        hp = 265
        return MonsterState("Donu", "Donu", hp, hp, "DONU_CIRCLE_OF_POWER", "BUFF")
    if monster_id == "Deca":
        hp = 265
        return MonsterState("Deca", "Deca", hp, hp, "DECA_BEAM", "ATTACK", move_base_damage=10, move_hits=2)
    if monster_id == "SpireShield":
        hp = 125
        monster = MonsterState("SpireShield", "Spire Shield", hp, hp, "SHIELD_BASH", "ATTACK_DEFEND", move_base_damage=12, move_hits=1)
        monster.block = 30
        return monster
    if monster_id == "SpireSpear":
        hp = 180
        return MonsterState("SpireSpear", "Spire Spear", hp, hp, "SPEAR_BURN_STRIKE", "ATTACK_DEBUFF", move_base_damage=6, move_hits=2)
    if monster_id == "CorruptHeart":
        hp = 750
        monster = MonsterState("CorruptHeart", "Corrupt Heart", hp, hp, "HEART_DEBILITATE", "STRONG_DEBUFF")
        monster.add_power("Beat of Death", 1)
        return monster
    raise KeyError(f"unsupported native simulator monster: {monster_id}")


def roll_act1_encounter(rng: random.Random, floor: int, act: int = 1, act_boss: str = "Hexaghost", elite: bool = False) -> list[MonsterState]:
    if floor == 53:
        return [make_monster("SpireShield", rng), make_monster("SpireSpear", rng)]
    if floor == 54:
        return [make_monster("CorruptHeart", rng)]
    if floor in {16, 33, 50}:
        if floor == 16:
            boss = {"Hexaghost": "Hexaghost", "Slime Boss": "SlimeBoss", "The Guardian": "TheGuardian"}.get(act_boss, "Hexaghost")
            return [make_monster(boss, rng)]
        if floor == 33:
            boss = {"The Champ": "TheChamp", "The Collector": "TheCollector", "Bronze Automaton": "BronzeAutomaton"}.get(act_boss, rng.choice(["TheChamp", "TheCollector", "BronzeAutomaton"]))
            if boss == "TheCollector":
                return [make_monster("TheCollector", rng), make_monster("TorchHead", rng), make_monster("TorchHead", rng)]
            if boss == "BronzeAutomaton":
                return [make_monster("BronzeAutomaton", rng), make_monster("BronzeOrb", rng), make_monster("BronzeOrb", rng)]
            return [make_monster(boss, rng)]
        choice = {"Awakened One": "AwakenedOne", "Time Eater": "TimeEater", "Donu and Deca": "DonuDeca"}.get(act_boss, rng.choice(["AwakenedOne", "TimeEater", "DonuDeca"]))
        if choice == "DonuDeca":
            return [make_monster("Donu", rng), make_monster("Deca", rng)]
        return [make_monster(choice, rng)]
    if elite:
        if act == 2:
            choice = rng.choice(["BookOfStabbing", "GremlinLeader", "Slavers"])
            if choice == "Slavers":
                return [make_monster("SlaverBlue", rng), make_monster("Taskmaster", rng), make_monster("SlaverRed", rng)]
            if choice == "GremlinLeader":
                smalls = ["GremlinFat", "GremlinWizard", "GremlinThief", "GremlinTsundere", "GremlinWarrior"]
                return [make_monster("GremlinLeader", rng), make_monster(rng.choice(smalls), rng), make_monster(rng.choice(smalls), rng)]
            return [make_monster(choice, rng)]
        if act >= 3:
            choice = rng.choice(["Nemesis", "GiantHead", "Reptomancer"])
            if choice == "Reptomancer":
                return [make_monster("Reptomancer", rng), make_monster("SnakeDagger", rng), make_monster("SnakeDagger", rng)]
            return [make_monster(choice, rng)]
        choice = rng.choice(["GremlinNob", "Lagavulin", "Sentries"])
        if choice == "Sentries":
            return [make_monster("Sentry", rng), make_monster("Sentry", rng), make_monster("Sentry", rng)]
        return [make_monster(choice, rng)]
    if act == 2:
        choice = rng.choice(["ByrdPair", "Chosen", "SphericGuardian", "SnakePlant", "Snecko", "ShelledParasite", "CenturionMystic", "Thieves"])
        if choice == "ByrdPair":
            return [make_monster("Byrd", rng), make_monster("Byrd", rng)]
        if choice == "CenturionMystic":
            return [make_monster("Centurion", rng), make_monster("Mystic", rng)]
        if choice == "Thieves":
            return [make_monster("Looter", rng), make_monster("Mugger", rng)]
        return [make_monster(choice, rng)]
    if act >= 3:
        choice = rng.choice(["Darklings", "OrbWalker", "TheMaw", "Transient", "WrithingMass", "Shapes"])
        if choice == "Darklings":
            return [make_monster("Darkling", rng), make_monster("Darkling", rng), make_monster("Darkling", rng)]
        if choice == "Shapes":
            return [make_monster(rng.choice(["Spiker", "Exploder", "Repulsor"]), rng) for _ in range(3)]
        return [make_monster(choice, rng)]
    choice = rng.choice(["Cultist", "JawWorm", "AcidSlime_M", "SpikeSlime_M", "Louses", "FungiBeast", "SlaverBlue", "Looter"])
    if choice == "Louses":
        return [make_monster(rng.choice(["RedLouse", "GreenLouse"]), rng), make_monster(rng.choice(["RedLouse", "GreenLouse"]), rng)]
    return [make_monster(choice, rng)]


def monster_adjusted_damage(monster: MonsterState, player: PlayerState, *, vulnerable_multiplier: float = 1.5) -> int:
    if monster.move_base_damage <= 0 or monster.move_hits <= 0:
        return 0
    damage = float(monster.move_base_damage + monster.power("Strength"))
    if monster.power("Weakened") > 0:
        damage *= 0.75
    if player.power("Vulnerable") > 0:
        damage *= vulnerable_multiplier
    return max(0, math.floor(damage))


def choose_next_move(monster: MonsterState, rng: random.Random) -> None:
    try:
        roll = int(rng.random(99))
    except TypeError:
        roll = int(rng.random() * 100)

    if monster.monster_id == "Cultist":
        _set_move(monster, "CULTIST_INCANTATION" if not monster.move_history else "CULTIST_DARK_STRIKE")
        return

    if monster.monster_id == "JawWorm":
        if not monster.move_history:
            _set_move(monster, "JAW_WORM_CHOMP")
            return
        if roll < 25:
            if _last_move(monster, "JAW_WORM_CHOMP"):
                _set_move(monster, "JAW_WORM_BELLOW" if rng.random_boolean(0.5625) else "JAW_WORM_THRASH")
            else:
                _set_move(monster, "JAW_WORM_CHOMP")
        elif roll < 55:
            if _last_two_moves(monster, "JAW_WORM_THRASH"):
                _set_move(monster, "JAW_WORM_CHOMP" if rng.random_boolean(0.357) else "JAW_WORM_BELLOW")
            else:
                _set_move(monster, "JAW_WORM_THRASH")
        elif _last_move(monster, "JAW_WORM_BELLOW"):
            _set_move(monster, "JAW_WORM_CHOMP" if rng.random_boolean(0.416) else "JAW_WORM_THRASH")
        else:
            _set_move(monster, "JAW_WORM_BELLOW")
        return

    if monster.monster_id in {"AcidSlime_S", "AcidSlime_M", "AcidSlime_L"}:
        if monster.monster_id == "AcidSlime_S":
            if _monster_ascension(monster) >= 17:
                _set_move(monster, "ACID_SLIME_S_LICK")
            elif rng.random_boolean():
                _set_move(monster, "ACID_SLIME_S_TACKLE")
            else:
                _set_move(monster, "ACID_SLIME_S_LICK")
            return
        if monster.monster_id == "AcidSlime_M":
            if _monster_ascension(monster) >= 17:
                if roll < 40:
                    if _last_two_moves(monster, "ACID_SLIME_M_CORROSIVE_SPIT"):
                        _set_move(monster, "ACID_SLIME_M_TACKLE" if rng.random_boolean() else "ACID_SLIME_M_LICK")
                    else:
                        _set_move(monster, "ACID_SLIME_M_CORROSIVE_SPIT")
                elif roll < 80:
                    if _last_two_moves(monster, "ACID_SLIME_M_TACKLE"):
                        _set_move(monster, "ACID_SLIME_M_CORROSIVE_SPIT" if rng.random_boolean(0.5) else "ACID_SLIME_M_LICK")
                    else:
                        _set_move(monster, "ACID_SLIME_M_TACKLE")
                elif _last_move(monster, "ACID_SLIME_M_LICK"):
                    _set_move(monster, "ACID_SLIME_M_CORROSIVE_SPIT" if rng.random_boolean(0.4) else "ACID_SLIME_M_TACKLE")
                else:
                    _set_move(monster, "ACID_SLIME_M_LICK")
            else:
                if roll < 30:
                    if _last_two_moves(monster, "ACID_SLIME_M_CORROSIVE_SPIT"):
                        _set_move(monster, "ACID_SLIME_M_TACKLE" if rng.random_boolean() else "ACID_SLIME_M_LICK")
                    else:
                        _set_move(monster, "ACID_SLIME_M_CORROSIVE_SPIT")
                elif roll < 70:
                    if _last_move(monster, "ACID_SLIME_M_TACKLE"):
                        _set_move(monster, "ACID_SLIME_M_CORROSIVE_SPIT" if rng.random_boolean(0.4) else "ACID_SLIME_M_LICK")
                    else:
                        _set_move(monster, "ACID_SLIME_M_TACKLE")
                elif _last_two_moves(monster, "ACID_SLIME_M_LICK"):
                    _set_move(monster, "ACID_SLIME_M_CORROSIVE_SPIT" if rng.random_boolean(0.4) else "ACID_SLIME_M_TACKLE")
                else:
                    _set_move(monster, "ACID_SLIME_M_LICK")
            return
        if monster.current_hp > 0 and monster.current_hp <= monster.max_hp // 2:
            _set_move(monster, "ACID_SLIME_L_SPLIT")
            return
        if _monster_ascension(monster) >= 17:
            if roll < 40:
                if _last_two_moves(monster, "ACID_SLIME_L_CORROSIVE_SPIT"):
                    _set_move(monster, "ACID_SLIME_L_TACKLE" if rng.random_boolean(0.6) else "ACID_SLIME_L_LICK")
                else:
                    _set_move(monster, "ACID_SLIME_L_CORROSIVE_SPIT")
            elif roll < 70:
                if _last_two_moves(monster, "ACID_SLIME_L_TACKLE"):
                    _set_move(monster, "ACID_SLIME_L_CORROSIVE_SPIT" if rng.random_boolean(0.6) else "ACID_SLIME_L_LICK")
                else:
                    _set_move(monster, "ACID_SLIME_L_TACKLE")
            elif _last_move(monster, "ACID_SLIME_L_LICK"):
                _set_move(monster, "ACID_SLIME_L_CORROSIVE_SPIT" if rng.random_boolean(0.4) else "ACID_SLIME_L_TACKLE")
            else:
                _set_move(monster, "ACID_SLIME_L_LICK")
        elif roll < 30:
            if _last_two_moves(monster, "ACID_SLIME_L_CORROSIVE_SPIT"):
                _set_move(monster, "ACID_SLIME_L_TACKLE" if rng.random_boolean() else "ACID_SLIME_L_LICK")
            else:
                _set_move(monster, "ACID_SLIME_L_CORROSIVE_SPIT")
        elif roll < 70:
            if _last_move(monster, "ACID_SLIME_L_TACKLE"):
                _set_move(monster, "ACID_SLIME_L_CORROSIVE_SPIT" if rng.random_boolean(0.4) else "ACID_SLIME_L_LICK")
            else:
                _set_move(monster, "ACID_SLIME_L_TACKLE")
        elif _last_two_moves(monster, "ACID_SLIME_L_LICK"):
            _set_move(monster, "ACID_SLIME_L_CORROSIVE_SPIT" if rng.random_boolean(0.4) else "ACID_SLIME_L_TACKLE")
        else:
            _set_move(monster, "ACID_SLIME_L_LICK")
        return

    if monster.monster_id in {"SpikeSlime_S", "SpikeSlime_M", "SpikeSlime_L"}:
        if monster.monster_id == "SpikeSlime_S":
            _set_move(monster, "SPIKE_SLIME_S_TACKLE")
            return
        if monster.monster_id == "SpikeSlime_M":
            if roll < 30:
                if _last_two_moves(monster, "SPIKE_SLIME_M_FLAME_TACKLE"):
                    _set_move(monster, "SPIKE_SLIME_M_LICK")
                else:
                    _set_move(monster, "SPIKE_SLIME_M_FLAME_TACKLE")
            elif _last_two_moves(monster, "SPIKE_SLIME_M_LICK") or (_monster_ascension(monster) >= 17 and _last_move(monster, "SPIKE_SLIME_M_LICK")):
                _set_move(monster, "SPIKE_SLIME_M_FLAME_TACKLE")
            else:
                _set_move(monster, "SPIKE_SLIME_M_LICK")
            return
        if monster.current_hp > 0 and monster.current_hp <= monster.max_hp // 2:
            _set_move(monster, "SPIKE_SLIME_L_SPLIT")
            return
        if roll < 30:
            if _last_two_moves(monster, "SPIKE_SLIME_L_FLAME_TACKLE"):
                _set_move(monster, "SPIKE_SLIME_L_LICK")
            else:
                _set_move(monster, "SPIKE_SLIME_L_FLAME_TACKLE")
        elif _last_two_moves(monster, "SPIKE_SLIME_L_LICK") or (_monster_ascension(monster) >= 17 and _last_move(monster, "SPIKE_SLIME_L_LICK")):
            _set_move(monster, "SPIKE_SLIME_L_FLAME_TACKLE")
        else:
            _set_move(monster, "SPIKE_SLIME_L_LICK")
        return

    if monster.monster_id in {"RedLouse", "GreenLouse"}:
        if monster.monster_id == "RedLouse":
            if roll < 25:
                if _last_move(monster, "RED_LOUSE_GROW") and (_monster_ascension(monster) >= 17 or _last_two_moves(monster, "RED_LOUSE_GROW")):
                    _set_move(monster, "RED_LOUSE_BITE")
                else:
                    _set_move(monster, "RED_LOUSE_GROW")
            elif _last_two_moves(monster, "RED_LOUSE_BITE"):
                _set_move(monster, "RED_LOUSE_GROW")
            else:
                _set_move(monster, "RED_LOUSE_BITE")
        else:
            if roll < 25:
                if _last_move(monster, "GREEN_LOUSE_SPIT_WEB") and (_monster_ascension(monster) >= 17 or _last_two_moves(monster, "GREEN_LOUSE_SPIT_WEB")):
                    _set_move(monster, "GREEN_LOUSE_BITE")
                else:
                    _set_move(monster, "GREEN_LOUSE_SPIT_WEB")
            elif _last_two_moves(monster, "GREEN_LOUSE_BITE"):
                _set_move(monster, "GREEN_LOUSE_SPIT_WEB")
            else:
                _set_move(monster, "GREEN_LOUSE_BITE")
        return

    if monster.monster_id == "FungiBeast":
        if roll < 60:
            if _last_two_moves(monster, "FUNGI_BEAST_BITE"):
                _set_move(monster, "FUNGI_BEAST_GROW")
            else:
                _set_move(monster, "FUNGI_BEAST_BITE")
        elif _last_move(monster, "FUNGI_BEAST_GROW"):
            _set_move(monster, "FUNGI_BEAST_BITE")
        else:
            _set_move(monster, "FUNGI_BEAST_GROW")
        return

    if monster.monster_id == "SlaverBlue":
        if roll >= 40 and not _last_two_moves(monster, "BLUE_SLAVER_STAB"):
            _set_move(monster, "BLUE_SLAVER_STAB")
        elif not _last_two_moves(monster, "BLUE_SLAVER_RAKE") or (_monster_ascension(monster) >= 17 and not _last_move(monster, "BLUE_SLAVER_RAKE")):
            _set_move(monster, "BLUE_SLAVER_RAKE")
        else:
            _set_move(monster, "BLUE_SLAVER_STAB")
        return

    if monster.monster_id == "SlaverRed":
        used_entangle = bool(monster.ai_state.get("used_entangle", 0))
        if not monster.move_history:
            _set_move(monster, "RED_SLAVER_STAB")
        elif roll >= 75 and not used_entangle:
            _set_move(monster, "RED_SLAVER_ENTANGLE")
        elif roll >= 50 and used_entangle and not _last_two_moves(monster, "RED_SLAVER_STAB"):
            _set_move(monster, "RED_SLAVER_STAB")
        elif not _last_two_moves(monster, "RED_SLAVER_SCRAPE") or (_monster_ascension(monster) >= 17 and not _last_move(monster, "RED_SLAVER_SCRAPE")):
            _set_move(monster, "RED_SLAVER_SCRAPE")
        else:
            _set_move(monster, "RED_SLAVER_STAB")
        return

    if monster.monster_id in {"Looter", "Mugger"}:
        if not monster.move_history:
            _set_move(monster, "LOOTER_MUG" if monster.monster_id == "Looter" else "MUGGER_MUG")
        elif _last_move(monster, "LOOTER_SMOKE_BOMB") or _last_move(monster, "MUGGER_SMOKE_BOMB"):
            _set_move(monster, "LOOTER_ESCAPE" if monster.monster_id == "Looter" else "MUGGER_ESCAPE")
        elif monster.monster_id == "Looter":
            _set_move(monster, "LOOTER_MUG")
        else:
            _set_move(monster, "MUGGER_MUG")
        return

    if monster.monster_id == "Bear":
        _set_move(monster, _random_move(monster, rng, ["BEAR_MAUL", "BEAR_LUNGE", "BEAR_BEAR_HUG"]))
        return

    if monster.monster_id == "Pointy":
        _set_move(monster, "POINTY_ATTACK")
        return

    if monster.monster_id == "Romeo":
        _set_move(monster, _random_move(monster, rng, ["ROMEO_AGONIZING_SLASH", "ROMEO_CROSS_SLASH", "ROMEO_MOCK"]))
        return

    if monster.monster_id == "GremlinWizard":
        charge = _advance_ai_counter(monster, "charge")
        _set_move(monster, "GREMLIN_WIZARD_ULTIMATE_BLAST" if charge >= 3 else "GREMLIN_WIZARD_CHARGING")
        if charge >= 3:
            monster.ai_state["charge"] = 0
        return

    if monster.monster_id in {"GremlinFat", "GremlinThief", "GremlinTsundere", "GremlinWarrior"}:
        move_by_id = {
            "GremlinFat": ["FAT_GREMLIN_SMASH"],
            "GremlinThief": ["SNEAKY_GREMLIN_PUNCTURE"],
            "GremlinTsundere": ["SHIELD_GREMLIN_PROTECT"],
            "GremlinWarrior": ["MAD_GREMLIN_SCRATCH"],
        }
        _set_move(monster, _random_move(monster, rng, move_by_id[monster.monster_id]))
        return

    if monster.monster_id == "GremlinNob":
        if not monster.move_history:
            _set_move(monster, "GREMLIN_NOB_BELLOW")
        elif _monster_ascension(monster) >= 18:
            if not _last_two_moves(monster, "GREMLIN_NOB_SKULL_BASH"):
                _set_move(monster, "GREMLIN_NOB_RUSH")
            elif _last_two_moves(monster, "GREMLIN_NOB_RUSH"):
                _set_move(monster, "GREMLIN_NOB_SKULL_BASH")
            else:
                _set_move(monster, "GREMLIN_NOB_RUSH")
        elif roll < 33 or _last_two_moves(monster, "GREMLIN_NOB_RUSH"):
            _set_move(monster, "GREMLIN_NOB_SKULL_BASH")
        else:
            _set_move(monster, "GREMLIN_NOB_RUSH")
        return

    if monster.monster_id == "Lagavulin":
        if monster.ai_state.get("asleep", 0) and not monster.ai_state.get("latent_awake", 0):
            _set_move(monster, "LAGAVULIN_SLEEP")
        elif not monster.move_history:
            _set_move(monster, "LAGAVULIN_SIPHON_SOUL")
        elif _last_move(monster, "LAGAVULIN_SIPHON_SOUL"):
            _set_move(monster, "LAGAVULIN_ATTACK")
        elif _last_two_moves(monster, "LAGAVULIN_ATTACK"):
            _set_move(monster, "LAGAVULIN_SIPHON_SOUL")
        else:
            _set_move(monster, "LAGAVULIN_ATTACK")
        return

    if monster.monster_id == "Sentry":
        if not monster.move_history:
            _set_move(monster, "SENTRY_BOLT" if int(monster.ai_state.get("spawn_index", 0)) % 2 == 0 else "SENTRY_BEAM")
        elif _last_move(monster, "SENTRY_BOLT"):
            _set_move(monster, "SENTRY_BEAM")
        else:
            _set_move(monster, "SENTRY_BOLT")
        return

    if monster.monster_id == "SlimeBoss":
        if monster.current_hp > 0 and monster.current_hp <= monster.max_hp // 2:
            _set_move(monster, "SLIME_BOSS_SPLIT")
        elif not monster.move_history:
            _set_move(monster, "SLIME_BOSS_GOOP_SPRAY")
        elif _last_move(monster, "SLIME_BOSS_GOOP_SPRAY"):
            _set_move(monster, "SLIME_BOSS_PREPARING")
        elif _last_move(monster, "SLIME_BOSS_PREPARING"):
            _set_move(monster, "SLIME_BOSS_SLAM")
        else:
            _set_move(monster, "SLIME_BOSS_GOOP_SPRAY")
            return
        return

    if monster.monster_id == "Hexaghost":
        if not monster.move_history:
            _set_move(monster, "HEXAGHOST_ACTIVATE")
            return
        if monster.move == "HEXAGHOST_ACTIVATE":
            _set_move(monster, "HEXAGHOST_DIVIDER")
            return
        cycle = ["HEXAGHOST_SEAR", "HEXAGHOST_TACKLE", "HEXAGHOST_SEAR", "HEXAGHOST_INFERNO", "HEXAGHOST_TACKLE", "HEXAGHOST_SEAR"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "TheGuardian":
        cycle = ["THE_GUARDIAN_CHARGING_UP", "THE_GUARDIAN_FIERCE_BASH", "THE_GUARDIAN_VENT_STEAM"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "Byrd":
        if monster.move == "BYRD_STUNNED":
            _set_move(monster, "BYRD_CAW")
            return
        _set_move(monster, _random_move(monster, rng, ["BYRD_CAW", "BYRD_PECK", "BYRD_HEADBUTT"]))
        return

    if monster.monster_id == "Chosen":
        if _monster_ascension(monster) >= 17:
            if not monster.move_history:
                _set_move(monster, "CHOSEN_HEX")
            elif not _last_move(monster, "CHOSEN_DEBILITATE") and not _last_move(monster, "CHOSEN_DRAIN"):
                _set_move(monster, "CHOSEN_DEBILITATE" if roll < 50 else "CHOSEN_DRAIN")
            else:
                _set_move(monster, "CHOSEN_ZAP" if roll < 40 else "CHOSEN_POKE")
        else:
            if not monster.move_history:
                _set_move(monster, "CHOSEN_POKE")
            elif len(monster.move_history) == 1:
                _set_move(monster, "CHOSEN_HEX")
            elif not _last_move(monster, "CHOSEN_DEBILITATE") and not _last_move(monster, "CHOSEN_DRAIN"):
                _set_move(monster, "CHOSEN_DEBILITATE" if roll < 50 else "CHOSEN_DRAIN")
            else:
                _set_move(monster, "CHOSEN_ZAP" if roll < 40 else "CHOSEN_POKE")
        return

    if monster.monster_id == "SphericGuardian":
        if not monster.move_history:
            _set_move(monster, "SPHERIC_GUARDIAN_ACTIVATE")
        elif monster.move == "SPHERIC_GUARDIAN_ACTIVATE":
            _set_move(monster, "SPHERIC_GUARDIAN_ATTACK_DEBUFF")
        elif monster.move in {"SPHERIC_GUARDIAN_ATTACK_DEBUFF", "SPHERIC_GUARDIAN_HARDEN"}:
            _set_move(monster, "SPHERIC_GUARDIAN_SLAM")
        else:
            _set_move(monster, "SPHERIC_GUARDIAN_HARDEN")
        return

    if monster.monster_id == "SnakePlant":
        _set_move(monster, _random_move(monster, rng, ["SNAKE_PLANT_CHOMP", "SNAKE_PLANT_ENFEEBLING_SPORES"]))
        return

    if monster.monster_id == "Snecko":
        _set_move(monster, _random_move(monster, rng, ["SNECKO_TAIL_WHIP", "SNECKO_BITE", "SNECKO_PERPLEXING_GLARE"]))
        return

    if monster.monster_id == "ShelledParasite":
        if not monster.move_history:
            if _monster_ascension(monster) >= 17:
                _set_move(monster, "SHELLED_FELL")
            else:
                _set_move(monster, "SHELLED_DOUBLE_STRIKE" if rng.random_boolean() else "SHELLED_SUCK")
            return
        roll2 = 100
        if roll < 20:
            if not _last_move(monster, "SHELLED_FELL"):
                _set_move(monster, "SHELLED_FELL")
                return
            try:
                roll2 = int(rng.random(20, 99))
            except TypeError:
                roll2 = 20 + int(rng.random() * 80)
        if roll < 60 or roll2 < 60:
            if not _last_two_moves(monster, "SHELLED_DOUBLE_STRIKE"):
                _set_move(monster, "SHELLED_DOUBLE_STRIKE")
            else:
                _set_move(monster, "SHELLED_SUCK")
        elif not _last_two_moves(monster, "SHELLED_SUCK"):
            _set_move(monster, "SHELLED_SUCK")
        else:
            _set_move(monster, "SHELLED_DOUBLE_STRIKE")
        return

    if monster.monster_id == "Centurion":
        mystic_alive = any(ally.alive and ally.monster_id == "Mystic" for ally in getattr(monster, "_group_ref", []) or [])
        if roll >= 65 and not _last_two_moves(monster, "CENTURION_DEFEND") and not _last_two_moves(monster, "CENTURION_FURY"):
            _set_move(monster, "CENTURION_DEFEND" if mystic_alive else "CENTURION_FURY")
        elif not _last_two_moves(monster, "CENTURION_SLASH"):
            _set_move(monster, "CENTURION_SLASH")
        else:
            _set_move(monster, "CENTURION_DEFEND" if mystic_alive else "CENTURION_FURY")
        return

    if monster.monster_id == "Mystic":
        centurion = None
        for ally in getattr(monster, "_group_ref", []) or []:
            if ally.monster_id == "Centurion":
                centurion = ally
                break
        heal_need = 21 if _monster_ascension(monster) >= 17 else 16
        centurion_needs_heal = centurion is not None and centurion.alive and (centurion.max_hp - centurion.current_hp >= heal_need)
        if monster.max_hp - monster.current_hp >= heal_need or centurion_needs_heal:
            _set_move(monster, "MYSTIC_HEAL")
        elif roll >= 40 and (not _last_move(monster, "MYSTIC_ATTACK") if _monster_ascension(monster) >= 17 else not _last_two_moves(monster, "MYSTIC_ATTACK")):
            _set_move(monster, "MYSTIC_ATTACK")
        elif not _last_two_moves(monster, "MYSTIC_BUFF"):
            _set_move(monster, "MYSTIC_BUFF")
        else:
            _set_move(monster, "MYSTIC_ATTACK")
        return

    if monster.monster_id == "BookOfStabbing":
        turn = _advance_ai_counter(monster, "turn")
        if turn % 4 == 0:
            _set_move(monster, "BOOK_BIG_STAB")
        else:
            hits = min(6, 2 + int(monster.ai_state.get("multi_stab_bonus", 0)))
            monster.ai_state["multi_stab_bonus"] = int(monster.ai_state.get("multi_stab_bonus", 0)) + 1
            _set_move(monster, "BOOK_MULTI_STAB")
            monster.move_hits = hits
        return

    if monster.monster_id == "GremlinLeader":
        cycle = ["GREMLIN_LEADER_RALLY", "GREMLIN_LEADER_ENCOURAGE", "GREMLIN_LEADER_STAB"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "Taskmaster":
        _set_move(monster, "TASKMASTER_SCOURING_WHIP")
        return

    if monster.monster_id == "TheChamp":
        if monster.current_hp <= monster.max_hp // 2 and not monster.ai_state.get("phase2", 0):
            monster.ai_state["phase2"] = 1
            _set_move(monster, "CHAMP_DEFENSIVE_STANCE")
            return
        cycle = ["CHAMP_FACE_SLAP", "CHAMP_TAUNT", "CHAMP_EXECUTE"] if monster.ai_state.get("phase2", 0) else ["CHAMP_TAUNT", "CHAMP_FACE_SLAP"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "TheCollector":
        cycle = ["COLLECTOR_BUFF", "COLLECTOR_FIREBALL", "COLLECTOR_MEGA_DEBUFF", "COLLECTOR_FIREBALL"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "TorchHead":
        _set_move(monster, "TORCH_HEAD_TACKLE")
        return

    if monster.monster_id == "BronzeOrb":
        _set_move(monster, _random_move(monster, rng, ["BRONZE_ORB_BEAM", "BRONZE_ORB_STASIS"]))
        return

    if monster.monster_id == "BronzeAutomaton":
        cycle = ["AUTOMATON_BOOST", "AUTOMATON_FLAIL", "AUTOMATON_FLAIL", "AUTOMATON_HYPER_BEAM"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "Darkling":
        _set_move(monster, _random_move(monster, rng, ["DARKLING_NIP", "DARKLING_CHOMP", "DARKLING_REINCARNATE"]))
        return

    if monster.monster_id == "OrbWalker":
        _set_move(monster, _random_move(monster, rng, ["ORB_WALKER_LASER", "ORB_WALKER_CLAW", "ORB_WALKER_CHARGE"]))
        return

    if monster.monster_id == "TheMaw":
        cycle = ["MAW_ROAR", "MAW_DROOL", "MAW_SLAM", "MAW_NOM"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "Transient":
        turn = _advance_ai_counter(monster, "turn")
        _set_move(monster, "TRANSIENT_ATTACK")
        monster.move_base_damage = 30 + max(0, turn - 1) * 10
        return

    if monster.monster_id == "WrithingMass":
        _set_move(monster, _random_move(monster, rng, ["WRITHING_FLAIL", "WRITHING_IMPLANT", "WRITHING_WITHER"]))
        return

    if monster.monster_id == "Exploder":
        turn = _advance_ai_counter(monster, "turn")
        _set_move(monster, "EXPLODER_EXPLODE" if turn >= 3 else "EXPLODER_SLAM")
        return

    if monster.monster_id in {"Spiker", "Repulsor"}:
        move_by_id = {
            "Spiker": ["SPIKER_CUT", "SPIKER_GROW"],
            "Repulsor": ["REPULSOR_REPULSE", "REPULSOR_ATTACK"],
        }
        _set_move(monster, _random_move(monster, rng, move_by_id[monster.monster_id]))
        return

    if monster.monster_id == "Nemesis":
        turn = _advance_ai_counter(monster, "turn")
        monster.ai_state["intangible_next"] = 1 if turn % 2 == 1 else 0
        _set_move(monster, _random_move(monster, rng, ["NEMESIS_ATTACK", "NEMESIS_SCYTHE", "NEMESIS_DEBUFF"]))
        return

    if monster.monster_id == "GiantHead":
        turn = _advance_ai_counter(monster, "turn")
        _set_move(monster, "GIANT_HEAD_IT_IS_TIME" if turn >= 5 else "GIANT_HEAD_COUNT")
        return

    if monster.monster_id == "Reptomancer":
        _set_move(monster, _random_move(monster, rng, ["REPTOMANCER_SUMMON", "REPTOMANCER_STAB", "REPTOMANCER_BIG_STAB"]))
        return

    if monster.monster_id == "SnakeDagger":
        turn = _advance_ai_counter(monster, "turn")
        _set_move(monster, "DAGGER_EXPLODE" if turn >= 2 else "DAGGER_STAB")
        return

    if monster.monster_id == "AwakenedOne":
        cycle = ["AWAKENED_SLASH", "AWAKENED_SOUL_STRIKE"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "TimeEater":
        if monster.current_hp <= monster.max_hp // 2 and not monster.ai_state.get("healed", 0):
            monster.ai_state["healed"] = 1
            _set_move(monster, "TIME_EATER_RIPPLE")
            return
        cycle = ["TIME_EATER_REVERBERATE", "TIME_EATER_HEAD_SLAM"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "Donu":
        cycle = ["DONU_CIRCLE_OF_POWER", "DONU_BEAM"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "Deca":
        cycle = ["DECA_BEAM", "DECA_SQUARE_OF_PROTECTION"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "SpireShield":
        cycle = ["SHIELD_BASH", "SHIELD_FORTIFY", "SHIELD_SMASH"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "SpireSpear":
        cycle = ["SPEAR_BURN_STRIKE", "SPEAR_SKEWER", "SPEAR_PIERCER"]
        _set_move(monster, _cycle_move(monster, cycle))
        return

    if monster.monster_id == "CorruptHeart":
        if monster.move == "HEART_DEBILITATE":
            _set_move(monster, "HEART_BLOOD_SHOTS")
            return
        cycle = ["HEART_ECHO", "HEART_BLOOD_SHOTS"]
        _set_move(monster, _cycle_move(monster, cycle))
        return


def _advance_ai_counter(monster: MonsterState, key: str) -> int:
    monster.ai_state[key] = int(monster.ai_state.get(key, 0)) + 1
    return monster.ai_state[key]


def _monster_ascension(monster: MonsterState) -> int:
    return int(monster.ai_state.get("ascension_level", 0))


def _last_move(monster: MonsterState, move: str) -> bool:
    return bool(monster.move_history) and monster.move_history[0] == move


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
        "BYRD_CAW", "BYRD_FLY", "MYSTIC_BUFF", "GREMLIN_LEADER_ENCOURAGE",
        "GREMLIN_LEADER_RALLY", "CHAMP_DEFENSIVE_STANCE", "COLLECTOR_BUFF", "AUTOMATON_BOOST",
        "DARKLING_REINCARNATE", "ORB_WALKER_CHARGE", "MAW_ROAR", "SPIKER_GROW",
        "GIANT_HEAD_COUNT", "REPTOMANCER_SUMMON", "AWAKENED_REBIRTH", "DONU_CIRCLE_OF_POWER",
        "RED_LOUSE_GROW",
        "CULTIST_INCANTATION", "GREMLIN_NOB_BELLOW", "FUNGI_BEAST_GROW",
    }:
        monster.intent = "BUFF"
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
        "SHIELD_FORTIFY", "SHIELD_GREMLIN_PROTECT",
    }:
        monster.intent = "DEFEND"
        monster.move_base_damage = 0
        monster.move_hits = 0
        return
    elif move in {
        "CHOSEN_DRAIN", "SNAKE_PLANT_ENFEEBLING_SPORES", "SNECKO_PERPLEXING_GLARE",
        "COLLECTOR_MEGA_DEBUFF", "MAW_DROOL", "WRITHING_IMPLANT", "WRITHING_WITHER",
        "REPULSOR_REPULSE", "NEMESIS_DEBUFF", "TIME_EATER_RIPPLE",
        "HEART_DEBILITATE", "LAGAVULIN_SIPHON_SOUL", "BEAR_BEAR_HUG", "GREEN_LOUSE_SPIT_WEB",
    }:
        monster.intent = "STRONG_DEBUFF" if move in {"CHOSEN_DRAIN", "LAGAVULIN_SIPHON_SOUL", "HEART_DEBILITATE"} else "DEBUFF"
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
            "CHAMP_FACE_SLAP": 12,
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
        "CENTURION_FURY", "BOOK_MULTI_STAB", "BOOK_BIG_STAB",
        "GREMLIN_LEADER_STAB", "TASKMASTER_SCOURING_WHIP", "CHAMP_EXECUTE", "CHAMP_TAUNT",
        "COLLECTOR_FIREBALL", "AUTOMATON_FLAIL", "AUTOMATON_HYPER_BEAM", "DARKLING_NIP",
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
            "BOOK_MULTI_STAB": (3, 6),
            "BOOK_BIG_STAB": (1, 21),
            "GREMLIN_LEADER_STAB": (3, 6),
            "TASKMASTER_SCOURING_WHIP": (1, 7),
            "CHAMP_EXECUTE": (2, 10),
            "CHAMP_TAUNT": (1, 18),
            "COLLECTOR_FIREBALL": (1, 18),
            "AUTOMATON_FLAIL": (2, 8),
            "AUTOMATON_HYPER_BEAM": (2, 26),
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
            "SHIELD_GREMLIN_SHIELD_BASH": (1, 6),
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
        "MYSTIC_ATTACK", "MYSTIC_BUFF", "BOOK_MULTI_STAB", "BOOK_BIG_STAB",
        "GREMLIN_LEADER_ENCOURAGE", "GREMLIN_LEADER_STAB", "GREMLIN_LEADER_RALLY",
        "TASKMASTER_SCOURING_WHIP", "TASKMASTER_RAKE", "CHAMP_DEFENSIVE_STANCE",
        "CHAMP_FACE_SLAP", "CHAMP_EXECUTE", "CHAMP_TAUNT", "COLLECTOR_BUFF",
        "COLLECTOR_FIREBALL", "COLLECTOR_MEGA_DEBUFF", "AUTOMATON_BOOST", "AUTOMATON_FLAIL",
        "AUTOMATON_HYPER_BEAM", "DARKLING_NIP", "DARKLING_CHOMP", "DARKLING_REINCARNATE",
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
        "GREMLIN_NOB_RUSH", "LAGAVULIN_ATTACK", "SENTRY_BEAM", "LAGAVULIN_SLEEP",
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
