from __future__ import annotations

import random

from spirecomm.native_sim.schema import MonsterState
from spirecomm.native_sim_v2.monster_support import (
    ACT_BOSSES,
    ELITE_ENCOUNTERS,
    STRONG_ENCOUNTERS,
    STRONG_WEIGHTS,
    WEAK_ENCOUNTERS,
    WEAK_WEIGHTS,
    _last_move_before,
    _populate_first_strong_enemy,
    _populate_monster_list,
    _roll_louse,
    _last_move,
    _last_two_moves,
    _set_move,
    generate_elite_schedule,
    generate_monster_schedules,
    generate_strong_monster_schedule,
    monster_adjusted_damage,
    _rng_ascension_level,
)


MONSTER_HP_RANGES: dict[str, tuple[tuple[int, int], tuple[int, int] | None, int | None, tuple[int, int] | None]] = {
    "AcidSlime_L": ((65, 69), (68, 72), 7, None),
    "AcidSlime_M": ((28, 32), (29, 34), 7, None),
    "AcidSlime_S": ((8, 12), (9, 13), 7, None),
    "AwakenedOne": ((300, 300), (320, 320), 9, None),
    "Bear": ((38, 52), (40, 44), 7, None),
    "BookOfStabbing": ((160, 164), (168, 172), 8, None),
    "BronzeAutomaton": ((300, 300), (320, 320), 9, None),
    "BronzeOrb": ((52, 58), (54, 60), 9, (52, 58)),
    "Byrd": ((25, 31), (26, 33), 7, None),
    "Centurion": ((76, 80), (76, 83), 7, None),
    "Chosen": ((95, 99), (98, 103), 7, None),
    "CorruptHeart": ((750, 750), (800, 800), 9, None),
    "Cultist": ((48, 54), (50, 56), 7, None),
    "Darkling": ((48, 56), (50, 59), 7, None),
    "Deca": ((250, 250), (265, 265), 9, None),
    "Donu": ((250, 250), (265, 265), 9, None),
    "Exploder": ((30, 30), (30, 35), 7, None),
    "FungiBeast": ((22, 28), (24, 28), 7, None),
    "GiantHead": ((500, 500), (520, 520), 8, None),
    "GremlinFat": ((13, 17), (14, 18), 7, None),
    "GremlinLeader": ((140, 148), (145, 155), 8, None),
    "GremlinNob": ((82, 86), (85, 90), 8, None),
    "GremlinWizard": ((21, 25), (22, 26), 7, None),
    "GreenLouse": ((11, 17), (12, 18), 7, None),
    "Hexaghost": ((250, 250), (264, 264), 9, None),
    "JawWorm": ((40, 44), (42, 46), 7, None),
    "Lagavulin": ((109, 111), (112, 115), 8, None),
    "Looter": ((44, 48), (46, 50), 7, None),
    "MadGremlin": ((20, 24), (21, 25), 7, None),
    "Mugger": ((48, 52), (50, 54), 7, None),
    "Mystic": ((48, 56), (50, 58), 7, None),
    "Nemesis": ((185, 185), (200, 200), 8, None),
    "OrbWalker": ((90, 96), (92, 102), 7, (90, 96)),
    "Pointy": ((30, 30), (34, 34), 7, None),
    "RedLouse": ((10, 15), (11, 16), 7, None),
    "RedSlaver": ((46, 50), (48, 52), 7, None),
    "Reptomancer": ((180, 190), (190, 200), 8, (180, 190)),
    "Repulsor": ((29, 35), (31, 38), 7, None),
    "Romeo": ((35, 39), (37, 41), 7, None),
    "Sentry": ((38, 42), (39, 45), 8, None),
    "ShelledParasite": ((68, 72), (70, 75), 7, None),
    "SlaverBlue": ((46, 50), (48, 52), 7, None),
    "SlimeBoss": ((140, 140), (150, 150), 9, None),
    "SnakeDagger": ((20, 25), (20, 25), 8, None),
    "SnakePlant": ((75, 79), (78, 82), 7, None),
    "Snecko": ((114, 120), (120, 125), 7, None),
    "SphericGuardian": ((20, 20), None, None, None),
    "Spiker": ((42, 56), (44, 60), 7, None),
    "SpikeSlime_L": ((64, 70), (67, 73), 7, None),
    "SpikeSlime_M": ((28, 32), (29, 34), 7, None),
    "SpikeSlime_S": ((10, 14), (11, 15), 7, None),
    "SpireGrowth": ((170, 170), (190, 190), 7, None),
    "SpireShield": ((110, 110), (125, 125), 8, None),
    "SpireSpear": ((160, 160), (180, 180), 8, None),
    "Taskmaster": ((54, 60), (57, 64), 8, (54, 60)),
    "TheChamp": ((420, 420), (440, 440), 9, None),
    "TheCollector": ((282, 282), (300, 300), 9, None),
    "TheGuardian": ((240, 240), (250, 250), 9, None),
    "TheMaw": ((300, 300), None, None, None),
    "TimeEater": ((456, 456), (480, 480), 9, None),
    "TorchHead": ((38, 40), (40, 45), 9, None),
    "Transient": ((999, 999), None, None, None),
    "WrithingMass": ((160, 160), (175, 175), 7, None),
}


def _hp_spec(monster_id: str) -> tuple[tuple[int, int], tuple[int, int] | None, int | None, tuple[int, int] | None]:
    aliases = {
        "GremlinWarrior": "MadGremlin",
        "GremlinThief": "SneakyGremlin",
        "GremlinTsundere": "ShieldGremlin",
        "LagavulinEvent": "Lagavulin",
        "SlaverRed": "RedSlaver",
    }
    canonical_id = aliases.get(monster_id, monster_id)
    if canonical_id == "SneakyGremlin":
        return ((10, 14), (11, 15), 7, None)
    if canonical_id == "ShieldGremlin":
        return ((12, 15), (13, 17), 7, None)
    return MONSTER_HP_RANGES.get(canonical_id, ((0, 0), None, None, None))


def _advance_ai_counter(monster: MonsterState, key: str) -> int:
    monster.ai_state[key] = int(monster.ai_state.get(key, 0)) + 1
    return monster.ai_state[key]


def _monster_ascension(monster: MonsterState) -> int:
    return int(monster.ai_state.get("ascension_level", 0))


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


def _make_monster_base(monster_id: str, rng: random.Random, ascension: int = 0) -> MonsterState:
    if monster_id == "INVALID = 0":
        return MonsterState("INVALID = 0", "INVALID = 0", 0, 0, "INVALID", "UNKNOWN")
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
        return MonsterState("SlimeBoss", "Slime Boss", 140, 140, "SLIME_BOSS_GOOP_SPRAY", "STRONG_DEBUFF")
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
        monster = MonsterState("TheGuardian", "The Guardian", 240, 240, "THE_GUARDIAN_CHARGING_UP", "DEFEND")
        monster.add_power("Mode Shift", 30)
        monster.ai_state["mode_shift_amount"] = 30
        monster.ai_state["mode_shift_overflow"] = 0
        return monster
    if monster_id == "Byrd":
        hp = rng.randint(25, 31)
        monster = MonsterState("Byrd", "Byrd", hp, hp, "BYRD_CAW", "BUFF")
        monster.add_power("Flight", 4 if ascension >= 17 else 3)
        return monster
    if monster_id == "Chosen":
        hp = rng.randint(95, 99)
        return MonsterState("Chosen", "Chosen", hp, hp, "CHOSEN_POKE", "ATTACK", move_base_damage=6 if ascension >= 2 else 5, move_hits=2)
    if monster_id == "SphericGuardian":
        monster = MonsterState("SphericGuardian", "Spheric Guardian", 20, 20, "SPHERIC_GUARDIAN_ACTIVATE", "DEFEND", move_base_damage=0, move_hits=0)
        monster.block = 40
        monster.add_power("Barricade", 1)
        monster.add_power("Artifact", 3)
        return monster
    if monster_id == "SnakePlant":
        hp = rng.randint(78, 82)
        monster = MonsterState("SnakePlant", "Snake Plant", hp, hp, "SNAKE_PLANT_CHOMP", "ATTACK", move_base_damage=8 if ascension >= 2 else 7, move_hits=3)
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
        monster = MonsterState("BookOfStabbing", "Book of Stabbing", hp, hp, "BOOK_MULTI_STAB", "ATTACK", move_base_damage=6, move_hits=2)
        monster.ai_state["stab_count"] = 1
        # Lightspeed models Book of Stabbing's hidden Painful Stabs power
        # separately from serialized monster powers. We keep the semantic flag
        # in ai_state so combat logic can inject Wounds without exposing an
        # extra visible power in battle-state comparisons.
        monster.ai_state["painful_stabs"] = 1
        return monster
    if monster_id == "GremlinLeader":
        hp = rng.randint(148, 155)
        return MonsterState("GremlinLeader", "Gremlin Leader", hp, hp, "GREMLIN_LEADER_ENCOURAGE", "BUFF")
    if monster_id == "Taskmaster":
        rng.randint(54, 60)
        hp = rng.randint(57, 64) if ascension >= 8 else rng.randint(54, 60)
        return MonsterState("Taskmaster", "Taskmaster", hp, hp, "TASKMASTER_SCOURING_WHIP", "ATTACK_DEBUFF", move_base_damage=7, move_hits=1)
    if monster_id == "TheChamp":
        monster = MonsterState("TheChamp", "The Champ", 440, 440, "CHAMP_HEAVY_SLASH", "ATTACK")
        monster.ai_state["champ_num_turns"] = 0
        monster.ai_state["champ_forge_times"] = 0
        monster.ai_state["champ_forge_threshold"] = 2
        monster.ai_state["champ_threshold_reached"] = 0
        return monster
    if monster_id == "TheCollector":
        monster = MonsterState("TheCollector", "The Collector", 300, 300, "THE_COLLECTOR_SPAWN", "MAGIC")
        monster.ai_state["collector_turn"] = 1
        monster.ai_state["fixed_opening_move"] = 1
        return monster
    if monster_id == "TorchHead":
        hp = rng.randint(38, 40)
        return MonsterState("TorchHead", "Torch Head", hp, hp, "TORCH_HEAD_TACKLE", "ATTACK", move_base_damage=7, move_hits=1)
    if monster_id == "BronzeOrb":
        hp = rng.randint(52, 58)
        return MonsterState("BronzeOrb", "Bronze Orb", hp, hp, "BRONZE_ORB_BEAM", "ATTACK", move_base_damage=8, move_hits=1)
    if monster_id == "BronzeAutomaton":
        return MonsterState("BronzeAutomaton", "Bronze Automaton", 300, 300, "BRONZE_AUTOMATON_SPAWN_ORBS", "MAGIC")
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
        monster = MonsterState("Transient", "Transient", 999, 999, "TRANSIENT_ATTACK", "ATTACK", move_base_damage=30, move_hits=1)
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
        monster = MonsterState("AwakenedOne", "Awakened One", 300, 300, "AWAKENED_SLASH", "ATTACK", move_base_damage=20, move_hits=1)
        monster.add_power("Curiosity", 1)
        return monster
    if monster_id == "TimeEater":
        return MonsterState("TimeEater", "Time Eater", 456, 456, "TIME_EATER_REVERBERATE", "ATTACK", move_base_damage=7, move_hits=3)
    if monster_id == "Donu":
        return MonsterState("Donu", "Donu", 265, 265, "DONU_CIRCLE_OF_POWER", "BUFF")
    if monster_id == "Deca":
        return MonsterState("Deca", "Deca", 265, 265, "DECA_BEAM", "ATTACK", move_base_damage=10, move_hits=2)
    if monster_id == "SpireShield":
        monster = MonsterState("SpireShield", "Spire Shield", 125, 125, "SHIELD_BASH", "ATTACK_DEFEND", move_base_damage=12, move_hits=1)
        monster.block = 30
        return monster
    if monster_id == "SpireSpear":
        return MonsterState("SpireSpear", "Spire Spear", 180, 180, "SPEAR_BURN_STRIKE", "ATTACK_DEBUFF", move_base_damage=6, move_hits=2)
    if monster_id == "CorruptHeart":
        monster = MonsterState("CorruptHeart", "Corrupt Heart", 750, 750, "HEART_DEBILITATE", "STRONG_DEBUFF")
        monster.add_power("Beat of Death", 1)
        return monster
    raise KeyError(f"unsupported native simulator monster: {monster_id}")


def _copy_rng_for_builder(rng: random.Random, monster_id: str) -> random.Random:
    builder_rng = rng.copy() if hasattr(rng, "copy") else rng
    _, _, _, pre_roll = _hp_spec(monster_id)
    if pre_roll is not None and hasattr(builder_rng, "random"):
        builder_rng.random(pre_roll[0], pre_roll[1])
    return builder_rng


def _consume_lightspeed_hp_roll(rng: random.Random, monster_id: str, ascension: int) -> int:
    low_range, high_range, threshold, pre_roll = _hp_spec(monster_id)
    if pre_roll is not None:
        rng.random(pre_roll[0], pre_roll[1])
    if high_range is None or threshold is None:
        return low_range[0]
    hp_range = high_range if ascension >= threshold else low_range
    return int(rng.random(hp_range[0], hp_range[1]))


def _consume_lightspeed_construct_rolls(monster, rng: random.Random, monster_id: str, ascension: int) -> None:
    canonical_id = {
        "GremlinWarrior": "MadGremlin",
        "GremlinThief": "SneakyGremlin",
        "GremlinTsundere": "ShieldGremlin",
        "LagavulinEvent": "Lagavulin",
        "SlaverRed": "RedSlaver",
    }.get(monster_id, monster_id)
    if canonical_id in {"GreenLouse", "RedLouse"}:
        bite_damage = int(rng.random(6, 8) if ascension >= 2 else rng.random(5, 7))
        monster.ai_state["bite_damage"] = bite_damage
        if monster.monster_id == "RedLouse":
            monster.move_base_damage = bite_damage
    elif canonical_id == "Darkling":
        monster.ai_state["darkling_reincarnate_hp"] = int(rng.random(9, 13) if ascension >= 2 else rng.random(7, 11))


def make_monster(monster_id: str, rng: random.Random, ascension: int = 0):
    builder_rng = _copy_rng_for_builder(rng, monster_id)
    monster = _make_monster_base(monster_id, builder_rng, ascension)
    if _hp_spec(monster_id)[0] != (0, 0):
        hp = _consume_lightspeed_hp_roll(rng, monster_id, ascension)
        monster.max_hp = hp
        monster.current_hp = hp
        _consume_lightspeed_construct_rolls(monster, rng, monster_id, ascension)
    monster.ai_state["ascension_level"] = ascension
    return monster


def encounter_to_monsters(
    encounter: str,
    hp_rng: random.Random,
    misc_rng: random.Random | None = None,
    ascension: int = 0,
):
    misc_rng = hp_rng if misc_rng is None else misc_rng

    if encounter == "GremlinLeader":
        gremlin_pool = [
            "GremlinWarrior",
            "GremlinWarrior",
            "GremlinThief",
            "GremlinThief",
            "GremlinFat",
            "GremlinFat",
            "GremlinTsundere",
            "GremlinWizard",
        ]
        monsters = [
            make_monster("INVALID = 0", hp_rng, ascension),
            make_monster(gremlin_pool[int(misc_rng.random(7))], hp_rng, ascension),
            make_monster(gremlin_pool[int(misc_rng.random(7))], hp_rng, ascension),
            make_monster("GremlinLeader", hp_rng, ascension),
        ]
        for monster in monsters[1:3]:
            monster.ai_state["leader_minion"] = 1
        return monsters

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

    monster_ids = encounter_to_monster_ids(encounter, misc_rng)
    return [make_monster(monster_id, hp_rng, ascension) for monster_id in monster_ids]


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
        return ["Byrd", "Chosen"]
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
        return ["INVALID = 0", "BronzeAutomaton", "INVALID = 0"]
    if encounter == "TheCollector":
        return ["INVALID = 0", "INVALID = 0", "TheCollector"]
    if encounter == "GremlinLeader":
        gremlin_pool = [
            "GremlinWarrior",
            "GremlinWarrior",
            "GremlinThief",
            "GremlinThief",
            "GremlinFat",
            "GremlinFat",
            "GremlinTsundere",
            "GremlinWizard",
        ]
        return [
            "INVALID = 0",
            gremlin_pool[int(rng.random(7))],
            gremlin_pool[int(rng.random(7))],
            "GremlinLeader",
        ]
    if encounter in {
        "Cultist", "JawWorm", "AcidSlime_M", "SpikeSlime_M", "FungiBeast", "SlaverBlue", "Looter", "SlaverRed",
        "SphericGuardian", "Chosen", "ShelledParasite", "SnakePlant", "Snecko", "OrbWalker", "TheMaw", "Transient",
        "WrithingMass", "GremlinNob", "Lagavulin", "BookOfStabbing", "GiantHead", "Nemesis",
        "Hexaghost", "SlimeBoss", "TheGuardian", "TheChamp", "AwakenedOne", "TimeEater", "SpireGrowth",
    }:
        return [encounter]
    return ["Cultist"]


def roll_act1_encounter(rng: random.Random, floor: int, act: int = 1, act_boss: str = "Hexaghost", elite: bool = False):
    if floor == 53:
        return [make_monster("SpireShield", rng), make_monster("SpireSpear", rng)]
    if floor == 54:
        return [make_monster("CorruptHeart", rng)]
    if floor in {16, 33, 50}:
        if floor == 16:
            boss = {"Hexaghost": "Hexaghost", "Slime Boss": "SlimeBoss", "The Guardian": "TheGuardian"}.get(act_boss, "Hexaghost")
            return [make_monster(boss, rng)]
        if floor == 33:
            boss = {"The Champ": "TheChamp", "The Collector": "TheCollector", "Bronze Automaton": "BronzeAutomaton"}.get(
                act_boss,
                rng.choice(["TheChamp", "TheCollector", "BronzeAutomaton"]),
            )
            if boss == "TheCollector":
                return [
                    make_monster("INVALID = 0", rng),
                    make_monster("INVALID = 0", rng),
                    make_monster("TheCollector", rng),
                ]
            if boss == "BronzeAutomaton":
                return [make_monster("INVALID = 0", rng), make_monster("BronzeAutomaton", rng), make_monster("INVALID = 0", rng)]
            return [make_monster(boss, rng)]
        choice = {"Awakened One": "AwakenedOne", "Time Eater": "TimeEater", "Donu and Deca": "DonuDeca"}.get(
            act_boss,
            rng.choice(["AwakenedOne", "TimeEater", "DonuDeca"]),
        )
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
                leader = make_monster("GremlinLeader", rng)
                add_a = make_monster(rng.choice(smalls), rng)
                add_b = make_monster(rng.choice(smalls), rng)
                add_a.ai_state["leader_minion"] = 1
                add_b.ai_state["leader_minion"] = 1
                return [leader, add_a, add_b]
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


def choose_next_move(monster, rng: random.Random) -> None:
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
            # When retaliation damage pushes the large Acid Slime to its split
            # threshold, lightspeed still rolls a normal follow-up move and can
            # repeat TACKLE once more at A0.
            repeated_tackle_blocked = _last_move(monster, "ACID_SLIME_L_TACKLE") and monster.current_hp > monster.max_hp // 2
            if repeated_tackle_blocked:
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
        if not monster.move_history:
            _set_move(monster, "SLIME_BOSS_GOOP_SPRAY")
        elif _last_move(monster, "SLIME_BOSS_GOOP_SPRAY"):
            _set_move(monster, "SLIME_BOSS_PREPARING")
        elif _last_move(monster, "SLIME_BOSS_PREPARING"):
            _set_move(monster, "SLIME_BOSS_SLAM")
        else:
            _set_move(monster, "SLIME_BOSS_GOOP_SPRAY")
        return
    if monster.monster_id == "Hexaghost":
        if not monster.move_history:
            _set_move(monster, "HEXAGHOST_ACTIVATE")
            return
        if monster.move == "HEXAGHOST_ACTIVATE":
            _set_move(monster, "HEXAGHOST_DIVIDER")
            return
        _set_move(monster, _cycle_move(monster, ["HEXAGHOST_SEAR", "HEXAGHOST_TACKLE", "HEXAGHOST_SEAR", "HEXAGHOST_INFERNO", "HEXAGHOST_TACKLE", "HEXAGHOST_SEAR"]))
        return
    if monster.monster_id == "TheGuardian":
        _set_move(monster, _cycle_move(monster, ["THE_GUARDIAN_CHARGING_UP", "THE_GUARDIAN_FIERCE_BASH", "THE_GUARDIAN_VENT_STEAM"]))
        return
    if monster.monster_id == "Byrd":
        if monster.move == "BYRD_STUNNED":
            _set_move(monster, "BYRD_HEADBUTT")
            return
        if not monster.move_history:
            if rng.random_boolean(0.375):
                _set_move(monster, "BYRD_CAW")
            else:
                _set_move(monster, "BYRD_PECK")
            return
        if roll < 50:
            if _last_two_moves(monster, "BYRD_PECK"):
                if rng.random_boolean(0.4):
                    _set_move(monster, "BYRD_SWOOP")
                else:
                    _set_move(monster, "BYRD_CAW")
            else:
                _set_move(monster, "BYRD_PECK")
            return
        if roll < 70:
            if _last_move(monster, "BYRD_SWOOP"):
                if rng.random_boolean(0.375):
                    _set_move(monster, "BYRD_CAW")
                else:
                    _set_move(monster, "BYRD_PECK")
            else:
                _set_move(monster, "BYRD_SWOOP")
            return
        if _last_move(monster, "BYRD_CAW"):
            if rng.random_boolean(0.2857):
                _set_move(monster, "BYRD_SWOOP")
            else:
                _set_move(monster, "BYRD_PECK")
        else:
            _set_move(monster, "BYRD_CAW")
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
        if _monster_ascension(monster) >= 17:
            if roll < 65:
                if _last_two_moves(monster, "SNAKE_PLANT_CHOMP"):
                    _set_move(monster, "SNAKE_PLANT_ENFEEBLING_SPORES")
                else:
                    _set_move(monster, "SNAKE_PLANT_CHOMP")
            elif not _last_two_moves(monster, "SNAKE_PLANT_ENFEEBLING_SPORES"):
                _set_move(monster, "SNAKE_PLANT_ENFEEBLING_SPORES")
            else:
                _set_move(monster, "SNAKE_PLANT_CHOMP")
        else:
            if roll < 65:
                if _last_two_moves(monster, "SNAKE_PLANT_CHOMP"):
                    _set_move(monster, "SNAKE_PLANT_ENFEEBLING_SPORES")
                else:
                    _set_move(monster, "SNAKE_PLANT_CHOMP")
            elif _last_move(monster, "SNAKE_PLANT_ENFEEBLING_SPORES"):
                _set_move(monster, "SNAKE_PLANT_CHOMP")
            else:
                _set_move(monster, "SNAKE_PLANT_ENFEEBLING_SPORES")
        return
    if monster.monster_id == "Snecko":
        if not monster.move_history:
            _set_move(monster, "SNECKO_PERPLEXING_GLARE")
        else:
            if roll < 40 or _last_two_moves(monster, "SNECKO_BITE"):
                _set_move(monster, "SNECKO_TAIL_WHIP")
            else:
                _set_move(monster, "SNECKO_BITE")
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
        stab_count = int(monster.ai_state.get("stab_count", 1))
        if roll < 15:
            if _last_move(monster, "BOOK_OF_STABBING_SINGLE_STAB"):
                stab_count += 1
                monster.ai_state["stab_count"] = stab_count
                _set_move(monster, "BOOK_OF_STABBING_MULTI_STAB")
            else:
                _set_move(monster, "BOOK_OF_STABBING_SINGLE_STAB")
        elif _last_two_moves(monster, "BOOK_OF_STABBING_MULTI_STAB"):
            _set_move(monster, "BOOK_OF_STABBING_SINGLE_STAB")
        else:
            stab_count += 1
            monster.ai_state["stab_count"] = stab_count
            _set_move(monster, "BOOK_OF_STABBING_MULTI_STAB")
        return
    if monster.monster_id == "GremlinLeader":
        alive_adds = sum(
            1
            for ally in getattr(monster, "_group_ref", []) or []
            if ally is not monster and ally.alive and ally.monster_id not in {"INVALID = 0", "GremlinLeader"}
        )
        if alive_adds == 0:
            if roll < 75:
                _set_move(monster, "GREMLIN_LEADER_STAB" if _last_move(monster, "GREMLIN_LEADER_RALLY") else "GREMLIN_LEADER_RALLY")
            elif _last_move(monster, "GREMLIN_LEADER_STAB"):
                _set_move(monster, "GREMLIN_LEADER_RALLY")
            else:
                _set_move(monster, "GREMLIN_LEADER_STAB")
            return
        if alive_adds == 1:
            if roll < 50:
                if _last_move(monster, "GREMLIN_LEADER_RALLY"):
                    roll2 = int(rng.random(50, 99))
                    _set_move(monster, "GREMLIN_LEADER_ENCOURAGE" if roll2 < 80 else "GREMLIN_LEADER_STAB")
                else:
                    _set_move(monster, "GREMLIN_LEADER_RALLY")
            elif roll < 80:
                _set_move(monster, "GREMLIN_LEADER_STAB" if _last_move(monster, "GREMLIN_LEADER_ENCOURAGE") else "GREMLIN_LEADER_ENCOURAGE")
            elif _last_move(monster, "GREMLIN_LEADER_STAB"):
                roll2 = int(rng.random(0, 80))
                _set_move(monster, "GREMLIN_LEADER_RALLY" if roll2 < 50 else "GREMLIN_LEADER_ENCOURAGE")
            else:
                _set_move(monster, "GREMLIN_LEADER_STAB")
            return
        if roll < 66:
            _set_move(monster, "GREMLIN_LEADER_STAB" if _last_move(monster, "GREMLIN_LEADER_ENCOURAGE") else "GREMLIN_LEADER_ENCOURAGE")
        elif _last_move(monster, "GREMLIN_LEADER_STAB"):
            _set_move(monster, "GREMLIN_LEADER_ENCOURAGE")
        else:
            _set_move(monster, "GREMLIN_LEADER_STAB")
        return
    if monster.monster_id == "Taskmaster":
        _set_move(monster, "TASKMASTER_SCOURING_WHIP")
        return
    if monster.monster_id == "TheChamp":
        monster.ai_state["champ_num_turns"] = int(monster.ai_state.get("champ_num_turns", 0)) + 1
        threshold_reached = bool(monster.ai_state.get("champ_threshold_reached", 0))
        if monster.current_hp < monster.max_hp / 2 and not threshold_reached:
            monster.ai_state["champ_threshold_reached"] = 1
            _set_move(monster, "CHAMP_ANGER")
            return
        if threshold_reached and not _last_move(monster, "CHAMP_EXECUTE") and not _last_move_before(monster, "CHAMP_EXECUTE"):
            _set_move(monster, "CHAMP_EXECUTE")
            return
        if int(monster.ai_state.get("champ_num_turns", 0)) == 4 and not threshold_reached:
            monster.ai_state["champ_num_turns"] = 0
            _set_move(monster, "CHAMP_TAUNT")
            return
        forge_times = int(monster.ai_state.get("champ_forge_times", 0))
        forge_threshold = int(monster.ai_state.get("champ_forge_threshold", 2))
        forge_roll_threshold = 30 if _monster_ascension(monster) >= 19 else 15
        if not _last_move(monster, "CHAMP_DEFENSIVE_STANCE") and forge_times < forge_threshold and roll <= forge_roll_threshold:
            monster.ai_state["champ_forge_times"] = forge_times + 1
            _set_move(monster, "CHAMP_DEFENSIVE_STANCE")
            return
        if not _last_move(monster, "CHAMP_GLOAT") and not _last_move(monster, "CHAMP_DEFENSIVE_STANCE") and roll <= 30:
            _set_move(monster, "CHAMP_GLOAT")
            return
        if not _last_move(monster, "CHAMP_FACE_SLAP") and roll <= 55:
            _set_move(monster, "CHAMP_FACE_SLAP")
            return
        if not _last_move(monster, "CHAMP_HEAVY_SLASH"):
            _set_move(monster, "CHAMP_HEAVY_SLASH")
            return
        _set_move(monster, "CHAMP_FACE_SLAP")
        return
    if monster.monster_id == "TheCollector":
        if (
            not monster.move_history
            and monster.move == "THE_COLLECTOR_SPAWN"
            and int(monster.ai_state.get("collector_turn", 0)) == 1
        ):
            _set_move(monster, "THE_COLLECTOR_SPAWN")
            return
        turn = int(monster.ai_state.get("collector_turn", 0)) + 1
        monster.ai_state["collector_turn"] = turn
        if turn == 4:
            _set_move(monster, "COLLECTOR_MEGA_DEBUFF")
            return
        alive_count = sum(1 for ally in getattr(monster, "_group_ref", []) or [] if ally.alive and ally.monster_id != "INVALID = 0")
        can_use_spawn = alive_count < 3 and not _last_move(monster, "THE_COLLECTOR_SPAWN")
        if roll <= 25 and can_use_spawn:
            _set_move(monster, "THE_COLLECTOR_SPAWN")
        elif roll <= 70 and not _last_two_moves(monster, "COLLECTOR_FIREBALL"):
            _set_move(monster, "COLLECTOR_FIREBALL")
        elif _last_move(monster, "COLLECTOR_BUFF"):
            _set_move(monster, "COLLECTOR_FIREBALL")
        else:
            _set_move(monster, "COLLECTOR_BUFF")
        return
    if monster.monster_id == "TorchHead":
        _set_move(monster, "TORCH_HEAD_TACKLE")
        return
    if monster.monster_id == "BronzeOrb":
        used_stasis = int(monster.ai_state.get("bronze_orb_used_stasis", 0))
        if not used_stasis and roll >= 25:
            _set_move(monster, "BRONZE_ORB_STASIS")
        elif roll >= 70 and not _last_two_moves(monster, "BRONZE_ORB_SUPPORT_BEAM"):
            _set_move(monster, "BRONZE_ORB_SUPPORT_BEAM")
        elif not _last_two_moves(monster, "BRONZE_ORB_BEAM"):
            _set_move(monster, "BRONZE_ORB_BEAM")
        else:
            _set_move(monster, "BRONZE_ORB_SUPPORT_BEAM")
        return
    if monster.monster_id == "BronzeAutomaton":
        if not monster.move_history:
            _set_move(monster, "BRONZE_AUTOMATON_SPAWN_ORBS")
        else:
            _set_move(monster, _cycle_move(monster, ["BRONZE_AUTOMATON_BOOST", "BRONZE_AUTOMATON_FLAIL", "BRONZE_AUTOMATON_FLAIL", "BRONZE_AUTOMATON_HYPER_BEAM"]))
        return
    if monster.monster_id == "Darkling":
        _set_move(monster, _random_move(monster, rng, ["DARKLING_NIP", "DARKLING_CHOMP", "DARKLING_REINCARNATE"]))
        return
    if monster.monster_id == "OrbWalker":
        _set_move(monster, _random_move(monster, rng, ["ORB_WALKER_LASER", "ORB_WALKER_CLAW", "ORB_WALKER_CHARGE"]))
        return
    if monster.monster_id == "TheMaw":
        _set_move(monster, _cycle_move(monster, ["MAW_ROAR", "MAW_DROOL", "MAW_SLAM", "MAW_NOM"]))
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
        _set_move(monster, _cycle_move(monster, ["AWAKENED_SLASH", "AWAKENED_SOUL_STRIKE"]))
        return
    if monster.monster_id == "TimeEater":
        if monster.current_hp <= monster.max_hp // 2 and not monster.ai_state.get("healed", 0):
            monster.ai_state["healed"] = 1
            _set_move(monster, "TIME_EATER_RIPPLE")
            return
        _set_move(monster, _cycle_move(monster, ["TIME_EATER_REVERBERATE", "TIME_EATER_HEAD_SLAM"]))
        return
    if monster.monster_id == "Donu":
        _set_move(monster, _cycle_move(monster, ["DONU_CIRCLE_OF_POWER", "DONU_BEAM"]))
        return
    if monster.monster_id == "Deca":
        _set_move(monster, _cycle_move(monster, ["DECA_BEAM", "DECA_SQUARE_OF_PROTECTION"]))
        return
    if monster.monster_id == "SpireShield":
        _set_move(monster, _cycle_move(monster, ["SHIELD_BASH", "SHIELD_FORTIFY", "SHIELD_SMASH"]))
        return
    if monster.monster_id == "SpireSpear":
        _set_move(monster, _cycle_move(monster, ["SPEAR_BURN_STRIKE", "SPEAR_SKEWER", "SPEAR_PIERCER"]))
        return
    if monster.monster_id == "CorruptHeart":
        if monster.move == "HEART_DEBILITATE":
            _set_move(monster, "HEART_BLOOD_SHOTS")
            return
        _set_move(monster, _cycle_move(monster, ["HEART_ECHO", "HEART_BLOOD_SHOTS"]))
        return


__all__ = [
    "ACT_BOSSES",
    "ELITE_ENCOUNTERS",
    "STRONG_ENCOUNTERS",
    "STRONG_WEIGHTS",
    "WEAK_ENCOUNTERS",
    "WEAK_WEIGHTS",
    "choose_next_move",
    "encounter_to_monster_ids",
    "encounter_to_monsters",
    "generate_elite_schedule",
    "generate_monster_schedules",
    "generate_strong_monster_schedule",
    "make_monster",
    "monster_adjusted_damage",
    "roll_act1_encounter",
]
