from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PlayerState:
    current_hp: int
    max_hp: int
    block: int = 0
    energy: int = 3
    base_energy: int = 3
    draw_per_turn: int = 5
    powers: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class MonsterState:
    monster_id: str
    current_hp: int
    max_hp: int
    name: str | None = None
    block: int = 0
    intent: str = "UNKNOWN"
    powers: list[dict[str, object]] = field(default_factory=list)
    move_adjusted_damage: int = 0
    move_hits: int = 0
    next_move: str = "UNKNOWN"
    move_history: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class CombatState:
    player: PlayerState
    monsters: list[MonsterState] = field(default_factory=list)
    hand: list[dict[str, object]] = field(default_factory=list)
    draw_pile: list[dict[str, object]] = field(default_factory=list)
    discard_pile: list[dict[str, object]] = field(default_factory=list)
    exhaust_pile: list[dict[str, object]] = field(default_factory=list)
    turn: int = 0
    cards_discarded_this_turn: int = 0
    encounter_name: str = ""
    room_type: str = "MonsterRoom"


@dataclass(slots=True)
class RunState:
    seed: int
    ascension_level: int
    act: int
    dungeon_id: str | None
    floor: int
    phase: str
    player: PlayerState
    deck: list[dict[str, object]] = field(default_factory=list)
    relics: list[dict[str, object]] = field(default_factory=list)
    potions: list[dict[str, object]] = field(default_factory=list)
    gold: int = 99
    has_ruby_key: bool = False
    has_emerald_key: bool = False
    has_sapphire_key: bool = False
    common_card_pool: list[str] = field(default_factory=list)
    uncommon_card_pool: list[str] = field(default_factory=list)
    rare_card_pool: list[str] = field(default_factory=list)
    colorless_card_pool: list[str] = field(default_factory=list)
    curse_card_pool: list[str] = field(default_factory=list)
    common_relic_pool: list[str] = field(default_factory=list)
    uncommon_relic_pool: list[str] = field(default_factory=list)
    rare_relic_pool: list[str] = field(default_factory=list)
    shop_relic_pool: list[str] = field(default_factory=list)
    src_common_card_pool: list[str] = field(default_factory=list)
    src_uncommon_card_pool: list[str] = field(default_factory=list)
    src_rare_card_pool: list[str] = field(default_factory=list)
    src_colorless_card_pool: list[str] = field(default_factory=list)
    src_curse_card_pool: list[str] = field(default_factory=list)
    act_boss: str | None = None
    boss_relic_pool: list[str] = field(default_factory=list)
    event_id: str | None = None
    implementation_status: str = "skeleton"
    combat: CombatState | None = None
