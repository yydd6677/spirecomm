from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CARD_TYPES = ("ATTACK", "SKILL", "POWER", "STATUS", "CURSE")
CARD_RARITIES = ("BASIC", "COMMON", "UNCOMMON", "RARE", "SPECIAL", "CURSE")


@dataclass(frozen=True)
class CardDef:
    card_id: str
    name: str
    card_type: str
    rarity: str
    cost: int
    has_target: bool = False
    exhausts: bool = False
    upgraded_cost: int | None = None
    ethereal: bool = False
    x_cost: bool = False


@dataclass(eq=False)
class CardInstance:
    card_def: CardDef
    upgrades: int = 0
    misc: int = 0
    uuid: str = ""
    cost_for_combat: int | None = None
    cost_for_turn: int | None = None
    free_to_play_once: bool = False

    @property
    def card_id(self) -> str:
        return self.card_def.card_id

    @property
    def name(self) -> str:
        if self.upgrades > 0 and not self.card_def.name.endswith("+"):
            return f"{self.card_def.name}+"
        return self.card_def.name

    @property
    def cost(self) -> int:
        if self.cost_for_turn is not None:
            return self.cost_for_turn
        if self.cost_for_combat is not None:
            return self.cost_for_combat
        if self.upgrades > 0 and self.card_def.upgraded_cost is not None:
            return self.card_def.upgraded_cost
        return self.card_def.cost


@dataclass
class Power:
    power_id: str
    amount: int

    def to_spirecomm(self) -> dict[str, Any]:
        return {
            "power_id": self.power_id,
            "id": self.power_id,
            "name": self.power_id,
            "amount": self.amount,
            "card": None,
            "damage": 0,
            "just_applied": False,
            "misc": self.amount,
        }


@dataclass(frozen=True)
class PotionDef:
    potion_id: str
    name: str
    rarity: str = "COMMON"
    requires_target: bool = False


@dataclass
class PotionInstance:
    potion_def: PotionDef | None = None
    price: int = 0

    @property
    def potion_id(self) -> str:
        return self.potion_def.potion_id if self.potion_def else "Potion Slot"

    @property
    def name(self) -> str:
        return self.potion_def.name if self.potion_def else "Potion Slot"

    @property
    def can_use(self) -> bool:
        return self.potion_def is not None

    @property
    def requires_target(self) -> bool:
        return bool(self.potion_def and self.potion_def.requires_target)

    def to_spirecomm(self) -> dict[str, Any]:
        return {
            "potion_id": self.potion_id,
            "id": self.potion_id,
            "name": self.name,
            "can_use": self.can_use,
            "can_discard": self.can_use,
            "requires_target": self.requires_target,
            "price": self.price,
        }


@dataclass
class PlayerState:
    current_hp: int = 80
    max_hp: int = 80
    block: int = 0
    energy: int = 3
    powers: dict[str, int] = field(default_factory=dict)

    def power(self, power_id: str) -> int:
        return int(self.powers.get(power_id, 0))

    def add_power(self, power_id: str, amount: int) -> None:
        self.powers[power_id] = self.power(power_id) + int(amount)
        if self.powers[power_id] == 0:
            del self.powers[power_id]

    def power_list(self) -> list[dict[str, Any]]:
        return [Power(power_id, amount).to_spirecomm() for power_id, amount in self.powers.items() if amount]


@dataclass
class MonsterState:
    monster_id: str
    name: str
    max_hp: int
    current_hp: int
    move: str
    intent: str
    move_base_damage: int = 0
    move_hits: int = 0
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)
    half_dead: bool = False
    is_gone: bool = False
    move_history: list[str] = field(default_factory=list)
    ai_state: dict[str, Any] = field(default_factory=dict)

    def power(self, power_id: str) -> int:
        return int(self.powers.get(power_id, 0))

    def add_power(self, power_id: str, amount: int) -> None:
        self.powers[power_id] = self.power(power_id) + int(amount)
        if self.powers[power_id] == 0:
            del self.powers[power_id]

    def power_list(self) -> list[dict[str, Any]]:
        return [Power(power_id, amount).to_spirecomm() for power_id, amount in self.powers.items() if amount]

    @property
    def alive(self) -> bool:
        return self.current_hp > 0 and not self.is_gone and not self.half_dead
