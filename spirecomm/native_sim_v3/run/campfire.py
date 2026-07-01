from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spirecomm.native_sim_v3.content.campfire_rules import rest_amount
from spirecomm.native_sim_v3.content.cards import can_upgrade_card


@dataclass(slots=True)
class CampfireState:
    can_recall: bool = False

    def actions(self, *, deck: list[dict[str, Any]], relics: list[dict[str, Any]] | None = None) -> list[dict[str, object]]:
        relic_ids = {str(relic.get("relic_id") or relic.get("id")) for relic in (relics or [])}
        actions: list[dict[str, object]] = []
        if "Coffee Dripper" not in relic_ids:
            actions.append({"kind": "campfire", "name": "rest", "label": "rest", "choice_index": len(actions)})
        if "Fusion Hammer" not in relic_ids and any(_is_upgradable(card) for card in deck):
            actions.append({"kind": "campfire", "name": "smith", "label": "smith", "choice_index": len(actions)})
        if "Peace Pipe" in relic_ids and any(_is_purgeable(card) for card in deck):
            actions.append({"kind": "campfire", "name": "toke", "label": "toke", "choice_index": len(actions)})
        if "Shovel" in relic_ids:
            actions.append({"kind": "campfire", "name": "dig", "label": "dig", "choice_index": len(actions)})
        if "Girya" in relic_ids:
            girya = next(
                (relic for relic in (relics or []) if str(relic.get("relic_id") or relic.get("id")) == "Girya"),
                None,
            )
            if int((girya or {}).get("counter") or 0) < 3:
                actions.append({"kind": "campfire", "name": "lift", "label": "lift", "choice_index": len(actions)})
        if self.can_recall:
            actions.append({"kind": "campfire", "name": "recall", "label": "recall", "choice_index": len(actions)})
        return actions

def _is_upgradable(card: dict[str, Any]) -> bool:
    return can_upgrade_card(card)


def _is_purgeable(card: dict[str, Any]) -> bool:
    return str(card.get("type") or "") not in {"STATUS"} and not bool(card.get("bottled"))
