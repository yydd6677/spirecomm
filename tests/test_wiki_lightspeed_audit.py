from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from spirecomm.wiki_lightspeed_audit import (
    WikiPage,
    _compare_dead_adventurer_page,
    _compare_elites_page,
    _compare_merchant_page,
    _costs_semantically_match,
    _extract_infobox_values,
    _parse_hp_ranges,
    _parse_lightspeed_cost_expression,
    _page_matches_scope,
    audit_page,
    classify_title,
    load_lightspeed_reference,
    run_audit,
)


REPO_ROOT = Path("/home/yydd/spirecomm")


class WikiLightspeedAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = load_lightspeed_reference(REPO_ROOT)

    def test_reference_contains_known_titles_and_current_burning_formula(self):
        self.assertEqual(self.reference.cards["accuracy"]["cost"], 1)
        self.assertEqual(self.reference.cards["accuracy"]["type"], "POWER")
        self.assertEqual(self.reference.relics["omamori"]["tier"], "COMMON")
        self.assertEqual(self.reference.potions["regenpotion"]["rarity"], "UNCOMMON")
        self.assertEqual(self.reference.systems["elites"]["strength_formula"], "act")

    def test_classify_title_maps_known_pages(self):
        self.assertEqual(classify_title("Accuracy", self.reference), "cards")
        self.assertEqual(classify_title("Omamori", self.reference), "relics")
        self.assertEqual(classify_title("Dead Adventurer", self.reference), "events")
        self.assertEqual(classify_title("Gremlin Nob", self.reference), "monsters")
        self.assertEqual(classify_title("Merchant", self.reference), "systems")

    def test_extract_infobox_values_reads_rendered_html(self):
        html = """
        <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="type">
          <h3 class="pi-data-label pi-secondary-font">Type</h3>
          <div class="pi-data-value pi-font"><a href="/wiki/Power">Power</a></div>
        </div>
        <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="cost">
          <h3 class="pi-data-label pi-secondary-font">Cost</h3>
          <div class="pi-data-value pi-font">1 (0)</div>
        </div>
        """
        infobox = _extract_infobox_values(html)
        self.assertEqual(infobox["type"], "Power")
        self.assertEqual(infobox["cost"], "1 (0)")

    def test_compare_card_page_uses_explicit_cost_plus_field(self):
        page = WikiPage(
            title="Apotheosis",
            categories=["Neutral_Cards", "Skill_Cards", "Rare_Cards"],
            wikitext="",
            html="""
            <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="type">
              <h3 class="pi-data-label pi-secondary-font">Type</h3>
              <div class="pi-data-value pi-font">Skill</div>
            </div>
            <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="rarity">
              <h3 class="pi-data-label pi-secondary-font">Rarity</h3>
              <div class="pi-data-value pi-font">Rare</div>
            </div>
            <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="cost">
              <h3 class="pi-data-label pi-secondary-font">Cost</h3>
              <div class="pi-data-value pi-font">2</div>
            </div>
            <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="cost_plus">
              <h3 class="pi-data-label pi-secondary-font">Cost+</h3>
              <div class="pi-data-value pi-font">1</div>
            </div>
            <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="effect">
              <h3 class="pi-data-label pi-secondary-font">Effect</h3>
              <div class="pi-data-value pi-font">Upgrade ALL your cards. Exhaust.</div>
            </div>
            <div class="pi-item pi-data pi-item-spacing pi-border-color" data-source="effect_plus">
              <h3 class="pi-data-label pi-secondary-font">Effect+</h3>
              <div class="pi-data-value pi-font">Upgrade ALL your cards. Exhaust.</div>
            </div>
            """,
        )
        claims, _ = audit_page(page, self.reference, REPO_ROOT)
        cost_claim = next(claim for claim in claims if claim.claim_kind == "cost")
        self.assertEqual(cost_claim.wiki_claim_text, "cost=2, cost_plus=1")

    def test_parse_lightspeed_cost_expression_handles_ternary_upgrade_costs(self):
        self.assertEqual(_parse_lightspeed_cost_expression("upgraded ? 3 : 4"), (4, 3))
        self.assertEqual(_parse_lightspeed_cost_expression("upgraded ? 0 : 1"), (1, 0))
        self.assertEqual(_parse_lightspeed_cost_expression("2"), (2, 2))

    def test_negative_lightspeed_cost_matches_wiki_unplayable_cost(self):
        self.assertTrue(_costs_semantically_match(None, None, -3, -3))
        self.assertTrue(_costs_semantically_match(None, None, -2, -2))
        self.assertFalse(_costs_semantically_match(None, None, 1, 1))

    def test_parse_hp_ranges_handles_single_value_boss_and_ascension_forms(self):
        self.assertEqual(_parse_hp_ranges("250 264 9+"), ([250, 250], [264, 264]))
        self.assertEqual(_parse_hp_ranges("300 320 (Ascension 9+)"), ([300, 300], [320, 320]))
        self.assertEqual(
            _parse_hp_ranges("300 (Both Phases 1 and 2) 320 (Ascension 9+, both Phases 1 and 2)"),
            ([300, 300], [320, 320]),
        )

    def test_elites_page_flags_burning_strength_formula_difference(self):
        page = WikiPage(
            title="Elites",
            categories=["Game_Mechanics", "Elite"],
            wikitext="""
            == Elite Buffs ==
            * (Act number + 1) increase in Strength.
            * +25% max HP.
            * (Act number*2 + 2) Metallicize.
            * (Act number*2 + 1) Regenerate.
            """,
            html="",
        )
        claims, counters = _compare_elites_page(page, self.reference, REPO_ROOT)
        self.assertEqual(counters["claims_checked"], 4)
        mismatches = [claim for claim in claims if claim.comparison_result == "mismatch"]
        self.assertTrue(any(claim.claim_kind == "burning_strength" for claim in mismatches))
        strength_claim = next(claim for claim in claims if claim.claim_kind == "burning_strength")
        self.assertEqual(strength_claim.v2_exposure, True)

    def test_merchant_page_flags_asc16_difference(self):
        page = WikiPage(
            title="Merchant",
            categories=["NPC", "Map_Location"],
            wikitext="""
            On [[Ascension]] 16+, shop prices are increased by 10%.

            === 5 Colored Cards (Class-Specific) ===
            * [[Common Cards|Common]]: 45 - 55 [[Gold]]
            * [[Uncommon Cards|Uncommon]]: 68 - 82 [[Gold]]
            * [[Rare Cards|Rare]]: 135 - 165 [[Gold]]

            === 2 [[Colorless Cards]] ===
            * [[Uncommon Cards|Uncommon]]: 81 - 99 [[Gold]]
            * [[Rare Cards|Rare]]: 162 - 198 [[Gold]]

            === 3 [[Relics]] ===
            * [[Relics#Common|Common]]: 143 - 157 [[Gold]]
            * [[Relics#Uncommon|Uncommon]]: 238 - 262 [[Gold]]
            * [[Relics#Rare|Rare]]: 285 - 315 [[Gold]]
            * [[Relics#Shop|Shop]]: 143 - 157 [[Gold]]

            === 3 [[Potions]] ===
            * Common: 48 - 52 [[Gold]]
            * Uncommon: 72 - 78 [[Gold]]
            * Rare: 95 - 105 [[Gold]]
            """,
            html="",
        )
        claims, counters = _compare_merchant_page(page, REPO_ROOT)
        self.assertGreaterEqual(counters["claims_checked"], 10)
        self.assertTrue(any(claim.claim_kind == "shop_asc16_modifier" for claim in claims))

    def test_dead_adventurer_page_is_auto_validated(self):
        page = WikiPage(
            title="Dead Adventurer",
            categories=["Event"],
            wikitext="""
            This event only appears on floor 7 and above.
            *'''[Search]''' Find Loot. 25% (35%) that an Elite will return to fight you.
            * Each subsequent [Search] increases the chance to encounter an Elite by 25%.
            """,
            html="",
        )
        claims, counters = _compare_dead_adventurer_page(page, REPO_ROOT)
        self.assertEqual(counters["claims_checked"], 3)
        self.assertFalse(any(claim.comparison_result == "mismatch" for claim in claims))

    def test_unknown_system_page_is_marked_unvalidated(self):
        page = WikiPage(
            title="Neow",
            categories=["NPC", "Event"],
            wikitext="Neow welcomes the player at the beginning of every run.",
            html="",
        )
        claims, manifest = audit_page(page, self.reference, REPO_ROOT)
        self.assertEqual(manifest["status"], "unvalidated")
        self.assertEqual(claims[0].comparison_result, "unvalidated")

    def test_ironclad_scope_filters_non_ironclad_cards_but_keeps_shared_content(self):
        self.assertFalse(_page_matches_scope("A Thousand Cuts", "cards", ["Silent_Cards", "Power_Cards"], "ironclad"))
        self.assertTrue(_page_matches_scope("Bash", "cards", ["Ironclad_Cards", "Attack_Cards"], "ironclad"))
        self.assertTrue(_page_matches_scope("Panache", "cards", ["Neutral_Cards", "Power_Cards"], "ironclad"))
        self.assertTrue(_page_matches_scope("Burn", "cards", ["Neutral_Cards", "Status_Cards"], "ironclad"))
        self.assertTrue(_page_matches_scope("Gremlin Nob", "monsters", ["Elite", "Act_I"], "ironclad"))
        self.assertFalse(_page_matches_scope("Defend", "cards", ["Disambiguation_pages"], "ironclad"))

    def test_run_audit_recovers_from_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            inventory = ["Elites"]
            output_dir.joinpath("inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
            page = WikiPage(
                title="Elites",
                categories=["Game_Mechanics", "Elite"],
                wikitext="""
                == Elite Buffs ==
                * (Act number + 1) increase in Strength.
                * +25% max HP.
                * (Act number*2 + 2) Metallicize.
                * (Act number*2 + 1) Regenerate.
                """,
                html="",
            )
            output_dir.joinpath("pages.json").write_text(
                json.dumps(
                    {
                        "Elites": {
                            "title": page.title,
                            "categories": page.categories,
                            "wikitext": page.wikitext,
                            "html": page.html,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_dir.joinpath("checkpoint.json").write_text(
                json.dumps({"processed_titles": ["Elites"], "inventory_size": 1, "last_title": "Elites"}),
                encoding="utf-8",
            )

            summary = run_audit(repo_root=REPO_ROOT, output_dir=output_dir)
            self.assertEqual(summary["validated_pages"], 1)
            self.assertEqual(summary["failed_pages"], 0)
            self.assertTrue(output_dir.joinpath("coverage_manifest.json").exists())
            self.assertTrue(output_dir.joinpath("findings.jsonl").exists())
            self.assertTrue(output_dir.joinpath("summary.md").exists())

    def test_run_audit_records_fetch_failures_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            output_dir.joinpath("inventory.json").write_text(json.dumps(["Accuracy"]), encoding="utf-8")
            with mock.patch("spirecomm.wiki_lightspeed_audit.fetch_wiki_page", side_effect=RuntimeError("boom")):
                summary = run_audit(repo_root=REPO_ROOT, output_dir=output_dir)
            self.assertEqual(summary["failed_pages"], 1)
            coverage = json.loads(output_dir.joinpath("coverage_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(coverage[0]["status"], "fetch_failed")
            findings = output_dir.joinpath("findings.jsonl").read_text(encoding="utf-8")
            self.assertIn("fetch_failed", findings)


if __name__ == "__main__":
    unittest.main()
