# spirecomm-native simulator

This document mainly describes the older native simulator family. The current
primary backend status for the repo now lives under:

- `/home/yydd/spirecomm/spirecomm/native_sim_v3/README.md`
- `/home/yydd/spirecomm/spirecomm/native_sim_v3/STATUS.md`

This package is a from-scratch simulator owned by `spirecomm`. It intentionally
keeps the observation schema close to the CommunicationMod state format so the
existing combat and run-level models can reason over familiar inputs.

Current implemented slice:

- Ironclad starter deck, full Ironclad card definition list, and a broad
  colorless/event-card subset that Ironclad can obtain from shops, events,
  potions, and relics.
- Execution paths for Ironclad attacks, skills, and powers, with several
  complex cards still implemented as conservative approximations.
- Several card-specific persistent/trigger rules are modeled, including
  Rampage growth and Sentinel exhaust energy.
- Status/curse cards needed by Ironclad: Wound, Burn, Dazed, Slimed, Void,
  Ascender's Bane.
- Act 1 normal monster, elite, and boss subset with simplified moves and
  stateful key elite/boss patterns:
  Cultist, Jaw Worm, medium slimes, Louses, Fungi Beast, Blue Slaver,
  Gremlin Nob, Lagavulin, Sentries, Slime Boss, Hexaghost, Guardian.
- Simplified Act 2 and Act 3 normal, elite, and boss pools, including common
  hallway fights and the major bosses.
- Act 4 route skeleton with ruby/sapphire/emerald key handling, Shield/Spear,
  Corrupt Heart, and A20 second Act 3 boss support.
- Structural minion encounters for Gremlin Leader, The Collector, Bronze
  Automaton, and Reptomancer, plus Red Slaver Entangle/No Attack handling.
- Bronze Orb Stasis now temporarily removes a draw/discard/hand card and
  returns it when the orb dies.
- Slime split chains now cover Slime Boss, large slimes, and medium slimes,
  producing the expected smaller slime children rather than only handling the
  boss split.
- Chosen Hex, Snecko Confusion, slime status attacks, and Hexaghost Burn
  insertion now have concrete combat effects. Confusion randomizes newly drawn
  non-X attack/skill/power costs, and Artifact can block player debuffs such as
  Hex, Confusion, and No Attack.
- Additional common monster status/debuff effects now mutate the simulated deck
  and powers, including Snecko Tail Whip Vulnerable, Shelled Parasite Frail,
  Repulsor Dazed insertion, Nemesis Burn insertion, and the Masked Bandits'
  Bear/Romeo debuffs.
- Looter/Mugger encounters are included in Act 1/2 pools with gold theft,
  escape intent, and
  stolen-gold return when defeated before escaping.
- Opening-hand innate behavior is modeled for important native cards such as
  Ascender's Bane, Writhe, Dramatic Entrance, and Mind Blast.
- Spirecomm-compatible combat state serialization.
- Legal action generation for playable cards, potions, and hard-disabled end
  turn while any card is playable.
- Optional approximate Neow reward phase (`scripts/native/run_native_run.py --neow`).
- Run-level state machine for combat reward, map, campfire, shop, event, and
  chest phases, plus boss relic rewards after Act 1/2 bosses.
- Expanded relic/potion subsystem with rarity-weighted potion/relic rolls, shop
  relics, core inventory serialization, shop/reward acquisition, combat potion
  actions, boss relics, key relics, common curses, colorless potion/card
  generation, and many high-impact relic triggers.
- Runic Dome now hides monster intent, move id, move base damage, and hit count
  from the spirecomm-compatible observation rather than only masking the intent
  label.
- Basic Artifact/Ginger/Turnip debuff blocking, Sadistic Nature debuff damage,
  and several turn/reward relic counters such as Sundial, Centennial Puzzle,
  Ancient Tea Set, Art of War, Pocketwatch, Singing Bowl, Matryoshka, and
  Necronomicon.
- Rupture now distinguishes card/self-damage from enemy attack damage, and
  Necronomicurse returns when exhausted.
- Resource and routing relic approximations for Omamori, Darkstone Periapt,
  Maw Bank, Meal Ticket, Membership Card, The Courier, Smiling Mask, White Beast
  Statue, Peace Pipe, Shovel, Girya, Tiny Chest, Wing Boots, and the Bottled
  relics.
- Shop/special relic approximations for Ceramic Fish, Clockwork Souvenir,
  Medical Kit, Chemical X, Brimstone, Self-Forming Clay, Magic Flower, Orange
  Pellets, Cauldron, Dolly's Mirror, Lee's Waffle, Orrery, Warped Tongs, and
  related Ironclad/shop relics.
- Existing spirecomm models can drive combat, card reward, map, campfire, shop,
  event, boss relic, potion, upgrade target, and purge target choices through
  `/home/yydd/spirecomm/scripts/native/run_native_run.py`.
- Randomness now uses a Python port of `sts_lightspeed`'s `sts::Random`, with
  separate run-level streams for Neow, treasure, event, relic, potion, card,
  merchant, monster, shuffle, misc, math-util, and map generation. This is the
  first layer of official distribution parity; many distribution formulas below
  still remain approximations.
- Card rewards now follow the lightspeed-style rarity roll shape with
  `cardRarityFactor`, elite/normal rarity thresholds, cross-act card RNG counter
  synchronization, reward-card de-duplication, and Act 2/3 upgrade chances.
  Shop cards use a separate shop rarity roll that reads, but does not mutate,
  the reward rarity factor.
- Act map generation now follows the lightspeed-style 7-column path algorithm,
  first-row redundant edge filtering, room-count ratios, parent/sibling room
  constraints, early elite/rest restrictions, and burning elite assignment. It
  is mapped onto the native simulator's existing absolute-floor state machine,
  so the path topology is much closer to official while preserving native boss
  flow.
- Shop generation now follows the lightspeed slot shape more closely: two
  attack cards, two skills, one non-common power, two colorless cards, two
  ordinary relic-tier rolls, one shop relic, three potions, card sale slot,
  A16/Courier/Membership discounts, and scaling card-removal cost.
- Question-mark rooms now use lightspeed-style dynamic outcome chances for
  monster/shop/treasure/event, including Tiny Chest counter handling,
  Juzu Bracelet monster prevention, and chance reset/increase updates after
  each question-room outcome.
- Event generation now keeps per-act event/shrine/special pools and removes
  drawn events from the current act pool instead of sampling a fresh list every
  time.
- Summon-heavy fights keep native target indices for execution while compressing
  visible live monster slots for the combat model, avoiding out-of-range target
  logits when old dead/split monsters remain in the native list.
- Broader event coverage now includes several common Act 2/3 events such as
  Bonfire Spirits, Designer In-Spire, Face Trader, Forgotten Altar, Lab, Match
  and Keep, Moai Head, Sensory Stone, The Woman in Blue, Transmogrifier,
  Upgrade Shrine, Wheel of Change, Pleading Vagrant, Dead Adventurer, and
  Liars Game.
- Event generation is roughly act-scoped, so early runs no longer draw
  late-game-only event pools such as Mind Bloom or Masked Bandits in Act 1.
- Several high-risk events now enter native event combats instead of being
  treated as pure resource buttons: Masked Bandits, Dead Adventurer,
  Mysterious Sphere, Colosseum, and Mind Bloom's fight branch.
- Enemy-observed powers now include several important non-card mechanics:
  Byrd Flight break/stun, Nemesis Intangible turns, Giant Head Slow stacks,
  Spheric Guardian Artifact/Barricade, Snake Plant Malleable, Shelled Parasite
  Plated Armor, and Spiker Thorns growth.
- Monster block now follows the normal turn cadence: ordinary enemy block is
  cleared before the monster turn, while block gained by monster actions or
  Plated Armor remains visible during the next player turn.

Known incomplete areas:

- Card effects that require grid/card-select sub-screens are approximated.
- Some relic trigger timing is still approximate, especially effects with card
  selection sub-screens, exact restock behavior, or delayed queue interactions.
- Event, shop, chest, Neow, reward, question-room, and map generation now
  consume dedicated StS-style RNG streams. The largest remaining distribution
  gaps are the exact official event eligibility predicates, all one-time-event
  removal rules, shop restock edge cases, and exact map/path compatibility at
  the byte-for-byte seed level.
- Act 4 content and A20 double boss are implemented structurally and can be
  reached in native runs, but Heart, Shield/Spear, and key reward details remain
  approximate.
- Many monsters now have phase/cycle state and structural minions, but the full
  official probability tables, summon placement, and no-repeat move rules are
  not completely reproduced yet.
- Native smoke runs get harsher as monster rules are made more realistic; low
  average floors here are expected while combat still uses the naked policy
  rather than the live `UndoPlanner` branch evaluator.
- Exact StS action-queue ordering is not fully reproduced.

Expansion order:

1. Replace approximated Ironclad card effects with exact rules.
2. Continue expanding relic and potion pools, prioritizing effects that change
   legal actions or model-observed powers.
3. Continue tightening the lightspeed distribution layer: exact event
   eligibility/removal rules, shop restock edge cases, and byte-for-byte map
   validation against lightspeed for representative seeds.
4. Continue replacing approximate monster AI with exact move probabilities,
   no-repeat rules, and phase transitions.
5. Add true map/event/shop reward distributions and Neow downside options.

The goal is not to imitate `sts_lightspeed`'s agent. The simulator should be a
deterministic rules engine that feeds the existing `spirecomm` models directly.
