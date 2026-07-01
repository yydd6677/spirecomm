from __future__ import annotations

import unittest

from spirecomm.native_sim.randoms import NativeRandomStreams, StsRandom


class TestNativeRandoms(unittest.TestCase):
    def test_sts_random_matches_lightspeed_reference_sequence(self):
        rng = StsRandom(123456789)

        self.assertEqual(rng.random(99), 47)
        self.assertEqual(rng.random(0, 10), 4)
        self.assertAlmostEqual(rng.random(), 0.265211820602417, places=12)
        self.assertFalse(rng.random_boolean(0.35))
        self.assertEqual(rng.random_long(), 979175008931098231)
        self.assertEqual(rng.counter, 5)

    def test_sts_random_target_counter_matches_lightspeed_constructor(self):
        rng = StsRandom(1, 3)

        self.assertEqual(rng.random(999), 32)
        self.assertEqual(rng.counter, 4)

    def test_native_random_streams_are_independent(self):
        streams = NativeRandomStreams(42)

        first_card_roll = streams.card.random(99)
        first_relic_roll = streams.relic.random(99)
        self.assertEqual(first_card_roll, first_relic_roll)

        streams.card.random(99)
        self.assertNotEqual(streams.card.counter, streams.relic.counter)

    def test_sts_random_copy_does_not_advance_original(self):
        rng = StsRandom(7)
        clone = rng.copy()

        self.assertEqual(clone.random(99), rng.random(99))
        clone.random(99)
        self.assertNotEqual(clone.counter, rng.counter)


if __name__ == "__main__":
    unittest.main()
