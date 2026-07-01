from __future__ import annotations

import unittest

from spirecomm.ai import runtime_decision
from spirecomm.native_sim_v3 import NativeRunEnv
from spirecomm.native_sim_v3.content.relics import make_relic
from spirecomm.native_sim_v3.run.map import MapEdge, MapNode


def _map_env_with_row1_options(symbols: dict[int, str], *, current_x: int = 1) -> NativeRunEnv:
    env = NativeRunEnv(seed=71, character="IRONCLAD", ascension_level=0, enable_neow=False, start_on_map=True)
    nodes = [[MapNode(x=x, y=y) for x in range(5)] for y in range(3)]
    nodes[0][current_x].room_symbol = "M"
    nodes[0][current_x].add_edge(MapEdge(src_x=current_x, src_y=0, dst_x=current_x, dst_y=1))
    for x, symbol in symbols.items():
        nodes[1][x].room_symbol = symbol
        nodes[1][x].add_edge(MapEdge(src_x=x, src_y=1, dst_x=x, dst_y=2))
        nodes[2][x].room_symbol = "M"
    env.map = nodes
    env.phase = "MAP"
    env.current_map_node = (current_x, 0)
    env.first_room_chosen = True
    env.floor = 1
    return env


class RuntimeMapDpTest(unittest.TestCase):
    def test_green_elite_penalty_default_is_minus_40(self) -> None:
        self.assertEqual(runtime_decision.MAP_DP_GREEN_ELITE_PENALTY, 40)
        self.assertEqual(runtime_decision.MAP_DP_WINGED_OFFPATH_PENALTY, 20)

    def test_winged_greaves_dp_can_choose_off_path_best_node(self) -> None:
        env = _map_env_with_row1_options({1: "M", 3: "T"})
        env.relics.append(make_relic("WingedGreaves"))

        action, scores, source = runtime_decision._choose_map_dynamic_programming(env)

        self.assertEqual(source, "map_dp")
        self.assertEqual(action["x"], 3)
        self.assertEqual(action["symbol"], "T")
        self.assertGreater(scores[action["choice_index"]], scores[0])

    def test_winged_greaves_dp_penalizes_off_path_and_prefers_normal_edge(self) -> None:
        env = _map_env_with_row1_options({1: "T", 3: "T"})
        env.relics.append(make_relic("WingedGreaves"))

        action, scores, source = runtime_decision._choose_map_dynamic_programming(env)

        self.assertEqual(source, "map_dp")
        self.assertEqual(scores[0] - scores[1], runtime_decision.MAP_DP_WINGED_OFFPATH_PENALTY)
        self.assertEqual(action["x"], 1)


if __name__ == "__main__":
    unittest.main()
