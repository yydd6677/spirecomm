from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from spirecomm.native_sim.randoms import StsRandom


MAP_HEIGHT = 15
MAP_WIDTH = 7
PATH_DENSITY = 6
ROW_END_NODE = MAP_WIDTH - 1

SHOP_ROOM_CHANCE = 0.05
REST_ROOM_CHANCE = 0.12
TREASURE_ROOM_CHANCE = 0.0
EVENT_ROOM_CHANCE = 0.22
ELITE_ROOM_CHANCE_A0 = 0.08
ELITE_ROOM_CHANCE_A1 = ELITE_ROOM_CHANCE_A0 * 1.6

ROOM_VALUES = {
    "$": 0,
    "R": 1,
    "?": 2,
    "E": 3,
    "M": 4,
    "T": 5,
    "BOSS": 6,
}


@dataclass
class NativeMapNode:
    x: int
    y: int
    symbol: str = ""
    parents: list[int] = field(default_factory=list)
    edges: list[int] = field(default_factory=list)

    def add_parent(self, parent: int) -> None:
        self.parents.append(parent)

    def add_edge(self, edge: int) -> None:
        if edge in self.edges:
            return
        self.edges.append(edge)
        self.edges.sort()

    def remove_edge_at(self, index: int) -> None:
        del self.edges[index]

    def remove_parent(self, parent: int) -> None:
        self.parents = [item for item in self.parents if item != parent]

    def max_edge(self) -> int:
        return self.edges[-1]

    def min_edge(self) -> int:
        return self.edges[0]

    def max_parent(self) -> int:
        return max(self.parents)

    def min_parent(self) -> int:
        return min(self.parents)


def _rand_range(rng: StsRandom, minimum: int, maximum: int) -> int:
    return int(rng.random(maximum - minimum)) + minimum


def _sts_round(value: float) -> int:
    # Match C++ std::round used by lightspeed / the game: half values round
    # away from zero, unlike Python's bankers rounding.
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _get_common_ancestor(nodes: list[list[NativeMapNode]], x1: int, x2: int, y: int) -> int:
    if y < 0:
        return -1

    # Mirrors sts_lightspeed's current implementation, including the x1 < y
    # comparison. The original game has the same odd edge-case behavior through
    # its path-loop avoidance.
    if x1 < y:
        left_node = x1
        right_node = x2
    else:
        left_node = x2
        right_node = x1

    left = nodes[y][left_node]
    right = nodes[y][right_node]
    if not left.parents or not right.parents:
        return -1
    left_x = left.max_parent()
    if left_x == right.min_parent():
        return left_x
    return -1


def _choose_path_parent_loop_randomizer(
    nodes: list[list[NativeMapNode]],
    rng: StsRandom,
    cur_x: int,
    cur_y: int,
    new_x: int,
) -> int:
    dest = nodes[cur_y + 1][new_x]
    for parent_x in dest.parents:
        if cur_x == parent_x:
            continue
        if _get_common_ancestor(nodes, parent_x, cur_x, cur_y) == -1:
            continue
        if new_x > cur_x:
            new_x = cur_x + _rand_range(rng, -1, 0)
            if new_x < 0:
                new_x = cur_x
        elif new_x == cur_x:
            new_x = cur_x + _rand_range(rng, -1, 1)
            if new_x > ROW_END_NODE:
                new_x = cur_x - 1
            elif new_x < 0:
                new_x = cur_x + 1
        else:
            new_x = cur_x + _rand_range(rng, 0, 1)
            if new_x > ROW_END_NODE:
                new_x = cur_x
    return new_x


def _choose_path_adjust_new_x(nodes: list[list[NativeMapNode]], cur_x: int, cur_y: int, new_x: int) -> int:
    if cur_x != 0:
        left_node = nodes[cur_y][cur_x - 1]
        if left_node.edges:
            left_edge = left_node.max_edge()
            if left_edge > new_x:
                new_x = left_edge
    if cur_x < ROW_END_NODE:
        right_node = nodes[cur_y][cur_x + 1]
        if right_node.edges:
            right_edge = right_node.min_edge()
            if right_edge < new_x:
                new_x = right_edge
    return new_x


def _choose_new_path(nodes: list[list[NativeMapNode]], rng: StsRandom, cur_x: int, cur_y: int) -> int:
    if cur_x == 0:
        minimum, maximum = 0, 1
    elif cur_x == ROW_END_NODE:
        minimum, maximum = -1, 0
    else:
        minimum, maximum = -1, 1
    new_x = cur_x + _rand_range(rng, minimum, maximum)
    new_x = _choose_path_parent_loop_randomizer(nodes, rng, cur_x, cur_y, new_x)
    return _choose_path_adjust_new_x(nodes, cur_x, cur_y, new_x)


def _create_paths_iteration(nodes: list[list[NativeMapNode]], rng: StsRandom, start_x: int) -> None:
    cur_x = start_x
    for cur_y in range(MAP_HEIGHT - 1):
        new_x = _choose_new_path(nodes, rng, cur_x, cur_y)
        nodes[cur_y][cur_x].add_edge(new_x)
        nodes[cur_y + 1][new_x].add_parent(cur_x)
        cur_x = new_x
    nodes[14][cur_x].add_edge(3)


def _create_paths(nodes: list[list[NativeMapNode]], rng: StsRandom) -> None:
    first_start_x = _rand_range(rng, 0, MAP_WIDTH - 1)
    _create_paths_iteration(nodes, rng, first_start_x)
    for path_index in range(1, PATH_DENSITY):
        start_x = _rand_range(rng, 0, MAP_WIDTH - 1)
        while start_x == first_start_x and path_index == 1:
            start_x = _rand_range(rng, 0, MAP_WIDTH - 1)
        _create_paths_iteration(nodes, rng, start_x)


def _filter_redundant_edges_from_first_row(nodes: list[list[NativeMapNode]]) -> None:
    visited = [False] * MAP_WIDTH
    for src_x in range(MAP_WIDTH):
        node = nodes[0][src_x]
        for index in range(len(node.edges) - 1, -1, -1):
            dest_x = node.edges[index]
            if visited[dest_x]:
                nodes[1][dest_x].remove_parent(src_x)
                node.remove_edge_at(index)
            else:
                visited[dest_x] = True


@dataclass
class _RoomCounts:
    total: int = 0
    unassigned: int = 0


def _get_room_counts_and_assign_fixed(nodes: list[list[NativeMapNode]]) -> _RoomCounts:
    counts = _RoomCounts()
    for row in range(MAP_HEIGHT):
        for node in nodes[row]:
            if not node.edges:
                continue
            if row == 0:
                node.symbol = "M"
                counts.total += 1
            elif row == 8:
                node.symbol = "T"
                counts.total += 1
            elif row == MAP_HEIGHT - 1:
                node.symbol = "R"
                counts.total += 1
            elif row == MAP_HEIGHT - 2:
                counts.unassigned += 1
            else:
                counts.unassigned += 1
                counts.total += 1
    return counts


def _fill_room_array(counts: _RoomCounts, elite_room_chance: float) -> list[str]:
    rooms: list[str] = []
    rooms.extend(["$"] * _sts_round(counts.total * SHOP_ROOM_CHANCE))
    rooms.extend(["R"] * _sts_round(counts.total * REST_ROOM_CHANCE))
    rooms.extend(["T"] * _sts_round(counts.total * TREASURE_ROOM_CHANCE))
    rooms.extend(["E"] * _sts_round(counts.total * elite_room_chance))
    rooms.extend(["?"] * _sts_round(counts.total * EVENT_ROOM_CHANCE))
    rooms.extend(["M"] * max(0, counts.unassigned - len(rooms)))
    return rooms


class _RoomConstructorData:
    masks = [
        0x0101010101010101,
        0x0202020202020202,
        0x0404040404040404,
        0x0808080808080808,
        0x1010101010101010,
        0x2020202020202020,
        0x4040404040404040,
    ]

    def __init__(self, rooms: list[str]):
        self.rooms = rooms
        self.offset = 0
        self.row_data = 0
        self.prev_row_data = 0
        self.sibling_masks = [0] * MAP_WIDTH
        self.next_sibling_masks = [0] * MAP_WIDTH
        self.parent_masks = [0] * MAP_WIDTH
        self.next_parent_masks = [0] * MAP_WIDTH

    def _room_bit(self, room: str, x: int) -> int:
        return 1 << (ROOM_VALUES[room] + x * 8)

    def set_cur_data_only(self, node: NativeMapNode) -> None:
        self.row_data |= self._room_bit(node.symbol, node.x)

    def set_next_data_only(self, node: NativeMapNode) -> None:
        if len(node.edges) == 1:
            for edge in node.edges:
                self.next_parent_masks[edge] |= 0xFF << (node.x * 8)
        else:
            sibling_mask = 0
            for edge in node.edges:
                sibling_mask |= 0xFF << (edge * 8)
                self.next_sibling_masks[edge] |= sibling_mask
                self.next_parent_masks[edge] |= 0xFF << (node.x * 8)

    def set_data(self, node: NativeMapNode) -> None:
        self.set_cur_data_only(node)
        self.set_next_data_only(node)

    def remove_element(self, index: int) -> None:
        # Mirror lightspeed's RoomConstructorData::removeElement exactly:
        # advance the active-window offset and shift the prefix right without
        # re-inserting the chosen room into the live tail.
        for current in range(index, self.offset, -1):
            self.rooms[current] = self.rooms[current - 1]
        self.offset += 1

    def next_row(self) -> None:
        self.prev_row_data = self.row_data
        self.row_data = 0
        self.sibling_masks = self.next_sibling_masks
        self.next_sibling_masks = [0] * MAP_WIDTH
        self.parent_masks = self.next_parent_masks
        self.next_parent_masks = [0] * MAP_WIDTH

    def sibling_match(self, node_x: int, room: str) -> bool:
        return bool(self.row_data & self.sibling_masks[node_x] & self.masks[ROOM_VALUES[room]])

    def parent_match(self, node_x: int, room: str) -> bool:
        return bool(self.prev_row_data & self.parent_masks[node_x] & self.masks[ROOM_VALUES[room]])


def _assign_room_to_node(node: NativeMapNode, data: _RoomConstructorData) -> None:
    tried: set[str] = set()
    for index in range(data.offset, len(data.rooms)):
        room = data.rooms[index]
        if room in tried:
            continue
        tried.add(room)
        if room == "E" and node.y <= 4:
            continue
        if room == "R" and (node.y <= 4 or node.y >= 13):
            continue
        if room in {"?", "M"}:
            if data.sibling_match(node.x, room):
                continue
            node.symbol = room
            data.row_data |= data._room_bit(room, node.x)
            data.remove_element(index)
            return
        if not data.parent_match(node.x, room) and not data.sibling_match(node.x, room):
            node.symbol = room
            data.row_data |= data._room_bit(room, node.x)
            data.remove_element(index)
            return
    node.symbol = "M"


def _assign_rooms_row(nodes: list[list[NativeMapNode]], data: _RoomConstructorData, row: int) -> None:
    for node in nodes[row]:
        if not node.edges:
            continue
        if row in {0, 8}:
            data.set_next_data_only(node)
        elif row in {7, 13}:
            _assign_room_to_node(node, data)
            data.set_cur_data_only(node)
        else:
            _assign_room_to_node(node, data)
            data.set_data(node)
    data.next_row()


def _assign_rooms(nodes: list[list[NativeMapNode]], rng: StsRandom, ascension_level: int) -> None:
    counts = _get_room_counts_and_assign_fixed(nodes)
    rooms = _fill_room_array(counts, ELITE_ROOM_CHANCE_A1 if ascension_level > 0 else ELITE_ROOM_CHANCE_A0)
    for i in range(counts.unassigned, 1, -1):
        j = rng.next_int(i)
        rooms[i - 1], rooms[j] = rooms[j], rooms[i - 1]
    data = _RoomConstructorData(rooms)
    for row in range(MAP_HEIGHT - 1):
        _assign_rooms_row(nodes, data, row)


def _assign_burning_elite(nodes: list[list[NativeMapNode]], rng: StsRandom) -> tuple[int, int, int] | None:
    elites = [node for row in nodes for node in row if node.symbol == "E"]
    if not elites:
        return None
    burning = elites[int(rng.random(len(elites) - 1))]
    burning.symbol = "E_GREEN"
    return burning.x, burning.y, int(rng.random(0, 3))


def generate_act_map(
    *,
    seed: int,
    ascension_level: int,
    act: int,
    start_floor: int,
    set_burning: bool,
) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
    offset = 1 if act == 1 else act * (100 * (act - 1))
    rng = StsRandom(seed + offset)
    nodes = [[NativeMapNode(x=x, y=y) for x in range(MAP_WIDTH)] for y in range(MAP_HEIGHT)]
    _create_paths(nodes, rng)
    _filter_redundant_edges_from_first_row(nodes)
    _assign_rooms(nodes, rng, ascension_level)
    burning = _assign_burning_elite(nodes, rng) if set_burning else None

    graph: dict[str, dict[str, Any]] = {}
    layers: dict[int, list[str]] = {}
    for row in range(MAP_HEIGHT):
        floor = start_floor + row
        layers[floor] = []
        for node in nodes[row]:
            node_id = f"a{act}-r{row}-x{node.x}"
            graph[node_id] = {
                "id": node_id,
                "act": act,
                "floor": floor,
                "x": node.x,
                "row": row,
                "symbol": node.symbol,
                "children_x": list(node.edges),
                "parents_x": list(node.parents),
                "children": [],
            }
            if node.edges:
                layers[floor].append(node_id)

    for row in range(MAP_HEIGHT - 1):
        next_floor = start_floor + row + 1
        next_by_x = {graph[node_id]["x"]: node_id for node_id in layers.get(next_floor, [])}
        for node_id in layers.get(start_floor + row, []):
            graph[node_id]["children"] = [
                next_by_x[child_x]
                for child_x in graph[node_id].get("children_x", [])
                if child_x in next_by_x
            ]

    boss_floor = start_floor + MAP_HEIGHT
    boss_id = f"a{act}-boss"
    graph[boss_id] = {
        "id": boss_id,
        "act": act,
        "floor": boss_floor,
        "x": 3,
        "row": MAP_HEIGHT,
        "symbol": "BOSS",
        "children": [],
        "children_x": [],
        "parents_x": [],
    }
    layers[boss_floor] = [boss_id]
    for node_id in layers.get(start_floor + MAP_HEIGHT - 1, []):
        if graph[node_id].get("children_x"):
            graph[node_id]["children"] = [boss_id]
            graph[boss_id]["parents_x"].append(int(graph[node_id]["x"]))
    if burning is not None:
        bx, by, buff = burning
        graph[f"a{act}-r{by}-x{bx}"]["burning_elite_buff"] = buff
    return graph, layers
