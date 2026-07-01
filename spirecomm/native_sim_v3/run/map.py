from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spirecomm.native_sim_v3.content.act_chances import act_chances
from spirecomm.native_sim_v3.content.ending_rules import ending_rules
from spirecomm.native_sim_v3.content.map_rules import map_rules

_MAP_RULES = map_rules()
MAP_HEIGHT = _MAP_RULES.map_height
MAP_WIDTH = _MAP_RULES.map_width
MAP_PATH_DENSITY = _MAP_RULES.map_path_density


def _java_round(value: float) -> int:
    return int(value + 0.5)


def _room_class_to_symbol(room_class: str) -> str:
    mapping = {
        "MonsterRoom": "M",
        "RestRoom": "R",
        "TreasureRoom": "T",
        "MonsterRoomElite": "E",
        "ShopRoom": "$",
        "MonsterRoomBoss": "BOSS",
        "TrueVictoryRoom": "VICTORY",
    }
    symbol = mapping.get(str(room_class))
    if symbol is None:
        raise ValueError(f"unsupported map room class {room_class!r}")
    return symbol


@dataclass(slots=True)
class MapEdge:
    src_x: int
    src_y: int
    dst_x: int
    dst_y: int


@dataclass(slots=True, frozen=True)
class RoomToken:
    symbol: str
    token_id: int


@dataclass(slots=True)
class MapNode:
    x: int
    y: int
    room_symbol: str | None = None
    has_emerald_key: bool = False
    edges: list[MapEdge] = field(default_factory=list)
    parents: list["MapNode"] = field(default_factory=list)

    def has_edges(self) -> bool:
        return bool(self.edges)

    def add_edge(self, edge: MapEdge) -> None:
        if all(edge.dst_x != existing.dst_x or edge.dst_y != existing.dst_y for existing in self.edges):
            self.edges.append(edge)

    def del_edge(self, edge: MapEdge) -> None:
        self.edges = [existing for existing in self.edges if not (existing.dst_x == edge.dst_x and existing.dst_y == edge.dst_y)]

    def add_parent(self, parent: "MapNode") -> None:
        self.parents.append(parent)


def _create_nodes(height: int = MAP_HEIGHT, width: int = MAP_WIDTH) -> list[list[MapNode]]:
    return [[MapNode(x=x, y=y) for x in range(width)] for y in range(height)]


def _node(nodes: list[list[MapNode]], x: int, y: int) -> MapNode:
    return nodes[y][x]


def _edge_sort_key(edge: MapEdge) -> tuple[int, int]:
    return (edge.dst_x, edge.dst_y)


def _get_max_edge(edges: list[MapEdge]) -> MapEdge:
    return sorted(edges, key=_edge_sort_key)[-1]


def _get_min_edge(edges: list[MapEdge]) -> MapEdge:
    return sorted(edges, key=_edge_sort_key)[0]


def _get_node_with_max_x(nodes: list[MapNode]) -> MapNode:
    return max(nodes, key=lambda node: node.x)


def _get_node_with_min_x(nodes: list[MapNode]) -> MapNode:
    return min(nodes, key=lambda node: node.x)


def _common_ancestor(node1: MapNode, node2: MapNode, max_depth: int) -> MapNode | None:
    assert node1.y == node2.y
    assert node1 is not node2
    # Mirror the decompiled MapGenerator bytecode literally here. The
    # runtime comparison is against node2.y, not node2.x.
    if node1.x < node2.y:
        left_node = node1
        right_node = node2
    else:
        left_node = node2
        right_node = node1
    current_y = node1.y
    while current_y >= 0 and current_y >= node1.y - max_depth:
        if not left_node.parents or not right_node.parents:
            return None
        left_node = _get_node_with_max_x(left_node.parents)
        right_node = _get_node_with_min_x(right_node.parents)
        if left_node is right_node:
            return left_node
        current_y -= 1
    return None


def _rand_range(rng: Any, start: int, end: int) -> int:
    return int(rng.random(start, end))


def _create_path(nodes: list[list[MapNode]], edge: MapEdge, rng: Any) -> list[list[MapNode]]:
    current_node = _node(nodes, edge.dst_x, edge.dst_y)
    if edge.dst_y + 1 >= len(nodes):
        current_node.add_edge(MapEdge(edge.dst_x, edge.dst_y, 3, edge.dst_y + 2))
        current_node.edges.sort(key=_edge_sort_key)
        return nodes

    row_end_node = len(nodes[edge.dst_y]) - 1
    if edge.dst_x == 0:
        min_delta, max_delta = 0, 1
    elif edge.dst_x == row_end_node:
        min_delta, max_delta = -1, 0
    else:
        min_delta, max_delta = -1, 1

    new_edge_x = edge.dst_x + _rand_range(rng, min_delta, max_delta)
    new_edge_y = edge.dst_y + 1
    target_candidate = _node(nodes, new_edge_x, new_edge_y)
    min_ancestor_gap = 3
    max_ancestor_gap = 5

    if target_candidate.parents:
        for parent in list(target_candidate.parents):
            if parent is current_node:
                continue
            ancestor = _common_ancestor(parent, current_node, max_ancestor_gap)
            if ancestor is None:
                continue
            ancestor_gap = new_edge_y - ancestor.y
            if ancestor_gap < min_ancestor_gap:
                if target_candidate.x > current_node.x:
                    new_edge_x = edge.dst_x + _rand_range(rng, -1, 0)
                    if new_edge_x < 0:
                        new_edge_x = edge.dst_x
                elif target_candidate.x == current_node.x:
                    new_edge_x = edge.dst_x + _rand_range(rng, -1, 1)
                    if new_edge_x > row_end_node:
                        new_edge_x = edge.dst_x - 1
                    elif new_edge_x < 0:
                        new_edge_x = edge.dst_x + 1
                else:
                    new_edge_x = edge.dst_x + _rand_range(rng, 0, 1)
                    if new_edge_x > row_end_node:
                        new_edge_x = edge.dst_x
                target_candidate = _node(nodes, new_edge_x, new_edge_y)

    if edge.dst_x != 0:
        left_node = nodes[edge.dst_y][edge.dst_x - 1]
        if left_node.has_edges():
            right_edge_of_left = _get_max_edge(left_node.edges)
            if right_edge_of_left.dst_x > new_edge_x:
                new_edge_x = right_edge_of_left.dst_x
    if edge.dst_x < row_end_node:
        right_node = nodes[edge.dst_y][edge.dst_x + 1]
        if right_node.has_edges():
            left_edge_of_right = _get_min_edge(right_node.edges)
            if left_edge_of_right.dst_x < new_edge_x:
                new_edge_x = left_edge_of_right.dst_x

    target_candidate = _node(nodes, new_edge_x, new_edge_y)
    new_edge = MapEdge(edge.dst_x, edge.dst_y, new_edge_x, new_edge_y)
    current_node.add_edge(new_edge)
    current_node.edges.sort(key=_edge_sort_key)
    target_candidate.add_parent(current_node)
    return _create_path(nodes, new_edge, rng)


def _filter_redundant_edges_from_first_row(nodes: list[list[MapNode]]) -> list[list[MapNode]]:
    existing_edges: list[MapEdge] = []
    delete_list: list[MapEdge] = []
    for node in nodes[0]:
        if not node.has_edges():
            continue
        for edge in node.edges:
            for prev_edge in existing_edges:
                if edge.dst_x == prev_edge.dst_x and edge.dst_y == prev_edge.dst_y:
                    delete_list.append(edge)
            existing_edges.append(edge)
        for edge in delete_list:
            node.del_edge(edge)
        delete_list.clear()
    return nodes


def _create_paths(nodes: list[list[MapNode]], path_density: int, rng: Any) -> list[list[MapNode]]:
    first_starting_node = -1
    row_size = len(nodes[0]) - 1
    for index in range(path_density):
        starting_node = _rand_range(rng, 0, row_size)
        if index == 0:
            first_starting_node = starting_node
        while starting_node == first_starting_node and index == 1:
            starting_node = _rand_range(rng, 0, row_size)
        _create_path(nodes, MapEdge(starting_node, -1, starting_node, 0), rng)
    return nodes


def _connected_non_assigned_node_count(map_rows: list[list[MapNode]]) -> int:
    count = 0
    for row in map_rows:
        for node in row:
            if node.has_edges() and node.room_symbol is None:
                count += 1
    return count


def _assign_row_as_room_type(row: list[MapNode], symbol: str) -> None:
    for node in row:
        if node.room_symbol is None:
            node.room_symbol = symbol


def _get_siblings(map_rows: list[list[MapNode]], parents: list[MapNode], node: MapNode) -> list[MapNode]:
    siblings: list[MapNode] = []
    for parent in parents:
        for parent_edge in parent.edges:
            sibling = map_rows[parent_edge.dst_y][parent_edge.dst_x]
            if sibling is node:
                continue
            siblings.append(sibling)
    return siblings


def _rule_sibling_matches(siblings: list[MapNode], room_symbol: str) -> bool:
    applicable = {"R", "M", "?", "E", "$"}
    if room_symbol not in applicable:
        return False
    return any(sibling.room_symbol == room_symbol for sibling in siblings if sibling.room_symbol is not None)


def _rule_parent_matches(parents: list[MapNode], room_symbol: str) -> bool:
    applicable = {"R", "T", "$", "E"}
    if room_symbol not in applicable:
        return False
    return any(parent.room_symbol == room_symbol for parent in parents if parent.room_symbol is not None)


def _rule_assignable_to_row(node: MapNode, room_symbol: str) -> bool:
    if node.y <= 4 and room_symbol in {"R", "E"}:
        return False
    return node.y < 13 or room_symbol != "R"


def _next_room_symbol_according_to_rules(
    map_rows: list[list[MapNode]], node: MapNode, room_list: list[RoomToken]
) -> RoomToken | None:
    parents = list(node.parents)
    siblings = _get_siblings(map_rows, parents, node)
    for token in room_list:
        symbol = token.symbol
        if not _rule_assignable_to_row(node, symbol):
            continue
        if not _rule_parent_matches(parents, symbol) and not _rule_sibling_matches(siblings, symbol):
            return token
        if node.y == 0:
            return token
    return None


def _assign_rooms_to_nodes(map_rows: list[list[MapNode]], room_list: list[RoomToken]) -> None:
    for row in map_rows:
        for node in row:
            if node is None or not node.has_edges() or node.room_symbol is not None:
                continue
            token = _next_room_symbol_according_to_rules(map_rows, node, room_list)
            if token is None:
                continue
            node.room_symbol = token.symbol
            room_list.remove(token)


def _last_minute_node_checker(map_rows: list[list[MapNode]]) -> None:
    for row in map_rows:
        for node in row:
            if node is not None and node.has_edges() and node.room_symbol is None:
                node.room_symbol = "M"


def _generate_room_types(available_room_count: int, *, act: int | str = 1, ascension_level: int = 0) -> list[RoomToken]:
    chances = act_chances(act)
    shop_count = _java_round(available_room_count * chances.shop_room_chance)
    rest_count = _java_round(available_room_count * chances.rest_room_chance)
    treasure_count = _java_round(available_room_count * chances.treasure_room_chance)
    if ascension_level >= 1:
        elite_count = _java_round(available_room_count * chances.elite_room_chance * 1.6)
    else:
        elite_count = _java_round(available_room_count * chances.elite_room_chance)
    event_count = _java_round(available_room_count * chances.event_room_chance)
    symbols = (
        ["$"] * shop_count
        + ["R"] * rest_count
        + ["E"] * elite_count
        + ["?"] * event_count
        + ["T"] * treasure_count
    )
    return [RoomToken(symbol=symbol, token_id=index) for index, symbol in enumerate(symbols)]


def _generate_special_ending_map() -> list[list[MapNode]]:
    rules = ending_rules()
    height = max(room.y for room in rules.rooms) + 1
    width = MAP_WIDTH
    nodes = _create_nodes(height=height, width=width)
    nodes_by_name: dict[str, MapNode] = {}
    for room in rules.rooms:
        node = nodes[room.y][room.x]
        node.room_symbol = _room_class_to_symbol(room.room_class)
        nodes_by_name[room.node_name] = node
    for edge in rules.edges:
        src = nodes_by_name[edge.src_name]
        dst = nodes_by_name[edge.dst_name]
        src.add_edge(MapEdge(src.x, src.y, dst.x, dst.y))
        dst.add_parent(src)
    return nodes


def _distribute_rooms_across_map(rng: Any, map_rows: list[list[MapNode]], room_list: list[RoomToken]) -> list[list[MapNode]]:
    node_count = _connected_non_assigned_node_count(map_rows)
    while len(room_list) < node_count:
        room_list.append(RoomToken(symbol="M", token_id=len(room_list)))
    rng.shuffle(room_list)
    _assign_rooms_to_nodes(map_rows, room_list)
    _last_minute_node_checker(map_rows)
    return map_rows


def generate_act_map(
    randoms: Any,
    *,
    act: int | str = 1,
    ascension_level: int = 0,
    endless: bool = False,
    blight_ids: set[str] | None = None,
    final_act_available: bool = False,
    has_emerald_key: bool = False,
) -> list[list[MapNode]]:
    if str(act) in {"4", "TheEnding"} or int(act) == 4:
        return _generate_special_ending_map()
    map_rng = randoms.stream("map")
    nodes = _create_nodes(MAP_HEIGHT, MAP_WIDTH)
    nodes = _create_paths(nodes, MAP_PATH_DENSITY, map_rng)
    nodes = _filter_redundant_edges_from_first_row(nodes)

    count = 0
    for row in nodes:
        for node in row:
            if node.has_edges() and node.y != len(nodes) - 2:
                count += 1

    room_list = _generate_room_types(count, act=act, ascension_level=ascension_level)
    _assign_row_as_room_type(nodes[-1], _room_class_to_symbol(_MAP_RULES.last_row_room_class))
    _assign_row_as_room_type(nodes[0], _room_class_to_symbol(_MAP_RULES.first_row_room_class))
    mimic_infestation = bool(endless) and "MimicInfestation" in {str(blight_id) for blight_id in set(blight_ids or ())}
    special_row_symbol = _room_class_to_symbol(
        _MAP_RULES.endless_mimic_room_class if mimic_infestation else _MAP_RULES.special_row_room_class
    )
    special_row_index = _MAP_RULES.endless_mimic_row_index if mimic_infestation else _MAP_RULES.special_row_index
    _assign_row_as_room_type(nodes[special_row_index], special_row_symbol)
    nodes = _distribute_rooms_across_map(map_rng, nodes, room_list)
    if final_act_available and not has_emerald_key:
        elite_nodes = [node for row in nodes for node in row if node.room_symbol == "E"]
        if elite_nodes:
            chosen = elite_nodes[int(map_rng.random(0, len(elite_nodes) - 1))]
            chosen.has_emerald_key = True
    return nodes


def available_map_actions(
    nodes: list[list[MapNode]],
    *,
    current_node: tuple[int, int] | None,
    first_room_chosen: bool,
    floor_offset: int,
    winged_charges: int = 0,
) -> list[dict[str, Any]]:
    if not first_room_chosen:
        candidates = [node for node in nodes[0] if node.has_edges()]
        synthetic_edges: list[MapEdge] = []
    elif current_node is None:
        candidates = []
        synthetic_edges = []
    else:
        x, y = current_node
        synthetic_edges = [edge for edge in nodes[y][x].edges if edge.dst_y >= MAP_HEIGHT]
        normal_candidates = [nodes[edge.dst_y][edge.dst_x] for edge in nodes[y][x].edges if edge.dst_y < MAP_HEIGHT]
        if winged_charges > 0 and normal_candidates:
            target_y = min(edge.dst_y for edge in nodes[y][x].edges if edge.dst_y < MAP_HEIGHT)
            candidates = [node for node in nodes[target_y] if node.room_symbol is not None and node.has_edges()]
        else:
            candidates = normal_candidates

    actions: list[dict[str, Any]] = []
    act_index = int(floor_offset) // 17 + 1
    for choice_index, node in enumerate(candidates):
        next_symbols = [nodes[edge.dst_y][edge.dst_x].room_symbol for edge in node.edges if edge.dst_y < MAP_HEIGHT]
        symbol = "E_GREEN" if node.room_symbol == "E" and node.has_emerald_key else node.room_symbol
        actions.append(
            {
                "kind": "map",
                "name": symbol,
                "symbol": symbol,
                "floor": node.y + 1,
                "choice_index": choice_index,
                "node_id": f"a{act_index}-r{node.y}-x{node.x}",
                "x": node.x,
                "child_count": len(node.edges),
                "next_symbols": next_symbols,
            }
        )
    for edge in synthetic_edges:
        actions.append(
            {
                "kind": "map",
                "name": "BOSS",
                "symbol": "BOSS",
                "floor": MAP_HEIGHT + 1,
                "choice_index": len(actions),
                "node_id": f"a{act_index}-rboss-x{edge.dst_x}",
                "x": edge.dst_x,
                "child_count": 0,
                "next_symbols": [],
            }
        )
    return actions
