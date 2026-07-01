from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

MASK_64 = (1 << 64) - 1
NORM_FLOAT = 1.0 / (1 << 24)
NORM_DOUBLE = 1.0 / (1 << 53)
JAVA_MASK_48 = (1 << 48) - 1
JAVA_MULTIPLIER = 0x5DEECE66D
JAVA_ADDEND = 0xB


def _mask(value: int) -> int:
    return value & MASK_64


def _unsigned_right_shift(value: int, shift: int) -> int:
    return (_mask(value) >> shift) & MASK_64


def _murmur_hash3(value: int) -> int:
    x = _mask(value)
    x ^= _unsigned_right_shift(x, 33)
    x = _mask(x * 0xFF51AFD7ED558CCD)
    x ^= _unsigned_right_shift(x, 33)
    x = _mask(x * 0xC4CEB9FE1A85EC53)
    x ^= _unsigned_right_shift(x, 33)
    return _mask(x)


@dataclass(slots=True)
class RandomCall:
    stream: str
    method: str
    args: tuple[Any, ...]
    result: Any


class RandomXS128:
    def __init__(self, seed: int) -> None:
        if seed == 0:
            seed = -(1 << 63)
        seed0 = _murmur_hash3(seed)
        seed1 = _murmur_hash3(seed0)
        self.seed0 = seed0
        self.seed1 = seed1

    def next_long(self) -> int:
        s1 = self.seed0
        s0 = self.seed1
        self.seed0 = s0
        s1 ^= _mask(s1 << 23)
        self.seed1 = _mask(s1 ^ s0 ^ _unsigned_right_shift(s1, 17) ^ _unsigned_right_shift(s0, 26))
        return _mask(self.seed1 + s0)

    def next_long_bound(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        while True:
            bits = _unsigned_right_shift(self.next_long(), 1)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value

    def next_int_bound(self, bound: int) -> int:
        return int(self.next_long_bound(bound))

    def next_boolean(self) -> bool:
        return (self.next_long() & 1) != 0

    def next_float(self) -> float:
        return float(_unsigned_right_shift(self.next_long(), 40)) * NORM_FLOAT

    def next_double(self) -> float:
        return float(_unsigned_right_shift(self.next_long(), 11)) * NORM_DOUBLE


class StsRandom:
    def __init__(self, seed: int, stream_name: str, counter: int = 0) -> None:
        self.seed = int(seed)
        self.stream_name = stream_name
        self.counter = 0
        self._random = RandomXS128(self.seed)
        self.calls: list[RandomCall] = []
        self.record_calls = True
        for _ in range(counter):
            self.random(999)

    def _record(self, method: str, args: tuple[Any, ...], result: Any) -> Any:
        self.counter += 1
        if bool(getattr(self, "record_calls", True)):
            self.calls.append(RandomCall(self.stream_name, method, args, result))
        return result

    def random(self, *args: int | float) -> int | float:
        if len(args) == 1:
            value = args[0]
            if isinstance(value, int):
                return self._record("random", (value,), self._random.next_long_bound(value + 1))
            return self._record("random", (value,), self._random.next_float() * value)
        start, end = args
        if isinstance(start, int) and isinstance(end, int):
            return self._record("random", (start, end), start + self._random.next_long_bound(end - start + 1))
        return self._record("random", (start, end), start + self._random.next_float() * (end - start))

    def random_boolean(self, chance: float | None = None) -> bool:
        if chance is None:
            return self._record("random_boolean", tuple(), self._random.next_boolean())
        return self._record("random_boolean", (chance,), self._random.next_float() < chance)

    def set_counter(self, target_counter: int) -> None:
        target = int(target_counter)
        if self.counter >= target:
            return
        for _ in range(target - self.counter):
            self.random_boolean()

    def random_long(self) -> int:
        return self._record("random_long", tuple(), self._random.next_long())

    def shuffle(self, seq: list[Any]) -> None:
        for index in range(len(seq) - 1, 0, -1):
            swap_index = self._random.next_int_bound(index + 1)
            seq[index], seq[swap_index] = seq[swap_index], seq[index]
        if bool(getattr(self, "record_calls", True)):
            self.calls.append(RandomCall(self.stream_name, "shuffle", (len(seq),), list(seq)))


class JavaRandom:
    def __init__(self, seed: int) -> None:
        self.seed = (int(seed) ^ JAVA_MULTIPLIER) & JAVA_MASK_48

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * JAVA_MULTIPLIER + JAVA_ADDEND) & JAVA_MASK_48
        return self.seed >> (48 - bits)

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound & (bound - 1) == 0:
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value


def java_shuffle_in_place(seq: list[Any], seed: int) -> None:
    rng = JavaRandom(seed)
    for index in range(len(seq), 1, -1):
        swap_index = rng.next_int(index)
        seq[index - 1], seq[swap_index] = seq[swap_index], seq[index - 1]


def _map_seed(base_seed: int, act: int) -> int:
    base = int(base_seed)
    act_num = int(act)
    if act_num <= 1:
        return base + 1
    if act_num == 2:
        return base + act_num * 100
    if act_num == 3:
        return base + act_num * 200
    return base + act_num


@dataclass(slots=True)
class NativeRandomSet:
    seed: int
    act: int = 1
    floor: int = 0
    streams: dict[str, StsRandom] = field(init=False)

    def __post_init__(self) -> None:
        base = int(self.seed)
        self.streams = {
            "neow": StsRandom(base, "neow"),
            "monster": StsRandom(base, "monster"),
            "event": StsRandom(base, "event"),
            "merchant": StsRandom(base, "merchant"),
            "card": StsRandom(base, "card"),
            "treasure": StsRandom(base, "treasure"),
            "relic": StsRandom(base, "relic"),
            "potion": StsRandom(base, "potion"),
            "map": StsRandom(_map_seed(base, self.act), "map"),
            "monster_hp": StsRandom(base + int(self.floor), "monster_hp"),
            "ai": StsRandom(base + int(self.floor), "ai"),
            "shuffle": StsRandom(base + int(self.floor), "shuffle"),
            "card_random": StsRandom(base + int(self.floor), "card_random"),
            "misc": StsRandom(base + int(self.floor), "misc"),
        }

    def stream(self, name: str) -> StsRandom:
        return self.streams[name]

    def duplicate_stream(self, name: str, *, alias: str | None = None) -> StsRandom:
        source = self.streams[name]
        return StsRandom(source.seed, alias or source.stream_name, counter=source.counter)

    def reset_floor_streams(self, floor: int) -> None:
        self.floor = int(floor)
        base = int(self.seed) + self.floor
        for name in ("monster_hp", "ai", "shuffle", "card_random", "misc"):
            self.streams[name] = StsRandom(base, name)

    def reset_act_stream(self, act: int) -> None:
        self.act = int(act)
        self.streams["map"] = StsRandom(_map_seed(int(self.seed), self.act), "map")

    def debug_trace(self) -> dict[str, list[dict[str, Any]]]:
        def _jsonable(value: Any) -> Any:
            if is_dataclass(value):
                return {key: _jsonable(inner) for key, inner in asdict(value).items()}
            if isinstance(value, list):
                return [_jsonable(item) for item in value]
            if isinstance(value, tuple):
                return [_jsonable(item) for item in value]
            if isinstance(value, dict):
                return {str(key): _jsonable(inner) for key, inner in value.items()}
            return value

        return {
            name: [
                {
                    "stream": call.stream,
                    "method": call.method,
                    "args": _jsonable(call.args),
                    "result": _jsonable(call.result),
                }
                for call in random.calls
            ]
            for name, random in self.streams.items()
        }

    def debug_state(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "seed": int(random.seed),
                "counter": int(random.counter),
                "call_count": len(random.calls),
                "seed0": int(random._random.seed0),
                "seed1": int(random._random.seed1),
            }
            for name, random in self.streams.items()
        }
