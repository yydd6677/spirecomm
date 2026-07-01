from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, MutableSequence, Sequence, TypeVar


T = TypeVar("T")


class JavaRandom:
    """java.util.Random compatible RNG used by Slay the Spire shuffles."""

    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1

    def __init__(self, seed: int):
        self.seed = (int(seed) ^ self.MULTIPLIER) & self.MASK

    def next(self, bits: int) -> int:
        self.seed = (self.seed * self.MULTIPLIER + self.ADDEND) & self.MASK
        return self.seed >> (48 - int(bits))

    def next_int(self, bound: int | None = None) -> int:
        if bound is None:
            value = self.next(32)
            return value - (1 << 32) if value >= (1 << 31) else value
        bound = int(bound)
        if bound <= 0:
            raise ValueError("bound must be positive")
        value = self.next(31)
        mask = bound - 1
        if (bound & mask) == 0:
            return (bound * value) >> 31
        while True:
            result = value % bound
            if value - result + mask >= 0:
                return result
            value = self.next(31)


def java_collections_shuffle(sequence: MutableSequence[T], seed: int) -> None:
    rng = JavaRandom(seed)
    for size in range(len(sequence), 1, -1):
        index = rng.next_int(size)
        sequence[size - 1], sequence[index] = sequence[index], sequence[size - 1]


class StsRandom:
    """Slay the Spire style xorshift RNG, ported from sts_lightspeed."""

    NORM_DOUBLE = 1.1102230246251565e-16
    NORM_FLOAT = 5.9604644775390625e-8
    ONE_IN_MOST_SIGNIFICANT = 1 << 63
    MASK_64 = (1 << 64) - 1

    def __init__(self, seed: int = 0, target_counter: int = 0):
        self.counter = 0
        seed = int(seed) & self.MASK_64
        self.seed0 = self.murmur_hash3(self.ONE_IN_MOST_SIGNIFICANT if seed == 0 else seed)
        self.seed1 = self.murmur_hash3(self.seed0)
        for _ in range(max(0, int(target_counter))):
            self.random(999)

    def copy(self) -> "StsRandom":
        clone = StsRandom.__new__(StsRandom)
        clone.counter = self.counter
        clone.seed0 = self.seed0
        clone.seed1 = self.seed1
        return clone

    @classmethod
    def murmur_hash3(cls, value: int) -> int:
        value &= cls.MASK_64
        value ^= value >> 33
        value = (value * ((-49064778989728563) & cls.MASK_64)) & cls.MASK_64
        value ^= value >> 33
        value = (value * ((-4265267296055464877) & cls.MASK_64)) & cls.MASK_64
        value ^= value >> 33
        return value & cls.MASK_64

    def _next_long_raw(self) -> int:
        s1 = self.seed0
        s0 = self.seed1
        self.seed0 = s0
        s1 ^= (s1 << 23) & self.MASK_64
        self.seed1 = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26)) & self.MASK_64
        return (self.seed1 + s0) & self.MASK_64

    def next_long(self, bound: int | None = None) -> int:
        if bound is None:
            return self._next_long_raw()
        return self.next_bounded_long(bound)

    def next_bounded_long(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        while True:
            bits = self._next_long_raw() >> 1
            candidate = bits % bound
            if ((bits - candidate + bound - 1) & self.MASK_64) < (1 << 63):
                return candidate

    def next_int(self, bound: int | None = None) -> int:
        if bound is None:
            value = self.next_long() & 0xFFFFFFFF
            return value - (1 << 32) if value >= (1 << 31) else value
        return int(self.next_bounded_long(bound))

    def next_float(self) -> float:
        return float(self.next_long() >> 40) * self.NORM_FLOAT

    def next_double(self) -> float:
        return float(self.next_long() >> 11) * self.NORM_DOUBLE

    def random(self, *args):
        self.counter += 1
        if not args:
            return self.next_float()
        if len(args) == 1:
            (upper,) = args
            if isinstance(upper, int):
                return self.next_int(upper + 1)
            return self.next_float() * float(upper)
        if len(args) == 2:
            start, end = args
            if isinstance(start, int) and isinstance(end, int):
                return int(start) + self.next_int(int(end) - int(start) + 1)
            return float(start) + self.next_float() * (float(end) - float(start))
        raise TypeError("random expected 0, 1, or 2 arguments")

    def random_boolean(self, chance: float | None = None) -> bool:
        self.counter += 1
        if chance is None:
            return bool(self.next_long() & 1)
        return self.next_float() < chance

    def set_counter(self, target_counter: int) -> None:
        while self.counter < target_counter:
            self.random_boolean()

    def randint(self, start: int, end: int) -> int:
        return int(self.random(start, end))

    def randrange(self, *args: int) -> int:
        if len(args) == 1:
            stop = int(args[0])
            if stop <= 0:
                raise ValueError("empty range for randrange()")
            return int(self.random(stop - 1))
        if len(args) == 2:
            start, stop = map(int, args)
            if stop <= start:
                raise ValueError("empty range for randrange()")
            return int(self.random(start, stop - 1))
        raise TypeError("randrange expected 1 or 2 arguments")

    def choice(self, sequence: Sequence[T]) -> T:
        if not sequence:
            raise IndexError("Cannot choose from an empty sequence")
        return sequence[int(self.random(len(sequence) - 1))]

    def shuffle(self, sequence: MutableSequence[T]) -> None:
        for i in range(len(sequence), 1, -1):
            j = int(self.random(i - 1))
            sequence[i - 1], sequence[j] = sequence[j], sequence[i - 1]

    def choices(self, population: Sequence[T], weights: Iterable[float] | None = None, k: int = 1) -> list[T]:
        if weights is None:
            return [self.choice(population) for _ in range(k)]
        weights_list = [float(weight) for weight in weights]
        total = sum(weights_list)
        if total <= 0:
            raise ValueError("total of weights must be greater than zero")
        result: list[T] = []
        for _ in range(k):
            roll = self.random() * total
            cumulative = 0.0
            for item, weight in zip(population, weights_list):
                cumulative += weight
                if roll < cumulative:
                    result.append(item)
                    break
            else:
                result.append(population[-1])
        return result

    def random_long(self) -> int:
        self.counter += 1
        value = self.next_long()
        return value - (1 << 64) if value >= (1 << 63) else value


@dataclass
class NativeRandomStreams:
    seed: int
    neow: StsRandom = field(init=False)
    treasure: StsRandom = field(init=False)
    event: StsRandom = field(init=False)
    relic: StsRandom = field(init=False)
    potion: StsRandom = field(init=False)
    card: StsRandom = field(init=False)
    card_random: StsRandom = field(init=False)
    merchant: StsRandom = field(init=False)
    monster: StsRandom = field(init=False)
    shuffle: StsRandom = field(init=False)
    misc: StsRandom = field(init=False)
    math_util: StsRandom = field(init=False)

    def __post_init__(self) -> None:
        self.neow = StsRandom(self.seed)
        self.treasure = StsRandom(self.seed)
        self.event = StsRandom(self.seed)
        self.relic = StsRandom(self.seed)
        self.potion = StsRandom(self.seed)
        self.card = StsRandom(self.seed)
        self.card_random = StsRandom(self.seed)
        self.merchant = StsRandom(self.seed)
        self.monster = StsRandom(self.seed)
        self.shuffle = StsRandom(self.seed)
        self.misc = StsRandom(self.seed)
        self.math_util = StsRandom(self.seed - 897897)

    @property
    def potion_rng(self) -> StsRandom:
        return self.potion

    @property
    def shop_rng(self) -> StsRandom:
        return self.merchant

    @property
    def event_rng(self) -> StsRandom:
        return self.event

    @property
    def relic_rng(self) -> StsRandom:
        return self.relic

    @property
    def treasure_rng(self) -> StsRandom:
        return self.treasure

    @property
    def card_rng(self) -> StsRandom:
        return self.card

    @property
    def card_random_rng(self) -> StsRandom:
        return self.card_random

    @property
    def monster_rng(self) -> StsRandom:
        return self.monster

    @property
    def shuffle_rng(self) -> StsRandom:
        return self.shuffle

    @property
    def misc_rng(self) -> StsRandom:
        return self.misc

    @property
    def neow_rng(self) -> StsRandom:
        return self.neow

    def map_rng(self, act: int) -> StsRandom:
        offset = 1 if act == 1 else act * (100 * (act - 1))
        return StsRandom(self.seed + offset)


__all__ = ["JavaRandom", "NativeRandomStreams", "StsRandom", "java_collections_shuffle"]
