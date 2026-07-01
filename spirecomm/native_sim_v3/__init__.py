"""Independent real-game-first simulator namespace.

`native_sim_v3` is intentionally rebuilt as a clean project. The simulator
core must not import or delegate rule logic to `v1`, `v2`, or `lightspeed`.
It remains an explicitly independent backend, but the primary repo CLI
entrypoints now default to `v3`.
"""

from spirecomm.native_sim_v3.env import NativeCombatEnv, NativeRunEnv

__all__ = ["NativeCombatEnv", "NativeRunEnv"]
