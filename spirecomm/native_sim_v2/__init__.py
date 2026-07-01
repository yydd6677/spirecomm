"""Lightspeed-aligned native simulator backend.

This package keeps the spirecomm-facing interface stable while we replace the
old approximation layer subsystem-by-subsystem.
"""

from spirecomm.native_sim_v2.env import NativeCombatEnv, NativeRunEnv

__all__ = ["NativeCombatEnv", "NativeRunEnv"]
