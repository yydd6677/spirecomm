try:
    from spirecomm.ai.agent import HybridAgent, SimpleAgent
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    HybridAgent = None
    SimpleAgent = None

try:
    from spirecomm.ai.lightspeed_combat_model import SerializedCombatSelector, V2CombatSelector
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    SerializedCombatSelector = None
    V2CombatSelector = None

from spirecomm.ai.policy import CombatPolicy, RuleBasedCombatPolicy, load_policy
from spirecomm.ai.recording import TrajectoryRecorder
