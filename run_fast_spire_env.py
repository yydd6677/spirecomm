#!/usr/bin/env python3
"""Run the spirecomm models against the lightspeed rule engine.

This is the spirecomm-side entry point for the model-driven fast environment.
lightspeed is imported only as a deterministic rules module; its built-in agent
and search policy are not used for strategy.
"""

import sys
from pathlib import Path


LIGHTSPEED_ROOT = Path("/home/yydd/sts_lightspeed")
LIGHTSPEED_TOOLS = LIGHTSPEED_ROOT / "tools"

if str(LIGHTSPEED_TOOLS) not in sys.path:
    sys.path.insert(0, str(LIGHTSPEED_TOOLS))

from run_model_lightspeed_env import main


if __name__ == "__main__":
    main()
