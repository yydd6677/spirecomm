#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spirecomm.wiki_lightspeed_audit import DEFAULT_USER_AGENT, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gameplay pages on the Slay the Spire wiki against sts_lightspeed logic.")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/yydd/spirecomm"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/yydd/spirecomm/_cache/wiki_lightspeed_audit"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--scope", choices=["all", "ironclad"], default="all")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_audit(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        user_agent=args.user_agent,
        limit=args.limit,
        refresh=args.refresh,
        scope=args.scope,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
