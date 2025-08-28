from __future__ import annotations

import argparse
import json
import sys

from src.core.optimization.strategy_selector import StrategySelector


def cmd_select(args: argparse.Namespace) -> int:
    ss = StrategySelector(config_path=args.config)
    decision = ss.select(task_type=args.task_type, context={"user": args.user or "cli"})
    out = {
        "decision_id": decision.decision_id,
        "task_type": decision.task_type,
        "arm_id": decision.arm_id,
        "arm_name": decision.arm_name,
        "metadata": decision.metadata,
        "algorithm": decision.algorithm,
        "confidence": decision.confidence,
        "timestamp": decision.timestamp,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    ss = StrategySelector(config_path=args.config)
    if ss.feedback(task_type=args.task_type, decision_id=args.decision_id, reward=float(args.reward)):
        print("ok")
        return 0
    print("not-found", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strategy selection CLI (A/B for reasoning styles)")
    p.add_argument("command", choices=["select", "feedback"])
    p.add_argument("task_type", help="Task type key in config/strategies.json")
    p.add_argument("--config", default="config/strategies.json")
    p.add_argument("--user", default=None)
    p.add_argument("--decision-id", dest="decision_id")
    p.add_argument("--reward", type=float)

    args = p.parse_args(argv)
    if args.command == "select":
        return cmd_select(args)
    elif args.command == "feedback":
        if not args.decision_id or args.reward is None:
            p.error("feedback requires --decision-id and --reward")
        return cmd_feedback(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

