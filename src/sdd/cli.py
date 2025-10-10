"""SDD CLI: thin wrapper around the constitutional SDD pipeline.

Usage examples:
- python -m src.sdd.cli specify "Build real-time chat" --workspace .
- python -m src.sdd.cli plan --spec specs/001-build-real-time-chat/spec.md --tech "FastAPI + WebSocket"
- python -m src.sdd.cli tasks --plan specs/001-build-real-time-chat/implementation-plan.md --priority integration-first --team-size 2
- python -m src.sdd.cli validate path/to/file_or_doc

This CLI intentionally avoids external deps (e.g., click) and uses argparse.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ..constitutional import ConstitutionalScorer
from .constitutional_pipeline import ConstitutionalSDDPipeline
from .models import PlanRequest, SpecifyRequest, TasksRequest


def _load_json_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Context file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError("Context JSON must be an object")
    return data  # type: ignore[return-value]


async def _cmd_specify(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    pipeline = ConstitutionalSDDPipeline(workspace)
    context = _load_json_file(args.context)
    req = SpecifyRequest(
        user_input=args.description,
        context=context,
        constitutional_gates=not args.no_gates,
    )
    res = await pipeline.specify(req)
    print(f"Created spec: {res.feature_path}")
    print(f"Compliance score: {res.overall_compliance_score:.2f}")
    if res.constitutional_compliance:
        print("Articles:")
        for k, v in res.constitutional_compliance.items():
            status = "OK" if v.compliant else "FAIL"
            print(f"  - {k}: {status} ({v.score:.2f})")
    return 0


async def _cmd_plan(args: argparse.Namespace) -> int:
    spec_path: Path
    if args.spec:
        spec_path = Path(args.spec)
    else:
        # Derive from feature dir
        feature_dir = Path(args.feature_dir or "").resolve()
        spec_path = feature_dir / "spec.md"
    pipeline = ConstitutionalSDDPipeline(Path(args.workspace).resolve())
    req = PlanRequest(
        specification_path=str(spec_path),
        technology_stack=args.tech or "",
        constraints={},
        constitutional_gates=not args.no_gates,
    )
    res = await pipeline.plan(req)
    print(f"Generated plan: {res.plan_path}")
    print(f"Compliance score: {res.overall_compliance_score:.2f}")
    if res.supporting_documents:
        print("Supporting docs:")
        for p in res.supporting_documents:
            print(f"  - {p}")
    return 0


async def _cmd_tasks(args: argparse.Namespace) -> int:
    plan_path: Path
    if args.plan:
        plan_path = Path(args.plan)
    else:
        feature_dir = Path(args.feature_dir or "").resolve()
        plan_path = feature_dir / "implementation-plan.md"
    pipeline = ConstitutionalSDDPipeline(Path(args.workspace).resolve())
    req = TasksRequest(
        plan_path=str(plan_path),
        priority_focus=args.priority,
        team_size=args.team_size,
        constitutional_gates=not args.no_gates,
    )
    res = await pipeline.tasks(req)
    print(f"Generated tasks: {res.tasks_path}")
    print(f"Compliance score: {res.overall_compliance_score:.2f}")
    print(f"Estimated total hours: {res.estimated_total_hours}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    content = path.read_text(encoding="utf-8")
    scorer = ConstitutionalScorer()
    if path.suffix == ".py":
        result = scorer.score_code(content, str(path))
    else:
        result = scorer.score_specification(content)
    print(
        f"Overall: {result.overall_score:.2f} (compliant={result.is_compliant})"
    )
    for article, score in result.article_scores.items():
        status = "OK" if score >= scorer.compliance_threshold else "FAIL"
        print(f"  - {article}: {status} ({score:.2f})")
    if result.violations:
        print("Violations:")
        for v in result.violations:
            where = f" (line {v.line})" if getattr(v, "line", None) else ""
            print(f"  - [{v.severity}] {v.article}{where}: {v.message}")
            if v.suggestion:
                print(f"    Suggestion: {v.suggestion}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Specification-Driven Development CLI"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # specify
    sp = sub.add_parser("specify", help="Create feature specification")
    sp.add_argument(
        "description", help="Feature description in natural language"
    )
    sp.add_argument(
        "--workspace", default=".", help="Workspace root directory"
    )
    sp.add_argument("--context", help="Path to JSON context file")
    sp.add_argument(
        "--no-gates", action="store_true", help="Disable constitutional gates"
    )
    sp.set_defaults(async_handler=_cmd_specify)

    # plan
    pp = sub.add_parser("plan", help="Generate implementation plan from spec")
    src = pp.add_mutually_exclusive_group(required=False)
    src.add_argument("--spec", help="Path to spec.md")
    src.add_argument(
        "--feature-dir", help="Feature directory containing spec.md"
    )
    pp.add_argument("--tech", help="Technology stack description")
    pp.add_argument(
        "--workspace", default=".", help="Workspace root directory"
    )
    pp.add_argument(
        "--no-gates", action="store_true", help="Disable constitutional gates"
    )
    pp.set_defaults(async_handler=_cmd_plan)

    # tasks
    tp = sub.add_parser("tasks", help="Generate task breakdown from plan")
    src2 = tp.add_mutually_exclusive_group(required=False)
    src2.add_argument("--plan", help="Path to implementation-plan.md")
    src2.add_argument(
        "--feature-dir", help="Feature directory containing plan"
    )
    tp.add_argument("--priority", default="test-first", help="Priority focus")
    tp.add_argument(
        "--team-size", type=int, default=1, help="Team size (1-10)"
    )
    tp.add_argument(
        "--workspace", default=".", help="Workspace root directory"
    )
    tp.add_argument(
        "--no-gates", action="store_true", help="Disable constitutional gates"
    )
    tp.set_defaults(async_handler=_cmd_tasks)

    # validate
    vp = sub.add_parser(
        "validate", help="Validate constitutional compliance for a file"
    )
    vp.add_argument("path", help="Path to .py code or .md/.txt specification")
    vp.set_defaults(handler=_cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "async_handler"):
        return asyncio.run(args.async_handler(args))
    if hasattr(args, "handler"):
        return args.handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
