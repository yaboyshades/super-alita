#!/usr/bin/env python3
"""Unified SDD + Mangle fact ingestion and decision script."""

from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from pathlib import Path
from typing import Any

try:
    from src.core.yaml_utils import safe_load as _repo_yaml_safe_load  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _repo_yaml_safe_load = None

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS file(file TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS symbol(symbol TEXT PRIMARY KEY, kind TEXT, file TEXT, start INT, end INT);
CREATE TABLE IF NOT EXISTS complexity(symbol TEXT, score REAL);
CREATE TABLE IF NOT EXISTS imports(file TEXT, module TEXT);
CREATE TABLE IF NOT EXISTS calls(caller TEXT, callee TEXT);
CREATE TABLE IF NOT EXISTS dep(fileA TEXT, fileB TEXT);
CREATE TABLE IF NOT EXISTS defines_test(test_sym TEXT, file TEXT);
CREATE TABLE IF NOT EXISTS tests_targets(test_sym TEXT, target_sym TEXT);
CREATE TABLE IF NOT EXISTS spec_feature(fid TEXT, title TEXT);
CREATE TABLE IF NOT EXISTS spec_symbol(fid TEXT, symbol TEXT);
CREATE TABLE IF NOT EXISTS spec_test(fid TEXT, test_sym TEXT);
CREATE TABLE IF NOT EXISTS spec_contract(fid TEXT, contract_id TEXT);
CREATE TABLE IF NOT EXISTS spec_metric(fid TEXT, metric TEXT);
CREATE TABLE IF NOT EXISTS spec_risk(fid TEXT, risk_id TEXT);
"""

COMPLEX_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.IfExp, ast.With)
SPEC_EXTENSIONS = {".yaml", ".yml", ".json"}
SPEC_MISSING_REASON = "Spec unavailable—reduced checks"

SpecStatus = dict[str, Any]


def _norm_complexity(raw: int) -> float:
    return min(1.0, raw / 10.0)


def _module_from_path(root: Path, py_path: Path) -> str:
    return ".".join(py_path.relative_to(root).with_suffix("").parts)


def ingest_code(conn: sqlite3.Connection, repo: Path) -> None:
    cursor = conn.cursor()
    py_files = [p for p in repo.rglob("*.py") if p.is_file()]

    for py in py_files:
        rel = py.relative_to(repo)
        cursor.execute("INSERT OR IGNORE INTO file(file) VALUES (?)", (str(rel),))

    name_index: dict[str, set[str]] = {}
    symbols: list[tuple[str, str, str, int, int]] = []
    complexities: list[tuple[str, float]] = []
    imports: list[tuple[str, str]] = []
    calls: list[tuple[str, str]] = []
    defined_tests: list[tuple[str, str]] = []
    test_edges: list[tuple[str, str]] = []

    for py in py_files:
        rel = py.relative_to(repo)
        module = _module_from_path(repo, py)
        source = py.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((str(rel), alias.name.split(".")[0]))
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((str(rel), node.module.split(".")[0]))

        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_callable(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
                self._visit_callable(node)

            def _visit_callable(self, node: ast.AST) -> None:
                name = getattr(node, "name", None)
                if not isinstance(name, str):
                    return
                symbol = f"{module}::{name}"
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                complexity_score = 1 + sum(1 for n in ast.walk(node) if isinstance(n, COMPLEX_NODES))
                symbols.append((symbol, "function", str(rel), start, end))
                complexities.append((symbol, _norm_complexity(complexity_score)))
                self.stack.append(symbol)
                self.generic_visit(node)
                self.stack.pop()

            def visit_Call(self, node: ast.Call) -> None:
                callee: str | None = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if self.stack and callee:
                    calls.append((self.stack[-1], callee))
                self.generic_visit(node)

        FunctionVisitor().visit(tree)

        if str(rel).startswith("tests/") or "/tests/" in str(rel):
            class TestVisitor(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.stack: list[str] = []

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    self._visit_test(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
                    self._visit_test(node)

                def _visit_test(self, node: ast.AST) -> None:
                    name = getattr(node, "name", None)
                    if not isinstance(name, str):
                        return
                    symbol = f"{module}::{name}"
                    if name.startswith("test"):
                        defined_tests.append((symbol, str(rel)))
                        self.stack.append(symbol)
                        self.generic_visit(node)
                        self.stack.pop()
                    else:
                        self.generic_visit(node)

                def visit_Call(self, node: ast.Call) -> None:
                    callee: str | None = None
                    if isinstance(node.func, ast.Name):
                        callee = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        callee = node.func.attr
                    if self.stack and callee:
                        test_edges.append((self.stack[-1], callee))
                    self.generic_visit(node)

            TestVisitor().visit(tree)

        for symbol, _, file_name, *_ in [s for s in symbols if s[2] == str(rel)]:
            simple_name = symbol.split("::")[-1]
            name_index.setdefault(simple_name, set()).add(symbol)

    resolved_calls: list[tuple[str, str]] = []
    for caller, callee_simple in calls:
        if callee_simple in name_index:
            for full in name_index[callee_simple]:
                resolved_calls.append((caller, full))
        else:
            resolved_calls.append((caller, callee_simple))

    resolved_tests: list[tuple[str, str]] = []
    for test_symbol, callee_simple in test_edges:
        if callee_simple in name_index:
            for full in name_index[callee_simple]:
                resolved_tests.append((test_symbol, full))

    cursor.executemany(
        "INSERT OR IGNORE INTO symbol(symbol, kind, file, start, end) VALUES (?, ?, ?, ?, ?)",
        symbols,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO complexity(symbol, score) VALUES (?, ?)",
        complexities,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO imports(file, module) VALUES (?, ?)",
        imports,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO calls(caller, callee) VALUES (?, ?)",
        resolved_calls,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO defines_test(test_sym, file) VALUES (?, ?)",
        defined_tests,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO tests_targets(test_sym, target_sym) VALUES (?, ?)",
        resolved_tests,
    )

    files_in_db = {row[0] for row in cursor.execute("SELECT file FROM file")}
    dependencies: set[tuple[str, str]] = set()
    for src_file, module in imports:
        candidate = module.replace(".", "/") + ".py"
        if candidate in files_in_db:
            dependencies.add((src_file, candidate))

    cursor.executemany(
        "INSERT OR IGNORE INTO dep(fileA, fileB) VALUES (?, ?)",
        list(dependencies),
    )
    conn.commit()


def _load_structured_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        if _repo_yaml_safe_load is not None:
            return _repo_yaml_safe_load(text)
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot read YAML spec")
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported spec file extension: {path.suffix}")


def ingest_spec(conn: sqlite3.Connection, spec_path: Path) -> SpecStatus:
    status: SpecStatus = {
        "present": False,
        "warnings": [],
        "had_errors": False,
        "path": str(spec_path),
    }

    if not spec_path.exists():
        status["warnings"].append(f"Spec path '{spec_path}' not found")
        status["had_errors"] = True
        return status

    if spec_path.is_file():
        files = [spec_path]
    elif spec_path.is_dir():
        files = [p for p in spec_path.rglob("*") if p.suffix.lower() in SPEC_EXTENSIONS and p.is_file()]
    else:
        status["warnings"].append(f"Spec path '{spec_path}' is neither file nor directory")
        status["had_errors"] = True
        return status

    if not files:
        status["warnings"].append(f"No spec files detected under '{spec_path}'")
        status["had_errors"] = True
        return status

    cursor = conn.cursor()
    any_written = False

    for file_path in files:
        try:
            payload = _load_structured_spec(file_path)
        except Exception as exc:
            status["warnings"].append(f"{file_path}: {exc}")
            status["had_errors"] = True
            continue

        if not isinstance(payload, dict):
            status["warnings"].append(f"{file_path}: expected mapping at top level")
            status["had_errors"] = True
            continue

        feature_block = payload.get("feature")
        if not isinstance(feature_block, dict):
            feature_block = {}

        fid = feature_block.get("id") if isinstance(feature_block.get("id"), str) else payload.get("id")
        if not isinstance(fid, str) or not fid:
            fid = file_path.stem

        title = ""
        if isinstance(feature_block.get("title"), str):
            title = feature_block["title"]
        elif isinstance(payload.get("title"), str):
            title = payload["title"]

        cursor.execute(
            "INSERT OR REPLACE INTO spec_feature(fid, title) VALUES (?, ?)",
            (fid, title),
        )

        symbols: list[str] = []
        scope = feature_block.get("scope") if isinstance(feature_block.get("scope"), dict) else {}
        if isinstance(scope, dict) and isinstance(scope.get("symbols"), list):
            symbols.extend(str(sym) for sym in scope["symbols"])
        if isinstance(payload.get("symbols"), list):
            symbols.extend(str(sym) for sym in payload["symbols"])

        tests: list[str] = []
        tests_required = feature_block.get("tests_required")
        if isinstance(tests_required, list):
            tests.extend(str(test) for test in tests_required)
        if isinstance(payload.get("tests_required"), list):
            tests.extend(str(test) for test in payload["tests_required"])

        contracts: list[str] = []
        contracts_block = feature_block.get("contracts")
        if isinstance(contracts_block, list):
            for contract in contracts_block:
                if isinstance(contract, dict) and "id" in contract:
                    contracts.append(str(contract["id"]))
                else:
                    contracts.append(str(contract))
        top_contracts = payload.get("contracts")
        if isinstance(top_contracts, list):
            for contract in top_contracts:
                if isinstance(contract, dict) and "id" in contract:
                    contracts.append(str(contract["id"]))
                else:
                    contracts.append(str(contract))

        metrics: list[str] = []
        observability = feature_block.get("observability")
        if isinstance(observability, dict) and isinstance(observability.get("metrics"), list):
            metrics.extend(str(metric) for metric in observability["metrics"])
        if isinstance(payload.get("metrics"), list):
            metrics.extend(str(metric) for metric in payload["metrics"])

        risks: list[str] = []
        risk_block = feature_block.get("risks")
        if isinstance(risk_block, list):
            for risk in risk_block:
                if isinstance(risk, dict) and "id" in risk:
                    risks.append(str(risk["id"]))
                else:
                    risks.append(str(risk))
        top_risks = payload.get("risks")
        if isinstance(top_risks, list):
            for risk in top_risks:
                if isinstance(risk, dict) and "id" in risk:
                    risks.append(str(risk["id"]))
                else:
                    risks.append(str(risk))

        cursor.executemany(
            "INSERT INTO spec_symbol(fid, symbol) VALUES (?, ?)",
            [(fid, sym) for sym in symbols],
        )
        cursor.executemany(
            "INSERT INTO spec_test(fid, test_sym) VALUES (?, ?)",
            [(fid, test) for test in tests],
        )
        cursor.executemany(
            "INSERT INTO spec_contract(fid, contract_id) VALUES (?, ?)",
            [(fid, contract) for contract in contracts],
        )
        cursor.executemany(
            "INSERT INTO spec_metric(fid, metric) VALUES (?, ?)",
            [(fid, metric) for metric in metrics],
        )
        cursor.executemany(
            "INSERT INTO spec_risk(fid, risk_id) VALUES (?, ?)",
            [(fid, risk) for risk in risks],
        )
        any_written = True

    conn.commit()
    status["present"] = any_written
    if not any_written:
        status["had_errors"] = True
    if status["warnings"]:
        status["had_errors"] = True
    return status


RULES = {
    "untested_function": """
SELECT s.symbol, s.file, c.score AS complexity
FROM symbol s
JOIN complexity c ON c.symbol = s.symbol
WHERE s.kind = 'function' AND c.score >= 0.3
  AND s.symbol NOT IN (SELECT target_sym FROM tests_targets)
""",
    "cycle": """
SELECT a.fileA, a.fileB
FROM dep a
JOIN dep b ON a.fileA = b.fileB AND a.fileB = b.fileA
WHERE a.fileA <> a.fileB
""",
    "missing_impl": """
SELECT ss.fid, ss.symbol
FROM spec_symbol ss
LEFT JOIN symbol s ON s.symbol = ss.symbol
WHERE s.symbol IS NULL
""",
    "missing_test": """
SELECT st.fid, st.test_sym
FROM spec_test st
LEFT JOIN defines_test dt ON dt.test_sym = st.test_sym
WHERE dt.test_sym IS NULL
""",
    "cycle_spec": """
WITH spec_files AS (
    SELECT DISTINCT file FROM symbol
    WHERE symbol IN (SELECT symbol FROM spec_symbol)
)
SELECT a.fileA, a.fileB
FROM dep a
JOIN dep b ON a.fileA = b.fileB AND a.fileB = b.fileA
WHERE a.fileA <> a.fileB AND (a.fileA IN spec_files OR a.fileB IN spec_files)
""",
    "reinvention_json": """
SELECT s.symbol, s.file
FROM symbol s
WHERE s.kind = 'function'
  AND (LOWER(s.symbol) LIKE '%::to_json%' OR LOWER(s.symbol) LIKE '%::json_%')
EXCEPT
SELECT s.symbol, s.file
FROM symbol s
JOIN imports i ON i.file = s.file AND i.module = 'json'
""",
    "spec_creep": """
SELECT s.symbol, s.file
FROM symbol s
WHERE s.symbol LIKE 'analyzer.serialize.%'
  AND s.symbol NOT IN (SELECT symbol FROM spec_symbol)
""",
}


def run_rules(conn: sqlite3.Connection) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    report = {"rules": {}, "summary": {}}
    for name, sql in RULES.items():
        rows = [dict(row) for row in conn.execute(sql)]
        report["rules"][name] = rows
        report["summary"][name] = len(rows)
    return report


def fuse_to_advice(report: dict[str, Any], workflow_label: str, spec_status: SpecStatus) -> dict[str, Any]:
    summary = report["summary"]

    library_first = max(0.0, 1.0 - min(1.0, 0.2 * summary.get("reinvention_json", 0) + 0.1 * summary.get("spec_creep", 0)))
    test_first = max(0.0, 1.0 - (0.3 if summary.get("missing_test", 0) > 0 else 0.0) - min(0.5, 0.1 * summary.get("untested_function", 0)))
    simplicity = 0.4 if (summary.get("cycle", 0) > 0 or summary.get("cycle_spec", 0) > 0) else 1.0
    articles = {
        "library_first": round(library_first, 3),
        "test_first": round(test_first, 3),
        "simplicity_gate": round(simplicity, 3),
    }
    constitution = round(0.45 * test_first + 0.35 * simplicity + 0.20 * library_first, 3)

    mangle_score = round(max(0.0, 1.0 - min(1.0, 0.15 * summary.get("untested_function", 0) + 0.15 * summary.get("cycle", 0))), 3)
    workflow_conf = 0.8 if workflow_label in {"new_feature", "refactor", "debug"} else 0.6

    w_m, w_c, w_w = 0.35, 0.45, 0.20
    if workflow_label in {"new_feature", "refactor", "debug"}:
        w_m += 0.15
    total = w_m + w_c + w_w
    w_m, w_c, w_w = w_m / total, w_c / total, w_w / total

    fused = round(w_m * mangle_score + w_c * constitution + w_w * workflow_conf, 3)

    if constitution < 0.40:
        decision = "revise"
    elif fused >= 0.70:
        decision = "proceed"
    elif fused >= 0.50:
        decision = "revise"
    else:
        decision = "block"

    reasons: list[str] = []
    if summary.get("missing_test", 0) > 0:
        reasons.append("Required spec tests missing")
    if summary.get("cycle", 0) or summary.get("cycle_spec", 0):
        reasons.append("Cycle detected in module dependencies")
    if summary.get("reinvention_json", 0):
        reasons.append("JSON handling without stdlib usage")
    if summary.get("missing_impl", 0):
        reasons.append("Spec symbol lacks implementation")
    if summary.get("spec_creep", 0):
        reasons.append("Analyzer serialize path not declared in spec")
    if spec_status.get("had_errors") or not spec_status.get("present"):
        reasons.append(SPEC_MISSING_REASON)

    return {
        "decision": decision,
        "reasons": reasons,
        "scores": {
            "fused": fused,
            "contributors": {
                "mangle": mangle_score,
                "constitution": constitution,
                "workflow": round(workflow_conf, 3),
            },
            "articles": articles,
        },
        "notes": {"spec": spec_status},
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified SDD + Mangle ingestion")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--spec", default=".spec")
    parser.add_argument("--db", default="facts.sqlite")
    parser.add_argument("--report", default="report.json")
    parser.add_argument("--advice", default="advice.json")
    parser.add_argument("--workflow", default="refactor")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    spec_path = Path(args.spec).resolve()
    db_path = Path(args.db).resolve()
    report_path = Path(args.report).resolve()
    advice_path = Path(args.advice).resolve()

    _ensure_parent(db_path)
    _ensure_parent(report_path)
    _ensure_parent(advice_path)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    ingest_code(conn, repo_root)
    spec_status = ingest_spec(conn, spec_path)

    report = run_rules(conn)
    report["meta"] = {"spec": spec_status}
    advice = fuse_to_advice(report, args.workflow, spec_status)

    report_path.write_text(json.dumps(report, indent=2))
    advice_path.write_text(json.dumps(advice, indent=2))

    print(
        json.dumps(
            {
                "ok": True,
                "report": str(report_path),
                "advice": str(advice_path),
                "summary": report["summary"],
                "decision": advice["decision"],
                "spec_present": bool(spec_status.get("present")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
