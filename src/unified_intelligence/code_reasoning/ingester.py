"""
Code ingestion module for extracting facts from Python code.

Adapted from mangle_code_scaffold_v2/scripts/ingest_code.py
"""

import ast
import logging
import pathlib
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

COMPLEX_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.BoolOp,
    ast.IfExp,
    ast.With,
)


class CodeIngester:
    """Ingests Python code and extracts facts into SQLite database."""

    def __init__(self, db_path: str):
        # Handle in-memory database specially
        if db_path == ":memory:":
            self.db_path = db_path  # type: Union[str, pathlib.Path]
            self.is_memory_db = True
        else:
            self.db_path = pathlib.Path(
                db_path
            ).resolve()  # type: Union[str, pathlib.Path]
            self.is_memory_db = False

    def norm_complexity(self, score: int) -> float:
        """Normalize complexity score to 0-1 range."""
        return min(1.0, score / 10.0)

    def module_name_from_path(
        self, root: pathlib.Path, path: pathlib.Path
    ) -> str:
        """Convert file path to module name."""
        rel = path.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        return ".".join(parts)

    def collect_symbols_calls_imports(
        self, src: str, mod: str, file_path: str
    ) -> tuple[list, list, list]:
        """Extract symbols, calls, and imports from source code."""
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return [], [], []

        symbols = []  # (symbol, kind, file, start, end, complexity)
        calls = []  # (caller_symbol, callee_symbol_or_name)
        imports = []  # (file, module)

        # Collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    (file_path, n.name.split(".")[0]) for n in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((file_path, node.module.split(".")[0]))

        class FnVisitor(ast.NodeVisitor):
            def __init__(self, norm_complexity_func):
                self.stack = []
                self.norm_complexity = norm_complexity_func

            def visit_FunctionDef(self, node: ast.FunctionDef):
                qual = f"{mod}::{node.name}"
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                comp = 1
                for n in ast.walk(node):
                    if isinstance(n, COMPLEX_NODES):
                        comp += 1
                symbols.append(
                    (
                        qual,
                        "function",
                        file_path,
                        start,
                        end,
                        self.norm_complexity(comp),
                    )
                )
                self.stack.append(qual)
                self.generic_visit(node)
                self.stack.pop()

            def visit_Call(self, node: ast.Call):
                callee = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if self.stack and callee:
                    calls.append((self.stack[-1], callee))
                self.generic_visit(node)

        FnVisitor(self.norm_complexity).visit(tree)
        return symbols, calls, imports

    def discover_tests_targets(
        self, test_src: str, test_mod: str
    ) -> tuple[list, list]:
        """Extract test functions and their call targets."""
        try:
            tree = ast.parse(test_src)
        except SyntaxError as e:
            logger.warning(f"Syntax error in test file {test_mod}: {e}")
            return [], []

        test_syms = []
        edges = []

        class TVisitor(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_FunctionDef(self, node: ast.FunctionDef):
                qual = f"{test_mod}::{node.name}"
                if node.name.startswith("test"):
                    start = getattr(node, "lineno", 0)
                    end = getattr(node, "end_lineno", start)
                    test_syms.append((qual, start, end))
                    self.stack.append(qual)
                    self.generic_visit(node)
                    self.stack.pop()
                else:
                    self.generic_visit(node)

            def visit_Call(self, node: ast.Call):
                callee = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if self.stack and callee:
                    edges.append((self.stack[-1], callee))
                self.generic_visit(node)

        TVisitor().visit(tree)
        return test_syms, edges

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connection."""
        # Handle in-memory database specially
        if str(self.db_path) == ":memory:":
            conn = sqlite3.connect(":memory:")
        else:
            conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize_database(self, conn: sqlite3.Connection):
        """Initialize database schema."""
        c = conn.cursor()
        c.executescript(
            """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS file(file TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS symbol(symbol TEXT PRIMARY KEY, kind TEXT, file TEXT, start INT, end INT);
        CREATE TABLE IF NOT EXISTS complexity(symbol TEXT, score REAL);
        CREATE TABLE IF NOT EXISTS imports(file TEXT, module TEXT);
        CREATE TABLE IF NOT EXISTS calls(caller TEXT, callee TEXT);
        CREATE TABLE IF NOT EXISTS dep(fileA TEXT, fileB TEXT);
        CREATE TABLE IF NOT EXISTS covered(symbol TEXT);
        CREATE TABLE IF NOT EXISTS defines_test(test_sym TEXT, file TEXT);
        CREATE TABLE IF NOT EXISTS tests_targets(test_sym TEXT, target_sym TEXT);
        """
        )
        conn.commit()

    def ingest_repository(
        self, repo_root: str, include_tests: bool = True
    ) -> dict[str, int]:
        """
        Ingest entire repository and return statistics.

        Args:
            repo_root: Path to repository root
            include_tests: Whether to include test files

        Returns:
            Dictionary with ingestion statistics
        """
        root = pathlib.Path(repo_root).resolve()

        if not root.exists():
            raise ValueError(f"Repository path does not exist: {root}")

        # Remove existing database (only for file-based databases)
        if not self.is_memory_db and pathlib.Path(self.db_path).exists():
            pathlib.Path(self.db_path).unlink()

        with self.get_db_connection() as conn:
            self.initialize_database(conn)
            c = conn.cursor()

            # Find all Python files
            py_files = list(root.rglob("*.py"))
            logger.info(f"Found {len(py_files)} Python files in {root}")

            # Insert files
            for path in py_files:
                rel = str(path.relative_to(root))
                rel_path = rel.replace("\\", "/")
                c.execute(
                    "INSERT OR IGNORE INTO file(file) VALUES (?)", (rel_path,)
                )

            # Process files
            all_symbols = []
            all_calls = []
            all_imports = []
            all_complexity = []
            all_def_tests = []
            all_tests_targets_edges = []
            name_to_symbols: dict[str, set[str]] = {}

            for path in py_files:
                rel = str(path.relative_to(root))
                rel_path = rel.replace("\\", "/")
                mod = self.module_name_from_path(root, path)

                try:
                    src = path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    logger.warning(f"Could not read {path}: {e}")
                    continue

                if include_tests and (
                    "/tests/" in rel_path or rel_path.startswith("tests/")
                ):
                    tests, edges = self.discover_tests_targets(src, mod)
                    for qual, start, end in tests:
                        all_def_tests.append((qual, rel_path))
                        all_symbols.append(
                            (qual, "function", rel_path, start, end)
                        )
                        all_complexity.append((qual, 0.0))
                    for e in edges:
                        all_tests_targets_edges.append(e)
                else:
                    symbols, calls, imports = (
                        self.collect_symbols_calls_imports(src, mod, rel_path)
                    )
                    for s in symbols:
                        all_symbols.append(s[:5])
                        all_complexity.append((s[0], s[5]))
                        simple = s[0].split("::")[-1]
                        name_to_symbols.setdefault(simple, set()).add(s[0])
                    for call in calls:
                        all_calls.append(call)
                    for imp in imports:
                        all_imports.append(imp)

            # Resolve calls
            resolved_calls = []
            for caller, callee_simple in all_calls:
                if callee_simple in name_to_symbols:
                    for fq in name_to_symbols[callee_simple]:
                        resolved_calls.append((caller, fq))
                else:
                    resolved_calls.append((caller, callee_simple))

            # Resolve test targets
            resolved_tests_targets = []
            for test_sym, callee_simple in all_tests_targets_edges:
                if callee_simple in name_to_symbols:
                    for fq in name_to_symbols[callee_simple]:
                        resolved_tests_targets.append((test_sym, fq))

            # Insert data
            c.executemany(
                "INSERT OR IGNORE INTO symbol(symbol, kind, file, start, end) VALUES (?,?,?,?,?)",
                all_symbols,
            )
            c.executemany(
                "INSERT OR IGNORE INTO complexity(symbol, score) VALUES (?,?)",
                all_complexity,
            )
            c.executemany(
                "INSERT OR IGNORE INTO imports(file, module) VALUES (?,?)",
                all_imports,
            )
            c.executemany(
                "INSERT OR IGNORE INTO calls(caller, callee) VALUES (?,?)",
                resolved_calls,
            )
            c.executemany(
                "INSERT OR IGNORE INTO defines_test(test_sym, file) VALUES (?,?)",
                all_def_tests,
            )
            c.executemany(
                "INSERT OR IGNORE INTO tests_targets(test_sym, target_sym) VALUES (?,?)",
                resolved_tests_targets,
            )

            # Calculate dependencies
            files_index = {
                f: True
                for (f,) in conn.execute("SELECT file FROM file").fetchall()
            }

            def module_to_path(mod: str) -> str:
                return mod.replace(".", "/") + ".py"

            deps = set()
            for f, m in all_imports:
                target = module_to_path(m)
                if target in files_index:
                    deps.add((f, target))
            c.executemany(
                "INSERT OR IGNORE INTO dep(fileA, fileB) VALUES (?,?)",
                list(deps),
            )

            conn.commit()

            stats = {
                "files_processed": len(py_files),
                "symbols_extracted": len(all_symbols),
                "calls_extracted": len(resolved_calls),
                "imports_extracted": len(all_imports),
                "tests_extracted": len(all_def_tests),
                "dependencies_calculated": len(deps),
            }

            logger.info(f"Ingestion complete: {stats}")
            return stats
