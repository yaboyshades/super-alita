import ast
import pathlib
import sqlite3
import sys

COMPLEX_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.IfExp, ast.With)

def norm_complexity(score: int) -> float:
    return min(1.0, score / 10.0)

def module_name_from_path(root: pathlib.Path, path: pathlib.Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    return ".".join(parts)

def collect_symbols_calls_imports(src: str, mod: str, file_path: str):
    tree = ast.parse(src)
    symbols = []  # (symbol, kind, file, start, end, complexity)
    calls = []    # (caller_symbol, callee_symbol_or_name)
    imports = []  # (file, module)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append((file_path, n.name.split('.')[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((file_path, node.module.split('.')[0]))

    class FnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node: ast.FunctionDef):
            qual = f"{mod}::{node.name}"
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            comp = 1
            for n in ast.walk(node):
                if isinstance(n, COMPLEX_NODES):
                    comp += 1
            symbols.append((qual, "function", file_path, start, end, norm_complexity(comp)))
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

    FnVisitor().visit(tree)
    return symbols, calls, imports

def discover_tests_targets(test_src: str, test_mod: str):
    tree = ast.parse(test_src)
    test_syms = []
    edges = []

    class TVisitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node: ast.FunctionDef):
            qual = f"{test_mod}::{node.name}"
            if node.name.startswith("test"):
                test_syms.append((qual,))
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python ingest_code.py <repo_root> <out_db>")
        sys.exit(1)
    root = pathlib.Path(sys.argv[1]).resolve()
    out_db = pathlib.Path(sys.argv[2]).resolve()

    if out_db.exists():
        out_db.unlink()

    conn = sqlite3.connect(str(out_db))
    c = conn.cursor()
    c.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE file(file TEXT PRIMARY KEY);
    CREATE TABLE symbol(symbol TEXT PRIMARY KEY, kind TEXT, file TEXT, start INT, end INT);
    CREATE TABLE complexity(symbol TEXT, score REAL);
    CREATE TABLE imports(file TEXT, module TEXT);
    CREATE TABLE calls(caller TEXT, callee TEXT);
    CREATE TABLE dep(fileA TEXT, fileB TEXT);
    CREATE TABLE covered(symbol TEXT);
    CREATE TABLE defines_test(test_sym TEXT, file TEXT);
    CREATE TABLE tests_targets(test_sym TEXT, target_sym TEXT);
    """)

    py_files = [p for p in root.rglob("*.py")]
    for path in py_files:
        rel = str(path.relative_to(root))
        c.execute("INSERT OR IGNORE INTO file(file) VALUES (?)", (rel,))

    all_symbols = []
    all_calls = []
    all_imports = []
    all_complexity = []
    all_def_tests = []
    all_tests_targets_edges = []
    name_to_symbols = {}

    for path in py_files:
        rel = str(path.relative_to(root))
        mod = module_name_from_path(root, path)
        src = path.read_text(encoding="utf-8", errors="ignore")
        if "/tests/" in rel or rel.startswith("tests/"):
            tests, edges = discover_tests_targets(src, mod)
            for t in tests:
                all_def_tests.append((t[0], rel))
            for e in edges:
                all_tests_targets_edges.append(e)
        else:
            symbols, calls, imports = collect_symbols_calls_imports(src, mod, rel)
            for s in symbols:
                all_symbols.append(s[:5])
                all_complexity.append((s[0], s[5]))
                simple = s[0].split("::")[-1]
                name_to_symbols.setdefault(simple, set()).add(s[0])
            for call in calls:
                all_calls.append(call)
            for imp in imports:
                all_imports.append(imp)

    resolved_calls = []
    for caller, callee_simple in all_calls:
        if callee_simple in name_to_symbols:
            for fq in name_to_symbols[callee_simple]:
                resolved_calls.append((caller, fq))
        else:
            resolved_calls.append((caller, callee_simple))

    resolved_tests_targets = []
    for test_sym, callee_simple in all_tests_targets_edges:
        if callee_simple in name_to_symbols:
            for fq in name_to_symbols[callee_simple]:
                resolved_tests_targets.append((test_sym, fq))

    c.executemany("INSERT OR IGNORE INTO symbol(symbol, kind, file, start, end) VALUES (?,?,?,?,?)", all_symbols)
    c.executemany("INSERT OR IGNORE INTO complexity(symbol, score) VALUES (?,?)", all_complexity)
    c.executemany("INSERT OR IGNORE INTO imports(file, module) VALUES (?,?)", all_imports)
    c.executemany("INSERT OR IGNORE INTO calls(caller, callee) VALUES (?,?)", resolved_calls)
    c.executemany("INSERT OR IGNORE INTO defines_test(test_sym, file) VALUES (?,?)", all_def_tests)
    c.executemany("INSERT OR IGNORE INTO tests_targets(test_sym, target_sym) VALUES (?,?)", resolved_tests_targets)

    files_index = {f: True for (f,) in conn.execute("SELECT file FROM file").fetchall()}
    def module_to_path(mod: str) -> str:
        return mod.replace(".", "/") + ".py"

    deps = set()
    for f, m in all_imports:
        target = module_to_path(m)
        if target in files_index:
            deps.add((f, target))
    c.executemany("INSERT OR IGNORE INTO dep(fileA, fileB) VALUES (?,?)", list(deps))

    conn.commit()
    print(f"Wrote facts to {out_db}")

if __name__ == "__main__":
    main()
