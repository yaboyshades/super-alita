"""Control-flow graph hashing middleware for Copilot Agent mode."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx

_CACHE_PATH = Path('.cache/copilot_cfg_hashes.json')
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Dict[str, Dict[str, str]]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: Dict[str, Dict[str, str]]) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding='utf-8')


class CFGBuilder(ast.NodeVisitor):
    def __init__(self, graph: nx.DiGraph, prefix: str) -> None:
        self.graph = graph
        self.prefix = prefix
        self._counter = 0

    def _new_node(self, label: str, lineno: int) -> str:
        node_id = f"{self.prefix}_{self._counter}"
        self._counter += 1
        self.graph.add_node(node_id, label=label, lineno=lineno)
        return node_id

    def build(self, body: Iterable[ast.stmt]) -> None:
        previous: str | None = None
        for stmt in body:
            label = type(stmt).__name__
            node = self._new_node(label, getattr(stmt, 'lineno', -1))
            if previous is not None:
                self.graph.add_edge(previous, node)
            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                self._handle_branch(node, stmt)
            elif isinstance(stmt, ast.Try):
                self._handle_try(node, stmt)
            previous = node

    def _handle_branch(self, parent: str, stmt: ast.AST) -> None:
        body = getattr(stmt, 'body', [])
        orelse = getattr(stmt, 'orelse', [])
        if body:
            body_start = self._new_node(f"{type(stmt).__name__}_body", getattr(body[0], 'lineno', -1))
            self.graph.add_edge(parent, body_start)
            self.build(body)
        if orelse:
            else_start = self._new_node(f"{type(stmt).__name__}_else", getattr(orelse[0], 'lineno', -1))
            self.graph.add_edge(parent, else_start)
            self.build(orelse)

    def _handle_try(self, parent: str, stmt: ast.Try) -> None:
        if stmt.body:
            body_start = self._new_node('TryBody', getattr(stmt.body[0], 'lineno', -1))
            self.graph.add_edge(parent, body_start)
            self.build(stmt.body)
        for handler in stmt.handlers:
            if handler.body:
                handler_start = self._new_node('ExceptHandler', getattr(handler.body[0], 'lineno', -1))
                self.graph.add_edge(parent, handler_start)
                self.build(handler.body)
        if stmt.orelse:
            else_start = self._new_node('TryElse', getattr(stmt.orelse[0], 'lineno', -1))
            self.graph.add_edge(parent, else_start)
            self.build(stmt.orelse)
        if stmt.finalbody:
            finally_start = self._new_node('TryFinally', getattr(stmt.finalbody[0], 'lineno', -1))
            self.graph.add_edge(parent, finally_start)
            self.build(stmt.finalbody)


def build_cfg(text: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            prefix = f"{node.name}_{node.lineno}"
            builder = CFGBuilder(graph, prefix)
            builder.build(node.body)
    return graph


def compute_hashes(text: str) -> Dict[str, str]:
    graph = build_cfg(text)
    hashes: Dict[str, str] = {}
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            prefix = f"{node.name}_{node.lineno}"
            sub_nodes = [n for n in graph.nodes if n.startswith(prefix)]
            subgraph = graph.subgraph(sub_nodes).copy()
            if subgraph.number_of_nodes() == 0:
                continue
            canonical = nx.weisfeiler_lehman_graph_hash(
                subgraph, node_attr='label', edge_attr=None
            )
            hashes[prefix] = canonical
    return hashes


def check_cfg_uniqueness(text: str, *, file_path: str | None = None) -> dict:
    if file_path is None:
        return {'passed': True, 'skipped': 'no file provided'}
    cache = _load_cache()
    hashes = compute_hashes(text)
    duplicates: List[dict] = []
    for prefix, canonical in hashes.items():
        entry = cache.get(canonical)
        if entry and entry['path'] != file_path:
            duplicates.append({
                'function': prefix,
                'existing_path': entry['path'],
                'existing_function': entry['function'],
            })
        else:
            cache[canonical] = {'path': file_path, 'function': prefix}
    if duplicates:
        return {
            'passed': False,
            'duplicates': duplicates,
            'message': 'Detected duplicate control-flow graph across files.',
        }
    _save_cache(cache)
    return {'passed': True, 'tracked': list(hashes.keys())}


def noop_inject(text: str) -> str:
    return text


def main(argv: List[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print('Usage: cfg_hash_guard.py <python-file>', file=sys.stderr)
        return 2
    target = Path(argv[0])
    if not target.exists():
        print(f'File {target} not found', file=sys.stderr)
        return 2
    text = target.read_text(encoding='utf-8')
    report = check_cfg_uniqueness(text, file_path=str(target))
    print(json.dumps(report, indent=2))
    return 0 if report.get('passed', False) else 1


if __name__ == '__main__':  # pragma: no cover - CLI hook
    sys.exit(main())
