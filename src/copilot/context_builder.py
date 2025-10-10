from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256


@dataclass(slots=True)
class OpenFile:
    """Representation of a file opened in the editor."""

    path: str
    content: str
    selection: tuple[int, int] | None = None


@dataclass(slots=True)
class ChatContext:
    """Inputs provided by the client to build the copilot context."""

    open_files: list[OpenFile]
    attachments: list[OpenFile] = field(default_factory=list)
    quality_mode: str = "fast"


@dataclass(slots=True)
class CopilotContext:
    """Context derived from the chat request for the model."""

    top_comments: dict[str, str]
    imports: dict[str, list[str]]
    symbols: dict[str, list[str]]
    selections: dict[str, str]
    attachments: list[str]
    content_hash: str
    quality_mode: str


def _extract_top_comment(source: str) -> str | None:
    """Return the leading comment block at the top of ``source`` if present."""

    lines: list[str] = source.splitlines()
    comment_lines: list[str] = []
    for line in lines:
        striped = line.lstrip()
        if striped.startswith("#"):
            comment_lines.append(striped[1:].lstrip())
        elif striped == "":
            if comment_lines:
                comment_lines.append("")
        else:
            break
    return "\n".join(comment_lines).strip() or None


def _extract_imports(source: str) -> list[str]:
    """Return top-level import statements from ``source``."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                imports.append(name)
    return imports


def _extract_symbols(source: str) -> list[str]:
    """Return names of top-level classes and functions defined in ``source``."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols: list[str] = []
    for node in tree.body:
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            symbols.append(node.name)
    return symbols


def _extract_selection(content: str, selection: tuple[int, int]) -> str:
    start, end = selection
    lines = content.splitlines()
    selected = lines[start - 1 : end]
    return "\n".join(selected)


def _compute_content_hash(
    open_files: Iterable[OpenFile], attachments: Iterable[OpenFile]
) -> str:
    data = {
        "open_files": [
            {"path": f.path, "content": f.content, "selection": f.selection}
            for f in sorted(open_files, key=lambda f: f.path)
        ],
        "attachments": [
            {"path": f.path, "content": f.content, "selection": f.selection}
            for f in sorted(attachments, key=lambda f: f.path)
        ],
    }
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(canonical).hexdigest()


def build_copilot_context(chat: ChatContext) -> CopilotContext:
    """Derive a :class:`CopilotContext` from ``chat`` inputs."""

    top_comments: dict[str, str] = {}
    imports: dict[str, list[str]] = {}
    symbols: dict[str, list[str]] = {}
    selections: dict[str, str] = {}

    for file in chat.open_files:
        if (comment := _extract_top_comment(file.content)) is not None:
            top_comments[file.path] = comment
        imports[file.path] = _extract_imports(file.content)
        symbols[file.path] = _extract_symbols(file.content)
        if file.selection is not None:
            selections[file.path] = _extract_selection(
                file.content, file.selection
            )

    attachments_paths = [a.path for a in chat.attachments]
    content_hash = _compute_content_hash(chat.open_files, chat.attachments)

    return CopilotContext(
        top_comments=top_comments,
        imports=imports,
        symbols=symbols,
        selections=selections,
        attachments=attachments_paths,
        content_hash=content_hash,
        quality_mode=chat.quality_mode,
    )
