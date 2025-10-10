"""Repository MCP tool for self-wrapping codebase access.

Provides safe, read/write/search capabilities on the local repository with
lightweight git history access. Designed to be wired into the runtime ability
registry as dynamic tools.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

try:  # Optional pydantic for structured returns
    from pydantic import BaseModel
except Exception:  # pragma: no cover - fallback if pydantic unavailable
    BaseModel = object  # type: ignore

import contextlib

from src.core.proc import arun


class RepoFileInfo(BaseModel):  # type: ignore[misc]
    path: str
    size: int
    modified: float
    language: str | None = None


class RepoCommitInfo(BaseModel):  # type: ignore[misc]
    hash: str
    message: str
    author: str
    timestamp: float


class RepositoryMCPTool:
    """MCP-style tool for repository self-inspection and modification."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.max_file_size = 1024 * 1024  # 1MB safety limit
        self.event_bus: Any | None = None

    async def initialize(self, event_bus: Any, **_: Any) -> bool:
        self.event_bus = event_bus
        return True

    # --------------------------- File listing --------------------------- #
    async def list_files(
        self,
        directory: str = "",
        pattern: str = "**/*",
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            base = (
                (self.repo_root / directory).resolve()
                if directory
                else self.repo_root
            )
            if not base.exists() or not base.is_dir():
                return {"error": f"Directory {directory!r} not found"}

            default_excludes = [
                "**/__pycache__/**",
                "**/node_modules/**",
                "**/.git/**",
                "**/dist/**",
                "**/out/**",
            ]
            excludes = exclude_patterns or default_excludes

            files: list[dict[str, Any]] = []
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.resolve().relative_to(self.repo_root)
                rel_posix = rel.as_posix()
                if any(fnmatch.fnmatch(rel_posix, pat) for pat in excludes):
                    continue
                if pattern and not fnmatch.fnmatch(rel_posix, pattern):
                    continue
                st = p.stat()
                files.append(
                    {
                        "path": rel_posix,
                        "size": st.st_size,
                        "modified": st.st_mtime,
                        "language": self._detect_language(p),
                    }
                )
            return {
                "files": files,
                "directory": directory,
                "count": len(files),
            }
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"Failed to list files: {e}"}

    # --------------------------- File read ------------------------------ #
    async def read_file(
        self, file_path: str, max_lines: int | None = None
    ) -> dict[str, Any]:
        try:
            target = (self.repo_root / file_path).resolve()
            if not str(target).startswith(str(self.repo_root)):
                return {"error": "File path outside repository bounds"}
            if not target.exists() or not target.is_file():
                return {"error": f"File {file_path!r} not found"}
            if target.stat().st_size > self.max_file_size:
                return {"error": f"File {file_path!r} too large (>1MB)"}

            content = target.read_text(encoding="utf-8", errors="ignore")
            if max_lines is not None:
                lines = content.splitlines()
                truncated = len(lines) > max_lines
                content = "\n".join(lines[:max_lines]) + (
                    f"\n... (truncated, {len(lines) - max_lines} more lines)"
                    if truncated
                    else ""
                )
            return {
                "content": content,
                "path": Path(file_path).as_posix(),
                "size": target.stat().st_size,
                "language": self._detect_language(target),
            }
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"Failed to read file: {e}"}

    # --------------------------- File write ----------------------------- #
    async def write_file(
        self, file_path: str, content: str, create_dirs: bool = True
    ) -> dict[str, Any]:
        try:
            target = (self.repo_root / file_path).resolve()
            if not str(target).startswith(str(self.repo_root)):
                return {"error": "File path outside repository bounds"}
            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            backup_available = False
            if target.exists():
                backup_available = True
            target.write_text(content, encoding="utf-8")

            # Emit a simple event for observability (best-effort)
            if self.event_bus is not None:
                with contextlib.suppress(Exception):
                    await self.event_bus.emit(
                        {
                            "type": "file_modified",
                            "path": Path(file_path).as_posix(),
                            "size": len(content),
                            "backup_available": backup_available,
                        }
                    )

            return {
                "success": True,
                "path": Path(file_path).as_posix(),
                "size": len(content),
                "backup_available": backup_available,
            }
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"Failed to write file: {e}"}

    # --------------------------- Git history ---------------------------- #
    async def get_git_history(
        self, file_path: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        try:
            cmd = [
                "git",
                "log",
                f"--max-count={int(limit)}",
                "--pretty=format:%H|%s|%an|%ct",
            ]
            if file_path:
                cmd.append(file_path)
            out = await arun(cmd, timeout=30)
            commits: list[dict[str, Any]] = []
            for line in out.splitlines():
                if not line.strip():
                    continue
                parts = line.split("|", 3)
                if len(parts) != 4:
                    continue
                h, msg, author, ts = parts
                try:
                    tsf = float(ts)
                except Exception:
                    tsf = 0.0
                commits.append(
                    {
                        "hash": h,
                        "message": msg,
                        "author": author,
                        "timestamp": tsf,
                    }
                )
            return {
                "commits": commits,
                "file_path": file_path,
                "count": len(commits),
            }
        except Exception as e:
            return {"error": f"Failed to get git history: {e}"}

    # --------------------------- Code search ---------------------------- #
    async def search_code(
        self, query: str, file_pattern: str = "**/*.py", context_lines: int = 3
    ) -> dict[str, Any]:
        try:
            results: list[dict[str, Any]] = []
            max_hits = 200
            for p in self.repo_root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.resolve().relative_to(self.repo_root)
                rel_posix = rel.as_posix()
                if not fnmatch.fnmatch(rel_posix, file_pattern):
                    continue
                try:
                    if p.stat().st_size > self.max_file_size:
                        continue
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    lines = text.splitlines()
                    lower_q = query.lower()
                    for i, line in enumerate(lines):
                        if lower_q in line.lower():
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            results.append(
                                {
                                    "file": rel_posix,
                                    "line_number": i + 1,
                                    "match": line.strip(),
                                    "context": lines[start:end],
                                }
                            )
                            if len(results) >= max_hits:
                                break
                except Exception:
                    continue
            return {
                "results": results[:max_hits],
                "query": query,
                "count": len(results),
            }
        except Exception as e:
            return {"error": f"Failed to search code: {e}"}

    # --------------------------- Helpers -------------------------------- #
    def _detect_language(self, file_path: Path) -> str | None:
        ext = file_path.suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }.get(ext)
