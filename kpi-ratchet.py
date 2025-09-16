#!/usr/bin/env python3
"""Generate the deep dive report from the static template."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "docs" / "deep-dive" / "REPORT.tpl.md"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "deep-dive" / "REPORT.md"


@dataclass(frozen=True)
class ReportContext:
    """Values that are interpolated into the report template."""

    date: str
    commit_sha: str
    file_count: int
    loc_count: int

    @property
    def repo_size_label(self) -> str:
        """Display representation for the "Repo Size" placeholder."""

        return f"{self.file_count} files / {self.loc_count} LOC"


def load_template(path: Path) -> str:
    """Load the template contents from ``path``."""

    return path.read_text(encoding="utf-8")


def write_report(path: Path, content: str) -> None:
    """Persist ``content`` to ``path`` using UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_git_command(args: Iterable[str], repo_root: Path) -> str | None:
    """Execute a git command relative to ``repo_root`` and return trimmed output."""

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def _tracked_files(repo_root: Path) -> list[str]:
    """Return the list of tracked files for ``repo_root``."""

    output = _run_git_command(["ls-files"], repo_root)
    return [] if output is None else [line for line in output.splitlines() if line]


def _count_lines(repo_root: Path, files: Iterable[str]) -> int:
    """Count the total number of lines contained in ``files`` relative to ``repo_root``."""

    total = 0
    for relative_path in files:
        path = repo_root / relative_path
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                total += sum(1 for _ in handle)
        except OSError:
            # Skip files that cannot be opened (e.g., removed or permission issues).
            continue
    return total


def build_context(repo_root: Path) -> ReportContext:
    """Construct the context injected into the template."""

    date = datetime.now(timezone.utc).date().isoformat()
    commit_sha = _run_git_command(["rev-parse", "--short", "HEAD"], repo_root) or "unknown"
    tracked = _tracked_files(repo_root)
    file_count = len(tracked)
    loc_count = _count_lines(repo_root, tracked)
    return ReportContext(date=date, commit_sha=commit_sha, file_count=file_count, loc_count=loc_count)


def fill_template(template: str, context: ReportContext) -> str:
    """Replace placeholder tokens in ``template`` using ``context`` values."""

    replacements = {
        "<fill>": context.date,
        "<sha>": context.commit_sha,
        "<files/LOC>": context.repo_size_label,
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def generate_report(
    *,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    repo_root: Path | None = None,
) -> Path:
    """Generate the report using ``template_path`` and write it to ``output_path``."""

    resolved_repo_root = repo_root or REPO_ROOT
    template = load_template(template_path)
    context = build_context(resolved_repo_root)
    rendered = fill_template(template, context)
    write_report(output_path, rendered)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the generator."""

    parser = argparse.ArgumentParser(description="Generate the deep dive KPI report.")
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="Path to the report template.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the rendered report should be written.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used for gathering metadata.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the CLI."""

    args = parse_args()
    generate_report(
        template_path=args.template,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
