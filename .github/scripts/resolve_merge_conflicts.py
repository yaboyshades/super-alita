"""Utility for CI to sanitize Git merge conflict markers.

The script scans tracked files for merge conflict markers and, when found,
removes the markers while keeping the "ours" section (the lines that appeared
before the `=======` divider). The goal is to produce a deterministic cleanup
that CI can surface back to the author as a patch artifact.

If conflict markers are resolved, a patch file is created so contributors can
apply the proposed cleanup locally. The script exits with status code 1 in this
case to ensure the workflow blocks the pull request until the author reviews
and commits the resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path
from typing import Iterable

MARKER_START = "<<<<<<<"
MARKER_MID = "======="
MARKER_END = ">>>>>>>"
PATCH_PATH = Path("merge-conflict-resolution.patch")
SUMMARY_PATH = Path("merge-conflict-summary.txt")


@dataclass
class ConflictResolution:
    """Result information for a processed file."""

    path: Path
    resolved_blocks: int


class ConflictMarkerError(RuntimeError):
    """Raised when merge conflict markers are unbalanced."""


def tracked_files() -> Iterable[Path]:
    """Yield paths tracked by Git relative to the repository root."""

    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    root = Path.cwd()
    for relative_path in result.stdout.splitlines():
        path = root / relative_path
        if path.is_file():
            yield path


def resolve_conflicts_in_file(path: Path) -> ConflictResolution | None:
    """Remove conflict markers from *path* keeping the top section.

    The algorithm preserves original newline characters by working with
    ``splitlines(keepends=True)``. If unbalanced markers are encountered, a
    :class:`ConflictMarkerError` is raised so CI can surface the issue.
    """

    original_text = path.read_text(encoding="utf-8", errors="ignore")
    if MARKER_START not in original_text:
        return None

    lines = original_text.splitlines(keepends=True)
    rewritten: list[str] = []
    i = 0
    resolved_blocks = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith(MARKER_START):
            resolved_blocks += 1
            i += 1
            top_segment: list[str] = []
            bottom_segment: list[str] = []
            collecting_top = True

            while i < len(lines):
                segment_line = lines[i]
                if segment_line.startswith(MARKER_MID):
                    collecting_top = False
                    i += 1
                    continue
                if segment_line.startswith(MARKER_END):
                    i += 1
                    break
                if collecting_top:
                    top_segment.append(segment_line)
                else:
                    bottom_segment.append(segment_line)
                i += 1
            else:  # pragma: no cover - defensive guard for malformed markers
                raise ConflictMarkerError(
                    f"Unterminated conflict in {path.as_posix()}"
                )

            rewritten.extend(top_segment)
            continue

        rewritten.append(line)
        i += 1

    new_text = "".join(rewritten)
    if new_text != original_text:
        path.write_text(new_text, encoding="utf-8")
        return ConflictResolution(path=path, resolved_blocks=resolved_blocks)

    return None


def write_patch_and_summary(resolutions: list[ConflictResolution]) -> None:
    """Persist a git patch and a human-readable summary for CI."""

    diff_result = subprocess.run(
        ["git", "diff"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    PATCH_PATH.write_text(diff_result.stdout, encoding="utf-8")

    summary_lines = [
        "Merge conflict markers were detected and sanitized.\n",
        "The CI workflow kept the 'ours' sections (above =======) for each conflict.\n",
        "\nFiles updated:\n",
    ]
    summary_lines.extend(
        f"- {resolution.path.as_posix()} (blocks resolved: {resolution.resolved_blocks})\n"
        for resolution in resolutions
    )
    summary_lines.append(
        "\nDownload the attached patch artifact and review the changes before committing.\n"
    )
    SUMMARY_PATH.write_text("".join(summary_lines), encoding="utf-8")


def main() -> int:
    resolutions: list[ConflictResolution] = []
    for path in tracked_files():
        resolution = resolve_conflicts_in_file(path)
        if resolution is not None:
            resolutions.append(resolution)

    if not resolutions:
        return 0

    write_patch_and_summary(resolutions)
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    print(summary_text)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConflictMarkerError as exc:  # pragma: no cover - surfaced in CI logs
        print(f"::error ::{exc}")
        sys.exit(2)
