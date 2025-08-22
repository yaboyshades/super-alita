from __future__ import annotations

from difflib import unified_diff


def rewrite_function_to_result(
    src: str, function_name: str
) -> tuple[str | None, str | None]:
    # TODO: Replace with a real libcst transform.
    if f"def {function_name}(" not in src:
        return None, "Function not found"
    new_src = src  # placeholder; no-op transform for demo
    diff = "".join(
        unified_diff(
            src.splitlines(keepends=True),
            new_src.splitlines(keepends=True),
            fromfile="a.py",
            tofile="b.py",
        )
    )
    return new_src, diff
