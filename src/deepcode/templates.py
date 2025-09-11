"""DeepCode templates loader and simple renderer.

Provides access to reusable prompt/code templates stored under
`templates/deepcode/` in the repository.

This intentionally avoids adding external dependencies (e.g., Jinja2).
Rendering uses a minimal `.format_map` with safe defaults.
"""

from __future__ import annotations

import functools
from pathlib import Path


def _repo_root() -> Path:
    # src/deepcode/templates.py -> src -> ROOT
    return Path(__file__).resolve().parents[2]


def get_templates_dir() -> Path:
    """Return the directory containing DeepCode templates."""
    return _repo_root() / "templates" / "deepcode"


@functools.lru_cache(maxsize=1)
def list_deepcode_templates() -> list[str]:
    """List available DeepCode template basenames.

    Returns file basenames (e.g., 'code-review.md').
    """
    tdir = get_templates_dir()
    if not tdir.exists():
        return []
    names: list[str] = []
    for p in sorted(tdir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            names.append(p.name)
    return names


def load_deepcode_template(name: str) -> str:
    """Load a DeepCode template by filename.

    Args:
        name: Filename within `templates/deepcode` (e.g., 'code-review.md').
    """
    path = get_templates_dir() / name
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"DeepCode template not found: {path}")
    return path.read_text(encoding="utf-8")


class _SafeDict(dict):
    def __missing__(self, key):  # type: ignore[override]
        # Leave unknown placeholders intact
        return "{" + key + "}"


def render_deepcode_template(name: str, **params: object) -> str:
    """Render a template with simple `.format_map` substitution.

    Unknown placeholders are preserved as `{placeholder}`.
    """
    tpl = load_deepcode_template(name)
    return tpl.format_map(_SafeDict(params))


def render_inline(template_text: str, **params: object) -> str:
    """Render an inline template string using safe substitution."""
    return template_text.format_map(_SafeDict(params))


__all__ = [
    "get_templates_dir",
    "list_deepcode_templates",
    "load_deepcode_template",
    "render_deepcode_template",
    "render_inline",
]

