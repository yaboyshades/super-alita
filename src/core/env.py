"""Environment loading utilities.

Ensures a single, safe load of the repository's .env file so abilities and
scripts consistently see environment variables (e.g., GITHUB_TOKEN).

Design:
- Prefer loading the .env at the repository root (three parents up from this
  module: src/core/env.py -> core -> src -> repo root).
- Do nothing if python-dotenv isn't installed (graceful no-op).
- Avoid re-loading to prevent overriding environment variables set by the OS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_ENV_LOADED: bool = False


def ensure_env_loaded(*, silent: bool = True) -> None:
    """Load the .env file from the repo root once.

    Parameters:
        silent: When False, prints a brief message about load status.

    Behavior:
        - If python-dotenv is not available, returns quietly.
        - If already loaded, returns without reloading.
        - Does not override existing environment variables.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        # python-dotenv is optional at runtime; just no-op if missing
        if not silent:
            print("? dotenv not available; skipping .env load")
        _ENV_LOADED = True
        return

    # Compute repo root relative to this file: src/core/env.py -> repo root
    this_file: Final[Path] = Path(__file__).resolve()
    repo_root: Final[Path] = this_file.parents[2]
    env_path = repo_root / ".env"

    # Only attempt to load if the file exists; do not override existing vars
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)
        if not silent:
            print(f"? Loaded .env from {env_path}")
    else:
        if not silent:
            print(f"? No .env found at {env_path}; skipping load")

    _ENV_LOADED = True
