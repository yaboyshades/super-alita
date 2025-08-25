"""Small helpers for dynamic terminal loading animations."""

from __future__ import annotations

import sys
import time
from typing import List, TextIO

SPINNER_FRAMES = ["-", "\\", "|", "/"]

def build_spinner_frames(prompt: str) -> List[str]:
    """Build frames for a loading animation with dynamic prompt text.

    Each frame shows a spinner character followed by the growing prefix
    of the provided prompt. This allows callers to preview what is being
    "made" as the animation progresses.
    """
    frames: List[str] = []
    for i in range(1, len(prompt) + 1):
        spinner = SPINNER_FRAMES[(i - 1) % len(SPINNER_FRAMES)]
        frames.append(f"{spinner} {prompt[:i]}")
    return frames

def animate_prompt(prompt: str, delay: float = 0.1, stream: TextIO | None = None) -> None:
    """Render a loading animation for the given prompt.

    Parameters
    ----------
    prompt:
        The text to gradually reveal alongside the spinner.
    delay:
        Seconds to pause between frames. Use ``0`` in tests.
    stream:
        Optional text stream. Defaults to ``sys.stdout``.
    """
    stream = stream or sys.stdout
    for frame in build_spinner_frames(prompt):
        stream.write(f"\r{frame}")
        stream.flush()
        time.sleep(delay)
    stream.write("\n")
    stream.flush()
