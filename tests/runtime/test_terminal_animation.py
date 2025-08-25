"""Tests for terminal animation utilities."""

# The animation helpers should provide deterministic frames so they can
# be verified without rendering in a real terminal.

from io import StringIO

from src.utils.terminal_animation import animate_prompt, build_spinner_frames


def test_build_spinner_frames_dynamic():
    frames = build_spinner_frames("Hi")
    assert frames == ["- H", "\\ Hi"]


def test_animate_prompt_writes_frames():
    buf = StringIO()
    animate_prompt("OK", delay=0, stream=buf)
    output = buf.getvalue()
    assert output.startswith("\r- O")
    assert "\r\\ OK" in output
    assert output.endswith("\n")
