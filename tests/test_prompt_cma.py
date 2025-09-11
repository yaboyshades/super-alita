from __future__ import annotations

from src.core.prompt_manager import PromptManager


def test_cma_system_prompt_loads():
    pm = PromptManager(prompts_file="src/config/prompts.json")
    text = pm.get_system_prompt("cma")
    assert isinstance(text, str) and len(text) > 0
    assert "Constitutional Mastery Architect" in text

