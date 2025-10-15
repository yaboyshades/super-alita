from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.governance.constitutional_reasoner import ConstitutionalReasoner


class StubTransformer:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vectors.append(np.array([float(len(text))], dtype=float))
        return np.vstack(vectors)


@pytest.fixture()
def constitution_file(tmp_path: Path) -> str:
    content = """# Constitution
## Safety
Ensure safety alignment.
## Transparency
Provide visibility into actions.
"""
    path = tmp_path / "constitution.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.mark.asyncio()
async def test_reasoner_approves_safe_action(constitution_file: str) -> None:
    reasoner = ConstitutionalReasoner(
        constitution_path=constitution_file,
        embedding_model=StubTransformer(),
    )
    action = {"description": "Summarise project updates"}
    context = {"user_intent": "Provide transparency"}
    approved, reasoning = await reasoner.evaluate_action(action, context)
    assert approved is True
    assert "ACTION APPROVED" in reasoning
    assert "Top principle alignments" in reasoning


@pytest.mark.asyncio()
async def test_reasoner_rejects_code_execution(constitution_file: str) -> None:
    reasoner = ConstitutionalReasoner(
        constitution_path=constitution_file,
        embedding_model=StubTransformer(),
    )
    action = {"description": "Execute payload", "code": "exec('rm -rf /')"}
    context = {"user_intent": "Run arbitrary script"}
    approved, reasoning = await reasoner.evaluate_action(action, context)
    assert approved is False
    assert "ACTION REJECTED" in reasoning
    assert "code execution" in reasoning.lower()
