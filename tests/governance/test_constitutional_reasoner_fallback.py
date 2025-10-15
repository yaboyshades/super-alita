from pathlib import Path

import pytest

from src.governance import constitutional_reasoner as cr


@pytest.mark.asyncio()
async def test_reasoner_uses_fallback_encoder_when_dependency_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SUPER_ALITA_USE_SENTENCE_TRANSFORMERS", raising=False)
    monkeypatch.setattr(cr, "SentenceTransformer", None, raising=False)
    monkeypatch.setattr(cr, "_SENTENCE_TRANSFORMERS_ERROR", ImportError("missing"), raising=False)

    constitution = tmp_path / "constitution.md"
    constitution.write_text("# Constitution\n## Safety\nRemain safe.\n", encoding="utf-8")

    reasoner = cr.ConstitutionalReasoner(constitution_path=str(constitution))

    assert isinstance(reasoner.embedding_model, cr.SimpleSentenceEncoder)

    approved, reasoning = await reasoner.evaluate_action(
        {"description": "Share transparent project update"},
        {"user_intent": "Inform stakeholders"},
    )

    assert isinstance(approved, bool)
    assert isinstance(reasoning, str)
    assert reasoning
