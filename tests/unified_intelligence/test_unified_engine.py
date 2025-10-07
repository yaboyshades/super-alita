import asyncio

from src.unified_intelligence import UnifiedIntelligenceEngine


class FakeMangleBridge:
    """Test double emulating the asynchronous Mangle bridge contract."""

    def __init__(self):
        self.initialized = False
        self.questions_asked: list[str] = []

    async def initialize(self) -> None:  # pragma: no cover - simple state update
        self.initialized = True

    async def get_insights(self, user_input: str, pattern: str | None = None) -> dict:
        return {
            "available": True,
            "question": user_input,
            "pattern": pattern,
            "analysis_type": "test_double",
            "code_quality_issues": False,
            "quality_recommendations": [],
        }

    async def ask_question(self, question: str) -> dict:
        self.questions_asked.append(question)
        return {"available": True, "question": question, "answer": "stub"}


def test_enhance_interaction_with_fake_mangle() -> None:
    async def _run() -> None:
        engine = UnifiedIntelligenceEngine()
        engine.mangle_bridge = FakeMangleBridge()
        engine._initialized = False  # ensure we invoke fake initialization

        result = await engine.enhance_interaction(
            "Create a new feature plan for telemetry"
        )

        assert result["mangle_insights"]["available"] is True
        assert result["enhanced_guidance"]["enhancement"]["enhanced"] is True
        assert "overall_score" in result["constitutional_compliance"]
        assert result["recommendations"]  # expect at least one recommendation branch

    asyncio.run(_run())


def test_ask_code_question_uses_fake_bridge() -> None:
    async def _run() -> None:
        engine = UnifiedIntelligenceEngine()
        fake_bridge = FakeMangleBridge()
        engine.mangle_bridge = fake_bridge
        engine._initialized = False

        answer = await engine.ask_code_question(
            "What tests cover the workflow detector?"
        )

        assert answer["available"] is True
        assert fake_bridge.questions_asked == [
            "What tests cover the workflow detector?"
        ]

    asyncio.run(_run())
