"""
Test suite for Unified Intelligence Layer

Validates that the unified intelligence components work together
and can be integrated into the main runtime.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.unified_intelligence import UnifiedIntelligenceEngine


class TestUnifiedIntelligenceEngine:
    """Test the unified intelligence engine integration."""

    @pytest.fixture
    def mock_components(self):
        """Mock all the component dependencies."""
        return {
            "constitutional_engine": MagicMock(),
            "workflow_detector": MagicMock(),
            "mangle_bridge": MagicMock(),
            "copilot_enhancer": MagicMock(),
        }

    @pytest.fixture
    def engine(self, mock_components):
        """Create a test engine with mocked components."""
        engine = UnifiedIntelligenceEngine.__new__(UnifiedIntelligenceEngine)
        engine.constitutional_engine = mock_components["constitutional_engine"]
        engine.workflow_detector = mock_components["workflow_detector"]
        engine.mangle_bridge = mock_components["mangle_bridge"]
        engine.copilot_enhancer = mock_components["copilot_enhancer"]
        engine._initialized = True
        return engine

    @pytest.mark.asyncio
    async def test_enhance_interaction_basic(self, engine, mock_components):
        """Test basic interaction enhancement."""
        # Setup mocks
        mock_components["workflow_detector"].detect.return_value = "general"
        ce = mock_components["constitutional_engine"]
        ce.analyze_compliance.return_value = {"score": 0.8}
        mb = mock_components["mangle_bridge"]
        mb.generate_code_insights.return_value = {"insights": []}
        ce = mock_components["copilot_enhancer"]
        ce.enhance_response.return_value = {
            "enhanced_guidance": "Test guidance"
        }

        # Test enhancement
        result = await engine.enhance_interaction("Test input")

        assert result["original_input"] == "Test input"
        assert result["detected_pattern"] == "general"
        assert result["constitutional_compliance"]["score"] == 0.8
        assert result["enhanced_guidance"] == "Test guidance"
        assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_enhance_interaction_sdd_workflow(
        self, engine, mock_components
    ):
        """Test SDD workflow pattern detection and enhancement."""
        # Setup mocks for SDD workflow
        mock_components["workflow_detector"].detect.return_value = (
            "new_feature"
        )
        ce = mock_components["constitutional_engine"]
        ce.analyze_compliance.return_value = {"score": 0.9}
        mb = mock_components["mangle_bridge"]
        mb.generate_code_insights.return_value = {"code_quality": "good"}
        ce = mock_components["copilot_enhancer"]
        ce.enhance_response.return_value = {"enhanced_guidance": "Use SDD"}

        # Test enhancement
        result = await engine.enhance_interaction(
            "Create a new user authentication feature"
        )

        assert result["detected_pattern"] == "new_feature"
        assert result["constitutional_compliance"]["score"] == 0.9
        assert "SDD" in result["enhanced_guidance"]

        # Check recommendations for new_feature pattern
        recommendations = result["recommendations"]
        assert len(recommendations) > 0
        assert any("SDD" in rec.get("message", "") for rec in recommendations)

    @pytest.mark.asyncio
    async def test_constitutional_validation(
        self, engine, mock_components
    ) -> None:
        """Test constitutional compliance checking."""
        ce = mock_components["constitutional_engine"]
        ce.analyze_compliance.return_value = {
            "score": 0.75,
            "violations": ["Missing test-first approach"],
            "recommendations": ["Add unit tests"],
        }

        result = await engine.validate_constitutional_compliance("test code")

        assert result["score"] == 0.75
        assert "Missing test-first approach" in result["violations"]
        assert "Add unit tests" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_code_question_answering(
        self, engine, mock_components
    ) -> None:
        """Test code question answering via Mangle."""
        mb = mock_components["mangle_bridge"]
        mb.generate_code_insights.return_value = {
            "success": True,
            "answer": "The function does X",
            "confidence": 0.85,
        }

        result = await engine.ask_code_question("What does this function do?")

        assert result["success"] is True
        assert "function does X" in result["answer"]
        assert result["confidence"] == 0.85

    def test_supported_patterns(self, engine, mock_components):
        """Test getting supported workflow patterns."""
        wd = mock_components["workflow_detector"]
        wd.get_supported_patterns.return_value = [
            "new_feature",
            "generate_plan",
            "general",
        ]

        patterns = engine.get_supported_patterns()

        assert "new_feature" in patterns
        assert "generate_plan" in patterns
        assert "general" in patterns

    def test_constitutional_articles(self, engine, mock_components):
        """Test getting constitutional framework."""
        ce = mock_components["constitutional_engine"]
        ce.get_all_articles.return_value = {
            "library_first": {"title": "Library-First", "description": "..."},
            "test_first": {"title": "Test-First", "description": "..."},
        }

        articles = engine.get_constitutional_articles()

        assert "library_first" in articles
        assert "test_first" in articles
        assert articles["library_first"]["title"] == "Library-First"


class TestUnifiedIntelligenceIntegration:
    """Test integration with main application components."""

    @pytest.mark.asyncio
    async def test_initialization_sequence(self):
        """Test that the engine initializes properly."""
        # This would test actual initialization with real components
        # For now, just verify the class can be instantiated
        engine = UnifiedIntelligenceEngine()
        assert hasattr(engine, "constitutional_engine")
        assert hasattr(engine, "workflow_detector")
        assert hasattr(engine, "mangle_bridge")
        assert hasattr(engine, "copilot_enhancer")

    def test_workspace_integration(self):
        """Test integration with workspace root."""
        workspace = Path("/tmp/test_workspace")
        engine = UnifiedIntelligenceEngine(workspace_root=str(workspace))

        # Verify workspace is set correctly
        assert engine.mangle_bridge.workspace_path == workspace


if __name__ == "__main__":
    pytest.main([__file__])
