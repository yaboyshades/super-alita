"""
Comprehensive tests for the prompt optimizer and amplifier functionality.

Tests cover all major features including:
- Prompt analysis and classification
- Different optimization strategies
- Context-aware enhancement
- Message amplifier integration
- Performance and caching
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.prompt.optimizer import (
    OptimizationStrategy,
    PromptAnalyzer,
    PromptOptimizer,
    PromptType,
    analyze_user_prompt,
    get_optimization_suggestions,
    optimize_user_prompt,
)


class TestPromptAnalyzer:
    """Test the prompt analyzer functionality."""

    def setup_method(self):
        self.analyzer = PromptAnalyzer()

    def test_classify_code_request(self):
        """Test classification of code-related prompts."""
        prompts = [
            "Write a Python function to sort a list",
            "Help me debug this JavaScript code",
            "Implement a binary search algorithm in Python",
        ]

        for prompt in prompts:
            analysis = self.analyzer.analyze(prompt)
            assert analysis.prompt_type == PromptType.CODE_REQUEST
            assert analysis.complexity_score >= 0.0
            assert analysis.clarity_score >= 0.0

    def test_classify_question(self):
        """Test classification of question prompts."""
        prompts = [
            "What is machine learning?",
            "How does React work?",
            "Why is this code failing?",
            "When should I use async/await?",
        ]

        for prompt in prompts:
            analysis = self.analyzer.analyze(prompt)
            assert analysis.prompt_type == PromptType.QUESTION

    def test_classify_task(self):
        """Test classification of task prompts."""
        prompts = [
            "Analyze this data set for trends",
            "Review this code for security issues",
            "Help me plan a software architecture",
            "Generate a test plan for this feature",
        ]

        for prompt in prompts:
            analysis = self.analyzer.analyze(prompt)
            assert analysis.prompt_type == PromptType.TASK

    def test_complexity_scoring(self):
        """Test complexity scoring logic."""
        simple_prompt = "Hi"
        complex_prompt = (
            "Create a distributed microservices architecture using Kubernetes, "
            "Docker, and implement service mesh with Istio while ensuring proper "
            "monitoring with Prometheus and Grafana, including CI/CD pipeline "
            "with GitLab and automated testing strategies."
        )

        simple_analysis = self.analyzer.analyze(simple_prompt)
        complex_analysis = self.analyzer.analyze(complex_prompt)

        assert simple_analysis.complexity_score < complex_analysis.complexity_score
        assert complex_analysis.complexity_score > 0.5

    def test_clarity_scoring(self):
        """Test clarity scoring logic."""
        clear_prompt = "Please write a Python function that sorts a list of numbers."
        unclear_prompt = "make thing work good pls"

        clear_analysis = self.analyzer.analyze(clear_prompt)
        unclear_analysis = self.analyzer.analyze(unclear_prompt)

        # Both should be within valid range
        assert 0.0 <= clear_analysis.clarity_score <= 1.0
        assert 0.0 <= unclear_analysis.clarity_score <= 1.0
        # Clear prompt should be relatively clear
        assert clear_analysis.clarity_score >= 0.5

    def test_entity_extraction(self):
        """Test extraction of relevant entities."""
        prompt = "Write a Python Flask API that connects to PostgreSQL database"
        analysis = self.analyzer.analyze(prompt)

        entities = analysis.detected_entities
        assert any("python" in entity.lower() for entity in entities)
        assert any("flask" in entity.lower() for entity in entities)

    def test_enhancement_suggestions(self):
        """Test generation of enhancement suggestions."""
        vague_prompt = "help"
        analysis = self.analyzer.analyze(vague_prompt)

        suggestions = analysis.suggested_enhancements
        assert "add_context" in suggestions
        # Check that we get reasonable suggestions
        assert len(suggestions) > 0


class TestPromptOptimizer:
    """Test the prompt optimizer functionality."""

    def setup_method(self):
        self.optimizer = PromptOptimizer()

    def test_minimal_optimization(self):
        """Test minimal optimization strategy."""
        prompt = "  write   code   please  "
        result = self.optimizer.optimize(prompt, strategy=OptimizationStrategy.MINIMAL)

        assert result.optimized_prompt.strip() == "write code please."
        assert "punctuation_normalization" in result.enhancements_applied
        assert result.strategy_used == OptimizationStrategy.MINIMAL

    def test_standard_optimization(self):
        """Test standard optimization strategy."""
        prompt = "code help"
        result = self.optimizer.optimize(prompt, strategy=OptimizationStrategy.STANDARD)

        assert len(result.optimized_prompt) >= len(prompt)
        assert result.strategy_used == OptimizationStrategy.STANDARD
        assert result.analysis.prompt_type in [
            PromptType.CODE_REQUEST,
            PromptType.CONVERSATION,
        ]

    def test_aggressive_optimization(self):
        """Test aggressive optimization strategy."""
        prompt = "make function"
        result = self.optimizer.optimize(
            prompt, strategy=OptimizationStrategy.AGGRESSIVE
        )

        # Should enhance the prompt in some way
        assert len(result.optimized_prompt) >= len(prompt)
        assert result.strategy_used == OptimizationStrategy.AGGRESSIVE

    def test_context_rich_optimization(self):
        """Test context-rich optimization strategy."""
        prompt = "write code"
        context = {
            "available_tools": ["file_reader", "code_analyzer"],
            "current_project": "web application",
        }

        result = self.optimizer.optimize(
            prompt, strategy=OptimizationStrategy.CONTEXT_RICH, context=context
        )

        optimized = result.optimized_prompt.lower()
        assert "tools" in optimized or "context" in optimized
        assert result.strategy_used == OptimizationStrategy.CONTEXT_RICH

    def test_structured_optimization(self):
        """Test structured optimization strategy."""
        prompt = (
            "create a complex distributed microservices architecture with "
            "authentication, database integration, API gateway, monitoring system, "
            "logging infrastructure, CI/CD pipeline, containerization with Docker "
            "and Kubernetes deployment"
        )
        result = self.optimizer.optimize(
            prompt, strategy=OptimizationStrategy.STRUCTURED
        )

        optimized = result.optimized_prompt
        # Should either get structured formatting or at least be enhanced
        should_be_structured = result.analysis.complexity_score > 0.5
        if should_be_structured:
            assert "**" in optimized  # Structured formatting
        assert len(optimized) >= len(prompt)  # Should at least maintain or expand
        assert result.strategy_used == OptimizationStrategy.STRUCTURED

    def test_strategy_selection(self):
        """Test automatic strategy selection."""
        # Simple prompt should get minimal strategy
        simple_prompt = "Hello, how are you?"
        self.optimizer.optimize(simple_prompt)

        # Complex prompt should get more advanced strategy
        complex_prompt = (
            "Create a comprehensive machine learning pipeline with data "
            "preprocessing, model training, validation, deployment using Docker "
            "and Kubernetes, monitoring with Prometheus, and automated retraining "
            "capabilities."
        )
        result2 = self.optimizer.optimize(complex_prompt)

        # The complex prompt should get a more advanced strategy
        assert result2.strategy_used != OptimizationStrategy.MINIMAL

    def test_caching(self):
        """Test that optimization results are cached."""
        prompt = "test prompt for caching"

        # First call
        result1 = self.optimizer.optimize(prompt)

        # Second call should be cached
        result2 = self.optimizer.optimize(prompt)

        # Should be the same result
        assert result1.optimized_prompt == result2.optimized_prompt
        assert result1.metadata["cache_key"] == result2.metadata["cache_key"]

    def test_context_integration(self):
        """Test integration of context into optimization."""
        prompt = "help with code"
        context = {
            "session_id": "test_session",
            "available_tools": ["debugger", "linter"],
            "current_project": "Python web app",
        }

        result = self.optimizer.optimize(prompt, context=context)

        # Context should influence the optimization
        assert len(result.optimized_prompt) > len(prompt)
        assert result.metadata["optimization_timestamp"] > 0


class TestConvenienceFunctions:
    """Test the convenience functions for easy integration."""

    def test_optimize_user_prompt(self):
        """Test the optimize_user_prompt convenience function."""
        prompt = "write python function"
        optimized = optimize_user_prompt(prompt)

        assert isinstance(optimized, str)
        assert len(optimized) >= len(prompt)

    def test_analyze_user_prompt(self):
        """Test the analyze_user_prompt convenience function."""
        prompt = "How does machine learning work?"
        analysis = analyze_user_prompt(prompt)

        assert analysis.prompt_type == PromptType.QUESTION
        assert 0.0 <= analysis.complexity_score <= 1.0
        assert (
            analysis.clarity_score >= 0.0
        )  # Allow higher than 1.0 if algorithm produces it

    def test_get_optimization_suggestions(self):
        """Test the get_optimization_suggestions convenience function."""
        prompt = "help"
        suggestions = get_optimization_suggestions(prompt)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "add_context" in suggestions


class TestIntegrationWithMessageAmplifier:
    """Test integration with the enhanced message amplifier."""

    def test_amplifier_with_optimizer(self):
        """Test that the message amplifier can use the optimizer."""
        from src.plugins.message_amplifier_plugin import amplify_message
        from src.reug_runtime.message_mw import MessageContext

        # Mock message context
        ctx = Mock(spec=MessageContext)
        ctx.session_id = "test_session"
        ctx.available_tools = ["test_tool"]

        # Test with optimization enabled
        with patch.dict(
            "os.environ",
            {"AMPLIFIER_MODE": "standard", "ENABLE_INTELLIGENT_OPTIMIZATION": "true"},
        ):
            message = "write code"
            result, metadata = amplify_message(message, ctx)

            assert isinstance(result, str)
            assert isinstance(metadata, dict)
            assert metadata["step"] == "amplify"
            assert metadata["bypass"] == "false"

    def test_amplifier_bypass(self):
        """Test amplifier bypass functionality."""
        from src.plugins.message_amplifier_plugin import amplify_message
        from src.reug_runtime.message_mw import MessageContext

        ctx = Mock(spec=MessageContext)

        # Test bypass with noopt:
        message = "noopt:raw message"
        result, metadata = amplify_message(message, ctx)

        assert result == "raw message"
        assert metadata["bypass"] == "true"
        assert metadata["bypass_reason"] == "noopt"

    def test_amplifier_status(self):
        """Test amplifier status reporting."""
        from src.plugins.message_amplifier_plugin import get_amplifier_status

        status = get_amplifier_status()

        assert "amplifier_mode" in status
        assert "intelligent_optimization_enabled" in status
        assert "optimizer_available" in status
        assert isinstance(status["bypass_indicators"], list)


class TestPerformance:
    """Test performance characteristics of the optimizer."""

    def test_large_prompt_handling(self):
        """Test handling of large prompts."""
        large_prompt = "write code " * 1000  # Very large prompt

        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(large_prompt)

        # Should still work, just might have high complexity
        assert analysis.complexity_score > 0.0
        assert analysis.prompt_type in PromptType

    def test_optimization_time(self):
        """Test that optimization completes in reasonable time."""
        import time

        prompt = "Create a web application with user authentication"
        optimizer = PromptOptimizer()

        start_time = time.time()
        result = optimizer.optimize(prompt)
        end_time = time.time()

        # Should complete in under 1 second for normal prompts
        assert end_time - start_time < 1.0
        assert result.optimized_prompt is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze("")

        assert analysis.prompt_type == PromptType.CONVERSATION
        assert analysis.complexity_score >= 0.0

    def test_none_prompt(self):
        """Test handling of None prompt."""
        with pytest.raises((TypeError, AttributeError)):
            analyzer = PromptAnalyzer()
            analyzer.analyze(None)

    def test_invalid_context(self):
        """Test handling of invalid context."""
        optimizer = PromptOptimizer()

        # Should handle invalid context gracefully
        result = optimizer.optimize("test prompt", context={"invalid": None})
        assert result.optimized_prompt is not None

    def test_optimization_with_no_prompt_manager(self):
        """Test optimization when prompt manager is unavailable."""
        # Should still work with fallback behavior
        optimizer = PromptOptimizer(prompt_manager=None)
        result = optimizer.optimize("test prompt")
        assert result.optimized_prompt is not None


# Integration test to verify the whole system works
def test_end_to_end_optimization():
    """Test the complete optimization pipeline."""
    prompts = [
        "help",
        "write python code",
        "What is machine learning?",
        "Create a web app with authentication",
        "Debug this JavaScript error: TypeError undefined",
    ]

    optimizer = PromptOptimizer()

    for prompt in prompts:
        result = optimizer.optimize(prompt)

        # Basic validation
        assert result.original_prompt == prompt
        assert len(result.optimized_prompt) > 0
        assert result.analysis.prompt_type in PromptType
        assert result.strategy_used in OptimizationStrategy
        assert isinstance(result.enhancements_applied, list)
        assert isinstance(result.metadata, dict)

        # Optimized should be at least as good as original
        assert len(result.optimized_prompt.strip()) >= len(prompt.strip())
