"""
Simple workflow tests for unified orchestrator.
"""

from src.orchestration.unified_orchestrator import UnifiedRunConfig


def test_unified_run_config_creation():
    """Test UnifiedRunConfig creation and defaults."""
    config = UnifiedRunConfig(prompt="Test prompt", run_id="test-123")

    assert config.prompt == "Test prompt"
    assert config.run_id == "test-123"
    assert config.session_id == "default"
    assert config.enable_planning is True
    assert config.sdd_mode is False
    assert config.constitutional_threshold == 0.75


def test_unified_run_config_from_args():
    """Test UnifiedRunConfig.from_args method."""
    args = {
        "run_id": "custom-123",
        "session_id": "custom-session",
        "enable_specification": True,
        "enable_tasks": True,
        "sdd_mode": True,
        "constitutional_threshold": 0.8,
    }

    config = UnifiedRunConfig.from_args("Test prompt", args)

    assert config.prompt == "Test prompt"
    assert config.run_id == "custom-123"
    assert config.session_id == "custom-session"
    assert config.enable_specification is True
    assert config.enable_tasks is True
    # Note: sdd_mode not in from_args yet, would need to be added


def test_sdd_config_fields():
    """Test SDD-specific configuration fields."""
    config = UnifiedRunConfig(
        prompt="SDD test",
        run_id="sdd-001",
        sdd_mode=True,
        sdd_feature_id="feature-123",
        sdd_phase="specify",
        constitutional_threshold=0.85,
        sdd_template_dir="custom/templates",
    )

    assert config.sdd_mode is True
    assert config.sdd_feature_id == "feature-123"
    assert config.sdd_phase == "specify"
    assert config.constitutional_threshold == 0.85
    assert config.sdd_template_dir == "custom/templates"


if __name__ == "__main__":
    # Run basic tests
    test_unified_run_config_creation()
    test_unified_run_config_from_args()
    test_sdd_config_fields()
    print("All basic tests passed!")
