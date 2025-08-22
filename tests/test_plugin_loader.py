"""
Tests for plugin loader functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from src.core.plugin_loader import (
    PluginLoadError,
    PluginManifestError,
    discover_plugins,
    get_plugin_info,
    load_plugin_manifest,
    validate_dependencies,
    validate_plugin_order,
)


class TestPluginManifest:
    """Test plugin manifest loading and validation."""

    def test_load_valid_manifest(self):
        """Test loading a valid plugin manifest."""
        manifest_data = {
            "version": 1,
            "plugins": [
                {
                    "name": "test_plugin",
                    "module": "test.module:TestClass",
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(manifest_data, f)
            temp_path = f.name

        try:
            plugins = load_plugin_manifest(temp_path)
            assert len(plugins) == 1
            assert plugins[0]["name"] == "test_plugin"
            assert plugins[0]["module"] == "test.module:TestClass"
        finally:
            Path(temp_path).unlink()

    def test_load_missing_file(self):
        """Test loading non-existent manifest file."""
        with pytest.raises(PluginManifestError, match="Plugin manifest not found"):
            load_plugin_manifest("nonexistent.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(PluginManifestError, match="Failed to parse YAML"):
                load_plugin_manifest(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_missing_plugins_key(self):
        """Test loading manifest without plugins key."""
        manifest_data = {"version": 1}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(manifest_data, f)
            temp_path = f.name

        try:
            with pytest.raises(PluginManifestError, match="missing 'plugins' key"):
                load_plugin_manifest(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_plugin_format(self):
        """Test loading manifest with invalid plugin format."""
        manifest_data = {"version": 1, "plugins": ["not_a_dict"]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(manifest_data, f)
            temp_path = f.name

        try:
            with pytest.raises(
                PluginManifestError, match="Plugin 0 is not a dictionary"
            ):
                load_plugin_manifest(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_missing_required_fields(self):
        """Test loading manifest with missing required fields."""
        manifest_data = {
            "version": 1,
            "plugins": [{"name": "test"}],  # Missing module
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(manifest_data, f)
            temp_path = f.name

        try:
            with pytest.raises(
                PluginManifestError, match="missing required field: module"
            ):
                load_plugin_manifest(temp_path)
        finally:
            Path(temp_path).unlink()


class TestDependencyValidation:
    """Test plugin dependency validation."""

    def test_validate_no_dependencies(self):
        """Test validation with no dependencies."""
        plugins = [
            {"name": "plugin1", "module": "test:Test1"},
            {"name": "plugin2", "module": "test:Test2"},
        ]

        missing = validate_dependencies(plugins)
        assert len(missing) == 0

    def test_validate_satisfied_dependencies(self):
        """Test validation with satisfied dependencies."""
        plugins = [
            {"name": "plugin1", "module": "test:Test1"},
            {"name": "plugin2", "module": "test:Test2", "depends_on": ["plugin1"]},
        ]

        missing = validate_dependencies(plugins)
        assert len(missing) == 0

    def test_validate_missing_dependencies(self):
        """Test validation with missing dependencies."""
        plugins = [
            {
                "name": "plugin1",
                "module": "test:Test1",
                "depends_on": ["missing_plugin"],
            }
        ]

        missing = validate_dependencies(plugins)
        assert "missing_plugin" in missing

    def test_validate_invalid_dependency_format(self):
        """Test validation with invalid dependency format."""
        plugins = [
            {"name": "plugin1", "module": "test:Test1", "depends_on": "not_a_list"}
        ]

        missing = validate_dependencies(plugins)
        assert len(missing) == 0  # Invalid format is ignored


class TestPluginDiscovery:
    """Test plugin discovery and loading."""

    def test_discover_enabled_plugins_only(self):
        """Test that only enabled plugins are discovered."""
        plugins = [
            {"name": "enabled", "module": "test:MockPlugin", "enabled": True},
            {"name": "disabled", "module": "test:MockPlugin", "enabled": False},
        ]

        with patch("src.core.plugin_loader.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_plugin_class = MagicMock()
            mock_module.MockPlugin = mock_plugin_class
            mock_import.return_value = mock_module

            result = discover_plugins(plugins)

            assert len(result) == 1
            assert result[0][0] == "enabled"

    def test_discover_priority_ordering(self):
        """Test that plugins are ordered by priority."""
        plugins = [
            {"name": "low", "module": "test:MockPlugin", "priority": 100},
            {"name": "high", "module": "test:MockPlugin", "priority": 1},
            {"name": "medium", "module": "test:MockPlugin", "priority": 50},
        ]

        with patch("src.core.plugin_loader.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_plugin_class = MagicMock()
            mock_module.MockPlugin = mock_plugin_class
            mock_import.return_value = mock_module

            result = discover_plugins(plugins)

            # Should be ordered by priority: high(1), medium(50), low(100)
            assert [name for name, _ in result] == ["high", "medium", "low"]

    def test_discover_invalid_module_spec(self):
        """Test discovery with invalid module specification."""
        plugins = [{"name": "invalid", "module": "no_colon_separator"}]

        with pytest.raises(PluginLoadError, match="must be 'module:class'"):
            discover_plugins(plugins)

    def test_discover_import_error(self):
        """Test discovery with import error."""
        plugins = [{"name": "missing", "module": "nonexistent:Class"}]

        with patch("src.core.plugin_loader.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            with pytest.raises(PluginLoadError, match="Cannot import module"):
                discover_plugins(plugins)

    def test_discover_missing_class(self):
        """Test discovery with missing class in module."""
        plugins = [{"name": "missing_class", "module": "test:MissingClass"}]

        with patch("src.core.plugin_loader.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            del mock_module.MissingClass  # Ensure class doesn't exist
            mock_import.return_value = mock_module

            with pytest.raises(PluginLoadError, match="Class 'MissingClass' not found"):
                discover_plugins(plugins)


class TestPluginInfo:
    """Test plugin information extraction."""

    def test_get_plugin_info_complete(self):
        """Test extracting complete plugin information."""
        plugins = [
            {
                "name": "test_plugin",
                "module": "test:TestClass",
                "enabled": True,
                "priority": 10,
                "depends_on": ["dep1", "dep2"],
                "description": "Test plugin",
                "category": "test",
            }
        ]

        info = get_plugin_info(plugins)

        assert "test_plugin" in info
        plugin_info = info["test_plugin"]
        assert plugin_info["module"] == "test:TestClass"
        assert plugin_info["enabled"] is True
        assert plugin_info["priority"] == 10
        assert plugin_info["depends_on"] == ["dep1", "dep2"]
        assert plugin_info["description"] == "Test plugin"
        assert plugin_info["category"] == "test"

    def test_get_plugin_info_defaults(self):
        """Test extracting plugin information with defaults."""
        plugins = [{"name": "minimal_plugin", "module": "test:TestClass"}]

        info = get_plugin_info(plugins)

        plugin_info = info["minimal_plugin"]
        assert plugin_info["enabled"] is True  # Default
        assert plugin_info["priority"] == 100  # Default
        assert plugin_info["depends_on"] == []  # Default
        assert plugin_info["description"] == ""  # Default
        assert plugin_info["category"] == "uncategorized"  # Default


class TestPluginOrderValidation:
    """Test plugin order validation."""

    def test_validate_correct_order(self):
        """Test validation of correct dependency order."""
        plugins = [
            {"name": "base", "module": "test:Base", "priority": 10},
            {
                "name": "dependent",
                "module": "test:Dependent",
                "priority": 20,
                "depends_on": ["base"],
            },
        ]

        warnings = validate_plugin_order(plugins)
        assert len(warnings) == 0

    def test_validate_incorrect_order(self):
        """Test validation of incorrect dependency order."""
        plugins = [
            {
                "name": "dependent",
                "module": "test:Dependent",
                "priority": 10,
                "depends_on": ["base"],
            },
            {"name": "base", "module": "test:Base", "priority": 20},
        ]

        warnings = validate_plugin_order(plugins)
        assert len(warnings) == 1
        assert "loads later" in warnings[0]

    def test_validate_disabled_dependency(self):
        """Test validation with disabled dependency."""
        plugins = [
            {"name": "base", "module": "test:Base", "enabled": False},
            {"name": "dependent", "module": "test:Dependent", "depends_on": ["base"]},
        ]

        warnings = validate_plugin_order(plugins)
        assert len(warnings) == 1
        assert "loads later (or is disabled)" in warnings[0]
