"""
Tests for MCP capabilities tool functionality.
"""

from unittest.mock import MagicMock, patch

from src.mcp_local.capabilities_tool import (
    _collect_event_capabilities,
    _collect_plugin_capabilities,
    _collect_system_capabilities,
    _collect_tool_capabilities,
    collect_capabilities,
)


class TestCapabilitiesCollection:
    """Test main capabilities collection functionality."""

    def test_collect_capabilities_structure(self):
        """Test that collect_capabilities returns expected structure."""
        capabilities = collect_capabilities()

        # Check top-level structure
        expected_keys = ["timestamp", "version", "plugins", "events", "tools", "system"]
        for key in expected_keys:
            assert key in capabilities

        assert isinstance(capabilities["timestamp"], (int, float))
        assert capabilities["version"] == "1.0"
        assert isinstance(capabilities["plugins"], list)
        assert isinstance(capabilities["events"], dict)
        assert isinstance(capabilities["tools"], dict)
        assert isinstance(capabilities["system"], dict)


class TestPluginCapabilities:
    """Test plugin capability collection."""

    @patch("src.mcp_local.capabilities_tool.load_plugin_manifest")
    @patch("src.mcp_local.capabilities_tool.get_plugin_info")
    def test_collect_plugins_with_manifest(self, mock_get_info, mock_load_manifest):
        """Test collecting plugin capabilities with manifest available."""
        # Mock manifest data
        mock_load_manifest.return_value = [
            {"name": "test_plugin", "module": "test:TestPlugin"}
        ]

        mock_get_info.return_value = {
            "test_plugin": {
                "module": "test:TestPlugin",
                "enabled": True,
                "priority": 10,
                "depends_on": [],
                "description": "Test plugin",
                "category": "test",
            }
        }

        plugins = _collect_plugin_capabilities()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "test_plugin"
        assert plugins[0]["status"] == "configured"
        assert plugins[0]["enabled"] is True

    @patch("src.mcp_local.capabilities_tool.load_plugin_manifest")
    def test_collect_plugins_no_manifest(self, mock_load_manifest):
        """Test collecting plugin capabilities with no manifest."""
        mock_load_manifest.return_value = []

        plugins = _collect_plugin_capabilities()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "fallback_detection"
        assert plugins[0]["status"] == "detection_limited"

    def test_collect_plugins_import_error(self):
        """Test collecting plugin capabilities with import error."""
        with patch("src.mcp_local.capabilities_tool.load_plugin_manifest") as mock_load:
            mock_load.side_effect = ImportError("Module not found")

            plugins = _collect_plugin_capabilities()

            assert len(plugins) == 1
            assert plugins[0]["name"] == "fallback_detection"

    @patch("src.mcp_local.capabilities_tool.load_plugin_manifest")
    def test_collect_plugins_general_error(self, mock_load_manifest):
        """Test collecting plugin capabilities with general error."""
        mock_load_manifest.side_effect = Exception("General error")

        plugins = _collect_plugin_capabilities()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "error_state"
        assert plugins[0]["status"] == "error"
        assert "error" in plugins[0]


class TestEventCapabilities:
    """Test event capability collection."""

    @patch("src.mcp_local.capabilities_tool.list_events")
    def test_collect_events_with_registry(self, mock_list_events):
        """Test collecting event capabilities with registry available."""
        from src.core.event_types import EventDescriptor

        mock_events = {
            "test_event": EventDescriptor(
                name="test_event",
                version=1,
                schema={"required": ["field1"]},
                description="Test event",
            ),
            "deprecated_event": EventDescriptor(
                name="deprecated_event",
                version=2,
                deprecated=True,
                successor="new_event",
            ),
        }
        mock_list_events.return_value = mock_events

        events = _collect_event_capabilities()

        assert events["total_count"] == 2
        assert events["deprecated_count"] == 1
        assert len(events["types"]) == 2

        # Check event details
        event_names = [e["name"] for e in events["types"]]
        assert "test_event" in event_names
        assert "deprecated_event" in event_names

        # Find deprecated event
        deprecated = next(e for e in events["types"] if e["name"] == "deprecated_event")
        assert deprecated["deprecated"] is True
        assert deprecated["successor"] == "new_event"

    def test_collect_events_import_error(self):
        """Test collecting event capabilities with import error."""
        with patch("src.mcp_local.capabilities_tool.list_events") as mock_list:
            mock_list.side_effect = ImportError("Module not found")

            events = _collect_event_capabilities()

            assert events["status"] == "registry_not_available"

    @patch("src.mcp_local.capabilities_tool.list_events")
    def test_collect_events_general_error(self, mock_list_events):
        """Test collecting event capabilities with general error."""
        mock_list_events.side_effect = Exception("General error")

        events = _collect_event_capabilities()

        assert events["status"] == "error"
        assert "error" in events


class TestToolCapabilities:
    """Test tool capability collection."""

    def test_collect_tools_placeholder(self):
        """Test tool capability collection (placeholder implementation)."""
        tools = _collect_tool_capabilities()

        # Current implementation is placeholder
        assert tools["total_count"] == 0
        assert tools["status"] == "detection_pending"
        assert "note" in tools


class TestSystemCapabilities:
    """Test system capability collection."""

    def test_collect_system_capabilities(self):
        """Test system capability collection."""
        system = _collect_system_capabilities()

        # Check required fields
        required_fields = ["python_version", "platform", "memory_usage", "uptime"]
        for field_name in required_fields:
            assert field_name in system

        # Python version should be in expected format
        assert isinstance(system["python_version"], str)
        assert "." in system["python_version"]

        # Platform should be non-empty
        assert isinstance(system["platform"], str)
        assert len(system["platform"]) > 0

    @patch("src.mcp_local.capabilities_tool.psutil")
    def test_collect_system_with_psutil(self, mock_psutil):
        """Test system capability collection with psutil available."""
        # Mock psutil
        mock_process = MagicMock()
        mock_memory = MagicMock()
        mock_memory.rss = 1024 * 1024 * 100  # 100MB
        mock_memory.vms = 1024 * 1024 * 200  # 200MB
        mock_process.memory_info.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process
        mock_psutil.boot_time.return_value = 1234567890.0

        system = _collect_system_capabilities()

        assert system["memory_usage"] is not None
        assert "rss_mb" in system["memory_usage"]
        assert "vms_mb" in system["memory_usage"]
        assert system["uptime"] == 1234567890.0

    def test_collect_system_without_psutil(self):
        """Test system capability collection without psutil."""
        with patch("src.mcp_local.capabilities_tool.psutil", side_effect=ImportError):
            system = _collect_system_capabilities()

            assert system["memory_usage"] is None
            assert system["uptime"] is None

    def test_optional_dependencies_check(self):
        """Test optional dependencies checking."""
        system = _collect_system_capabilities()

        assert "optional_dependencies" in system
        deps = system["optional_dependencies"]

        # Should check common optional modules
        expected_modules = [
            "redis",
            "psutil",
            "yaml",
            "requests",
            "beautifulsoup4",
            "selenium",
        ]
        for module in expected_modules:
            assert module in deps
            assert isinstance(deps[module], bool)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_timestamp(self):
        """Test timestamp generation."""
        from src.mcp_local.capabilities_tool import _get_timestamp

        timestamp = _get_timestamp()
        assert isinstance(timestamp, float)
        assert timestamp > 0

    def test_get_python_version(self):
        """Test Python version detection."""
        from src.mcp_local.capabilities_tool import _get_python_version

        version = _get_python_version()
        assert isinstance(version, str)
        assert version.count(".") >= 2  # Should be like "3.11.0"

    def test_get_platform_info(self):
        """Test platform information detection."""
        from src.mcp_local.capabilities_tool import _get_platform_info

        platform_info = _get_platform_info()
        assert isinstance(platform_info, str)
        assert len(platform_info) > 0

    def test_check_optional_dependencies(self):
        """Test optional dependencies checking."""
        from src.mcp_local.capabilities_tool import _check_optional_dependencies

        deps = _check_optional_dependencies()
        assert isinstance(deps, dict)

        # yaml should be available (used in tests)
        assert "yaml" in deps
        assert deps["yaml"] is True
