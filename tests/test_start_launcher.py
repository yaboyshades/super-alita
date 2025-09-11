#!/usr/bin/env python3
"""
Test Suite for Super-Alita Unified Launcher

Tests the mode registry, argument parsing, and dispatch functionality
of the constitutional launcher implementation.

Constitutional Compliance:
- Article III (Simplicity Gate): Each test has single responsibility
- Test-First Development: Comprehensive coverage before deployment
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def launcher_module():
    """Import the start.py module for testing."""
    import start

    return start


@pytest.fixture
def mock_uvicorn():
    """Mock uvicorn to prevent actual server startup in tests."""
    with patch("uvicorn.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_web_ui():
    """Mock web UI launch to prevent actual UI startup."""
    with patch("src.ui.web_ui.launch_ui") as mock_launch:
        yield mock_launch


@pytest.fixture(autouse=True)
def _stub_uvicorn_if_missing(request):
    """Provide a lightweight stub for uvicorn if it's not installed.

    Some tests patch `uvicorn.run`. Ensure the module exists so patching works,
    unless the test explicitly opts out via marker `no_uvicorn_stub`.
    """
    if request.node.get_closest_marker("no_uvicorn_stub"):
        return
    if "uvicorn" not in sys.modules:
        sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)  # type: ignore[attr-defined]


class TestArgumentParsing:
    def test_default_arguments(self, launcher_module):
        parser = launcher_module.create_parser()
        args = parser.parse_args([])

        assert args.model == "llama3.2:3b"
        assert args.port == 8081
        assert args.host == "0.0.0.0"
        assert args.op == "act"
        assert args.launcher == "server"
        assert args.consensus is True
        assert args.debug is False

    def test_launcher_mode_argument(self, launcher_module):
        parser = launcher_module.create_parser()
        args = parser.parse_args(["--launcher", "demo-enhanced-consensus"])
        assert args.launcher == "demo-enhanced-consensus"

        args = parser.parse_args(["--launcher", "debug-reug-streaming", "--debug"])
        assert args.launcher == "debug-reug-streaming"
        assert args.debug is True

    def test_operational_mode_argument(self, launcher_module):
        parser = launcher_module.create_parser()
        for op_mode in ["shadow", "act", "batch"]:
            args = parser.parse_args(["--op", op_mode])
            assert args.op == op_mode

    def test_legacy_mode_compatibility(self, launcher_module):
        parser = launcher_module.create_parser()
        for legacy_mode in ["shadow", "act", "batch"]:
            args = parser.parse_args(["--mode", legacy_mode])
            assert args.mode == legacy_mode

        args = parser.parse_args(["--mode", "demo-enhanced-consensus"])
        assert args.mode == "demo-enhanced-consensus"

    def test_list_modes_argument(self, launcher_module):
        parser = launcher_module.create_parser()
        args = parser.parse_args(["--list-modes"])
        assert args.list_modes is True

    def test_complex_argument_combinations(self, launcher_module):
        parser = launcher_module.create_parser()
        args = parser.parse_args(
            [
                "--launcher",
                "live-agent-demo",
                "--op",
                "shadow",
                "--model",
                "gpt-4",
                "--port",
                "9000",
                "--host",
                "127.0.0.1",
                "--debug",
                "--ui",
            ]
        )
        assert args.launcher == "live-agent-demo"
        assert args.op == "shadow"
        assert args.model == "gpt-4"
        assert args.port == 9000
        assert args.host == "127.0.0.1"
        assert args.debug is True
        assert args.ui is True


class TestModeRegistry:
    def test_mode_registration(self, launcher_module):
        core_modes = ["server", "runtime", "api", "consensus"]
        for mode in core_modes:
            assert mode in launcher_module._MODES
            assert "handler" in launcher_module._MODES[mode]
            assert "description" in launcher_module._MODES[mode]
            assert "category" in launcher_module._MODES[mode]

    def test_demo_modes_registration(self, launcher_module):
        demo_modes = [
            "demo-enhanced-consensus",
            "demo-consensus-chat",
            "demo-github-integration",
            "live-agent-demo",
            "complete-agent-demo",
        ]
        for mode in demo_modes:
            assert mode in launcher_module._MODES
            assert launcher_module._MODES[mode]["category"] == "demo"

    def test_debug_modes_registration(self, launcher_module):
        debug_modes = [
            "debug-reug-streaming",
            "debug-consensus-500-error",
            "debug-autogen-pipeline",
        ]
        for mode in debug_modes:
            assert mode in launcher_module._MODES
            assert launcher_module._MODES[mode]["category"] == "debug"

    def test_mangle_modes_registration(self, launcher_module):
        mangle_modes = ["run-mangle-demo", "mangle-simple-demo"]
        for mode in mangle_modes:
            assert mode in launcher_module._MODES
            assert launcher_module._MODES[mode]["category"] == "mangle"

    def test_mode_registry_decorator(self, launcher_module):
        @launcher_module.register_mode("test-mode", "Test mode description", "test")
        def test_handler(args):  # noqa: ANN001
            return "test result"

        assert "test-mode" in launcher_module._MODES
        assert launcher_module._MODES["test-mode"]["description"] == "Test mode description"
        assert launcher_module._MODES["test-mode"]["category"] == "test"
        assert launcher_module._MODES["test-mode"]["handler"] == test_handler


class TestModeListingAndValidation:
    def test_list_modes_output(self, launcher_module, capsys):
        launcher_module.list_modes()
        captured = capsys.readouterr()
        assert "Available launcher modes:" in captured.out
        assert "[core]" in captured.out
        assert "[demo]" in captured.out
        assert "[debug]" in captured.out
        assert "[mangle]" in captured.out
        assert "server" in captured.out
        assert "demo-enhanced-consensus" in captured.out
        assert "debug-reug-streaming" in captured.out
        assert "run-mangle-demo" in captured.out

    def test_mode_categorization(self, launcher_module, capsys):
        launcher_module.list_modes()
        captured = capsys.readouterr()
        lines = captured.out.split("\n")
        core_section = any("[core]" in l for l in lines)
        demo_section = any("[demo]" in l for l in lines)
        debug_section = any("[debug]" in l for l in lines)
        mangle_section = any("[mangle]" in l for l in lines)
        assert core_section and demo_section and debug_section and mangle_section


class TestModeExecution:
    def test_server_mode_execution(self, launcher_module, mock_uvicorn):
        args = argparse.Namespace(launcher="server", ui=False, debug=False, port=8081, host="0.0.0.0")
        handler = launcher_module._MODES["server"]["handler"]
        handler(args)
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 8081
        assert call_args[1]["reload"] is False

    def test_demo_mode_execution(self, launcher_module, mock_uvicorn):
        args = argparse.Namespace(
            launcher="demo-enhanced-consensus",
            ui=False,
            debug=False,
            port=8081,
            host="0.0.0.0",
        )
        handler = launcher_module._MODES["demo-enhanced-consensus"]["handler"]
        handler(args)
        assert args.debug is True
        mock_uvicorn.assert_called_once()

    def test_mangle_mode_environment_setup(self, launcher_module, mock_uvicorn):
        args = argparse.Namespace(launcher="run-mangle-demo", ui=False, debug=False, port=8081, host="0.0.0.0")
        if "MANGLE_TIMEOUT" in os.environ:
            del os.environ["MANGLE_TIMEOUT"]
        handler = launcher_module._MODES["run-mangle-demo"]["handler"]
        handler(args)
        assert os.environ.get("MANGLE_TIMEOUT") == "30"
        mock_uvicorn.assert_called_once()

    def test_ui_mode_execution(self, launcher_module, mock_web_ui, mock_uvicorn):
        args = argparse.Namespace(launcher="server", ui=True, debug=False, port=8081, host="0.0.0.0")
        handler = launcher_module._MODES["server"]["handler"]
        handler(args)
        mock_web_ui.assert_called_once_with(port=8081)
        mock_uvicorn.assert_not_called()

    @patch("src.ui.web_ui.launch_ui", side_effect=ImportError("UI module not found"))
    def test_ui_fallback_to_api(self, mock_web_ui, launcher_module, mock_uvicorn):  # noqa: ARG001
        args = argparse.Namespace(launcher="server", ui=True, debug=False, port=8081, host="0.0.0.0")
        handler = launcher_module._MODES["server"]["handler"]
        handler(args)
        mock_web_ui.assert_called_once()
        mock_uvicorn.assert_called_once()
        assert args.ui is False


class TestEnvironmentConfiguration:
    def test_setup_environment_basic(self, launcher_module):
        args = argparse.Namespace(
            model="test-model",
            port=9000,
            op="shadow",
            launcher="test-launcher",
            consensus=True,
            debug=True,
            config=None,
        )
        launcher_module.setup_environment(args)
        assert os.environ["SUPER_ALITA_MODEL"] == "test-model"
        assert os.environ["SUPER_ALITA_PORT"] == "9000"
        assert os.environ["SUPER_ALITA_MODE"] == "shadow"
        assert os.environ["SUPER_ALITA_LAUNCHER"] == "test-launcher"
        assert os.environ["CONSENSUS_ENABLED"] == "true"
        assert os.environ["LOG_LEVEL"] == "DEBUG"

    def test_consensus_environment_setup(self, launcher_module):
        args = argparse.Namespace(
            model="llama3.2:3b",
            port=8081,
            op="act",
            launcher="server",
            consensus=True,
            debug=False,
            config=None,
        )
        launcher_module.setup_environment(args)
        assert os.environ["CONSENSUS_ENABLED"] == "true"
        assert os.environ["CONSENSUS_METHOD"] == "weighted_vote"
        assert os.environ["PROMPT_OPTIMIZATION_ENABLED"] == "true"
        assert os.environ["LADDER_AOG_ENABLED"] == "true"

    def test_debug_environment_setup(self, launcher_module):
        args = argparse.Namespace(
            model="llama3.2:3b",
            port=8081,
            op="act",
            launcher="server",
            consensus=False,
            debug=True,
            config=None,
        )
        launcher_module.setup_environment(args)
        assert os.environ["LOG_LEVEL"] == "DEBUG"
        assert os.environ["REUG_LOG_LEVEL"] == "DEBUG"


class TestMainFunction:
    @patch("start.list_modes")
    def test_main_list_modes_exit(self, mock_list_modes, launcher_module):  # noqa: ARG001
        with patch("sys.argv", ["start.py", "--list-modes"]):
            launcher_module.main()
            mock_list_modes.assert_called_once()

    def test_legacy_mode_compatibility_in_main(self, launcher_module, mock_uvicorn):  # noqa: ARG001
        with patch("sys.argv", ["start.py", "--mode", "shadow"]):
            with patch.object(launcher_module, "setup_environment"):
                try:
                    launcher_module.main()
                except SystemExit:
                    pass

        with patch("sys.argv", ["start.py", "--mode", "demo-enhanced-consensus"]):
            with patch.object(launcher_module, "setup_environment"):
                try:
                    launcher_module.main()
                except SystemExit:
                    pass

    def test_unknown_mode_handling(self, launcher_module, mock_uvicorn, capsys):  # noqa: ARG001
        args = argparse.Namespace(
            launcher="nonexistent-mode",
            op="act",
            model="llama3.2:3b",
            port=8081,
            debug=False,
            consensus=True,
            config=None,
            list_modes=False,
            host="0.0.0.0",
        )
        with patch.object(launcher_module, "create_parser") as mock_parser:
            mock_parser.return_value.parse_args.return_value = args
            with patch.object(launcher_module, "setup_environment"):
                try:
                    launcher_module.main()
                except SystemExit:
                    pass
        captured = capsys.readouterr()
        assert "Unknown launcher mode: nonexistent-mode" in captured.out


class TestIntegration:
    def test_full_demo_workflow(self, launcher_module, mock_uvicorn):  # noqa: ARG001
        with patch(
            "sys.argv",
            [
                "start.py",
                "--launcher",
                "demo-enhanced-consensus",
                "--op",
                "act",
                "--model",
                "gpt-4",
                "--port",
                "9000",
                "--debug",
            ],
        ):
            with patch.object(launcher_module, "setup_environment") as mock_setup:
                try:
                    launcher_module.main()
                except SystemExit:
                    pass
                mock_setup.assert_called_once()
                mock_uvicorn.assert_called_once()

    def test_constitutional_compliance_single_responsibility(self, launcher_module):
        for mode_name, mode_info in launcher_module._MODES.items():
            assert "handler" in mode_info
            assert callable(mode_info["handler"])  # noqa: PT018

    def test_mode_registry_completeness(self, launcher_module):
        expected_core = ["server", "runtime", "api", "consensus"]
        expected_demo = [
            "demo-enhanced-consensus",
            "demo-consensus-chat",
            "demo-github-integration",
            "demo-jest-for-python",
            "live-agent-demo",
            "complete-agent-demo",
        ]
        expected_debug = [
            "debug-reug-streaming",
            "debug-consensus-500-error",
            "debug-autogen-pipeline",
        ]
        expected_mangle = ["run-mangle-demo", "mangle-simple-demo"]
        all_expected = expected_core + expected_demo + expected_debug + expected_mangle
        for mode in all_expected:
            assert mode in launcher_module._MODES, f"Mode {mode} not registered"


class TestPerformance:
    def test_mode_lookup_performance(self, launcher_module):
        import time

        start = time.time()
        for mode_name in list(launcher_module._MODES.keys()):
            assert launcher_module._MODES.get(mode_name) is not None
        total = time.time() - start
        assert total < 0.01  # dictionary lookups should be very fast

    def test_argument_parsing_performance(self, launcher_module):
        import time

        parser = launcher_module.create_parser()
        start = time.time()
        vectors = [
            ["--launcher", "server"],
            ["--launcher", "demo-enhanced-consensus", "--debug"],
            ["--op", "shadow", "--model", "gpt-4", "--port", "9000"],
            ["--list-modes"],
        ]
        for v in vectors:
            parser.parse_args(v)
        total = time.time() - start
        assert total < 0.05


class TestErrorHandling:
    def test_invalid_port_handling(self, launcher_module):
        parser = launcher_module.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--port", "invalid"])

    def test_invalid_op_mode_handling(self, launcher_module):
        parser = launcher_module.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--op", "invalid-mode"])

    @pytest.mark.no_uvicorn_stub
    def test_missing_dependencies_handling(self, launcher_module):
        args = argparse.Namespace(launcher="server", ui=False, debug=False, port=8081, host="0.0.0.0")
        # Ensure any pre-stubbed module is cleared for this test
        sys.modules.pop("uvicorn", None)
        with patch("builtins.__import__", side_effect=ImportError("uvicorn not found")):
            with pytest.raises(ImportError):
                handler = launcher_module._MODES["server"]["handler"]
                handler(args)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "--tb=short"])
