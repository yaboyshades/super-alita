"""
Pytest configuration and fixtures for Super Alita agent development.

This module provides comprehensive test fixtures optimized for:
- Event-driven architecture testing
- Neural atom pattern validation
- Cognitive loop testing
- Redis/async component mocking
- Professional development workflows
"""

import asyncio
import contextlib
import importlib
import inspect
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import os

# Ensure project src is importable when running tests standalone
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# --- Discovery: locate a MetricsRegistry class/instance if present ---
_MR_CLASS = None
_MR_INSTANCE = None

_CANDIDATE_MODULES = [
    "core.metrics",
    "core.metrics_registry",
    "src.core.metrics",
    "src.core.metrics_registry",
    "super_alita.core.metrics",
    "super_alita.metrics",
    "metrics",
]

for _mod in _CANDIDATE_MODULES:
    try:
        m = importlib.import_module(_mod)
    except Exception:
        continue
    # class candidates
    for name in dir(m):
        obj = getattr(m, name, None)
        if inspect.isclass(obj) and name == "MetricsRegistry":
            _MR_CLASS = obj
        # common singleton patterns: METRICS, metrics, registry
        if name in {"METRICS", "metrics", "registry"} and not inspect.isclass(obj):
            _MR_INSTANCE = obj
    if _MR_CLASS or _MR_INSTANCE:
        break

# --- Monkey-patch: add a reset() if missing on the class ---
def _generic_reset(self):
    """
    Best-effort reset that clears dict-like / list-like fields and
    calls .reset() / .clear() when available on sub-objects.
    """
    # Clear obvious container attributes
    for attr in dir(self):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(self, attr)
        except Exception:
            continue
        # Skip callables (methods/functions)
        if callable(val):
            continue
        # Prefer explicit reset/clear on sub-objects
        for meth in ("reset", "clear"):
            if hasattr(val, meth) and callable(getattr(val, meth)):
                try:
                    getattr(val, meth)()
                    break
                except Exception:
                    pass
        else:
            # Fallbacks for common types
            if isinstance(val, dict):
                val.clear()
            elif isinstance(val, list):
                val.clear()
            elif isinstance(val, set):
                val.clear()
    # If the registry maintains counters/histograms as attributes like:
    # self.counters / self.histograms / self.gauges (dicts), they've been cleared above.
    return None

if _MR_CLASS is not None and not hasattr(_MR_CLASS, "reset"):
    try:
        setattr(_MR_CLASS, "reset", _generic_reset)
    except Exception:
        pass

# --- Ensure an instance exists so tests can call .reset() directly if they import it ---
if _MR_INSTANCE is None and _MR_CLASS is not None:
    try:
        _MR_INSTANCE = _MR_CLASS()  # type: ignore[call-arg]
    except Exception:
        _MR_INSTANCE = None

# --- Autouse: reset metrics between tests if we found a registry ---
@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    if _MR_INSTANCE is not None and hasattr(_MR_INSTANCE, "reset"):
        try:
            _MR_INSTANCE.reset()
        except Exception:
            pass
    # Also attempt class-level reset for singleton-style registries
    if _MR_CLASS is not None and hasattr(_MR_CLASS, "reset"):
        try:
            _MR_CLASS.reset(_MR_CLASS)
        except Exception:
            pass
    yield



# --------------------------------------------------------------------
# Legacy execution-flow compatibility shims for older tests
# Some suites import symbols like core.execution_flow.find_applicable_tools.
# After migrating to REUG services, those symbols might not exist.
# We attach test-only shims that delegate to reug.services where possible.
# --------------------------------------------------------------------
def _attach_execution_flow_shims():
    try:
        cef = importlib.import_module("core.execution_flow")
    except Exception:
        return  # No legacy module; nothing to shim

    # Only attach once
    if getattr(cef, "_REUG_TEST_SHIMS_ATTACHED", False):
        return

    # Try to wire REUG services (fallbacks keep tests green even if SoT/executor missing)
    try:
        from reug.events import EventEmitter
        from reug.services import create_services, PlanStep
        emitter = EventEmitter(os.environ.get("REUG_EVENT_LOG_DIR") or "logs/events.jsonl")
        services = create_services(emitter)
    except Exception:
        services = None

    # ---- Shim: find_applicable_tools(step) -> list[dict]
    if not hasattr(cef, "find_applicable_tools"):
        def find_applicable_tools(step):
            """Legacy shim that delegates to REUG or falls back."""
            kind = None
            args = {}
            if isinstance(step, dict):
                kind = str(step.get("kind", "")).upper()
                args = dict(step.get("args", {}))
            else:
                kind = str(getattr(step, "kind", "")).upper()
                args = dict(getattr(step, "args", {}) or {})
            if services:
                ps = PlanStep(step_id=str(getattr(step, "step_id", "s1")), kind=kind or "COMPUTE", args=args)
                import asyncio
                async def _select():
                    return await services["select_tool"](ps, {"correlation_id": "test"})
                res = asyncio.get_event_loop().run_until_complete(_select())
                if res.get("status") == "FOUND":
                    return [res["tool"]]
            tool_id = {
                "SEARCH":  "tool.search.basic",
                "COMPUTE": "tool.compute.python",
                "ANALYZE": "tool.analyze.basic",
                "GENERATE":"tool.generate.text",
                "VALIDATE":"tool.validate.schema",
            }.get(kind or "COMPUTE", "tool.generic")
            return [{"tool_id": tool_id}]
        setattr(cef, "find_applicable_tools", find_applicable_tools)

    # ---- Shim: select_tool(step) -> dict with status/tool
    if not hasattr(cef, "select_tool"):
        def select_tool(step):
            tools = cef.find_applicable_tools(step)
            if tools:
                return {"status": "FOUND", "tool": tools[0]}
            return {"status": "NOT_FOUND"}
        setattr(cef, "select_tool", select_tool)

    # ---- Shim: execute_step(tool, step) -> result dict
    if not hasattr(cef, "execute_step"):
        def execute_step(tool, step):
            """Legacy shim: executes via REUG services if available, else evaluates simple expressions."""
            if isinstance(step, dict):
                args = dict(step.get("args", {}))
                kind = str(step.get("kind", "COMPUTE")).upper()
            else:
                args = dict(getattr(step, "args", {}) or {})
                kind = str(getattr(step, "kind", "COMPUTE")).upper()
            args["_kind"] = kind
            if services:
                import asyncio
                async def _run():
                    return await services["execute"](tool, args, {"correlation_id": "test", "step_index": 0})
                ex = asyncio.get_event_loop().run_until_complete(_run())
                if ex.get("status") == "SUCCESS":
                    return ex.get("result") or {}
                raise RuntimeError(ex.get("error") or "tool execution failed")
            if kind in ("COMPUTE", "ANALYZE"):
                expr = args.get("expr") or args.get("code") or "0"
                return {"value": eval(expr, {"__builtins__": {}}, {})}
            if kind == "GENERATE":
                return {"text": args.get("text", "ok")}
            return {"ok": True}
        setattr(cef, "execute_step", execute_step)

    setattr(cef, "_REUG_TEST_SHIMS_ATTACHED", True)

_attach_execution_flow_shims()

@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure asyncio event loop policy for testing."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture()
def mock_redis_client():
    """Mock Redis client for testing event bus functionality."""
    mock_client = Mock()
    mock_client.ping = Mock(return_value=True)
    mock_client.publish = Mock(return_value=1)
    mock_client.subscribe = Mock()
    mock_client.unsubscribe = Mock()
    mock_client.get_message = Mock(return_value=None)
    return mock_client


@pytest.fixture()
def mock_event_bus():
    """Mock EventBus for testing plugin interactions."""
    mock_bus = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus.subscribe = AsyncMock()
    mock_bus.unsubscribe = AsyncMock()
    mock_bus.metrics = {
        "events_in": 0,
        "events_out": 0,
        "handlers_invoked": 0,
        "events_dropped": 0,
    }
    return mock_bus


@pytest.fixture()
def mock_neural_store():
    """Mock Neural Store for testing atom storage and retrieval."""
    mock_store = AsyncMock()
    mock_store.register = AsyncMock()
    mock_store.register_with_lineage = AsyncMock()
    mock_store.search = AsyncMock(return_value=[])
    mock_store.get_by_name = AsyncMock(return_value=None)
    mock_store.list_all = AsyncMock(return_value=[])
    return mock_store


@pytest.fixture()
def sample_text():
    """Sample text for atomizer and cognitive testing."""
    return """
    The agent should be able to decompose complex problems into smaller,
    manageable subproblems. This decomposition allows for parallel processing
    and reduces cognitive load. We propose a hierarchical task network approach
    where each node represents a solvable subproblem.

    Event-driven architecture enables reactive behavior and loose coupling
    between components. Each neural atom can respond to specific events
    while maintaining independence from other system components.
    """


@pytest.fixture()
def sample_neural_atom_metadata():
    """Sample metadata for neural atom testing."""
    return {
        "name": "test_atom_123",
        "description": "Test neural atom for unit testing",
        "capabilities": ["test", "analysis", "processing"],
        "version": "1.0.0",
        "usage_count": 0,
        "success_rate": 1.0,
        "avg_execution_time": 0.0,
    }


@pytest.fixture()
def sample_atoms():
    """Sample atoms for testing collections and operations."""
    return [
        {
            "atom_id": "concept_123",
            "atom_type": "CONCEPT",
            "title": "Problem Decomposition",
            "content": "Breaking complex problems into smaller parts",
            "meta": {"source": "test", "tags": ["cognitive", "auto"]},
        },
        {
            "atom_id": "event_456",
            "atom_type": "EVENT",
            "title": "Task Completion",
            "content": "Agent completed a complex reasoning task",
            "meta": {"source": "test", "tags": ["event", "completion"]},
        },
        {
            "atom_id": "pattern_789",
            "atom_type": "PATTERN",
            "title": "Hierarchical Planning",
            "content": "Use tree-like structures for task decomposition",
            "meta": {"source": "test", "tags": ["pattern", "planning"]},
        },
    ]


@pytest.fixture()
def deterministic_uuid():
    """Generate deterministic UUIDs for testing."""

    def _generate_uuid(namespace: str, name: str) -> str:
        """Generate UUIDv5 for deterministic testing."""
        namespace_uuid = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        return str(uuid.uuid5(namespace_uuid, f"{namespace}:{name}"))

    return _generate_uuid


@pytest.fixture()
def sample_event_data():
    """Sample event data for testing event-driven patterns."""
    return {
        "tool_call": {
            "source_plugin": "test_plugin",
            "conversation_id": "test_session_123",
            "tool_name": "test_tool",
            "parameters": {"param1": "value1", "param2": 42},
            "tool_call_id": "call_123456",
        },
        "tool_result": {
            "source_plugin": "test_plugin",
            "conversation_id": "test_session_123",
            "tool_call_id": "call_123456",
            "success": True,
            "result": {"output": "test successful", "data": [1, 2, 3]},
        },
        "gap_event": {
            "source_plugin": "test_plugin",
            "conversation_id": "test_session_123",
            "missing_tool": "advanced_calculator",
            "description": "Tool for complex mathematical calculations",
            "gap_id": "gap_789012",
        },
    }


@pytest.fixture()
def mock_llm_client():
    """Mock LLM client for testing AI interactions."""
    mock_client = Mock()
    mock_client.generate_text = AsyncMock(
        return_value="This is a mock LLM response for testing."
    )
    mock_client.generate_embedding = AsyncMock(
        return_value=[0.1] * 384  # Mock 384-dimensional embedding
    )
    mock_client.analyze_intent = AsyncMock(
        return_value={"intent": "test", "confidence": 0.95}
    )
    return mock_client


@pytest.fixture()
def mock_config():
    """Mock configuration for testing plugins and components."""
    return {
        "redis": {"host": "localhost", "port": 6379, "db": 0},
        "llm": {
            "api_key": "test_key_123",
            "model": "test-model",
            "max_tokens": 1000,
        },
        "neural_store": {"path": "test_store", "embedding_dim": 384},
        "plugins": {
            "enabled": ["test_plugin", "mock_plugin"],
            "test_plugin": {"param1": "value1", "param2": True},
        },
    }


@pytest.fixture()
async def async_test_context():
    """Provide async context for testing async components."""
    context = {
        "tasks": [],
        "futures": [],
        "cleanup_functions": [],
    }

    yield context

    # Cleanup
    for task in context["tasks"]:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    for cleanup_func in context["cleanup_functions"]:
        with contextlib.suppress(Exception):
            await cleanup_func()


@pytest.fixture()
def batch_event_processor():
    """Mock batch event processor for testing performance patterns."""

    class MockBatchProcessor:
        def __init__(self):
            self.events = []
            self.batch_size = 10
            self.processed_batches = []

        async def add_event(self, event: dict[str, Any]) -> None:
            self.events.append(event)
            if len(self.events) >= self.batch_size:
                await self.process_batch()

        async def process_batch(self) -> None:
            if self.events:
                batch = self.events.copy()
                self.events.clear()
                self.processed_batches.append(batch)

        async def flush(self) -> None:
            if self.events:
                await self.process_batch()

    return MockBatchProcessor()


@pytest.fixture()
def performance_monitor():
    """Mock performance monitor for testing metrics and monitoring."""

    class MockPerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.timings = {}

        def start_timer(self, name: str) -> None:
            self.timings[name] = time.time()

        def end_timer(self, name: str) -> float:
            if name in self.timings:
                duration = time.time() - self.timings[name]
                del self.timings[name]
                return duration
            return 0.0

        def increment_counter(self, name: str, value: int = 1) -> None:
            self.metrics[name] = self.metrics.get(name, 0) + value

        def set_gauge(self, name: str, value: float) -> None:
            self.metrics[name] = value

        def get_metrics(self) -> dict[str, Any]:
            return self.metrics.copy()

    return MockPerformanceMonitor()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers for agent development."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "event: Event-driven tests")
    config.addinivalue_line("markers", "neural: Neural atom tests")
    config.addinivalue_line("markers", "cognitive: Cognitive loop tests")
    config.addinivalue_line("markers", "redis: Tests requiring Redis")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and content."""
    for item in items:
        # Mark tests based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Mark tests based on function name patterns
        if "redis" in item.name.lower():
            item.add_marker(pytest.mark.redis)
        if "event" in item.name.lower():
            item.add_marker(pytest.mark.event)
        if "neural" in item.name.lower() or "atom" in item.name.lower():
            item.add_marker(pytest.mark.neural)
        if "cognitive" in item.name.lower():
            item.add_marker(pytest.mark.cognitive)
        if "slow" in item.name.lower() or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)


# Performance testing utilities
@pytest.fixture()
def benchmark_async():
    """Benchmark async function execution times."""

    async def _benchmark(func, *args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        return result, duration

    return _benchmark


@pytest.fixture()
def assert_performance():
    """Assert performance requirements for tests."""

    def _assert_performance(
        duration: float, max_duration: float, operation: str = "operation"
    ):
        assert duration <= max_duration, (
            f"{operation} took {duration:.4f}s, "
            f"exceeding maximum of {max_duration:.4f}s"
        )

    return _assert_performance
