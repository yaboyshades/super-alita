"""Performance benchmark tests for critical components."""

# Import components to benchmark
import sys
import time

import pytest

sys.path.insert(0, "src")

from core.telemetry_broker import TelemetryBroker
from reug_runtime.llm_client import OllamaClient, get_llm_client


class TestLLMClientBenchmarks:
    """Benchmark LLM client performance."""

    @pytest.mark.benchmark
    def test_ollama_client_creation(self, benchmark):
        """Benchmark Ollama client instantiation."""

        def create_client():
            return OllamaClient(
                base_url="http://localhost:11434", model="test-model", api_key=None
            )

        result = benchmark(create_client)
        assert result is not None

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_llm_client_selection(self, benchmark):
        """Benchmark LLM client selection logic."""

        async def select_client():
            # Mock config to avoid real service calls
            return await get_llm_client(
                preferred="ollama", fallback_chain=["ollama", "gpt-4o-mini"]
            )

        # Use async benchmark
        start = time.perf_counter()
        result = await select_client()
        duration = time.perf_counter() - start

        assert result is not None
        assert duration < 0.1  # Should be fast


class TestTelemetryBrokerBenchmarks:
    """Benchmark telemetry broker performance."""

    @pytest.fixture
    def broker(self):
        """Create a telemetry broker for testing."""
        return TelemetryBroker(ring_buffer_size=1000)

    @pytest.mark.benchmark
    def test_ingest_single_event(self, benchmark, broker):
        """Benchmark single event ingestion."""

        def ingest_event():
            broker.ingest(
                {
                    "event": "test.action",
                    "timestamp": time.time(),
                    "data": {"key": "value"},
                }
            )

        benchmark(ingest_event)
        assert broker.get_event_count() > 0

    @pytest.mark.benchmark
    def test_ingest_batch_events(self, benchmark, broker):
        """Benchmark batch event ingestion."""

        events = []
        for i in range(100):
            events.append(
                {
                    "event": f"test.batch.{i}",
                    "timestamp": time.time(),
                    "data": {"index": i},
                }
            )

        def ingest_batch():
            for event in events:
                broker.ingest(event)

        benchmark(ingest_batch)
        assert broker.get_event_count() >= 100

    @pytest.mark.benchmark
    def test_envelope_building(self, benchmark, broker):
        """Benchmark context envelope building."""

        # Pre-populate with events
        for i in range(50):
            broker.ingest(
                {
                    "event": f"test.envelope.{i}",
                    "timestamp": time.time(),
                    "data": {"index": i},
                }
            )

        def build_envelope():
            return broker.build_context_envelope(
                context_hint="test context", max_events=20
            )

        result = benchmark(build_envelope)
        assert "events" in result
        assert "summary" in result


class TestExtensionBenchmarks:
    """Benchmark extension performance metrics."""

    @pytest.mark.benchmark
    def test_wasm_analyzer_initialization(self, benchmark):
        """Benchmark WASM analyzer initialization time."""

        def create_analyzer():
            # Simulate analyzer creation without actual VS Code context
            class MockAnalyzer:
                def __init__(self):
                    self.history = {}
                    self.buffer = []
                    self.initialized = True

                def initialize(self):
                    # Simulate initialization work
                    time.sleep(0.001)  # 1ms simulation
                    return True

            analyzer = MockAnalyzer()
            analyzer.initialize()
            return analyzer

        result = benchmark(create_analyzer)
        assert hasattr(result, "initialized")

    @pytest.mark.benchmark
    def test_code_complexity_calculation(self, benchmark):
        """Benchmark code complexity analysis."""

        # Sample code for analysis
        sample_code = """
        def complex_function(x, y, z):
            if x > 0:
                for i in range(y):
                    if i % 2 == 0:
                        try:
                            result = z / i
                        except ZeroDivisionError:
                            result = 0
                        finally:
                            print(result)
                    else:
                        while z > 0:
                            z -= 1
                            if z < 10:
                                break
            else:
                match x:
                    case -1:
                        return "negative"
                    case 0:
                        return "zero"
                    case _:
                        return "other"
        """

        def calculate_complexity():
            # Simulate complexity calculation
            complexity = 1
            patterns = [
                r"\bif\b",
                r"\belse\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bswitch\b",
                r"\bcatch\b",
                r"\bmatch\b",
                r"\btry\b",
            ]

            import re

            for pattern in patterns:
                matches = re.findall(pattern, sample_code)
                complexity += len(matches)

            return complexity

        result = benchmark(calculate_complexity)
        assert result > 1  # Should detect some complexity


class TestIntegrationBenchmarks:
    """Benchmark full integration scenarios."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    def test_end_to_end_latency(self, benchmark):
        """Benchmark end-to-end operation latency."""

        def simulate_full_cycle():
            # Simulate: telemetry -> broker -> analysis -> response
            broker = TelemetryBroker()

            # Ingest some events
            broker.ingest(
                {
                    "event": "user.action",
                    "timestamp": time.time(),
                    "data": {"action": "code_analysis"},
                }
            )

            # Build context
            envelope = broker.build_context_envelope("analysis context")

            # Simulate analysis
            analysis_result = {
                "complexity": len(envelope.get("events", [])),
                "recommendations": ["optimize loops", "reduce nesting"],
            }

            return analysis_result

        result = benchmark(simulate_full_cycle)
        assert "complexity" in result
        assert "recommendations" in result


# Custom benchmark configuration
def pytest_configure(config):
    """Configure pytest benchmarks."""
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle benchmark markers."""
    benchmark_items = []
    regular_items = []

    for item in items:
        if "benchmark" in item.keywords:
            benchmark_items.append(item)
        else:
            regular_items.append(item)

    # Run regular tests first, then benchmarks
    items[:] = regular_items + benchmark_items
