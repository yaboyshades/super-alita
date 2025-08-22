#!/usr/bin/env python3
"""
Microbenchmark Suite for Super-Alita Performance Optimization
Uses pytest-benchmark for detailed performance profiling of core components.
"""

import asyncio
import random
import time

import pytest

# Import core components for benchmarking
from src.core.cache import CacheManager, LocalCache
from src.core.clock import FakeClock, MonotonicClock
from src.core.event_bus import EventBus
from src.core.metrics import NoOpMetricsCollector
from src.core.net import AdaptiveConcurrencyGate
from src.core.race_safe_breaker import BreakerConfig, RaceSafeCircuitBreaker


class BenchmarkFixtures:
    """Shared fixtures for benchmarks"""

    @pytest.fixture
    def cache(self):
        return LocalCache("bench_cache", max_size=1000, default_ttl=300.0)

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def clock(self):
        return FakeClock(0.0)

    @pytest.fixture
    def breaker(self, clock):
        config = BreakerConfig(failure_threshold=5, window_seconds=10.0)
        return RaceSafeCircuitBreaker("bench_breaker", config, clock)

    @pytest.fixture
    def concurrency_gate(self):
        return AdaptiveConcurrencyGate(initial_limit=50, metrics=NoOpMetricsCollector())


class TestCacheBenchmarks(BenchmarkFixtures):
    """Benchmark cache operations"""

    @pytest.mark.asyncio
    async def test_cache_set_performance(self, benchmark, cache):
        """Benchmark cache set operations"""

        async def cache_set():
            await cache.set(f"key_{random.randint(1, 1000)}", "test_value")

        await benchmark.pedantic(cache_set, rounds=100, iterations=10)

    @pytest.mark.asyncio
    async def test_cache_get_performance(self, benchmark, cache):
        """Benchmark cache get operations"""

        # Pre-populate cache
        for i in range(100):
            await cache.set(f"key_{i}", f"value_{i}")

        async def cache_get():
            return await cache.get(f"key_{random.randint(0, 99)}")

        await benchmark.pedantic(cache_get, rounds=100, iterations=10)

    @pytest.mark.asyncio
    async def test_cache_hit_ratio_under_load(self, benchmark, cache):
        """Benchmark cache performance with realistic hit ratios"""

        # Pre-populate with hot data
        hot_keys = [f"hot_key_{i}" for i in range(20)]
        for key in hot_keys:
            await cache.set(key, f"hot_value_{key}")

        async def mixed_workload():
            # 80% hot data, 20% cold data (realistic distribution)
            if random.random() < 0.8:
                key = random.choice(hot_keys)
            else:
                key = f"cold_key_{random.randint(1, 1000)}"

            result = await cache.get(key)
            if result is None and not key.startswith("cold"):
                await cache.set(key, f"value_{key}")

            return result

        await benchmark.pedantic(mixed_workload, rounds=100, iterations=5)


class TestEventBusBenchmarks(BenchmarkFixtures):
    """Benchmark event bus operations"""

    @pytest.mark.asyncio
    async def test_event_emission_performance(self, benchmark, event_bus):
        """Benchmark event emission"""

        async def emit_event():
            await event_bus.emit_event("test_event", {"data": "test_value"})

        await benchmark.pedantic(emit_event, rounds=50, iterations=10)

    @pytest.mark.asyncio
    async def test_event_handling_performance(self, benchmark, event_bus):
        """Benchmark event handling with multiple subscribers"""

        # Add event handlers
        handler_call_count = 0

        async def handler1(event):
            nonlocal handler_call_count
            handler_call_count += 1

        async def handler2(event):
            nonlocal handler_call_count
            handler_call_count += 1

        event_bus.subscribe("bench_event", handler1)
        event_bus.subscribe("bench_event", handler2)

        async def emit_with_handlers():
            await event_bus.emit_event("bench_event", {"data": "test"})

        await benchmark.pedantic(emit_with_handlers, rounds=50, iterations=5)


class TestCircuitBreakerBenchmarks(BenchmarkFixtures):
    """Benchmark circuit breaker operations"""

    @pytest.mark.asyncio
    async def test_breaker_allow_request_performance(self, benchmark, breaker):
        """Benchmark circuit breaker request checking"""

        async def check_request():
            return await breaker.allow_request()

        await benchmark.pedantic(check_request, rounds=100, iterations=20)

    @pytest.mark.asyncio
    async def test_breaker_record_result_performance(self, benchmark, breaker):
        """Benchmark result recording"""

        async def record_success():
            await breaker.record_result(True)

        await benchmark.pedantic(record_success, rounds=100, iterations=10)

    @pytest.mark.asyncio
    async def test_breaker_execute_performance(self, benchmark, breaker):
        """Benchmark function execution through breaker"""

        async def dummy_func():
            return "success"

        async def execute_with_breaker():
            return await breaker.execute_with_breaker(dummy_func)

        await benchmark.pedantic(execute_with_breaker, rounds=50, iterations=10)


class TestConcurrencyBenchmarks(BenchmarkFixtures):
    """Benchmark concurrency control components"""

    @pytest.mark.asyncio
    async def test_concurrency_gate_acquire_performance(
        self, benchmark, concurrency_gate
    ):
        """Benchmark concurrency gate acquisition"""

        async def acquire_and_release():
            async with concurrency_gate.acquire():
                await asyncio.sleep(0.001)  # Simulate small amount of work

        await benchmark.pedantic(acquire_and_release, rounds=20, iterations=5)

    @pytest.mark.asyncio
    async def test_concurrent_gate_performance(self, benchmark, concurrency_gate):
        """Benchmark gate under concurrent load"""

        async def concurrent_workload():
            tasks = []
            for _ in range(10):

                async def worker():
                    async with concurrency_gate.acquire():
                        await asyncio.sleep(0.001)

                tasks.append(asyncio.create_task(worker()))

            await asyncio.gather(*tasks)

        await benchmark.pedantic(concurrent_workload, rounds=10, iterations=2)


class TestClockBenchmarks(BenchmarkFixtures):
    """Benchmark clock operations"""

    def test_monotonic_clock_performance(self, benchmark):
        """Benchmark monotonic clock calls"""
        clock = MonotonicClock()

        def get_time():
            return clock.now()

        benchmark(get_time)

    def test_fake_clock_performance(self, benchmark, clock):
        """Benchmark fake clock operations"""

        def fake_clock_ops():
            current = clock.now()
            clock.tick(0.1)
            return clock.now()

        benchmark(fake_clock_ops)


class TestMemoryBenchmarks:
    """Benchmark memory usage patterns"""

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, benchmark):
        """Test memory usage under realistic load"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        async def memory_intensive_workload():
            # Simulate typical Super-Alita workload
            cache_manager = CacheManager()
            cache = cache_manager.get_cache("memory_test", max_size=500)

            # Fill cache
            for i in range(100):
                await cache.set(f"key_{i}", {"data": list(range(100))})

            # Access patterns
            for _ in range(50):
                key = f"key_{random.randint(0, 99)}"
                await cache.get(key)

            await cache_manager.shutdown()

            # Return memory usage in MB
            return process.memory_info().rss / 1024 / 1024

        result = await benchmark.pedantic(
            memory_intensive_workload, rounds=5, iterations=1
        )
        print(
            f"Peak memory usage: {max(result) if isinstance(result, list) else result:.2f} MB"
        )


class TestE2EBenchmarks:
    """End-to-end performance benchmarks"""

    @pytest.mark.asyncio
    async def test_full_request_pipeline_performance(self, benchmark):
        """Benchmark complete request processing pipeline"""

        async def full_pipeline():
            # Simulate full request lifecycle
            cache_manager = CacheManager()
            cache = cache_manager.get_cache("e2e_test")

            # Check cache first
            result = await cache.get("e2e_key")

            if result is None:
                # Simulate processing
                await asyncio.sleep(0.01)  # Simulate I/O
                result = {"processed": True, "timestamp": time.time()}
                await cache.set("e2e_key", result, ttl=30.0)

            await cache_manager.shutdown()
            return result

        await benchmark.pedantic(full_pipeline, rounds=20, iterations=3)


# Performance baseline configuration
PERFORMANCE_BASELINES = {
    "cache_set_p95": 0.001,  # 1ms for cache set
    "cache_get_p95": 0.0005,  # 0.5ms for cache get
    "event_emit_p95": 0.002,  # 2ms for event emission
    "breaker_check_p95": 0.0001,  # 0.1ms for breaker check
    "full_pipeline_p95": 0.015,  # 15ms for full pipeline
    "memory_usage_mb": 100,  # 100MB memory usage limit
}


def validate_performance_baselines(benchmark_results):
    """Validate benchmark results against baselines"""
    failures = []

    for metric, baseline in PERFORMANCE_BASELINES.items():
        if metric in benchmark_results:
            actual = benchmark_results[metric]
            if actual > baseline:
                failures.append(f"{metric}: {actual:.4f} > {baseline:.4f}")

    if failures:
        pytest.fail("Performance baseline violations:\n" + "\n".join(failures))


if __name__ == "__main__":
    # Run benchmarks with detailed output
    pytest.main(
        [
            __file__,
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-verbose",
            "--benchmark-autosave",
            "-v",
        ]
    )
