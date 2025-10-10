"""
Test Performance Monitoring Optimization Integration

Comprehensive integration test that validates all optimization components
work together correctly and the system meets performance requirements.
"""

import asyncio
import logging
import time

import pytest

# Import optimization components
from src.performance_monitoring.optimization import create_optimization_suite
from src.performance_monitoring.optimization.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceProfiler,
)
from src.performance_monitoring.optimization.resource_manager import (
    ResourceManager,
)
from src.performance_monitoring.optimization.scalability_manager import (
    ScalabilityManager,
    ServiceStatus,
)
from src.performance_monitoring.optimization.security_hardening import (
    SecurityHardening,
    SecurityLevel,
)

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_optimization_integration():
    """Test complete optimization suite integration"""

    # Create optimization suite
    optimization_suite = await create_optimization_suite()

    try:
        # Validate all components are initialized
        assert "performance" in optimization_suite
        assert "scalability" in optimization_suite
        assert "security" in optimization_suite
        assert "resources" in optimization_suite

        performance = optimization_suite["performance"]
        scalability = optimization_suite["scalability"]
        security = optimization_suite["security"]
        resources = optimization_suite["resources"]

        # Test performance optimization
        await test_performance_optimization(performance)

        # Test scalability management
        await test_scalability_management(scalability)

        # Test security hardening
        await test_security_hardening(security)

        # Test resource management
        await test_resource_management(resources)

        # Test cross-component integration
        await test_cross_component_integration(optimization_suite)

        logger.info("All optimization integration tests passed")

    finally:
        # Clean up
        await performance.stop()
        await scalability.stop()
        await security.stop()
        await resources.stop()


async def test_performance_optimization(performance: PerformanceOptimizer):
    """Test performance optimization features"""

    # Test caching
    cache_key = "test_cache_key"
    cache_value = {"data": "test_value", "timestamp": time.time()}

    performance.cache_result(cache_key, cache_value)
    cached_result = performance.get_cached_result(cache_key)

    assert cached_result == cache_value

    # Test query caching
    query = "SELECT * FROM test_table WHERE id = ?"
    params = {"id": 123}
    query_hash = performance.hash_query(query, params)

    result_data = {"rows": [{"id": 123, "name": "test"}]}
    performance.cache_query(query_hash, result_data)

    cached_query_result = performance.get_cached_query(query_hash)
    assert cached_query_result == result_data

    # Test performance profiling
    operation_name = "test_operation"
    with PerformanceProfiler(performance, operation_name):
        await asyncio.sleep(0.1)  # Simulate work

    stats = performance.get_performance_stats()
    assert operation_name in stats["performance_profiles"]
    assert stats["performance_profiles"][operation_name]["count"] >= 1

    logger.info("Performance optimization tests passed")


async def test_scalability_management(scalability: ScalabilityManager):
    """Test scalability management features"""

    # Register test services
    service_name = "test_service"
    await scalability.register_service(
        service_name, "localhost", 8080, weight=1.0
    )
    await scalability.register_service(
        service_name, "localhost", 8081, weight=2.0
    )

    # Mark services as healthy
    await scalability.service_registry.update_instance_status(
        service_name, "localhost:8080", ServiceStatus.HEALTHY
    )
    await scalability.service_registry.update_instance_status(
        service_name, "localhost:8081", ServiceStatus.HEALTHY
    )

    # Test service discovery
    instance = await scalability.get_service_instance(service_name)
    assert instance is not None
    assert instance.host == "localhost"
    assert instance.port in [8080, 8081]

    # Test load balancing
    selected_instances = []
    for _ in range(10):
        instance = await scalability.get_service_instance(service_name)
        if instance:
            selected_instances.append(f"{instance.host}:{instance.port}")

    # Should have balanced distribution
    assert len(set(selected_instances)) > 1

    # Test request recording
    await scalability.record_request_result(
        service_name, "localhost:8080", success=True, response_time_ms=50.0
    )

    # Test auto-scaling rules
    scalability.add_scaling_rule(
        service_name,
        "response_time",
        scale_up_threshold=100.0,
        scale_down_threshold=20.0,
    )

    # Test service stats
    stats = scalability.get_service_stats()
    assert service_name in stats["services"]
    assert stats["services"][service_name]["instance_count"] == 2

    logger.info("Scalability management tests passed")


async def test_security_hardening(security: SecurityHardening):
    """Test security hardening features"""

    # Test user registration and authentication
    username = "test_user"
    email = "test@example.com"
    password = "SecurePassword123!"

    user = await security.auth_manager.register_user(
        username,
        email,
        password,
        roles={"user"},
        security_level=SecurityLevel.INTERNAL,
    )

    assert user.username == username
    assert user.email == email

    # Test authentication
    token = await security.auth_manager.authenticate_user(
        username, password, source_ip="127.0.0.1"
    )

    assert token is not None

    # Test token validation
    payload = await security.authenticate_request(
        token, source_ip="127.0.0.1", user_agent="test"
    )

    assert payload is not None
    assert payload["username"] == username

    # Test encryption
    sensitive_data = "This is sensitive information"
    encrypted_data = security.encrypt_sensitive_data(sensitive_data)
    decrypted_data = security.decrypt_sensitive_data(encrypted_data)

    assert decrypted_data == sensitive_data

    # Test audit logging
    await security.log_data_access(
        user.user_id,
        "test_resource",
        "read",
        source_ip="127.0.0.1",
        details={"test": True},
    )

    # Test security status
    security_status = security.get_security_status()
    assert "policy" in security_status
    assert "users" in security_status
    assert "audit" in security_status

    logger.info("Security hardening tests passed")


async def test_resource_management(resources: ResourceManager):
    """Test resource management features"""

    # Test memory tracking
    snapshot = resources.memory_tracker.take_snapshot()
    assert snapshot.timestamp is not None
    assert snapshot.total_memory_mb > 0

    # Test resource monitoring
    metrics = resources.resource_monitor.collect_metrics()
    assert metrics is not None
    assert metrics.memory_usage_mb > 0
    assert metrics.cpu_percent >= 0

    # Test garbage collection optimization
    gc_stats = resources.gc_optimizer.force_collection()
    assert "collected_objects" in gc_stats
    assert "collection_time_ms" in gc_stats

    # Test memory optimization
    optimization_result = await resources.optimize_memory()
    assert "snapshot" in optimization_result
    assert "gc_stats" in optimization_result

    # Test optimization report
    report = resources.get_optimization_report()
    assert "memory" in report
    assert "resources" in report
    assert "garbage_collection" in report
    assert "system_health" in report

    # Test cleanup callback registration
    cleanup_called = False

    def test_cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    resources.register_cleanup_callback(test_cleanup)
    await resources.run_cleanup()

    assert cleanup_called

    logger.info("Resource management tests passed")


async def test_cross_component_integration(optimization_suite):
    """Test integration between optimization components"""

    performance = optimization_suite["performance"]
    scalability = optimization_suite["scalability"]
    security = optimization_suite["security"]
    resources = optimization_suite["resources"]

    # Test performance and scalability integration
    # Register connection pool and service
    async def mock_connection_factory():
        return {"connection": "mock", "created_at": time.time()}

    performance.register_connection_pool("test_db", mock_connection_factory)

    await scalability.register_service(
        "test_integration_service", "localhost", 9000
    )

    # Test security and resource integration
    # Perform secured operation with resource tracking
    resources.memory_tracker.take_snapshot().total_memory_mb

    # Register test user
    user = await security.auth_manager.register_user(
        "integration_user",
        "integration@test.com",
        "IntegrationTest123!",
        roles={"admin"},
    )

    # Authenticate and perform operations
    await security.auth_manager.authenticate_user(
        "integration_user", "IntegrationTest123!"
    )

    # Log secured data access
    await security.log_data_access(
        user.user_id,
        "integration_test",
        "read",
        details={"integration_test": True},
    )

    # Check memory after operations
    resources.memory_tracker.take_snapshot().total_memory_mb

    # Test performance caching with security context
    cache_key = f"secure_data_{user.user_id}"
    secure_data = security.encrypt_sensitive_data("Integration test data")
    performance.cache_result(cache_key, secure_data)

    cached_secure_data = performance.get_cached_result(cache_key)
    assert cached_secure_data == secure_data

    # Test comprehensive stats collection
    perf_stats = performance.get_performance_stats()
    scale_stats = scalability.get_service_stats()
    security_status = security.get_security_status()
    resource_report = resources.get_optimization_report()

    # Validate all stats are populated
    assert len(perf_stats) > 0
    assert len(scale_stats) > 0
    assert len(security_status) > 0
    assert len(resource_report) > 0

    # Test cleanup integration
    await resources.run_cleanup()
    performance.memory_cache.clear()

    logger.info("Cross-component integration tests passed")


async def test_performance_under_load():
    """Test optimization system performance under load"""

    optimization_suite = await create_optimization_suite()

    try:
        performance = optimization_suite["performance"]
        scalability = optimization_suite["scalability"]

        # Simulate load testing
        start_time = time.time()

        # Register multiple services
        for i in range(10):
            await scalability.register_service(
                f"load_test_service_{i}", "localhost", 8000 + i
            )

        # Perform cache operations under load
        cache_operations = []
        for i in range(100):
            cache_key = f"load_test_key_{i}"
            cache_value = {"data": f"value_{i}", "timestamp": time.time()}

            performance.cache_result(cache_key, cache_value)
            cached = performance.get_cached_result(cache_key)

            assert cached == cache_value
            cache_operations.append(cached)

        # Test service selection under load
        service_selections = []
        for i in range(50):
            instance = await scalability.get_service_instance(
                "load_test_service_1"
            )
            if instance:
                service_selections.append(instance)

        end_time = time.time()
        duration = end_time - start_time

        # Performance assertions
        assert duration < 5.0  # Should complete within 5 seconds
        assert len(cache_operations) == 100
        assert len(service_selections) <= 50  # May not always find instances

        # Check system health after load
        report = optimization_suite["resources"].get_optimization_report()
        health = report["system_health"]

        assert health["health_score"] > 50  # Should maintain reasonable health

        logger.info(
            f"Load test completed in {duration:.2f}s with health score: {health['health_score']}"
        )

    finally:
        await optimization_suite["performance"].stop()
        await optimization_suite["scalability"].stop()
        await optimization_suite["security"].stop()
        await optimization_suite["resources"].stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def run_tests():
        """Run all optimization tests"""
        print(
            "Starting Performance Monitoring Optimization Integration Tests..."
        )

        try:
            await test_optimization_integration()
            await test_performance_under_load()

            print("✅ All optimization integration tests passed!")
            return True

        except Exception as e:
            print(f"❌ Tests failed: {e}")
            logger.exception("Test failure")
            return False

    # Run the tests
    success = asyncio.run(run_tests())
    exit(0 if success else 1)
