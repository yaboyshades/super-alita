#!/usr/bin/env python3
"""
Performance Monitoring and Rule Automation System Demo

Demonstrates the complete performance monitoring and constitutional compliance
system with real-time monitoring, quality gates, and dashboard interface.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the performance monitoring system
from src.performance_monitoring.integration import PerformanceMonitoringSystem


async def demo_system_initialization():
    """Demonstrate system initialization and basic functionality."""
    logger.info("=== Performance Monitoring System Demo ===")
    
    # Create and start the monitoring system
    logger.info("Initializing Performance Monitoring System...")
    system = PerformanceMonitoringSystem(
        constitutional_threshold=0.75
    )
    
    await system.start_system()
    logger.info("System started successfully!")
    
    return system


async def demo_extension_monitoring(system):
    """Demonstrate extension interaction monitoring."""
    logger.info("\n--- Extension Interaction Monitoring Demo ---")
    
    # Simulate extension interactions
    interactions = []
    
    for i in range(5):
        interaction = system.performance_monitor.track_extension_interaction(
            f"demo_extension_{i % 2}", f"function_{i}"
        )
        interactions.append(interaction)
        
        # Simulate processing time
        await asyncio.sleep(0.1 + (i * 0.05))
        
        # Complete interaction
        success = i != 2  # Make one fail for demo
        error_msg = "Demo error" if not success else None
        system.performance_monitor.complete_interaction(interaction, success, error_msg)
        
        logger.info(f"Completed interaction {i}: {success}")
    
    # Show performance summary
    summary = system.performance_monitor.get_performance_summary()
    logger.info("Performance Summary:")
    logger.info(f"  Total interactions: {summary['interactions_count']}")
    logger.info(f"  Average response time: {summary['average_response_time_ms']:.2f}ms")
    logger.info(f"  Success rate: {summary['success_rate']:.2%}")


async def demo_telemetry_collection(system):
    """Demonstrate telemetry data collection."""
    logger.info("\n--- Telemetry Collection Demo ---")
    
    # Simulate host API calls
    call1 = system.telemetry_bridge.track_host_api_call(
        "read_file", {"path": "/demo/file.txt"}, "file_access_constitutional_check"
    )
    await asyncio.sleep(0.05)
    system.telemetry_bridge.complete_host_api_call(call1, result="file_content")
    
    # Simulate WASM operations
    wasm_op = system.telemetry_bridge.track_wasm_operation(
        "constitutional_analyzer", "validate_compliance", 2048
    )
    await asyncio.sleep(0.08)
    system.telemetry_bridge.complete_wasm_operation(
        wasm_op, result="compliance_score", memory_usage_bytes=4096, prediction_accuracy=0.85
    )
    
    # Simulate LSP messages
    lsp_msg = system.telemetry_bridge.track_lsp_message(
        "request", "textDocument/completion", 512, "code_completion_constitutional_relevance"
    )
    system.telemetry_bridge.update_lsp_message(lsp_msg, latency_ms=25.5, diagnostic_count=3)
    
    # Show telemetry summary
    telemetry_summary = system.telemetry_bridge.get_telemetry_summary()
    logger.info("Telemetry Summary:")
    logger.info(f"  Total events: {telemetry_summary['total_events']}")
    logger.info(f"  Event rate: {telemetry_summary['event_rate_per_minute']}/min")
    
    # Show constitutional compliance data
    const_data = system.telemetry_bridge.get_constitutional_compliance_data()
    if const_data.get("status") != "no_data":
        logger.info(f"  Constitutional events: {const_data['total_constitutional_events']}")


async def demo_constitutional_validation(system):
    """Demonstrate constitutional compliance validation."""
    logger.info("\n--- Constitutional Compliance Demo ---")
    
    # Test code change validation
    test_code_changes = [
        "def new_feature():",
        "    # This is a new feature implementation",
        "    return process_data()",
        "",
        "def process_data():",
        "    data = fetch_external_data()",
        "    return transform(data)"
    ]
    
    result = await system.validate_code_change(
        "src/new_feature.py",
        test_code_changes,
        {"author": "demo_user", "component": "core"}
    )
    
    compliance = result["compliance_score"]
    logger.info("Code Change Validation:")
    logger.info(f"  Overall score: {compliance['overall_score']:.3f}")
    logger.info(f"  Compliant: {compliance['is_compliant']}")
    logger.info(f"  Violations: {len(compliance['violations'])}")
    
    for violation in compliance['violations'][:3]:  # Show first 3 violations
        logger.info(f"    - {violation['description']}")


async def demo_commit_validation(system):
    """Demonstrate commit validation through quality gates."""
    logger.info("\n--- Commit Validation Demo ---")
    
    # Test commit validation
    commit_message = "feat: implement constitutional compliance monitoring"
    changed_files = [
        "src/performance_monitoring/core/constitutional_engine.py",
        "src/performance_monitoring/dashboard/dashboard_interface.py",
        "tests/test_constitutional_compliance.py"
    ]
    diff_data = """
+def validate_constitutional_compliance(data):
+    '''Validate data against constitutional principles.'''
+    score = calculate_compliance_score(data)
+    return score >= COMPLIANCE_THRESHOLD
    """
    
    validation_result = await system.validate_commit(
        commit_message, changed_files, diff_data
    )
    
    logger.info("Commit Validation:")
    logger.info(f"  Overall passed: {validation_result['overall_passed']}")
    logger.info(f"  Execution time: {validation_result['execution_time_ms']:.2f}ms")
    logger.info(f"  Gates passed: {validation_result['summary']['passed_gates']}/{validation_result['summary']['total_gates']}")
    
    for gate_result in validation_result['gate_results']:
        logger.info(f"  {gate_result['gate_name']}: {'PASS' if gate_result['passed'] else 'FAIL'} (score: {gate_result['score']:.3f})")
        if gate_result['violations']:
            for violation in gate_result['violations'][:2]:  # Show first 2 violations
                logger.info(f"    - {violation}")


async def demo_dashboard_interface(system):
    """Demonstrate dashboard interface functionality."""
    logger.info("\n--- Dashboard Interface Demo ---")
    
    # Get dashboard state
    dashboard_state = system.dashboard.get_dashboard_state()
    logger.info("Dashboard State:")
    logger.info(f"  Status: {dashboard_state['status']}")
    logger.info(f"  Metrics count: {len(dashboard_state['metrics'])}")
    logger.info(f"  Alerts count: {len(dashboard_state['alerts'])}")
    
    # Show key metrics
    logger.info("Key Metrics:")
    for name, metric in list(dashboard_state['metrics'].items())[:5]:  # Show first 5 metrics
        logger.info(f"  {name}: {metric['value']} {metric['unit']} ({metric['status']})")
    
    # Get constitutional dashboard
    const_dashboard = system.get_constitutional_dashboard()
    logger.info("Constitutional Dashboard:")
    const_metrics = const_dashboard['constitutional_dashboard']['constitutional_metrics']
    if const_metrics:
        for name, metric in const_metrics.items():
            logger.info(f"  {name}: {metric['value']} ({metric['status']})")
    else:
        logger.info("  No constitutional metrics available yet")


async def demo_system_health_check(system):
    """Demonstrate system health monitoring."""
    logger.info("\n--- System Health Check Demo ---")
    
    health_status = await system.run_health_check()
    logger.info("System Health Check:")
    logger.info(f"  Overall health: {health_status['overall_health']}")
    
    for component, health in health_status['component_health'].items():
        status = health.get('status', 'unknown')
        logger.info(f"  {component}: {status}")
        if status == 'unhealthy':
            logger.warning(f"    Error: {health.get('error', 'Unknown error')}")
    
    if health_status['issues']:
        logger.warning("Health Issues:")
        for issue in health_status['issues']:
            logger.warning(f"  - {issue}")


async def demo_system_status(system):
    """Show comprehensive system status."""
    logger.info("\n--- System Status Overview ---")
    
    status = system.get_system_status()
    logger.info("Component Status:")
    for component, running in status['component_status'].items():
        logger.info(f"  {component}: {'Running' if running else 'Stopped'}")
    
    # Show constitutional compliance trend
    const_trend = status.get('constitutional_compliance', {})
    if const_trend.get('status') != 'no_data':
        logger.info("Constitutional Compliance Trend:")
        logger.info(f"  Average score: {const_trend.get('average_score', 0):.3f}")
        logger.info(f"  Compliance rate: {const_trend.get('compliance_rate', 0):.2%}")
        logger.info(f"  Trend: {const_trend.get('trend', 'unknown')}")


async def main():
    """Main demo function."""
    system = None
    
    try:
        # Initialize system
        system = await demo_system_initialization()
        
        # Run demonstration scenarios
        await demo_extension_monitoring(system)
        await demo_telemetry_collection(system)
        await demo_constitutional_validation(system)
        await demo_commit_validation(system)
        await demo_dashboard_interface(system)
        await demo_system_health_check(system)
        await demo_system_status(system)
        
        logger.info("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
        
    finally:
        if system:
            logger.info("Shutting down system...")
            await system.stop_system()
            logger.info("System shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())