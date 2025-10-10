"""
Performance Monitoring and Rule Automation System

This module provides comprehensive monitoring capabilities for extension interactions,
constitutional compliance validation, and system performance tracking.

Architecture Components:
- Performance Monitor: Tracks extension interactions and system metrics
- Constitutional Engine: Validates compliance against six-article framework
- Rule Automation Engine: Automates constitutional validation workflows
- Telemetry Bridge: Collects real-time performance data
- Dashboard Interface: Provides visual monitoring and alerts

Constitutional Compliance:
- Article I: Library-First - Uses standard monitoring libraries and patterns
- Article II: Test-First - Comprehensive test coverage for all components
- Article III: Simplicity - Clean, maintainable architecture
- Article IV: Integration-First - Seamless integration with existing systems
- Article V: Clarity - Clear documentation and interfaces
- Article VI: Versioning - Proper change management and versioning
"""

from .core.constitutional_engine import ConstitutionalEngine
from .core.performance_monitor import PerformanceMonitor
from .core.telemetry_bridge import TelemetryBridge
from .dashboard.dashboard_interface import DashboardInterface
from .optimization import create_optimization_suite

__version__ = "1.0.0"
__author__ = "Super Alita Development Team"

__all__ = [
    "PerformanceMonitor",
    "TelemetryBridge", 
    "ConstitutionalEngine",
    "DashboardInterface",
    "create_optimization_suite",
]