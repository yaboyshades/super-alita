"""
Performance Monitoring System - Optimization Module

This module provides performance optimization, scalability
enhancements, and security hardening features for the
constitutional performance monitoring system.

Components:
- Performance Optimizer: Caching layers and connection pooling
- Scalability Manager: Horizontal scaling and load balancing
- Security Hardening: Encryption, authentication, and audit logging
- Resource Manager: Memory optimization and cleanup
"""

from .performance_optimizer import PerformanceOptimizer
from .resource_manager import ResourceManager
from .scalability_manager import ScalabilityManager
from .security_hardening import SecurityHardening

__all__ = [
    'PerformanceOptimizer',
    'ScalabilityManager',
    'SecurityHardening',
    'ResourceManager'
]


async def create_optimization_suite(config=None):
    """Create and configure complete optimization suite"""

    # Initialize all optimization components
    performance_optimizer = PerformanceOptimizer()
    scalability_manager = ScalabilityManager()
    security_hardening = SecurityHardening()
    resource_manager = ResourceManager()

    # Start all services
    await performance_optimizer.start()
    await scalability_manager.start()
    await security_hardening.start()
    await resource_manager.start()

    return {
        'performance': performance_optimizer,
        'scalability': scalability_manager,
        'security': security_hardening,
        'resources': resource_manager
    }