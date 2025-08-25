"""
Prometheus Metrics Collector for Super Alita Agent System

Provides comprehensive metrics collection and exposure for monitoring
the agent system using Prometheus/OpenMetrics format.
"""

import time
from typing import Dict, List, Optional, Any
import threading
from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge, 
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)


class PrometheusMetricsCollector:
    """
    Prometheus metrics collector for Super Alita Agent system.
    
    Collects and exposes metrics for all agent components including:
    - Cortex processing cycles
    - Knowledge graph operations
    - Optimization decisions
    - System health and performance
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector with optional custom registry."""
        self.registry = registry or REGISTRY
        self._lock = threading.Lock()
        self._initialized = False
        
        # Metrics will be initialized on first use
        self._metrics: Dict[str, Any] = {}
        
    def _ensure_initialized(self) -> None:
        """Ensure metrics are initialized (thread-safe)."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self._initialize_metrics()
            self._initialized = True
    
    def _initialize_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        
        # System metrics
        self._metrics['system_info'] = Info(
            'super_alita_system', 
            'Super Alita system information',
            registry=self.registry
        )
        
        self._metrics['system_uptime'] = Gauge(
            'super_alita_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self._metrics['component_health'] = Gauge(
            'super_alita_component_health',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        # Cortex metrics
        self._metrics['cortex_cycles_total'] = Counter(
            'super_alita_cortex_cycles_total',
            'Total number of Cortex processing cycles',
            ['session_id', 'status'],
            registry=self.registry
        )
        
        self._metrics['cortex_cycle_duration'] = Histogram(
            'super_alita_cortex_cycle_duration_seconds',
            'Time spent processing Cortex cycles',
            ['session_id', 'module'],
            registry=self.registry
        )
        
        self._metrics['cortex_active_sessions'] = Gauge(
            'super_alita_cortex_active_sessions',
            'Number of active Cortex sessions',
            registry=self.registry
        )
        
        # Event metrics
        self._metrics['events_emitted_total'] = Counter(
            'super_alita_events_emitted_total',
            'Total number of events emitted',
            ['event_type', 'source_plugin'],
            registry=self.registry
        )
        
        self._metrics['events_processed_total'] = Counter(
            'super_alita_events_processed_total',
            'Total number of events processed',
            ['event_type', 'handler'],
            registry=self.registry
        )
        
        # Knowledge graph metrics
        self._metrics['knowledge_atoms_total'] = Gauge(
            'super_alita_knowledge_atoms_total',
            'Total number of atoms in knowledge graph',
            ['atom_type'],
            registry=self.registry
        )
        
        self._metrics['knowledge_bonds_total'] = Gauge(
            'super_alita_knowledge_bonds_total',
            'Total number of bonds in knowledge graph',
            ['bond_type'],
            registry=self.registry
        )
        
        self._metrics['knowledge_operations_total'] = Counter(
            'super_alita_knowledge_operations_total',
            'Total knowledge graph operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self._metrics['knowledge_operation_duration'] = Histogram(
            'super_alita_knowledge_operation_duration_seconds',
            'Time spent on knowledge graph operations',
            ['operation_type'],
            registry=self.registry
        )
        
        # Optimization metrics
        self._metrics['optimization_policies_total'] = Gauge(
            'super_alita_optimization_policies_total',
            'Total number of optimization policies',
            registry=self.registry
        )
        
        self._metrics['optimization_decisions_total'] = Counter(
            'super_alita_optimization_decisions_total',
            'Total optimization decisions made',
            ['policy_id', 'algorithm', 'arm_id'],
            registry=self.registry
        )
        
        self._metrics['optimization_rewards_total'] = Counter(
            'super_alita_optimization_rewards_total',
            'Total rewards received',
            ['policy_id', 'arm_id'],
            registry=self.registry
        )
        
        self._metrics['optimization_reward_value'] = Histogram(
            'super_alita_optimization_reward_value',
            'Reward values received',
            ['policy_id', 'arm_id'],
            registry=self.registry
        )
        
        self._metrics['optimization_arm_performance'] = Gauge(
            'super_alita_optimization_arm_performance',
            'Performance metrics for bandit arms',
            ['policy_id', 'arm_id', 'metric'],
            registry=self.registry
        )
        
        # Plugin metrics
        self._metrics['plugins_loaded_total'] = Gauge(
            'super_alita_plugins_loaded_total',
            'Total number of loaded plugins',
            ['plugin_type'],
            registry=self.registry
        )
        
        self._metrics['plugin_execution_total'] = Counter(
            'super_alita_plugin_execution_total',
            'Total plugin executions',
            ['plugin_name', 'method', 'status'],
            registry=self.registry
        )
        
        self._metrics['plugin_execution_duration'] = Histogram(
            'super_alita_plugin_execution_duration_seconds',
            'Time spent in plugin executions',
            ['plugin_name', 'method'],
            registry=self.registry
        )
        
        # gRPC metrics
        self._metrics['grpc_requests_total'] = Counter(
            'super_alita_grpc_requests_total',
            'Total gRPC requests',
            ['method', 'status'],
            registry=self.registry
        )
        
        self._metrics['grpc_request_duration'] = Histogram(
            'super_alita_grpc_request_duration_seconds',
            'Time spent processing gRPC requests',
            ['method'],
            registry=self.registry
        )
        
        # Set initial system info
        self._metrics['system_info'].info({
            'version': '2.0.0',
            'component': 'super_alita_agent',
            'python_version': '3.13'
        })
    
    # System metrics methods
    
    def set_system_uptime(self, uptime_seconds: float) -> None:
        """Set system uptime."""
        self._ensure_initialized()
        self._metrics['system_uptime'].set(uptime_seconds)
    
    def set_component_health(self, component: str, healthy: bool) -> None:
        """Set component health status."""
        self._ensure_initialized()
        self._metrics['component_health'].labels(component=component).set(1 if healthy else 0)
    
    # Cortex metrics methods
    
    def inc_cortex_cycles(self, session_id: str, status: str = "success") -> None:
        """Increment Cortex cycle counter."""
        self._ensure_initialized()
        self._metrics['cortex_cycles_total'].labels(session_id=session_id, status=status).inc()
    
    def observe_cortex_cycle_duration(self, session_id: str, module: str, duration: float) -> None:
        """Record Cortex cycle duration."""
        self._ensure_initialized()
        self._metrics['cortex_cycle_duration'].labels(session_id=session_id, module=module).observe(duration)
    
    def set_cortex_active_sessions(self, count: int) -> None:
        """Set number of active Cortex sessions."""
        self._ensure_initialized()
        self._metrics['cortex_active_sessions'].set(count)
    
    # Event metrics methods
    
    def inc_events_emitted(self, event_type: str, source_plugin: str) -> None:
        """Increment events emitted counter."""
        self._ensure_initialized()
        self._metrics['events_emitted_total'].labels(event_type=event_type, source_plugin=source_plugin).inc()
    
    def inc_events_processed(self, event_type: str, handler: str) -> None:
        """Increment events processed counter."""
        self._ensure_initialized()
        self._metrics['events_processed_total'].labels(event_type=event_type, handler=handler).inc()
    
    # Knowledge graph metrics methods
    
    def set_knowledge_atoms(self, atom_type: str, count: int) -> None:
        """Set knowledge graph atom count."""
        self._ensure_initialized()
        self._metrics['knowledge_atoms_total'].labels(atom_type=atom_type).set(count)
    
    def set_knowledge_bonds(self, bond_type: str, count: int) -> None:
        """Set knowledge graph bond count."""
        self._ensure_initialized()
        self._metrics['knowledge_bonds_total'].labels(bond_type=bond_type).set(count)
    
    def inc_knowledge_operations(self, operation_type: str, status: str = "success") -> None:
        """Increment knowledge operations counter."""
        self._ensure_initialized()
        self._metrics['knowledge_operations_total'].labels(operation_type=operation_type, status=status).inc()
    
    def observe_knowledge_operation_duration(self, operation_type: str, duration: float) -> None:
        """Record knowledge operation duration."""
        self._ensure_initialized()
        self._metrics['knowledge_operation_duration'].labels(operation_type=operation_type).observe(duration)
    
    # Optimization metrics methods
    
    def set_optimization_policies(self, count: int) -> None:
        """Set optimization policies count."""
        self._ensure_initialized()
        self._metrics['optimization_policies_total'].set(count)
    
    def inc_optimization_decisions(self, policy_id: str, algorithm: str, arm_id: str) -> None:
        """Increment optimization decisions counter."""
        self._ensure_initialized()
        self._metrics['optimization_decisions_total'].labels(
            policy_id=policy_id, algorithm=algorithm, arm_id=arm_id
        ).inc()
    
    def inc_optimization_rewards(self, policy_id: str, arm_id: str) -> None:
        """Increment optimization rewards counter."""
        self._ensure_initialized()
        self._metrics['optimization_rewards_total'].labels(policy_id=policy_id, arm_id=arm_id).inc()
    
    def observe_optimization_reward_value(self, policy_id: str, arm_id: str, reward: float) -> None:
        """Record optimization reward value."""
        self._ensure_initialized()
        self._metrics['optimization_reward_value'].labels(policy_id=policy_id, arm_id=arm_id).observe(reward)
    
    def set_optimization_arm_performance(self, policy_id: str, arm_id: str, metric: str, value: float) -> None:
        """Set optimization arm performance metric."""
        self._ensure_initialized()
        self._metrics['optimization_arm_performance'].labels(
            policy_id=policy_id, arm_id=arm_id, metric=metric
        ).set(value)
    
    # Plugin metrics methods
    
    def set_plugins_loaded(self, plugin_type: str, count: int) -> None:
        """Set loaded plugins count."""
        self._ensure_initialized()
        self._metrics['plugins_loaded_total'].labels(plugin_type=plugin_type).set(count)
    
    def inc_plugin_execution(self, plugin_name: str, method: str, status: str = "success") -> None:
        """Increment plugin execution counter."""
        self._ensure_initialized()
        self._metrics['plugin_execution_total'].labels(
            plugin_name=plugin_name, method=method, status=status
        ).inc()
    
    def observe_plugin_execution_duration(self, plugin_name: str, method: str, duration: float) -> None:
        """Record plugin execution duration."""
        self._ensure_initialized()
        self._metrics['plugin_execution_duration'].labels(
            plugin_name=plugin_name, method=method
        ).observe(duration)
    
    # gRPC metrics methods
    
    def inc_grpc_requests(self, method: str, status: str) -> None:
        """Increment gRPC requests counter."""
        self._ensure_initialized()
        self._metrics['grpc_requests_total'].labels(method=method, status=status).inc()
    
    def observe_grpc_request_duration(self, method: str, duration: float) -> None:
        """Record gRPC request duration."""
        self._ensure_initialized()
        self._metrics['grpc_request_duration'].labels(method=method).observe(duration)
    
    # Metrics exposure methods
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        self._ensure_initialized()
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get the content type for metrics."""
        return CONTENT_TYPE_LATEST
    
    def get_metric_families(self):
        """Get metric families for gRPC response."""
        self._ensure_initialized()
        return self.registry.collect()
    
    def collect_system_metrics(self, start_time: float) -> None:
        """Collect and update system-level metrics."""
        self._ensure_initialized()
        
        # Update uptime
        uptime = time.time() - start_time
        self.set_system_uptime(uptime)
    
    def collect_plugin_metrics(self, plugins: Dict[str, Any]) -> None:
        """Collect metrics from plugin registry."""
        self._ensure_initialized()
        
        # Count plugins by type
        plugin_counts: Dict[str, int] = {}
        for plugin in plugins.values():
            plugin_type = plugin.__class__.__module__.split('.')[-1]
            plugin_counts[plugin_type] = plugin_counts.get(plugin_type, 0) + 1
        
        for plugin_type, count in plugin_counts.items():
            self.set_plugins_loaded(plugin_type, count)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        self._ensure_initialized()
        
        try:
            # Test metric collection
            test_time = time.time()
            self.set_system_uptime(test_time)
            
            return {
                "status": "healthy",
                "metrics_count": len(self._metrics),
                "registry": str(self.registry),
                "timestamp": test_time
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }