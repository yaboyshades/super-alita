
# Planning Metrics

<cite>
**Referenced Files in This Document**   
- [planning_metrics.py](file://src/core/planning_metrics.py)
- [metrics_registry.py](file://src/core/metrics_registry.py)
- [cortex_adapter.py](file://src/ladder/integration/cortex_adapter.py)
- [energy_enhanced_adapter.py](file://src/ladder/prioritization/energy_enhanced_adapter.py)
- [telemetry.py](file://src/ecosystem/telemetry.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Metrics Domain Model](#metrics-domain-model)
3. [Data Collection Mechanisms](#data-collection-mechanisms)
4. [Storage and Aggregation](#storage-and-aggregation)
5. [Integration with Planning System](#integration-with-planning-system)
6. [Telemetry System Integration](#telemetry-system-integration)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
The Planning Metrics system in the Super Alita framework provides real-time operational insights that directly influence planning decisions and todo prioritization. This system collects key performance indicators from various components and transforms them into actionable intelligence for the planning engine. The metrics are designed to identify system health issues, performance bottlenecks, and optimization opportunities that affect planning efficiency and reliability.

The implementation focuses on integrating operational metrics into the planning workflow, allowing the agent to make informed decisions about priorities and resource allocation based on real-time system performance data. This documentation details the architecture, implementation, and usage patterns of the planning metrics system, providing guidance for both beginners and experienced developers.

## Metrics Domain Model

The planning metrics system is built around a comprehensive domain model that captures various aspects of system performance and health. The core metrics are organized into logical categories that reflect different dimensions of system operation.

### Core Metric Categories

```mermaid
classDiagram
class PlanningMetricsProvider {
+get_planning_priority_metrics() dict[str, Any]
+get_todo_integration_summary() dict[str, Any]
+suggest_todo_priorities(current_todos) list[dict[str, Any]]
-_derive_planning_implications(mailbox_pressure, stale_rate, concurrency_load, ignored_triggers) dict[str, Any]
-_get_system_status_summary(health_metrics) str
}
class SystemHealthMetrics {
+mailbox_pressure : float
+stale_completion_rate : float
+concurrency_load : float
+ignored_trigger_count : int
}
class PerformanceIndicators {
+current_mailbox_size : int
+peak_mailbox_size : int
+active_operations : int
+total_operations_processed : int
}
class PlanningImplications {
+priority_adjustments : list[dict]
+suggested_actions : list[str]
+risk_factors : list[dict]
+performance_trends : str
}
PlanningMetricsProvider --> SystemHealthMetrics : "includes"
PlanningMetricsProvider --> PerformanceIndicators : "includes"
PlanningMetricsProvider --> PlanningImplications : "includes"
PlanningMetricsProvider --> MetricsRegistry : "uses"
```

**Diagram sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L17-L297)

**Section sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L17-L297)

### Key Planning Metrics

The system measures several critical aspects of planning performance and system health:

| Metric Category | Specific Metrics | Purpose | Data Type |
|----------------|------------------|---------|---------|
| **System Health** | mailbox_pressure | Measures input queue pressure (0.0-1.0) | float |
| | stale_completion_rate | Tracks concurrency issues from stale completions | float |
| | concurrency_load | Indicates system load level | float |
| | ignored_trigger_count | Counts ignored triggers that may indicate design issues | int |
| **Performance Indicators** | current_mailbox_size | Current number of items in FSM mailbox | int |
| | peak_mailbox_size | Maximum mailbox size observed | int |
| | active_operations | Number of currently active operations | int |
| | total_operations_processed | Total operations processed by FSM | int |
| **Planning Implications** | priority_adjustments | Suggested priority changes based on metrics | list[dict] |
| | suggested_actions | Recommended actions to address issues | list[str] |
| | risk_factors | Identified risk factors with impact assessment | list[dict] |
| | performance_trends | Overall trend (stable, improving, degrading) | str |

These metrics are derived from lower-level system metrics collected by the metrics registry and transformed into planning-relevant insights. The system uses these metrics to assess the overall health of the planning process and identify areas that require attention or optimization.

## Data Collection Mechanisms

The planning metrics system employs a multi-layered approach to data collection, gathering information from various sources across the Super Alita framework.

### Metrics Collection Flow

```mermaid
sequenceDiagram
participant Registry as MetricsRegistry
participant Provider as PlanningMetricsProvider
participant Collector as Metric Collectors
participant System as System Components
System->>Registry : Update counters and gauges
Registry->>Registry : Store metrics in thread-safe collections
Provider->>Registry : Request specific metrics
Registry-->>Provider : Return current metric values
Provider->>Provider : Calculate derived metrics
Provider->>Provider : Generate planning implications
Provider-->>Planning System : Provide integrated metrics summary
```

**Diagram sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L34-L81)
- [metrics_registry.py](file://src/core/metrics_registry.py#L0-L194)

**Section sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L34-L81)
- [metrics_registry.py](file://src/core/metrics_registry.py#L0-L194)

### Primary Data Sources

The planning metrics are primarily collected from the Finite State Machine (FSM) components, which serve as the central coordination mechanism in the Super Alita framework. The key metrics sources include:

- **FSM Mailbox**: Tracks the size and pressure of the input queue, indicating how quickly the system is processing incoming requests.
- **Operation Lifecycle**: Monitors active operations and completion rates, identifying potential concurrency issues.
- **Event Processing**: Measures trigger handling and identifies ignored triggers that may indicate design problems.
- **System Resources**: Collects CPU, memory, and other system-level metrics that affect planning performance.

The `PlanningMetricsProvider` class serves as the central collection point, retrieving raw metrics from the `MetricsRegistry` and transforming them into planning-relevant insights. This provider uses the `get_metrics_registry()` function to access the global metrics registry instance, ensuring consistent and thread-safe access to metric data.

### Derived Metrics Calculation

The system calculates several derived metrics that provide deeper insights into planning performance:

```mermaid
flowchart TD
A[Raw Metrics] --> B[Derived Metrics]
B --> C[Planning Implications]
subgraph Raw Metrics
A1[mailbox_size]
A2[mailbox_max]
A3[active_ops]
A4[ignored_triggers]
A5[stale_completions]
A6[total_operations]
end
subgraph Derived Metrics
B1[mailbox_pressure = mailbox_size / mailbox_max]
B2[stale_rate = stale_completions / total_operations]
B3[concurrency_load = active_ops / 5.0]
end
subgraph Planning Implications
C1[High mailbox pressure → concurrency optimization]
C2[High stale rate → FSM stability improvements]
C3[High concurrency load → resource management]
C4[Many ignored triggers → design review]
end
A1 --> B1
A2 --> B1
A5 --> B2
A6 --> B2
A3 --> B3
B1 --> C1
B2 --> C2
B3 --> C3
A4 --> C4
```

**Diagram sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L34-L81)
- [planning_metrics.py](file://src/core/planning_metrics.py#L83-L160)

**Section sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L83-L160)

The `_derive_planning_implications` method analyzes these derived metrics and generates actionable recommendations for the planning system. For example, when mailbox pressure exceeds 0.7, the system suggests prioritizing concurrency improvements. Similarly, a stale completion rate above 0.1 triggers recommendations for FSM stability improvements.

## Storage and Aggregation

The planning metrics system implements a sophisticated storage and aggregation strategy to maintain historical context while providing real-time insights.

### Metrics Storage Architecture

```mermaid
classDiagram
class MetricsRegistry {
+_counters : dict[str, int]
+_gauges : dict[str, float]
+_histograms : dict[str, dict[str, Any]]
+_data_lock : RLock
+increment_counter(name, value, labels)
+set_gauge(name, value, labels)
+observe_histogram(name, value, labels)
+get_prometheus_metrics() str
+get_counter(name, labels) int
+get_gauge(name, labels) float
}
class PlanningMetricsProvider {
+registry : MetricsRegistry
+_last_metrics_snapshot : dict
+_trend_history : list[dict]
+get_planning_priority_metrics() dict[str, Any]
}
MetricsRegistry <|-- PlanningMetricsProvider : "uses"
MetricsRegistry --> ThreadSafety : "implements"
```

**Diagram sources**
- [metrics_registry.py](file://src/core/metrics_registry.py#L0-L194)
- [planning_metrics.py](file://src/core/planning_metrics.py#L17-L297)

**Section sources**
- [metrics_registry.py](file://src/core/metrics_registry.py#L0-L194)

### Data Retention and History

The system maintains a rolling history of metric snapshots to enable trend analysis and historical comparisons. The `PlanningMetricsProvider` stores up to 100 recent metric snapshots in its `_trend_history` list, automatically removing older entries when the limit is exceeded.

```python
# Store for trend analysis
self._trend_history.append(planning_metrics)
if len(self._trend_history) > 100:  # Keep last 100 snapshots
    self._trend_history.pop(0)
```

This historical data enables the system to detect performance trends by comparing recent metrics with past values. For example, the system can identify whether mailbox pressure is increasing, decreasing, or remaining stable over time, providing valuable context for planning decisions.

### Aggregation Strategies

The metrics registry implements several aggregation strategies to handle different types of metrics:

- **Counters**: Monotonically increasing values that track the number of events (e.g., total operations processed).
- **Gauges**: Point-in-time measurements that can go up and down (e.g., current mailbox size).
- **Histograms**: Distribution of values across predefined buckets, useful for understanding latency distributions.

The registry uses thread-safe data structures and locking mechanisms to ensure data consistency in multi-threaded environments. The `threading.RLock` ensures that metric operations are atomic and prevents race conditions when multiple components access the registry simultaneously.

## Integration with Planning System

The planning metrics are tightly integrated with the planning system, directly influencing todo prioritization and planning decisions.

### Todo List Enhancement

The system provides functions to enhance todo lists with metrics-driven insights, making performance data actionable for the planning engine.

```mermaid
sequenceDiagram
participant TodoSystem as Todo Management
participant Metrics as PlanningMetricsProvider
participant Enhanced as Enhanced Todos
TodoSystem->>Metrics : get_todo_integration_summary()
Metrics-->>TodoSystem : Return integration summary
TodoSystem->>TodoSystem : Add system status to in-progress items
alt Critical or Warning Status
TodoSystem->>TodoSystem : Create metric-driven todos
end
TodoSystem-->>Enhanced : Return enhanced todo list
```

**Diagram sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L249-L284)
- [planning_metrics.py](file://src/core/planning_metrics.py#L162-L178)

**Section sources**
- [planning_metrics.py](file://src/core/planning_metrics.py#L249