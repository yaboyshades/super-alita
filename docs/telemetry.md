# Ecosystem Telemetry

The ecosystem includes a lightweight, dependency-free telemetry system for performance monitoring and observability.

## Components

-   **`Telemetry` Class:** The main entry point for recording telemetry data. It is designed to be injected into components like the `EcosystemOrchestrator`.
-   **`IMetricsSink` Protocol:** An interface for backend systems (like Prometheus or OpenTelemetry) that receive the telemetry data. A `NoopMetricsSink` is provided as a safe default.

## Usage

### Timing Code Blocks

The primary way to measure performance is with the `timer` context manager.

```python
telemetry = Telemetry()

with telemetry.timer("database.query.duration_ms", tags={"query_type": "select"}):
    # Code to be measured
    run_database_query()
```

### Incrementing Counters

You can also increment counters for tracking event occurrences.

```python
telemetry.increment_counter("files.processed", tags={"file_type": ".py"})
```

This simple system provides the essential hooks for production-grade observability without introducing heavy dependencies initially.