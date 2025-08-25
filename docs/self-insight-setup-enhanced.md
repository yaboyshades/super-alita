# Super Alita Self-Insight Setup Guide - Enhanced Edition

## 🎯 Overview

Super Alita is a production-ready autonomous agent system with comprehensive self-insight capabilities, concurrency safety, and intelligent planning. This guide covers setup, configuration, and monitoring of the complete system.

## 🏗️ Architecture Components

### Core Systems
- **FSM (Finite State Machine)**: Handles agent state transitions with concurrency safety
- **Knowledge Graph API**: Manages persistent knowledge and learning insights
- **Planning Engine**: Metrics-driven todo prioritization with anti-thrash protection
- **Circuit Breaker**: Protects against overload with automatic recovery
- **Telemetry System**: Comprehensive metrics and monitoring

### New Features (Enhanced)
- **Mailbox Queuing**: Re-entrant input handling with pressure monitoring
- **Stale Detection**: Identifies and handles outdated operations
- **Risk Scoring**: Unified priority mapping with smoothing algorithms
- **Learning Velocity**: Adaptive feedback based on performance trends
- **Degraded Mode**: Automatic protection under stress conditions

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Clone and setup
git clone <repository>
cd super-alita

# Install dependencies (CPU-only)
pip install -r requirements.txt
pip install -r requirements-test.txt

# Add GPU acceleration
# pip install -r requirements-gpu.txt

# Verify environment
python final_integration.py
```

### 2. Start Core Services

```powershell
# Method 1: VS Code Tasks (Recommended)
# Open VS Code and use:
# - Ctrl+Shift+P -> "Tasks: Run Task" -> "Start KG API Server"
# - Ctrl+Shift+P -> "Tasks: Run Task" -> "Launch Agent"

# Method 2: Manual Terminal Commands
# Terminal 1: KG API Server
python start_kg_api.py

# Terminal 2: Agent System
python src/main.py

# Terminal 3: Metrics Sync (Optional)
python src/planning/sync_once.py
```

### 3. Verify System Health

```powershell
# Check API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Run system validation
python final_integration.py

# Test metrics pipeline
python src/planning/sync_once.py
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# Use: config/prometheus/super-alita-prometheus-enhanced.yml
scrape_configs:
  - job_name: 'super-alita-kg-api'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 10s

rule_files:
  - "super-alita-alerts.yml"
```

### Grafana Dashboard

Import `config/grafana/super-alita-dashboard-enhanced.json` for comprehensive monitoring:

**Key Panels:**
- System Health Overview
- FSM State Distribution
- Mailbox Pressure Gauge
- Concurrency Metrics Timeline
- Circuit Breaker Status
- Planning & Risk Metrics
- Learning Velocity
- Decision Confidence Distribution
- Tool Performance
- Error & Fallback Rates

### Critical Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| `sa_fsm_mailbox_pressure` | > 0.8 | Scale processing or enable degraded mode |
| `sa_planning_risk_score` | > 0.8 | Review active todos and priorities |
| `sa_fsm_circuit_breaker_open` | = 1 | Investigate cause, reduce load |
| `sa_agent_learning_velocity` | < 0.2 | Provide more stimulating tasks |
| `sa_fsm_stale_completions_total` | High rate | Check for timing issues |

## 🛡️ Concurrency & Safety Features

### Circuit Breaker Protection

**Automatic Protection Against:**
- Mailbox overflow (>100 items)
- Transition rate limiting (>10/second)
- Excessive failures
- Resource exhaustion

**Configuration:**
```python
MAILBOX_MAX_SIZE = 100
TRANSITION_RATE_LIMIT = 10
CIRCUIT_BREAKER_TIMEOUT = 30  # seconds
```

### Fallback Behavior

**Triggers:**
- No applicable tools found
- Tool execution failures
- Timeout conditions
- Circuit breaker activation

**Response Strategy:**
1. Generate contextual fallback response
2. Log event for learning
3. Update metrics
4. Maintain user experience

### Anti-Thrash Protection

**Mechanisms:**
- **Hysteresis**: Prevents oscillating decisions
- **Debounce**: Delays rapid state changes
- **Deduplication**: Removes duplicate alerts/todos
- **Smoothing**: Uses EWMA for trend analysis

## 🧪 Testing & Validation

### Concurrency Tests

```powershell
# Run all concurrency tests
python run_concurrency_tests.py

# Run by category
python run_concurrency_tests.py --categories

# Test specific scenarios
python -m pytest tests/core/test_concurrency.py::TestFallbackBehavior -v
python -m pytest tests/core/test_concurrency.py::TestCircuitBreaker -v
```

### Performance Validation

```powershell
# System integration test
python final_integration.py

# Anti-thrash demonstration
python src/planning/sync_once.py --demo

# Metrics pipeline test
python src/planning/sync_once.py
```

### Load Testing

```powershell
# High concurrency simulation
python tests/core/test_concurrency.py TestIntegrationScenarios::test_high_load_integration

# Mailbox pressure test
python tests/core/test_concurrency.py TestReEntrantInput::test_mailbox_pressure_under_load
```

## 🔧 Configuration Reference

### Environment Variables

```bash
# Core Configuration
SUPER_ALITA_LOG_LEVEL=INFO
SUPER_ALITA_DEBUG=false

# API Configuration
KG_API_HOST=localhost
KG_API_PORT=8000

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_EXPORT_INTERVAL=15

# Concurrency Settings
MAX_CONCURRENT_OPERATIONS=5
MAILBOX_SIZE_LIMIT=100
CIRCUIT_BREAKER_ENABLED=true

# Planning Configuration
RISK_SCORE_THRESHOLD=0.8
TODO_PRIORITY_LEVELS=4
AUTO_ESCALATION_ENABLED=true
```

### Advanced Configuration

```python
# src/core/config.py
CIRCUIT_BREAKER_CONFIG = {
    "mailbox_max_size": 100,
    "transition_rate_limit": 10,
    "timeout_seconds": 30,
    "failure_threshold": 5
}

PLANNING_CONFIG = {
    "risk_weights": {
        "mailbox_pressure": 0.4,
        "stale_rate": 0.3,
        "error_rate": 0.3
    },
    "smoothing_alpha": 0.3,
    "trend_window": 10
}
```

## 🚨 Troubleshooting Guide

### Common Issues

**1. High Mailbox Pressure**
```
Symptoms: sa_fsm_mailbox_pressure > 0.8
Causes: Excessive input rate, slow processing, resource constraints
Solutions:
- Enable degraded mode
- Scale processing capacity
- Implement input rate limiting
- Check for resource bottlenecks
```

**2. Circuit Breaker Tripping**
```
Symptoms: sa_fsm_circuit_breaker_open = 1
Causes: System overload, rapid failures, resource exhaustion
Solutions:
- Reduce input load
- Check system resources
- Review error logs
- Wait for automatic reset (30s default)
```

**3. Low Learning Velocity**
```
Symptoms: sa_agent_learning_velocity < 0.2
Causes: Insufficient stimulation, repetitive tasks, system issues
Solutions:
- Provide diverse, challenging tasks
- Check knowledge graph connectivity
- Review decision confidence trends
- Validate feedback mechanisms
```

**4. High Stale Completion Rate**
```
Symptoms: High rate of sa_fsm_stale_completions_total
Causes: Timing issues, race conditions, slow responses
Solutions:
- Check operation timeouts
- Review concurrency settings
- Validate state transition logic
- Monitor system performance
```

### Debug Commands

```powershell
# System health check
python -c "from src.core.kg_api_server import app; print('API OK')"

# FSM state inspection
python -c "from src.core.states import StateMachine; print('FSM OK')"

# Metrics registry check
python -c "from src.core.metrics_registry import get_metrics_registry; print(get_metrics_registry().get_all_metrics())"

# Circuit breaker status
python -c "
from src.core.states import StateMachine
from src.core.session import get_session
from src.core.metrics_registry import get_metrics_registry
fsm = StateMachine(get_session('debug'), get_metrics_registry())
print(f'Circuit breaker open: {fsm.circuit_breaker.is_open}')
print(f'Failure count: {fsm.circuit_breaker.failure_count}')
"
```

### Log Analysis

**Key Log Patterns to Watch:**
```
# Circuit breaker events
"Circuit breaker tripped: mailbox_overflow"
"Circuit breaker reset"

# Concurrency issues
"Mailbox size * approaching limit"
"Stale completion detected"
"Re-entrant input queued"

# Performance warnings
"High transition rate detected"
"Fallback response generated"
"Planning risk score elevated"
```

## 📈 Performance Optimization

### Scaling Recommendations

**Low Load (< 10 req/min):**
- Single instance sufficient
- Basic monitoring
- 30s circuit breaker timeout

**Medium Load (10-100 req/min):**
- Enable all concurrency features
- 15s monitoring intervals
- Degraded mode planning

**High Load (> 100 req/min):**
- Load balancing consideration
- Reduced circuit breaker timeout (15s)
- Aggressive anti-thrash settings
- SLO monitoring

### Resource Planning

```yaml
# Minimum Requirements
CPU: 2 cores
Memory: 4GB
Disk: 10GB
Network: 100Mbps

# Recommended Production
CPU: 4+ cores
Memory: 8GB+
Disk: 50GB SSD
Network: 1Gbps
Database: PostgreSQL/Neo4j cluster
```

## 🔮 Advanced Features

### Multi-Agent Coordination (Future)
- Shared event bus with isolation
- Agent-to-agent communication
- Distributed consensus protocols
- Cross-agent learning

### Machine Learning Integration (Future)
- Anomaly detection for metrics
- Predictive health scoring
- Auto-tuning thresholds
- Reinforcement learning feedback

### Neo4j Knowledge Graph (Optional)
```python
# Enable with feature flag
NEO4J_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## 📋 Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Database connections tested
- [ ] Prometheus/Grafana configured
- [ ] Alert rules deployed
- [ ] Circuit breaker thresholds set
- [ ] Monitoring dashboards imported
- [ ] Load testing completed
- [ ] Backup procedures established
- [ ] Log rotation configured
- [ ] SSL certificates installed (production)
- [ ] Security scanning completed

## 🎯 Success Metrics

**Operational Excellence:**
- Uptime > 99.9%
- Response time P95 < 2s
- Error rate < 0.1%
- Circuit breaker trips < 1/day

**Learning Effectiveness:**
- Learning velocity > 0.5
- Decision confidence > 0.7
- Knowledge graph growth rate > 0
- Todo completion rate > 80%

**System Health:**
- Mailbox pressure < 0.5
- Risk score < 0.6
- Stale completion rate < 0.05
- Resource utilization < 70%

---

## 🆘 Support & Contact

For issues, questions, or contributions:
- GitHub Issues: [Repository Issues](link)
- Documentation: This file and `/docs/` folder
- Monitoring: Grafana dashboards
- Logs: Check system logs with debug commands above

**System Status: PRODUCTION-READY**
**Last Updated: {{ current_date }}**
**Version: Enhanced Edition with Concurrency Safety**
