# Super Alita Agent Detailed Monitoring

This document describes the enhanced monitoring system for capturing and analyzing Super Alita agent events with maximum fidelity.

## Overview

The detailed monitoring system provides:
- **Full payload capture** for all agent events
- **Millisecond-precision timestamps** for latency analysis
- **Structured JSONL logging** for automated analysis
- **Real-time event streaming** via Redis pub/sub
- **UTF-8 safe console output** for Windows environments
- **PowerShell integration** for interactive monitoring

## Components

### 1. Detailed Monitor (`monitor_agent_detailed.py`)

The core monitoring script that captures all agent events with full detail.

**Features:**
- Connects to Redis/Memurai for event streaming
- Logs to both structured JSONL and human-readable formats
- Environment-aware configuration (supports Memurai and Redis)
- Comprehensive channel subscription (18+ event types)
- Error-resilient with graceful degradation

**Usage:**
```bash
# Basic monitoring
python monitor_agent_detailed.py

# With environment variables
set MEMURAI_HOST=localhost
set MEMURAI_PORT=6379
python monitor_agent_detailed.py
```

**Output Locations:**
- `logs/agent_detailed_monitor.log` - Human-readable log
- `logs/telemetry.jsonl` - Structured JSON Lines format

### 2. PowerShell Tail Helper (`scripts/monitor_agent_tail.ps1`)

Interactive PowerShell script for real-time log monitoring with filtering.

**Features:**
- Color-coded output by log level
- Regex filtering for specific events
- Support for both log formats
- Configurable tail length and follow mode
- UTF-8 safe for Windows console

**Usage:**
```powershell
# Basic tailing
.\scripts\monitor_agent_tail.ps1

# Filter for tool events only
.\scripts\monitor_agent_tail.ps1 -Filter "tool_call|tool_result"

# Show errors only
.\scripts\monitor_agent_tail.ps1 -Filter "ERROR|FATAL"

# JSONL format only
.\scripts\monitor_agent_tail.ps1 -LogType jsonl
```

## Event Channels Monitored

The system monitors all critical Super Alita event channels:

| Channel | Description | Key Data |
|---------|-------------|----------|
| `conversation_message` | User interactions | message, user_id, timestamp |
| `tool_call` | Tool execution requests | tool_name, parameters, call_id |
| `tool_result` | Tool execution results | success, result, execution_time |
| `gap_detection` | Capability gap identification | missing_tool, description |
| `neural_atom_execution` | Neural atom processing | atom_name, input, output |
| `memory_operation` | Memory system operations | operation, data, success |
| `agent_status` | Agent state changes | status, context |
| `pythonic_preprocessor` | DTA preprocessing | input, output, confidence |
| `llm_planner` | LLM planning decisions | plan, confidence, tool_selection |
| `creator_plugin` | Tool creation pipeline | stage, tool_name, code |
| `memory_manager` | Memory management | operation, atom_id, metadata |
| `event_bus_health` | Event system status | throughput, latency, errors |
| `system_events` | System-level events | component, event, severity |
| `validation_events` | Validation results | test_name, result, details |
| `error_events` | Error conditions | error_type, message, stack_trace |
| `performance_metrics` | Performance data | metric_name, value, timestamp |
| `user_interaction` | UI/UX events | action, context, duration |
| `telemetry_broadcast` | Telemetry meta-events | source, metrics, health |

## Event Structure

Each event is captured with comprehensive metadata:

```json
{
  "timestamp": "2025-01-13 15:30:45.123",
  "timestamp_unix": 1705157445.123,
  "channel": "tool_call",
  "event_type": "TOOLCALL",
  "data": {
    "tool_name": "web_search",
    "parameters": {"query": "AI research"},
    "call_id": "call_abc123",
    "source_plugin": "llm_planner"
  },
  "metadata": {
    "monitor_version": "1.0",
    "redis_host": "127.0.0.1",
    "redis_port": 6379,
    "latency_tracking": true,
    "tool_name": "web_search"
  }
}
```

## Analysis Workflows

### Real-time Monitoring

1. **Start the detailed monitor:**
   ```bash
   python monitor_agent_detailed.py
   ```

2. **In another terminal, tail the logs:**
   ```powershell
   .\scripts\monitor_agent_tail.ps1
   ```

3. **Run the agent and interact:**
   ```bash
   python launch_super_alita.py
   ```

### Performance Analysis

Query the JSONL logs for performance insights:

```bash
# Tool execution times
cat logs/telemetry.jsonl | jq 'select(.channel=="tool_result") | {tool: .metadata.tool_name, time: .data.execution_time}'

# Error frequency
cat logs/telemetry.jsonl | jq 'select(.channel=="error_events") | .data.error_type' | sort | uniq -c

# Event throughput by channel
cat logs/telemetry.jsonl | jq -r '.channel' | sort | uniq -c | sort -nr
```

### Debugging Workflows

1. **Find specific events:**
   ```powershell
   .\scripts\monitor_agent_tail.ps1 -Filter "gap_detection"
   ```

2. **Track tool creation pipeline:**
   ```powershell
   .\scripts\monitor_agent_tail.ps1 -Filter "creator_plugin|gap_detection"
   ```

3. **Monitor memory operations:**
   ```powershell
   .\scripts\monitor_agent_tail.ps1 -Filter "memory_operation|memory_manager"
   ```

## Configuration

### Environment Variables

The monitor supports flexible configuration via environment variables:

```bash
# Memurai (preferred)
MEMURAI_HOST=127.0.0.1
MEMURAI_PORT=6379
MEMURAI_PASSWORD=your_password

# Redis (fallback)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_DB=0
```

### Log Rotation

For production use, configure log rotation:

```bash
# Example logrotate configuration
/path/to/logs/agent_detailed_monitor.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 user group
}
```

## Integration with Development Workflow

### VS Code Integration

Add tasks to `.vscode/tasks.json`:

```json
{
    "label": "Monitor: Start Detailed",
    "type": "shell",
    "command": "python",
    "args": ["monitor_agent_detailed.py"],
    "group": "test",
    "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "new"
    }
},
{
    "label": "Monitor: Tail Logs",
    "type": "shell",
    "command": "pwsh",
    "args": ["-File", "scripts/monitor_agent_tail.ps1"],
    "group": "test"
}
```

### PowerShell Profile Integration

Add to your PowerShell profile:

```powershell
# Super Alita monitoring aliases
function Start-AgentMonitor { python monitor_agent_detailed.py }
function Tail-AgentLogs { .\scripts\monitor_agent_tail.ps1 @args }

Set-Alias -Name "monitor" -Value Start-AgentMonitor
Set-Alias -Name "tail-agent" -Value Tail-AgentLogs
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Ensure Redis/Memurai is running: `redis-cli ping`
   - Check host/port configuration
   - Verify firewall settings

2. **No Events Captured**
   - Verify agent is publishing to correct channels
   - Check event bus configuration in agent
   - Confirm Redis pub/sub is working: `redis-cli monitor`

3. **Unicode/Encoding Issues**
   - Monitor script includes UTF-8 hardening
   - PowerShell script uses ASCII fallback
   - Check console encoding settings

4. **Log Files Not Created**
   - Ensure `logs/` directory exists
   - Check file permissions
   - Verify disk space

### Debug Commands

```bash
# Test Redis connection
redis-cli ping

# Monitor Redis pub/sub activity
redis-cli monitor

# Check log file creation
ls -la logs/

# Test PowerShell script
pwsh -File scripts/monitor_agent_tail.ps1 -Lines 10 -Follow false
```

## Best Practices

1. **Start monitoring before agent**: Capture startup events
2. **Use filters for focus**: Avoid information overload
3. **Regular log rotation**: Prevent disk space issues
4. **Backup JSONL files**: For historical analysis
5. **Monitor performance impact**: Detailed logging has overhead
6. **Use structured queries**: Leverage JSONL format for analysis

This comprehensive monitoring system provides maximum visibility into Super Alita agent behavior, enabling detailed debugging, performance analysis, and system optimization.
