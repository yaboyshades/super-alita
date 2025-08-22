# Super Alita Self-Insight Setup & Usage Guide

## Overview

This setup guide helps you deploy Super Alita with its enhanced self-insight capabilities, including telemetry, learning velocity tracking, hypothesis lifecycle management, personality drift monitoring, and cross-session insight consolidation.

## Architecture Components

### 1. Self-Insight Loop
- **Decision Confidence**: Tracks agent confidence in tool choices
- **Learning Velocity**: Measures improvement rate over time
- **Hypothesis Lifecycle**: Active/confirmed/rejected hypothesis states
- **Personality Tracking**: Risk tolerance, confidence thresholds, learning rates
- **Cross-Session Consolidation**: Long-term pattern recognition

### 2. Telemetry Stack
- **VS Code Extension**: Status bar pulse, metrics HTTP endpoint
- **Prometheus**: Metrics collection and storage
- **Grafana**: Real-time dashboards and visualizations
- **Knowledge Graph API**: REST endpoints for policy/personality/consolidation

### 3. Integration Points
- **MCP Router**: Emits decision events with confidence scores
- **Knowledge Graph**: Stores insights as atoms and bonds
- **Event Bus**: Real-time event streaming
- **Neo4j**: Graph database for insight relationships

## Setup Instructions

### 1. Install Dependencies

```bash
# Core Python dependencies
pip install fastapi uvicorn neo4j prometheus-client

# VS Code extension dependencies (if developing)
npm install vscode @types/vscode
```

### 2. Configure Environment

Create `.env` file in project root:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
REDIS_URL=redis://localhost:6379
PROMETHEUS_ENABLED=true
TELEMETRY_ENABLED=true
```

### 3. Start Services

#### Option A: Start Everything
```bash
# Start Redis (for event bus)
redis-server

# Start Neo4j (for knowledge graph)
neo4j start

# Start Prometheus (for metrics collection)
prometheus --config.file=config/prometheus/super-alita-prometheus.yml

# Start Grafana (for dashboards)
grafana-server --config=config/grafana/grafana.ini

# Start Knowledge Graph API server
python start_kg_api.py

# Start VS Code with extension loaded
code . --extensionDevelopmentPath=./extension
```

#### Option B: Use Docker Compose (Recommended)
```bash
# Create docker-compose.yml and start all services
docker-compose up -d
```

### 4. VS Code Extension Setup

1. Open the workspace in VS Code
2. Run "Extensions: Install from VSIX" command
3. Select `automatalabs.copilot-mcp-*.vsix`
4. Reload VS Code
5. Check status bar for insight pulse indicator

## Usage

### 1. Monitor Self-Insight in Real-Time

#### Status Bar
- **Green pulse**: Learning velocity is positive
- **Yellow pulse**: Hypothesis testing in progress
- **Blue pulse**: Policy changes being applied
- **Red pulse**: Confidence below threshold

#### Console Output
Look for `[INSIGHTS]` log entries:
```
[INSIGHTS] Decision confidence: 0.85 → Tool choice: semantic_search
[INSIGHTS] Learning velocity: +0.12 (improving)
[INSIGHTS] Hypothesis confirmed: "High confidence decisions lead to better outcomes"
[INSIGHTS] Personality shift: Risk tolerance 0.6 → 0.65
```

### 2. Access Dashboards

#### Grafana Dashboard
Open http://localhost:3000/d/super-alita-dashboard

**Key Panels:**
- Decision Confidence Distribution
- Learning Velocity Timeline
- Hypothesis Status (active/confirmed/rejected)
- Agent Personality - Risk Tolerance
- Policy Changes vs Success Rate
- Self-Insight Health Timeline

#### Prometheus Metrics
Open http://localhost:9090

**Core Metrics:**
- `sa_decision_confidence_avg`
- `sa_learning_velocity`
- `sa_hypothesis_total{status="confirmed"}`
- `sa_personality_risk_tolerance`
- `sa_policy_changes_total`

### 3. API Endpoints

#### Knowledge Graph API (http://localhost:8000)

**Policy Management:**
```bash
# Get current policies
curl http://localhost:8000/api/kg/policy

# Apply new policy
curl -X POST http://localhost:8000/api/kg/policy \
  -H "Content-Type: application/json" \
  -d '{"content": "Prefer high-confidence decisions", "confidence": 0.9}'
```

**Personality Tracking:**
```bash
# Get personality metrics
curl http://localhost:8000/api/kg/personality

# Update personality traits
curl -X POST http://localhost:8000/api/kg/personality \
  -H "Content-Type: application/json" \
  -d '{"risk_tolerance": 0.7, "confidence_threshold": 0.8}'
```

**Insight Consolidation:**
```bash
# Get cross-session insights
curl http://localhost:8000/api/kg/consolidation

# Create new insight
curl -X POST http://localhost:8000/api/kg/consolidation \
  -H "Content-Type: application/json" \
  -d '{"content": "Pattern: High confidence → better outcomes", "session_count": 5}'
```

### 4. Drive Traffic and Observe

#### Generate Agent Activity
1. **Use Copilot extensively**: Ask for code reviews, refactoring, debugging
2. **Make complex requests**: Multi-step tasks that require decision-making
3. **Test edge cases**: Deliberately trigger confidence variations
4. **Work across sessions**: Close and reopen VS Code to test persistence

#### Monitor Hypothesis Evolution
1. Watch status bar pulse patterns
2. Check Grafana dashboard for hypothesis state changes
3. Look for policy adaptations in the logs
4. Observe personality drift over time

#### Validate Learning
1. **Decision Confidence**: Should improve over repeated similar tasks
2. **Learning Velocity**: Should show positive trends during active use
3. **Hypothesis Confirmations**: Should accumulate evidence over time
4. **Policy Changes**: Should become more targeted and effective

## Expected Behaviors

### Positive Learning Indicators
- **Increasing decision confidence** on familiar tasks
- **Hypothesis confirmations** outpacing rejections
- **Learning velocity** trending upward
- **Policy adaptations** becoming more specific
- **Personality traits** stabilizing around effective values

### Adaptive Responses
- **Lower confidence** in novel/complex situations (appropriate caution)
- **Policy adjustments** based on success/failure patterns
- **Hypothesis generation** for unexplained patterns
- **Cross-session consolidation** of validated insights

### Red Flags
- **Persistently low confidence** (may indicate poor calibration)
- **No hypothesis confirmations** (learning mechanism not working)
- **Erratic personality changes** (instability in trait tracking)
- **Zero policy adaptations** (static behavior, no learning)

## Troubleshooting

### Common Issues

1. **No metrics showing in Grafana**
   - Check VS Code extension is loaded
   - Verify http://localhost:3000/metrics returns data
   - Check Prometheus target status

2. **API endpoints returning 500 errors**
   - Ensure Neo4j is running and accessible
   - Check database connection configuration
   - Verify FastAPI server logs

3. **No insight pulse in status bar**
   - Reload VS Code window
   - Check extension activation events
   - Look for errors in VS Code Developer Tools

4. **Hypotheses not changing state**
   - Generate more varied agent activity
   - Check event bus connectivity
   - Verify insight oracle configuration

### Debug Commands

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:3000/metrics

# Test Neo4j connectivity
cypher-shell -u neo4j -p password "RETURN 1"

# Verify Redis event bus
redis-cli ping

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Log Locations

- **VS Code Extension**: Developer Tools → Console
- **FastAPI Server**: Console where `start_kg_api.py` is running
- **Neo4j**: `~/neo4j/logs/neo4j.log`
- **Prometheus**: Console output or system logs
- **Grafana**: `/var/log/grafana/grafana.log`

## Advanced Configuration

### Custom Metrics
Add new metrics in `extension/telemetry.ts`:
```typescript
// Example: Track tool usage patterns
const toolUsageCounter = new prometheus.Counter({
  name: 'sa_tool_usage_total',
  help: 'Total tool usage by type',
  labelNames: ['tool_type', 'success']
});
```

### Custom Policies
Create domain-specific policies via API:
```bash
curl -X POST http://localhost:8000/api/kg/policy \
  -H "Content-Type: application/json" \
  -d '{
    "content": "For code refactoring tasks, prefer confidence > 0.8",
    "category": "code_quality",
    "confidence": 0.95
  }'
```

### Hypothesis Templates
Define hypothesis patterns in the insight oracle:
```typescript
const hypothesisTemplates = [
  "When ${condition}, then ${outcome} with confidence ${confidence}",
  "Tool ${tool_name} performs better when ${context_condition}",
  "Decision confidence correlates with ${environmental_factor}"
];
```

## Success Metrics

After 1-2 weeks of usage, you should observe:

1. **Quantitative Improvements**
   - Decision confidence trending upward (>0.8 average)
   - Learning velocity consistently positive
   - 80%+ hypothesis confirmation rate
   - Policy adaptations leading to measurable improvements

2. **Qualitative Behaviors**
   - Agent makes more targeted tool choices
   - Faster resolution of repeated problem types
   - Better handling of edge cases over time
   - More contextually appropriate confidence levels

3. **System Health**
   - All dashboards showing live data
   - API endpoints responding < 100ms
   - No error spikes in telemetry
   - Consistent insight pulse patterns

This represents a fully operational self-insight loop where the agent continuously learns about its own decision-making patterns and adapts accordingly.
