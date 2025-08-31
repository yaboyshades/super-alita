# Super Alita Production Deployment Guide

## 🎯 Overview

This guide provides comprehensive instructions for deploying Super Alita with enhanced consensus algorithms and Ollama integration in a production environment.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.11 or higher
- **Memory**: 16GB+ RAM (32GB+ recommended for large models)
- **Storage**: 50GB+ free space
- **Network**: Stable internet connection for model downloads

### Required Services
- **Ollama**: For local LLM serving
- **Redis** (optional): For enhanced event bus performance
- **Nginx/Apache** (optional): For reverse proxy and SSL

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yaboyshades/super-alita.git
cd super-alita

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies (takes 5+ minutes)
pip install -r requirements.txt -r requirements-test.txt

# Create environment file
cp .env.example .env
```

### 2. Ollama Installation and Configuration

```bash
# Install Ollama
# Windows: Download from https://ollama.ai/download
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
# macOS:
brew install ollama

# Start Ollama service
ollama serve

# Pull recommended models
ollama pull gpt-oss:20b      # Primary model (20.9B parameters)
ollama pull llama2:13b       # Alternative model
ollama pull mistral:7b       # Lightweight option
ollama pull codellama:13b    # Code-specific tasks
```

### 3. Super Alita Deployment

```bash
# Validate deployment
python validate_deployment.py

# Start production server
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4

# Or with gunicorn for production
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
```

## 🔧 Advanced Configuration

### Enhanced Consensus Algorithms

Super Alita now includes five consensus methods:

1. **Simple Vote** (`simple_vote`): Majority voting
2. **Weighted Vote** (`weighted_vote`): Confidence-weighted aggregation
3. **Confidence Based** (`confidence_based`): Threshold-based selection
4. **Semantic Similarity** (`semantic_similarity`): Word overlap clustering
5. **Ensemble Ranking** (`ensemble_ranking`): Multi-factor scoring

#### Configuration Example

```python
# In .env file
CONSENSUS_DEFAULT_METHOD=weighted_vote
CONSENSUS_NUM_SAMPLES=3
CONSENSUS_TEMPERATURE=0.7
CONSENSUS_CONFIDENCE_THRESHOLD=0.75
```

### Model Configuration

#### Ollama Models Comparison

| Model | Size | Parameters | Use Case | Memory |
|-------|------|------------|----------|---------|
| gpt-oss:20b | 11GB | 20.9B | General purpose, high quality | 16GB+ |
| llama2:13b | 7GB | 13B | Balanced performance | 12GB+ |
| mistral:7b | 4GB | 7B | Fast inference | 8GB+ |
| codellama:13b | 7GB | 13B | Code generation | 12GB+ |

#### Model Selection Strategy

```bash
# Production: High quality
ollama pull gpt-oss:20b

# Development: Balanced
ollama pull llama2:13b

# Edge deployment: Lightweight
ollama pull mistral:7b
```

### Environment Variables

Create a comprehensive `.env` file:

```bash
# Core Configuration
SUPER_ALITA_MODE=act
PYTHONPATH=./src

# Server Configuration
HOST=0.0.0.0
PORT=8080
WORKERS=4
LOG_LEVEL=info

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_TIMEOUT=120

# Consensus Configuration
CONSENSUS_DEFAULT_METHOD=weighted_vote
CONSENSUS_NUM_SAMPLES=3
CONSENSUS_TEMPERATURE=0.7
CONSENSUS_MAX_TOKENS=512
CONSENSUS_CONFIDENCE_THRESHOLD=0.75
CONSENSUS_TEMPERATURE_RANGE=0.2

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Security Configuration
ALITA_REQUIRE_API_KEY=true
ALITA_API_KEY=your-secure-api-key-here

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 🔒 Security Configuration

### API Key Authentication

```bash
# Enable API authentication
export ALITA_REQUIRE_API_KEY=true
export ALITA_API_KEY="your-secure-key"

# Multiple keys
export ALITA_API_KEYS="key1,key2,key3"
```

### Rate Limiting

```python
# Built-in rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600  # 1 hour
```

### Network Security

```nginx
# Nginx reverse proxy configuration
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 Monitoring and Health Checks

### Health Endpoints

```bash
# System health
curl http://localhost:8080/healthz

# Tools catalog
curl http://localhost:8080/tools/catalog

# Metrics (if enabled)
curl http://localhost:9090/metrics
```

### Monitoring Setup

```bash
# Install monitoring dependencies
pip install prometheus-client grafana-client

# Start monitoring
python -m src.monitoring.prometheus_exporter
```

### Health Check Validation

```python
# test_production_health.py
import requests
import json

def validate_production_health():
    """Comprehensive production health validation."""

    # Health check
    health = requests.get("http://localhost:8080/healthz").json()
    assert health["status"] == "healthy"

    # Tools availability
    tools = requests.get("http://localhost:8080/tools/catalog").json()
    assert any(tool["tool_id"] == "deepconf_consensus" for tool in tools)

    # Consensus functionality
    consensus_test = requests.post(
        "http://localhost:8080/tools/execute",
        json={
            "tool_id": "deepconf_consensus",
            "args": {
                "prompt": "What is 2+2?",
                "method": "weighted_vote",
                "num_samples": 2
            }
        }
    )
    assert consensus_test.status_code == 200

    print("✅ Production health validation passed!")

if __name__ == "__main__":
    validate_production_health()
```

## 🚀 Production Deployment Scenarios

### Scenario 1: Single Server Deployment

```bash
# Simple production deployment
gunicorn app:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100
```

### Scenario 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install -r requirements.txt -r requirements-test.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8080 11434

# Start services
CMD ["sh", "-c", "ollama serve & sleep 10 && ollama pull gpt-oss:20b && python -m uvicorn app:app --host 0.0.0.0 --port 8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  super-alita:
    build: .
    ports:
      - "8080:8080"
      - "11434:11434"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - OLLAMA_BASE_URL=http://localhost:11434/v1
      - OLLAMA_MODEL=gpt-oss:20b
      - SUPER_ALITA_MODE=act
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - super-alita
    restart: unless-stopped
```

### Scenario 3: Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: super-alita
spec:
  replicas: 3
  selector:
    matchLabels:
      app: super-alita
  template:
    metadata:
      labels:
        app: super-alita
    spec:
      containers:
      - name: super-alita
        image: super-alita:latest
        ports:
        - containerPort: 8080
        env:
        - name: OLLAMA_BASE_URL
          value: "http://ollama-service:11434/v1"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: super-alita-service
spec:
  selector:
    app: super-alita
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Issues

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
sudo systemctl restart ollama  # Linux
# or
ollama serve  # Manual start
```

#### 2. Memory Issues

```bash
# Monitor memory usage
htop

# Reduce model size or samples
export CONSENSUS_NUM_SAMPLES=2
export OLLAMA_MODEL=mistral:7b
```

#### 3. Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :8080

# Kill conflicting processes
sudo kill -9 $(lsof -t -i:8080)
```

#### 4. Import Errors

```bash
# Verify Python path
export PYTHONPATH=./src

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Tuning

#### Model Optimization

```bash
# Quantized models for faster inference
ollama pull llama2:7b-q4_0
ollama pull mistral:7b-q4_0

# GPU acceleration (if available)
export OLLAMA_GPU=true
```

#### Server Optimization

```python
# gunicorn.conf.py
bind = "0.0.0.0:8080"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

## 📈 Scaling Strategies

### Horizontal Scaling

1. **Load Balancer**: Nginx/HAProxy for request distribution
2. **Multiple Instances**: Run multiple Super Alita instances
3. **Ollama Clustering**: Distribute models across nodes

### Vertical Scaling

1. **Memory**: Increase RAM for larger models
2. **CPU**: More cores for parallel processing
3. **GPU**: NVIDIA GPUs for acceleration

### Model Scaling

```bash
# Progressive model loading
ollama pull mistral:7b     # Start small
ollama pull llama2:13b     # Scale up
ollama pull gpt-oss:20b    # Full capacity
```

## 🔒 Security Best Practices

### 1. API Security

```bash
# Strong API keys
export ALITA_API_KEY=$(openssl rand -hex 32)

# Header-based authentication
export ALITA_API_HEADER=X-API-Key
```

### 2. Network Security

```bash
# Firewall configuration
sudo ufw allow 8080/tcp
sudo ufw allow 11434/tcp
sudo ufw enable

# TLS/SSL termination
certbot --nginx -d your-domain.com
```

### 3. Container Security

```dockerfile
# Non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Read-only filesystem
docker run --read-only --tmpfs /tmp super-alita
```

## 📊 Production Monitoring

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

CONSENSUS_REQUESTS = Counter('consensus_requests_total', 'Total consensus requests')
CONSENSUS_LATENCY = Histogram('consensus_latency_seconds', 'Consensus request latency')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active WebSocket connections')
```

### Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
- name: super-alita
  rules:
  - alert: HighConsensusLatency
    expr: consensus_latency_seconds > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High consensus latency detected"

  - alert: ConsensusFailureRate
    expr: rate(consensus_failures_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High consensus failure rate"
```

## 🚀 Continuous Deployment

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy Super Alita

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt -r requirements-test.txt
    - name: Run tests
      run: |
        python validate_deployment.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        ssh user@production-server "cd /app && git pull && systemctl restart super-alita"
```

### Blue-Green Deployment

```bash
# Blue-Green deployment script
#!/bin/bash

# Start new version (green)
docker run -d --name super-alita-green -p 8081:8080 super-alita:latest

# Health check
sleep 30
curl -f http://localhost:8081/healthz || exit 1

# Switch traffic
docker stop super-alita-blue
docker run -d --name super-alita-blue-new -p 8080:8080 super-alita:latest

# Cleanup
docker rm super-alita-blue
docker rename super-alita-blue-new super-alita-blue
docker rm super-alita-green
```

## 🎯 Success Criteria

Your production deployment is successful when:

- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ All components (event_bus, ability_registry, kg, llm) show `"ok"`
- ✅ Tools catalog includes `deepconf_consensus`
- ✅ Consensus sampling works with all 5 algorithms
- ✅ Response time < 30 seconds for 3 samples
- ✅ Error rate < 1%
- ✅ 99.9% uptime achieved

## 📞 Support and Maintenance

### Regular Maintenance

```bash
# Weekly tasks
docker system prune -f
pip check
python validate_deployment.py

# Monthly tasks
pip install --upgrade -r requirements.txt
ollama pull gpt-oss:20b  # Update models
```

### Backup Strategy

```bash
# Backup configuration
tar -czf super-alita-backup-$(date +%Y%m%d).tar.gz \
    .env \
    src/ \
    requirements*.txt \
    validate_deployment.py

# Backup models
cp -r ~/.ollama/models/ ./model-backup/
```

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Check ADVANCED_DEVELOPMENT_PATTERNS.md
- **Health Checks**: Run `python validate_deployment.py`

---

## 🎉 Conclusion

This production deployment guide provides everything needed to deploy Super Alita with enhanced consensus algorithms in a production environment. The system now includes:

1. **✅ Fixed Health Checks**: All components report healthy status
2. **✅ Enhanced Consensus**: 5 different aggregation algorithms
3. **✅ Ollama Integration**: Multiple model support
4. **✅ Production Ready**: Security, monitoring, and scaling

The deployment is validated and ready for production use with robust consensus sampling capabilities!
## Redis-backed Rate Limiting

To ensure consistent limits across multiple processes/containers, enable the Redis limiter:

- Environment:
  - `ALITA_RATE_LIMIT_ENABLED=true`
  - `ALITA_RATE_LIMIT=120` (requests/window)
  - `ALITA_RATE_WINDOW=60` (window seconds)
  - `ALITA_REDIS_URL=redis://redis:6379`

- docker-compose example (excerpt):
```

### 4. Complete Orchestration (Dev Helper)

For a one‑command local setup that starts Ollama, ensures `gpt-oss:20b`, boots the API, validates the system, and smoke‑tests the enhanced consensus tool, use the orchestration script:

```bash
python tools/complete_startup.py
```

This script:
- Starts `ollama serve` if not already running
- Pulls `gpt-oss:20b` when missing (large download)
- Launches `uvicorn app:app` on port 8080
- Runs `validate_deployment.py`
- Verifies `deepconf_consensus` via `/tools/catalog` + `/ability/execute/deepconf_consensus`
- Provides an interactive prompt that streams via `/tools/reug_start_turn` + `/tools/reug_stream_next`
services:
  api:
    image: your/alita-image
    environment:
      - ALITA_RATE_LIMIT_ENABLED=true
      - ALITA_RATE_LIMIT=120
      - ALITA_RATE_WINDOW=60
      - ALITA_REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "no"]
    ports:
      - "6379:6379"
```

When set, the runtime uses `redis.asyncio` automatically. If Redis is unavailable or `ALITA_REDIS_URL` isn’t provided, it falls back to the in-process limiter.
