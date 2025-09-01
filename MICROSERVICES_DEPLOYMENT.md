# Microservices Deployment Guide

This guide covers deploying Super Alita with the new microservices architecture using Docker and Redis.

## Architecture Overview

The microservices setup includes:

- **Redis**: Distributed event bus for inter-service communication
- **Consensus Service**: gRPC server for distributed consensus algorithms
- **Main App**: FastAPI application with adapter-based service discovery
- **Development Mode**: Hot-reload setup for development

## Quick Start

### 1. Local Development (Recommended)

Start Redis and consensus service, then run the main app locally:

```bash
# Start supporting services
docker-compose --profile microservices up -d redis consensus-service

# Run main app locally with live reload
export CONSENSUS_SERVICE_MODE=grpc
export CONSENSUS_GRPC_URL=127.0.0.1:50051
export REDIS_URL=redis://localhost:6379/0
python -m uvicorn app:app --reload --port 8080
```

### 2. Full Containerized Deployment

```bash
# Start all services
docker-compose --profile microservices --profile app up -d

# Check service health
curl http://localhost:8080/healthz
curl http://localhost:8080/tools/catalog
```

### 3. Development with Hot Reload

```bash
# Start Redis only
docker-compose --profile redis up -d

# Run in local mode (no gRPC consensus)
export CONSENSUS_SERVICE_MODE=local
export REDIS_URL=redis://localhost:6379/1  # Use different DB
python -m uvicorn app:app --reload --port 8080
```

## Configuration Modes

### Local Mode (Default)

- `CONSENSUS_SERVICE_MODE=local`
- Uses in-memory consensus algorithms
- Good for development and testing

### gRPC Mode (Microservices)

- `CONSENSUS_SERVICE_MODE=grpc`
- `CONSENSUS_GRPC_URL=consensus-service:50051` (Docker) or `127.0.0.1:50051` (local)
- Distributed consensus via gRPC service

### Hybrid Mode (Resilient)

- `CONSENSUS_SERVICE_MODE=hybrid`
- Tries gRPC first, falls back to local on failure
- Best for production environments

## Environment Variables

```bash
# Consensus Service
CONSENSUS_SERVICE_MODE=grpc|local|hybrid
CONSENSUS_GRPC_URL=host:port
CONSENSUS_GRPC_BIND=0.0.0.0:50051  # For server only

# Redis Event Bus
REDIS_URL=redis://host:port/db
USE_REDIS_EVENT_BUS=true|false

# Development
ALITA_AUTO_DISCOVER_ABILITIES=on
```

## Docker Profiles

Use profiles to control which services start:

- `redis`: Just Redis
- `microservices`: Redis + Consensus Service
- `app`: Main application
- `dev`: Development services

Examples:

```bash
# Minimal setup for development
docker-compose --profile redis up -d

# Full microservices stack
docker-compose --profile microservices --profile app up -d

# Development mode
docker-compose --profile dev up -d
```

## Testing the Setup

1. **Health Check**:

   ```bash
   curl http://localhost:8080/healthz
   ```

2. **Test Consensus (gRPC mode)**:

   ```bash
   curl -X POST http://localhost:8080/ability/execute/deepconf_consensus \
     -H "Content-Type: application/json" \
     -d '{"prompt":"List two primary colors","method":"weighted_vote","num_samples":3}'
   ```

3. **Test Mangle Abilities**:

   ```bash
   curl -X POST http://localhost:8080/ability/execute/mangle_rule_catalog \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

4. **View Tools Catalog**:
   ```bash
   curl http://localhost:8080/tools/catalog
   ```

## Troubleshooting

### Connection Issues

If consensus service fails to connect:

```bash
# Check if gRPC server is running
docker-compose logs consensus-service

# Test gRPC connection directly
python -c "import grpc; grpc.channel_ready_future(grpc.insecure_channel('localhost:50051')).result(timeout=5)"
```

### Redis Issues

If Redis connection fails:

```bash
# Check Redis health
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis
```

### Port Conflicts

If ports are already in use:

```bash
# Check what's using the port
netstat -tulpn | grep :8080
netstat -tulpn | grep :50051
netstat -tulpn | grep :6379

# Update port mappings in docker-compose.yml
```

## Production Considerations

1. **Security**:

   - Use Redis AUTH (`REDIS_PASSWORD`)
   - Configure TLS for gRPC in production
   - Run services as non-root users

2. **Scaling**:

   - Multiple consensus service replicas behind load balancer
   - Redis Cluster for high availability
   - Horizontal scaling of main app instances

3. **Monitoring**:

   - Health checks for all services
   - Redis monitoring with RedisInsight
   - gRPC metrics and tracing

4. **Data Persistence**:
   - Redis data persistence configured
   - Volume mounts for application data
   - Backup strategies for stateful services

## Logs and Debugging

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f redis
docker-compose logs -f consensus-service
docker-compose logs -f super-alita-app

# Follow logs with timestamps
docker-compose logs -f -t
```
