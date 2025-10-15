# Super Alita v4.0 Configuration Guide

## Environment Variables

### Core Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ALITA_PROFILE` | string | `production` | Application profile (production, development, test) |
| `API_PREFIX` | string | `` | API route prefix (e.g., `/api/v1`) |
| `LOG_LEVEL` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | string | `json` | Log format (json, text) |
| `REUG_LOG_DIR` | string | `./logs` | Log directory path |

### Security Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ALITA_REQUIRE_API_KEY` | boolean | `false` | Enable API key authentication |
| `ALITA_API_KEY` | string | - | Single API key for authentication |
| `ALITA_API_KEYS` | comma-separated | - | Multiple API keys (comma-separated) |
| `ALITA_ADMIN_KEY` | string | - | Admin key for privileged operations |
| `ALITA_RATE_LIMIT_ENABLED` | boolean | `false` | Enable rate limiting |
| `ALITA_RATE_LIMIT` | integer | `60` | Requests per window |
| `ALITA_RATE_WINDOW` | integer | `60` | Rate limit window in seconds |

### LLM Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_MODEL` | string | `ollama:gpt-oss:20b` | LLM model specification |
| `OLLAMA_HOST` | string | `http://127.0.0.1:11434` | Ollama server URL |
| `LLM_TIMEOUT` | integer | `60` | LLM request timeout in seconds |
| `LLM_TEMPERATURE` | float | `0.7` | Default temperature for generation |
| `LLM_MAX_TOKENS` | integer | `2048` | Maximum tokens per response |

### Feature Flags

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_ENHANCED_CONSENSUS` | boolean | `true` | Enable consensus sampling |
| `ALITA_ENABLE_Z3` | boolean | `false` | Enable Z3 constraint solving |
| `RESEARCH_ENABLED` | boolean | `false` | Enable research mode features |
| `SUPER_ALITA_DEV` | boolean | `false` | Enable development mode |

## Configuration Profiles

### Development Profile

```bash
# .env.development
ALITA_PROFILE=development
LOG_LEVEL=DEBUG
LOG_FORMAT=text
ALITA_REQUIRE_API_KEY=false
SUPER_ALITA_DEV=true
ENABLE_GITHUB_DEMO=true
```

### Production Profile

```bash
# .env.production
ALITA_PROFILE=production
LOG_LEVEL=INFO
LOG_FORMAT=json
ALITA_REQUIRE_API_KEY=true
ALITA_API_KEYS=prod-key-1,prod-key-2
ALITA_ADMIN_KEY=admin-key-secure
ALITA_RATE_LIMIT_ENABLED=true
```

## Security Best Practices

### API Key Management

1. **Never commit API keys** to version control
2. **Use strong, unique keys** for each environment
3. **Rotate keys regularly** using the auth endpoints
4. **Monitor key usage** through logs and metrics

### Constitutional Safety

Configure constitutional principles in `.github/CONSTITUTION.md`:

```markdown
## Privacy Protection
Never access, store, or transmit personally identifiable information.

## Security First
Never execute arbitrary code or bypass security measures.

## Transparency
Provide clear reasoning for all decisions and actions.
```

## Monitoring and Observability

### Health Endpoints

- `GET /health` - Basic health check
- `GET /healthz` - Kubernetes-compatible health check
- `GET /health/simple` - Load balancer health check

### Key Metrics to Monitor

- **Request latency** (P50, P95, P99)
- **Constitutional approval rate**
- **Tool execution success rate**
- **LLM response times**
- **Error rates by endpoint**