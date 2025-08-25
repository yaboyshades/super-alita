# Super Alita Copilot Agent

A GitHub Copilot Agent that provides SSE streaming interface to the Super Alita AI system.

## Features

- **SSE Streaming**: Proper Server-Sent Events with GitHub Copilot platform event types
- **Preview SDK Integration**: Uses @copilot-extensions/preview-sdk for request verification and response formatting
- **gRPC Backend**: Connects to Super Alita gRPC server for cognitive processing
- **Multi-Armed Bandit**: Optimization decisions through Thompson Sampling and UCB1
- **Knowledge Graph**: Query and explore the deterministic knowledge store
- **Cortex Integration**: Full perception-reasoning-action cognitive cycle

## Event Types Supported

- `copilot_confirmation` - Interactive confirmation dialogs
- `copilot_errors` - Error reporting with proper categorization  
- `copilot_references` - Code/content references with metadata
- Default SSE - Standard text streaming

## Commands

### System Commands
- `health` - Check system health status
- `status` - Get detailed system metrics
- `help` - Show command reference

### Knowledge Graph
- `kg <query>` - Query the knowledge graph
- Example: `kg machine learning algorithms`

### Optimization
- `decide <policy_id>` - Get optimization decision
- `reward <decision_id> <value>` - Provide decision feedback
- Example: `decide exploration` then `reward decision_12345 0.8`

### Testing
- `confirm test` - Test confirmation dialog flow
- `refs test` - Test reference attachment
- `error test` - Test error reporting

### General Processing
- Any other text is processed through the Cortex cognitive cycle

## Development

```bash
# Install dependencies
npm install

# Development server (with hot reload)
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run tests
npm test
```

## Environment Variables

- `PORT` - Server port (default: 8787)
- `GITHUB_TOKEN` - GitHub token for request verification
- `SUPER_ALITA_GRPC_HOST` - gRPC server address (default: localhost:50051)

## Architecture

```
GitHub Copilot ──SSE──> Agent Server ──gRPC──> Super Alita Backend
                           │
                           ├── Health/Status
                           ├── Cortex Processing  
                           ├── Knowledge Graph
                           └── Optimization Engine
```

## Security

- Request verification via GitHub's signature validation
- Token-based authentication with Preview SDK
- Proper error handling and sanitization
- Rate limiting through platform controls