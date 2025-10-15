# Super Alita v4.0 Architecture Guide

## System Overview

Super Alita v4.0 is a modular AI agent platform built with clean architecture principles, featuring constitutional safety, advanced memory management, and extensible plugin systems.

## Architecture Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                      │
├──────────────────────────────────────────────────────────────────┤
│  Chat Router  │  API Router   │  Health Router │  Auth Router   │
│  Streaming    │  Unified API  │  Monitoring    │  Key Mgmt      │
├──────────────────────────────────────────────────────────────────┤
│                        Middleware Stack                         │
│  Logging   │  Security   │  Rate Limiting   │  Authentication   │
├──────────────────────────────────────────────────────────────────┤
│                        Service Layer                            │
│  Event Bus │  LLM Client │  Knowledge Graph │  Ability Registry │
│  Registry  │  Service    │  Service         │  Service          │
├──────────────────────────────────────────────────────────────────┤
│                      Governance Layer                           │
│         Constitutional Reasoner  │  Safety Evaluation           │
├──────────────────────────────────────────────────────────────────┤
│                        Core Layer                               │
│    Configuration   │    Event System    │    Process Manager   │
└──────────────────────────────────────────────────────────────────┘
```

## Key Components

### Application Factory (`src/app/factory.py`)

**Purpose**: Clean application creation with dependency injection

**Key Features**:
- Configuration-driven setup
- Service lifecycle management
- Router registration
- Middleware stack assembly

### Service Registry (`src/services/registry.py`)

**Purpose**: Dependency injection container for services

**Key Features**:
- Service initialization in dependency order
- Type-safe service access
- Health monitoring per service
- Graceful shutdown handling

### Constitutional Reasoner (`src/governance/constitutional_reasoner.py`)

**Purpose**: AI safety through constitutional principles

**Safety Mechanisms**:
- Embedding-based principle matching
- Regex pattern violation detection
- Contextual evaluation
- Explainable decisions

### Event Bus Service (`src/services/event_bus.py`)

**Purpose**: Event-driven architecture foundation

**Event Types**:
- System lifecycle events
- Tool execution events
- Constitutional evaluation events
- Custom application events

## Data Flow

### Chat Request Flow

```
1. HTTP Request → Middleware Stack
   ├── Logging (request tracking)
   ├── Security (headers, validation)
   ├── Rate Limiting (abuse prevention)
   └── Authentication (API key check)

2. Router Processing → Chat Router
   ├── Message extraction
   ├── Session management
   └── Response generation

3. Service Layer → LLM Service
   ├── Message formatting
   ├── Provider selection
   └── Streaming generation

4. Constitutional Check → Safety Evaluation
   ├── Action analysis
   ├── Principle matching
   └── Approval/rejection
```

## Security Architecture

### Multi-Layer Security

1. **Network Layer**: HTTPS, CORS, security headers
2. **Application Layer**: API key authentication, rate limiting
3. **Service Layer**: Input validation, constitutional checking
4. **Data Layer**: Encrypted storage, secure key management

### Constitutional Safety

The constitutional reasoner provides:
- **Proactive Safety**: Actions evaluated before execution
- **Explainable Decisions**: Clear reasoning for approvals/rejections
- **Contextual Evaluation**: Considers user intent and action context
- **Domain Customization**: Extensible for organization-specific rules

## Performance Characteristics

### Startup Performance
- **Cold start**: ~2-3 seconds
- **Warm start**: ~1 second
- **Service initialization**: Parallel where possible

### Runtime Performance
- **Chat response**: <2 seconds (LLM dependent)
- **Tool execution**: <1 second (tool dependent)
- **Constitutional check**: <100ms
- **Health check**: <50ms

### Resource Usage
- **Memory**: ~200-500MB base (model dependent)
- **CPU**: Low when idle, scales with requests
- **Storage**: Logs and knowledge graph growth

This architecture provides a solid foundation for building sophisticated AI agents with safety, observability, and extensibility built in.