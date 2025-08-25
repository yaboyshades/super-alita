# Architecture

Super Alita follows **minimal predefinition, maximal self-evolution**.

- **MCP + Registry**: creates/loads tools at runtime. Tools are **atom factories**.
- **Atoms/Bonds**: a unified cognitive fabric. All tool outputs are atoms/bonds.
- **Memory**: idempotent graph store with lineage queries.
- **Flow**:

```mermaid
sequenceDiagram
    participant Orchestrator
    participant MCP
    participant Tool
    participant Memory
    Orchestrator->>MCP: invoke(tool, args)
    MCP->>Tool: run(args)
    Tool-->>MCP: {atoms,bonds}
    MCP->>Memory: store(atoms,bonds)
    Memory-->>Orchestrator: ack(ids)
```

- **Deterministic Identity**: UUIDv5 seeded by normalized content + type + title.
- **Provenance**: standardized in `meta.provenance` (source/activity/timestamp/context/parents).

## Runtime dependency graph

```mermaid
graph TD
    Router --> EventBus
    Router --> Registry
    Router --> LLM
    Router --> KG
    EventBus --> Redis[(Redis)]
    EventBus --> File[(NDJSON)]
    Registry --> FS[(Disk)]
    KG --> Store[(Adapters)]
```
