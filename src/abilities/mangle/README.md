# Mangle Integration for Super Alita

## Overview

This integration adds Google's Mangle deductive database programming language to Super Alita, enabling advanced logical reasoning, security analysis, and knowledge graph capabilities. Mangle is a Datalog extension designed for deductive database programming with powerful recursive rule processing.

## Key Features

- **Logical Reasoning**: Define facts and rules to perform complex logical inference
- **Security Analysis**: Analyze dependencies for vulnerabilities using knowledge bases
- **Knowledge Graph**: Build and traverse knowledge graphs with rich relationships
- **Explainability**: Get detailed explanations of reasoning steps for transparency

## Installation

1. Ensure you have the Mangle executable installed:

   ```
   go install github.com/google/mangle/cmd/mangle@latest
   ```

2. Set the `MANGLE_BIN_PATH` environment variable to point to your Mangle executable:

   ```
   export MANGLE_BIN_PATH=/path/to/mangle
   ```

3. The integration will be automatically loaded when Super Alita starts.

## Usage

### Through API Endpoints

The following endpoints are available:

- `POST /ability/execute/mangle_query`
- `POST /ability/execute/mangle_add_fact`
- `POST /ability/execute/mangle_add_rule`
- `POST /ability/execute/mangle_analyze_dependencies`
- `POST /ability/execute/mangle_knowledge_graph`
- `POST /ability/execute/mangle_explain`

Example request to analyze dependencies:

```json
POST /ability/execute/mangle_analyze_dependencies
{
  "dependencies": [
    {"name": "log4j", "version": "2.14.0"},
    {"name": "spring-core", "version": "5.3.20"}
  ]
}
```

### Through REUG Streaming

Example using the streaming API:

```json
POST /v1/chat/stream
{
  "message": "Check if log4j 2.14.0 is vulnerable",
  "session_id": "abc123"
}
```

### Through Event Bus

Example publishing an event:

```python
event = create_event(
    "request_mangle_query",
    query="vulnerable(Name, Version)?",
    params={"timeout": 30}
)
await event_bus.publish(event)
```

## Mangle Concepts

### Facts

Facts are assertions about entities and their relationships:

```
dependency('log4j', '2.14.0').
vulnerability('log4j', '2.14.0', 'CVE-2021-44228').
```

### Rules

Rules define logical implications and can be recursive:

```
vulnerable(Lib, Ver) :-
  dependency(Lib, Ver),
  vulnerability(Lib, Ver, _).

transitive_depends_on(X, Y) :- depends_on(X, Y).
transitive_depends_on(X, Z) :- depends_on(X, Y), transitive_depends_on(Y, Z).
```

### Queries

Queries retrieve information from the knowledge base:

```
vulnerable(Name, Version)?
transitive_depends_on('Frontend', Component)?
```

## Examples

### Vulnerability Analysis

```python
from src.abilities.mangle.mangle_ability import MangleAbility

# Initialize Mangle
mangle = MangleAbility()

# Add facts
await mangle.add_fact("dependency('log4j', '2.14.0')")
await mangle.add_fact("known_vulnerability('log4j', '2.14.0')")

# Add rule
await mangle.add_rule(
    "vuln_check",
    "vulnerable(Name, Version) :- dependency(Name, Version), known_vulnerability(Name, Version)."
)

# Query for vulnerabilities
result = await mangle.query("vulnerable(Name, Version)")
```

### Knowledge Graph Analysis

```python
# Build knowledge graph
await mangle.add_fact("component('Frontend')")
await mangle.add_fact("component('Backend')")
await mangle.add_fact("depends_on('Frontend', 'Backend')")

# Add transitive dependency rule
await mangle.add_rule(
    "transitive",
    """
    transitive_depends_on(X, Y) :- depends_on(X, Y).
    transitive_depends_on(X, Z) :- depends_on(X, Y), transitive_depends_on(Y, Z).
    """
)

# Query with context
context = [
    {"relation": "has_vulnerability", "subject": "Backend", "object": "CVE-2023-1234"}
]

result = await mangle.knowledge_graph_query(
    "affected(X) :- component(X), transitive_depends_on(X, Y), has_vulnerability(Y, _)",
    context
)
```

## Integration Architecture

The Mangle integration consists of:

1. **MangleAbility**: Core class for interacting with Mangle
2. **ManglePluginInterface**: Plugin integration with Super Alita event bus
3. **Tool Registration**: API endpoints exposed through ability registry

The knowledge base is persisted in `./data/mangle/knowledge_base.json`.

## Advanced Features

### Explanation

Get detailed reasoning steps for any query:

```python
explanation = await mangle.explain_query_results("vulnerable('log4j', '2.14.0')")
print(explanation["explanation"])
```

### Batch Operations

Add multiple facts or rules at once:

```python
for fact in facts:
    await mangle.add_fact(fact)
```

### Custom Configurations

Customize timeouts, paths, and more:

```python
mangle = MangleAbility({
    "binary_path": "/custom/path/to/mangle",
    "timeout": 60,
    "knowledge_base_dir": "./custom/kb/path"
})
```

## Common Use Cases

1. **Dependency Vulnerability Analysis**: Identify vulnerable components
2. **Compliance Verification**: Check if systems meet policy requirements
3. **Attack Path Analysis**: Find potential security exploit paths
4. **Knowledge Graph Reasoning**: Traverse complex relationships

## Debugging

If you encounter issues:

1. Check Mangle executable path with `echo $MANGLE_BIN_PATH`
2. Verify knowledge base at `./data/mangle/knowledge_base.json`
3. Enable debug logging with `export LOG_LEVEL=DEBUG`

## Resources

- [Mangle GitHub Repository](https://github.com/google/mangle)
- [Datalog Tutorial](https://docs.racket-lang.org/datalog/)
- [Deductive Database Programming](https://en.wikipedia.org/wiki/Deductive_database)
