# Memory API

- `store_atoms_bonds(atoms, bonds) -> {stored_atoms, stored_bonds}`
- `query_atoms(atom_type=None) -> List[Atom]`
- `get_children(atom_id) -> List[str]`
- `get_parents(atom_id) -> List[str]`
- `bfs(start_ids, direction="forward"|"backward"|"both", max_hops=2) -> {id:distance}`
- `export_json() -> str`

Deduplication uses canonical bond keys: sorted `(source_id, target_id)` and `bond_type`.

## Consolidated Memory Snapshots

The knowledge store periodically captures **snapshots** of node embeddings.
Snapshots are stored as lightweight dictionaries of `node_id -> vector` and
preserved in a rolling buffer.

During subsequent updates the store performs a *rehearsal* step that averages
current embeddings with those from recent snapshots. This consolidation helps
reinforce earlier knowledge and prevents catastrophic forgetting when the graph
is rebuilt.

Snapshots can be restored by loading the stored vectors back into the graph
before performing new embedding updates.

## ACE-Evolved Cross-Domain Memory Stack

The runtime now tracks a three-layer memory architecture that mirrors the
Complementary Learning Systems (CLS) theory and integrates ACE's context
evolution loop with the LLM context window.

```
ACE Context Evolution
        ↓
Cross-Domain Memory System (bio-inspired)
        ↓
LLM Context Window
```

### Biological Fidelity Layers

| Layer | Inspired Region | Responsibilities |
| ----- | ---------------- | ---------------- |
| Working Memory | Prefrontal cortex | High-salience context buffer (≈7 ±2 chunks) |
| Episodic Memory | Hippocampus | Fast capture of recent experience; consolidation staging |
| Semantic Memory | Neocortex | Slow, structured knowledge graph consolidation |
| Procedural Memory | Basal ganglia | Skill/strategy library surfaced as reusable abilities |

```python
class BioInspiredMemorySystem:
    working_buffer: ContextBuffer
    episodic_store: VectorStore
    semantic_store: KnowledgeGraph
    skill_library: SkillStore
```

Each tier emits atoms/bonds so that rehearsal, replay, and schema evolution can
be measured via telemetry. Episodic entries age according to configurable
forgetting curves; once consolidated, distilled gists keep the semantic store
compact while preserving provenance links to the original observations.

### Evolvable Memory Strategies

ACE optimization steers consolidation and retrieval policies. Strategies are
tracked as first-class entities so the runtime can evaluate and mutate them.

```python
class EvolvableMemoryStrategy:
    stability_plasticity_balance: float
    forgetting_curves: dict[str, float]
    consolidation_schedules: dict[str, str]
    strategy_performance: dict[str, float]
    context_transformation_history: list[dict[str, object]]
```

- **Spaced reinforcement**: Consolidation queues respect Ebbinghaus-inspired
  decay profiles and leverage the runtime TODO scorer to resurface knowledge
  before it fades.
- **Progressive abstraction**: Episodic traces are synthesized into semantic
  principles and the resulting bonds are annotated with their abstraction level
  so the router can pick executive summaries or raw evidence on demand.
- **Graceful forgetting**: When the utility score of an episodic item drops
  below threshold and a consolidated gist exists, the system compresses the
  item while preserving backlinks for auditability.
- **Cross-modal association**: Visual, textual, and audio embeddings are
  stitched via shared concept atoms, enabling multimodal retrieval paths.

### Self-Optimizing Context Windows

The streaming orchestrator can request evolved contexts that blend episodic and
semantic memories based on historical success metrics.

```python
context = system.get_evolved_context(
    query="Analyze Q4 strategy",
    evolution_feedback={
        "previous_success_rate": 0.85,
        "preferred_abstraction_level": "executive_summary",
        "contradiction_resolution_style": "evidence_weighted",
    },
)
```

### Emergent Problem Solving Loop

Instead of a static retrieval step, the runtime executes an ACE-guided loop that
measures tool efficacy, updates the bandit prior, and feeds the evolved context
into the LLM stream.

```python
response = llm(
    query,
    context=ace_evolver.evolve(
        memory_system.retrieve_with_strategy(query, current_strategy),
        performance_feedback,
    ),
)
```

### Business-Facing Patterns

- **Enterprise copilot**: Learns user preferences as episodic traces, distills
  semantic workflows (for example: `data_first → risk_assessment →
  stakeholder_alignment`), and updates strategy weights when success metrics
  drift.
- **Autonomous researcher**: Processes literature, fuses cross-domain
  relationships, and evolves reading strategies with reward signals sourced
  from evaluation tasks.
- **Personalized education**: Promotes multimodal delivery strategies (e.g.,
  `diagram_expansion` vs. `text_detailing`) based on learner telemetry.

### Metrics and Observability

| Metric | Traditional RAG | ACE + Cross-Domain Memory |
| ------ | --------------- | ------------------------- |
| Learning velocity | Requires periodic retraining | Real-time ingestion with consolidation |
| Context relevance | Static retrieval set | Evolves per turn via ACE feedback |
| Knowledge retention | Keep/discard binary | Graceful decay with gist compression |
| Cross-domain reasoning | Limited | Strengthened via associative bonds |
| User adaptation | Minimal | Continuous preference modeling |

Telemetry already captures `STATE_TRANSITION`, `AbilityCalled`, `AbilitySucceeded`
and related events. The memory stack augments them with atoms/bonds for
`USED_STRATEGY`, `CONSOLIDATED_ITEM`, and `FORGOTTEN_ITEM` so that bandit
updates and audit dashboards can trace why a given context was presented.

### Implementation Roadmap

1. **Core integration (0–4 months)** – Extend memory models with biological
   layers, wire episodic→semantic consolidation, and implement forgetting curve
   scheduling with regression tests in `tests/runtime`.
2. **Self-optimization (4–7 months)** – Record strategy performance, add
   cross-domain association learning, and prototype adaptive forgetting
   policies.
3. **Emergent capabilities (7–12 months)** – Meta-learn consolidation
   strategies, unify multimodal inputs, and experiment with autonomous
   hypothesis generation loops.

### Definition of Success

- **6 months** – ≥40% improvement on complex reasoning benchmarks, user-facing
  evidence that the system “remembers what matters,” and observable strategy
  evolution in telemetry logs.
- **12 months** – Stable reasoning styles tuned to user cohorts, fewer
  contradictory answers (target −60%), and novel insight generation through
  cross-domain bonds.
- **18+ months** – Autonomous knowledge discovery, transfer of high-performing
  reasoning patterns across teams, and predictive retrieval that anticipates
  information needs before the next turn.
