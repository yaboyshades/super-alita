
# LADDER Planner System Documentation

## Overview
The LADDER Planner system implements a hierarchical task planning and execution framework based on the LADDER methodology: Localize → Assess → Decompose → Decide → Execute → Review. The system provides enhanced planning capabilities with knowledge graph integration, multi-armed bandit learning, and energy-based task prioritization.

## Core Components

### LadderStage
```python
class LadderStage(str, Enum):
    LOCALIZE = "L"
    ASSESS = "A"
    DECOMPOSE = "D1"
    DECIDE = "D2"
    EXECUTE = "E"
    REVIEW = "R"
```

### Todo Model
```python
class Todo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depends_on: set[str] = Field(default_factory=set)
    stage: LadderStage = LadderStage.LOCALIZE
    status: TodoStatus = TodoStatus.PENDING
    energy: float = 0.0
    priority: float = 0.0
    confidence: float = 0.0
    owner: str | None = None
    tool_hint: str | None = None
    exit_criteria: list[ExitCriteria] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### Interfaces
```python
class TodoStore(Protocol):
    def upsert(self, t: Any) -> None: ...
    def get(self, todo_id: str) -> Any: ...
    def children_of(self, todo_id: str): ...
```

## Enhanced Ladder Planner

### EnhancedLadderPlanner Class
The EnhancedLadderPlanner class extends the base LADDER implementation with advanced features:

- Multi-armed bandit tool selection with ε-greedy algorithm
- Energy-based task prioritization
- Shadow/Active execution modes
- Knowledge base integration for learning

```python
class EnhancedLadderPlanner:
    """Enhanced LADDER planner with multi-armed bandit learning and advanced task decomposition."""

    def __init__(
        self,
        kg: KG,
        bandit: Bandit,
        store: TodoStore,
        orchestrator: Orchestrator,
        mode: str = "shadow",
    ):
        self.kg = kg
        self.bandit = bandit
        self.store = store
        self.orch = orchestrator
        self.mode = mode  # "shadow" or "active"
        self.bandit_stats: dict[str, dict[str, Any]] = {}
        self.knowledge_base: dict[str, Any] = {}
        self.exploration_rate = 0.1  # ε for ε-greedy bandit
        self.active_inference = FLAGS.ladder_active_inference
        self.state_model = GenerativeStateModel() if self.active_inference else None
        self.latent_state: dict[str, float] | None = None
```

### Key Methods

#### _enhanced_ladder
```python
async def _enhanced_ladder(self, root: Todo) -> None:
    """Enhanced LADDER implementation with energy-based prioritization."""
    logger.info(
        f"🎯 Starting enhanced LADDER for: {root.title} (mode: {self.mode})"
    )

    # L -> A
    self._advance_stage(root, LadderStage.ASSESS)
    root = self.store.get(root.id) or root
    root = self._enhanced_assess(root)

    # A -> D1
    self._advance_stage(root, LadderStage.DECOMPOSE)
    root = self.store.get(root.id) or root
    children = await self._enhanced_decompose(root)
    root = self.store.get(root.id)  # Refresh after decomposition

    # D1 -> D2
    self._advance_stage(root, LadderStage.DECIDE)
    root = self.store.get(root.id) or root
    children = self._enhanced_decide(children)

    # D2 -> E
    self._advance_stage(root, LadderStage.EXECUTE)
    root = self.store.get(root.id) or root
    await self._enhanced_execute(root, children)

    # E -> R
    self._advance_stage(root, LadderStage.REVIEW)
    root = self.store.get(root.id) or root
    await self._enhanced_review(root, children)
```

#### _enhanced_assess
```python
def _enhanced_assess(self, t: Todo) -> Todo:
    """A: Enhanced assessment with knowledge base integration."""
    logger.info(f"🔍 Enhanced assessment for: {t.title}")

    # Get context from knowledge graph
    ctx = self.kg.get_context_for_title(t.title)

    # Check knowledge base for similar tasks
    similar_tasks = self._find_similar_tasks(t.title)

    # Calculate confidence based on historical performance
    confidence = self._calculate_confidence(t.title, similar_tasks)

    # Enhanced energy calculation
    kg_energy = self.kg.compute_energy_for_title(t.title)
    final_energy = (t.energy + kg_energy) / 2.0 if kg_energy > 0 else t.energy

    updated_data = {
        "energy": final_energy,
        "confidence": confidence,
    }

    if ctx and len(ctx) > 10:
        updated_data["description"] = (
            t.description or ""
        ) + f"\n\n[KG Context]: {ctx[:500]}"

    if similar_tasks:
        similar_summary = ", ".join([task["title"] for task in similar_tasks[:3]])
        updated_data["evidence"] = t.evidence + [
            Evidence(
                kind="note",
                summary=f"Similar tasks found: {similar_summary}",
                score=confidence,
            )
        ]

    updated_t = t.model_copy(update=updated_data)
    self.store.upsert(updated_t)

    self._emit_sync(
        "plan.assessed",
        updated_t.id,
        {
            "energy": updated_t.energy,
            "confidence": updated_t.confidence,
            "similar_tasks": len(similar_tasks),
        },
    )

    return updated_t
```

#### _enhanced_decompose
```python
async def _enhanced_decompose(self, root: Todo) -> list[Todo]:
    """Enhanced decomposition with task-specific strategies."""
    logger.info(f"🧩 Enhanced decomposition for: {root.title}")

    # Select decomposition strategy based on task type
    strategy = self._select_decomposition_strategy(root)
    children = await strategy(root)

    if not children:
        # Fallback to default decomposition
        children = await self._de