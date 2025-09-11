# Consolidation Roadmap
Date: 2025-09-11
Scope: Core orchestration & reliability primitives (event bus variants, plan executor, decision policy, reliability layer)

## Objective
Eliminate duplicative variant modules, unify reliability and event routing layers, and standardize decision + planning logic under a single orchestrator-compatible contract.

## Drivers
- Reduced cognitive load
- Consistent metrics / observability
- Simplified dependency graph
- Easier constitutional & SDD integration

## Target End State
| Concern | Unified Module | Replaces | Mode / Strategy | Notes |
|---------|---------------|----------|-----------------|-------|
| Reliability | `reliability_unified.py` (`UnifiedReliabilityManager`) | reliability.py, reliability_optimized.py | fast / balanced / strict | Feature flags & pluggable components |
| Event Bus | `event_bus_unified.py` | event_bus.py, event_bus_clean.py, event_bus_old.py, in_memory_event_bus.py, reliable_event_bus.py, event_bus_redis.py | backend=memory|redis, reliability=strict|off | Reliability optional adapter |
| Planning Execution | `plan_executor_unified.py` | plan_executor.py, plan_executor_clean.py | async staged pipeline | Emits standard events |
| Decision Policy | `decision_policy_unified.py` | decision_policy.py, decision_policy_v1.py | strategy plug-ins | Weighted + consensus aware |

## Phased Plan
### Phase 1 (Design & Shims)
- Finalize unified reliability design (DONE)
- Draft unified event bus spec (pending)
- Introduce deprecation warnings in legacy modules (partial: reliability)

### Phase 2 (Implementation)
- Implement `UnifiedReliabilityManager` + tests
- Implement unified decision policy interface (strategy registry)
- Implement unified plan executor (stage contracts + metrics)
- Create unified event bus with backend abstraction & reliability hook

### Phase 3 (Migration)
- Switch orchestrator imports to unified modules
- Add re-export shims in legacy modules
- Run regression scenarios (consensus, SDD, streaming)

### Phase 4 (Hardening)
- Load + soak tests (high EPS event simulation)
- Chaos tests: Redis disconnect, circuit open, DLQ flood
- Performance benchmarks vs baseline

### Phase 5 (Removal)
- Remove deprecated variants after one release cycle
- Update docs & architecture diagrams

## Key Interfaces (Sketch)
```python
class UnifiedEventBus:
    async def publish(self, event: BaseEvent, *, reliability_mode: str | None = None): ...
    async def subscribe(self, event_type: str, handler: EventHandler, *, semantic=None, threshold: float = 0.7): ...
    def get_metrics(self) -> dict: ...

class UnifiedPlanExecutor:
    async def execute(self, plan: Plan, *, context: ExecutionContext) -> ExecutionResult: ...

class DecisionPolicyEngine:
    def decide(self, options: list[DecisionOption], context: DecisionContext) -> DecisionOutcome: ...

class UnifiedReliabilityManager:  # (already designed)
    async def process_event(self, event, publish_cb, enable_idempotency=True) -> dict: ...
```

## Metrics Convergence
- Use single namespace: `orchestrator.*`
- Event bus: `orchestrator.events.emitted`, `...processed`, `...duplicates`
- Reliability: `orchestrator.reliability.mode`, `...duplicates_blocked`, `...circuit_state`
- Decision: `orchestrator.decision.latency_ms`, `...strategy_used`
- Planning: `orchestrator.plan.stages_completed`, `...failure_stage`

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Hidden coupling to legacy modules | Add temporary import tracing wrapper |
| Performance regression | Benchmark before + after per phase |
| Incomplete test coverage | Enforce coverage gates in CI for unified modules |
| Feature drift during transition | Freeze variant modules (doc: read-only) |

## Open Items
- Define semantic similarity adapter for UnifiedEventBus
- Decide on storage for run ledger (Redis vs file)
- Align decision policy with consensus tool hooks

## Decision Log
- Proceed with reliability first due to highest duplication + safety impact
- Event bus unification deferred until reliability stabilized

-- End Document --
