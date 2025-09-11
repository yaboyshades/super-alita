# Core Plan Executor Variants Audit

Date: 2025-09-11
Scope: `src/core/plan_executor.py`, `src/core/plan_executor_clean.py`
Objective: Identify functional deltas, risks, and propose unification path.

## Summary Classification
- `plan_executor.py`: ACTIVE + ENRICHED (Gap detection via `AtomGapEvent`, memory persistence using `TextualMemoryAtom`, tool registry integration, web search normalization, detailed result summarization with snippet extraction, CREATOR integration path).
- `plan_executor_clean.py`: DUPLICATE / LITE (Simplified version without gap detection, no subscription setup for tool results (bug risk), simpler persistence using `NeuralAtom`, lacks memory/content extraction helpers, less robust summary content).

## Feature Comparison
| Feature | plan_executor.py | plan_executor_clean.py |
|---------|------------------|------------------------|
| Tool Gap Detection / AtomGapEvent | Yes | No |
| Tool Registry Integration | Yes (`get_tool_registry`) | No |
| Result Waiter Key Strategy | `{plan_id}_{step_idx}` | Same |
| Tool Result Subscription Setup | Yes (async task subscribes to `tool_result`) | Missing (never subscribes) |
| Web Agent Normalization | Yes (`web_agent` & legacy `web_search`) | Yes (only `web_agent`) |
| Memory Manager Handling | Yes (extract memory content) | No |
| LLM Summary Fallback | Yes | Yes |
| Rich Web Result Summaries | Yes (top web & GitHub results, structured) | No (basic summary) |
| Service Offline Fast-Fail | Yes | Yes |
| Persistence Object | `TextualMemoryAtom` with metadata | `NeuralAtom` minimal |
| Gap Recovery Auto-Retry | Yes (after CREATOR) | No |
| Param Normalization | Query extraction + memory extraction | Query only |

## Key Issues Identified
1. `plan_executor_clean.py` never subscribes to tool result events → waiter events never fire (logical defect if used directly).
2. Duplicate logic splits feature development effort (gap detection, memory, summarization divergence).
3. Inconsistent storage model (TextualMemoryAtom vs NeuralAtom) could fragment retrieval/analytics.
4. Lack of unified interface for future capabilities (rollback, partial re-run, dependency graphs).

## Recommended Unification Strategy
1. Canonicalize on `plan_executor.py` (retain enriched features) and fold in any small improvements from clean version if present (none critical besides slightly smaller surface area).
2. Add interface boundary `IPlanExecutor` (Protocol or ABC) to stabilize API for orchestrator.
3. Extract gap detection and memory extraction helpers into private methods for clarity.
4. Implement optional semantic step scheduling (future) via dependency annotations.
5. Mark `plan_executor_clean.py` as deprecated with import-time warning; plan removal after deprecation window.

## Immediate Actions
- [ ] Add deprecation banner & warning to `plan_executor_clean.py`.
- [ ] Add lightweight protocol `plan_executor_interface.py` (optional if needed by orchestrator soon).
- [ ] Add unit test skeleton validating: gap detection path, tool result subscription functioning, memory manager branch, web_agent summary formatting.

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Hidden imports of clean variant persist | Provide shim alias inside deprecated file exporting canonical class |
| Future refactors break orchestrator | Introduce protocol + orchestrator type hints |
| Test coverage gap for gap-event path | Add focused test with mocked event_bus & tool registry |

## Decision Log
- Canonical base: `plan_executor.py` (feature-complete, aligns with closed-loop philosophy).
- Clean variant retained short-term only to avoid sudden breakage; immediate deprecation.

---
Document will update post-consolidation.
