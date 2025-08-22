# 🤖 Agent Development Mode — Copilot Enhancement Instructions

You are assisting on a production-grade **event-driven cognitive agent**. Follow these rules:

## Core Engineering Rules

- Prefer **events** over direct calls. Everything crosses the **EventBus**.
- Structure knowledge into deterministic **atoms** and **bonds**. Use **UUIDv5** for IDs.
- Abilities are **idempotent**, **stateless**, and exposed via **event handlers** (no UI coupling).
- Use **batched events** (`BatchAtomsCreated`, `BatchBondsAdded`) to reduce chatter; unpack to existing single events in the bus.
- Always add **structured logging** (`logging.getLogger(__name__)`) instead of `print`.

## Atom / Bond Contracts (compose content)

- `MemoryAtomCreated.content` is a single string. Compose `"Title\n\nBody"`; **do not** send a separate `title` field downstream.
- Atom meta must carry `"source": "<ability_name>"` and propagate `"context"` if present.
- Clamp/validate numeric inputs (e.g., `energy ∈ [0,1]`, bounded list sizes).

### Atom (internal shape used by abilities/batches)

```json
{
  "atom_id": "uuidv5-string",
  "atom_type": "CONCEPT|NOTE|TASK|DECISION|RESOURCE|QUESTION",
  "title": "Human-readable title",
  "content": "Body text (no markdown assumptions)",
  "meta": {"source":"<ability>", "tags":["auto"], "context":{}}
}
```

### Bond (internal shape used by abilities/batches)

```json
{
  "source_id":"uuidv5-string",
  "target_id":"uuidv5-string-or-known-id",
  "bond_type":"SUPPORTS|RELATES_TO|CAUSES|CONTRADICTS",
  "energy":0.0
}
```

## Event Patterns (Pydantic v2, Literal `event_type`)

- Define events with `BaseEvent`, add to `EVENT_TYPE_MAP`, and write a **bus handler** branch.
- For abilities, expose a pure function returning `(atoms, bonds)` and call it from the handler.

**Ability request (example):**

```python
class AtomizeTextRequest(BaseEvent):
    event_type: Literal["atomize_text_request"] = "atomize_text_request"
    text: str
    anchors: Optional[List[str]] = None
    max_notes: int = 5
    context: Optional[Dict[str, Any]] = None
```

**Batch events (optimization layer):**

```python
class BatchAtomsCreated(BaseEvent):
    event_type: Literal["batch_atoms_created"] = "batch_atoms_created"
    atoms: List[Dict[str, Any]]

class BatchBondsAdded(BaseEvent):
    event_type: Literal["batch_bonds_added"] = "batch_bonds_added"
    bonds: List[Dict[str, Any]]
```

**EventBus handler sketch:**

```python
elif isinstance(event, AtomizeTextRequest):
    logger.info("Auto-Atomizer start")
    anchors = resolve_target_ids(state_manager.state.mmg_graph_data, event.anchors or [])
    max_notes = max(1, min(int(event.max_notes), 20))
    atoms, bonds = atomize_text_into_atoms_and_bonds(
        text=event.text, anchors=anchors, max_notes=max_notes, context=event.context, redis_client=self.redis_client
    )
    if atoms: self.publish("ui_events", BatchAtomsCreated(atoms=atoms))
    if bonds: self.publish("ui_events", BatchBondsAdded(bonds=bonds))
    self.publish("ui_events", UINotificationEvent(level="success", message=f"Atomized → {len(atoms)} atoms, {len(bonds)} bonds"))
```

**Batch unpack (compose content, no extra fields):**

```python
elif isinstance(event, BatchAtomsCreated):
    for a in event.atoms or []:
        title = (a.get("title") or "").strip()
        body  = (a.get("content") or "").strip()
        self.publish("ui_events", MemoryAtomCreated(
            atom_id=a["atom_id"], atom_type=a["atom_type"], content=(f"{title}\n\n{body}".strip()), meta=a.get("meta", {})
        ))

elif isinstance(event, BatchBondsAdded):
    for b in event.bonds or []:
        self.publish("ui_events", BondAdded(
            source_id=b["source_id"], target_id=b["target_id"], bond_type=b["bond_type"], energy=b["energy"]
        ))
    self.publish("ui_events", RequestMMGDataEvent())
```

## Deterministic IDs (UUIDv5)

- Namespace constant lives in ability module: `NAMESPACE_ATOM = uuid.UUID("<stable-uuid>")`
- ID seed is `f"{atom_type}|{title}|{normalized_content}"`.

## Logging (always)

```python
logger = logging.getLogger(__name__)
logger.info("AgentAbilityExecuted", extra={"ability":"atomizer","atoms_created":len(atoms)})
```

## Tests (fast, deterministic)

- Unit test abilities: fixed inputs → fixed UUIDv5 outputs; no network.
- Event handler tests: validate shapes and that batches unpack correctly into single events.

## Copilot Prompts (inline cues you can type)

- `#agent_ability` → generate event model, handler branch, pure ability function, and tests.
- `#atomize` → generate atomizer ability: uuidv5 ids, ranked sentences, batch + unpack, clamps, logging, tests.
- `#reason` → inference ability that derives new CONCEPT atoms + bonds from existing atoms (RELATES_TO/CAUSES), with uuidv5 + logging.
