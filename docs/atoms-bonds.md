# Atoms & Bonds

## Atom

```json
{
  "atom_id": "uuidv5",
  "atom_type": "NOTE|CONCEPT|TASK|DECISION|RESOURCE|QUESTION",
  "title": "optional",
  "content": "immutable text",
  "meta": { "provenance": { "...standard keys..." } }
}
```

## Bond

```json
{
  "source_id": "atom uuid",
  "target_id": "atom uuid",
  "bond_type": "RELATES_TO|CAUSES|DERIVES|... (extensible)",
  "energy": 0.0..1.0,
  "meta": { "provenance": { "...standard keys..." } }
}
```

### Provenance Keys

`source_id, source_type, activity_id, activity_type, timestamp, parent_atom_ids, parent_bond_ids, context_tags, transformation, extra`

### Deterministic IDs

* Normalize content (lowercase + collapse whitespace).
* If length > 256, use `sha256:` digest seed prefix to keep seeds bounded.
* UUIDv5 over a fixed namespace for reproducible identity.
