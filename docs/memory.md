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
