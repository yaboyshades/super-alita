# Memory API

- `store_atoms_bonds(atoms, bonds) -> {stored_atoms, stored_bonds}`
- `query_atoms(atom_type=None) -> List[Atom]`
- `get_children(atom_id) -> List[str]`
- `get_parents(atom_id) -> List[str]`
- `bfs(start_ids, direction="forward"|"backward"|"both", max_hops=2) -> {id:distance}`
- `export_json() -> str`

Deduplication uses canonical bond keys: sorted `(source_id, target_id)` and `bond_type`.
