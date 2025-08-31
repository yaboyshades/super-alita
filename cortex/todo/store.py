from __future__ import annotations

from .models import Todo


class InMemoryTodoStore:
    """Simple in-memory store. Replace with persistent backend as needed."""

    def __init__(self):
        self._by_id: dict[str, Todo] = {}

    def upsert(self, t: Todo) -> None:
        self._by_id[t.id] = t

    def get(self, todo_id: str) -> Todo | None:
        return self._by_id.get(todo_id)

    def children_of(self, todo_id: str) -> list[Todo]:
        t = self.get(todo_id)
        if not t:
            return []
        return [self.get(cid) for cid in t.children_ids if self.get(cid) is not None]

    def all(self) -> list[Todo]:
        return list(self._by_id.values())
