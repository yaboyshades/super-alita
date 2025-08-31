from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

StateMap = MutableMapping[str, Any]
ComponentFunc = Callable[[dict[str, Any]], str]


class GUIRegistry:
    """In-memory registry for simple server-rendered components.

    Features:
      - Function component registration via decorator
      - Ephemeral state store for simple counters, etc.
      - JSON Schema -> form field mapping (basic types only)
    """

    def __init__(self) -> None:  # pragma: no cover - trivial
        self._components: dict[str, ComponentFunc] = {}
        self._state: StateMap = {}

    # --- Component Management -------------------------------------------------
    def register(self, name: str, func: ComponentFunc) -> None:
        if not name or not callable(func):  # defensive
            raise ValueError("Invalid component registration")
        self._components[name] = func

    def get(self, name: str) -> ComponentFunc | None:
        return self._components.get(name)

    def list_components(self) -> list[str]:
        return sorted(self._components.keys())

    # Backwards compatibility alias (avoid potential name shadowing confusion)
    def list_all(self) -> list[str]:  # pragma: no cover - simple alias
        return self.list_components()

    # Provide lazy alias via attribute lookup to avoid static analyzer confusion
    def __getattr__(self, name: str) -> object:  # pragma: no cover
        if name == "list":
            return self.list_all
        raise AttributeError(name)

    def render(self, name: str, props: dict[str, Any] | None = None) -> str:
        comp = self.get(name)
        if not comp:
            raise KeyError(name)
        return comp(props or {})

    # --- State Management -----------------------------------------------------
    def get_state(self, key: str, default: Any | None = None) -> Any:
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def update_state(self, key: str, updater: Callable[[Any], Any]) -> Any:
        cur = self._state.get(key)
        new_val = updater(cur)
        self._state[key] = new_val
        return new_val

    def dump_state(self) -> dict[str, Any]:
        return dict(self._state)

    # --- Schema Helpers -------------------------------------------------------
    def schema_to_fields(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        order = list(props.keys())
        fields: list[dict[str, Any]] = []
        for name in order:
            spec = props.get(name) or {}
            ftype = spec.get("type", "string")
            input_type = {
                "string": "text",
                "integer": "number",
                "number": "number",
                "boolean": "checkbox",
            }.get(ftype, "text")
            fields.append(
                {
                    "name": name,
                    "title": spec.get("title", name.title()),
                    "type": input_type,
                    "required": name in schema.get("required", []),
                }
            )
        return fields


gui_registry = GUIRegistry()


def register_component(name: str) -> Callable[[ComponentFunc], ComponentFunc]:
    """Decorator helper for registering a component."""

    def _wrap(fn: ComponentFunc) -> ComponentFunc:
        gui_registry.register(name, fn)
        return fn

    return _wrap


# --- Built-in baseline components -------------------------------------------
@register_component("status_badge")
def _status_badge(props: dict[str, Any]) -> str:
    text = props.get("text", "OK")
    level = props.get("level", "info")
    return f"<span class='badge badge-{level}'>{text}</span>"


@register_component("panel")
def _panel(props: dict[str, Any]) -> str:
    title = props.get("title", "Panel")
    body = props.get("body", "")
    return (
        "<div class='panel'>"
        f"<div class='panel-header'>{title}</div>"
        f"<div class='panel-body'>{body}</div>"
        "</div>"
    )


# --- Stateful & Form Components ---------------------------------------------
@register_component("counter")
def _counter(props: dict[str, Any]) -> str:
    key = props.get("key", "counter_default")
    initial = int(props.get("initial", 0))
    if gui_registry.get_state(key) is None:
        gui_registry.set_state(key, initial)
    val = gui_registry.get_state(key, initial)
    return (
        "<div class='panel'>"
        "<div class='panel-header'>Counter</div>"
        f"<div class='panel-body'><strong>{val}</strong></div>"
        "</div>"
    )


@register_component("simple_form")
def _simple_form(props: dict[str, Any]) -> str:
    fields = props.get("fields") or []
    action = props.get("action", "#")
    parts = [f"<form method='post' action='{action}'>"]
    for f in fields:
        fname = f.get("name")
        ftype = f.get("type", "text")
        title = f.get("title", fname)
        required = " required" if f.get("required") else ""
        if ftype == "checkbox":
            parts.append(
                "<label><input type='checkbox' name='"
                f"{fname}'{required}/> {title}</label>"
            )
        else:
            parts.append(
                f"<label>{title}<br/><input type='{ftype}' name='"
                f"{fname}'{required}/></label>"
            )
    parts.append("<button type='submit'>Submit</button></form>")
    return "".join(parts)


@register_component("schema_form")
def _schema_form(props: dict[str, Any]) -> str:  # pragma: no cover - delegates
    schema = props.get("schema") or {}
    action = props.get("action", "#")
    fields = gui_registry.schema_to_fields(schema)
    return _simple_form({"fields": fields, "action": action})


__all__ = [
    "GUIRegistry",
    "gui_registry",
    "register_component",
]
