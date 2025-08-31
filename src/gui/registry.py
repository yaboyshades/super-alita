from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any
from pathlib import Path
import os

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


@register_component("mcp_index")
def _mcp_index(props: dict[str, Any]) -> str:
    """List persisted MCP-like specs from the MCP Box.

    Reads JSON spec filenames from MCP_BOX_DIR (default ./.mcp_box).
    """
    box_dir = Path(os.getenv("MCP_BOX_DIR", ".mcp_box"))
    items: list[str] = []
    if box_dir.exists():
        for p in sorted(box_dir.glob("*.json")):
            items.append(p.name)
    body = ["<div class='panel'>",
            "<div class='panel-header'>MCP Box</div>",
            "<div class='panel-body'>"]
    if items:
        body.append("<ul>")
        for name in items[:200]:
            body.append(f"<li><code>{name}</code></li>")
        body.append("</ul>")
    else:
        body.append("<em>No MCP specs found.</em>")
    body.append("</div></div>")
    return "".join(body)


@register_component("mcp_console")
def _mcp_console(props: dict[str, Any]) -> str:
    """Interactive MCP console: brainstorm → register → execute.

    Provides a minimal UI to drive the self-evolution loop without predefined tools.
    """
    return (
        "<div class='panel'>"
        "<div class='panel-header'>MCP Console</div>"
        "<div class='panel-body'>"
        "<style>.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}</style>"
        "<div><label>Task:</label><br/>"
        "<input id='mcp_task' type='text' style='width:100%'"
        " placeholder='Describe the capability you need...'/></div>"
        "<button id='mcp_btn_bs'>Brainstorm</button>"
        "<div id='mcp_bs' class='mono' style='white-space:pre-wrap;margin-top:8px'></div>"
        "<hr/>"
        "<div><label>Register Proposal (index):</label> "
        "<input id='mcp_idx' type='number' min='0' style='width:5em'/>"
        " <button id='mcp_btn_reg'>Register</button></div>"
        "<div id='mcp_reg' class='mono' style='white-space:pre-wrap;margin-top:8px'></div>"
        "<hr/>"
        "<div><label>Execute Registered Tool:</label><br/>"
        "<input id='mcp_tool' type='text' placeholder='tool_id' style='width:60%'/>"
        "<br/><textarea id='mcp_args' class='mono' style='width:100%;height:100px'"
        " placeholder='{""arg"": ""value""}'></textarea><br/>"
        "<button id='mcp_btn_exec'>Execute</button></div>"
        "<div id='mcp_exec' class='mono' style='white-space:pre-wrap;margin-top:8px'></div>"
        "</div>"
        "</div>"
        "<script>(function(){\n"
        "const el=(id)=>document.getElementById(id); let props=[];\n"
        "el('mcp_btn_bs').onclick=async()=>{\n"
        "  const task=el('mcp_task').value||'';\n"
        "  el('mcp_bs').textContent='… brainstorming …';\n"
        "  try{\n"
        "    const r=await fetch('/tools/mcp/brainstorm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});\n"
        "    const j=await r.json(); props=j.proposals||[];\n"
        "    el('mcp_bs').textContent=JSON.stringify(props,null,2);\n"
        "  }catch(e){ el('mcp_bs').textContent=String(e); }\n"
        "};\n"
        "el('mcp_btn_reg').onclick=async()=>{\n"
        "  const i=parseInt(el('mcp_idx').value||'0'); const spec=props[i];\n"
        "  if(!spec){ el('mcp_reg').textContent='No proposal at index'; return;}\n"
        "  el('mcp_reg').textContent='… registering …';\n"
        "  try{\n"
        "    const r=await fetch('/tools/mcp/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(spec)});\n"
        "    const j=await r.json(); el('mcp_reg').textContent=JSON.stringify(j,null,2);\n"
        "    if(j && j.registered){ el('mcp_tool').value=j.registered; }\n"
        "  }catch(e){ el('mcp_reg').textContent=String(e);}\n"
        "};\n"
        "el('mcp_btn_exec').onclick=async()=>{\n"
        "  const tid=el('mcp_tool').value||'';\n"
        "  let args={}; try{ args=JSON.parse(el('mcp_args').value||'{}'); }catch(e){ args={}; }\n"
        "  el('mcp_exec').textContent='… executing …';\n"
        "  try{\n"
        "    const r=await fetch('/tools/execute/'+encodeURIComponent(tid),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args})});\n"
        "    const j=await r.json(); el('mcp_exec').textContent=JSON.stringify(j,null,2);\n"
        "  }catch(e){ el('mcp_exec').textContent=String(e);}\n"
        "};\n"
        "})();</script>"
    )


@register_component("mcp_index_abstracted")
def _mcp_index_abstracted(props: dict[str, Any]) -> str:
    """Render the abstracted MCP index (index.json) from the MCP Box."""
    from pathlib import Path
    import json

    box_dir = Path(os.getenv("MCP_BOX_DIR", ".mcp_box"))
    index_path = box_dir / "index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
    else:
        data = None

    body = ["<div class='panel'>",
            "<div class='panel-header'>MCP Box (Abstracted Index)</div>",
            "<div class='panel-body'>"]
    if data:
        tools = data.get("tools", [])
        body.append(f"<p><strong>Canonical tools:</strong> {len(tools)}</p>")
        body.append("<ul>")
        for t in tools[:200]:
            tid = t.get("tool_id")
            act = t.get("action")
            props_l = ", ".join(t.get("properties", []))
            body.append(f"<li><code>{tid}</code> &mdash; action: <code>{act}</code>; props: {props_l}</li>")
        body.append("</ul>")
    else:
        body.append("<em>No index.json found. Use /tools/mcp/abstract to generate.</em>")
    body.append("</div></div>")
    return "".join(body)
