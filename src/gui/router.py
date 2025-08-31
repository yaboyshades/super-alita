from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .registry import gui_registry

router = APIRouter(prefix="/gui", tags=["gui"])

# Compact dark theme CSS (lines kept < 88 chars where possible)
_BASE_CSS = (
    "body{font-family:system-ui,sans-serif;margin:0;padding:0;"
    "background:#0f1115;color:#eee;}"
    ".gui-container{padding:1.25rem 1.5rem;}"
    ".badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;"
    "background:#444;}"
    ".badge-info{background:#2d6cdf;}"
    ".badge-warn{background:#d19a00;}"
    ".badge-error{background:#c43;}"
    ".panel{background:#1b1f27;border:1px solid #2a303b;border-radius:8px;"
    "margin-bottom:1rem;box-shadow:0 2px 4px rgba(0,0,0,.4);}"
  ".panel-header{font-weight:600;padding:.75rem 1rem;"
  "border-bottom:1px solid #2a303b;}"
    ".panel-body{padding:.75rem 1rem;}"
    ".grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));}"
    "a{color:#4da3ff;text-decoration:none;}a:hover{text-decoration:underline;}"
    ".footer{margin-top:2rem;font-size:12px;opacity:.6;}"
)

_BASE_JS = """
async function loadComponent(name, target){
  try {
    const res = await fetch(`/gui/components/${name}`);
    if(!res.ok){ throw new Error('Failed: '+res.status); }
    const html = await res.text();
    const el = document.getElementById(target);
    if(el){ el.innerHTML = html; }
  } catch(e){
    console.error(e);
  }
}
window.addEventListener('DOMContentLoaded', ()=>{
  // Lazy load demo components
  loadComponent('status_badge','status-slot');
  loadComponent('panel','panel-slot');
});
""".strip()


def _layout(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <title>{title}</title>
  <meta name='viewport' content='width=device-width,initial-scale=1' />
  <style>{_BASE_CSS}</style>
</head>
<body>
  <div class='gui-container'>
    {body}
    <div class='footer'>Super Alita GUI • Components: 
  {', '.join(gui_registry.list_components())}
    </div>
  </div>
  <script>{_BASE_JS}</script>
</body>
</html>"""


@router.get("", response_class=HTMLResponse)
async def gui_index() -> HTMLResponse:
    body = """
    <h1>Super Alita GUI</h1>
  <p>Lightweight server-rendered component library. Extend via Python registration.</p>
    <div id='status-slot'></div>
    <div id='panel-slot'></div>
    <h2>Available Components</h2>
    <ul>
    {items}
    </ul>
  """.replace(
  "{items}", "\n".join(
    f"<li><code>{c}</code></li>" for c in gui_registry.list_components()
  )
  )
    return HTMLResponse(_layout("Super Alita GUI", body))


@router.get("/components", response_class=JSONResponse)
async def list_components() -> JSONResponse:
    return JSONResponse({"components": gui_registry.list_components()})


@router.get("/components/{name}", response_class=HTMLResponse)
async def get_component(name: str, props: str | None = Query(None)) -> HTMLResponse:
    # Parse optional JSON props (ignore errors silently returning empty props)
    try:
        parsed_props: dict[str, Any] = json.loads(props) if props else {}
    except (ValueError, TypeError):
        parsed_props = {}

    try:
        html = gui_registry.render(name, parsed_props)
    except KeyError:
        return HTMLResponse(
            f"<div class='error'>Unknown component: {name}</div>",
            status_code=404,
        )
    return HTMLResponse(html)
