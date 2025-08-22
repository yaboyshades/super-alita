import json, os
from typing import Dict, Any

REGISTRY_PATH = os.path.join(os.getenv("REUG_TOOL_REGISTRY_DIR", "tools_registry"), "tools.json")

def load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH,"r") as f:
        return json.load(f)

def save_registry(reg: Dict[str, Any]):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH,"w") as f:
        json.dump(reg,f,indent=2)

def register_tool(tool: Dict[str, Any]):
    reg = load_registry()
    reg[tool["tool_id"]] = tool
    save_registry(reg)
    return tool

