#!/usr/bin/env python
import sys, json, pathlib, yaml

# Simple structural checks; replace with JSON Schema if needed.

REQUIRED_STRATEGY_KEYS = {"arm_id", "description", "base_weight"}

def validate_strategies(path: pathlib.Path, problems: list[str]):
    file = path / "strategies.json"
    if not file.exists():
        problems.append("Missing strategies.json")
        return
    data = json.loads(file.read_text())
    if not isinstance(data, list):
        problems.append("strategies.json must be a list")
        return
    for idx, arm in enumerate(data):
        missing = REQUIRED_STRATEGY_KEYS - set(arm.keys())
        if missing:
            problems.append(f"strategies[{idx}] missing keys: {missing}")

def validate_tool_manifests(path: pathlib.Path, problems: list[str]):
    tools_dir = path / "tool_manifest"
    if not tools_dir.exists():
        return
    for f in tools_dir.glob("**/*.y*ml"):
        try:
            doc = yaml.safe_load(f.read_text())
        except Exception as e:
            problems.append(f"{f}: YAML parse error {e}")
            continue
        if not isinstance(doc, dict):
            problems.append(f"{f}: must be object")
            continue
        for key in ("name", "version", "cost", "endpoints"):
            if key not in doc:
                problems.append(f"{f}: missing '{key}'")

def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "config")
    problems: list[str] = []
    validate_strategies(root, problems)
    validate_tool_manifests(root, problems)
    if problems:
        print("Config validation issues:")
        print("\n".join(problems))
        sys.exit(1)
    print("Config validation passed.")

if __name__ == "__main__":
    main()