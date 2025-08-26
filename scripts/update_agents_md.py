#!/usr/bin/env python3
"""
Update docs/agents.md as a living document.

- Scans src/{abilities,plugins}/**/*.py and src/reug_runtime/** for registry info
- Pulls owners from CODEOWNERS if present
- Syncs .alita/sessions/ledger.json (creates if missing)
- Rewrites guarded blocks in docs/agents.md
- Appends a CHANGELOG entry for the current PR (if env PR number set)

ENV:
  GITHUB_PR_NUMBER (optional)
  GITHUB_ACTOR      (optional)
  RELEASE_TAG       (optional)
"""
from __future__ import annotations
import os, re, sys, json, time, pathlib, subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "agents.md"
LEDGER = ROOT / ".alita" / "sessions" / "ledger.json"
NOTES_DIR = ROOT / ".alita" / "sessions" / "notes"

MARKERS = {
    "AGENTS": ("<!-- AGENTS:START -->", "<!-- AGENTS:END -->"),
    "OWNERS": ("<!-- AGENTS:OWNERS_START -->", "<!-- AGENTS:OWNERS_END -->"),
    "ABILITIES": ("<!-- ABILITIES:START -->", "<!-- ABILITIES:END -->"),
    "PLUGINS": ("<!-- PLUGINS:START -->", "<!-- PLUGINS:END -->"),
    "SESSIONS": ("<!-- SESSIONS:START -->", "<!-- SESSIONS:END -->"),
    "CHANGELOG": ("<!-- CHANGELOG:START -->", "<!-- CHANGELOG:END -->"),
    # inline markers (start == end) retain the marker so future runs can update
    "LAST_UPDATED": ("<!-- AGENTS:LAST_UPDATED -->", "<!-- AGENTS:LAST_UPDATED -->"),
    "RELEASE": ("<!-- AGENTS:RELEASE -->", "<!-- AGENTS:RELEASE -->"),
}

def _find_py(mod_root: str) -> list[pathlib.Path]:
    return [p for p in (ROOT / mod_root).rglob("*.py") if p.is_file()]

def _scan_abilities():
    rows = []
    for p in _find_py("src/abilities"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(class .*Ability)|(register_ability\()|(@ability)", text):
            sig = re.search(r"def\s+([a-zA-Z_][\w]*)\(", text)
            name = sig.group(1) if sig else p.stem
            guard = "yes" if ("try:" in text or "Timeout" in text or "guard" in text) else "unknown"
            rows.append((name, str(p.relative_to(ROOT)), "(…)", guard, "Ability* events", ""))
    return rows

def _scan_plugins():
    rows = []
    for p in _find_py("src/plugins"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(class .*Plugin)|(register_plugin\()|(@plugin)", text):
            caps = ", ".join(
                sorted(
                    set(
                        re.findall(
                            r"capab(?:ility|ilities)\s*[:=]\s*\[?([^\]\n]+)",
                            text,
                            flags=re.I,
                        )
                    )
                )
            )
            rows.append((p.stem, str(p.relative_to(ROOT)), caps or "(…)", "ENV_*", "function() => ok", ""))
    return rows

def _scan_agents_top():
    has_router = (ROOT / "src" / "reug_runtime" / "router.py").exists()
    abilities = len(_scan_abilities())
    plugins = len(_scan_plugins())
    rows = [
        (
            "super-alita",
            "runtime",
            "src/main.py" if has_router else "(unknown)",
            abilities,
            plugins,
            "@owners",
            "beta",
            "",
        )
    ]
    return rows

def _codeowners():
    rows = []
    p = ROOT / "CODEOWNERS"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        try:
            path, owners = line.split(maxsplit=1)
            rows.append((path, owners, "(slack?)", "(oncall?)"))
        except ValueError:
            continue
    return rows

def _render_table(headers, rows):
    cols = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:" if i == 0 else "----" for i in range(cols)]) + "|",
    ]
    for r in rows:
        cells = [str(x).replace("\n", " ") for x in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def _replace_block(text, key, body):
    start, end = MARKERS[key]
    if start == end:
        pattern = re.compile(re.escape(start) + r".*?(?=\n|$)")
        return pattern.sub(start + body, text)
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    return pattern.sub(start + "\n" + body + "\n" + end, text)

def _load_json(path: pathlib.Path, default):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default, indent=2))
        return default
    return json.loads(path.read_text())

def _update_ledger(pr_number: str | None):
    data = _load_json(LEDGER, {"series": []})
    if pr_number:
        series_id = f"series-{datetime.now(timezone.utc).strftime('%Y%W')}"
        series = next((s for s in data["series"] if s["series_id"] == series_id), None)
        if not series:
            series = {"series_id": series_id, "prs": [], "branches": [], "session_notes": []}
            data["series"].append(series)
        if int(pr_number) not in series["prs"]:
            series["prs"].append(int(pr_number))
        LEDGER.write_text(json.dumps(data, indent=2))
    lines = []
    for s in sorted(data["series"], key=lambda x: x["series_id"], reverse=True)[:6]:
        lines.append(f"- **{s['series_id']}** · PRs: {sorted(s['prs'])}")
    return "\n".join(lines) if lines else "- (none)"

def main():
    md = DOC.read_text(encoding="utf-8")

    agents = _render_table(
        [
            "Agent",
            "Kind",
            "Entrypoint",
            "Abilities (count)",
            "Plugins (count)",
            "Owner(s)",
            "Stability",
            "Notes",
        ],
        _scan_agents_top(),
    )
    md = _replace_block(md, "AGENTS", agents)

    owners = _render_table(
        ["Component", "CODEOWNERS", "Slack", "Escalation"], _codeowners() or []
    )
    md = _replace_block(md, "OWNERS", owners or "| (no CODEOWNERS) | | | |")

    abilities = _render_table(
        ["Ability", "Module", "Signature", "Guardrails", "Telemetry Events", "Notes"],
        _scan_abilities(),
    )
    md = _replace_block(md, "ABILITIES", abilities)

    plugins = _render_table(
        ["Plugin", "Module", "Capabilities", "Config Keys", "Health Check", "Notes"],
        _scan_plugins(),
    )
    md = _replace_block(md, "PLUGINS", plugins)

    pr = os.getenv("GITHUB_PR_NUMBER")
    session_idx = _update_ledger(pr)
    md = _replace_block(md, "SESSIONS", session_idx)

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md = _replace_block(md, "LAST_UPDATED", now_iso)
    md = _replace_block(md, "RELEASE", os.getenv("RELEASE_TAG", "unreleased"))

    if pr:
        actor = os.getenv("GITHUB_ACTOR", "unknown")
        try:
            summary = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%s"], text=True
            ).strip()
        except Exception:
            summary = "update"
        change = f"- {now_iso} #{pr} {summary} (owner: @{actor})"
        start, end = MARKERS["CHANGELOG"]
        parts = md.split(start)
        head, tail = parts[0], start + parts[1]
        md = head + start + "\n" + change + "\n" + tail

    DOC.write_text(md, encoding="utf-8")
    print(f"Updated {DOC}")

if __name__ == "__main__":
    sys.exit(main())
