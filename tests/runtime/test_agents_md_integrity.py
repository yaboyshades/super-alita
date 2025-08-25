import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs" / "agents.md"


def test_header_fields_clean():
    lines = DOC.read_text().splitlines()
    last = next(l for l in lines if l.startswith("- Last Updated:"))
    assert re.fullmatch(r"- Last Updated: <!-- AGENTS:LAST_UPDATED -->\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", last)
    release = next(l for l in lines if l.startswith("- Current Release:"))
    assert re.fullmatch(r"- Current Release: <!-- AGENTS:RELEASE -->[\w\.\-]+", release)


def test_plugin_table_integrity():
    lines = DOC.read_text().splitlines()
    start = lines.index("<!-- PLUGINS:START -->")
    end = lines.index("<!-- PLUGINS:END -->")
    table_lines = lines[start + 2 : end]  # skip header and separator
    assert table_lines, "plugin table empty"
    for row in table_lines:
        assert row.startswith("| ") and row.endswith(" |"), row
