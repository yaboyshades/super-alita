from pathlib import Path

from src.tools.spec_generator import draft_tool_spec, write_tool_spec


def test_spec_generator_writes_files(tmp_path: Path) -> None:
    spec = draft_tool_spec("Convert a PDF to text")
    paths = write_tool_spec(spec, root=str(tmp_path))
    md = Path(paths["markdown"])
    js = Path(paths["json"])
    assert md.exists()
    assert js.exists()
    assert md.read_text(encoding="utf-8").startswith("# Tool Spec:")
