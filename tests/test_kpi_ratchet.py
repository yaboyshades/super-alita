from __future__ import annotations

import importlib.util
import sys
import subprocess
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest


def load_kpi_ratchet() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "kpi-ratchet.py"
    spec = importlib.util.spec_from_file_location("kpi_ratchet_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load kpi-ratchet module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def kpi_ratchet() -> ModuleType:
    return load_kpi_ratchet()


def test_generate_report_uses_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kpi_ratchet: ModuleType
) -> None:
    template_path = tmp_path / "REPORT.tpl.md"
    template_path.write_text(
        "Date: <fill> | Commit: <sha> | Size: <files/LOC>\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "REPORT.md"

    context = kpi_ratchet.ReportContext(
        date="2024-01-02",
        commit_sha="abc1234",
        file_count=4,
        loc_count=200,
    )
    monkeypatch.setattr(kpi_ratchet, "build_context", lambda repo_root: context)

    generated_path = kpi_ratchet.generate_report(
        template_path=template_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert generated_path == output_path
    assert output_path.read_text(encoding="utf-8") == "Date: 2024-01-02 | Commit: abc1234 | Size: 4 files / 200 LOC\n"
    # Ensure the template remained untouched.
    assert template_path.read_text(encoding="utf-8") == "Date: <fill> | Commit: <sha> | Size: <files/LOC>\n"


def test_build_context_counts_files(tmp_path: Path, kpi_ratchet: ModuleType) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)

    (repo_root / "alpha.txt").write_text("first\nsecond\n", encoding="utf-8")
    (repo_root / "bravo.py").write_text("only\n", encoding="utf-8")
    subprocess.run(["git", "add", "alpha.txt", "bravo.py"], cwd=repo_root, check=True, capture_output=True)

    context = kpi_ratchet.build_context(repo_root)

    assert context.date == date.today().isoformat()
    assert context.commit_sha == "unknown"
    assert context.file_count == 2
    assert context.loc_count == 3
    assert context.repo_size_label == "2 files / 3 LOC"
