import json
import os
import subprocess
from pathlib import Path

import pytest  # type: ignore[import-not-found]

CREATE_SCRIPT = Path("scripts") / "create-new-feature.sh"
SETUP_PLAN_SCRIPT = Path("scripts") / "setup-plan.sh"

SPEC_TEMPLATE_TEXT = "# Spec Template\n"
PLAN_TEMPLATE_TEXT = "# Plan Template\n"
DEFAULT_DESCRIPTION = "Realtime inference tuning plan"


def _run_git_command(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def _run_script(
    script_path: Path, repo_dir: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy() | {
        "GIT_DIR": str(repo_dir / ".git"),
        "GIT_WORK_TREE": str(repo_dir),
    }
    return subprocess.run(
        ["bash", str(script_path), *args],
        cwd=script_path.parent,
        env=env,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "feature-repo"
    repo_dir.mkdir()

    _run_git_command(repo_dir, "init")
    _run_git_command(repo_dir, "config", "user.email", "test@example.com")
    _run_git_command(repo_dir, "config", "user.name", "Test User")
    _run_git_command(repo_dir, "commit", "--allow-empty", "-m", "init")

    templates_dir = repo_dir / "templates"
    templates_dir.mkdir()
    (templates_dir / "spec-template.md").write_text(
        SPEC_TEMPLATE_TEXT, encoding="utf-8"
    )
    (templates_dir / "plan-template.md").write_text(
        PLAN_TEMPLATE_TEXT, encoding="utf-8"
    )

    return repo_dir


def _slugify(text: str) -> str:
    """
    Call the slugify function from scripts/lib/sdd-common.sh via a subprocess.

    Assumes that the sourced script defines a slugify helper.
    """
    # Compose a shell command that sources the script and calls slugify
    # The shell script should print the slugified result to stdout
    # This assumes sdd-common.sh is in scripts/lib/sdd-common.sh
    script = f'source scripts/lib/sdd-common.sh && slugify "{text}"'
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("München", "munchen"),
        ("Straße", "strasse"),
    ],
)
def test_slugify_transliterates_unicode_characters(text: str, expected: str) -> None:
    assert _slugify(text) == expected


def _branch_suffix(slug: str) -> str:
    parts = [part for part in slug.split("-") if part]
    if not parts:
        return "feature"
    return "-".join(parts[:3])


def test_create_new_feature_generates_branch_and_spec(tmp_path: Path) -> None:
    repo_dir = _init_repo(tmp_path)

    result = _run_script(CREATE_SCRIPT, repo_dir, "--json", DEFAULT_DESCRIPTION)

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)

    slug = _slugify(DEFAULT_DESCRIPTION)
    branch_suffix = _branch_suffix(slug)
    expected_feature_num = "001"
    expected_branch = f"{expected_feature_num}-{branch_suffix}"
    expected_spec = repo_dir / "specs" / expected_branch / "spec.md"

    assert data == {
        "BRANCH_NAME": expected_branch,
        "SPEC_FILE": str(expected_spec),
        "FEATURE_NUM": expected_feature_num,
    }

    assert expected_spec.exists()
    assert expected_spec.read_text(encoding="utf-8") == SPEC_TEMPLATE_TEXT

    head_branch = _run_git_command(
        repo_dir, "rev-parse", "--abbrev-ref", "HEAD"
    ).stdout.strip()
    assert head_branch == expected_branch


def test_setup_plan_populates_template_and_reports_paths(tmp_path: Path) -> None:
    repo_dir = _init_repo(tmp_path)
    create_result = _run_script(CREATE_SCRIPT, repo_dir, "--json", DEFAULT_DESCRIPTION)
    create_data = json.loads(create_result.stdout)
    branch_name = create_data["BRANCH_NAME"]

    result = _run_script(SETUP_PLAN_SCRIPT, repo_dir, "--json")

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)

    feature_dir = repo_dir / "specs" / branch_name
    expected_spec = feature_dir / "spec.md"
    expected_plan = feature_dir / "plan.md"

    assert data == {
        "FEATURE_SPEC": str(expected_spec),
        "IMPL_PLAN": str(expected_plan),
        "SPECS_DIR": str(feature_dir),
        "BRANCH": branch_name,
    }

    assert expected_plan.exists()
    assert expected_plan.read_text(encoding="utf-8") == PLAN_TEMPLATE_TEXT

    head_branch = _run_git_command(
        repo_dir, "rev-parse", "--abbrev-ref", "HEAD"
    ).stdout.strip()
    assert head_branch == branch_name
