import pytest
from pathlib import Path
import subprocess

from agents.jules.indexer import RepoIndexer

pytestmark = pytest.mark.core

@pytest.fixture
def test_repo(tmp_path: Path) -> Path:
    """Creates a temporary test repository structure and stages files."""
    repo_root = tmp_path / "test_repo"
    repo_root.mkdir()

    # Initialize git repo and stage all files so `git ls-files` works as expected.
    subprocess.run(['git', 'init'], cwd=repo_root, check=True, capture_output=True)

    # Create files and directories
    (repo_root / "src").mkdir()
    (repo_root / "src/main.py").write_text("print('hello')")
    (repo_root / "src/utils.py").write_text("def helper(): pass")
    (repo_root / "data").mkdir()
    (repo_root / "data/items.json").write_text('{"key": "value"}')
    (repo_root / "README.md").write_text("# Test Repo")
    (repo_root / "app.log").write_text("Log entry")
    (repo_root / "build").mkdir()
    (repo_root / "build/artifact.bin").write_text("binary")

    # Create .gitignore
    (repo_root / ".gitignore").write_text("*.log\nbuild/")

    # Add all files to git staging area
    subprocess.run(['git', 'add', '.'], cwd=repo_root, check=True, capture_output=True)

    return repo_root

def test_code_map_all_files(test_repo: Path):
    """Tests that code_map returns all non-ignored files when using broad globs."""
    indexer = RepoIndexer(str(test_repo))
    files = indexer.code_map(globs=['*', '*/*', '.*', 'data/*', 'src/*'])

    expected = [
        ".gitignore",
        "README.md",
        "data/items.json",
        "src/main.py",
        "src/utils.py",
    ]
    assert set(files) == set(expected)

def test_code_map_python_only(test_repo: Path):
    """Tests glob filtering for Python files."""
    indexer = RepoIndexer(str(test_repo))
    files = indexer.code_map(globs=["src/*.py"])

    expected = [
        "src/main.py",
        "src/utils.py",
    ]
    assert set(files) == set(expected)

def test_code_map_with_ignore_pattern(test_repo: Path):
    """Tests the explicit ignore pattern functionality."""
    indexer = RepoIndexer(str(test_repo))
    files = indexer.code_map(globs=['*', '*/*', '.*'], ignore=["*.json"])

    expected = [
        ".gitignore",
        "README.md",
        "src/main.py",
        "src/utils.py",
    ]
    assert set(files) == set(expected)

def test_code_map_no_matches(test_repo: Path):
    """Tests a glob that matches no files."""
    indexer = RepoIndexer(str(test_repo))
    files = indexer.code_map(globs=["*.nonexistent"])
    assert files == []

def test_code_map_empty_globs_returns_all(test_repo: Path):
    """Tests that an empty globs list returns all non-ignored files."""
    indexer = RepoIndexer(str(test_repo))
    files = indexer.code_map(globs=[])

    expected = [
        ".gitignore",
        "README.md",
        "data/items.json",
        "src/main.py",
        "src/utils.py",
    ]
    assert set(files) == set(expected)
