"""Test module for Jules Repository Indexer

IMPORTANT: Run this test through pytest, not directly as a Python script.

Correct usage:
    python -m pytest tests/test_indexer.py -v

Incorrect usage (will cause import errors):
    python tests/test_indexer.py

See docs/TESTING_GUIDE.md for more information.
"""

import sys
from pathlib import Path
import pytest

# Verify that src directory is in sys.path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"

# Check if we're running in a properly configured environment
if str(src_path) not in sys.path:
    print("⚠️  WARNING: src directory not in sys.path!")
    print("   This usually means you're running the test incorrectly.")
    print("   Use: python -m pytest tests/test_indexer.py")
    print("   See docs/TESTING_GUIDE.md for more information.")
    print()

# Debug information (only shown with -s flag)
print("DEBUG: Current sys.path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print(f"DEBUG: Repository root: {repo_root}")
print(f"DEBUG: Source path: {src_path}")
print(f"DEBUG: Source path exists: {src_path.exists()}")
print(f"DEBUG: Source path is in sys.path: {str(src_path) in sys.path}")

# Check if the jules module files exist (for debug output)
if src_path.exists():
    agents_path = src_path / "agents"
    if agents_path.exists():
        print(f"DEBUG: Contents of {agents_path}:")
        for item in agents_path.iterdir():
            print(f"  - {item.name}")
            
        # List contents of jules directory if it exists
        jules_path = agents_path / "jules"
        if jules_path.exists():
            print(f"DEBUG: Contents of {jules_path}:")
            for item in jules_path.iterdir():
                print(f"  - {item.name}")
    else:
        print(f"DEBUG: Agents directory does not exist: {agents_path}")
else:
    print(f"DEBUG: Source directory does not exist: {src_path}")

print()  # Add spacing before tests
def test_debug_sys_path():
    """Test that we can see the sys.path correctly."""
    if str(src_path) not in sys.path:
        pytest.fail(
            f"Source path {src_path} not in sys.path. "
            f"Make sure you're running tests through pytest: "
            f"python -m pytest tests/test_indexer.py"
        )


def test_jules_module_files_exist():
    """Test that the Jules module files exist on the filesystem."""
    jules_init_path = src_path / "agents" / "jules" / "__init__.py"
    jules_indexer_path = src_path / "agents" / "jules" / "indexer.py"
    
    assert jules_init_path.exists(), f"Jules __init__.py should exist at {jules_init_path}"
    assert jules_indexer_path.exists(), f"Jules indexer.py should exist at {jules_indexer_path}"


def test_jules_indexer_import():
    """Test importing the Jules indexer module."""
    try:
        from agents.jules.indexer import RepositoryIndexer
        print("SUCCESS: Import worked!")
        
        # Test basic functionality
        indexer = RepositoryIndexer(Path("/tmp"))
        assert indexer is not None
        assert indexer.repo_path == Path("/tmp")
        
        # Test indexing functionality
        index_result = indexer.index_repository()
        assert isinstance(index_result, dict)
        assert "repo_path" in index_result
        assert "files" in index_result
        assert "directories" in index_result
        
    except ModuleNotFoundError as e:
        pytest.fail(
            f"Failed to import agents.jules.indexer: {e}\n"
            f"Make sure you're running through pytest: python -m pytest tests/test_indexer.py\n"
            f"See docs/TESTING_GUIDE.md for more information."
        )


def test_jules_indexer_functionality():
    """Test Jules indexer core functionality."""
    from agents.jules.indexer import RepositoryIndexer
    
    # Test with the current repository
    indexer = RepositoryIndexer(repo_root)
    index_result = indexer.index_repository()
    
    # Verify the index contains expected data
    assert index_result["repo_path"] == str(repo_root)
    assert isinstance(index_result["files"], list)
    assert isinstance(index_result["directories"], list)
    assert isinstance(index_result["python_modules"], list)
    
    # Should find some Python files in the repo
    assert len(index_result["python_modules"]) > 0
    
    # Test search functionality
    search_results = indexer.search_code("indexer")
    assert isinstance(search_results, list)
    
    # Test other methods
    deps = indexer.get_dependencies("some_file.py")
    assert isinstance(deps, list)
    
    symbols = indexer.get_symbols("some_file.py")
    assert isinstance(symbols, dict)
    assert "functions" in symbols
    assert "classes" in symbols



if __name__ == "__main__":
    print("❌ ERROR: Do not run this test file directly!")
    print("   Use: python -m pytest tests/test_indexer.py")
    print("   See docs/TESTING_GUIDE.md for more information.")
    print()
    print("   If you want to run a quick test, use:")
    print("   python -c \"import sys; from pathlib import Path; sys.path.insert(0, str(Path('src'))); from agents.jules.indexer import RepositoryIndexer; print('Import successful!')\"")
    sys.exit(1)