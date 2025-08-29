"""Test module for Jules Repository Indexer"""

import sys
import os
from pathlib import Path
import pytest

# Debug: Show the current sys.path
print("DEBUG: Current sys.path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# Verify that src directory is in sys.path
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
print(f"DEBUG: Repository root: {repo_root}")
print(f"DEBUG: Source path: {src_path}")
print(f"DEBUG: Source path exists: {src_path.exists()}")
print(f"DEBUG: Source path is in sys.path: {str(src_path) in sys.path}")

# Check if the jules module files exist
jules_init_path = src_path / "agents" / "jules" / "__init__.py"
jules_indexer_path = src_path / "agents" / "jules" / "indexer.py"
print(f"DEBUG: Jules __init__.py exists: {jules_init_path.exists()}")
print(f"DEBUG: Jules indexer.py exists: {jules_indexer_path.exists()}")

# Try to list the contents of the agents directory
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

# Test basic import capability
def test_debug_sys_path():
    """Test that we can see the sys.path correctly."""
    assert str(src_path) in sys.path, f"Source path {src_path} should be in sys.path"


def test_jules_module_files_exist():
    """Test that the Jules module files exist on the filesystem."""
    assert jules_init_path.exists(), f"Jules __init__.py should exist at {jules_init_path}"
    assert jules_indexer_path.exists(), f"Jules indexer.py should exist at {jules_indexer_path}"


def test_jules_indexer_import():
    """Test importing the Jules indexer module - this is where the issue occurs."""
    # This is the import that fails with ModuleNotFoundError
    try:
        from agents.jules.indexer import RepositoryIndexer
        print("SUCCESS: Import worked!")
        
        # Test basic functionality
        indexer = RepositoryIndexer(Path("/tmp"))
        assert indexer is not None
        assert indexer.repo_path == Path("/tmp")
        
    except ModuleNotFoundError as e:
        print(f"FAILED: ModuleNotFoundError: {e}")
        raise


def test_alternative_import_methods():
    """Test alternative import methods to debug the issue."""
    # Method 1: Try absolute import with full path
    try:
        import agents.jules.indexer
        print("SUCCESS: agents.jules.indexer import worked")
    except ModuleNotFoundError as e:
        print(f"FAILED: agents.jules.indexer import failed: {e}")
    
    # Method 2: Try importlib
    import importlib
    try:
        module = importlib.import_module("agents.jules.indexer")
        print("SUCCESS: importlib.import_module worked")
    except ModuleNotFoundError as e:
        print(f"FAILED: importlib.import_module failed: {e}")
    
    # Method 3: Try adding specific path to sys.path
    agents_jules_path = src_path / "agents" / "jules"
    if str(agents_jules_path) not in sys.path:
        sys.path.insert(0, str(agents_jules_path))
        try:
            import indexer
            print("SUCCESS: Direct indexer import worked")
        except ModuleNotFoundError as e:
            print(f"FAILED: Direct indexer import failed: {e}")


if __name__ == "__main__":
    # Run tests directly for debugging
    test_debug_sys_path()
    test_jules_module_files_exist()
    print("\nAttempting import...")
    test_jules_indexer_import()
    print("\nTrying alternative methods...")
    test_alternative_import_methods()