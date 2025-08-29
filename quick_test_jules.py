#!/usr/bin/env python3
"""
Quick test utility for Jules agent development.

This script properly sets up the Python path and allows testing
the Jules indexer without running through pytest.

Usage:
    python quick_test_jules.py
"""

import sys
from pathlib import Path

# Add src to path for standalone execution
repo_root = Path(__file__).parent
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def main():
    """Test Jules indexer functionality."""
    print("🔍 Quick Jules Agent Test")
    print("=" * 40)
    
    try:
        from agents.jules.indexer import RepositoryIndexer
        print("✅ Import successful!")
        
        # Test basic functionality
        print(f"📁 Testing with repository: {repo_root}")
        indexer = RepositoryIndexer(repo_root)
        
        print("🔄 Indexing repository...")
        result = indexer.index_repository()
        
        print(f"📊 Results:")
        print(f"   - Files found: {len(result['files'])}")
        print(f"   - Directories: {len(result['directories'])}")
        print(f"   - Python modules: {len(result['python_modules'])}")
        
        # Test search functionality
        search_results = indexer.search_code("indexer")
        print(f"   - Search results for 'indexer': {len(search_results)}")
        
        print("✅ All tests passed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Try running through pytest instead:")
        print("   python -m pytest tests/test_indexer.py -v")
        return 1
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())