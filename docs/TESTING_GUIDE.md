# Testing Guide for Super Alita

## Running Tests Correctly

**IMPORTANT**: Always run tests through pytest to ensure proper module import paths.

### Correct Test Execution

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_indexer.py

# Run specific test function
python -m pytest tests/test_indexer.py::test_jules_indexer_import

# Run with verbose output
python -m pytest tests/test_indexer.py -v

# Run with stdout/stderr output
python -m pytest tests/test_indexer.py -s
```

### ❌ Incorrect Test Execution (Will Cause Import Errors)

```bash
# DON'T run tests directly - this bypasses conftest.py path setup
python tests/test_indexer.py

# DON'T use python -c to import modules without proper path setup
python -c "from agents.jules.indexer import RepositoryIndexer"
```

## Import Path Setup

The project uses a `src/` directory structure. The `tests/conftest.py` file automatically adds the `src` directory to Python's import path when running through pytest:

```python
# From tests/conftest.py lines 27-30
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
```

## Troubleshooting Import Issues

### Common Issue: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'agents.jules.indexer'`

**Solution**: Use pytest instead of running Python scripts directly.

### Debugging Steps

1. **Verify files exist**:
   ```bash
   ls -la src/agents/jules/
   # Should show: __init__.py, indexer.py
   ```

2. **Check if running through pytest**:
   ```bash
   python -m pytest tests/test_indexer.py::test_debug_sys_path -s
   # Should show src path in sys.path
   ```

3. **Verify environment setup**:
   ```bash
   cd /path/to/super-alita
   cp .env.example .env
   pip install -r requirements.txt -r requirements-test.txt
   ```

## Project Structure

```
super-alita/
├── src/                    # Source code (added to sys.path by conftest.py)
│   ├── agents/
│   │   └── jules/
│   │       ├── __init__.py
│   │       └── indexer.py
│   └── ...
├── tests/                  # Test files
│   ├── conftest.py        # Pytest configuration and path setup
│   ├── test_indexer.py    # Jules indexer tests
│   └── ...
└── pytest.ini            # Pytest configuration
```

## Best Practices

1. **Always use pytest**: Never run test files directly as Python scripts
2. **Use relative imports in tests**: Import from the module name, not file paths
3. **Check conftest.py**: When adding new modules, ensure conftest.py includes necessary path setup
4. **Environment setup**: Always run `cp .env.example .env` and install dependencies before testing