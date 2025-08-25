# Testing

> For environment setup, see the [project README](../README.md).

## With pytest (recommended)

```
make deps
make test

# Optional property-based tests
pip install hypothesis
PYTHONPATH=src pytest -q
```

### Property-based Tests


# Testing

## Setup and Test Commands

```bash
make deps               # install runtime + test deps (CPU only)
# For GPU acceleration, install extras: pip install -r requirements-gpu.txt (optional)
make lint               # run pre-commit hooks
pre-commit run --all-files
make test               # run full test suite (target ≥70% coverage)
```

Coverage should remain at or above 70%; use `pytest --cov -q` to check locally.

### Property-based Tests


Hypothesis checks:

- ID determinism under whitespace/case changes
- Hashing path for long content
- Sanity on tool/registry roundtrips
