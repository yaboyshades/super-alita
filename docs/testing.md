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

Hypothesis checks:

- ID determinism under whitespace/case changes
- Hashing path for long content
- Sanity on tool/registry roundtrips
