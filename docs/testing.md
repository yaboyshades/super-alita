# Testing

## With pytest (recommended)

```
pip install -r requirements.txt
pip install hypothesis   # optional
PYTHONPATH=src pytest -q
```

### Property-based Tests

Hypothesis checks:

- ID determinism under whitespace/case changes
- Hashing path for long content
- Sanity on tool/registry roundtrips
