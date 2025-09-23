#!/usr/bin/env python3
"""Compare legacy streaming output with canonical event ledger."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path


def load_ndjson(path: Path) -> Iterator[dict[str, object]]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compare_streams(legacy: Iterable[dict[str, object]], canonical: Iterable[dict[str, object]]) -> dict[str, object]:
    legacy_events = list(legacy)
    canonical_events = list(canonical)

    legacy_count = len(legacy_events)
    canonical_count = len(canonical_events)

    legacy_kinds = Counter(event.get('type') for event in legacy_events)
    canonical_kinds = Counter(event.get('kind') for event in canonical_events)

    mismatched_kind_counts = {}
    for key in set(legacy_kinds) | set(canonical_kinds):
        if legacy_kinds.get(key) != canonical_kinds.get(key):
            mismatched_kind_counts[key] = {
                'legacy': legacy_kinds.get(key, 0),
                'canonical': canonical_kinds.get(key, 0),
            }

    sequences = [event.get('sequence') for event in canonical_events if 'sequence' in event]
    sequence_issues = {}
    if sequences and sequences != sorted(sequences):
        sequence_issues['out_of_order'] = True
    if sequences and sequences[0] != 0:
        sequence_issues['starts_at'] = sequences[0]

    termination = any(event.get('kind') in {'RunTerminated', 'RunFailed'} for event in canonical_events)

    return {
        'legacy_event_count': legacy_count,
        'canonical_event_count': canonical_count,
        'mismatched_kind_counts': mismatched_kind_counts,
        'sequence_issues': sequence_issues,
        'has_termination_event': termination,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare legacy stream output with canonical events.')
    parser.add_argument('--legacy', type=Path, required=True, help='Path to legacy NDJSON stream file.')
    parser.add_argument('--canonical', type=Path, required=True, help='Path to canonical NDJSON ledger file.')
    args = parser.parse_args()

    legacy_path = args.legacy
    canonical_path = args.canonical

    if not legacy_path.exists():
        raise SystemExit(f'Legacy stream file not found: {legacy_path}')
    if not canonical_path.exists():
        raise SystemExit(f'Canonical ledger file not found: {canonical_path}')

    report = compare_streams(load_ndjson(legacy_path), load_ndjson(canonical_path))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
