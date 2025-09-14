#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: pel_update_yaml.py <file.yaml>")
        sys.exit(1)

    p = Path(sys.argv[1])
    y = yaml.safe_load(p.read_text())
    y.setdefault("provenance", {})["updated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    p.write_text(yaml.safe_dump(y, sort_keys=False))
    print("PEL updated:", p)


if __name__ == "__main__":
    main()
