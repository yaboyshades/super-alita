#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os
from pathlib import Path
from typing import Dict, Any, List

SAFE_ROOT = Path("./deepcode_lab").resolve()

def ensure_safe_path(path: str) -> Path:
    p = (SAFE_ROOT / path).resolve()
    if SAFE_ROOT not in p.parents and p != SAFE_ROOT:
        raise ValueError("Path containment violated (.. escape)")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def repo_download_integrity(repo_dir: str) -> Dict[str, Any]:
    d = Path(repo_dir)
    status = "ok"
    missing = []
    if not d.exists() or not any(d.iterdir()):
        status = "partial"; missing.append("empty_dir")
    if not (d/"README.md").exists():
        status = "partial"; missing.append("README.md")
    if not ((d/"pyproject.toml").exists() or (d/"setup.py").exists()):
        status = "partial"; missing.append("build_descriptor")
    if not ((d/"LICENSE").exists() or (d/"LICENSE.md").exists()):
        status = "partial"; missing.append("license")
    return {"status": status, "missing": missing}

def hash_top_files(repo_dir: str, top_n: int = 10) -> List[Dict[str, Any]]:
    d = Path(repo_dir)
    files = sorted([p for p in d.rglob("*") if p.is_file()])[:top_n]
    out = []
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        out.append({"path": str(f.relative_to(d)), "sha256_16": h})
    return out
