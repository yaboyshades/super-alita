#!/usr/bin/env python3
from pathlib import Path; import shutil
def move_if_exists(src: Path, dst: Path):
    if src.exists(): dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst))
def main():
    legacy = Path("archive/legacy_startup"); demos = Path("examples/demos")
    for f in ["start_super_alita.py","start_with_20b.py","start_mangle.py"]: move_if_exists(Path(f), legacy/f)
    for f in ["demo_github_integration.py","demo_perplexica_integration.py"]: move_if_exists(Path(f), demos/f)
if __name__=="__main__": main()