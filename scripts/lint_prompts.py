#!/usr/bin/env python
import sys, re, json, pathlib

BANNED_PATTERNS = [
    r"OPENAI_API_KEY",
    r"internal_only",
    r"(?i)do\s+anything\s+now",
]

def scan_file(path: pathlib.Path) -> list[str]:
    issues = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pat in BANNED_PATTERNS:
        if re.search(pat, text):
            issues.append(f"{path}: pattern '{pat}' found")
    # Example injection heuristic
    if "IGNORE ALL PREVIOUS" in text.upper():
        issues.append(f"{path}: injection-like phrase detected")
    return issues

def main():
    roots = [pathlib.Path(p) for p in sys.argv[1:]]
    files = []
    for r in roots:
        if r.is_file():
            files.append(r)
        else:
            files.extend(r.glob("**/*"))
    issues = []
    for f in files:
        if f.suffix.lower() in {".md", ".txt", ".json"}:
            issues.extend(scan_file(f))
    if issues:
        print("Prompt lint issues:")
        print("\n".join(issues))
        sys.exit(1)
    print("Prompt lint passed.")

if __name__ == "__main__":
    main()