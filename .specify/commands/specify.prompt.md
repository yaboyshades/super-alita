```prompt
# /specify
- Run `.specify/scripts/create-new-feature.sh --json "$ARG"` to create `NNN-<slug>` branch and scaffold `spec.md`.
- Output JSON: BRANCH_NAME, FEATURE_NUM, SPEC_FILE, FEATURE_DIR (absolute paths).
- The repository uses a canonical spec generation mode by default; generated specs include Implementation Readiness and Test Scenarios. If you pass a JSON payload include `{"description":"...","author":"...","mode":"..."}` — `mode` will be ignored and logged (canonical mode enforced).
- After creation, open SPEC_FILE for editing.

```
# /specify
- Run `.specify/scripts/create-new-feature.sh --json "$ARG"` to create `NNN-<slug>` branch and scaffold `spec.md`.
- Output JSON: BRANCH_NAME, FEATURE_NUM, SPEC_FILE, FEATURE_DIR (absolute paths).
- After creation, open SPEC_FILE for editing.
