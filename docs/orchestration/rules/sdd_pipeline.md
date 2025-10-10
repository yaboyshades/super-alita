# SDD Pipeline Routing Rules

These instructions scope dynamic routing when the Spec-Driven Development pipeline is active.

- Prioritize planners defined in `src/sdd/router.py` and `src/sdd/enhanced_sdd_framework.py`.
- Pull contextual memories from `memory/sdd/constitutional_sdd_framework.md` and templates in `templates/sdd/`.
- Require regression coverage across `tests/runtime/` and `tests/sdd/` before merging planning changes.
- Invoke `.github/workflows/sdd-validation.yml` when the plan emits new execution steps.
