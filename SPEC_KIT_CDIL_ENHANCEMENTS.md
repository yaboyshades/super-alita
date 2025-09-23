# Spec Kit + CDIL Enhancements

This document describes the enhancements made to the Spec Kit + CDIL integration to make it "hard to bypass and easy to live with."

## 1. Hardening Components

### Pre-commit Hook

The pre-commit hook (`tools/cdil/pre-commit-hook.sh`) prevents commits with spec or signature drift:

- Checks for modified spec files and verifies spec lock
- Checks for modified source files and verifies signature lock
- Provides fast local feedback before CI runs

To install the pre-commit hook:
```bash
# Add to .git/hooks/pre-commit
cp tools/cdil/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### SARIF Emitter

The SARIF emitter (`tools/cdil/sarif-emitter.py`) generates standardized reports for CI and PR annotations:

- Creates SARIF reports for API differences and contract violations
- Generates PR comments summarizing changes
- Integrates with GitHub's code scanning alerts

### Function Implementation Tool

The function implementation tool (`tools/cdil/implement-function.py`) provides a one-button workflow:

- Assembles specification context (spec + stubs + validators)
- Prompts LLM for implementation
- Applies FunctionBodyPatch with AST guard
- Runs type checks and contract tests
- Provides validation report

## 2. Developer UX Improvements

### Editor Integration

Planned VS Code extension features:
- Show canonical signature above functions
- Warn on signature divergence
- Provide "Generate migration plan" quick-fix

### One-button Implementation

The `implement-function.py` script provides a complete workflow:
```bash
python tools/cdil/implement-function.py user.create src/user/create.py create_user
```

## 3. Security Enhancements

### Lock File Provenance

Planned enhancements for lock file signing:
- Sign `.contracts/spec.lock.json` and `.contracts/signature.lock.json`
- Use CI-managed keys (cosign or HMAC with GH secrets)
- Fail CI if signatures are missing or invalid

### SemVer Policy Gate

Planned SemVer enforcement:
- Require version bump in spec YAML for surface changes
- Map to package version bump in language ecosystem
- Fail CI if semver policy and diff don't match

## 4. Performance Optimizations

### Partial Extraction

Planned caching for faster verification:
- Only re-extract symbol graph for changed modules
- Cache results keyed by file hash
- Reduce cold start time for local workflows

### Mutation Testing Cadence

Planned mutation testing strategy:
- Per-PR: smoke on touched symbols only
- Nightly: full module suites
- Track mutation score trend with alerts

## 5. Specification Quality

### Spec Lint Rules

Planned specification linting:
- Enforce pre/post conditions for all public functions
- Require at least one example per function
- Validate generator mappings for composite types

### Golden I/O Vectors

Planned golden case persistence:
- Store golden cases under `tests/contract_goldens/`
- Use as regression anchors
- Provide readable documentation examples

## 6. Multi-language Support

### TypeScript Integration

Planned TypeScript support:
- Use `*.d.ts` as SSOT
- Extract with `ts-morph`
- Generate Zod validators and fast-check arbitraries

### Java/Kotlin Integration

Planned JVM support:
- Use Revapi for public surface changes
- Propagate `@Deprecated` annotations from spec

### Rust Integration

Planned Rust support:
- Use `cargo public-api`
- Export public API JSON
- Normalize into symbol graph

## 7. Observability & Audit

### Dashboards

Planned metrics to track:
- Contract Failures by Spec ID
- API Drift Attempts
- Mutation Score by Package
- P95 Compile Time of Spec Kit
- CI Wall Time for CDIL

### Traceability

Planned audit trail:
- Store spec hash, signature hash, test reports
- Track changed symbols per run
- Enable incident reconstruction

## 8. Rollout Plan

### Phase 1 (Current)
- Add spec+signature provenance signing
- Add SemVer gate

### Phase 2
- Add SARIF outputs + PR bot
- Add pre-commit verify for faster feedback

### Phase 3
- Editor tooling + partial extraction cache
- Nightly full mutation runs with score threshold alert

## 9. Checklists

### Definition of Done (new or changed API)
- [ ] Spec YAML updated; version bumped; lints pass
- [ ] Compiler regenerated stubs/validators/tests; spec.lock.json updated/signed
- [ ] Implementation via body-only patches; AST guard enforced
- [ ] signature.lock.json verified (or updated in a migration PR)
- [ ] Type checks, generated contract tests, golden tests pass
- [ ] (If breaking) Deprecation + codemod + sunset window captured

### Incident Response (drift detected)
- [ ] Capture failing lock diff + spec diff + CI artifacts
- [ ] If accidental: revert or re-generate body-only fix
- [ ] If intended: open spec-first migration PR with SemVer bump and codemod

## Conclusion

These enhancements make the Spec Kit + CDIL stack more robust and developer-friendly while maintaining the core principle of "can't break interfaces." The combination of fast local feedback, standardized reporting, and automated validation creates a system that's hard to bypass but easy to work with.