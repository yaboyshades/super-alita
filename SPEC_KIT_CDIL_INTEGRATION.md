# Spec Kit + CDIL Integration

This document describes the integration of Spec Kit with Contract-Driven Interface Locks (CDIL) in the Super Alita repository.

## Overview

The combination of Spec Kit and CDIL creates a robust system where:
- **Spec Kit** serves as the authoritative, human-readable specification
- **CDIL** enforces conformance at all stages (edit, build, test, merge)
- This transforms "don't break interfaces" into "can't break interfaces"

## Components

### 1. Spec Kit Compiler (`tools/spec-kit/compile.js`)

Compiles specifications to:
- Stub files (`.pyi`, `.d.ts`)
- Validator functions
- Test files (PBT strategies, golden cases)
- Lock files

### 2. CDIL Extractor (`tools/cdil/main.py`)

Extracts symbol graphs from code and verifies signatures against spec locks.

### 3. Specification Files (`specs/*.yaml`)

Human-readable specification files that define:
- API contracts
- Pre/post conditions
- Invariants
- Examples
- Policies
- Generators

### 4. Generated Artifacts

- **Stub files**: Type definitions in `contracts/` directory
- **Validators**: Pre/post condition checkers
- **Tests**: Property-based and example-based tests
- **Lock files**: `spec.lock.json` and `signature.lock.json`

## Workflow

1. **Author Time**:
   - Edit spec YAML files
   - Run Spec Kit compiler to generate stubs and tests
   - Commit updated lock files

2. **Edit/CI Time**:
   - CDIL extracts symbol graph from code
   - Verifies signature lock matches stubs
   - Runs generated contract tests
   - Blocks merge if any check fails

3. **Migration Flow**:
   - Edit spec YAML → bump version → recompile
   - Compiler emits new stubs + deprecation tags
   - Codemod recipe for call sites
   - Updated lock files

## Example

A simple `user.create` specification demonstrates the system:

```yaml
spec_id: user.create
version: 1.3.0
io:
  params:
    - name: data
      type: UserIn
      required: true
    - name: referrer
      type: string | null
      default: null
  returns: UserOut
contracts:
  pre:
    - "data.email is RFC5322-valid"
    - "len(data.name) > 0"
  post:
    - "result.id is nonempty"
    - "result.email == data.email"
    - "result.created_at is ISO8601Z"
```

This generates:
- Python stubs with type definitions
- Validator functions for pre/post conditions
- Property-based tests using Hypothesis
- Example-based tests

## CI Integration

The GitHub workflow (`.github/workflows/spec-cdil.yml`) runs:
1. Spec compilation
2. Spec lock verification
3. CDIL signature extraction and verification
4. Static type checks
5. Contract tests
6. Mutation testing

## Benefits

1. **Prevention of Interface Drift**: Lock files prevent unintended changes
2. **LLM Compliance**: Constrains AI assistants to only modify function bodies
3. **Automated Testing**: Property-based tests generated directly from specifications
4. **Gradual Adoption**: Legacy code can be incorporated incrementally
5. **Spec-First Changes**: All API changes must start with spec updates