---
# /sync_spec Mini-Protocol

**Inputs:**
- feature_slug
- files_changed
- impacted_contracts

**Outputs:**
- deltas
- impacted_tests
- required_gates

**Flow:**
1. Detect changes to spec artifacts
2. Emit deltas (JSON) for code/tests to update
3. Block merge if gates not met

**Article Coverage:**
- Article XI: Spec/code parity
- Article XV: Definition of Done

---