---
# /sync_code Mini-Protocol

**Inputs:**
- feature_slug
- files_changed
- impacted_contracts

**Outputs:**
- deltas
- impacted_tests
- required_gates

**Flow:**
1. Detect code changes
2. Emit deltas for spec/tests to catch up
3. Block merge if parity not met

**Article Coverage:**
- Article XI: Spec/code parity
- Article XV: Definition of Done

---