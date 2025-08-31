# Mangle Integration for Enhanced Consensus Control

## New Capabilities

### 1. Output Validation Gates

- Detect policy violations in generated text
- Apply confidence penalties for non-compliant output
- Track violation details in metadata
- Use explicit violation rules in Mangle

### 2. Tool/Action Gating

- Pre-execution checks for high-risk tools
- Policy-based permission control with clear rejection reasons
- Context-sensitive authorization decisions
- Security guardrails for dangerous operations

### 3. Dynamic Consensus Method Selection

- Rule-based selection of optimal consensus method
- Domain-specific method optimization
- Sample count and metadata consideration
- Override default weighted_vote when appropriate

### 4. LLM Claim Verification

- Post-hoc verification of factual claims
- Identify problematic or incorrect statements
- Domain-specific claim validation
- Apply confidence adjustments based on validity

## Implementation Components

1. `mangle_validator.py` - Core validator implementation
2. `integration.py` - Integration with DeepConf consensus
3. `mangle_rules_example.json` - Example rule definitions
4. `VALIDATION_README.md` - Documentation and usage guide
5. `mangle_validation_example.py` - Usage examples

## Rule Development

Rules are defined in `./data/mangle/rules.json` using Mangle's logic programming syntax:

```json
{
  "rule_id": {
    "name": "Human-readable name",
    "description": "Description of what the rule does",
    "body": "predicate(X) :- condition1(X), condition2(Y).",
    "tags": ["category1", "category2"]
  }
}
```

### Key Rule Types

- **Confidence adjustment**: `adjust_confidence(New) :- base_confidence(C), ...`
- **Policy violations**: `violation(Reason) :- output_text(Text), ...`
- **Tool authorization**: `deny_tool(Reason) :- tool_name(Name), ...`
- **Method selection**: `select_method(Method, Reason) :- domain(D), ...`
- **Claim verification**: `invalid_claim(Claim, Reason) :- output_text(Text), ...`

## Usage Examples

```python
# Initialize validator
mangle = MangleAbility()
validator = MangleValidator(mangle)

# Validate output against policy rules
validation = await validator.validate_output(
    output_text="Generated text...",
    domain="finance"
)

# Check if tool execution is allowed by policy
authorization = await validator.validate_tool_execution(
    tool_name="file_write",
    params={"path": "/etc/passwd"}
)

# Select optimal consensus method
method = await validator.select_consensus_method(
    domain="programming",
    sample_count=4
)

# Verify claims in output
verification = await validator.verify_llm_claims(
    output_text="Python 3.9 introduced...",
    claims_type="software"
)
```

## DeepConf Integration

```python
from src.abilities.mangle.integration import integrate_mangle_with_deepconf

# Enhance DeepConf with Mangle validation
enhanced_deepconf = integrate_mangle_with_deepconf(deepconf_instance)
```

This adds automatic:

- Method selection based on domain and request context
- Output validation with confidence penalty application
- Claim verification for appropriate domains
- Detailed validation metadata in responses
