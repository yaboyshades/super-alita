# Mangle Validation Integration for Super Alita

This integration enhances Super Alita with advanced validation and policy enforcement capabilities using Mangle's deductive database system.

## Features

### 1. Output Validation Gates

Use Mangle rules to detect policy violations in generated outputs and apply confidence penalties:

```python
validation_result = await mangle_validator.validate_output(
    output_text="Your financial advice text...",
    domain="finance",
    meta={"samples_valid": 3, "response_length": 250}
)

if not validation_result["valid"]:
    print(f"Violations: {validation_result['violations']}")
    print(f"Confidence penalty: {validation_result['confidence_penalty']}")
```

Example rule (in `./data/mangle/rules.json`):

```
"detect_financial_advice_violation": {
  "name": "Policy Violation - Financial Advice",
  "body": "violation('Financial advice requires disclaimer') :- domain('finance'), output_text(Text), contains_substring(Text, 'invest'), not(contains_substring(Text, 'not financial advice'))."
}
```

### 2. Tool/Action Gating

Pre-execution checks for high-risk tools to prevent unsafe operations:

```python
authorization = await mangle_validator.validate_tool_execution(
    tool_name="file_write",
    params={"path": "/etc/passwd", "content": "new_user:x:0:0::/:/bin/bash"},
    context={"domain": "finance", "user_role": "standard"}
)

if not authorization["authorized"]:
    print(f"Tool execution blocked: {authorization['reason']}")
```

Example rule:

```
"deny_filesystem_tool_for_financial": {
  "name": "Tool Policy - Restrict Filesystem Access for Financial",
  "body": "deny_tool('File system write operations not permitted for financial data') :- tool_name('file_write'), context('domain', 'finance')."
}
```

### 3. Consensus Method Selection

Dynamically select the best consensus method based on context:

```python
method_selection = await mangle_validator.select_consensus_method(
    domain="programming",
    sample_count=4,
    meta={"temperature": 0.7, "max_tokens": 500}
)

print(f"Selected method: {method_selection['method']}")
print(f"Reason: {method_selection['reason']}")
```

Example rule:

```
"select_ensemble_ranking_for_technical": {
  "name": "Method Selection - Ensemble for Technical",
  "body": "select_method('ensemble_ranking', 'Better precision for technical content') :- domain(D), (D = 'technical' ; D = 'programming' ; D = 'scientific'), sample_count(N), N >= 3."
}
```

### 4. LLM Claim Verification

Validate claims made in model outputs against known facts:

```python
verification = await mangle_validator.verify_llm_claims(
    output_text="Python 3.9 introduced the match statement...",
    claims_type="software"
)

if not verification["verified"]:
    print(f"Invalid claims: {verification['invalid_claims']}")
    print(f"Confidence adjustment: {verification['confidence_adjustment']}")
```

Example rule:

```
"invalid_claim_version_mismatch": {
  "name": "Claim Verification - Version Numbers",
  "body": "invalid_claim('Software version claim is outdated', 'Referenced version is not current') :- output_text(Text), claims_type('software'), contains_substring(Text, 'Python 3.9'), not(contains_substring(Text, 'newer versions are available'))."
}
```

## Integration with DeepConf

The Mangle validation system is designed to integrate seamlessly with Super Alita's DeepConf consensus system:

```python
from src.abilities.mangle.integration import integrate_mangle_with_deepconf

# Enhance a DeepConf instance with Mangle validation
enhanced_deepconf = integrate_mangle_with_deepconf(deepconf_instance)

# Now consensus generation will automatically:
# 1. Select the best consensus method using Mangle rules
# 2. Validate the output against policy rules
# 3. Apply confidence penalties for violations
# 4. Add metadata about validation results
```

## Rule Development

Mangle rules are stored in `./data/mangle/rules.json` as a collection of named rules:

```json
{
  "rule_id": {
    "name": "Human-readable name",
    "description": "Description of what the rule does",
    "body": "actual_predicate(X) :- condition1(X), condition2(Y).",
    "created_at": "ISO date string",
    "tags": ["category1", "category2"]
  }
}
```

### Key Predicates

1. **Output Validation**:

   - `violation(Reason)` - Define conditions that constitute a policy violation
   - `domain(D)` - Match specific domains
   - `output_text(Text)` - The generated text to validate

2. **Tool Gating**:

   - `deny_tool(Reason)` - Define conditions to block tool execution
   - `tool_name(Name)` - Match specific tool names
   - `param(Key, Value)` - Match tool parameters
   - `context(Key, Value)` - Match execution context

3. **Method Selection**:

   - `select_method(Method, Reason)` - Select a consensus method
   - `sample_count(N)` - Number of samples generated
   - Various metadata like domain, temperature, etc.

4. **Claim Verification**:
   - `invalid_claim(Claim, Reason)` - Define conditions for invalid claims
   - `claims_type(Type)` - Type of claims being verified (factual, code, etc.)

### Helper Predicates

- `contains_substring(String, Substring)` - Check if a string contains a substring

## Setup Instructions

1. Ensure Mangle is properly installed:

   ```
   export MANGLE_BIN_PATH=/path/to/mangle
   ```

2. Create the rules directory:

   ```
   mkdir -p ./data/mangle
   ```

3. Copy the example rules:

   ```
   cp ./examples/mangle_rules_example.json ./data/mangle/rules.json
   ```

4. Import the validator in your code:

   ```python
   from src.abilities.mangle.mangle_ability import MangleAbility
   from src.abilities.mangle.mangle_validator import MangleValidator

   mangle = MangleAbility()
   validator = MangleValidator(mangle)
   ```

## Example Rules

See `./examples/mangle_rules_example.json` for complete examples of all rule types.
