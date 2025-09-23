# Quickstart: Calculus Runtime Derivative Gate

**Phase 1 Quickstart** | **Date**: 2025-09-16 | **Plan**: specs/018-calculus-runtime-derivative-gate/plan.md

## Overview

This guide walks through running the calculus gate locally to analyze function runtime derivatives and detect performance regressions through mathematical analysis.

## Prerequisites

- Python 3.11+
- Git repository with target functions
- Virtual environment (recommended)

## Installation

### 1. Install Dependencies

```bash
pip install numpy scipy matplotlib rich pytest hypothesis
```

### 2. Clone/Navigate to Repository

```bash
cd /path/to/super-alita-clean
```

### 3. Set Up Environment

```bash
# Create virtual environment if needed
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install project dependencies
pip install -r requirements.txt
```

## Quick Start: Analyzing a Function

### Step 1: Basic Analysis

Analyze a function's runtime derivatives:

```bash
python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --verbose
```

Expected output:
```
🔬 Analyzing function 'search_documents' in src/core/search.py
📊 Sampling runtime across input sizes...

🏛️  CALCULUS GATE ANALYSIS RESULTS
==================================================
📊 Function: search_documents
📈 Slope Gate: ✅ PASS
📐 Curvature Gate: ✅ PASS
📏 Lipschitz Gate: ✅ PASS
🎯 Overall Grade: A
--------------------------------------------------
🎉 PERFORMANCE APPROVED: Mathematical bounds satisfied!
```

### Step 2: Save Certificate

Generate a JSON certificate for CI/MCP integration:

```bash
python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --output artifacts/calculus_gate/search_certificate.json \
    --fail-on-violation
```

This creates:
- `artifacts/calculus_gate/search_certificate.json` - Full certificate
- Console output for human review
- Exit code 0 (pass) or 1 (fail) for CI integration

### Step 3: Custom Thresholds

Configure performance thresholds for strict monitoring:

```bash
python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --slope-limit 1.0 \
    --curvature-limit 0.5 \
    --lipschitz-limit 5.0 \
    --verbose
```

## Understanding Results

### Grade System

- **Grade A**: All gates pass, excellent performance characteristics
- **Grade B**: Minor violations, acceptable with monitoring
- **Grade F**: Major violations, requires investigation

### Gate Meanings

- **Slope Gate**: Monitors |df/dn| to detect constant-time violations
- **Curvature Gate**: Monitors |d²f/dn²| for complexity class changes
- **Lipschitz Gate**: Monitors sensitivity for performance stability

### Violation Examples

If you see violations:

```
⚠️  PERFORMANCE VIOLATIONS: Review derivative analysis.

📈 Slope Violations:
   Size 1000: |df/dn| = 3.456789 > 2.0

📐 Curvature Changes:
   Size 5000: |d²f/dn²| = 1.234567 > 1.0

📏 Lipschitz Violation:
   Constant = 15.678901 > 10.0
```

## Certificate Structure

The JSON certificate contains:

```json
{
  "function_name": "search_documents",
  "timestamp": 1694865600.0,
  "commit_hash": "HEAD",
  "passes_slope_gate": true,
  "passes_curvature_gate": true,
  "passes_lipschitz_gate": true,
  "overall_pass": true,
  "certificate_grade": "A",
  "analysis": {
    "input_sizes": [1, 2, 4, 8, 16, ...],
    "runtime_values": [0.001, 0.002, 0.004, ...],
    "first_derivative": [0.1, 0.15, 0.18, ...],
    "second_derivative": [0.01, 0.02, 0.01, ...],
    "lipschitz_constant": 5.2
  }
}
```

## Integration Examples

### CI Integration (GitHub Actions)

```yaml
# .github/workflows/calculus-gate.yml
name: Calculus Gate
on: [push, pull_request]

jobs:
  performance-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install numpy scipy matplotlib rich

    - name: Run calculus gate
      run: |
        python .vscode/copilot-middleware/calculus_gate_cli.py \
          src/core/search.py \
          --function search_documents \
          --output artifacts/search_certificate.json \
          --fail-on-violation

    - name: Upload certificate
      uses: actions/upload-artifact@v3
      with:
        name: calculus-certificates
        path: artifacts/calculus_gate/
```

### MCP Integration

Start MCP server for Copilot integration:

```python
# Example MCP endpoint usage
import requests

response = requests.post('http://localhost:8080/calculus/analyze', json={
    "function_path": "src/core/search.py",
    "function_name": "search_documents",
    "commit_hash": "abc123"
})

result = response.json()
print(f"Grade: {result['analysis_summary']['overall_grade']}")
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
echo "Running calculus gate analysis..."

python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --fail-on-violation \
    --quiet

if [ $? -ne 0 ]; then
    echo "❌ Performance regression detected. Commit blocked."
    exit 1
fi

echo "✅ Performance analysis passed."
```

## Troubleshooting

### Common Issues

**Error: "Need ≥3 unique input sizes"**
```bash
# Increase sample count or adjust size range
--samples 25 --min-size 1 --max-size 50000
```

**Error: "Curve fitting failed"**
```bash
# Use more conservative sampling
--samples 15 --max-size 10000
```

**High noise warnings**
```bash
# Increase warmup runs or use smaller range
--samples 30 --max-size 5000
```

### Debug Mode

For detailed analysis:

```bash
python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --verbose \
    --output debug_certificate.json

# Examine certificate for detailed metrics
cat debug_certificate.json | jq '.analysis.measurement_noise'
```

### Performance Tuning

For faster analysis:

```bash
# Reduce sample count for development
python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --samples 10 \
    --max-size 1000
```

For production monitoring:

```bash
# Comprehensive analysis
python .vscode/copilot-middleware/calculus_gate_cli.py \
    src/core/search.py \
    --function search_documents \
    --samples 30 \
    --max-size 100000 \
    --slope-limit 1.0 \
    --output production_certificate.json
```

## Next Steps

1. **Configure Monitoring**: Set up automated analysis for critical functions
2. **Establish Baselines**: Run analysis on known-good builds to establish performance baselines
3. **CI Integration**: Add calculus gate to your build pipeline
4. **Dashboard Integration**: Connect certificates to observability dashboards
5. **Team Training**: Share results with team for performance awareness

## Advanced Usage

### Multiple Function Analysis

```bash
# Analyze multiple functions
for func in search_documents process_data filter_results; do
    python .vscode/copilot-middleware/calculus_gate_cli.py \
        src/core/search.py \
        --function $func \
        --output "artifacts/calculus_gate/${func}_certificate.json"
done
```

### Batch Processing

```bash
# Create analysis script
cat << 'EOF' > analyze_all.sh
#!/bin/bash
FUNCTIONS=(
    "src/core/search.py:search_documents"
    "src/core/data.py:process_dataset"
    "src/utils/helpers.py:sort_items"
)

for entry in "${FUNCTIONS[@]}"; do
    IFS=':' read -r file func <<< "$entry"
    echo "Analyzing $func in $file..."

    python .vscode/copilot-middleware/calculus_gate_cli.py \
        "$file" \
        --function "$func" \
        --output "artifacts/calculus_gate/${func}_certificate.json" \
        --fail-on-violation
done
EOF

chmod +x analyze_all.sh
./analyze_all.sh
```

**Quickstart Status**: ✅ COMPLETE
**Integration Examples**: ✅ PROVIDED
**Troubleshooting Guide**: ✅ INCLUDED
**Ready for Implementation**: ✅ YES
