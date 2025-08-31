# Generic Autogen (Need → Generate → Gate → Iterate → Apply)

**What it does**
- Detects needed capability kinds from task text (policy patterns).
- Builds strict requirements for DeepCode.
- Enforces gates: *required artifacts*, *safety scan*, *pytest pass*.
- Iterates with explicit "Fix:" refinements until gates pass, then applies.
- Emits EventBus events for telemetry and learning:
  - `autogen.started|iteration_checked|refine|applied|failed|skipped`
  - **OaK**: `oak.plan_proposed`, `oak.outcome_feedback`
  - **Bandit**: `bandit.reward_event {reward∈[0,1]}`

## CLI Usage

```bash
python scripts/run_autogen.py --desc "we need to extract product prices from ecommerce pages into a csv"
```

## Integration

- Enable `AutogenCreatorPlugin` so gap events automatically trigger autogen.
- Gap topics: `capability_gap_detected`, `atom_gap_request`, `knowledge_gap_detected`.

## Capability Templates

The system recognizes these capability kinds:

### web_scraper
Patterns: "scrape", "extract price/product", "e-commerce", "price tracker"
Generates: Generic web scraping ability with configurable selectors

### etl_task
Patterns: "ETL", "data pipeline", "data ingestion", "normalize"
Generates: Configurable data transformation pipeline

### api_client
Patterns: "api client", "call api", "fetch via http/rest"
Generates: Typed API client with retry/auth capabilities

### report_generator
Patterns: "report", "summary", "export csv/xlsx/pdf"
Generates: Data-to-report transformation ability

## Safety Gates

All generated code must pass:

1. **RequiredPathsGate**: Enforces presence of implementation, tests, and docs
2. **SafetyGate**: Blocks dangerous patterns (eval, os.system, shell=True)
3. **PytestGate**: All tests must pass before applying changes

## Event Integration

### OaK Integration
- `oak.plan_proposed`: Notifies planning layer of available autogen option
- `oak.outcome_feedback`: Reports success/failure for option evaluation

### Bandit Learning
- `bandit.reward_event`: Emits normalized rewards (1.0=success, 0.0=failure)

### Telemetry Events
- `autogen.started`: Beginning capability generation
- `autogen.iteration_checked`: Gate validation result
- `autogen.refine`: Adding requirements based on gate failures
- `autogen.applied`: Successfully applied generated capability
- `autogen.failed`: Could not satisfy gates within iteration limit
- `autogen.skipped`: No recognizable capability patterns found

## Examples

### Manual Trigger
```bash
# Generate a web scraper ability
python scripts/run_autogen.py --desc "scrape product listings from ecommerce sites"

# Generate an ETL pipeline
python scripts/run_autogen.py --desc "normalize customer data from CSV files"
```

### Programmatic Usage
```python
from src.pipelines.autogen_pipeline import autogen_any

# Basic usage
result = autogen_any("need to generate reports from sales data")

# With custom settings
result = autogen_any(
    description="create api client for payment processing",
    repo_path="/path/to/project",
    iterations=3,
    event_bus=my_event_bus,
    api=my_local_api
)
```

### Plugin Integration
The `AutogenCreatorPlugin` automatically responds to capability gap events:

```python
# Publish a gap event to trigger autogen
from src.core.events import create_event

bus.publish(create_event(
    event_type="capability_gap_detected",
    description="need robust price extractor"
))
```

## Configuration

Add to your plugin configuration to enable automatic gap handling:

```python
# In your plugin manager setup
plugins = [
    AutogenCreatorPlugin(),
    # ... other plugins
]
```

## Integration with Super Alita

The autogen system integrates with Super Alita's existing infrastructure:

- **DeepCode**: Uses `/deepcode/*` endpoints for code generation
- **Testing**: Leverages `/tools/pytest_run` for validation
- **Security**: Calls `/ability/execute/secure_scan_code` for safety
- **Events**: Publishes to the existing EventBus system
- **Plugins**: Implements standard `PluginInterface`

This ensures autogen capabilities are available throughout the agent ecosystem while maintaining security and quality standards.
