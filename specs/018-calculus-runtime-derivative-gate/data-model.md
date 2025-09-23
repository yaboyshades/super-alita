# Data Model: Calculus Runtime Derivative Gate

**Phase 1 Design** | **Date**: 2025-09-16 | **Plan**: specs/018-calculus-runtime-derivative-gate/plan.md

## Core Entities

### TargetFunction
Represents a function to monitor with its configuration and metadata.

```python
@dataclass
class TargetFunction:
    """Configuration for a function to be monitored by calculus gate."""

    # Identity
    name: str                    # Function name (e.g., "search_documents")
    file_path: str              # Absolute path to source file
    module_path: str            # Python import path (e.g., "src.core.search")

    # Sampling Configuration
    min_input_size: int = 1     # Minimum input size for testing
    max_input_size: int = 10000 # Maximum input size for testing
    sample_count: int = 20      # Number of sample points
    warmup_runs: int = 3        # Warmup iterations per sample

    # Threshold Configuration
    slope_limit: float = 2.0         # Max |df/dn| before violation
    curvature_limit: float = 1.0     # Max |d²f/dn²| before violation
    lipschitz_limit: float = 10.0    # Max Lipschitz constant

    # Input Generation Strategy
    input_generator: str = "default"  # Strategy name for generating test inputs
    input_config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime
    updated_at: datetime
    active: bool = True
```

### RuntimeSampleSet
Collection of runtime measurements for a specific function and build.

```python
@dataclass
class RuntimeSampleSet:
    """Set of runtime measurements for derivative analysis."""

    # Identity
    target_function: str        # Function name
    build_id: str              # Git commit hash or build identifier
    measurement_timestamp: datetime

    # Measurement Data
    input_sizes: List[int]      # Input sizes tested (strictly increasing)
    wall_times: List[float]     # Wall clock times in seconds
    cpu_times: List[float]      # CPU times in seconds
    memory_peaks: List[int]     # Peak memory usage in bytes
    memory_deltas: List[int]    # Memory allocated during execution

    # Sampling Metadata
    warmup_runs: int           # Number of warmup runs per sample
    measurement_conditions: Dict[str, Any]  # System state during measurement

    # Quality Indicators
    measurement_noise: float   # Coefficient of variation across runs
    convergence_achieved: bool # Whether measurements stabilized
    outliers_removed: int     # Number of outlier measurements dropped
```

### DerivativeCertificate
Complete analysis results with mathematical validation and compliance status.

```python
@dataclass
class DerivativeCertificate:
    """Mathematical certificate of runtime derivative analysis."""

    # Identity and Tracking
    function_name: str
    build_id: str
    analysis_timestamp: datetime
    certificate_version: str = "1.0"

    # Analysis Results
    sample_set: RuntimeSampleSet
    fitted_curve_params: Dict[str, Any]     # Spline coefficients or fit parameters

    # Mathematical Derivatives
    first_derivatives: List[float]         # df/dn at each input size
    second_derivatives: List[float]        # d²f/dn² at each input size
    lipschitz_constant: float             # Max |f(x1)-f(x2)|/|x1-x2|

    # Statistical Confidence
    derivative_confidence_intervals: List[Tuple[float, float]]  # 95% CI for df/dn
    curvature_confidence_intervals: List[Tuple[float, float]]   # 95% CI for d²f/dn²
    bootstrap_iterations: int = 1000       # Bootstrap sample count

    # Compliance Assessment
    slope_violations: List[Tuple[int, float]]     # (input_size, df_dn_value)
    curvature_violations: List[Tuple[int, float]] # (input_size, d2f_dn2_value)
    lipschitz_violation: bool                     # True if exceeds limit

    # Thresholds Applied
    slope_limit: float
    curvature_limit: float
    lipschitz_limit: float

    # Quality Gates
    passes_slope_gate: bool
    passes_curvature_gate: bool
    passes_lipschitz_gate: bool
    overall_compliance: bool
    certificate_grade: str  # "A", "B", or "F"

    # Analysis Quality
    fitting_method: str        # "cubic_spline", "savgol_fallback", etc.
    fitting_quality_score: float   # R² or similar goodness-of-fit
    noise_handling_applied: bool   # Whether noise mitigation was used

    # Historical Context
    baseline_comparison: Optional[str]  # Previous certificate ID for comparison
    trend_analysis: Dict[str, Any]      # Trend indicators vs baseline
```

### AlertEvent
Structured event emitted when thresholds are violated.

```python
@dataclass
class AlertEvent:
    """Alert event for CI/MCP consumption when violations occur."""

    # Event Identity
    event_id: str              # UUID for tracking
    event_type: str            # "slope_violation", "curvature_violation", "lipschitz_violation"
    timestamp: datetime
    severity: str              # "warning", "error", "critical"

    # Context
    function_name: str
    build_id: str
    certificate_id: str        # Reference to full certificate

    # Violation Details
    threshold_name: str        # "slope_limit", "curvature_limit", "lipschitz_limit"
    threshold_value: float     # Configured limit that was exceeded
    actual_value: float        # Measured value that exceeded limit
    violation_magnitude: float # How much the limit was exceeded (ratio)

    # Location Information
    input_size_at_violation: Optional[int]  # Input size where violation occurred
    derivative_type: str       # "first", "second", "lipschitz"

    # Actionable Information
    suggested_actions: List[str]  # Recommended next steps
    related_files: List[str]      # Files that might need review

    # Integration Fields
    ci_failure_recommended: bool  # Whether CI should fail
    mcp_notification_sent: bool   # MCP delivery status
    dashboard_alert_level: str    # For observability dashboards
```

## Entity Relationships

```
TargetFunction (1) -----> (N) RuntimeSampleSet
                                    |
                                    | (1)
                                    v
RuntimeSampleSet (1) -----> (1) DerivativeCertificate
                                    |
                                    | (0..N)
                                    v
DerivativeCertificate (1) ---> (N) AlertEvent
```

## Data Flow

1. **Configuration**: `TargetFunction` defines what to monitor and how
2. **Measurement**: `RuntimeSampleSet` captures raw performance data
3. **Analysis**: `DerivativeCertificate` computes derivatives and compliance
4. **Alerting**: `AlertEvent` triggers when violations occur

## Storage Schema

### File System Layout
```
artifacts/calculus_gate/
├── functions/
│   └── {function_name}/
│       ├── config.json          # TargetFunction serialized
│       └── samples/
│           └── {build_id}/
│               ├── measurements.json    # RuntimeSampleSet
│               ├── certificate.json     # DerivativeCertificate
│               ├── alerts.json         # AlertEvent[]
│               └── plots/              # Optional visualizations
│                   ├── runtime_curve.png
│                   └── derivatives.png
└── baselines/
    └── {function_name}_baseline.json   # Historical baseline for comparison
```

### JSON Schema Examples

#### TargetFunction Schema
```json
{
  "name": "search_documents",
  "file_path": "/repo/src/core/search.py",
  "module_path": "src.core.search",
  "min_input_size": 1,
  "max_input_size": 10000,
  "sample_count": 20,
  "slope_limit": 2.0,
  "curvature_limit": 1.0,
  "lipschitz_limit": 10.0,
  "input_generator": "document_list",
  "input_config": {"doc_type": "synthetic"},
  "created_at": "2025-09-16T12:00:00Z",
  "active": true
}
```

#### DerivativeCertificate Schema
```json
{
  "function_name": "search_documents",
  "build_id": "abc123def",
  "analysis_timestamp": "2025-09-16T12:30:00Z",
  "certificate_version": "1.0",
  "first_derivatives": [0.1, 0.15, 0.18, ...],
  "second_derivatives": [0.01, 0.02, 0.01, ...],
  "lipschitz_constant": 5.2,
  "derivative_confidence_intervals": [[0.08, 0.12], [0.13, 0.17], ...],
  "slope_violations": [],
  "curvature_violations": [],
  "lipschitz_violation": false,
  "passes_slope_gate": true,
  "passes_curvature_gate": true,
  "passes_lipschitz_gate": true,
  "overall_compliance": true,
  "certificate_grade": "A",
  "fitting_method": "cubic_spline",
  "fitting_quality_score": 0.98
}
```

## Data Validation Rules

### TargetFunction Validation
- `name` must be valid Python identifier
- `file_path` must exist and be readable
- `min_input_size` < `max_input_size`
- `sample_count` ≥ 3 for meaningful analysis
- All limits must be positive numbers

### RuntimeSampleSet Validation
- `input_sizes` must be strictly increasing
- All measurement arrays must have same length
- `wall_times` and `cpu_times` must be non-negative
- `measurement_noise` should be < 0.5 for reliable analysis

### DerivativeCertificate Validation
- Must have valid `sample_set` reference
- Derivative arrays must match input size count
- Confidence intervals must be valid ranges
- `certificate_grade` must be "A", "B", or "F"
- `overall_compliance` must match individual gate results

## Performance Considerations

### Memory Usage
- Stream processing for large sample sets (>1000 points)
- Lazy loading of historical certificates
- Configurable retention policy (default 30 days)

### Storage Efficiency
- Compress large measurement arrays using msgpack
- Store only essential data in primary certificates
- Optional detail storage for debugging

### Query Patterns
- Index by function name for trend analysis
- Index by build_id for CI integration
- Time-based indexing for cleanup operations

## Security and Privacy

### Sensitive Data Handling
- No source code stored in measurements
- Function names only (no implementation details)
- Sanitized error messages in alerts
- Access control on certificate directories

### Data Retention
- Configurable retention periods per function
- Automatic cleanup of expired measurements
- Preserve baseline certificates indefinitely
- Audit trail for configuration changes

**Data Model Status**: ✅ COMPLETE
**Schema Validation**: ✅ DEFINED
**Storage Strategy**: ✅ SPECIFIED
**Ready for Contracts Phase**: ✅ YES
