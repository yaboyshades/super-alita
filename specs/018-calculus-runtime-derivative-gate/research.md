# Research: Calculus Runtime Derivative Gate

**Phase 0 Research** | **Date**: 2025-09-16 | **Plan**: specs/018-calculus-runtime-derivative-gate/plan.md

## Research Objectives

Based on the spec requirements, we need to research the following unknowns:

1. **Fallback strategies for noisy data** (Savitzky–Golay filtering, bootstrap CI)
2. **Best practices for curve fitting stability** (spline vs polynomial, regularization)
3. **Statistical confidence methods for derivatives** (confidence intervals, bootstrap)
4. **Visualization options** (text-based vs optional plots)

## Decision/Rationale/Alternatives Analysis

### 1. Noise Mitigation Strategies

**Decision**: Multi-tier approach with Cubic Spline + Savitzky-Golay fallback

**Research Findings**:
- **Cubic Spline**: Excellent for smooth, well-behaved runtime curves with sufficient unique points
- **Savitzky-Golay Filter**: Robust for noisy data, preserves shape better than moving averages
- **Bootstrap Confidence Intervals**: Statistical rigor for derivative uncertainty

**Rationale**:
- Primary: CubicSpline provides smooth analytical derivatives
- Fallback: Savitzky-Golay when data is too noisy for stable spline fitting
- Statistical validation: Bootstrap sampling for confidence bounds

**Alternatives Considered**:
- Polynomial fitting: Less stable with high-degree polynomials, prone to oscillation
- LOWESS/LOESS: More complex, slower, overkill for runtime curves
- Simple moving average: Too aggressive smoothing, loses important features

### 2. Curve Fitting Stability

**Decision**: Cubic Spline with input validation + regularization fallbacks

**Research Findings**:
- **Strictly Increasing Requirement**: CubicSpline requires x-values to be strictly increasing
- **Minimum Points**: Need ≥3 unique points for meaningful derivatives
- **Numerical Stability**: Handle edge cases with boundary condition specification

**Implementation Strategy**:
```python
# Primary approach
fitted_curve = CubicSpline(sizes, times, bc_type='natural')

# Validation checks
if len(unique_sizes) < 3:
    raise ValueError("Need ≥3 unique input sizes")

# Fallback for ill-conditioned data
if fitting_fails:
    # Use linear interpolation + finite differences
    fitted_curve = interp1d(sizes, times, kind='linear')
```

**Alternatives Considered**:
- Polynomial regression: Less stable, requires degree selection
- Exponential fitting: Too restrictive, assumes specific growth model
- Piecewise linear: Loses smoothness, poor derivative estimation

### 3. Statistical Confidence Methods

**Decision**: Bootstrap confidence intervals for derivative estimates

**Research Findings**:
- **Bootstrap Sampling**: Resample runtime measurements to estimate derivative uncertainty
- **Finite Difference Accuracy**: Central difference more accurate than forward/backward
- **Confidence Bounds**: 95% confidence intervals for derivative significance testing

**Mathematical Foundation**:
```python
# Central difference for first derivative
df_dx = (f(x + h) - f(x - h)) / (2 * h)

# Second difference for curvature
d2f_dx2 = (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# Bootstrap confidence interval
bootstrap_derivatives = []
for _ in range(1000):
    resampled_data = bootstrap_resample(measurements)
    bootstrap_derivatives.append(compute_derivative(resampled_data))

confidence_interval = np.percentile(bootstrap_derivatives, [2.5, 97.5])
```

**Alternatives Considered**:
- Analytical error propagation: Complex for spline derivatives
- Monte Carlo simulation: More expensive than bootstrap
- Simple standard deviation: Doesn't capture derivative uncertainty properly

### 4. Visualization and Reporting

**Decision**: Rich text-based reporting with optional plot export

**Research Findings**:
- **Primary**: Rich console output with Unicode charts for CLI
- **Optional**: Matplotlib plots saved to artifacts directory
- **CI/MCP**: JSON schema with embedded base64 plots if needed

**Implementation Approach**:
```python
# Console visualization using Rich
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Derivative Analysis")
table.add_column("Input Size", style="cyan")
table.add_column("Runtime (s)", style="green")
table.add_column("df/dn", style="yellow")
table.add_column("d²f/dn²", style="red")

# Optional plot generation
if generate_plots:
    import matplotlib.pyplot as plt
    plt.plot(sizes, times, 'o-', label='Runtime')
    plt.plot(sizes, derivatives, '--', label='df/dn')
    plt.savefig(f'artifacts/calculus_gate/{function_name}_curve.png')
```

**Alternatives Considered**:
- Web dashboard: Out of scope for initial implementation
- ASCII art plots: Lower quality than Rich Unicode charts
- Only JSON output: Less developer-friendly for debugging

## Implementation Recommendations

### Primary Technology Stack
- **Numerical Libraries**: SciPy (CubicSpline), NumPy (arrays, statistics)
- **Visualization**: Rich (console), optional Matplotlib (plots)
- **Statistical Analysis**: Bootstrap sampling with NumPy
- **Data Persistence**: JSON certificates with optional plot artifacts

### Error Handling Strategy
1. **Insufficient Data**: Require ≥3 unique input sizes, clear error messages
2. **Noisy Measurements**: Savitzky-Golay filtering fallback
3. **Fitting Failures**: Linear interpolation with finite differences
4. **Edge Cases**: Graceful degradation with warning logs

### Performance Considerations
- **Sampling Overhead**: Limit to configurable timeouts (5min CI, 60s local)
- **Memory Usage**: Stream processing for large datasets
- **Caching**: Persist intermediate results for retry scenarios

### Security and Reliability
- **Input Validation**: Sanitize file paths, function names
- **Resource Limits**: Memory caps, execution timeouts
- **Error Reporting**: Sanitized error messages for CI logs

## Research Validation

All research objectives have been addressed with concrete technical decisions:

✅ **Noise mitigation**: Cubic Spline + Savitzky-Golay fallback strategy
✅ **Fitting stability**: Input validation + regularization with clear fallbacks
✅ **Statistical confidence**: Bootstrap confidence intervals for derivatives
✅ **Visualization**: Rich console + optional plots for comprehensive reporting

## Next Phase Requirements

Phase 1 can proceed with:
- Data model design based on research findings
- Contract schemas incorporating bootstrap confidence intervals
- Technical implementation patterns validated through research
- Clear fallback strategies for all identified edge cases

**Research Status**: ✅ COMPLETE
**Blocking Issues**: None identified
**Ready for Phase 1**: ✅ YES
