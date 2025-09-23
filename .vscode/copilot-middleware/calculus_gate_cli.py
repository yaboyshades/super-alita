#!/usr/bin/env python3
"""
Calculus Gate CLI
================

Command-line interface for the calculus-based runtime derivative gate.
Addresses spec requirement FR-006 for developer tooling.

SPECIFICATION COMPLIANCE:
- FR-006: CLI command for local developer usage
- Performance: < 60 seconds for default sampling profile
- Output: JSON artifact + human-readable summary
- Integration: Compatible with CI and MCP workflows

Usage:
    python calculus_gate_cli.py <file> --function <name> [options]

Example:
    python calculus_gate_cli.py src/core/search.py --function search_documents
"""

import argparse
import json
import sys
from pathlib import Path

# Import calculus gate implementation
from calculus_gate import analyze_function_runtime, PerformanceCertificate


def main() -> int:
    """Main CLI entry point matching spec requirements."""
    parser = argparse.ArgumentParser(
        description="Calculus-based runtime derivative analysis gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myfile.py --function process_data
  %(prog)s src/core.py --function search --output results.json
  %(prog)s lib/utils.py --function sort_items --min-size 10 --max-size 1000
        """,
    )

    # Required arguments per spec
    parser.add_argument("file_path", help="Python file containing target function")
    parser.add_argument("--function", required=True, help="Function name to analyze")

    # Configuration options per spec FR-001
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Minimum input size for sampling (default: 1)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10000,
        help="Maximum input size for sampling (default: 10000)",
    )
    parser.add_argument(
        "--samples", type=int, default=20, help="Number of sample points (default: 20)"
    )

    # Threshold configuration per spec FR-004
    parser.add_argument(
        "--slope-limit",
        type=float,
        default=2.0,
        help="Maximum allowed |df/dn| (default: 2.0)",
    )
    parser.add_argument(
        "--curvature-limit",
        type=float,
        default=1.0,
        help="Maximum allowed |d²f/dn²| (default: 1.0)",
    )
    parser.add_argument(
        "--lipschitz-limit",
        type=float,
        default=10.0,
        help="Maximum Lipschitz constant (default: 10.0)",
    )

    # Output options per spec FR-005
    parser.add_argument("--output", "-o", help="Save JSON certificate to file")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed analysis"
    )

    # CI integration per spec FR-006
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit with error code on threshold violations",
    )
    parser.add_argument(
        "--commit-hash",
        default="HEAD",
        help="Git commit hash for tracking (default: HEAD)",
    )

    args = parser.parse_args()

    # Validate inputs
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}", file=sys.stderr)
        return 1

    try:
        # Run calculus analysis per spec requirements
        if not args.quiet:
            print(f"🔬 Analyzing function '{args.function}' in {file_path}")
            print("📊 Sampling runtime across input sizes...")

        # Configure custom limits if provided
        from calculus_gate import CalculusAnalyzer, RuntimeProfiler

        profiler = RuntimeProfiler(
            min_size=args.min_size, max_size=args.max_size, num_samples=args.samples
        )

        analyzer = CalculusAnalyzer(
            slope_limit=args.slope_limit,
            curvature_limit=args.curvature_limit,
            lipschitz_limit=args.lipschitz_limit,
        )

        # Analyze function runtime per spec FR-002, FR-003
        certificate = analyze_function_runtime(
            str(file_path), args.function, args.commit_hash
        )

        # Generate output per spec FR-005
        if not args.quiet:
            print("\n🏛️  CALCULUS GATE ANALYSIS RESULTS")
            print("=" * 50)
            print(f"📊 Function: {certificate.function_name}")
            print(
                f"📈 Slope Gate: {'✅ PASS' if certificate.passes_slope_gate else '❌ FAIL'}"
            )
            print(
                f"📐 Curvature Gate: {'✅ PASS' if certificate.passes_curvature_gate else '❌ FAIL'}"
            )
            print(
                f"📏 Lipschitz Gate: {'✅ PASS' if certificate.passes_lipschitz_gate else '❌ FAIL'}"
            )
            print(f"🎯 Overall Grade: {certificate.certificate_grade}")
            print("-" * 50)

            if certificate.overall_pass:
                print("🎉 PERFORMANCE APPROVED: Mathematical bounds satisfied!")
            else:
                print("⚠️  PERFORMANCE VIOLATIONS: Review derivative analysis.")

            # Show violations if verbose
            if args.verbose and not certificate.overall_pass:
                analysis = certificate.analysis
                if analysis.slope_violations:
                    print(f"\n📈 Slope Violations:")
                    for size, slope in analysis.slope_violations:
                        print(
                            f"   Size {size}: |df/dn| = {slope:.6f} > {args.slope_limit}"
                        )

                if analysis.curvature_changes:
                    print(f"\n📐 Curvature Changes:")
                    for size, curve in analysis.curvature_changes:
                        print(
                            f"   Size {size}: |d²f/dn²| = {curve:.6f} > {args.curvature_limit}"
                        )

                if analysis.lipschitz_constant > args.lipschitz_limit:
                    print(f"\n📏 Lipschitz Violation:")
                    print(
                        f"   Constant = {analysis.lipschitz_constant:.6f} > {args.lipschitz_limit}"
                    )

        # Save JSON artifact per spec FR-005
        if args.output:
            certificate_data = {
                "function_name": certificate.function_name,
                "timestamp": certificate.timestamp,
                "commit_hash": certificate.commit_hash,
                "slope_limit": certificate.slope_limit,
                "curvature_limit": certificate.curvature_limit,
                "lipschitz_limit": certificate.lipschitz_limit,
                "passes_slope_gate": certificate.passes_slope_gate,
                "passes_curvature_gate": certificate.passes_curvature_gate,
                "passes_lipschitz_gate": certificate.passes_lipschitz_gate,
                "overall_pass": certificate.overall_pass,
                "certificate_grade": certificate.certificate_grade,
                "analysis": {
                    "input_sizes": certificate.analysis.input_sizes,
                    "runtime_values": certificate.analysis.runtime_values,
                    "first_derivative": certificate.analysis.first_derivative,
                    "second_derivative": certificate.analysis.second_derivative,
                    "lipschitz_constant": certificate.analysis.lipschitz_constant,
                    "slope_violations": certificate.analysis.slope_violations,
                    "curvature_changes": certificate.analysis.curvature_changes,
                },
            }

            with open(args.output, "w") as f:
                json.dump(certificate_data, f, indent=2, default=str)

            if not args.quiet:
                print(f"\n📜 Certificate saved to {args.output}")

        # Return appropriate exit code per spec FR-006
        if args.fail_on_violation and not certificate.overall_pass:
            return 1
        else:
            return 0

    except Exception as e:
        print(f"❌ Error during analysis: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
