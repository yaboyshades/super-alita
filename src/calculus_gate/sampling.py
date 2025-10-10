"""
Runtime profiling and sampling for calculus gate analysis.

This module provides facilities to profile function performance across
different input sizes, collecting timing and memory usage data for
mathematical analysis.
"""

import gc
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

from .models import RuntimeSampleSet


class RuntimeProfiler:
    """Profiles function runtime performance across different input sizes."""

    def __init__(self, warmup_runs: int = 3, measurement_runs: int = 5):
        """
        Initialize the runtime profiler.

        Args:
            warmup_runs: Number of warmup iterations before measurement
            measurement_runs: Number of measurement runs to average
        """
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.enable_memory_tracking = True

    def profile_function(
        self,
        target_function: Callable[[int], Any],
        input_sizes: list[int],
        build_id: str = "unknown",
    ) -> RuntimeSampleSet:
        """
        Profile a function across multiple input sizes.

        Args:
            target_function: Function to profile (must accept single int argument)
            input_sizes: List of input sizes to test (must be strictly increasing)
            build_id: Build or commit identifier

        Returns:
            RuntimeSampleSet with measurement data
        """
        if not input_sizes:
            raise ValueError("input_sizes cannot be empty")

        if not all(
            input_sizes[i] < input_sizes[i + 1]
            for i in range(len(input_sizes) - 1)
        ):
            raise ValueError("input_sizes must be strictly increasing")

        sample_set = RuntimeSampleSet(
            target_function=(
                target_function.__name__
                if hasattr(target_function, "__name__")
                else "unknown"
            ),
            build_id=build_id,
            warmup_runs=self.warmup_runs,
        )

        total_outliers = 0
        measurement_noise_values = []

        for input_size in input_sizes:
            try:
                (
                    wall_time,
                    cpu_time,
                    memory_peak,
                    memory_delta,
                    outliers,
                    noise,
                ) = self._measure_single_input(target_function, input_size)

                sample_set.add_sample(
                    input_size, wall_time, cpu_time, memory_peak, memory_delta
                )
                total_outliers += outliers
                measurement_noise_values.append(noise)

            except Exception as e:
                # Record measurement failure but continue with other sizes
                print(
                    f"Warning: Failed to measure input size {input_size}: {e}"
                )
                # Add minimal data to maintain array consistency
                sample_set.add_sample(input_size, 0.0, 0.0, 0, 0)
                measurement_noise_values.append(
                    1.0
                )  # High noise for failed measurement

        # Update quality indicators
        sample_set.outliers_removed = total_outliers
        sample_set.measurement_noise = (
            sum(measurement_noise_values) / len(measurement_noise_values)
            if measurement_noise_values
            else 0.0
        )
        sample_set.convergence_achieved = all(
            noise < 0.2 for noise in measurement_noise_values
        )  # 20% CV threshold

        return sample_set

    def _measure_single_input(
        self, target_function: Callable[[int], Any], input_size: int
    ) -> tuple[float, float, int, int, int, float]:
        """
        Measure function performance for a single input size.

        Returns:
            (wall_time, cpu_time, memory_peak, memory_delta, outliers_removed, measurement_noise)
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                target_function(input_size)
            except Exception:
                pass  # Ignore warmup failures

        # Force garbage collection before measurement
        gc.collect()

        # Measurement runs
        wall_times = []
        cpu_times = []
        memory_peaks = []
        memory_deltas = []

        for _ in range(self.measurement_runs):
            # Start memory tracking
            if self.enable_memory_tracking:
                tracemalloc.start()

            # Measure timing
            start_wall = time.perf_counter()
            start_cpu = time.process_time()

            try:
                result = target_function(input_size)
                # Keep reference to prevent optimization
                _ = result
            except Exception as e:
                # Function failed - record minimal timing
                end_wall = time.perf_counter()
                end_cpu = time.process_time()
                wall_times.append(end_wall - start_wall)
                cpu_times.append(end_cpu - start_cpu)
                memory_peaks.append(0)
                memory_deltas.append(0)
                print(f"Warning: Function failed for input {input_size}: {e}")
                continue

            end_wall = time.perf_counter()
            end_cpu = time.process_time()

            wall_time = end_wall - start_wall
            cpu_time = end_cpu - start_cpu

            # Memory measurement
            memory_peak = 0
            memory_delta = 0
            if self.enable_memory_tracking:
                try:
                    current_peak, peak_bytes = tracemalloc.get_traced_memory()
                    memory_peak = peak_bytes
                    memory_delta = current_peak
                    tracemalloc.stop()
                except Exception:
                    # Memory tracking failed
                    tracemalloc.stop()

            wall_times.append(wall_time)
            cpu_times.append(cpu_time)
            memory_peaks.append(memory_peak)
            memory_deltas.append(memory_delta)

        # Remove outliers (values > 2 std devs from mean)
        wall_times_clean, outliers_wall = self._remove_outliers(wall_times)
        cpu_times_clean, outliers_cpu = self._remove_outliers(cpu_times)

        total_outliers = outliers_wall + outliers_cpu

        # Calculate averages
        avg_wall_time = (
            sum(wall_times_clean) / len(wall_times_clean)
            if wall_times_clean
            else 0.0
        )
        avg_cpu_time = (
            sum(cpu_times_clean) / len(cpu_times_clean)
            if cpu_times_clean
            else 0.0
        )
        avg_memory_peak = max(memory_peaks) if memory_peaks else 0
        avg_memory_delta = (
            sum(memory_deltas) // len(memory_deltas) if memory_deltas else 0
        )

        # Calculate measurement noise (coefficient of variation)
        if len(wall_times_clean) > 1 and avg_wall_time > 0:
            variance = sum(
                (t - avg_wall_time) ** 2 for t in wall_times_clean
            ) / len(wall_times_clean)
            std_dev = variance**0.5
            noise = std_dev / avg_wall_time
        else:
            noise = 0.0

        return (
            avg_wall_time,
            avg_cpu_time,
            avg_memory_peak,
            avg_memory_delta,
            total_outliers,
            noise,
        )

    def _remove_outliers(self, values: list[float]) -> tuple[list[float], int]:
        """
        Remove outliers using 2-sigma rule.

        Returns:
            (cleaned_values, number_of_outliers_removed)
        """
        if len(values) <= 2:
            return values, 0

        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance**0.5

        if std_dev == 0:
            return values, 0

        # Keep values within 2 standard deviations
        threshold = 2.0 * std_dev
        cleaned = [v for v in values if abs(v - mean_val) <= threshold]

        # Ensure we keep at least half the values
        if len(cleaned) < len(values) // 2:
            cleaned = values
            outliers_removed = 0
        else:
            outliers_removed = len(values) - len(cleaned)

        return cleaned, outliers_removed

    def generate_exponential_input_sizes(
        self, min_size: int = 1, max_size: int = 10000, sample_count: int = 20
    ) -> list[int]:
        """
        Generate exponentially spaced input sizes for testing.

        Args:
            min_size: Minimum input size
            max_size: Maximum input size
            sample_count: Number of sample points

        Returns:
            List of exponentially spaced input sizes
        """
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("Invalid size range")
        if sample_count < 3:
            raise ValueError("Need at least 3 sample points")

        import math

        # Generate exponential spacing
        log_min = math.log(min_size)
        log_max = math.log(max_size)
        log_step = (log_max - log_min) / (sample_count - 1)

        sizes = []
        for i in range(sample_count):
            log_size = log_min + i * log_step
            size = int(math.exp(log_size))
            sizes.append(size)

        # Ensure uniqueness and strict ordering
        unique_sizes = []
        for size in sizes:
            if not unique_sizes or size > unique_sizes[-1]:
                unique_sizes.append(size)

        # Ensure we have the exact endpoints
        if unique_sizes[0] != min_size:
            unique_sizes[0] = min_size
        if unique_sizes[-1] != max_size:
            unique_sizes[-1] = max_size

        return unique_sizes

    def validate_function_signature(self, target_function: Callable) -> bool:
        """
        Validate that function accepts a single integer argument.

        Args:
            target_function: Function to validate

        Returns:
            True if function signature is valid
        """
        try:
            # Try calling with a small test value
            target_function(1)
            return True
        except TypeError as e:
            if "takes" in str(e) and "positional argument" in str(e):
                return False
            # Other TypeError might be due to implementation, not signature
            return True
        except Exception:
            # Other exceptions are okay - just means function works with int input
            return True

    def estimate_runtime_complexity(self, sample_set: RuntimeSampleSet) -> str:
        """
        Provide a rough estimate of runtime complexity based on measurements.

        Args:
            sample_set: Measured runtime data

        Returns:
            String description of estimated complexity
        """
        if len(sample_set.input_sizes) < 3:
            return "insufficient_data"

        # Calculate ratios between consecutive measurements
        ratios = []
        for i in range(1, len(sample_set.wall_times)):
            if sample_set.wall_times[i - 1] > 0:
                time_ratio = (
                    sample_set.wall_times[i] / sample_set.wall_times[i - 1]
                )
                size_ratio = (
                    sample_set.input_sizes[i] / sample_set.input_sizes[i - 1]
                )
                if size_ratio > 1:
                    ratios.append(time_ratio / size_ratio)

        if not ratios:
            return "unknown"

        avg_ratio = sum(ratios) / len(ratios)

        # Classify based on average ratio
        if avg_ratio < 1.2:
            return "constant_or_logarithmic"
        elif avg_ratio < 2.0:
            return "linear"
        elif avg_ratio < 4.0:
            return "quadratic_or_polynomial"
        else:
            return "exponential_or_worse"
