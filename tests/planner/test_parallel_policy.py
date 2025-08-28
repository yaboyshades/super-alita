from dataclasses import dataclass

from cortex.planner.parallel_wrapper import should_parallelize


@dataclass
class Step:
    estimated_time: float
    tool: str | None = None


def test_parallelization_when_conditions_met():
    substeps = [
        Step(estimated_time=5, tool="a"),
        Step(estimated_time=4, tool="b"),
        Step(estimated_time=6, tool="c"),
    ]
    assert should_parallelize(substeps, parallel_threshold=2, min_parallel_benefit=3)


def test_no_parallelization_with_shared_dependencies():
    substeps = [
        Step(estimated_time=5, tool="a"),
        Step(estimated_time=4, tool="a"),
        Step(estimated_time=6, tool="b"),
    ]
    assert not should_parallelize(
        substeps, parallel_threshold=2, min_parallel_benefit=3
    )


def test_no_parallelization_when_threshold_not_met():
    substeps = [
        Step(estimated_time=5, tool="a"),
        Step(estimated_time=4, tool="b"),
    ]
    assert not should_parallelize(
        substeps, parallel_threshold=2, min_parallel_benefit=3
    )


def test_no_parallelization_when_benefit_insufficient():
    substeps = [
        Step(estimated_time=1, tool="a"),
        Step(estimated_time=1, tool="b"),
        Step(estimated_time=1, tool="c"),
    ]
    assert not should_parallelize(
        substeps, parallel_threshold=2, min_parallel_benefit=5
    )
