from src.core.optimization.reward_aggregator import compute_reward_from_result


def test_compute_reward_from_result_success_latency_cost() -> None:
    result = {
        "success": True,
        "execution_time": 1.0,
        "performance_metrics": {"cost_usd": 0.1},
    }
    r = compute_reward_from_result(result)
    assert 0.5 < r <= 1.0

    slow = {
        "success": True,
        "execution_time": 10.0,
        "performance_metrics": {"cost_usd": 0.1},
    }
    r2 = compute_reward_from_result(slow)
    assert r2 < r

    fail = {
        "success": False,
        "execution_time": 0.1,
        "performance_metrics": {"cost_usd": 0.0},
    }
    r3 = compute_reward_from_result(fail)
    assert 0.0 <= r3 < 0.5
