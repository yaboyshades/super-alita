import json
from pathlib import Path

from src.core.optimization.strategy_selector import StrategySelector


def test_strategy_selector_select_and_feedback(tmp_path: Path) -> None:
    # Copy baseline config
    cfg = tmp_path / "strategies.json"
    cfg.write_text(
        json.dumps(
            {
                "version": "0.1.0",
                "task_types": {
                    "code_review": {
                        "algorithm": "thompson",
                        "arms": [
                            {"id": "a", "name": "A", "metadata": {"style": "concise"}},
                            {"id": "b", "name": "B", "metadata": {"style": "steps"}},
                        ],
                        "stats": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    ss = StrategySelector(config_path=str(cfg))
    d = ss.select("code_review")
    assert d.task_type == "code_review"
    assert d.arm_id in {"a", "b"}

    ok = ss.feedback("code_review", decision_id=d.decision_id, reward=0.9)
    assert ok
    updated = json.loads(cfg.read_text(encoding="utf-8"))
    assert "stats" in updated["task_types"]["code_review"]
