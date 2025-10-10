from datetime import datetime

import pytest

from src.constitutional.engine import ConstitutionalComplianceEngine
from src.constitutional.events import ComplianceEvent


@pytest.fixture
def engine() -> ConstitutionalComplianceEngine:
    return ConstitutionalComplianceEngine.from_config_path(
        "configs/constitutional/framework.yaml"
    )


def test_engine_fails_when_score_below_threshold(
    engine: ConstitutionalComplianceEngine,
) -> None:
    report = engine.evaluate(
        events=[
            ComplianceEvent(
                article_id="article_1",
                check_id="retry_policy_defined",
                status="failed",
                timestamp=datetime.utcnow(),
                details={"component": "eventbus"},
            ),
            ComplianceEvent(
                article_id="article_2",
                check_id="telemetry_emitted",
                status="passed",
                timestamp=datetime.utcnow(),
                details={"component": "eventbus"},
            ),
        ]
    )

    assert report.score < 0.75
    assert not report.is_compliant


def test_engine_requires_all_articles_to_pass(
    engine: ConstitutionalComplianceEngine,
) -> None:
    report = engine.evaluate(
        events=[
            ComplianceEvent(
                article_id="article_1",
                check_id="retry_policy_defined",
                status="passed",
                timestamp=datetime.utcnow(),
                details={"component": "eventbus"},
            ),
            ComplianceEvent(
                article_id="article_2",
                check_id="telemetry_emitted",
                status="passed",
                timestamp=datetime.utcnow(),
                details={"component": "eventbus"},
            ),
            ComplianceEvent(
                article_id="article_3",
                check_id="egress_guard_enabled",
                status="passed",
                timestamp=datetime.utcnow(),
                details={"component": "eventbus"},
            ),
        ]
    )

    assert report.score >= 0.75
    assert report.is_compliant
