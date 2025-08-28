#!/usr/bin/env python3
"""Test autogen pipeline with fully mocked components."""

import time
import uuid

from src.core.event_bus import EventBus
from src.pipelines.autogen_pipeline import CAPABILITY_TEMPLATES, _emit
from src.policies.need_detector import NeedDetector


class MockGate:
    """Mock gate that always passes validation."""

    def validate_latest(self, latest):
        print("✅ Mock gate validation - all checks pass")
        return True, {
            "paths": latest.get("paths", []),
            "reasons": [],
            "checks_passed": ["safety", "tests", "docs"],
        }


class OfflineMockAPI:
    """Fully offline mock API for testing autogen pipeline."""

    def deepcode_request(self, **kwargs):
        print(f'✅ Mock DeepCode request: {kwargs["task_kind"]}')
        return {"status": "queued"}

    def deepcode_latest(self):
        print("✅ Mock DeepCode latest - simulating successful generation")
        return {
            "paths": [
                "src/abilities/web_scraper.py",
                "tests/abilities/test_web_scraper.py",
                "docs/web_scraper.md",
            ],
            "status": "complete",
            "files": [
                {
                    "path": "src/abilities/web_scraper.py",
                    "content": "def scrape(): pass",
                },
                {
                    "path": "tests/abilities/test_web_scraper.py",
                    "content": "def test_scrape(): pass",
                },
            ],
        }

    def deepcode_apply(self, paths=None):
        print(f"✅ Mock DeepCode apply: {len(paths or [])} files")
        return {"applied": paths or [], "status": "success"}


def test_autogen_simplified(description: str) -> dict:
    """Simplified autogen test with mocked components."""
    bus = EventBus()
    api = OfflineMockAPI()
    convo_id = f"autogen-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    # Detect capability needs
    kinds = NeedDetector().detect(description)
    print(f"🔍 Detected capabilities: {kinds}")

    if not kinds:
        return {"status": "skipped", "reason": "no_signals"}

    results = {"status": "complete", "applied": []}

    for kind in kinds:
        tpl = CAPABILITY_TEMPLATES.get(kind)
        if not tpl:
            print(f"❌ Unknown capability: {kind}")
            continue

        print(f"🎯 Processing capability: {kind}")

        # Emit planning event
        _emit(
            bus,
            "oak.plan_proposed",
            {
                "kind": kind,
                "desc": description,
                "conversation_id": convo_id,
                "option": "autogen_create_ability",
            },
        )

        # Start autogen
        _emit(
            bus,
            "autogen.started",
            {"kind": kind, "desc": description, "conversation_id": convo_id},
        )

        # Make DeepCode request
        req = tpl["requirements"](description)
        api.deepcode_request(
            task_kind=tpl["task_kind"],
            requirements=req,
            repo_path=".",
            conversation_id=convo_id,
        )

        # Get latest and validate with mock gate
        latest = api.deepcode_latest()
        gate = MockGate()
        ok, info = gate.validate_latest(latest)

        print(f'🔒 Gate validation: {"✅ PASS" if ok else "❌ FAIL"}')

        if ok:
            paths = info.get("paths") or []
            apply_res = api.deepcode_apply(paths=paths)

            _emit(
                bus,
                "autogen.applied",
                {
                    "kind": kind,
                    "iteration": 1,
                    "paths": paths,
                    "apply_result": apply_res,
                },
            )

            # Success reward
            _emit(
                bus,
                "bandit.reward_event",
                {
                    "source": "autogen",
                    "kind": kind,
                    "reward": 1.0,
                    "desc": description,
                },
            )

            # OaK feedback
            _emit(
                bus,
                "oak.outcome_feedback",
                {"kind": kind, "success": True, "paths": paths},
            )

            results["applied"].append(
                {"kind": kind, "paths": paths, "apply": apply_res}
            )

            print(f"✅ Successfully applied {kind} capability")
        else:
            print(f"❌ Failed to validate {kind} capability")
            results.setdefault("failed", []).append(kind)

    return results


def main():
    """Test the simplified autogen pipeline."""
    print("🧪 Testing Simplified Autogen Pipeline")
    print("=" * 50)

    test_cases = [
        "scrape product prices from e-commerce websites and save to CSV",
        "build an ETL pipeline to transform CSV data",
        "create an API client for REST services",
        "generate reports from data",
    ]

    for i, description in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {description[:50]}...")
        print("-" * 50)

        result = test_autogen_simplified(description)

        print("\n📊 Result:")
        print(f'  Status: {result.get("status")}')
        print(f'  Applied: {len(result.get("applied", []))} capabilities')
        if result.get("applied"):
            for cap in result["applied"]:
                print(f'    - {cap["kind"]}: {len(cap["paths"])} files')
        if result.get("failed"):
            print(f'  Failed: {result["failed"]}')

        print("\n" + "=" * 50)

    print("\n🎉 All test cases completed!")


if __name__ == "__main__":
    main()
