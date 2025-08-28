#!/usr/bin/env python3
"""Test autogen pipeline offline with mock API."""

from src.core.event_bus import EventBus
from src.pipelines.autogen_pipeline import autogen_any


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

    def pytest_run(self, _args=None):
        print("✅ Mock pytest run - all tests pass")
        return {"exit_code": 0, "output": "All tests passed"}

    def secure_scan(self, _code=""):
        print("✅ Mock security scan - no issues")
        return {"issues": [], "status": "clean"}


def main():
    """Test the autogen pipeline."""
    print("🧪 Testing Autogen Pipeline with Offline Mock")
    print("=" * 50)

    result = autogen_any(
        description="scrape product prices from e-commerce websites and save to CSV",
        repo_path=".",
        iterations=1,
        event_bus=EventBus(),
        api=OfflineMockAPI(),
    )

    print("\n📊 Final Result:")
    print(f'Status: {result.get("status")}')
    print(f'Applied: {len(result.get("applied", []))} capabilities')
    if result.get("applied"):
        for cap in result["applied"]:
            print(f'  - {cap["kind"]}: {len(cap["paths"])} files')

    if result.get("failed"):
        print(f'Failed: {result["failed"]}')

    return result


if __name__ == "__main__":
    main()
