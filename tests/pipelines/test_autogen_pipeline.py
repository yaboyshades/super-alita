from __future__ import annotations

from src.core.event_bus import EventBus


class StubAPI:
    def __init__(self):
        self.step = 0

    def deepcode_request(self, **kw):
        return {"status": "accepted", "request": kw}

    def deepcode_latest(self):
        self.step += 1
        # First pass: missing required files -> fail gate
        if self.step == 1:
            return {
                "diffs": [
                    {
                        "path": "src/misc/foo.py",
                        "new_content": "def x(): return 1",
                    }
                ]
            }
        # Second pass: provide required artifacts
        return {
            "diffs": [
                {"path": "docs/scraper_overview.md", "new_content": "# doc"},
                {
                    "path": "src/abilities/web_scraper_dynamic.py",
                    "new_content": "def ok(): return 1",
                },
                {
                    "path": "tests/abilities/test_web_scraper_dynamic.py",
                    "new_content": "def test_ok(): assert True",
                },
            ]
        }

    def deepcode_apply(self, paths=None):
        return {"status": "ok", "applied": True, "paths": paths}

    def pytest_run(self, args=None):
        _ = args  # Acknowledged unused
        return {"ok": True, "stdout": "1 passed"}

    def secure_scan_code(self, code: str):
        _ = code  # Acknowledged unused
        return {
            "tool": "secure_scan_code",
            "result": {"issues": [], "issue_count": 0},
        }


class SinkBus(EventBus):
    def __init__(self):
        super().__init__()
        self.events = []

    async def publish(self, evt):
        self.events.append(evt)


def test_generic_autogen_happy_path():
    bus = SinkBus()
    # monkeypatch LocalAPI in pipeline to use stub
    import src.pipelines.autogen_pipeline as P

    P.LocalAPI = lambda: StubAPI()
    res = P.autogen_any(
        description="We need to scrape product prices from ecommerce pages.",
        iterations=3,
        event_bus=bus,
    )
    assert res["applied"], "should apply at least one capability"
    topics = [e.event_type for e in bus.events]
    assert "autogen.started" in topics
    assert "autogen.iteration_checked" in topics
    assert "autogen.applied" in topics


def test_skips_when_no_need():
    bus = SinkBus()
    import src.pipelines.autogen_pipeline as P

    P.LocalAPI = lambda: StubAPI()
    res = P.autogen_any(
        description="Refactor internal string util.",
        iterations=2,
        event_bus=bus,
    )
    assert res["status"] == "skipped"
