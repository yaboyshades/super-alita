import pytest

from src.security.maestro_hardening import MaestroSecurity
from src.reug_runtime import loop


def test_maestro_security_triggers_remediation_calls():
    calls: list[tuple[str, object]] = []

    def analyzer() -> list[str]:
        return ["exposed-surface"]

    def make_handler(name: str):
        def handler(vulnerabilities):
            calls.append((name, vulnerabilities))

        return handler

    security = MaestroSecurity(
        analyzer=analyzer,
        auth_handler=make_handler("auth"),
        authorization_handler=make_handler("authorization"),
        sandbox_handler=make_handler("sandbox"),
    )

    result = security.enforce()

    assert result == ["exposed-surface"]
    assert calls == [
        ("auth", ["exposed-surface"]),
        ("authorization", ["exposed-surface"]),
        ("sandbox", ["exposed-surface"]),
    ]


def test_maestro_security_skips_when_no_vulnerabilities():
    calls: list[str] = []

    def analyzer() -> list[str]:
        return []

    def recorder(_vulnerabilities):
        calls.append("called")

    security = MaestroSecurity(
        analyzer=analyzer,
        auth_handler=recorder,
        authorization_handler=recorder,
        sandbox_handler=recorder,
    )

    result = security.enforce()

    assert result == []
    assert calls == []


def test_ensure_maestro_hardening_respects_toggle(monkeypatch):
    calls: list[str] = []

    class FakeSecurity:
        def enforce(self) -> None:
            calls.append("enforced")

    monkeypatch.setattr(loop, "_maestro_security", FakeSecurity())
    monkeypatch.setattr(loop, "_maestro_hardening_applied", False)
    monkeypatch.setattr(loop.SETTINGS, "maestro_hardening_enabled", True)

    loop.ensure_maestro_hardening()
    loop.ensure_maestro_hardening()  # second invocation should be a no-op

    assert calls == ["enforced"]
    assert loop._maestro_hardening_applied is True


def test_ensure_maestro_hardening_disabled(monkeypatch):
    calls: list[str] = []

    class FakeSecurity:
        def enforce(self) -> None:
            calls.append("enforced")

    monkeypatch.setattr(loop, "_maestro_security", FakeSecurity())
    monkeypatch.setattr(loop, "_maestro_hardening_applied", False)
    monkeypatch.setattr(loop.SETTINGS, "maestro_hardening_enabled", False)

    loop.ensure_maestro_hardening()

    assert calls == []
    assert loop._maestro_hardening_applied is True
