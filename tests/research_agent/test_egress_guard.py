"""Unit tests for the egress guard."""

from research_agent.egress_guard import BLOCKED_NETWORKS, is_allowed


def test_non_http_blocked() -> None:
    allowed, reason = is_allowed("ftp://example.com")
    assert not allowed
    assert reason == "non-http(s) scheme"


def test_local_ip_blocked() -> None:
    network = BLOCKED_NETWORKS[0]
    ip = next(iter(network.hosts()))
    allowed, reason = is_allowed(f"http://{ip}")
    assert not allowed
    assert "blocked network" in reason


def test_dns_host_allowed() -> None:
    allowed, reason = is_allowed("https://example.com")
    assert allowed
    assert reason == "ok"
