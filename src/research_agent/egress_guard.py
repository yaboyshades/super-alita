"""Egress guard enforcing CMA v5.3 Codecraft rules."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

# RFC1918 + link-local + loopback networks disallowed for outbound research calls.
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def is_allowed(url: str) -> tuple[bool, str]:
    """Return (allowed, reason) for the supplied URL."""
    try:
        parsed = urlparse(url)
    except Exception:  # pragma: no cover - defensive guard
        return False, "url parse error"

    if parsed.scheme not in {"http", "https"}:
        return False, "non-http(s) scheme"

    host = parsed.hostname or ""
    if not host:
        return False, "missing host"

    if host.endswith(".local") or host.endswith(".internal"):
        return False, "blocked host suffix"

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Host is not a plain IP; allow DNS resolution to enforce policies downstream.
        return True, "ok"

    allow_loopback = os.environ.get("RESEARCH_AGENT_ALLOW_LOOPBACK") == "1"
    for network in BLOCKED_NETWORKS:
        if ip in network:
            if allow_loopback and ip.is_loopback:
                continue
            return False, f"blocked network: {network}"
    return True, "ok"
