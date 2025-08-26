import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from backend.skillset_server import app, verify_signature


def _sign(body: bytes, secret: str = "secret") -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_signature() -> None:
    body = b"{}"
    sig = _sign(body)
    assert verify_signature(body, sig)
    assert not verify_signature(body, "bad")


def test_search_issues_skill() -> None:
    client = TestClient(app)
    payload = {"query": "dark"}
    body = json.dumps(payload).encode()
    headers = {"x-signature": _sign(body)}
    resp = client.post("/search/issues", data=body, headers=headers)
    assert resp.status_code == 200
    issues = resp.json()["issues"]
    assert issues == [{"id": 2, "title": "Dark mode feature"}]


def test_search_issues_skill_bad_signature() -> None:
    client = TestClient(app)
    payload = {"query": "dark"}
    resp = client.post("/search/issues", json=payload, headers={"x-signature": "bad"})
    assert resp.status_code == 401
