"""Simple skillset server exposing a signed ``search_issues_skill``.

The server uses FastAPI and verifies HMAC signatures on incoming
requests.  This mirrors a typical Skillset setup where the caller
authenticates using a shared secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request

_SECRET = os.environ.get("SKILLSET_SECRET", "secret")
app = FastAPI()


def verify_signature(body: bytes, signature: str, secret: str = _SECRET) -> bool:
    """Verify an HMAC signature for ``body`` using ``secret``."""
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


@app.post("/search/issues")
async def search_issues_skill(request: Request) -> dict[str, list[dict[str, Any]]]:
    """Search a static set of issues.

    The request body must contain ``{"query": "..."}`` and include an
    ``X-Signature`` header with an HMAC SHA256 digest of the raw body.
    """
    body = await request.body()
    signature = request.headers.get("x-signature", "")
    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="invalid signature")

    payload = json.loads(body.decode())
    query = payload.get("query", "").lower()
    issues = [
        {"id": 1, "title": "Login bug"},
        {"id": 2, "title": "Dark mode feature"},
        {"id": 3, "title": "Docs update"},
    ]
    matches = [issue for issue in issues if query in issue["title"].lower()]
    return {"issues": matches}


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
