from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_hex(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return f"sk-{secrets.token_urlsafe(32)}"


@dataclass
class APIKeyRecord:
    key_id: str
    key_hash: str
    owner: str
    created_at: str
    revoked_at: str | None = None
    expires_at: str | None = None
    metadata: dict[str, Any] | None = None

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("key_hash", None)
        return d


class APIKeyStore:
    """Simple JSON file-backed API key store (hash-at-rest)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._db: dict[str, APIKeyRecord] = {}
        self._loaded = False

    @classmethod
    def from_env(cls) -> APIKeyStore:
        path = os.getenv("ALITA_API_KEY_STORE", str(Path("config") / "api_keys.json"))
        return cls(path)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            for kid, rec in data.items():
                self._db[kid] = APIKeyRecord(**rec)
        except Exception:
            # Corrupt or empty file -> reset
            self._db = {}

    def _save(self) -> None:
        serialized = {kid: asdict(rec) for kid, rec in self._db.items()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    def verify(self, raw_key: str) -> bool:
        self._ensure_loaded()
        h = _sha256_hex(raw_key)
        for rec in self._db.values():
            if rec.key_hash == h and not rec.revoked_at:
                # Check expiry
                if rec.expires_at:
                    try:
                        exp = datetime.fromisoformat(rec.expires_at)
                        if exp < datetime.now(UTC):
                            return False
                    except Exception:
                        return False
                return True
        return False

    def add(
        self,
        owner: str,
        metadata: dict[str, Any] | None = None,
        ttl_hours: int | None = None,
    ) -> dict[str, str]:
        self._ensure_loaded()
        raw = generate_api_key()
        kid = secrets.token_urlsafe(16)
        rec = APIKeyRecord(
            key_id=kid,
            key_hash=_sha256_hex(raw),
            owner=owner,
            created_at=_now_iso(),
            metadata=metadata or {},
        )
        if ttl_hours is not None:
            try:
                # ttl_hours == 0 => immediately expired
                exp = datetime.now(UTC).timestamp() + (ttl_hours * 3600)
                rec.expires_at = datetime.fromtimestamp(exp, UTC).isoformat()
            except Exception:
                rec.expires_at = None
        self._db[kid] = rec
        self._save()
        return {"key_id": kid, "api_key": raw}

    def revoke(self, raw_key: str) -> bool:
        self._ensure_loaded()
        h = _sha256_hex(raw_key)
        changed = False
        for rec in self._db.values():
            if rec.key_hash == h and not rec.revoked_at:
                rec.revoked_at = _now_iso()
                changed = True
        if changed:
            self._save()
        return changed

    def rotate(self, raw_key: str) -> dict[str, str] | None:
        if not self.revoke(raw_key):
            return None
        # Use previous owner if found
        self._ensure_loaded()
        h = _sha256_hex(raw_key)
        owner = "unknown"
        for rec in self._db.values():
            if rec.key_hash == h:
                owner = rec.owner
                break
        return self.add(owner)

    def list_public(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return [rec.to_public() for rec in self._db.values()]

    def get_by_raw(self, raw_key: str) -> APIKeyRecord | None:
        self._ensure_loaded()
        h = _sha256_hex(raw_key)
        for rec in self._db.values():
            if rec.key_hash == h:
                return rec
        return None

    def revoke_by_id(self, key_id: str) -> bool:
        self._ensure_loaded()
        rec = self._db.get(key_id)
        if rec and not rec.revoked_at:
            rec.revoked_at = _now_iso()
            self._save()
            return True
        return False
