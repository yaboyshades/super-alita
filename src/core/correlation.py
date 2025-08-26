from __future__ import annotations
import uuid, contextvars

correlation_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)
session_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)

def get_correlation_id() -> str:
    cid = correlation_var.get()
    if cid is None:
        cid = uuid.uuid4().hex
        correlation_var.set(cid)
    return cid

def set_correlation_id(cid: str) -> None:
    correlation_var.set(cid)

def get_session_id() -> str | None:
    return session_var.get()

def set_session_id(sid: str | None) -> None:
    session_var.set(sid)