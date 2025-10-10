"""
Lightweight gRPC client for Super Alita Agent (used as Mangle gateway health).

This client checks connectivity to the SuperAlita gRPC server and provides
an extension point for future Mangle-specific RPCs. When unavailable, callers
should gracefully fall back to local/CLI execution paths.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import grpc

from src.core.mangle import super_alita_pb2 as pb2  # type: ignore
from src.core.mangle import super_alita_pb2_grpc as pb2_grpc  # type: ignore


@dataclass
class GrpcHealthStatus:
    ok: bool
    status: str
    message: str


class MangleGrpcClient:
    """Thin client for SuperAlitaAgent gRPC health and future RPCs.

    This is intentionally minimal: health checks are used to decide whether
    a remote path is available; query execution should fall back when remote
    capabilities are not present.
    """

    def __init__(self, target: str, *, timeout: float = 3.0) -> None:
        self.target = target
        self.timeout = timeout
        self._channel: grpc.Channel | None = None
        self._stub: pb2_grpc.SuperAlitaAgentStub | None = None

    def connect(self) -> None:
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.target)
            self._stub = pb2_grpc.SuperAlitaAgentStub(self._channel)

    def close(self) -> None:
        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.close()
        self._channel = None
        self._stub = None

    def get_health(self) -> GrpcHealthStatus:
        self.connect()
        assert self._stub is not None
        try:
            resp = self._stub.GetHealth(
                pb2.google_dot_protobuf_dot_empty__pb2.Empty(),
                timeout=self.timeout,
            )
            status = getattr(resp, "status", 0)
            message = getattr(resp, "message", "")
            ok = bool(
                status == pb2.HealthResponse.HEALTHY
                or status == pb2.HealthResponse.DEGRADED
            )
            return GrpcHealthStatus(ok=ok, status=str(status), message=message)
        except Exception as e:  # pragma: no cover
            return GrpcHealthStatus(ok=False, status="error", message=str(e))

    # Placeholder for future remote query API.
    # def mangle_query(self, program: str) -> dict: ...
