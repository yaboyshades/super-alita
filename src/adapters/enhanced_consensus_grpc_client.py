"""Async gRPC client for external consensus service.

This is a thin wrapper mirroring the local provider's consensus_sampling
interface so callers can remain agnostic.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

try:  # optional dependency
    import grpc  # type: ignore
except Exception:  # pragma: no cover
    grpc = None  # type: ignore

def _import_stubs():  # lazy import / generation
    try:  # type: ignore
        from protos.consensus_pb2 import ConsensusRequest  # type: ignore
        from protos.consensus_pb2_grpc import (  # type: ignore
            ConsensusServiceStub,
        )
        return ConsensusRequest, ConsensusServiceStub
    except Exception:  # pragma: no cover
        # Attempt generation
        try:
            import subprocess
            import sys
            import pathlib

            root = pathlib.Path(__file__).resolve().parent.parent
            script = root / "scripts" / "gen_protos.py"
            if script.exists():
                subprocess.check_call([sys.executable, str(script)])
                from protos.consensus_pb2 import (  # type: ignore
                    ConsensusRequest,  # noqa: WPS433
                )
                from protos.consensus_pb2_grpc import (  # type: ignore
                    ConsensusServiceStub,  # noqa: WPS433
                )
                return ConsensusRequest, ConsensusServiceStub
        except Exception:
            return None, None
        return None, None

ConsensusRequest, ConsensusServiceStub = _import_stubs()


@dataclass
class ConsensusResult:
    result: str
    confidence: float
    method_used: str
    raw_responses: list[str]
    confidence_scores: dict[str, float]


class EnhancedConsensusGrpcClient:
    def __init__(self, config: dict[str, Any]):
        if (
            grpc is None
            or ConsensusServiceStub is None
            or ConsensusRequest is None
        ):
            raise RuntimeError("gRPC consensus stubs not available")
        self.config = config
        self.server_url = self.config.get("grpc_url", "localhost:50051")
        self.timeout = float(self.config.get("timeout", 60))
        self._channel = grpc.aio.insecure_channel(  # type: ignore[attr-defined]
            self.server_url,
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        self._stub = ConsensusServiceStub(self._channel)  # type: ignore

    async def consensus_sampling(
        self, prompt: str, **kwargs: Any
    ) -> dict[str, Any]:
        req = ConsensusRequest(  # type: ignore
            prompt=prompt,
            method=kwargs.get("method", "weighted_vote"),
            num_samples=kwargs.get("num_samples", 3),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
            confidence_threshold=kwargs.get("confidence_threshold", 0.7),
            temperature_range=kwargs.get("temperature_range", 0.2),
        )
        try:
            resp = await asyncio.wait_for(
                self._stub.GetConsensus(req),
                timeout=self.timeout,  # type: ignore
            )
            return {
                "consensus_text": resp.result,
                "consensus_confidence": resp.confidence,
                "aggregation_method": resp.method_used,
                "individual_responses": list(resp.raw_responses),
                "confidence_scores": list(resp.confidence_scores.values()),
                "metadata": {"source": "grpc"},
            }
        except asyncio.TimeoutError as e:  # noqa: PERF203
            raise TimeoutError(
                f"Consensus gRPC timeout after {self.timeout}s"
            ) from e
        except grpc.RpcError as e:  # type: ignore[attr-defined]
            raise RuntimeError(f"Consensus gRPC error: {e}") from e
