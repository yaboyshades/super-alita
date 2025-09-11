"""Async gRPC server exposing the enhanced consensus provider.

Run directly:

    python src/consensus_grpc/server.py

Environment variables:
    CONSENSUS_GRPC_BIND   host:port bind address (default: 0.0.0.0:50051)
    CONSENSUS_MODEL_NAME  underlying model name (default: gpt-oss:20b)
    CONSENSUS_BASE_URL    underlying model base url
                        (default: http://localhost:11434/v1)
    CONSENSUS_TIMEOUT     provider timeout seconds (default: 60)

Implements the consensus.proto service:
    rpc GetConsensus(ConsensusRequest) returns (ConsensusResponse)
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from typing import Any

# Ensure project root on sys.path so 'src' top-level package is importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR, SRC_DIR / "protos"):
    if str(p) not in sys.path:  # pragma: no cover - startup path fix
        sys.path.insert(0, str(p))

try:  # Import generated stubs; generate if missing.
    # Generated code expects plain 'consensus_pb2' import; ensure cwd in path
    from protos import consensus_pb2, consensus_pb2_grpc  # type: ignore
except Exception:  # pragma: no cover - generate stubs dynamically
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    script = ROOT / "scripts" / "gen_protos.py"
    if script.exists():
        import subprocess

        subprocess.check_call([sys.executable, str(script)])
        from protos import (  # type: ignore  # noqa: E501
            consensus_pb2,
            consensus_pb2_grpc,
        )
    else:  # pragma: no cover
        raise

import grpc  # type: ignore  # noqa: E402

from src.abilities.enhanced_consensus_ability import (  # noqa: E402
    EnhancedConsensusProvider,
)


class ConsensusService(consensus_pb2_grpc.ConsensusServiceServicer):  # type: ignore
    """gRPC servicer bridging to EnhancedConsensusProvider."""

    def __init__(self, provider: EnhancedConsensusProvider):
        self._provider = provider

    async def GetConsensus(self, request, context):  # type: ignore[override]
        """Handle consensus request via provider."""
        try:
            result: dict[str, Any] = await self._provider.consensus_sampling(
                prompt=request.prompt,
                num_samples=request.num_samples or 3,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 512,
                method=request.method or "weighted_vote",
                confidence_threshold=request.confidence_threshold or 0.7,
                temperature_range=request.temperature_range or 0.2,
            )
        except Exception as e:  # pragma: no cover
            await context.abort(  # type: ignore[attr-defined]
                grpc.StatusCode.INTERNAL,
                str(e),
            )

        raw_responses = result.get("individual_responses") or []
        scores = result.get("confidence_scores") or []
        score_map = {f"r{i}": float(s) for i, s in enumerate(scores)}

        return consensus_pb2.ConsensusResponse(  # type: ignore
            result=result.get("consensus_text", ""),
            confidence=float(result.get("consensus_confidence", 0.0)),
            method_used=result.get("aggregation_method", "weighted_vote"),
            raw_responses=list(raw_responses),
            confidence_scores=score_map,
        )


async def serve() -> None:
    """Start the gRPC server and block forever."""
    bind_addr = os.getenv("CONSENSUS_GRPC_BIND", "0.0.0.0:50051")
    model_name = os.getenv("CONSENSUS_MODEL_NAME", "gpt-oss:20b")
    base_url = os.getenv("CONSENSUS_BASE_URL", "http://localhost:11434/v1")
    timeout = float(os.getenv("CONSENSUS_TIMEOUT", "60"))

    provider = EnhancedConsensusProvider(
        {
            "model_name": model_name,
            "base_url": base_url,
            "timeout": timeout,
        }
    )
    await provider.initialize()

    server = grpc.aio.server(  # type: ignore[attr-defined]
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )
    consensus_pb2_grpc.add_ConsensusServiceServicer_to_server(  # type: ignore
        ConsensusService(provider), server
    )
    server.add_insecure_port(bind_addr)
    print(f"🚀 Consensus gRPC server listening on {bind_addr}")
    await server.start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:  # pragma: no cover
        print("Shutting down consensus gRPC server...")
        await server.stop(5)


def _main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":  # pragma: no cover
    _main()
