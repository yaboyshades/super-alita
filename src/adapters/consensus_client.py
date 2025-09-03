"""Consensus service adapter providing local/grpc/hybrid modes.

Env vars:
    CONSENSUS_SERVICE_MODE = local|grpc|hybrid (default: local)
    CONSENSUS_GRPC_URL     = host:port for remote service (default: localhost:50051)

Falls back to local provider if gRPC unavailable or errors (hybrid mode).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from src.abilities.enhanced_consensus_ability import EnhancedConsensusProvider

try:  # optional import
    import grpc  # type: ignore
except Exception:  # pragma: no cover
    grpc = None  # type: ignore


class ConsensusServiceMode(str, Enum):
    LOCAL = "local"
    GRPC = "grpc"
    HYBRID = "hybrid"


class ConsensusClient:
    """Adapter that abstracts consensus implementation source."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        mode_raw = os.getenv("CONSENSUS_SERVICE_MODE", "local").lower()
        valid_modes = {m.value for m in ConsensusServiceMode}
        self.mode = (
            ConsensusServiceMode(mode_raw)
            if mode_raw in valid_modes
            else ConsensusServiceMode.LOCAL
        )
        self.grpc_url = os.getenv(
            "CONSENSUS_GRPC_URL",
            self.config.get("grpc_url", "localhost:50051"),
        )
        self.timeout = float(self.config.get("timeout", 60.0))

        # Always have a local provider for fallback or direct usage
        self.local_provider = EnhancedConsensusProvider(
            {
                "base_url": self.config.get("base_url", "http://localhost:11434/v1"),
                "model_name": self.config.get("model_name", "gpt-oss:20b"),
                "timeout": self.timeout,
            }
        )

        self._grpc_client = None
        wants_grpc = self.mode in (
            ConsensusServiceMode.GRPC,
            ConsensusServiceMode.HYBRID,
        )
        if wants_grpc and grpc is not None:
            try:
                from .enhanced_consensus_grpc_client import (
                    EnhancedConsensusGrpcClient,
                )

                self._grpc_client = EnhancedConsensusGrpcClient(
                    {"grpc_url": self.grpc_url, "timeout": self.timeout}
                )
            except Exception as e:  # pragma: no cover
                print(f"⚠️  gRPC consensus client init failed: {e}")
                if self.mode == ConsensusServiceMode.GRPC:
                    raise

    async def initialize(self) -> None:
        await self.local_provider.initialize()

    async def consensus_sampling(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        # Local only or missing remote client
        if self.mode == ConsensusServiceMode.LOCAL or self._grpc_client is None:
            return await self.local_provider.consensus_sampling(prompt, **kwargs)

        # Pure gRPC mode
        if self.mode == ConsensusServiceMode.GRPC:
            return await self._grpc_client.consensus_sampling(prompt, **kwargs)

        # Hybrid: try remote then fallback
        try:
            return await self._grpc_client.consensus_sampling(prompt, **kwargs)
        except Exception as e:  # pragma: no cover
            print(f"⚠️  Remote consensus failed, fallback to local: {e}")
            return await self.local_provider.consensus_sampling(prompt, **kwargs)
