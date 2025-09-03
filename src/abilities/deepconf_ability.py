"""
DeepConf consensus sampling ability for Super Alita

This provides a high-level interface for performing consensus-based
sampling using the DeepConf methodology with multiple consensus modes.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.abilities.mangle.mangle_ability import MangleAbility
from src.plugins.plugin_interface import PluginInterface


class ConsensusMode(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    HYBRID = "hybrid"


@dataclass
class DeepConfSampleRequest:
    """Request for DeepConf consensus sampling"""

    prompt: str
    temperature: float | list[float] = 0.7
    max_tokens: int = 512
    num_samples: int = 3
    mode: ConsensusMode = ConsensusMode.OFFLINE
    consensus_method: str = "weighted_vote"
    confidence_threshold: float = 0.7
    domain: str | None = None


@dataclass
class DeepConfResponse:
    """Response from DeepConf consensus sampling"""

    consensus_text: str
    consensus_confidence: float
    individual_responses: list[dict[str, Any]]
    metadata: dict[str, Any]
    aggregation_method: str
    raw_results: dict[str, Any] | None = None


class DeepConfAbility(PluginInterface):
    """
    High-level DeepConf consensus sampling ability

    Provides multiple consensus modes:
    - OFFLINE: Pre-computed consensus with cached results
    - ONLINE: Real-time consensus generation
    - HYBRID: Adaptive switching between offline/online based on context
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.vllm_client: VLLMDeepConfClient | None = None
        self.pipeline: EnhancedDeepConfPipeline | None = None
        self.initialized = False
        self._mangle: MangleAbility | None = None

        # Default configuration (updated for Ollama integration)
        self.default_config = {
            "vllm_base_url": "http://localhost:11434/v1",
            "model_name": "gpt-oss:20b",
            "timeout": 60.0,
            "max_retries": 3,
            "enable_caching": True,
            "cache_ttl": 3600,
        }

    async def initialize(self, event_bus, **kwargs) -> bool:
        """Initialize the DeepConf ability"""
        try:
            # Merge configuration
            final_config = {**self.default_config, **self.config, **kwargs}

            # Initialize vLLM client
            self.vllm_client = VLLMDeepConfClient(
                base_url=final_config["vllm_base_url"],
                model_name=final_config["model_name"],
                timeout=final_config["timeout"],
                max_retries=final_config["max_retries"],
            )

            # Initialize pipeline
            self.pipeline = EnhancedDeepConfPipeline(model_api=self.vllm_client)
            # Initialize Mangle (optional)
            try:
                self._mangle = MangleAbility()
            except Exception:
                self._mangle = None

            self.initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize DeepConf ability: {e}")
            return False

    async def cleanup(self) -> None:
        """Clean shutdown of the DeepConf ability"""
        if self.vllm_client:
            await self.vllm_client.close()
        self.initialized = False

    def get_plugin_info(self) -> dict[str, Any]:
        """Get plugin information"""
        return {
            "name": "DeepConfAbility",
            "version": "1.0.0",
            "description": "High-level DeepConf consensus sampling ability",
            "type": "ability",
            "capabilities": [
                "consensus_sampling",
                "confidence_calibration",
                "multi_mode_operation",
                "caching",
                "batch_processing",
            ],
            "supported_modes": ["offline", "online", "hybrid"],
            "supported_aggregation": [
                "weighted_vote",
                "borda_count",
                "condorcet",
                "similarity_clustering",
            ],
        }

    async def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """
        Process an event and return result if applicable

        Args:
            event: Event data dictionary

        Returns:
            Result dictionary or None if event not handled
        """
        if not self.initialized:
            return None

        # Handle consensus sampling events
        if event.get("type") == "consensus_request":
            try:
                response = await self.sample_consensus(
                    prompt=event.get("prompt", ""), **event.get("params", {})
                )
                return {
                    "type": "consensus_response",
                    "consensus_text": response.consensus_text,
                    "confidence": response.consensus_confidence,
                    "metadata": response.metadata,
                }
            except Exception as e:
                return {
                    "type": "error",
                    "message": f"Consensus sampling failed: {e}",
                }

        # Event not handled by this ability
        return None

    async def sample_consensus(
        self,
        prompt: str,
        temperature: float | list[float] = 0.7,
        max_tokens: int = 512,
        num_samples: int = 3,
        mode: ConsensusMode = ConsensusMode.OFFLINE,
        consensus_method: str = "weighted_vote",
        confidence_threshold: float = 0.7,
        domain: str | None = None,
    ) -> DeepConfResponse:
        """
        Generate consensus response using DeepConf methodology

        Args:
            prompt: Input prompt for consensus generation
            temperature: Sampling temperature(s)
            max_tokens: Maximum tokens to generate
            num_samples: Number of samples for consensus
            mode: Consensus mode (offline/online/hybrid)
            consensus_method: Aggregation method
            confidence_threshold: Minimum confidence for acceptance
            domain: Optional domain for calibration

        Returns:
            DeepConfResponse with consensus text and metadata
        """
        if not self.initialized:
            raise RuntimeError("DeepConf ability not initialized")

        request = DeepConfSampleRequest(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            num_samples=num_samples,
            mode=mode,
            consensus_method=consensus_method,
            confidence_threshold=confidence_threshold,
            domain=domain,
        )

        if mode == ConsensusMode.OFFLINE:
            return await self._offline_consensus(request)
        elif mode == ConsensusMode.ONLINE:
            return await self._online_consensus(request)
        else:  # HYBRID
            return await self._hybrid_consensus(request)

    async def _offline_consensus(
        self, request: DeepConfSampleRequest
    ) -> DeepConfResponse:
        """Generate consensus using offline/cached approach"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self.pipeline.get_cached_result(cache_key)

            if cached_result:
                return DeepConfResponse(
                    consensus_text=cached_result["consensus_text"],
                    consensus_confidence=cached_result["confidence"],
                    individual_responses=cached_result["responses"],
                    metadata={"source": "cache", "cache_key": cache_key},
                    aggregation_method=request.consensus_method,
                    raw_results=cached_result,
                )

            # Generate new consensus
            result = await self.pipeline.generate_consensus(
                prompt=request.prompt,
                num_samples=request.num_samples,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                method=request.consensus_method,
            )

            # Cache the result
            await self.pipeline.cache_result(cache_key, result, ttl=3600)

            return DeepConfResponse(
                consensus_text=result["consensus_text"],
                consensus_confidence=result["confidence"],
                individual_responses=result["individual_responses"],
                metadata={"source": "generated", "mode": "offline"},
                aggregation_method=request.consensus_method,
                raw_results=result,
            )

        except Exception as e:
            raise RuntimeError(f"Offline consensus failed: {e}")

    async def _online_consensus(
        self, request: DeepConfSampleRequest
    ) -> DeepConfResponse:
        """Generate consensus using online/real-time approach"""
        try:
            # Generate samples concurrently
            tasks = []
            temperatures = self._prepare_temperatures(
                request.temperature, request.num_samples
            )

            for i in range(request.num_samples):
                task = self.vllm_client.generate_with_confidence(
                    prompt=request.prompt,
                    temperature=temperatures[i],
                    max_tokens=request.max_tokens,
                )
                tasks.append(task)

            # Wait for all samples
            samples = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful samples
            valid_samples = [s for s in samples if not isinstance(s, Exception)]

            if not valid_samples:
                raise RuntimeError("No valid samples generated")

            # Aggregate using pipeline
            aggregation_result = (
                await self.pipeline.consensus_aggregator.aggregate_responses(
                    valid_samples, method=request.consensus_method
                )
            )

            # Calibrate confidence if domain specified
            final_confidence = aggregation_result["confidence"]
            if request.domain:
                calibration_result = (
                    await self.pipeline.confidence_calibrator.calibrate_confidence(
                        aggregation_result["confidence"], domain=request.domain
                    )
                )
                final_confidence = calibration_result["calibrated_confidence"]

            # Optional Mangle-based adjustment
            if self._mangle:
                meta = {
                    "samples_generated": len(samples),
                    "samples_valid": len(valid_samples),
                    "response_length": len(
                        aggregation_result.get("consensus_text", "")
                    ),
                }
                final_confidence = await self._mangle.evaluate_confidence(
                    final_confidence, domain=request.domain, meta=meta
                )

            return DeepConfResponse(
                consensus_text=aggregation_result["consensus_text"],
                consensus_confidence=final_confidence,
                individual_responses=valid_samples,
                metadata={
                    "source": "generated",
                    "mode": "online",
                    "samples_generated": len(samples),
                    "samples_valid": len(valid_samples),
                },
                aggregation_method=request.consensus_method,
                raw_results=aggregation_result,
            )

        except Exception as e:
            raise RuntimeError(f"Online consensus failed: {e}")

    async def _hybrid_consensus(
        self, request: DeepConfSampleRequest
    ) -> DeepConfResponse:
        """Generate consensus using hybrid approach"""
        try:
            # Decision logic for offline vs online
            cache_key = self._generate_cache_key(request)
            cached_result = await self.pipeline.get_cached_result(cache_key)

            # Use cache if available and confidence meets threshold
            if (
                cached_result
                and cached_result.get("confidence", 0) >= request.confidence_threshold
            ):
                return DeepConfResponse(
                    consensus_text=cached_result["consensus_text"],
                    consensus_confidence=cached_result["confidence"],
                    individual_responses=cached_result["responses"],
                    metadata={"source": "cache", "mode": "hybrid"},
                    aggregation_method=request.consensus_method,
                    raw_results=cached_result,
                )

            # Fall back to online generation
            online_result = await self._online_consensus(request)
            online_result.metadata["mode"] = "hybrid"

            # Cache the new result
            await self.pipeline.cache_result(
                cache_key, online_result.raw_results, ttl=3600
            )

            return online_result

        except Exception as e:
            raise RuntimeError(f"Hybrid consensus failed: {e}")

    def _generate_cache_key(self, request: DeepConfSampleRequest) -> str:
        """Generate cache key for request"""
        import hashlib

        key_data = f"{request.prompt}_{request.num_samples}_{request.consensus_method}_{request.temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _prepare_temperatures(
        self, temperature: float | list[float], num_samples: int
    ) -> list[float]:
        """Prepare temperature list for sampling"""
        if isinstance(temperature, list):
            if len(temperature) >= num_samples:
                return temperature[:num_samples]
            else:
                # Extend with last value
                return temperature + [temperature[-1]] * (
                    num_samples - len(temperature)
                )
        else:
            return [temperature] * num_samples

    async def get_consensus_direct(self, prompt: str, **kwargs) -> str:
        """Direct API for getting consensus text"""
        response = await self.sample_consensus(prompt, **kwargs)
        return response.consensus_text

    async def batch_consensus(
        self, prompts: list[str], **kwargs
    ) -> list[DeepConfResponse]:
        """Generate consensus for multiple prompts"""
        tasks = []
        for prompt in prompts:
            task = self.sample_consensus(prompt, **kwargs)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def get_confidence_calibration(
        self, responses: list[str], domain: str
    ) -> dict[str, float]:
        """Get confidence calibration for responses"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        calibrations = {}
        for i, response in enumerate(responses):
            # Heuristic confidence extraction based on response length
            raw_confidence = len(response) / 1000.0  # Simple heuristic

            result = await self.pipeline.confidence_calibrator.calibrate_confidence(
                raw_confidence, domain=domain
            )
            calibrations[f"response_{i}"] = result["calibrated_confidence"]

        return calibrations
