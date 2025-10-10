"""
vLLM DeepConf Client - Enhanced client for consensus sampling with logprobs
Provides vLLM integration with confidence scoring and token probability analysis
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp
import numpy as np


@dataclass
class VLLMGenerationResult:
    """Result structure for vLLM generation with confidence metrics"""

    text: str
    logprobs: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"
    confidence_score: float = 0.0
    token_count: int = 0
    generation_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        if self.logprobs and not self.confidence_score:
            self.confidence_score = self._calculate_confidence_from_logprobs()

        if not self.token_count and self.logprobs:
            self.token_count = len(self.logprobs)

    def _calculate_confidence_from_logprobs(self) -> float:
        """Calculate confidence score from token log probabilities"""
        if not self.logprobs:
            return 0.0

        # Extract log probabilities for top tokens
        log_probs = []
        for token_data in self.logprobs:
            if isinstance(token_data, dict) and "logprob" in token_data:
                log_probs.append(token_data["logprob"])
            elif isinstance(token_data, dict) and "top_logprobs" in token_data:
                # Get the highest probability token
                top_tokens = token_data["top_logprobs"]
                if top_tokens:
                    log_probs.append(max(top_tokens.values()))

        if not log_probs:
            return 0.0

        # Convert to probabilities and calculate geometric mean
        probabilities = [np.exp(lp) for lp in log_probs]
        geometric_mean = np.power(
            np.prod(probabilities), 1.0 / len(probabilities)
        )

        # Normalize to [0, 1] range
        return min(1.0, max(0.0, geometric_mean))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "logprobs": self.logprobs,
            "finish_reason": self.finish_reason,
            "confidence_score": self.confidence_score,
            "token_count": self.token_count,
            "generation_time": self.generation_time,
            "metadata": self.metadata,
        }


@dataclass
class VLLMBatchRequest:
    """Batch request structure for multiple generations"""

    prompts: list[str]
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    logprobs: int = 1
    echo: bool = False
    seed: int | None = None

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert to OpenAI API format for vLLM"""
        requests = []
        for prompt in self.prompts:
            request = {
                "model": "default",  # Will be overridden by client
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "logprobs": self.logprobs,
                "echo": self.echo,
            }

            if self.stop:
                request["stop"] = self.stop
            if self.seed is not None:
                request["seed"] = self.seed

            requests.append(request)

        return requests


class VLLMDeepConfClient:
    """
    Enhanced vLLM client for DeepConf consensus sampling
    Provides high-performance generation with confidence scoring
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        model_name: str = "microsoft/DialoGPT-medium",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Performance tracking
        self.request_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

        # Session management
        self.session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper configuration"""
        if self.session is None or self.session.closed:
            async with self._session_lock:
                if self.session is None or self.session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "DeepConf-vLLM-Client/1.0",
                    }

                    self.session = aiohttp.ClientSession(
                        timeout=timeout,
                        headers=headers,
                        connector=aiohttp.TCPConnector(
                            limit=100,
                            limit_per_host=20,
                            enable_cleanup_closed=True,
                        ),
                    )

        return self.session

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def generate_with_logprobs(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        logprobs: int = 1,
        seed: int | None = None,
    ) -> VLLMGenerationResult:
        """
        Generate text with log probabilities for confidence calculation
        """
        start_time = time.time()

        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "logprobs": logprobs,
            "echo": False,
        }

        if stop:
            request_data["stop"] = stop
        if seed is not None:
            request_data["seed"] = seed

        # Execute with retry logic
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/completions", json=request_data
                ) as response:

                    if response.status == 200:
                        result_data = await response.json()
                        generation_time = time.time() - start_time

                        # Update performance metrics
                        self.request_count += 1
                        self.total_generation_time += generation_time

                        # Parse response
                        choice = result_data["choices"][0]
                        text = choice["text"]
                        finish_reason = choice.get("finish_reason", "stop")
                        logprobs_data = choice.get("logprobs")

                        # Extract token information
                        token_logprobs = []
                        if logprobs_data and "token_logprobs" in logprobs_data:
                            for i, (token, logprob) in enumerate(
                                zip(
                                    logprobs_data.get("tokens", []),
                                    logprobs_data.get("token_logprobs", []),
                                    strict=False,
                                )
                            ):
                                token_info = {
                                    "token": token,
                                    "logprob": logprob,
                                    "position": i,
                                }

                                # Add top alternatives if available
                                if (
                                    "top_logprobs" in logprobs_data
                                    and i < len(logprobs_data["top_logprobs"])
                                ):
                                    token_info["top_logprobs"] = logprobs_data[
                                        "top_logprobs"
                                    ][i]

                                token_logprobs.append(token_info)

                        # Create result object
                        result = VLLMGenerationResult(
                            text=text,
                            logprobs=token_logprobs,
                            finish_reason=finish_reason,
                            generation_time=generation_time,
                            metadata={
                                "prompt_length": len(prompt),
                                "response_length": len(text),
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "attempt": attempt + 1,
                                "model": self.model_name,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )

                        self.total_tokens_generated += result.token_count
                        return result

                    else:
                        error_text = await response.text()
                        if attempt == self.max_retries - 1:
                            raise Exception(
                                f"vLLM API error {response.status}: {error_text}"
                            )

                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(
                        f"vLLM generation failed after {self.max_retries} attempts: {str(e)}"
                    )

                await asyncio.sleep(self.retry_delay * (2**attempt))
                continue

        raise Exception("Unexpected failure in vLLM generation")

    async def batch_generate(
        self, batch_request: VLLMBatchRequest
    ) -> list[VLLMGenerationResult]:
        """
        Generate multiple completions in batch for improved efficiency
        """
        if not batch_request.prompts:
            return []

        # For now, process sequentially (vLLM batch API varies by version)
        results = []

        # Use asyncio.gather for concurrent requests
        tasks = []
        for prompt in batch_request.prompts:
            task = self.generate_with_logprobs(
                prompt=prompt,
                temperature=batch_request.temperature,
                max_tokens=batch_request.max_tokens,
                top_p=batch_request.top_p,
                frequency_penalty=batch_request.frequency_penalty,
                presence_penalty=batch_request.presence_penalty,
                stop=batch_request.stop,
                logprobs=batch_request.logprobs,
                seed=batch_request.seed,
            )
            tasks.append(task)

        # Execute concurrently with limited parallelism
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Batch generation failed for prompt {i}: {result}")
                # Create a fallback result
                fallback = VLLMGenerationResult(
                    text="[Generation failed]",
                    confidence_score=0.0,
                    finish_reason="error",
                    metadata={"error": str(result), "prompt_index": i},
                )
                valid_results.append(fallback)
            else:
                valid_results.append(result)

        return valid_results

    async def health_check(self) -> dict[str, Any]:
        """Check vLLM server health and model status"""
        try:
            session = await self._get_session()

            # Try a simple generation request
            test_request = {
                "model": self.model_name,
                "prompt": "Hello",
                "max_tokens": 1,
                "temperature": 0.0,
            }

            async with session.post(
                f"{self.base_url}/completions", json=test_request
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "model": self.model_name,
                        "base_url": self.base_url,
                        "response_time": response.headers.get(
                            "X-Response-Time"
                        ),
                        "server_info": data.get("model", {}),
                        "performance_stats": {
                            "total_requests": self.request_count,
                            "total_tokens": self.total_tokens_generated,
                            "avg_generation_time": (
                                self.total_generation_time
                                / max(1, self.request_count)
                            ),
                        },
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}: {error_text}",
                        "model": self.model_name,
                        "base_url": self.base_url,
                    }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.model_name,
                "base_url": self.base_url,
            }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get client performance statistics"""
        return {
            "request_count": self.request_count,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time": self.total_generation_time,
            "average_tokens_per_request": (
                self.total_tokens_generated / max(1, self.request_count)
            ),
            "average_generation_time": (
                self.total_generation_time / max(1, self.request_count)
            ),
            "tokens_per_second": (
                self.total_tokens_generated
                / max(0.001, self.total_generation_time)
            ),
        }


# Export main classes
__all__ = ["VLLMDeepConfClient", "VLLMGenerationResult", "VLLMBatchRequest"]
