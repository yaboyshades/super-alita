"""LLM client service with provider abstraction."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, AsyncGenerator, Optional

import httpx

from .base import BaseService

try:
    from reug_runtime.llm_client import get_llm_client, LLMClient
except ImportError:
    # Fallback LLM client
    class LLMClient:
        def __init__(self, model: str):
            self.model = model
        
        async def identify(self) -> Dict[str, str]:
            return {"model": self.model, "provider": "fallback"}
        
        async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
            # Simple fallback streaming
            response = "I'm running in fallback mode. Please configure your LLM properly."
            for word in response.split():
                yield {"type": "content", "content": word + " "}
                await asyncio.sleep(0.1)
    
    def get_llm_client(model: str) -> LLMClient:
        return LLMClient(model)

class LLMService(BaseService):
    """LLM service with provider management."""
    
    def __init__(self, config, registry):
        super().__init__(config, registry)
        self.client: LLMClient = None
        self.model_identity: Dict[str, str] = {}
    
    async def initialize(self) -> None:
        """Initialize LLM client."""
        try:
            model = self.config.llm.model
            self.client = get_llm_client(model)
            
            # Get model identity
            if hasattr(self.client, 'identify'):
                self.model_identity = await self.client.identify()
            else:
                self.model_identity = {"model": model, "provider": "unknown"}
            
            self._initialized = True
            self.logger.info(f"LLM service initialized: {self.model_identity}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            # Create fallback client
            self.client = LLMClient(self.config.llm.model)
            self.model_identity = {"model": self.config.llm.model, "provider": "fallback"}
            self._initialized = True
    
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion."""
        if not self.client:
            await self.initialize()
        
        # Try Ollama direct connection first
        if self.config.llm.model.startswith("ollama:") or "gpt-oss" in self.config.llm.model:
            async for chunk in self._stream_ollama(messages, **kwargs):
                yield chunk
            return
        
        # Use configured client
        if hasattr(self.client, 'stream_chat'):
            async for chunk in self.client.stream_chat(messages, **kwargs):
                yield chunk
        else:
            # Fallback response
            response = "LLM streaming not available. Please check configuration."
            for word in response.split():
                yield {"type": "content", "content": word + " "}
                await asyncio.sleep(0.05)
    
    async def _stream_ollama(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Direct Ollama streaming."""
        try:
            ollama_model = self.config.llm.model.replace("ollama:", "") or "gpt-oss:20b"
            
            async with httpx.AsyncClient(timeout=self.config.llm.timeout) as client:
                payload = {
                    "model": ollama_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.llm.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.llm.max_tokens)
                    }
                }
                
                async with client.stream(
                    "POST", f"{self.config.llm.ollama_host}/api/chat", json=payload
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content")
                                if content:
                                    yield {"type": "content", "content": content}
                                if data.get("done"):
                                    return
                            except json.JSONDecodeError:
                                continue
                    else:
                        self.logger.warning(f"Ollama HTTP error: {response.status_code}")
                        # Fall back to error message
                        yield {"type": "content", "content": "LLM service temporarily unavailable."}
        
        except Exception as e:
            self.logger.error(f"Ollama streaming error: {e}")
            yield {"type": "content", "content": f"LLM error: {str(e)}"}
    
    def get_model_identity(self) -> Dict[str, str]:
        """Get current model identity."""
        return self.model_identity.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM service health."""
        base_health = await super().health_check()
        
        try:
            # Test a simple completion
            messages = [{"role": "user", "content": "test"}]
            chunks = []
            async for chunk in self.stream_chat(messages):
                chunks.append(chunk)
                if len(chunks) > 3:  # Don't wait for full completion
                    break
            
            ollama_reachable = False
            if self.config.llm.model.startswith("ollama:"):
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        response = await client.get(f"{self.config.llm.ollama_host}/api/tags")
                        ollama_reachable = response.status_code == 200
                except:
                    pass
            
            return {
                **base_health,
                "model_identity": self.model_identity,
                "ollama_reachable": ollama_reachable,
                "streaming_works": len(chunks) > 0
            }
            
        except Exception as e:
            return {
                **base_health,
                "status": "unhealthy",
                "error": str(e)
            }