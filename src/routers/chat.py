"""Chat router with streaming support."""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, List
from uuid import uuid4

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .base import BaseRouter

class ChatRouter(BaseRouter):
    """Chat endpoints with streaming support."""
    
    def __init__(self, config, services):
        super().__init__(config, services)
        self.chat_sessions: Dict[str, List[Dict[str, str]]] = {}
    
    async def initialize(self) -> None:
        """Initialize chat router."""
        self.router = self.create_router(prefix="/v1/chat", tags=["chat"])
        
        # Chat history endpoint
        @self.router.get("/history")
        async def get_chat_history(
            session: str = Query(default="default")
        ) -> JSONResponse:
            """Get chat history for session."""
            history = self._get_session_messages(session)
            return JSONResponse({
                "session": session,
                "messages": history,
                "count": len(history)
            })
        
        # Clear history endpoint
        @self.router.delete("/history")
        async def clear_chat_history(
            session: str = Query(default="default")
        ) -> JSONResponse:
            """Clear chat history for session."""
            self.chat_sessions[session] = []
            return JSONResponse({
                "session": session,
                "cleared": True
            })
        
        # Streaming chat endpoint
        @self.router.post("/stream")
        async def chat_stream(
            request: Request
        ) -> StreamingResponse:
            """Stream chat responses."""
            body = await request.json()
            message = body.get("message", "")
            session_id = body.get("session_id", "default")
            
            if not message:
                return JSONResponse(
                    {"error": "Message required"}, 
                    status_code=400
                )
            
            return StreamingResponse(
                self._stream_chat_response(message, session_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Non-streaming chat endpoint
        @self.router.post("")
        async def chat(
            request: Request
        ) -> JSONResponse:
            """Non-streaming chat endpoint."""
            body = await request.json()
            message = body.get("message", "")
            session_id = body.get("session_id", "default")
            
            if not message:
                return JSONResponse(
                    {"error": "Message required"}, 
                    status_code=400
                )
            
            # Add to history
            self._get_session_messages(session_id).append({
                "role": "user",
                "content": message
            })
            
            # Generate response
            response = await self._generate_response(message, session_id)
            
            # Add response to history
            self._get_session_messages(session_id).append({
                "role": "assistant", 
                "content": response
            })
            
            # Get model identity
            llm_service = self.get_service("llm_client")
            model_identity = llm_service.get_model_identity() if llm_service else {"model": "unknown"}
            
            return JSONResponse({
                "response": response,
                "session_id": session_id,
                "model": model_identity
            })
        
        self._initialized = True
        self.logger.info("Chat router initialized")
    
    def get_router(self) -> APIRouter:
        """Get the FastAPI router."""
        return self.router
    
    def _get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Get messages for a chat session."""
        return self.chat_sessions.setdefault(session_id, [])
    
    async def _stream_chat_response(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """Stream chat response using Server-Sent Events."""
        event_id = str(uuid4())
        
        # Add user message to history
        self._get_session_messages(session_id).append({
            "role": "user",
            "content": message
        })
        
        # Get LLM service
        llm_service = self.get_service("llm_client")
        model_identity = llm_service.get_model_identity() if llm_service else {"model": "unknown"}
        
        # Send start event
        yield self._sse_pack("start", {
            "id": event_id,
            "session": session_id,
            "model": model_identity
        })
        
        # Stream response
        accumulated_response = []
        
        if llm_service:
            # Build conversation history
            history = self._get_session_messages(session_id)
            messages = [
                {"role": "system", "content": "You are Super Alita, a helpful AI assistant."},
                *history
            ]
            
            try:
                async for chunk in llm_service.stream_chat(messages):
                    if chunk.get("type") == "content":
                        content = chunk.get("content", "")
                        accumulated_response.append(content)
                        yield self._sse_pack("content", {"content": content})
                        
            except Exception as e:
                error_content = f"Sorry, I encountered an error: {str(e)}"
                accumulated_response.append(error_content)
                yield self._sse_pack("content", {"content": error_content})
        else:
            # Fallback response when LLM service unavailable
            fallback = "I'm running in fallback mode. LLM service is not available."
            accumulated_response.append(fallback)
            
            for word in fallback.split():
                yield self._sse_pack("content", {"content": word + " "})
                await asyncio.sleep(0.05)
        
        # Add assistant response to history
        full_response = "".join(accumulated_response)
        self._get_session_messages(session_id).append({
            "role": "assistant",
            "content": full_response
        })
        
        # Send completion event
        yield self._sse_pack("done", {"reason": "complete"})
    
    async def _generate_response(self, message: str, session_id: str) -> str:
        """Generate non-streaming chat response."""
        llm_service = self.get_service("llm_client")
        
        if llm_service:
            # Build conversation
            history = self._get_session_messages(session_id)
            messages = [
                {"role": "system", "content": "You are Super Alita, a helpful AI assistant."},
                *history,
                {"role": "user", "content": message}
            ]
            
            # Collect streaming response
            response_parts = []
            try:
                async for chunk in llm_service.stream_chat(messages):
                    if chunk.get("type") == "content":
                        response_parts.append(chunk.get("content", ""))
                        
                return "".join(response_parts)
                
            except Exception as e:
                return f"Sorry, I encountered an error: {str(e)}"
        
        # Fallback response
        return self._fallback_response(message)
    
    def _fallback_response(self, message: str) -> str:
        """Generate fallback response when LLM unavailable."""
        message_lower = message.lower().strip()
        
        if any(greeting in message_lower for greeting in ["hi", "hello", "hey"]):
            return "Hello! I'm Super Alita, your AI assistant. How can I help you today?"
        
        if any(help_word in message_lower for help_word in ["help", "what can you do"]):
            return "I can help with code generation, research paper implementation, web scraping, and more!"
        
        return f"I understand you're asking about: '{message}'. I'm currently running in fallback mode. Please ensure the LLM is properly configured."
    
    def _sse_pack(self, event_type: str, data: Dict[str, Any]) -> str:
        """Pack data for Server-Sent Events."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"