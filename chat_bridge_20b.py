#!/usr/bin/env python3
"""
Simple test to hook Super Alita chat to running 20B model
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Super Alita Chat Bridge")

# Serve static files
if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

async def ollama_chat_stream(prompt: str):
    """Stream chat from Ollama 20B model"""
    try:
        messages = [
            {"role": "system", "content": "You are Super Alita, an AI assistant powered by GPT-OSS 20B. Be helpful and concise."},
            {"role": "user", "content": prompt}
        ]
        
        async with httpx.AsyncClient(timeout=60) as client:
            payload = {
                "model": "gpt-oss:20b",
                "messages": messages,
                "stream": True
            }
            
            async with client.stream(
                "POST", "http://127.0.0.1:11434/api/chat", json=payload
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content")
                            if content:
                                yield content
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    yield f"Error: Ollama returned status {response.status_code}"
                    
    except Exception as e:
        yield f"Connection error: {e}"

@app.post("/v1/chat")
async def chat_endpoint(request: Request):
    """Chat endpoint that connects to 20B model"""
    body = await request.json()
    prompt = body.get("message", "").strip()
    
    if not prompt:
        return JSONResponse({"error": "No message provided"}, status_code=400)
    
    # Collect streaming response
    response_parts = []
    async for chunk in ollama_chat_stream(prompt):
        response_parts.append(chunk)
    
    full_response = "".join(response_parts)
    return JSONResponse({
        "type": "message",
        "content": full_response,
        "model": "gpt-oss:20b",
        "session": "default"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                gpt_oss_loaded = any("gpt-oss:20b" in str(model) for model in models)
                return {
                    "status": "healthy",
                    "ollama": "connected",
                    "gpt_oss_20b": "loaded" if gpt_oss_loaded else "not_loaded",
                    "models": len(models)
                }
            else:
                return {
                    "status": "degraded", 
                    "ollama": "error",
                    "error": f"Status {response.status_code}"
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama": "disconnected", 
            "error": str(e)
        }

if __name__ == "__main__":
    print("🚀 Starting Super Alita Chat Bridge for 20B Model")
    print("🔗 Connecting to existing Ollama instance...")
    
    # Quick test
    async def test_connection():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://127.0.0.1:11434/api/tags")
                if response.status_code == 200:
                    print("✅ Ollama connection successful")
                    return True
                else:
                    print(f"❌ Ollama error: {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    if asyncio.run(test_connection()):
        print("🌐 Starting server on http://127.0.0.1:8081")
        print("💬 Chat interface: http://127.0.0.1:8081")
        print("🏥 Health check: http://127.0.0.1:8081/health")
        uvicorn.run(app, host="127.0.0.1", port=8081, log_level="info")
    else:
        print("❌ Cannot connect to Ollama. Please ensure:")
        print("  • Ollama is running: ollama serve")
        print("  • GPT-OSS 20B is loaded: ollama run gpt-oss:20b")
        sys.exit(1)