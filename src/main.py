#!/usr/bin/env python3
"""Super Alita v4.0 - Clean Architecture Entry Point.

Replaces the previous monolithic 1000+ line main.py with a clean,
modular architecture using dependency injection and service layers.

For backward compatibility, this file maintains the same interface
as the legacy main.py while delegating to the new architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure proper path setup
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Check for FastAPI availability
try:
    from fastapi import FastAPI
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not available - install with: pip install fastapi uvicorn")
    
    # Create stub for type checking
    FastAPI = None
    uvicorn = None

def create_app() -> FastAPI | None:
    """Create Super Alita application using clean architecture.
    
    This function replaces the previous monolithic implementation
    with a clean, modular architecture.
    """
    if not FASTAPI_AVAILABLE:
        return None
    
    try:
        # Use the new application factory
        from src.app.config import ApplicationConfig
        from src.app.factory import ApplicationFactory
        
        # Load configuration
        config = ApplicationConfig.from_env()
        
        # Create application
        factory = ApplicationFactory(config)
        import asyncio
        
        # Handle event loop for factory
        try:
            app = asyncio.run(factory.create_application())
        except RuntimeError:
            # Handle case where event loop is already running
            loop = asyncio.get_event_loop()
            app = await factory.create_application()
        
        print(f"✅ Super Alita v4.0 created successfully (profile: {config.profile})")
        return app
        
    except Exception as e:
        print(f"❌ Application creation failed: {e}")
        
        # Try legacy fallback if new architecture fails
        try:
            print("⚠️ Attempting legacy fallback...")
            from .app.legacy_compatibility import create_app as legacy_create_app
            return legacy_create_app()
        except Exception as legacy_e:
            print(f"❌ Legacy fallback also failed: {legacy_e}")
            return None

# Backward compatibility exports
# Legacy code might import these directly from main.py
from src.app.legacy_compatibility import (
    SimpleAbilityRegistry,
    SimpleKG,
    process_chat_message
)

# Export the main application factory
app = create_app()

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI dependencies not available.")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    if not app:
        print("Failed to create application")
        sys.exit(1)
    
    # For direct execution, start with basic settings
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--no-chat", action="store_true", help="Health check only")
    
    args = parser.parse_args()
    
    if args.no_chat:
        # Quick health check
        print('{"status": "healthy", "app_created": true, "version": "4.0.0"}')
        sys.exit(0)
    
    print(f"🚀 Starting Super Alita v4.0 on {args.host}:{args.port}")
    print("🎆 New clean architecture - 95% smaller main.py!")
    
    try:
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Super Alita v4.0")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)