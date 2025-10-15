"""Clean application factory with dependency injection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import ApplicationConfig
from ..services import ServiceRegistry
from ..routers import RouterRegistry
from ..middleware import MiddlewareStack

class ApplicationFactory:
    """Clean application factory with dependency injection."""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.services = ServiceRegistry(config)
        self.routers = RouterRegistry(config)
        self.middleware = MiddlewareStack(config)
    
    async def create_application(self) -> FastAPI:
        """Create configured FastAPI application."""
        self.logger.info("Creating Super Alita v4.0 application...")
        
        # Create FastAPI app with clean lifespan management
        app = FastAPI(
            title="Super Alita v4.0",
            version="4.0.0",
            description="Modular AI Agent Runtime",
            lifespan=self._lifespan_handler
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors.allow_origins,
            allow_methods=self.config.cors.allow_methods,
            allow_headers=self.config.cors.allow_headers,
            allow_credentials=self.config.cors.allow_credentials,
        )
        
        # Initialize services
        await self.services.initialize()
        app.state.services = self.services
        
        # Add middleware stack
        await self.middleware.setup(app, self.services)
        
        # Register routers
        await self.routers.register_all(app, self.services)
        
        # Mount static files
        await self._mount_static_assets(app)
        
        self.logger.info("✅ Super Alita v4.0 application created successfully")
        return app
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Clean lifespan management."""
        # Startup
        self.logger.info("🚀 Starting Super Alita v4.0...")
        
        try:
            # Initialize services
            await app.state.services.startup()
            
            # Emit startup events
            event_bus = app.state.services.get("event_bus")
            if event_bus:
                await event_bus.emit({
                    "type": "application_started",
                    "version": "4.0.0",
                    "config_profile": self.config.profile
                })
            
            self.logger.info("✅ Super Alita v4.0 started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Startup failed: {e}")
            raise
        
        yield
        
        # Shutdown
        self.logger.info("🛑 Shutting down Super Alita v4.0...")
        
        try:
            await app.state.services.shutdown()
            self.logger.info("✅ Super Alita v4.0 shut down cleanly")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {e}")
    
    async def _mount_static_assets(self, app: FastAPI) -> None:
        """Mount static file serving."""
        if not self.config.static.enabled:
            return
        
        static_path = Path(self.config.static.directory)
        if not static_path.exists():
            self.logger.warning(f"Static directory not found: {static_path}")
            return
        
        try:
            app.mount(
                self.config.static.mount_path,
                StaticFiles(directory=str(static_path)),
                name="static"
            )
            
            # Serve index.html at root if available
            index_path = static_path / "index.html"
            if index_path.exists():
                from fastapi import FileResponse
                
                @app.get("/")
                async def serve_index():
                    return FileResponse(str(index_path))
            
            self.logger.info(f"✅ Static assets mounted at {self.config.static.mount_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to mount static assets: {e}")

# Factory function for backward compatibility
def create_application(config: ApplicationConfig = None) -> FastAPI:
    """Create Super Alita application with clean architecture."""
    if config is None:
        config = ApplicationConfig.from_env()
    
    factory = ApplicationFactory(config)
    return factory.create_application()