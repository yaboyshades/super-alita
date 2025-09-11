"""
Constitutional gRPC Server for Super Alita Agent

Production-ready gRPC server with constitutional compliance validation,
health checking, reflection, and lifecycle management. Integrates with
the unified intelligence layer following constitutional principles.

Constitutional Compliance:
- Article I: Library-First (gRPC, reflection, health checking libraries)
- Article II: Test-First (comprehensive test coverage required)
- Article III: Simplicity Gate (clear separation of concerns)
- Article IV: Integration-First (end-to-end testing)
- Article V: Clarity (unambiguous error handling)
- Article VI: Counterfactual (justification for all design decisions)
"""

import asyncio
import logging
import signal

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from src.core.mangle import super_alita_pb2_grpc as pb2_grpc
from src.main_unified import UnifiedSuperAlita
from src.sdd.mangle_reasoner import MangleReasoner

from .super_alita_servicer import ConstitutionalSuperAlitaServicer

logger = logging.getLogger(__name__)


class ConstitutionalGrpcServer:
    """
    Constitutional-compliant gRPC server for Super Alita Agent.

    Provides production-ready gRPC server with constitutional validation,
    health checking, reflection, and graceful lifecycle management.

    Design decisions (Article VI: Counterfactual Justification):
    - Uses grpc-health for standardized health checking
    - Uses grpc-reflection for development and debugging
    - Async server for better resource utilization
    - Constitutional validation at every RPC call
    - Integration with unified intelligence layer
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50052,
        max_workers: int = 10,
        enable_reflection: bool = True,
        enable_health_checking: bool = True,
    ):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.enable_reflection = enable_reflection
        self.enable_health_checking = enable_health_checking

        # Server components
        self.server: grpc.aio.Server | None = None
        self.servicer: ConstitutionalSuperAlitaServicer | None = None
        self.health_servicer: health.HealthServicer | None = None

        # Integrated components
        self.unified_agent: UnifiedSuperAlita | None = None
        self.mangle_reasoner: MangleReasoner | None = None

        # Server state
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info(f"Constitutional gRPC server initialized on {host}:{port}")

    async def setup(
        self,
        unified_agent: UnifiedSuperAlita | None = None,
        mangle_reasoner: MangleReasoner | None = None,
    ) -> None:
        """
        Setup server with constitutional components.

        Args:
            unified_agent: Unified intelligence orchestrator
            mangle_reasoner: Constitutional compliance reasoner
        """
        try:
            # Store component references
            self.unified_agent = unified_agent
            self.mangle_reasoner = mangle_reasoner

            # Create async gRPC server (Article I: Library-First)
            self.server = grpc.aio.server(
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                ]
            )

            # Initialize constitutional servicer
            self.servicer = ConstitutionalSuperAlitaServicer(
                unified_agent=unified_agent,
                mangle_reasoner=mangle_reasoner,
            )

            # Add main servicer to server
            pb2_grpc.add_SuperAlitaAgentServicer_to_server(self.servicer, self.server)

            # Setup health checking (Article I: Library-First)
            if self.enable_health_checking:
                self.health_servicer = health.HealthServicer()
                health_pb2_grpc.add_HealthServicer_to_server(
                    self.health_servicer, self.server
                )

                # Set initial health status
                self.health_servicer.set(
                    "super_alita.SuperAlitaAgent",
                    health_pb2.HealthCheckResponse.SERVING,
                )

                logger.info("Health checking enabled")

            # Setup reflection for development (Article I: Library-First)
            if self.enable_reflection:
                service_names = [
                    "super_alita.SuperAlitaAgent",
                    reflection.SERVICE_NAME,
                ]
                if self.enable_health_checking:
                    service_names.append(health.SERVICE_NAME)

                reflection.enable_server_reflection(service_names, self.server)
                logger.info("Server reflection enabled")

            # Add listening port
            listen_addr = f"{self.host}:{self.port}"
            self.server.add_insecure_port(listen_addr)

            logger.info(f"Constitutional gRPC server setup complete on {listen_addr}")

        except Exception as e:
            logger.error(f"Constitutional server setup failed: {e}")
            raise

    async def start(self) -> None:
        """Start the constitutional gRPC server."""
        if not self.server:
            raise RuntimeError("Server not setup. Call setup() first.")

        try:
            await self.server.start()
            self._running = True

            # Update health status to serving
            if self.health_servicer:
                self.health_servicer.set(
                    "super_alita.SuperAlitaAgent",
                    health_pb2.HealthCheckResponse.SERVING,
                )

            logger.info(
                f"🚀 Constitutional SuperAlita gRPC server started on "
                f"{self.host}:{self.port}"
            )
            logger.info("✅ Constitutional compliance validation active")
            logger.info("✅ Unified intelligence integration ready")

        except Exception as e:
            logger.error(f"Failed to start constitutional server: {e}")
            self._running = False
            raise

    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully with constitutional cleanup."""
        if not self.server or not self._running:
            return

        try:
            logger.info("🔄 Stopping constitutional gRPC server...")

            # Update health status to not serving
            if self.health_servicer:
                self.health_servicer.set(
                    "super_alita.SuperAlitaAgent",
                    health_pb2.HealthCheckResponse.NOT_SERVING,
                )

            # Graceful shutdown
            await self.server.stop(grace_period)
            self._running = False
            self._shutdown_event.set()

            logger.info("✅ Constitutional gRPC server stopped gracefully")

        except Exception as e:
            logger.error(f"Error stopping constitutional server: {e}")
            raise

    async def serve_forever(self) -> None:
        """Start server and wait for termination with signal handling."""
        if not self.server:
            raise RuntimeError("Server not setup. Call setup() first.")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.info("🛑 Received shutdown signal")
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            await self.start()

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Constitutional server error: {e}")
            raise
        finally:
            await self.stop()

    def is_running(self) -> bool:
        """Check if server is currently running."""
        return self._running

    async def health_check(self) -> bool:
        """Perform constitutional health check."""
        try:
            if not self.servicer:
                return False

            # Check constitutional middleware
            middleware_healthy = (
                self.servicer.constitutional_middleware.validation_enabled
            )

            # Check unified intelligence integration
            unified_healthy = self.unified_agent is not None

            # Check constitutional reasoning
            mangle_healthy = self.mangle_reasoner is not None

            return all([middleware_healthy, unified_healthy, mangle_healthy])

        except Exception as e:
            logger.error(f"Constitutional health check failed: {e}")
            return False

    def get_server_stats(self) -> dict:
        """Get constitutional server statistics."""
        stats = {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "constitutional_compliance": "enabled",
            "unified_intelligence": self.unified_agent is not None,
            "mangle_reasoning": self.mangle_reasoner is not None,
            "health_checking": self.enable_health_checking,
            "reflection": self.enable_reflection,
        }

        if self.servicer:
            stats.update(
                {
                    "requests_processed": self.servicer.request_count,
                    "uptime": time.time() - self.servicer.start_time,
                }
            )

        return stats


# Convenience function for easy deployment
async def create_constitutional_server(
    host: str = "0.0.0.0",
    port: int = 50052,
    unified_agent: UnifiedSuperAlita | None = None,
    mangle_reasoner: MangleReasoner | None = None,
    **kwargs,
) -> ConstitutionalGrpcServer:
    """
    Create and setup constitutional gRPC server.

    Args:
        host: Server host address
        port: Server port
        unified_agent: Unified intelligence orchestrator
        mangle_reasoner: Constitutional compliance reasoner
        **kwargs: Additional server configuration

    Returns:
        Configured constitutional gRPC server
    """
    server = ConstitutionalGrpcServer(host=host, port=port, **kwargs)
    await server.setup(
        unified_agent=unified_agent,
        mangle_reasoner=mangle_reasoner,
    )
    return server


# Export key components
__all__ = [
    "ConstitutionalGrpcServer",
    "create_constitutional_server",
]
