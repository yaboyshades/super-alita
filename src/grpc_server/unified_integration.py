"""
Constitutional gRPC Integration with Unified Intelligence Layer

Provides seamless integration between the constitutional gRPC server and
the UnifiedSuperAlita orchestrator. Ensures constitutional compliance
while leveraging the full power of the unified intelligence architecture.

Integration Points:
- Constitutional validation at gRPC entry points
- UnifiedSuperAlita orchestrator for processing
- Mangle reasoner for constitutional compliance
- Global workspace for consciousness integration
- Neural store for memory and reasoning

Constitutional Compliance:
- Article I: Library-First (leverages existing UnifiedSuperAlita)
- Article II: Test-First (comprehensive integration tests)
- Article III: Simplicity Gate (clear separation of concerns)
- Article IV: Integration-First (end-to-end validation)
- Article V: Clarity (unambiguous integration patterns)
- Article VI: Counterfactual (justified integration decisions)
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.grpc_server.server import (
    ConstitutionalGrpcServer,
    create_constitutional_server,
)
from src.main_unified import UnifiedSuperAlita
from src.sdd.mangle_reasoner import MangleReasoner

logger = logging.getLogger(__name__)


class ConstitutionalUnifiedIntegration:
    """
    Integration orchestrator for constitutional gRPC and unified intelligence.

    Manages the lifecycle and communication between:
    - Constitutional gRPC server
    - UnifiedSuperAlita orchestrator
    - Mangle constitutional reasoner
    - Constitutional compliance validation

    Design Decisions (Article VI: Counterfactual Justification):
    - Separate integration layer for clear concerns separation
    - Async initialization for non-blocking startup
    - Constitutional validation at integration boundaries
    - Graceful degradation when components unavailable
    """

    def __init__(
        self,
        grpc_host: str = "0.0.0.0",
        grpc_port: int = 50052,
        config_path: Path | None = None,
    ):
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.config_path = config_path or Path("src/config/agent.yaml")

        # Component instances
        self.unified_agent: UnifiedSuperAlita | None = None
        self.mangle_reasoner: MangleReasoner | None = None
        self.grpc_server: ConstitutionalGrpcServer | None = None

        # Integration state
        self._initialized = False
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"Constitutional unified integration initialized "
            f"(gRPC: {grpc_host}:{grpc_port})"
        )

    async def initialize(self) -> None:
        """
        Initialize all components with constitutional validation.

        Follows constitutional principles:
        - Article I: Uses existing UnifiedSuperAlita library
        - Article II: Includes validation of component initialization
        - Article III: Simple, clear initialization sequence
        """
        try:
            logger.info("🔄 Initializing constitutional unified integration...")

            # Initialize unified intelligence orchestrator (Article I: Library-First)
            logger.info("🧠 Starting unified intelligence layer...")
            self.unified_agent = UnifiedSuperAlita(config_path=self.config_path)

            # Initialize in background to avoid blocking
            # Note: In production, you might want to await this initialization
            asyncio.create_task(self.unified_agent.run())

            # Allow time for plugin initialization
            await asyncio.sleep(2)

            # Initialize constitutional reasoner (Article I: Library-First)
            logger.info("⚖️ Starting constitutional reasoner...")
            self.mangle_reasoner = MangleReasoner()

            # Validate constitutional compliance of the integration itself
            await self._validate_integration_compliance()

            # Initialize constitutional gRPC server
            logger.info("🚀 Starting constitutional gRPC server...")
            self.grpc_server = await create_constitutional_server(
                host=self.grpc_host,
                port=self.grpc_port,
                unified_agent=self.unified_agent,
                mangle_reasoner=self.mangle_reasoner,
                enable_reflection=True,
                enable_health_checking=True,
            )

            self._initialized = True
            logger.info(
                "✅ Constitutional unified integration initialized successfully"
            )

        except Exception as e:
            logger.error(f"❌ Constitutional integration initialization failed: {e}")
            await self.cleanup()
            raise

    async def start(self) -> None:
        """Start all components with constitutional monitoring."""
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("🚀 Starting constitutional unified integration...")

            # Start gRPC server
            if self.grpc_server:
                await self.grpc_server.start()

            self._running = True

            # Log constitutional compliance status
            await self._log_constitutional_status()

            logger.info("✅ Constitutional unified integration started successfully")
            logger.info(
                f"🌐 gRPC server available at {self.grpc_host}:{self.grpc_port}"
            )
            logger.info("⚖️ Constitutional compliance monitoring active")
            logger.info("🧠 Unified intelligence layer ready")

        except Exception as e:
            logger.error(f"❌ Failed to start constitutional integration: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop all components gracefully with constitutional cleanup."""
        if not self._running:
            return

        try:
            logger.info("🔄 Stopping constitutional unified integration...")

            # Stop gRPC server
            if self.grpc_server:
                await self.grpc_server.stop()

            # Stop unified agent
            if self.unified_agent:
                await self.unified_agent.shutdown()

            self._running = False
            self._shutdown_event.set()

            # Final constitutional compliance report
            await self._log_final_compliance_report()

            logger.info("✅ Constitutional unified integration stopped gracefully")

        except Exception as e:
            logger.error(f"❌ Error stopping constitutional integration: {e}")
            raise

    async def serve_forever(self) -> None:
        """Run integration until shutdown signal with constitutional monitoring."""
        try:
            await self.start()

            # Monitor constitutional compliance periodically
            monitoring_task = asyncio.create_task(
                self._monitor_constitutional_compliance()
            )

            # Wait for shutdown
            await self._shutdown_event.wait()

            # Cancel monitoring
            monitoring_task.cancel()

        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Constitutional integration error: {e}")
            raise
        finally:
            await self.stop()

    async def cleanup(self) -> None:
        """Cleanup resources with constitutional validation."""
        try:
            if self.grpc_server:
                await self.grpc_server.stop()

            if self.unified_agent:
                await self.unified_agent.shutdown()

            logger.info("✅ Constitutional integration cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during constitutional cleanup: {e}")

    def is_healthy(self) -> bool:
        """Check overall health of constitutional integration."""
        try:
            grpc_healthy = (
                self.grpc_server is not None and self.grpc_server.is_running()
            )
            unified_healthy = self.unified_agent is not None
            mangle_healthy = self.mangle_reasoner is not None

            return all([grpc_healthy, unified_healthy, mangle_healthy])

        except Exception:
            return False

    async def get_integration_status(self) -> dict[str, Any]:
        """Get comprehensive integration status with constitutional metrics."""
        status = {
            "initialized": self._initialized,
            "running": self._running,
            "healthy": self.is_healthy(),
            "components": {
                "grpc_server": {
                    "available": self.grpc_server is not None,
                    "running": (
                        self.grpc_server.is_running() if self.grpc_server else False
                    ),
                    "host": self.grpc_host,
                    "port": self.grpc_port,
                },
                "unified_agent": {
                    "available": self.unified_agent is not None,
                    "plugins_count": (
                        len(self.unified_agent.plugins) if self.unified_agent else 0
                    ),
                },
                "mangle_reasoner": {
                    "available": self.mangle_reasoner is not None,
                },
            },
            "constitutional_compliance": {
                "validation_enabled": True,
                "framework_version": "1.0",
                "articles_enforced": 6,
            },
        }

        # Add server stats if available
        if self.grpc_server:
            status["grpc_stats"] = self.grpc_server.get_server_stats()

        # Add constitutional compliance check
        if self.mangle_reasoner:
            try:
                compliance_results = (
                    self.mangle_reasoner.validate_constitutional_compliance()
                )
                total_violations = sum(
                    len(result.raw_results) for result in compliance_results.values()
                )
                status["constitutional_compliance"]["violations"] = total_violations
                status["constitutional_compliance"]["compliant"] = total_violations == 0
            except Exception as e:
                logger.warning(f"Constitutional compliance check failed: {e}")

        return status

    # Private Methods (Article III: Simplicity Gate)

    async def _validate_integration_compliance(self) -> None:
        """Validate constitutional compliance of the integration itself."""
        try:
            if not self.mangle_reasoner:
                logger.warning(
                    "Cannot validate integration compliance: "
                    "Mangle reasoner unavailable"
                )
                return

            # Run constitutional validation on the integration
            compliance_results = (
                self.mangle_reasoner.validate_constitutional_compliance()
            )

            total_violations = sum(
                len(result.raw_results) for result in compliance_results.values()
            )

            if total_violations > 0:
                logger.warning(
                    f"Constitutional violations detected: {total_violations}"
                )
                for article, result in compliance_results.items():
                    if result.raw_results:
                        logger.warning(f"{article}: {result.raw_results}")
            else:
                logger.info(
                    "✅ Integration passes constitutional compliance validation"
                )

        except Exception as e:
            logger.error(f"Constitutional compliance validation failed: {e}")

    async def _log_constitutional_status(self) -> None:
        """Log current constitutional compliance status."""
        try:
            status = await self.get_integration_status()
            compliance = status.get("constitutional_compliance", {})

            logger.info("⚖️ Constitutional Compliance Status:")
            logger.info(
                f"   Framework Version: "
                f"{compliance.get('framework_version', 'unknown')}"
            )
            logger.info(
                f"   Articles Enforced: {compliance.get('articles_enforced', 0)}"
            )
            logger.info(f"   Validation: {compliance.get('validation_enabled', False)}")

            if "violations" in compliance:
                violations = compliance["violations"]
                if violations == 0:
                    logger.info("   ✅ No constitutional violations detected")
                else:
                    logger.warning(
                        f"   ⚠️ {violations} constitutional violations detected"
                    )

        except Exception as e:
            logger.error(f"Failed to log constitutional status: {e}")

    async def _monitor_constitutional_compliance(self) -> None:
        """Continuously monitor constitutional compliance."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._log_constitutional_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Constitutional monitoring error: {e}")

    async def _log_final_compliance_report(self) -> None:
        """Log final constitutional compliance report on shutdown."""
        try:
            logger.info("📊 Final Constitutional Compliance Report:")
            status = await self.get_integration_status()

            if self.grpc_server:
                stats = self.grpc_server.get_server_stats()
                requests = stats.get("requests_processed", 0)
                uptime = stats.get("uptime", 0)

                logger.info(f"   Total Requests Processed: {requests}")
                logger.info(f"   Total Uptime: {uptime:.2f} seconds")

            compliance = status.get("constitutional_compliance", {})
            violations = compliance.get("violations", "unknown")
            logger.info(f"   Final Violation Count: {violations}")

        except Exception as e:
            logger.error(f"Failed to generate final compliance report: {e}")


# Convenience functions for easy deployment


async def create_constitutional_unified_integration(
    grpc_host: str = "0.0.0.0",
    grpc_port: int = 50052,
    config_path: Path | None = None,
) -> ConstitutionalUnifiedIntegration:
    """
    Create and initialize constitutional unified integration.

    Args:
        grpc_host: gRPC server host
        grpc_port: gRPC server port
        config_path: Path to unified agent configuration

    Returns:
        Initialized constitutional integration
    """
    integration = ConstitutionalUnifiedIntegration(
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        config_path=config_path,
    )

    await integration.initialize()
    return integration


async def run_constitutional_unified_integration(
    grpc_host: str = "0.0.0.0",
    grpc_port: int = 50052,
    config_path: Path | None = None,
) -> None:
    """
    Run constitutional unified integration until shutdown.

    Args:
        grpc_host: gRPC server host
        grpc_port: gRPC server port
        config_path: Path to unified agent configuration
    """
    integration = await create_constitutional_unified_integration(
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        config_path=config_path,
    )

    try:
        await integration.serve_forever()
    finally:
        await integration.cleanup()


# Export key components
__all__ = [
    "ConstitutionalUnifiedIntegration",
    "create_constitutional_unified_integration",
    "run_constitutional_unified_integration",
]
