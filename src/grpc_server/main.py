#!/usr/bin/env python3
"""
Constitutional gRPC Super Alita Agent Server

Main entry point for the constitutional gRPC server with unified intelligence
integration. Provides a production-ready deployment with constitutional
compliance monitoring and comprehensive error handling.

Usage:
    python -m src.grpc_server.main --host 0.0.0.0 --port 50052
    python -m src.grpc_server.main --config config/production.yaml

Constitutional Compliance:
- Article I: Library-First (leverages existing UnifiedSuperAlita)
- Article II: Test-First (comprehensive integration test coverage)
- Article III: Simplicity Gate (clean entry point, clear defaults)
- Article IV: Integration-First (end-to-end validation)
- Article V: Clarity (unambiguous command-line interface)
- Article VI: Counterfactual (justified architecture decisions)
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from src.grpc_server.unified_integration import (
    run_constitutional_unified_integration,
)


def setup_logging(level: str = "INFO", format_style: str = "detailed") -> None:
    """
    Configure logging for constitutional gRPC server.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_style: simple or detailed formatting
    """
    if format_style == "detailed":
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[PID:%(process)d] - %(message)s"
        )
    else:
        log_format = "%(levelname)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress overly verbose gRPC logs in production
    if level != "DEBUG":
        logging.getLogger("grpc").setLevel(logging.WARNING)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Constitutional gRPC Super Alita Agent Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="gRPC server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50052,
        help="gRPC server port",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to unified agent configuration file",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--log-format",
        choices=["simple", "detailed"],
        default="detailed",
        help="Log message format style",
    )

    # Development options
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (DEBUG logging, detailed errors)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit",
    )

    return parser


async def validate_configuration(
    host: str, port: int, config_path: Path | None
) -> bool:
    """
    Validate server configuration without starting.

    Args:
        host: Server host
        port: Server port
        config_path: Optional config file path

    Returns:
        True if configuration is valid
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(
            "🔍 Validating constitutional gRPC server configuration..."
        )

        # Validate host and port
        if not (1 <= port <= 65535):
            logger.error(f"❌ Invalid port: {port} (must be 1-65535)")
            return False

        logger.info(f"✅ Server address: {host}:{port}")

        # Validate config file if provided
        if config_path:
            if not config_path.exists():
                logger.error(f"❌ Config file not found: {config_path}")
                return False
            logger.info(f"✅ Config file: {config_path}")
        else:
            logger.info("✅ Using default configuration")

        # Test integration initialization (dry run)
        from src.grpc_server.unified_integration import (
            ConstitutionalUnifiedIntegration,
        )

        ConstitutionalUnifiedIntegration(
            grpc_host=host,
            grpc_port=port,
            config_path=config_path,
        )

        # Basic validation without full initialization
        logger.info("✅ Integration layer validation passed")

        logger.info("🎉 Configuration validation successful!")
        return True

    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


def handle_shutdown_signals(integration_task: asyncio.Task) -> None:
    """
    Set up graceful shutdown signal handlers.

    Args:
        integration_task: The main integration task to cancel
    """
    logger = logging.getLogger(__name__)

    def signal_handler(signum: int, frame) -> None:  # type: ignore
        signal_name = signal.Signals(signum).name
        logger.info(
            f"🛑 Received {signal_name}, initiating graceful shutdown..."
        )
        integration_task.cancel()

    # Handle common shutdown signals
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal_handler)


async def main() -> int:
    """
    Main entry point for constitutional gRPC server.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    if args.dev:
        log_level = "DEBUG"
        log_format = "detailed"
    else:
        log_level = args.log_level
        log_format = args.log_format

    setup_logging(log_level, log_format)
    logger = logging.getLogger(__name__)

    try:
        # Validate configuration
        is_valid = await validate_configuration(
            args.host, args.port, args.config
        )
        if not is_valid:
            logger.error("❌ Configuration validation failed")
            return 1

        if args.validate_only:
            logger.info("✅ Validation complete, exiting")
            return 0

        # Start constitutional gRPC server with unified intelligence
        logger.info(
            "🚀 Starting constitutional gRPC Super Alita Agent Server..."
        )
        logger.info("⚖️ Constitutional compliance framework: v1.0")
        logger.info("🧠 Unified intelligence integration: enabled")
        logger.info(f"🌐 Server: {args.host}:{args.port}")

        # Create and start integration
        integration_task = asyncio.create_task(
            run_constitutional_unified_integration(
                grpc_host=args.host,
                grpc_port=args.port,
                config_path=args.config,
            )
        )

        # Set up signal handlers for graceful shutdown
        handle_shutdown_signals(integration_task)

        # Run until shutdown
        await integration_task

        logger.info("✅ Constitutional gRPC server shutdown complete")
        return 0

    except asyncio.CancelledError:
        logger.info("🛑 Server shutdown requested")
        return 0
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
        return 0
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        if args.dev or log_level == "DEBUG":
            logger.exception("Full error details:")
        return 1


def cli_main() -> None:
    """CLI entry point that handles async main properly."""
    try:
        if sys.platform == "win32":
            # Windows-specific asyncio configuration
            asyncio.set_event_loop_policy(
                asyncio.WindowsProactorEventLoopPolicy()
            )

        exit_code = asyncio.run(main())
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
