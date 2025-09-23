#!/usr/bin/env python3
"""
Quick deployment script for Constitutional gRPC Super Alita Agent Server

This script provides easy deployment options for different environments
following constitutional compliance principles.

Usage:
    python deploy_grpc_server.py --env development
    python deploy_grpc_server.py --env production --port 50052
    python deploy_grpc_server.py --validate-only
"""

import asyncio
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Repository root
        )
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False


async def deploy_development() -> None:
    """Deploy in development mode with validation."""
    print("🚀 Deploying Constitutional gRPC Server (Development Mode)")

    # Validate configuration first
    if not run_command(
        [sys.executable, "-m", "src.grpc_server.main", "--validate-only"],
        "Validating configuration",
    ):
        print("❌ Deployment aborted due to validation failure")
        return

    # Start in development mode
    print("🌟 Starting server in development mode...")
    try:
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.grpc_server.main",
            "--dev",
            "--host",
            "localhost",
            "--port",
            "50052",
        )
    except KeyboardInterrupt:
        print("🛑 Development server stopped")


async def deploy_production(port: int = 50052) -> None:
    """Deploy in production mode."""
    print("🚀 Deploying Constitutional gRPC Server (Production Mode)")

    # Validate configuration
    if not run_command(
        [sys.executable, "-m", "src.grpc_server.main", "--validate-only"],
        "Validating production configuration",
    ):
        print("❌ Production deployment aborted")
        return

    # Start production server
    print(f"🌐 Starting production server on port {port}...")
    try:
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.grpc_server.main",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--log-level",
            "INFO",
        )
    except KeyboardInterrupt:
        print("🛑 Production server stopped")


def validate_only() -> None:
    """Run validation checks only."""
    print("🔍 Running Constitutional gRPC Server Validation...")

    success = run_command(
        [sys.executable, "-m", "src.grpc_server.main", "--validate-only"],
        "Configuration validation",
    )

    if success:
        print("🎉 All validation checks passed!")
    else:
        print("❌ Validation failed - check configuration")
        sys.exit(1)


async def main() -> None:
    """Main deployment function."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Constitutional gRPC Server")
    parser.add_argument(
        "--env",
        choices=["development", "production"],
        default="development",
        help="Deployment environment",
    )
    parser.add_argument(
        "--port", type=int, default=50052, help="Server port for production"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run validation checks"
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_only()
        return

    if args.env == "development":
        await deploy_development()
    elif args.env == "production":
        await deploy_production(args.port)


if __name__ == "__main__":
    asyncio.run(main())
