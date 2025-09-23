#!/usr/bin/env python3
"""
Super Alita with Mangle Startup Script

This script configures the environment and starts the Super Alita server
with Mangle integration enabled.
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def find_real_mangle_binary():
    """Try to find the real Mangle binary in the repository."""
    # Check if Mangle is in PATH
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "mangle"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "mangle"], capture_output=True, text=True)

        if result.returncode == 0:
            mangle_path = result.stdout.strip()
            print(f"✅ Found Mangle binary in PATH: {mangle_path}")
            return mangle_path
    except Exception:
        pass

    # Check for Go installation to build Mangle
    try:
        go_version = subprocess.run(["go", "version"], capture_output=True, text=True)
        if go_version.returncode == 0:
            # Check if we have cloned the mangle repository
            mangle_repo = Path("./mangle")
            if mangle_repo.exists() and (mangle_repo / "go.mod").exists():
                print("✅ Found Mangle repository, building binary...")

                # Build mangle using Go
                build_dir = Path("./bin")
                build_dir.mkdir(exist_ok=True)

                build_cmd = [
                    "go",
                    "build",
                    "-o",
                    str(
                        build_dir / "mangle"
                        + (".exe" if platform.system() == "Windows" else "")
                    ),
                    "./mangle/cmd/mangle",
                ]

                try:
                    subprocess.run(build_cmd, check=True)
                    mangle_path = str(
                        build_dir / "mangle"
                        + (".exe" if platform.system() == "Windows" else "")
                    )
                    print(f"✅ Built Mangle binary at: {mangle_path}")
                    return mangle_path
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to build Mangle: {e}")
    except Exception as e:
        print(f"⚠️ Go not available or error checking: {e}")

    return None


def setup_mangle_binary(mock_only=False):
    """Set up the Mangle binary - use real if available, otherwise use mock."""
    # Try to find or build real Mangle binary if not forced to use mock
    real_mangle = None
    if not mock_only:
        real_mangle = find_real_mangle_binary()

    if real_mangle:
        os.environ["MANGLE_BIN_PATH"] = real_mangle
        print(f"✅ Using real Mangle binary: {real_mangle}")
        return

    # Fall back to mock if real binary not available or mock forced
    reason = "mock only specified" if mock_only else "real binary not found"
    print(f"⚠️ Using mock implementation ({reason})")
    system = platform.system()

    # Create mock binary based on platform
    if system == "Windows":
        # Create a batch file for Windows
        mock_path = Path(tempfile.gettempdir()) / "mangle.bat"
        mock_content = '@echo off\necho [{"Name": "log4j", "Version": "2.14.0"}, {"Name": "junit", "Version": "4.13.1"}]'
        mock_path.write_text(mock_content)
    else:
        # Create a shell script for Unix-like systems
        mock_path = Path(tempfile.gettempdir()) / "mangle"
        mock_content = '#!/bin/sh\necho \'[{"Name": "log4j", "Version": "2.14.0"}, {"Name": "junit", "Version": "4.13.1"}]\''
        mock_path.write_text(mock_content)
        mock_path.chmod(0o755)  # Make executable

    # Set environment variable
    os.environ["MANGLE_BIN_PATH"] = str(mock_path)
    print(f"✅ Mock Mangle binary created at: {mock_path}")


def start_server(port=8080, reload=True):
    """Start the Super Alita server."""
    print(
        "🚀 Starting Super Alita server with Mangle integration " f"on port {port}..."
    )

    cmd = [sys.executable, "-m", "uvicorn", "app:app"]
    if reload:
        cmd.append("--reload")
    cmd.extend(["--port", str(port)])

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Start Super Alita with Mangle integration"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable automatic reload on code changes",
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Force use of mock Mangle binary even if real one exists",
    )

    return parser.parse_args()


def setup_environment(mock_only=False):
    """Set up the environment for Super Alita with Mangle."""
    # Enable auto-discovery of abilities
    os.environ["ALITA_AUTO_DISCOVER_ABILITIES"] = "on"

    # Create Mangle data directory if it doesn't exist
    mangle_data_dir = Path("./data/mangle")
    mangle_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Mangle data directory: {mangle_data_dir}")

    # Setup Mangle binary (real or mock)
    setup_mangle_binary(mock_only=mock_only)

    print("✅ Environment configured successfully!")
    print(f"- MANGLE_BIN_PATH: {os.environ.get('MANGLE_BIN_PATH')}")
    print(
        "- Auto-discover abilities: "
        + f"{os.environ.get('ALITA_AUTO_DISCOVER_ABILITIES')}"
    )


def main():
    """Main entry point."""
    print("==== Super Alita with Mangle Integration ====")

    # Parse command line arguments
    args = parse_args()

    # Set up environment
    setup_environment(mock_only=args.mock_only)

    # Start server with specified options
    start_server(port=args.port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
