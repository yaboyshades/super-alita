#!/usr/bin/env python3
"""
Run the Mangle integration demo with a mock Mangle binary.

This script sets up the mock Mangle binary and runs the demo integration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Set up the environment to use the mock Mangle binary
os.environ["MANGLE_BIN_PATH"] = str(Path(__file__).parent / "mock_mangle.py")
os.chmod(os.environ["MANGLE_BIN_PATH"], 0o755)  # Make it executable

# Make the main directory importable
parent_dir = str(Path(__file__).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import needed modules
if __name__ == "__main__":
    print("🔍 Running Mangle integration demo with mock implementation")
    print(f"📌 Using mock binary at: {os.environ['MANGLE_BIN_PATH']}")
    # Import and run the demo
    from examples.mangle_integration_demo import main

    asyncio.run(main())
