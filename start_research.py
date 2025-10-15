#!/usr/bin/env python3
"""Research launcher for Super Alita with research capabilities."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Launch research edition."""
    # Check if research dependencies are available
    try:
        import torch
        import transformers
        import peft
        print("✅ Research dependencies available")
    except ImportError as e:
        print(f"❌ Missing research dependencies: {e}")
        print("Install with: pip install -r requirements-research.txt")
        return False
    
    # Set environment for research mode
    os.environ["RESEARCH_ENABLED"] = "true"
    os.environ["ALITA_ENABLE_Z3"] = "true"
    
    # Import and run research application
    from src.main_research import main as research_main
    
    print("🔬 Starting Super Alita Research Edition...")
    try:
        asyncio.run(research_main())
        return True
    except KeyboardInterrupt:
        print("\n⚠️ Research demo interrupted")
        return True
    except Exception as e:
        print(f"❌ Research edition failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)