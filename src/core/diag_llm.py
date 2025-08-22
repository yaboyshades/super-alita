# src/core/diag_llm.py
"""
Diagnostic script to validate LLM provider integration.
Run this script to verify your LLM provider is correctly configured.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.llm_client import LLMUnavailable, generate  # noqa: E402
from src.core.settings import LLM_MODEL, LLM_RETRIES, LLM_TIMEOUT_SEC  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("diag_llm")


async def test_llm():
    """Test the LLM client to verify connectivity and response."""
    # Load environment variables from .env file
    dotenv_path = Path(project_root) / ".env"
    log.info("Loading environment from %s", dotenv_path)
    load_dotenv(dotenv_path=dotenv_path, verbose=True)

    log.info("=" * 60)
    log.info("LLM PROVIDER DIAGNOSTIC")
    log.info("=" * 60)
    log.info("Model: %s", LLM_MODEL)
    log.info("Timeout: %s seconds", LLM_TIMEOUT_SEC)
    log.info("Retries: %s", LLM_RETRIES)
    log.info("-" * 60)

    # Check for API key
    api_key_name = (
        "GEMINI_API_KEY" if "gemini" in LLM_MODEL.lower() else "OPENAI_API_KEY"
    )
    api_key = os.environ.get(api_key_name)
    if not api_key:
        log.error("❌ %s environment variable not set!", api_key_name)
        log.error("Please set the API key and try again.")
        return False
    log.info("✅ %s environment variable is set", api_key_name)

    # Test simple generation
    prompt = "Please provide a short hello world message."
    start_time = time.time()
    try:
        log.info("Sending test prompt to LLM provider...")
        response = await generate(prompt)
        elapsed = time.time() - start_time
        log.info("✅ LLM response received in %.2f seconds", elapsed)
        log.info("-" * 60)
        log.info("Response: %s", response)
        log.info("-" * 60)
        log.info("LLM integration test PASSED!")
    except LLMUnavailable:
        elapsed = time.time() - start_time
        log.exception("❌ LLM test FAILED after %.2f seconds", elapsed)
        return False
    except Exception:
        elapsed = time.time() - start_time
        log.exception("❌ Unexpected error during LLM test (%.2f seconds)", elapsed)
        return False
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose LLM provider integration")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = asyncio.run(test_llm())
    sys.exit(0 if success else 1)
