"""
Gemini utilities with safe import handling.
Handles protobuf conflicts and package compatibility issues.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Global flag to track Gemini availability
GEMINI_AVAILABLE = False
GEMINI_ERROR = None
genai = None


def _try_import_gemini() -> tuple[bool, str | None, Any]:
    """
    Attempt to import Google Generative AI with comprehensive error handling.

    Returns:
        tuple: (success, error_message, genai_module)
    """
    # Check if explicitly disabled
    if os.getenv("DISABLE_GEMINI", "").lower() in ("true", "1", "yes"):
        return False, "Gemini disabled via DISABLE_GEMINI environment variable", None

    try:
        import google.generativeai as genai_module
    except ImportError as e:
        error_msg = f"Google Generative AI package not installed: {e}"
        return False, error_msg, None
    except TypeError as e:
        if "descriptor pool" in str(e) or "duplicate symbol" in str(e):
            error_msg = f"Protobuf descriptor conflict in Google AI package: {e}"
            # This is a known issue with google-generativeai package
            return False, error_msg, None
        error_msg = f"TypeError importing Google Generative AI: {e}"
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Unexpected error importing Google Generative AI: {e}"
        return False, error_msg, None
    else:
        return True, None, genai_module


# Initialize on module import
GEMINI_AVAILABLE, GEMINI_ERROR, genai = _try_import_gemini()

if not GEMINI_AVAILABLE:
    logger.warning(f"Gemini AI not available: {GEMINI_ERROR}")
else:
    logger.info("Gemini AI package loaded successfully")


def get_gemini_client():
    """Get Gemini client if available, otherwise None."""
    if not GEMINI_AVAILABLE:
        return None
    return genai


def is_gemini_available() -> bool:
    """Check if Gemini AI is available for use."""
    return GEMINI_AVAILABLE


def get_gemini_error() -> str | None:
    """Get the error message if Gemini is not available."""
    return GEMINI_ERROR


def require_gemini():
    """Raise an exception if Gemini is not available."""
    if not GEMINI_AVAILABLE:
        raise ImportError(f"Google Generative AI not available: {GEMINI_ERROR}")
    return genai
