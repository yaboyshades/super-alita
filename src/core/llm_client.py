# src/core/llm_client.py
"""
Gemini-only LLM client for Super Alita.
Clean implementation without OpenAI dependencies.
"""

import asyncio
import logging
import os

# Gemini SDK (official google-generativeai)
import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """Raised when LLM service is unavailable."""

    pass


class LLMApiError(Exception):
    """Raised when LLM API returns an error."""

    pass


# Configuration
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Configure Gemini once at module level
if not _GEMINI_KEY:
    logger.warning("GEMINI_API_KEY/GOOGLE_API_KEY is not set - LLM features disabled")
    _gemini_model = None
else:
    genai.configure(api_key=_GEMINI_KEY)
    _gemini_model = genai.GenerativeModel(
        model_name=_GEMINI_MODEL,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        },
    )


async def _call_gemini(prompt: str, *, timeout: float = 20.0) -> str:
    """
    Async wrapper around google-generativeai's synchronous generate_content().
    Runs in a worker thread to avoid blocking the event loop.
    """
    if not _gemini_model:
        raise LLMUnavailable("Gemini API key not configured")

    try:

        def _sync_call() -> str:
            resp = _gemini_model.generate_content(prompt)
            # Handle text vs. parts gracefully
            if hasattr(resp, "text") and resp.text:
                return resp.text
            # Fallback: join parts if present
            parts = getattr(resp, "candidates", None) or []
            if parts:
                try:
                    return parts[0].content.parts[0].text
                except Exception:
                    pass
            return ""

        return await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=timeout)
    except TimeoutError as e:
        raise LLMUnavailable(f"Gemini call timed out after {timeout}s") from e
    except Exception as e:
        # Wrap any SDK errors
        raise LLMApiError(f"Gemini API error: {e}") from e


async def generate(
    prompt: str, *, timeout: float = 20.0, retries: int = 3, backoff: float = 0.8
) -> str:
    """
    Public entrypoint for LLM generation. Gemini-only with retry logic.
    """
    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            logger.debug(
                "LLM call start: attempt=%d timeout=%.1fs (Gemini %s)",
                attempt + 1,
                timeout,
                _GEMINI_MODEL,
            )
            text = await _call_gemini(prompt, timeout=timeout)
            logger.debug("LLM call success: %d chars returned", len(text))
            return text
        except (LLMUnavailable, LLMApiError) as e:
            last_err = e
            if attempt < retries - 1:
                delay = backoff * (2**attempt)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt + 1,
                    retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Gemini call failed after %d attempts: %s", retries, e)
                break

    # If we get here, all retries failed
    raise LLMUnavailable(f"All {retries} LLM attempts failed. Last error: {last_err}")


# Legacy compatibility
async def _call_gemini_async(prompt: str) -> str:
    """Legacy compatibility method for existing code."""
    return await generate(prompt)
