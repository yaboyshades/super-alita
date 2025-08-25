"""
Configuration for Puter plugin integration.
"""

import os

PUTER_CONFIG = {
    "base_url": "https://puter.com",
    "api_key": None,
    "timeout": 30,
    "max_retries": 3,

    # HTTP status codes we will retry on (in addition to network errors)
    "retriable_statuses": [502, 503, 504],
    "auto_create_dirs": True,
    "default_working_directory": "/",
    "log_level": "INFO",
    # Some deployments may not expose /api/health. Allow skipping.
    "skip_healthcheck": False,
    # Optional Worker/HMAC mode (for calling your secured bridge)
    "worker": {
        # If set, plugin will send requests to this base URL instead of base_url
        # (e.g., https://reug-bridge.puter.work)
        "base_url": None,
        # If set, plugin will sign each request body with HMAC SHA-256 and
        # include the signature in this header name.
        "shared_secret": None,
        "hmac_header": "x-reug-sig",
        "enabled": False,
    },

    "auto_create_dirs": True,
    "default_working_directory": "/",
    "log_level": "INFO",
  
}

if os.getenv("PUTER_API_KEY"):
    PUTER_CONFIG["api_key"] = os.getenv("PUTER_API_KEY")

if os.getenv("PUTER_BASE_URL"):
    PUTER_CONFIG["base_url"] = os.getenv("PUTER_BASE_URL")


# Worker/HMAC via env (optional)
if os.getenv("REUG_PUTER_WORKER_BASE"):
    PUTER_CONFIG["worker"]["base_url"] = os.getenv("REUG_PUTER_WORKER_BASE")
if os.getenv("REUG_PUTER_WORKER_SECRET"):
    PUTER_CONFIG["worker"]["shared_secret"] = os.getenv("REUG_PUTER_WORKER_SECRET")
if os.getenv("REUG_PUTER_WORKER_ENABLED"):
    val = os.getenv("REUG_PUTER_WORKER_ENABLED", "").strip().lower()
    PUTER_CONFIG["worker"]["enabled"] = val in {"1", "true", "yes"}

