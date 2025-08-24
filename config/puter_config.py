"""
Configuration for Puter plugin integration.
"""

import os

PUTER_CONFIG = {
    "base_url": "https://puter.com",
    "api_key": None,
    "timeout": 30,
    "max_retries": 3,
    "auto_create_dirs": True,
    "default_working_directory": "/",
    "log_level": "INFO",
}

if os.getenv("PUTER_API_KEY"):
    PUTER_CONFIG["api_key"] = os.getenv("PUTER_API_KEY")

if os.getenv("PUTER_BASE_URL"):
    PUTER_CONFIG["base_url"] = os.getenv("PUTER_BASE_URL")
