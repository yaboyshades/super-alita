"""Environment configuration utilities."""

import os
from pathlib import Path

def ensure_env_loaded(silent: bool = False):
    """Load .env file if it exists."""
    env_file = Path(".env")
    
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        if key and key not in os.environ:
                            os.environ[key] = value
            
            if not silent:
                print(f"✅ Loaded environment from {env_file}")
                
        except Exception as e:
            if not silent:
                print(f"⚠️  Failed to load .env: {e}")
    else:
        if not silent:
            print("ℹ️  No .env file found")

def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    return os.getenv(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in {'true', '1', 'yes', 'on'}