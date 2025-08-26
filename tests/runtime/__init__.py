from reug_runtime.config import SETTINGS

API_PREFIX = SETTINGS.api_prefix.rstrip("/")


def prefix_path(path: str) -> str:
    return f"{API_PREFIX}{path}"