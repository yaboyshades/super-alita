"""Settings for ACP server."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    require_auth: bool = False
    api_key: str = "test-key"
    log_level: str = "INFO"


settings = Settings(
    host=os.getenv("ACP_HOST", "0.0.0.0"),
    port=int(os.getenv("ACP_PORT", "8000")),
    require_auth=os.getenv("ACP_REQUIRE_AUTH", "false").lower() == "true",
    api_key=os.getenv("ACP_API_KEY", "test-key"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
)
