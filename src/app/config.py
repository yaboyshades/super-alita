"""Application configuration with environment integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

@dataclass
class DatabaseConfig:
    """Database configuration."""
    redis_url: str = "redis://localhost:6379"
    chromadb_path: str = "./data/chromadb"
    event_backup_path: str = "./data/events.jsonl"
    
    @classmethod
    def from_env(cls) -> DatabaseConfig:
        return cls(
            redis_url=os.getenv("ALITA_REDIS_URL", cls.redis_url),
            chromadb_path=os.getenv("CHROMADB_PATH", cls.chromadb_path),
            event_backup_path=os.getenv("EVENT_BACKUP_PATH", cls.event_backup_path)
        )

@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str = "ollama:gpt-oss:20b"
    ollama_host: str = "http://127.0.0.1:11434"
    timeout: int = 60
    temperature: float = 0.7
    max_tokens: int = 2048
    
    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            model=os.getenv("LLM_MODEL", cls.model),
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host),
            timeout=int(os.getenv("LLM_TIMEOUT", str(cls.timeout))),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.temperature))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(cls.max_tokens)))
        )

@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    require_api_key: bool = False
    api_keys: List[str] = field(default_factory=list)
    admin_key: str = ""
    rate_limit_enabled: bool = False
    rate_limit: int = 60
    rate_window: int = 60
    abilities_admin_only: bool = False
    ability_whitelist: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> SecurityConfig:
        # Parse API keys
        api_keys = []
        if single_key := os.getenv("ALITA_API_KEY", "").strip():
            api_keys.append(single_key)
        if multi_keys := os.getenv("ALITA_API_KEYS", "").strip():
            api_keys.extend([k.strip() for k in multi_keys.split(",") if k.strip()])
        
        # Parse whitelist
        whitelist_str = os.getenv("ALITA_ABILITY_WHITELIST", "")
        whitelist = [k.strip() for k in whitelist_str.split(",") if k.strip()]
        
        return cls(
            require_api_key=os.getenv("ALITA_REQUIRE_API_KEY", "false").lower() in {"1", "true", "yes", "on"},
            api_keys=api_keys,
            admin_key=os.getenv("ALITA_ADMIN_KEY", "").strip(),
            rate_limit_enabled=os.getenv("ALITA_RATE_LIMIT_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            rate_limit=int(os.getenv("ALITA_RATE_LIMIT", "60")),
            rate_window=int(os.getenv("ALITA_RATE_WINDOW", "60")),
            abilities_admin_only=os.getenv("ALITA_ABILITIES_ADMIN_ONLY", "false").lower() in {"1", "true", "yes", "on"},
            ability_whitelist=whitelist
        )

@dataclass
class CORSConfig:
    """CORS configuration."""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    
    @classmethod
    def from_env(cls) -> CORSConfig:
        origins_str = os.getenv("CORS_ALLOW_ORIGINS", "*")
        origins = [o.strip() for o in origins_str.split(",") if o.strip()]
        
        return cls(
            allow_origins=origins,
            allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() in {"1", "true", "yes"}
        )

@dataclass
class StaticConfig:
    """Static file serving configuration."""
    enabled: bool = True
    directory: str = "static"
    mount_path: str = "/static"
    serve_index: bool = True
    
    @classmethod
    def from_env(cls) -> StaticConfig:
        return cls(
            enabled=os.getenv("STATIC_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            directory=os.getenv("STATIC_DIRECTORY", cls.directory),
            mount_path=os.getenv("STATIC_MOUNT_PATH", cls.mount_path),
            serve_index=os.getenv("STATIC_SERVE_INDEX", "true").lower() in {"1", "true", "yes"}
        )

@dataclass
class FeatureFlags:
    """Feature flag configuration."""
    github_demo: bool = False
    perplexica_demo: bool = False
    autogen_demo: bool = False
    enhanced_consensus: bool = True
    z3_verifier: bool = False
    mcp_broadcast: bool = False
    research_mode: bool = False
    dev_mode: bool = False
    simple_chat_stream: bool = False
    deepcode_integration: bool = False
    
    @classmethod
    def from_env(cls) -> FeatureFlags:
        return cls(
            github_demo=os.getenv("ENABLE_GITHUB_DEMO", "false").lower() in {"1", "true", "yes", "on"},
            perplexica_demo=os.getenv("ENABLE_PERPLEXICA_DEMO", "false").lower() in {"1", "true", "yes", "on"},
            autogen_demo=os.getenv("ENABLE_AUTOGEN_DEMO", "false").lower() in {"1", "true", "yes", "on"},
            enhanced_consensus=os.getenv("ENABLE_ENHANCED_CONSENSUS", "true").lower() in {"1", "true", "yes", "on"},
            z3_verifier=os.getenv("ALITA_ENABLE_Z3", "false").lower() in {"1", "true", "yes", "on"},
            mcp_broadcast=os.getenv("MCP_BROADCAST_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            research_mode=os.getenv("RESEARCH_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            dev_mode=os.getenv("SUPER_ALITA_DEV", "false").lower() in {"1", "true", "yes", "on"},
            simple_chat_stream=os.getenv("ALITA_SIMPLE_CHAT_STREAM", "false").lower() in {"1", "true", "yes", "on"},
            deepcode_integration=os.getenv("DEEPCODE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        )

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"  # json or text
    directory: str = "./logs"
    file_name: str = "runtime.log"
    
    @classmethod
    def from_env(cls) -> LoggingConfig:
        return cls(
            level=os.getenv("REUG_LOG_LEVEL", cls.level),
            format=os.getenv("LOG_FORMAT", cls.format),
            directory=os.getenv("REUG_LOG_DIR", cls.directory),
            file_name=os.getenv("LOG_FILE_NAME", cls.file_name)
        )

@dataclass 
class DeepCodeConfig:
    """DeepCode integration configuration."""
    enabled: bool = False
    mode: str = "comprehensive"  # comprehensive, optimized
    install_location: str = "./deepcode"
    timeout: int = 300
    api_keys: Dict[str, str] = field(default_factory=dict)
    mcp_config_path: str = "./config/mcp_agent.config.yaml"
    
    @classmethod
    def from_env(cls) -> 'DeepCodeConfig':
        return cls(
            enabled=os.getenv("DEEPCODE_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            mode=os.getenv("DEEPCODE_MODE", "comprehensive"),
            install_location=os.getenv("DEEPCODE_HOME", "./deepcode"),
            timeout=int(os.getenv("DEEPCODE_TIMEOUT", "300")),
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
                "brave": os.getenv("BRAVE_API_KEY", "")
            },
            mcp_config_path=os.getenv("DEEPCODE_MCP_CONFIG", "./config/mcp_agent.config.yaml")
        )

@dataclass
class ApplicationConfig:
    """Main application configuration."""
    profile: str = "production"  # production, development, test
    api_prefix: str = ""
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    security: SecurityConfig = field(default_factory=SecurityConfig.from_env)
    cors: CORSConfig = field(default_factory=CORSConfig.from_env)
    static: StaticConfig = field(default_factory=StaticConfig.from_env)
    features: FeatureFlags = field(default_factory=FeatureFlags.from_env)
    logging: LoggingConfig = field(default_factory=LoggingConfig.from_env)
    deepcode: DeepCodeConfig = field(default_factory=DeepCodeConfig.from_env)
    
    @classmethod
    def from_env(cls) -> 'ApplicationConfig':
        """Load configuration from environment variables."""
        # Load .env file
        from ..core.env import ensure_env_loaded
        ensure_env_loaded(silent=True)
        
        profile = os.getenv("ALITA_PROFILE", "production")
        api_prefix = os.getenv("API_PREFIX", "")
        
        return cls(
            profile=profile,
            api_prefix=api_prefix,
            database=DatabaseConfig.from_env(),
            llm=LLMConfig.from_env(),
            security=SecurityConfig.from_env(),
            cors=CORSConfig.from_env(),
            static=StaticConfig.from_env(),
            features=FeatureFlags.from_env(),
            logging=LoggingConfig.from_env(),
            deepcode=DeepCodeConfig.from_env()
        )
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.profile == "development" or self.features.dev_mode
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.profile == "production" and not self.features.dev_mode

# Factory function - fix the import issue
def create_application(config: 'ApplicationConfig' = None):
    """Create Super Alita application."""
    if config is None:
        config = ApplicationConfig.from_env()
    
    # Import ApplicationFactory here to avoid circular imports
    from .factory import ApplicationFactory
    
    factory = ApplicationFactory(config)
    
    # Handle event loop properly
    import asyncio
    import nest_asyncio
    
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, apply nest_asyncio patch and use run_until_complete
            nest_asyncio.apply()
            return loop.run_until_complete(factory.create_application())
        else:
            # If no loop is running, use asyncio.run
            return asyncio.run(factory.create_application())
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(factory.create_application())