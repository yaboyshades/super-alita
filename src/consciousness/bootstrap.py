from __future__ import annotations

import logging
from pathlib import Path

from .components import ComponentOrchestrator
from .config import BootstrapConfig, load_config
from .infrastructure import InfrastructureBootstrap, record_boot_events
from .orchestrator import ConsciousnessOrchestrator, UnifiedConsciousness


async def bootstrap_consciousness(
    *,
    config_path: str | Path | None = None,
    log_level: str = "INFO",
    enable_all_subsystems: bool = False,  # kept for CLI parity, currently informational only
) -> UnifiedConsciousness:
    """Bring the unified consciousness system online."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO)
    )
    cfg = _prepare_config(config_path)

    infra = await InfrastructureBootstrap(cfg).initialize()
    components = await ComponentOrchestrator(infra, cfg).spawn_all()
    await record_boot_events(infra.eventbus, components=components.keys())

    orchestrator = ConsciousnessOrchestrator(infra, components)
    consciousness = await orchestrator.emerge()
    await consciousness.achieve_operational_coherence()
    return consciousness


def _prepare_config(path: str | Path | None) -> BootstrapConfig:
    location = Path(path or "configs/unified_consciousness.yaml")
    if not location.exists():
        raise FileNotFoundError(
            f"Unified consciousness configuration missing: {location}"
        )
    return load_config(location)
