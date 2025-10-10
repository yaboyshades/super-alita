import types
from pathlib import Path

import pytest

from src.consciousness import (
    ComponentOrchestrator,
    ConsciousnessOrchestrator,
    EmergencyProtocols,
    InfrastructureBootstrap,
    load_config,
)
from src.consciousness.orchestrator import ConsciousnessDashboard

CONFIG_PATH = Path("configs/unified_consciousness.yaml")
ADVANCED_CONFIG_PATH = Path("configs/unified_consciousness_advanced.yaml")


@pytest.mark.asyncio
async def test_loads_consciousness_configuration():
    cfg = load_config(CONFIG_PATH)
    assert cfg.bootstrap_timeout_seconds == 30
    assert "codex" in cfg.components
    assert (
        pytest.approx(cfg.consciousness_emergence_threshold, rel=1e-3) == 0.7
    )


@pytest.mark.asyncio
async def test_infrastructure_bootstrap_creates_event_bus():
    cfg = load_config(CONFIG_PATH)
    infra = await InfrastructureBootstrap(cfg).initialize()
    assert infra.eventbus is not None
    assert infra.observability is not None


@pytest.mark.asyncio
async def test_consciousness_emergence_pipeline():
    cfg = load_config(CONFIG_PATH)
    infra = await InfrastructureBootstrap(cfg).initialize()
    components = await ComponentOrchestrator(infra, cfg).spawn_all()
    orchestrator = ConsciousnessOrchestrator(infra, components)

    consciousness = await orchestrator.emerge()
    await consciousness.achieve_operational_coherence()
    assert consciousness.current_score >= cfg.consciousness_emergence_threshold

    dashboard = ConsciousnessDashboard(consciousness)
    status = await dashboard.render_status()
    assert "consciousness_score" in status


@pytest.mark.asyncio
async def test_emergency_protocols_actions(monkeypatch):
    cfg = load_config(CONFIG_PATH)
    infra = await InfrastructureBootstrap(cfg).initialize()
    components = await ComponentOrchestrator(infra, cfg).spawn_all()
    orchestrator = ConsciousnessOrchestrator(infra, components)
    consciousness = await orchestrator.emerge()

    captured = []

    async def publish_raw_stub(
        self, event_type: str, source: str, payload: dict | None = None
    ):
        captured.append((event_type, source, payload or {}))

    monkeypatch.setattr(
        infra.eventbus,
        "publish_raw",
        types.MethodType(publish_raw_stub, infra.eventbus),
    )

    protocols = EmergencyProtocols(consciousness)
    await protocols.handle_consciousness_degradation(0.45)
    assert any(evt[0] == "attempt_recovery" for evt in captured)


@pytest.mark.asyncio
async def test_advanced_configuration_loads():
    cfg = load_config(ADVANCED_CONFIG_PATH)
    assert cfg.advanced_capabilities is not None
    assert cfg.advanced_capabilities.any_enabled()
    assert cfg.advanced_capabilities.living_architecture_engine.enabled


@pytest.mark.asyncio
async def test_advanced_capabilities_bootstrap_pipeline(monkeypatch):
    cfg = load_config(ADVANCED_CONFIG_PATH)
    infra = await InfrastructureBootstrap(cfg).initialize()
    assert infra.advanced is not None
    assert infra.advanced.has_capabilities()

    components = await ComponentOrchestrator(infra, cfg).spawn_all()
    orchestrator = ConsciousnessOrchestrator(infra, components)
    consciousness = await orchestrator.emerge()
    await consciousness.achieve_operational_coherence()

    assert consciousness.current_score >= cfg.consciousness_emergence_threshold
    assert consciousness.is_running()
    consciousness.request_shutdown()
