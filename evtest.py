import asyncio

from src.orchestration.unified_orchestrator import UnifiedOrchestrator, UnifiedRunConfig
from tests.test_orchestrator_integration import MockAbilityRegistry, MockEventBus


async def main():
    orch = UnifiedOrchestrator(MockAbilityRegistry(), MockEventBus())
    cfg = UnifiedRunConfig(
        prompt='Build user authentication system',
        run_id='x',
        enable_specification=True,
        enable_planning=True,
        enable_tasks=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )
    evs=[]
    async for ev in orch.run_stream(cfg):
        evs.append(ev)
    print('events', len(evs))
    print([e.get('type')+':'+str(e.get('stage')) for e in evs if 'Stage' in e.get('type','')])
    print('spec_val', [e for e in evs if e.get('stage')=='specification_validation'])

asyncio.run(main())
