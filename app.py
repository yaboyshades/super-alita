from fastapi import FastAPI, Response

try:
    from reug_runtime.router import router as agent_router
    from reug_runtime.router_tools import router as tools_router
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "src"))
    from reug_runtime.router import router as agent_router
    from reug_runtime.router_tools import router as tools_router

try:
    from tests.runtime.fakes import (
        FakeAbilityRegistry as AbilityRegistry,
    )
    from tests.runtime.fakes import (
        FakeEventBus as EventBus,
    )
    from tests.runtime.fakes import (
        FakeKG as KnowledgeGraph,
    )
    from tests.runtime.fakes import (
        FakeLLM as LLMClient,
    )
except Exception:

    class EventBus:
        async def emit(self, event): ...
    class AbilityRegistry:
        def get_available_tools_schema(self):
            return []

        def validate_args(self, tool_name, args):
            return True

        async def execute(self, tool_name, args):
            return {}

    class KnowledgeGraph:
        async def retrieve_relevant_context(self, user_message):
            return ""

        async def get_goal_for_session(self, session_id):
            return {"id": "goal", "description": "Assist the user"}

        async def create_atom(self, atom_type, content):
            return {"id": "atom_0"}

        async def create_bond(self, bond_type, source_atom_id, target_atom_id): ...
    class LLMClient:
        async def stream_chat(self, messages, timeout):
            yield {"content": "Hello from mock LLM. "}


def create_app() -> FastAPI:
    app = FastAPI(title="REUG Runtime", version="0.1.0")

    @app.get("/healthz")
    async def health_check():
        return Response(status_code=200)

    app.include_router(agent_router)
    app.include_router(tools_router)

    app.state.event_bus = EventBus()
    app.state.ability_registry = AbilityRegistry()
    app.state.kg = KnowledgeGraph()
    app.state.llm_model = LLMClient()

    return app


app = create_app()
