# Add to your existing FastAPI app

from src.plugins.telemetry_pipeline.orchestrator import TelemetryPipelineOrchestrator


@app.post("/v1/telemetry/process")
async def process_telemetry(request: dict):
    """Process telemetry data through the pipeline"""
    
    orchestrator = TelemetryPipelineOrchestrator(
        app.state.ability_registry,
        app.state.llm_provider
    )
    
    prompt = await orchestrator.process_telemetry(
        task=request.get("task"),
        telemetry_items=request.get("telemetry", []),
        constraints=request.get("constraints", []),
        token_budget=request.get("token_budget", 2000)
    )
    
    return {"prompt": prompt}