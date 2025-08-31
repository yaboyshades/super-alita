# Add to your existing imports
from src.plugins.telemetry_pipeline import TelemetryPipelinePlugin


# In your application factory
def create_app():
    # ... existing code ...

    # Register telemetry pipeline plugin
    telemetry_plugin = TelemetryPipelinePlugin()
    app.plugin_manager.register(telemetry_plugin)

    # ... rest of your app setup ...
