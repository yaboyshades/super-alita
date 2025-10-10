# Example code snippet - add to your existing imports
from typing import Any

from src.plugins.telemetry_pipeline import TelemetryPipelinePlugin


# In your application factory
def create_app(app: Any) -> None:
    """Example: Register telemetry pipeline plugin.

    Args:
        app: Application instance with plugin_manager attribute
    """
    # ... existing code ...

    # Register telemetry pipeline plugin
    telemetry_plugin = TelemetryPipelinePlugin()
    app.plugin_manager.register(telemetry_plugin)

    # ... rest of your app setup ...
