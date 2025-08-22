import logging
from typing import Any

from src.core.auto_tool_pipeline import AutoToolPipeline
from src.core.plugin_interface import PluginInterface
from src.core.sandbox_runner import SandboxRunner

logger = logging.getLogger(__name__)


class AutoToolsPlugin(PluginInterface):
    """Boot-time plugin that registers the AutoToolPipeline for dynamic tool generation."""

    def __init__(self):
        super().__init__()
        self.pipeline = None

    @property
    def name(self) -> str:
        return "auto_tools"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the auto tools plugin."""
        await super().setup(event_bus, store, config)

        # Check if we have Gemini client available for LLM generation
        gemini_api_key = config.get("gemini_api_key")
        if not gemini_api_key:
            logger.warning(
                "No Gemini API key configured - auto-tools will not function"
            )
            return

        # Create LLM client (reuse from conversation plugin pattern)
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_api_key)

            class SimpleLLMClient:
                def __init__(self):
                    self.model = genai.GenerativeModel("gemini-1.5-flash")

                async def generate(self, prompt: str, max_tokens: int = 400) -> str:
                    response = await self.model.generate_content_async(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens, temperature=0.7
                        ),
                    )
                    return response.text

            llm_client = SimpleLLMClient()
            logger.info("✅ Gemini LLM client initialized for auto-tools")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return

        # Create sandbox runner
        sandbox = SandboxRunner()

        # Create and setup pipeline
        self.pipeline = AutoToolPipeline(store, llm_client, sandbox)
        logger.info("🔧 AutoToolPipeline created and configured")

    async def start(self) -> None:
        """Start the auto tools plugin."""
        await super().start()

        if self.pipeline:
            await self.pipeline.start(self.event_bus)
            logger.info(
                "🚀 AutoToolsPlugin started - ready for dynamic tool generation"
            )
        else:
            logger.warning(
                "AutoToolsPlugin started but pipeline not available (missing API key?)"
            )

    async def shutdown(self) -> None:
        """Shutdown the auto tools plugin."""
        logger.info("🛑 AutoToolsPlugin shutdown complete")
