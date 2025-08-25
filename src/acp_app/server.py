"""
ACP Server with multiple agents including search integration.
"""
import asyncio
import logging

from acp_sdk import Server
from .agents import EchoAgent, ClassifyAgent, RouterAgent, SearchAgent
from .middleware import auth_middleware, logging_middleware
from .settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger(__name__)


async def main():
    """Main server entry point."""
    server = Server()

    server.use(logging_middleware)
    if settings.require_auth:
        server.use(auth_middleware)

    server.register_agent("echo", EchoAgent())
    server.register_agent("classify", ClassifyAgent())
    server.register_agent("router", RouterAgent())
    server.register_agent("search", SearchAgent())

    logger.info(f"Starting ACP server on {settings.host}:{settings.port}")
    await server.start(host=settings.host, port=settings.port)

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Server task cancelled, stopping...")
        await server.stop()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
