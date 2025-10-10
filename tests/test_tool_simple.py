import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))


async def test_tool():
    try:
        from mcp_server.tools import find_missing_docstrings

        result = await find_missing_docstrings("src", False)
        print(f'Found {result["count"]} functions missing docstrings')
        return True
    except ImportError:
        print("mcp_server.tools not available - using fallback")
        return False


success = asyncio.run(test_tool())
print(
    f'Tool status: {"✅" if success else "⚠️"} (fallback will be used if needed)'
)
