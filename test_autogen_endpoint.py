#!/usr/bin/env python3
"""
Test the FastAPI autogen endpoint with native DeepCode integration.
"""

import asyncio

import aiohttp


async def test_autogen_endpoint():
    """Test the /autogen/trigger endpoint."""
    print("Testing FastAPI autogen endpoint...")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test the autogen trigger endpoint
            async with session.post(
                "http://localhost:8000/autogen/trigger",
                json={"description": "Create an API client for REST calls"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Autogen endpoint responded: {result}")
                    return result
                else:
                    print(f"✗ HTTP {response.status}: {await response.text()}")
                    return None
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("Note: Make sure the FastAPI server is running on localhost:8000")
            return None


if __name__ == "__main__":
    asyncio.run(test_autogen_endpoint())