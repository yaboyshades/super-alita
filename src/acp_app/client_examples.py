"""Client examples for ACP server."""

import asyncio
from acp_sdk import Client, Message, MessagePart


async def run_sync_example():
    """Synchronous call example."""
    client = Client(base_url="http://localhost:8000")

    messages = [Message(parts=[MessagePart(text="Hello, ACP!")])]
    response = await client.run_sync("echo", messages)
    print(f"Echo response: {response}")

    messages = [
        Message(
            parts=[
                MessagePart(
                    text="This is a longer text that should be classified as complex."
                )
            ]
        )
    ]
    response = await client.run_sync("classify", messages)
    print(f"Classify response: {response}")

    messages = [
        Message(parts=[MessagePart(text="what is retrieval augmented generation")])
    ]
    response = await client.run_sync("search", messages)
    print(f"Search response: {response}")


async def run_stream_example():
    """Streaming example."""
    client = Client(base_url="http://localhost:8000")

    messages = [Message(parts=[MessagePart(text="Route this message please")])]
    print("Streaming from router:")
    async for msg in client.run("router", messages):
        for part in msg.parts:
            if getattr(part, "text", None):
                print(f"  {part.text}")


async def main():
    """Run all examples."""
    print("=== Sync Examples ===")
    await run_sync_example()

    print("\n=== Stream Examples ===")
    await run_stream_example()


if __name__ == "__main__":
    asyncio.run(main())
