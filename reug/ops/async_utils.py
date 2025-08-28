import asyncio
import contextlib


async def cancel_and_await(task):
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
