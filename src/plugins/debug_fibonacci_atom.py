def debug_fibonacci(**kwargs):
    """
    Test fibonacci implementation for debugging

    Auto-generated tool that publishes results via event bus.
    """
    try:
        # Extract standard parameters
        event_bus = kwargs.get("event_bus")
        tool_call_id = kwargs.get("tool_call_id", "")
        session_id = kwargs.get("session_id", "")
        conversation_id = kwargs.get("conversation_id", "")

        # Remove non-tool parameters
        params = {
            k: v
            for k, v in kwargs.items()
            if k not in ["event_bus", "tool_call_id", "session_id", "conversation_id"]
        }

        # --- BEGIN USER LOGIC ---
        # Implementation for: Test fibonacci implementation for debugging
        n = params.get("n", 0)
        value = 0
        if n <= 0:
            value = 0
        elif n == 1:
            value = 1
        else:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            value = a
        result = {"value": value}
        # --- END USER LOGIC ---

        # If event_bus is available, emit result
        if event_bus:
            # Use pre-imported ToolResultEvent from safe globals
            result_event = ToolResultEvent(
                source_plugin="debug_fibonacci",
                tool_call_id=tool_call_id,
                session_id=session_id,
                conversation_id=conversation_id,
                success=True,
                result=result,
            )

            # Emit via event bus (handle both sync and async contexts)
            try:
                if hasattr(event_bus, "publish"):
                    if hasattr(
                        asyncio, "iscoroutinefunction"
                    ) and asyncio.iscoroutinefunction(event_bus.publish):
                        # Async context
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(event_bus.publish(result_event))
                        else:
                            loop.run_until_complete(event_bus.publish(result_event))
                    else:
                        # Sync publish method
                        event_bus.publish(result_event)
            except Exception:
                pass  # Silently fail if event publishing doesn't work

        return result.get("value", result)

    except Exception as e:
        # If event_bus is available, emit error result
        if kwargs.get("event_bus"):
            try:
                # Use pre-imported ToolResultEvent from safe globals
                error_event = ToolResultEvent(
                    source_plugin="debug_fibonacci",
                    tool_call_id=kwargs.get("tool_call_id", ""),
                    session_id=kwargs.get("session_id", ""),
                    conversation_id=kwargs.get("conversation_id", ""),
                    success=False,
                    result={"error": str(e)},
                )

                # Emit error via event bus
                if hasattr(kwargs["event_bus"], "publish"):
                    if hasattr(
                        asyncio, "iscoroutinefunction"
                    ) and asyncio.iscoroutinefunction(kwargs["event_bus"].publish):
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(
                                kwargs["event_bus"].publish(error_event)
                            )
                        else:
                            loop.run_until_complete(
                                kwargs["event_bus"].publish(error_event)
                            )
                    else:
                        kwargs["event_bus"].publish(error_event)
            except:
                pass

        raise e
