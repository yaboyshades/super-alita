#!/usr/bin/env python3
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "cortex.proxy.copilot_middleware:app", host="0.0.0.0", port=8000, reload=True
    )
