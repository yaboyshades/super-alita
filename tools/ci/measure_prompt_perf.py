from __future__ import annotations

import json
import os
import time


def main() -> int:
    start = time.perf_counter()
    # Placeholder: simulate a planner prompt run
    time.sleep(0.1)
    duration = time.perf_counter() - start

    # Cost can be integrated with real LLM billing; set 0 for now
    result = {
        "latency_seconds": duration,
        "estimated_cost_usd": 0.0,
        "timestamp": time.time(),
        "details": {"note": "placeholder measurement"},
    }
    print(json.dumps(result))

    # Budget gates from env
    max_latency = float(os.getenv("PROMPT_LATENCY_BUDGET", "2.0"))
    max_cost = float(os.getenv("PROMPT_COST_BUDGET", "0.25"))

    if result["latency_seconds"] > max_latency:
        print(f"Latency {result['latency_seconds']:.3f}s exceeds budget {max_latency}s", flush=True)
        return 2
    if result["estimated_cost_usd"] > max_cost:
        print(f"Cost ${result['estimated_cost_usd']:.3f} exceeds budget ${max_cost}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

