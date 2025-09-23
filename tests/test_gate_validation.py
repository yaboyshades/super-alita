#!/usr/bin/env python3
"""Test gate validation"""

import asyncio

from src.contracts.gates.common_gates import (
    CombinedGate,
    PytestGate,
    RequiredPathsGate,
    SafetyGate,
)
from src.core.plugin_registry import register_plugin
from src.native_deepcode_api import NativeDeepCodeAPI
from src.pipelines.autogen_pipeline import CAPABILITY_TEMPLATES
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test():
    # Setup plugin and API
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)

    api = NativeDeepCodeAPI()
    api.set_plugin(plugin)

    print(f"API configured with plugin: {api.native_plugin}")

    # Test paper2code capability
    tpl = CAPABILITY_TEMPLATES.get("paper2code")
    if not tpl:
        print("❌ No paper2code template found")
        return

    print(f"Template: {tpl}")

    # Generate requirements
    description = "Implement ResNet architecture from Deep Residual Learning for Image Recognition paper"
    req = tpl["requirements"](description)
    print(f"Generated requirements: {req[:100]}...")

    # Make request
    result = await api.deepcode_request(
        task_kind=tpl["task_kind"],
        requirements=req,
        repo_path=".",
        conversation_id="test",
    )
    print(f'Request result: {result.get("status")}')

    # Get latest
    latest = await api.deepcode_latest()
    print(f"Latest keys: {list(latest.keys())}")

    # Check gate validation
    gate = CombinedGate(
        RequiredPathsGate(
            required_paths=tpl["required_paths"],
            required_docs=tpl["required_docs"],
        ),
        SafetyGate(api),
        PytestGate(api),
    )

    ok, info = gate.validate_latest(latest)
    print(f"Gate validation: ok={ok}")
    print(f"Gate info: {info}")

    if not ok:
        print("❌ Gate validation failed")
        if "reasons" in info:
            for reason in info["reasons"]:
                print(f"  - {reason}")
    else:
        print("✅ Gate validation passed!")
        paths = info.get("paths", [])
        print(f"Validated paths: {paths}")

    await plugin.stop()


if __name__ == "__main__":
    asyncio.run(test())
