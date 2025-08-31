#!/usr/bin/env python3
"""Debug Paper2Code capability specifically"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.pipelines.autogen_pipeline import autogen_any
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_paper2code():
    """Test Paper2Code capability generation specifically"""

    # Setup plugin registry
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)
    print("Plugin registered: native_deepcode")

    # Test Paper2Code request
    paper_request = (
        "Implement the Transformer architecture from the "
        "'Attention is All You Need' paper"
    )

    # Debug the need detector response
    from src.policies.need_detector import NeedDetector

    detector = NeedDetector()
    needs = detector.detect(paper_request)
    print(f"\nNeed detector results: {needs}")

    # Run the capability directly through the plugin
    try:
        plugin_result = await plugin.native_deepcode_generate(
            task_kind="paper2code", requirements=paper_request, repo_path="."
        )
        print(f"\nPlugin result: {plugin_result}")

        # Check if gate validation would pass
        from src.contracts.gates.common_gates import COMMON_GATES

        if "paper2code" in COMMON_GATES:
            gate = COMMON_GATES["paper2code"]
            print(f"\nGate pattern: {gate['pattern']}")

            # Check each diff path
            if "diffs" in plugin_result:
                for diff in plugin_result["diffs"]:
                    path = diff.get("path", "")
                    matches = gate["pattern"].search(path)
                    print(f"Path '{path}' matches gate: {bool(matches)}")

    except Exception as e:
        print(f"Plugin error: {e}")
        import traceback

        traceback.print_exc()

    # Test full pipeline
    print("\nRunning full pipeline...")
    result = await autogen_any(paper_request)
    print(f"Pipeline result: {result}")

    await plugin.stop()


if __name__ == "__main__":
    asyncio.run(test_paper2code())
