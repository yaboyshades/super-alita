#!/usr/bin/env python3
"""Test Alita Research Paper Implementation with Paper2Code"""

import asyncio

from src.core.plugin_registry import register_plugin
from src.pipelines.autogen_pipeline import autogen_any
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin


async def test_alita_paper():
    """Test Paper2Code with the actual Alita research paper"""

    # Setup plugin registry
    plugin = NativeDeepCodePlugin()
    await plugin.start()
    register_plugin("native_deepcode", plugin)
    print("Plugin registered: native_deepcode")

    # Alita research paper implementation request
    alita_request = """
    Implement the Alita neural architecture from the research paper 'Alita: A Large-scale
    Information-seeking, Conversational, and Adversarial AI Assistant'.

    Key architectural components to implement:

    1. **Multi-Modal Fusion Architecture**:
       - Cross-attention mechanisms for text, image, and structured data integration
       - Adaptive gating for dynamic modality weighting
       - Multi-scale feature extraction and alignment

    2. **Conversational Memory System**:
       - Episodic memory with attention-based retrieval
       - Working memory for multi-turn context management
       - Long-term knowledge integration and updating

    3. **Information-Seeking Engine**:
       - Query understanding and decomposition
       - Evidence retrieval and ranking mechanisms
       - Source credibility assessment and fact verification

    4. **Adversarial Training Framework**:
       - Discriminator networks for response quality assessment
       - Adversarial loss functions for robust training
       - Safety alignment and bias mitigation components

    5. **Hierarchical Attention Mechanisms**:
       - Document-level attention for long-context processing
       - Sentence-level attention for fine-grained understanding
       - Token-level attention for precise information extraction

    6. **Neural Symbolic Integration**:
       - Symbolic reasoning components
       - Neural-symbolic interface layers
       - Logic-guided attention and inference

    The implementation should include:
    - PyTorch neural network modules for each component
    - Attention mechanisms with multi-head and cross-modal variants
    - Memory management systems with efficient storage/retrieval
    - Adversarial training loops and loss functions
    - Comprehensive testing and documentation

    This represents a state-of-the-art conversational AI architecture combining
    neural and symbolic approaches for robust information-seeking capabilities.
    """

    print("\nTesting Alita research paper implementation...")
    print(f"Request: {alita_request[:150]}...")

    # Run the autogen pipeline
    print("\nRunning autogen pipeline for Alita architecture...")
    result = await autogen_any(alita_request)
    print(f"\nPipeline result: {result}")

    # Check what files were generated
    if result.get("status") == "complete" and result.get("applied"):
        print("\n🎉 Alita implementation generated successfully!")
        for applied_item in result.get("applied", []):
            if isinstance(applied_item, dict) and "paths" in applied_item:
                print(f"\nGenerated files for {applied_item['kind']}:")
                for file_path in applied_item["paths"]:
                    print(f"  🧠 {file_path}")

                # Show some content if it's the main implementation
                main_files = [
                    p
                    for p in applied_item["paths"]
                    if "src/abilities" in p and p.endswith(".py")
                ]
                if main_files:
                    try:
                        from pathlib import Path

                        main_file = Path(main_files[0])
                        if main_file.exists():
                            content = main_file.read_text()
                            # Look for key Alita architecture components
                            components = [
                                "MultiModalFusion",
                                "ConversationalMemory",
                                "InformationSeeking",
                                "AdversarialTraining",
                                "HierarchicalAttention",
                                "NeuralSymbolic",
                            ]
                            found_components = [
                                c
                                for c in components
                                if c in content
                                or c.lower().replace("_", "")
                                in content.lower()
                            ]
                            if found_components:
                                print(
                                    f"\n✅ Found Alita components: {found_components}"
                                )
                            else:
                                print(
                                    "\n⚠️ Checking for neural architecture patterns..."
                                )
                                patterns = [
                                    "class",
                                    "attention",
                                    "memory",
                                    "fusion",
                                ]
                                found_patterns = [
                                    p for p in patterns if p in content.lower()
                                ]
                                print(f"Found patterns: {found_patterns}")
                    except Exception as e:
                        print(f"Could not analyze main file: {e}")
    else:
        print("\n❌ Alita implementation failed")
        if "failed" in result:
            print(f"Failed capabilities: {result['failed']}")
        if "applied" in result and not result["applied"]:
            print("No capabilities were successfully applied")

    await plugin.stop()
    print("\nAlita paper test completed!")


if __name__ == "__main__":
    asyncio.run(test_alita_paper())
