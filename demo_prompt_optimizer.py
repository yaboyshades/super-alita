#!/usr/bin/env python3
"""
Demo script showcasing the Super Alita Prompt Optimizer and Amplifier

This script demonstrates the various optimization strategies and capabilities
of the prompt optimizer system.
"""

from src.prompt.optimizer import (
    OptimizationStrategy,
    PromptOptimizer,
    analyze_user_prompt,
    optimize_user_prompt,
)


def demo_prompt_analysis():
    """Demonstrate prompt analysis capabilities."""
    print("🔍 PROMPT ANALYSIS DEMO")
    print("=" * 50)

    test_prompts = [
        "help",
        "write python function",
        "How does machine learning work?",
        "Create a REST API with authentication and database",
        "Debug this JavaScript error: TypeError undefined",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Original: {prompt}")
        analysis = analyze_user_prompt(prompt)
        print(f"   Type: {analysis.prompt_type.value}")
        print(f"   Complexity: {analysis.complexity_score:.2f}")
        print(f"   Clarity: {analysis.clarity_score:.2f}")
        print(f"   Completeness: {analysis.completeness_score:.2f}")
        if analysis.detected_entities:
            print(f"   Entities: {', '.join(analysis.detected_entities)}")
        if analysis.suggested_enhancements:
            print(f"   Suggestions: {', '.join(analysis.suggested_enhancements)}")


def demo_optimization_strategies():
    """Demonstrate different optimization strategies."""
    print("\n\n🚀 OPTIMIZATION STRATEGIES DEMO")
    print("=" * 50)

    prompt = "write code"
    optimizer = PromptOptimizer()

    strategies = [
        OptimizationStrategy.MINIMAL,
        OptimizationStrategy.STANDARD,
        OptimizationStrategy.AGGRESSIVE,
        OptimizationStrategy.CONTEXT_RICH,
    ]

    context = {
        "available_tools": ["file_reader", "code_analyzer", "debugger"],
        "current_project": "web application development",
        "session_context": "user is working on authentication system"
    }

    for strategy in strategies:
        print(f"\n📋 {strategy.value.upper()} Strategy:")
        result = optimizer.optimize(prompt, strategy=strategy, context=context)
        print(f"   Original: {result.original_prompt}")
        optimized = result.optimized_prompt
        if len(optimized) > 80:
            optimized = optimized[:77] + "..."
        print(f"   Optimized: {optimized}")
        print(f"   Enhancements: {', '.join(result.enhancements_applied)}")


def demo_bypass_functionality():
    """Demonstrate bypass functionality."""
    print("\n\n🛡️ BYPASS FUNCTIONALITY DEMO")
    print("=" * 50)

    bypass_prompts = [
        "noopt:raw message without optimization",
        "raw:keep this exactly as is",
        "literal:don't change this prompt",
    ]

    for prompt in bypass_prompts:
        print(f"\nInput: {prompt}")
        # Simulate the amplifier behavior
        if prompt.startswith(("noopt:", "raw:", "literal:")):
            prefix = prompt.split(":")[0]
            clean = prompt[len(prefix)+1:]
            print(f"Output: {clean}")
            print(f"Status: Bypassed ({prefix})")
        else:
            optimized = optimize_user_prompt(prompt)
            print(f"Output: {optimized}")
            print("Status: Optimized")


def demo_context_awareness():
    """Demonstrate context-aware optimization."""
    print("\n\n🧠 CONTEXT-AWARE OPTIMIZATION DEMO")
    print("=" * 50)

    prompt = "help me fix this bug"

    contexts = [
        {},
        {"available_tools": ["debugger"]},
        {"current_project": "Python web app", "available_tools": ["linter", "tester"]},
        {
            "current_project": "React application",
            "available_tools": ["debugger", "profiler", "network_analyzer"],
            "session_context": "user reported performance issues"
        }
    ]

    optimizer = PromptOptimizer()

    for i, context in enumerate(contexts, 1):
        print(f"\n{i}. Context: {context if context else 'None'}")
        result = optimizer.optimize(prompt, context=context)
        optimized = result.optimized_prompt
        if len(optimized) > 100:
            optimized = optimized[:97] + "..."
        print(f"   Result: {optimized}")


def demo_performance():
    """Demonstrate performance characteristics."""
    print("\n\n⚡ PERFORMANCE DEMO")
    print("=" * 50)

    import time

    # Test different prompt sizes
    prompts = [
        "help",
        "write a python function to sort numbers",
        "create a comprehensive web application with user authentication, "
        "database integration, API endpoints, frontend interface, and deployment pipeline",
        "implement a machine learning model " * 20,  # Very long prompt
    ]

    optimizer = PromptOptimizer()

    for i, prompt in enumerate(prompts, 1):
        start_time = time.time()
        result = optimizer.optimize(prompt)
        end_time = time.time()

        print(f"\n{i}. Length: {len(prompt)} chars")
        print(f"   Processing time: {(end_time - start_time) * 1000:.2f}ms")
        print(f"   Strategy used: {result.strategy_used.value}")
        print(f"   Expansion ratio: {len(result.optimized_prompt) / len(prompt):.2f}x")


if __name__ == "__main__":
    print("🎯 SUPER ALITA PROMPT OPTIMIZER & AMPLIFIER DEMO")
    print("=" * 60)

    try:
        demo_prompt_analysis()
        demo_optimization_strategies()
        demo_bypass_functionality()
        demo_context_awareness()
        demo_performance()

        print("\n\n✅ Demo completed successfully!")
        print("\nThe Super Alita Prompt Optimizer provides:")
        print("• Intelligent prompt analysis and classification")
        print("• Multiple optimization strategies for different needs")
        print("• Context-aware enhancement")
        print("• Bypass mechanisms for raw prompts")
        print("• High performance with caching")
        print("• Integration with existing message amplifier")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
