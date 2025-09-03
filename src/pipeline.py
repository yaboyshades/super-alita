#!/usr/bin/env python3
"""
Super Alita Prompt Optimization Pipeline

This module implements a prompt optimization and context enhancement pipeline
that leverages Super Alita's capabilities to generate high-quality prompts
for LLM interactions.

Usage:
    python pipeline.py "Your prompt here"
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path to facilitate imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Super Alita specific modules
try:
    from src.context.retriever import fetch_context, get_consensus_methods
    from src.abilities.enhanced_consensus_ability import ConsensusMethod
except ImportError:
    print("Warning: Could not import Super Alita modules. Using fallbacks.")
    ConsensusMethod = None


class PromptPipeline:
    """
    Main prompt optimization and context enhancement pipeline.
    Integrates with Super Alita's consensus and reasoning capabilities.
    """

    def __init__(self) -> None:
        """Initialize the prompt pipeline."""
        self.template_path = PROJECT_ROOT / "src" / "templates" / "templates.json"
        self.load_templates()
        self.base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-oss:20b")

    def load_templates(self) -> None:
        """Load prompt templates from file."""
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                self.templates = json.load(f)
        except Exception as e:
            print(f"Error loading templates: {e}")
            self.templates = {
                "askExpert": "You are an expert {role}. \"{input}\". "
                            "Provide the answer in {format}. Steps:",
                "contextBlock": "=== CONTEXT ===\n• {title}: {snippet}\n=== END ==="
            }

    def optimize_prompt(
        self, user_input: str, role: str = "AI developer",
        format_type: str = "detailed explanation", constraints: str = ""
    ) -> str:
        """
        Apply prompt optimization using templates.

        Args:
            user_input: The raw user query
            role: Expert role to assume
            format_type: Desired output format
            constraints: Additional constraints or guidelines

        Returns:
            Optimized prompt based on template
        """
        template = self.templates.get("askExpert",
            "You are an expert {role}. \"{input}\". Provide the answer in {format}.")
        optimized = template.replace("{role}", role)\
                           .replace("{input}", user_input)\
                           .replace("{format}", format_type)

        if constraints:
            optimized += f"\nConstraints: {constraints}"

        optimized += "\nLet's think step by step..."
        return optimized

    def format_context_block(self, contexts: List[Dict[str, str]]) -> str:
        """
        Format retrieved context into a block for prompt enhancement.

        Args:
            contexts: List of context snippets with titles

        Returns:
            Formatted context block
        """
        template = self.templates.get("contextBlock",
            "=== CONTEXT ===\n• {title}: {snippet}\n=== END ===")
        blocks = []

        for ctx in contexts:
            block = template.replace("{title}", ctx["title"])\
                           .replace("{snippet}", ctx["snippet"])
            blocks.append(block)

        return "\n".join(blocks)

    async def build_enhanced_prompt(
        self, user_input: str,
        role: str = "AI developer",
        format_type: str = "detailed explanation",
        constraints: str = "",
        consensus_method: str = "weighted_vote"
    ) -> str:
        """
        Build a complete enhanced prompt with context and optimization.

        Args:
            user_input: Raw user query
            role: Expert role to assume
            format_type: Desired output format
            constraints: Additional constraints
            consensus_method: DeepConf consensus method

        Returns:
            Complete enhanced prompt ready for LLM
        """
        # 1. Retrieve relevant context
        contexts = await fetch_context(user_input)

        # 2. Format context block
        context_block = self.format_context_block(contexts)

        # 3. Optimize base prompt
        optimized = self.optimize_prompt(user_input, role, format_type, constraints)

        # 4. Add DeepConf consensus information if applicable
        methods = await get_consensus_methods()
        if consensus_method in methods:
            consensus_template = self.templates.get("deepConfConsensus",
                "Using DeepConf consensus with method '{method}'")
            consensus_block = consensus_template.replace("{method}", consensus_method)\
                                              .replace("{num_samples}", "3")\
                                              .replace("{temperature}", "0.7")
            optimized = f"{optimized}\n\n{consensus_block}"

        # 5. Combine everything
        full_prompt = f"{context_block}\n\n{optimized}"
        return full_prompt

    async def invoke_model(self, prompt: str) -> str:
        """
        Invoke LLM with the enhanced prompt.
        Uses Super Alita's consensus mechanism if available, otherwise falls back
        to direct API calls.

        Args:
            prompt: Enhanced prompt to send to LLM

        Returns:
            Model response
        """
        try:
            # Try to use Super Alita's consensus ability
            from src.main import create_app
            app = create_app()
            registry = app.state.ability_registry

            if registry and registry.knows("deepconf_consensus"):
                print("Using Super Alita's DeepConf consensus...")
                result = await registry.execute("deepconf_consensus", {
                    "prompt": prompt,
                    "method": "weighted_vote",
                    "num_samples": 3,
                    "temperature": 0.7
                })
                return result.get("result", "No result from consensus")
            else:
                print("Falling back to direct API call...")
        except Exception as e:
            print(f"Could not use Super Alita consensus: {e}")
            print("Falling back to direct API call...")

        # Fallback: direct API call (using requests to avoid additional dependencies)
        import requests

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            print(f"Using model {self.model} via {self.base_url}...")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 800
                },
                timeout=60  # Extended timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error invoking model: {e}"


async def main() -> None:
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py 'Your prompt here'")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    pipeline = PromptPipeline()

    print("🔍 Optimizing prompt and retrieving context...")
    enhanced_prompt = await pipeline.build_enhanced_prompt(user_input)

    print("\n=== ENHANCED PROMPT ===")
    print(enhanced_prompt)
    print("=== END OF PROMPT ===\n")

    print("🧠 Generating response with Super Alita...")
    response = await pipeline.invoke_model(enhanced_prompt)

    print("\n=== RESPONSE ===")
    print(response)
    print("=== END OF RESPONSE ===")


if __name__ == "__main__":
    asyncio.run(main())
