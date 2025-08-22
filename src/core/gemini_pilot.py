"""
Gemini-2.5-Pro Pilot Client with JSON Schema Contract System

This module implements the production-grade cognitive contract system that
transforms Gemini into a structured action planner rather than a chat bot.
"""

import base64
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

# JSON Schema for Gemini responses
PILOT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Concise chain-of-thought for humans & logs.",
        },
        "action": {
            "type": "string",
            "enum": [
                "publish_event",
                "call_tool",
                "query_memory",
                "plan_steps",
                "escalate",
            ],
        },
        "parameters": {
            "type": "object",
            "description": "Key/value bag specific to the chosen action.",
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 300,
            "default": 30,
        },
        "schema_version": {
            "type": "integer",
            "description": "Schema version for compatibility checking.",
        },
    },
    "required": ["reasoning", "action", "parameters"],
    "additionalProperties": False,
}

SELF_HEAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Diagnostic analysis and rationale.",
        },
        "action": {
            "type": "string",
            "enum": ["retry", "restart", "rollback", "patch", "escalate"],
        },
        "parameters": {"type": "object", "description": "Action-specific parameters."},
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the recommended action.",
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 300,
            "default": 30,
        },
    },
    "required": ["reasoning", "action", "parameters", "confidence"],
    "additionalProperties": False,
}


class GeminiPilotClient:
    """
    Production-grade Gemini client for structured cognitive contracts.

    Features:
    - YAML-based prompt templates with variable substitution
    - JSON schema enforcement for structured responses
    - Token budgeting and compression for large contexts
    - Deterministic seeding for regression tests
    - Schema version validation
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-2.5-pro"
        # Contracts are stored at repository_root/prompts/contracts
        self.prompts_dir = (
            Path(__file__).resolve().parent.parent.parent / "prompts" / "contracts"
        )

        # HTTP client with timeouts and retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def load_contract(self, contract_name: str, variables: dict[str, Any]) -> str:
        """
        Load and process a cognitive contract YAML template.

        Args:
            contract_name: Name of the contract file (without .yaml)
            variables: Dictionary of variables for substitution

        Returns:
            Processed contract as string
        """
        contract_path = self.prompts_dir / f"{contract_name}.yaml"

        if not contract_path.exists():
            raise FileNotFoundError(f"Contract not found: {contract_path}")

        # Load YAML template
        with open(contract_path, encoding="utf-8") as f:
            template = f.read()

        # Substitute variables using os.path.expandvars pattern
        processed = template
        for key, value in variables.items():
            # Handle different value types
            if isinstance(value, dict | list):
                # Convert complex objects to YAML
                value_str = yaml.dump(value, default_flow_style=False)
            elif value is None:
                value_str = ""
            else:
                value_str = str(value)

            processed = processed.replace(f"${{{key}}}", value_str)

        return processed

    def _compress_large_context(
        self, context: str, threshold: int = 8000
    ) -> tuple[str, bool]:
        """
        Compress context if it exceeds token threshold.

        Args:
            context: Context string to potentially compress
            threshold: Token threshold for compression

        Returns:
            Tuple of (processed_context, was_compressed)
        """
        # Rough token estimation: ~4 chars per token
        estimated_tokens = len(context) // 4

        if estimated_tokens > threshold:
            # Compress and base64 encode
            compressed = gzip.compress(context.encode("utf-8"))
            encoded = base64.b64encode(compressed).decode("ascii")

            # Create compressed context marker
            compressed_context = f"""
# Large context compressed (original ~{estimated_tokens} tokens)
compressed_data: |
  {encoded}
compression: gzip
"""
            return compressed_context, True

        return context, False

    async def pilot_decide(
        self,
        goal: str,
        agent_state: str,
        last_error: str = "",
        memory_context: Any = None,
        recent_events: Any = None,
        seed: int | None = None,
        experimental: bool = False,
    ) -> dict[str, Any]:
        """
        Get a structured action plan from Gemini Pilot.

        Args:
            goal: Current objective or task
            agent_state: Current state of the agent
            last_error: Last error encountered (if any)
            memory_context: Relevant memories or context
            recent_events: Recent system events
            seed: Deterministic seed for testing
            experimental: Enable experimental features

        Returns:
            Structured action plan dictionary
        """
        # Prepare variables for contract
        variables = {
            "CURRENT_GOAL": goal,
            "AGENT_STATE": agent_state,
            "LAST_ERROR": last_error or "None",
            "MEMORY_CONTEXT": memory_context or [],
            "RECENT_EVENTS": recent_events or [],
        }

        # Load and process contract
        contract = await self.load_contract("pilot_decision", variables)

        # Parse contract to get meta information
        contract_data = yaml.safe_load(contract)
        max_tokens = contract_data.get("meta", {}).get("max_tokens", 2048)
        schema_version = contract_data.get("meta", {}).get("schema_version", 1)

        # Prepare generation config
        generation_config = {
            "temperature": 0.2,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": PILOT_RESPONSE_SCHEMA,
        }

        if seed is not None:
            generation_config["seed"] = seed

        # Prepare request payload
        payload = {
            "systemInstruction": {
                "parts": [{"text": contract_data.get("system_instruction", "")}]
            },
            "contents": [{"role": "user", "parts": [{"text": contract}]}],
            "generationConfig": generation_config,
        }

        # Make API call
        response = await self._call_gemini(payload)

        # Validate schema version
        if response.get("schema_version") != schema_version:
            logger.warning(
                f"Schema version mismatch: expected {schema_version}, got {response.get('schema_version')}"
            )

        # Filter experimental features if not enabled
        if not experimental and response.get("parameters", {}).get("experimental"):
            logger.info("Skipping experimental action (EXPERIMENTAL=0)")
            return await self.pilot_decide(
                goal,
                agent_state,
                last_error,
                memory_context,
                recent_events,
                seed,
                False,
            )

        return response

    async def self_heal_decide(
        self,
        failure_type: str,
        error_details: str,
        component: str,
        system_state: str,
        retry_count: int = 0,
        failure_history: Any = None,
        diagnostic_data: Any = None,
    ) -> dict[str, Any]:
        """
        Get a structured healing action from Gemini for system failures.

        Args:
            failure_type: Type of failure encountered
            error_details: Detailed error information
            component: Component that failed
            system_state: Current system state
            retry_count: Number of retry attempts so far
            failure_history: Similar past failures
            diagnostic_data: Additional diagnostic information

        Returns:
            Structured healing action plan
        """
        variables = {
            "FAILURE_TYPE": failure_type,
            "ERROR_DETAILS": error_details,
            "COMPONENT": component,
            "SYSTEM_STATE": system_state,
            "RETRY_COUNT": retry_count,
            "FAILURE_HISTORY": failure_history or [],
            "DIAGNOSTIC_DATA": diagnostic_data or {},
        }

        contract = await self.load_contract("self_heal_decision", variables)
        contract_data = yaml.safe_load(contract)

        payload = {
            "systemInstruction": {
                "parts": [{"text": contract_data.get("system_instruction", "")}]
            },
            "contents": [{"role": "user", "parts": [{"text": contract}]}],
            "generationConfig": {
                "temperature": 0.1,  # Lower temperature for healing decisions
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
                "responseSchema": SELF_HEAL_RESPONSE_SCHEMA,
            },
        }

        return await self._call_gemini(payload)

    async def _call_gemini(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make the actual HTTP call to Gemini API.

        Args:
            payload: Request payload

        Returns:
            Parsed response dictionary
        """
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()

            # Extract the actual response content
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(content)
            raise ValueError("No valid response from Gemini")

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Gemini API error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Gemini: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Factory function for easy usage
async def create_pilot_client(api_key: str | None = None) -> GeminiPilotClient:
    """Create and return a configured Gemini Pilot client."""
    return GeminiPilotClient(api_key)
