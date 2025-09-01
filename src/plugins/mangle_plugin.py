"""
Mangle Integration Plugin for Super Alita

This plugin integrates Google's Mangle deductive database programming language
with Super Alita, providing rule-based logical reasoning capabilities that
complement ChromaDB's vector similarity search.

Mangle is used for:
- Knowledge graph traversal and querying
- Software dependency analysis
- Security vulnerability detection
- Rule-based logical reasoning

The plugin provides both direct rule execution and an ability-based API.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.core.event_bus import EventBus
from src.plugins.plugin_interface import PluginInterface

# Configure logging
logger = logging.getLogger(__name__)


class MangleException(Exception):
    """Exception raised for Mangle execution errors."""
    pass


@dataclass
class MangleRule:
    """Represents a Mangle rule for deductive reasoning."""
    name: str
    body: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ManglePlugin(PluginInterface):
    """
    Plugin for integrating Mangle deductive database programming with Super Alita.

    This plugin provides:
    1. Rule management (create, store, retrieve)
    2. Rule execution and querying
    3. Fact database management
    4. Integration with Super Alita's event system
    """

    def __init__(self):
        """Initialize the Mangle plugin."""
        self.name = "mangle_deductive_reasoning"
        self.description = "Deductive database programming using Google's Mangle"

        # Storage for rules and facts
        self.rules: Dict[str, MangleRule] = {}
        self.facts: Dict[str, List[Dict[str, Any]]] = {}

        # Paths
        self.storage_dir = Path("./data/mangle")
        self.rules_file = self.storage_dir / "rules.json"
        self.facts_file = self.storage_dir / "facts.json"

        # Mangle binary path - will need to be configured properly
        self.mangle_bin = os.environ.get("MANGLE_BIN", "mangle")

        # Event bus will be set during initialization
        self.event_bus = None

        # Create storage directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self, event_bus: EventBus, **kwargs) -> bool:
        """Initialize the plugin with event bus and load persisted data."""
        logger.info(f"Initializing {self.name} plugin")
        self.event_bus = event_bus

        # Load persisted rules and facts
        self._load_rules()
        self._load_facts()

        # Check if Mangle binary is available
        try:
            # Run a simple version check or status command
            result = subprocess.run(
                [self.mangle_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Mangle detected: {result.stdout.strip()}")
            else:
                logger.warning(f"Mangle check returned non-zero: {result.stderr}")
                # We'll continue even if this fails, as we may mock the execution
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Mangle binary not found or error checking: {e}")
            # For development purposes, we'll continue initialization
            # In production, you might want to return False here

        # Subscribe to relevant events
        await event_bus.subscribe("knowledge_fact_added", self.handle_new_fact)
        await event_bus.subscribe("query_knowledge", self.handle_knowledge_query)

        logger.info(f"{self.name} plugin initialized with {len(self.rules)} rules "
                   f"and {sum(len(f) for f in self.facts.values())} facts")
        return True

    async def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process events from the event bus."""
        # This plugin primarily uses specific event handlers rather than this generic method
        return None

    async def handle_new_fact(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle new fact events from the event bus."""
        try:
            if "relation" in event and "data" in event:
                relation = event["relation"]
                data = event["data"]

                # Add the fact to our database
                if relation not in self.facts:
                    self.facts[relation] = []
                self.facts[relation].append(data)

                # Save facts to disk
                self._save_facts()

                logger.debug(f"Added new fact to relation '{relation}': {data}")
                return {"status": "success", "relation": relation}
        except Exception as e:
            logger.error(f"Error processing new fact: {e}")

        return None

    async def handle_knowledge_query(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle knowledge query events."""
        try:
            if "query" in event:
                query = event["query"]
                result = await self.execute_query(query)
                return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Error processing knowledge query: {e}")
            return {"status": "error", "error": str(e)}

        return None

    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a Mangle query against the current fact database."""
        try:
            # Create a temporary file for the program
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".mangle", delete=False) as program_file:
                program_path = program_file.name

                # Write all facts to the program file
                for relation, facts_list in self.facts.items():
                    for fact in facts_list:
                        # Convert facts to Mangle syntax
                        fact_str = self._convert_fact_to_mangle(relation, fact)
                        program_file.write(f"{fact_str}.\n")

                # Add rules to the program file
                for rule in self.rules.values():
                    program_file.write(f"\n{rule.body}\n")

                # Add the query
                program_file.write(f"\n{query}\n")

            # Execute the Mangle program
            try:
                result = subprocess.run(
                    [self.mangle_bin, "run", program_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    # Parse and return the results
                    return self._parse_mangle_output(result.stdout)
                else:
                    logger.error(f"Mangle execution error: {result.stderr}")
                    raise MangleException(f"Execution error: {result.stderr}")
            finally:
                # Clean up temporary file
                os.unlink(program_path)

        except Exception as e:
            logger.error(f"Error executing Mangle query: {e}")
            raise

    def add_rule(self, name: str, body: str, description: str = "", tags: List[str] = None) -> str:
        """Add a new rule to the database."""
        rule_id = name.lower().replace(" ", "_")

        self.rules[rule_id] = MangleRule(
            name=name,
            body=body,
            description=description,
            tags=tags or []
        )

        # Save rules to disk
        self._save_rules()

        logger.info(f"Added new rule: {rule_id}")
        return rule_id

    def get_rule(self, rule_id: str) -> Optional[MangleRule]:
        """Get a rule by its ID."""
        return self.rules.get(rule_id)

    def add_fact(self, relation: str, data: Dict[str, Any]) -> None:
        """Add a fact to the database."""
        if relation not in self.facts:
            self.facts[relation] = []

        self.facts[relation].append(data)
        self._save_facts()

    def get_facts(self, relation: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get facts, optionally filtered by relation."""
        if relation:
            return {relation: self.facts.get(relation, [])}
        return self.facts

    def _load_rules(self) -> None:
        """Load rules from disk."""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, "r") as f:
                    rules_data = json.load(f)

                self.rules = {
                    rule_id: MangleRule(**rule_data)
                    for rule_id, rule_data in rules_data.items()
                }
                logger.info(f"Loaded {len(self.rules)} rules from {self.rules_file}")
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            # Initialize with empty rules if loading fails
            self.rules = {}

    def _save_rules(self) -> None:
        """Save rules to disk."""
        try:
            rules_data = {
                rule_id: {
                    "name": rule.name,
                    "body": rule.body,
                    "description": rule.description,
                    "tags": rule.tags,
                    "created_at": rule.created_at
                }
                for rule_id, rule in self.rules.items()
            }

            with open(self.rules_file, "w") as f:
                json.dump(rules_data, f, indent=2)

            logger.debug(f"Saved {len(self.rules)} rules to {self.rules_file}")
        except Exception as e:
            logger.error(f"Error saving rules: {e}")

    def _load_facts(self) -> None:
        """Load facts from disk."""
        try:
            if self.facts_file.exists():
                with open(self.facts_file, "r") as f:
                    self.facts = json.load(f)
                logger.info(f"Loaded facts for {len(self.facts)} relations from {self.facts_file}")
        except Exception as e:
            logger.error(f"Error loading facts: {e}")
            # Initialize with empty facts if loading fails
            self.facts = {}

    def _save_facts(self) -> None:
        """Save facts to disk."""
        try:
            with open(self.facts_file, "w") as f:
                json.dump(self.facts, f, indent=2)

            logger.debug(f"Saved facts for {len(self.facts)} relations to {self.facts_file}")
        except Exception as e:
            logger.error(f"Error saving facts: {e}")

    def _convert_fact_to_mangle(self, relation: str, fact: Dict[str, Any]) -> str:
        """Convert a fact dictionary to Mangle syntax."""
        # Handle different data types appropriately
        values = []

        for key, value in fact.items():
            if isinstance(value, str):
                values.append(f'"{value}"')  # Strings are quoted
            elif isinstance(value, bool):
                values.append(str(value).lower())  # True -> true
            elif value is None:
                values.append("null")  # null value
            else:
                values.append(str(value))  # Numbers and other types

        return f"{relation}({', '.join(values)})"

    def _parse_mangle_output(self, output: str) -> Dict[str, Any]:
        """Parse the output from Mangle execution."""
        # This is a simple parser and would need to be enhanced for real use
        lines = output.strip().split("\n")
        results = []

        current_relation = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect relation name
            if ":" in line and not line.startswith(" "):
                current_relation = line.rstrip(":")
                continue

            # Extract result tuples
            if line.startswith(" ") and "=" in line:
                parts = [p.strip() for p in line.split(",")]
                result = {}

                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        # Basic parsing of value types
                        if value.startswith('"') and value.endswith('"'):
                            result[key.strip()] = value[1:-1]  # String
                        elif value.lower() == "true":
                            result[key.strip()] = True  # Boolean
                        elif value.lower() == "false":
                            result[key.strip()] = False  # Boolean
                        else:
                            try:
                                # Try to parse as number
                                result[key.strip()] = float(value) if "." in value else int(value)
                            except ValueError:
                                result[key.strip()] = value  # Fallback to string

                results.append(result)

        return {
            "relation": current_relation,
            "results": results
        }


# For standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import asyncio

    async def test():
        plugin = ManglePlugin()
        await plugin.initialize(EventBus())

        # Add some rules
        plugin.add_rule(
            "vulnerable_log4j",
            """vulnerable_log4j(P, V) :-
              project(P),
              depends_on(P, "log4j", V),
              V != "2.17.1",
              V != "2.12.4",
              V != "2.3.2".
            """,
            "Finds projects with vulnerable log4j versions",
            ["security", "vulnerability"]
        )

        # Add some facts
        plugin.add_fact("project", {"name": "project_a"})
        plugin.add_fact("project", {"name": "project_b"})
        plugin.add_fact("depends_on", {"project": "project_a", "lib": "log4j", "version": "2.15.0"})
        plugin.add_fact("depends_on", {"project": "project_b", "lib": "log4j", "version": "2.17.1"})

        # Execute a query
        result = await plugin.execute_query("vulnerable_log4j(P, V)")
        print(f"Query result: {result}")

    asyncio.run(test())
