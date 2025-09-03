#!/usr/bin/env python3
"""
Mangle Integration Plugin for Super Alita

This module provides a plugin to integrate Google's Mangle deductive database
programming language with Super Alita. It enables advanced logical reasoning,
security analysis, and knowledge graph traversal capabilities.

Mangle (https://github.com/google/mangle) is a Datalog extension designed for
deductive database programming, allowing for complex logical inference and
recursive rule processing.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.core import proc
from src.core.events import create_event
from src.plugins.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)

# Constants for Mangle integration
MANGLE_BIN_PATH = os.environ.get("MANGLE_BIN_PATH", "mangle")
DEFAULT_TIMEOUT_SECONDS = 30


class MangleAbility:
    """Mangle deductive database programming integration for Super Alita.

    This class provides a Python interface to Google's Mangle language,
    enabling AI agents to perform logical reasoning, security analysis,
    knowledge graph traversal, and complex decision-making with full
    explainability.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Mangle integration.

        Args:
            config: Configuration dictionary with optional parameters:
                - binary_path: Path to Mangle executable
                - timeout: Execution timeout in seconds
                - knowledge_base_dir: Directory to store persistent knowledge base files
        """
        self.config = config or {}
        self.binary_path = self.config.get("binary_path", MANGLE_BIN_PATH)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT_SECONDS)
        self.knowledge_base_dir = self.config.get("knowledge_base_dir", "./data/mangle")

        # Create knowledge base directory if it doesn't exist
        Path(self.knowledge_base_dir).mkdir(parents=True, exist_ok=True)

        # Track registered facts and rules
        self.facts: set[str] = set()
        self.rules: dict[str, str] = {}

        # Load any existing knowledge base
        self._load_knowledge_base()

        logger.info(
            f"Mangle ability initialized with {len(self.facts)} facts and {len(self.rules)} rules"
        )

    def _load_knowledge_base(self) -> None:
        """Load existing knowledge base from disk if available."""
        kb_path = Path(self.knowledge_base_dir) / "knowledge_base.json"
        if kb_path.exists():
            try:
                data = json.loads(kb_path.read_text(encoding="utf-8"))
                self.facts = set(data.get("facts", []))
                self.rules = data.get("rules", {})
                logger.info(f"Loaded knowledge base from {kb_path}")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")

    def _save_knowledge_base(self) -> None:
        """Persist knowledge base to disk."""
        kb_path = Path(self.knowledge_base_dir) / "knowledge_base.json"
        try:
            data = {"facts": list(self.facts), "rules": self.rules}
            kb_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info(f"Saved knowledge base to {kb_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

    async def add_fact(self, fact: str) -> dict[str, Any]:
        """Add a fact to the knowledge base.

        Args:
            fact: A Mangle fact (e.g., "project('super-alita').")

        Returns:
            Dictionary with operation result
        """
        if not fact.endswith("."):
            fact = f"{fact}."

        self.facts.add(fact)
        self._save_knowledge_base()

        return {"success": True, "fact": fact, "total_facts": len(self.facts)}

    async def add_rule(self, name: str, rule: str) -> dict[str, Any]:
        """Add a rule to the knowledge base.

        Args:
            name: Unique name for the rule
            rule: A Mangle rule (e.g., "vulnerable(P) :- project(P), uses(P, 'log4j', V), V < '2.17.0'.")

        Returns:
            Dictionary with operation result
        """
        if not rule.endswith("."):
            rule = f"{rule}."

        self.rules[name] = rule
        self._save_knowledge_base()

        return {
            "success": True,
            "rule_name": name,
            "rule": rule,
            "total_rules": len(self.rules),
        }

    async def query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a Mangle query against the knowledge base.

        Args:
            query: A Mangle query (e.g., "vulnerable(P)?")
            params: Optional parameters for the query

        Returns:
            Dictionary with query results
        """
        # Create temporary file with knowledge base and query
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mgl", delete=False) as f:
            # Add all facts
            for fact in self.facts:
                f.write(f"{fact}\n")

            # Add all rules
            for rule_name, rule in self.rules.items():
                f.write(f"{rule}\n")

            # Add the query
            if not query.endswith("?"):
                query = f"{query}?"
            f.write(f"{query}\n")
            f.flush()

            try:
                # Execute Mangle with the temporary file (async wrapper)
                cmd = [self.binary_path, "query", "--format", "json", f.name]
                stdout = await proc.arun(cmd, timeout=self.timeout)

                # Parse JSON output
                if stdout.strip():
                    parsed_result = json.loads(stdout)
                    return {
                        "success": True,
                        "results": parsed_result,
                        "count": len(parsed_result),
                    }
                else:
                    return {"success": True, "results": [], "count": 0}

            except proc.ProcError as e:
                return {
                    "success": False,
                    "error": f"Mangle execution error ({e.returncode}): {e.stderr}",
                }
            except Exception as e:
                return {"success": False, "error": f"Unexpected error: {str(e)}"}
            finally:
                # Clean up temporary file
                try:
                    os.unlink(f.name)
                except:
                    pass

    async def analyze_dependencies(
        self, project_deps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze project dependencies for security vulnerabilities.

        Args:
            project_deps: List of project dependencies with name, version info

        Returns:
            Analysis results with potential vulnerabilities
        """
        # First, add facts for each dependency
        for dep in project_deps:
            name = dep.get("name", "")
            version = dep.get("version", "")
            if name and version:
                await self.add_fact(f"dependency('{name}', '{version}')")

        # Add rule for vulnerable dependencies
        await self.add_rule(
            "vuln_check",
            """vulnerable(Name, Version) :-
                dependency(Name, Version),
                known_vulnerability(Name, Version).""",
        )

        # Query for vulnerabilities
        results = await self.query("vulnerable(Name, Version)")
        return results

    async def knowledge_graph_query(
        self, query: str, context: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Perform a knowledge graph query with additional context.

        Args:
            query: Knowledge graph query
            context: Optional additional context to enhance the query

        Returns:
            Query results
        """
        # Add context as temporary facts if provided
        temp_facts = []
        if context:
            for item in context:
                relation = item.get("relation", "")
                subject = item.get("subject", "")
                object_ = item.get("object", "")
                if relation and subject and object_:
                    fact = f"{relation}('{subject}', '{object}')"
                    await self.add_fact(fact)
                    temp_facts.append(fact)

        # Perform the query
        results = await self.query(query)

        # Remove temporary facts
        for fact in temp_facts:
            if fact in self.facts:
                self.facts.remove(fact)

        return results

    async def explain_query_results(self, query: str) -> dict[str, Any]:
        """Execute a query and provide explanation for the results.

        Args:
            query: A Mangle query

        Returns:
            Results with explanation of the logical reasoning steps
        """
        # Execute with explanation flag
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mgl", delete=False) as f:
            # Add all facts and rules
            for fact in self.facts:
                f.write(f"{fact}\n")
            for rule in self.rules.values():
                f.write(f"{rule}\n")
            if not query.endswith("?"):
                query = f"{query}?"
            f.write(f"{query}\n")
            f.flush()

            try:
                # Run with explain flag (need stderr), use to_thread with subprocess.run
                cmd = [
                    self.binary_path,
                    "query",
                    "--explain",
                    "--format",
                    "json",
                    f.name,
                ]

                def _call() -> subprocess.CompletedProcess[str]:
                    return subprocess.run(
                        cmd,
                        shell=False,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )

                result: subprocess.CompletedProcess[str] = await asyncio.to_thread(
                    _call
                )
                results = json.loads(result.stdout) if result.stdout.strip() else []
                explanation = (
                    result.stderr if result.stderr else "No explanation available"
                )
                return {
                    "success": True,
                    "results": results,
                    "explanation": explanation,
                    "count": len(results),
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
            finally:
                try:
                    os.unlink(f.name)
                except:
                    pass


class ManglePluginInterface(PluginInterface):
    """Mangle plugin interface for Super Alita.

    This plugin provides deductive database programming capabilities
    through Google's Mangle language, enabling advanced logical reasoning,
    security analysis, and knowledge graph traversal.
    """

    def __init__(self):
        """Initialize the Mangle plugin."""
        self.name = "mangle_plugin"
        self.mangle_ability = None
        self.event_bus = None

    async def cleanup(self):
        """Clean up resources used by the plugin."""
        logger.info("Cleaning up Mangle plugin resources")
        return True

    async def initialize(self, event_bus, **kwargs) -> bool:
        """Initialize the plugin with event bus and configuration.

        Args:
            event_bus: Super Alita event bus
            **kwargs: Additional configuration parameters

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.event_bus = event_bus

            # Extract Mangle-specific config
            mangle_config = kwargs.get("mangle_config", {})

            # Initialize Mangle ability
            self.mangle_ability = MangleAbility(mangle_config)

            # Subscribe to relevant events
            await event_bus.subscribe("request_mangle_query", self.handle_mangle_query)
            await event_bus.subscribe("request_mangle_fact_add", self.handle_add_fact)
            await event_bus.subscribe("request_mangle_rule_add", self.handle_add_rule)
            await event_bus.subscribe(
                "request_mangle_dependency_analysis", self.handle_dependency_analysis
            )
            await event_bus.subscribe(
                "request_mangle_knowledge_graph", self.handle_knowledge_graph
            )
            await event_bus.subscribe("request_mangle_explain", self.handle_explain)

            logger.info("Mangle plugin initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Mangle plugin: {e}")
            return False

    async def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Process an event from the event bus.

        Args:
            event: Event dictionary

        Returns:
            Optional response event
        """
        # Main event processing logic
        event_type = event.get("event_type")

        if not self.mangle_ability:
            logger.warning(f"Received {event_type} but Mangle ability not initialized")
            return None

        # Event handling is delegated to specific handlers
        return None

    async def handle_mangle_query(self, event: dict[str, Any]) -> None:
        """Handle a Mangle query event.

        Args:
            event: Event with query parameters
        """
        if not self.mangle_ability:
            await self._send_error_response(event, "Mangle ability not initialized")
            return

        try:
            query = event.get("query")
            params = event.get("params")

            if not query:
                await self._send_error_response(event, "Query parameter missing")
                return

            result = await self.mangle_ability.query(query, params)

            # Send response event
            response_event = create_event(
                "mangle_query_response", request_id=event.get("id"), result=result
            )
            await self.event_bus.publish(response_event)

        except Exception as e:
            await self._send_error_response(event, f"Error executing query: {str(e)}")

    async def handle_add_fact(self, event: dict[str, Any]) -> None:
        """Handle a request to add a fact to the Mangle knowledge base.

        Args:
            event: Event with fact information
        """
        if not self.mangle_ability:
            await self._send_error_response(event, "Mangle ability not initialized")
            return

        try:
            fact = event.get("fact")

            if not fact:
                await self._send_error_response(event, "Fact parameter missing")
                return

            result = await self.mangle_ability.add_fact(fact)

            # Send response event
            response_event = create_event(
                "mangle_fact_add_response", request_id=event.get("id"), result=result
            )
            await self.event_bus.publish(response_event)

        except Exception as e:
            await self._send_error_response(event, f"Error adding fact: {str(e)}")

    async def handle_add_rule(self, event: dict[str, Any]) -> None:
        """Handle a request to add a rule to the Mangle knowledge base.

        Args:
            event: Event with rule information
        """
        if not self.mangle_ability:
            await self._send_error_response(event, "Mangle ability not initialized")
            return

        try:
            name = event.get("name")
            rule = event.get("rule")

            if not name or not rule:
                await self._send_error_response(event, "Name or rule parameter missing")
                return

            result = await self.mangle_ability.add_rule(name, rule)

            # Send response event
            response_event = create_event(
                "mangle_rule_add_response", request_id=event.get("id"), result=result
            )
            await self.event_bus.publish(response_event)

        except Exception as e:
            await self._send_error_response(event, f"Error adding rule: {str(e)}")

    async def handle_dependency_analysis(self, event: dict[str, Any]) -> None:
        """Handle a request to analyze dependencies.

        Args:
            event: Event with dependency information
        """
        if not self.mangle_ability:
            await self._send_error_response(event, "Mangle ability not initialized")
            return

        try:
            dependencies = event.get("dependencies", [])

            if not dependencies:
                await self._send_error_response(
                    event, "Dependencies parameter missing or empty"
                )
                return

            result = await self.mangle_ability.analyze_dependencies(dependencies)

            # Send response event
            response_event = create_event(
                "mangle_dependency_analysis_response",
                request_id=event.get("id"),
                result=result,
            )
            await self.event_bus.publish(response_event)

        except Exception as e:
            await self._send_error_response(
                event, f"Error analyzing dependencies: {str(e)}"
            )

    async def handle_knowledge_graph(self, event: dict[str, Any]) -> None:
        """Handle a knowledge graph query request.

        Args:
            event: Event with query and context information
        """
        if not self.mangle_ability:
            await self._send_error_response(event, "Mangle ability not initialized")
            return

        try:
            query = event.get("query")
            context = event.get("context")

            if not query:
                await self._send_error_response(event, "Query parameter missing")
                return

            result = await self.mangle_ability.knowledge_graph_query(query, context)

            # Send response event
            response_event = create_event(
                "mangle_knowledge_graph_response",
                request_id=event.get("id"),
                result=result,
            )
            await self.event_bus.publish(response_event)

        except Exception as e:
            await self._send_error_response(
                event, f"Error executing knowledge graph query: {str(e)}"
            )

    async def handle_explain(self, event: dict[str, Any]) -> None:
        """Handle a request for explanation of a query.

        Args:
            event: Event with query information
        """
        if not self.mangle_ability:
            await self._send_error_response(event, "Mangle ability not initialized")
            return

        try:
            query = event.get("query")

            if not query:
                await self._send_error_response(event, "Query parameter missing")
                return

            result = await self.mangle_ability.explain_query_results(query)

            # Send response event
            response_event = create_event(
                "mangle_explain_response", request_id=event.get("id"), result=result
            )
            await self.event_bus.publish(response_event)

        except Exception as e:
            await self._send_error_response(
                event, f"Error generating explanation: {str(e)}"
            )

    async def _send_error_response(
        self, original_event: dict[str, Any], error_message: str
    ) -> None:
        """Send an error response for a failed request.

        Args:
            original_event: The original event that triggered this error
            error_message: Error message to include in the response
        """
        event_type = original_event.get("event_type", "unknown")
        response_type = event_type.replace("request_", "") + "_response"

        error_event = create_event(
            response_type,
            request_id=original_event.get("id"),
            success=False,
            error=error_message,
        )

        await self.event_bus.publish(error_event)

    async def run_rule(self, rule: str, query: str) -> dict[str, Any]:
        """Execute a temporary program consisting of stored facts, a rule, and a query."""
        if not rule.endswith("."):
            rule = f"{rule}."
        if not query.endswith("?"):
            query = f"{query}?"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mgl", delete=False) as f:
            for fact in self.facts:
                f.write(f"{fact}\n")
            for _name, r in self.rules.items():
                f.write(f"{r}\n")
            f.write(f"{rule}\n")
            f.write(f"{query}\n")
            f.flush()
            try:
                stdout = await proc.arun(
                    [self.binary_path, "query", "--format", "json", f.name],
                    timeout=self.timeout,
                )
                data = json.loads(stdout) if stdout.strip() else []
                return {"success": True, "results": data, "count": len(data)}
            except proc.ProcError as e:
                return {
                    "success": False,
                    "error": f"Mangle execution error ({e.returncode}): {e.stderr}",
                }
            finally:
                try:
                    os.unlink(f.name)
                except Exception:
                    pass

    async def rule_catalog(self) -> dict[str, Any]:
        """Return a list of discovered Mangle rules from disk or current session."""
        catalog: list[dict[str, Any]] = []
        # Prefer plugin rules storage if present
        try:
            rules_file = Path("./data/mangle/rules.json")
            if rules_file.exists():
                rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
                for rid, meta in rules_data.items():
                    catalog.append(
                        {
                            "id": rid,
                            "name": meta.get("name", rid),
                            "description": meta.get("description", ""),
                            "created_at": meta.get("created_at"),
                            "tags": meta.get("tags", []),
                        }
                    )
        except Exception:
            pass
        # Fallback to in-memory rules
        if not catalog and self.rules:
            for name, body in self.rules.items():
                catalog.append(
                    {
                        "id": name,
                        "name": name,
                        "description": body[:60] + ("..." if len(body) > 60 else ""),
                    }
                )
        return {"success": True, "rules": catalog, "count": len(catalog)}

    async def validate_output(self, output_text: str, domain=None, meta=None):
        """Validate output against Mangle policy rules.

        Uses rules with head violation(Reason) to detect policy violations in the output.
        If violations are found, returns details with rejection flag set to True.

        Args:
            output_text: The output text to validate
            domain: Optional domain for domain-specific rules
            meta: Optional metadata about the generation context

        Returns:
            Dictionary with keys:
                - valid: Boolean indicating if the output passes validation
                - violations: List of violation reasons found
                - confidence_penalty: Suggested confidence penalty (0.0-1.0)
        """
        try:
            import json
            from pathlib import Path

            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {"valid": True, "violations": [], "confidence_penalty": 0.0}

            # Build temp facts about the output and context
            tfacts = []

            # Add truncated output text fact (with simple escaping)
            safe_text = output_text[:500].replace("'", " ")
            tfacts.append(f"output_text('{safe_text}')")

            if domain:
                safe_domain = domain.replace("'", " ")
                tfacts.append(f"domain('{safe_domain}')")

            if meta:
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"{k}({v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", " ")
                        tfacts.append(f"{k}('{safe_v}')")

            # Check for violation rules
            violations = []
            confidence_penalty = 0.0

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for rule_id, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is a violation rule
                head = body.split(":-", 1)[0].strip().rstrip(".")
                if not head.startswith("violation("):
                    continue

                # Run the violation check
                res = await self.run_rule(body, "violation(Reason)", temp_facts=tfacts)
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            reason = result.get("Reason")
                            if reason:
                                violations.append(str(reason))
                    except Exception:
                        continue

            # If any violations found, calculate penalty and return details
            valid = len(violations) == 0
            if not valid:
                # Scale penalty based on violation count, max 0.5
                confidence_penalty = min(0.5, 0.1 * len(violations))

            return {
                "valid": valid,
                "violations": violations,
                "confidence_penalty": confidence_penalty,
            }
        except Exception as e:
            logger.warning(f"Output validation error: {e}")
            return {"valid": True, "violations": [], "confidence_penalty": 0.0}

    def _parse_head(self, body: str) -> tuple[str, list[str]]:
        """Parse a Mangle rule head to extract predicate and variables."""
        head = body.split(":-", 1)[0].strip().rstrip(".")
        if not head:
            return "", []
        l = head.find("(")
        r = head.rfind(")")
        if l == -1 or r == -1 or r < l:
            return head, []
        pred = head[:l].strip()
        args = [a.strip() for a in head[l + 1 : r].split(",") if a.strip()]
        return pred, args

    async def validate_output(
        self,
        output_text: str,
        *,
        domain: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate output against Mangle policy rules.

        Uses rules with head violation(Reason) to detect policy violations in the output.
        If violations are found, returns details with rejection flag set to True.

        Args:
            output_text: The output text to validate
            domain: Optional domain for domain-specific rules
            meta: Optional metadata about the generation context

        Returns:
            Dictionary with keys:
                - valid: Boolean indicating if the output passes validation
                - violations: List of violation reasons found
                - confidence_penalty: Suggested confidence penalty (0.0-1.0)
        """
        try:
            import json
            from pathlib import Path

            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {"valid": True, "violations": [], "confidence_penalty": 0.0}

            # Build temp facts about the output and context
            # Truncate text to avoid oversized facts
            safe_text = output_text[:500].replace("'", "\\'")
            tfacts = [f"output_text('{safe_text}')"]

            if domain:
                tfacts.append(f"domain('{domain}')")

            if meta:
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"{k}({v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", "\\'")
                        tfacts.append(f"{k}('{safe_v}')")

            # Check for violation rules
            violations = []
            confidence_penalty = 0.0

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for _rid, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is a violation rule
                pred, _vars = self._parse_head(body)
                if pred != "violation":
                    continue

                # Run the violation check
                res = await self.run_rule(body, "violation(Reason)", temp_facts=tfacts)
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            reason = result.get("Reason")
                            if reason:
                                violations.append(str(reason))
                    except Exception:
                        continue

            # If any violations found, calculate penalty and return details
            valid = len(violations) == 0
            if not valid:
                # Scale penalty based on violation count, max 0.5
                confidence_penalty = min(0.5, 0.1 * len(violations))

            return {
                "valid": valid,
                "violations": violations,
                "confidence_penalty": confidence_penalty,
            }
        except Exception as e:
            logger.warning(f"Output validation error: {e}")
            return {"valid": True, "violations": [], "confidence_penalty": 0.0}

    async def validate_output(
        self,
        output_text: str,
        *,
        domain: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate output against Mangle policy rules.

        Uses rules with head violation(Reason) to detect policy violations in the output.
        If violations are found, returns details with rejection flag set to True.

        Args:
            output_text: The output text to validate
            domain: Optional domain for domain-specific rules
            meta: Optional metadata about the generation context

        Returns:
            Dictionary with keys:
                - valid: Boolean indicating if the output passes validation
                - violations: List of violation reasons found
                - confidence_penalty: Suggested confidence penalty (0.0-1.0)
        """
        try:
            import json
            from pathlib import Path

            rules_file = Path("./data/mangle/rules.json")
            if not rules_file.exists():
                return {"valid": True, "violations": [], "confidence_penalty": 0.0}

            # Build temp facts about the output and context
            # Truncate to avoid oversized facts; escape single quotes
            _slice = output_text[:500].replace("'", "\\'")
            tfacts: list[str] = [f"output_text('{_slice}')"]
            if domain:
                tfacts.append(f"domain('{domain}')")
            if meta:
                for k, v in meta.items():
                    if isinstance(v, (int, float)):
                        tfacts.append(f"{k}({v})")
                    elif isinstance(v, str):
                        safe_v = v.replace("'", "\\'")
                        tfacts.append(f"{k}('{safe_v}')")

            # Check for violation rules
            violations: list[str] = []
            confidence_penalty = 0.0

            rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
            for _rid, meta_rule in rules_data.items():
                body = meta_rule.get("body") or meta_rule.get("rule") or ""
                if not body:
                    continue

                # Check if this is a violation rule
                pred, _vars = self._parse_head(body)
                if pred != "violation":
                    continue

                # Run the violation check
                res = await self.run_rule(body, "violation(Reason)", temp_facts=tfacts)
                if res.get("success") and res.get("count", 0) > 0:
                    try:
                        for result in res.get("results", []):
                            reason = result.get("Reason")
                            if reason:
                                violations.append(str(reason))
                    except Exception:
                        continue

            # If any violations found, calculate penalty and return details
            valid = len(violations) == 0
            if not valid:
                # Scale penalty based on violation count, max 0.5
                confidence_penalty = min(0.5, 0.1 * len(violations))

            return {
                "valid": valid,
                "violations": violations,
                "confidence_penalty": confidence_penalty,
            }
        except Exception as e:
            logger.warning(f"Output validation error: {e}")
            return {"valid": True, "violations": [], "confidence_penalty": 0.0}

    def _parse_head(self, body: str) -> tuple[str, list[str]]:
        """Parse a Mangle rule head to extract predicate and variables."""
        head = body.split(":-", 1)[0].strip().rstrip(".")
        if not head:
            return "", []
        l = head.find("(")
        r = head.rfind(")")
        if l == -1 or r == -1 or r < l:
            return head, []
        pred = head[:l].strip()
        args = [a.strip() for a in head[l + 1 : r].split(",") if a.strip()]
        return pred, args
