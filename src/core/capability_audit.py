#!/usr/bin/env python3
"""
Capability Audit System for Super Alita
=======================================

Comprehensive system to audit, track, and manage all agent capabilities including:
- Static plugins and their methods
- Dynamic tools and atoms
- MCP tools and external capabilities
- Memory and reasoning systems
- Knowledge bases and data sources

Provides gap analysis and capability mapping for intelligent tool routing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Import existing registries
from src.tools.dynamic_tools import dynamic_tool_registry

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of capabilities in the agent system"""

    PLUGIN = "plugin"
    DYNAMIC_TOOL = "dynamic_tool"
    ATOM = "atom"
    MCP_TOOL = "mcp_tool"
    MEMORY_SYSTEM = "memory_system"
    REASONING_ENGINE = "reasoning_engine"
    KNOWLEDGE_SOURCE = "knowledge_source"
    EXTERNAL_API = "external_api"
    WORKFLOW = "workflow"


class CapabilityStatus(Enum):
    """Status of a capability"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class CapabilityMetadata:
    """Metadata for a capability"""

    name: str
    capability_type: CapabilityType
    description: str
    version: str = "1.0.0"
    author: str = "super-alita"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None
    usage_count: int = 0
    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    error_message: str | None = None
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "use_cases": self.use_cases,
            "examples": self.examples,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "status": self.status.value,
            "error_message": self.error_message,
            "file_path": self.file_path,
        }


@dataclass
class CapabilityInterface:
    """Interface definition for a capability"""

    name: str
    methods: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    events_emitted: list[str] = field(default_factory=list)
    events_consumed: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    return_types: dict[str, str] = field(default_factory=dict)


class CapabilityRegistry:
    """Central registry for all agent capabilities"""

    def __init__(self):
        self.capabilities: dict[str, CapabilityMetadata] = {}
        self.interfaces: dict[str, CapabilityInterface] = {}
        self.capability_index: dict[str, list[str]] = {}  # keyword -> capability names
        self.dependency_graph: dict[str, list[str]] = {}
        self.audit_history: list[dict[str, Any]] = []

    def register_capability(
        self, metadata: CapabilityMetadata, interface: CapabilityInterface | None = None
    ) -> bool:
        """Register a new capability"""
        try:
            self.capabilities[metadata.name] = metadata

            if interface:
                self.interfaces[metadata.name] = interface

            # Build keyword index for search
            self._build_keyword_index(metadata)

            # Update dependency graph
            self._update_dependency_graph(metadata)

            logger.info(
                f"Registered capability: {metadata.name} ({metadata.capability_type.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register capability {metadata.name}: {e}")
            return False

    def get_capability(self, name: str) -> CapabilityMetadata | None:
        """Get capability by name"""
        return self.capabilities.get(name)

    def list_capabilities(
        self,
        capability_type: CapabilityType | None = None,
        status: CapabilityStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[CapabilityMetadata]:
        """List capabilities with optional filtering"""
        capabilities = list(self.capabilities.values())

        if capability_type:
            capabilities = [
                c for c in capabilities if c.capability_type == capability_type
            ]

        if status:
            capabilities = [c for c in capabilities if c.status == status]

        if tags:
            capabilities = [
                c for c in capabilities if any(tag in c.tags for tag in tags)
            ]

        return capabilities

    def search_capabilities(self, query: str) -> list[CapabilityMetadata]:
        """Search capabilities by keyword"""
        query_words = query.lower().split()
        matches = set()

        for word in query_words:
            if word in self.capability_index:
                matches.update(self.capability_index[word])

        return [
            self.capabilities[name] for name in matches if name in self.capabilities
        ]

    def get_capability_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        capabilities = list(self.capabilities.values())

        stats = {
            "total_capabilities": len(capabilities),
            "by_type": {},
            "by_status": {},
            "most_used": [],
            "least_used": [],
            "recent_additions": [],
            "error_capabilities": [],
        }

        # Group by type
        for cap_type in CapabilityType:
            count = len([c for c in capabilities if c.capability_type == cap_type])
            if count > 0:
                stats["by_type"][cap_type.value] = count

        # Group by status
        for status in CapabilityStatus:
            count = len([c for c in capabilities if c.status == status])
            if count > 0:
                stats["by_status"][status.value] = count

        # Most/least used
        sorted_by_usage = sorted(
            capabilities, key=lambda c: c.usage_count, reverse=True
        )
        stats["most_used"] = [
            {"name": c.name, "usage_count": c.usage_count} for c in sorted_by_usage[:5]
        ]
        stats["least_used"] = [
            {"name": c.name, "usage_count": c.usage_count} for c in sorted_by_usage[-5:]
        ]

        # Recent additions
        sorted_by_date = sorted(capabilities, key=lambda c: c.created_at, reverse=True)
        stats["recent_additions"] = [
            {"name": c.name, "created_at": c.created_at.isoformat()}
            for c in sorted_by_date[:5]
        ]

        # Error capabilities
        stats["error_capabilities"] = [
            {"name": c.name, "error": c.error_message}
            for c in capabilities
            if c.status == CapabilityStatus.ERROR
        ]

        return stats

    def export_capabilities(self, file_path: str) -> bool:
        """Export capabilities to JSON file"""
        try:
            export_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "capabilities": {
                    name: cap.to_dict() for name, cap in self.capabilities.items()
                },
                "interfaces": {
                    name: {
                        "name": iface.name,
                        "methods": iface.methods,
                        "properties": iface.properties,
                        "events_emitted": iface.events_emitted,
                        "events_consumed": iface.events_consumed,
                        "parameters": iface.parameters,
                        "return_types": iface.return_types,
                    }
                    for name, iface in self.interfaces.items()
                },
                "stats": self.get_capability_stats(),
            }

            Path(file_path).write_text(json.dumps(export_data, indent=2))
            logger.info(
                f"Exported {len(self.capabilities)} capabilities to {file_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to export capabilities: {e}")
            return False

    def _build_keyword_index(self, metadata: CapabilityMetadata):
        """Build searchable keyword index"""
        keywords = []

        # Add name words
        keywords.extend(metadata.name.lower().split("_"))
        keywords.extend(metadata.name.lower().split("-"))

        # Add description words
        if metadata.description:
            keywords.extend(metadata.description.lower().split())

        # Add tags
        keywords.extend([tag.lower() for tag in metadata.tags])

        # Add use cases
        for use_case in metadata.use_cases:
            keywords.extend(use_case.lower().split())

        # Update index
        for keyword in set(keywords):
            if keyword not in self.capability_index:
                self.capability_index[keyword] = []
            if metadata.name not in self.capability_index[keyword]:
                self.capability_index[keyword].append(metadata.name)

    def _update_dependency_graph(self, metadata: CapabilityMetadata):
        """Update capability dependency graph"""
        self.dependency_graph[metadata.name] = metadata.dependencies


class CapabilityAuditor:
    """Audits existing capabilities in the agent system"""

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
        self.plugins_path = Path("src/plugins")
        self.tools_path = Path("src/tools")

    async def audit_all_capabilities(self) -> dict[str, Any]:
        """Perform comprehensive audit of all capabilities"""
        logger.info("Starting comprehensive capability audit...")

        audit_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "audit_results": {
                "plugins": await self._audit_plugins(),
                "dynamic_tools": await self._audit_dynamic_tools(),
                "atoms": await self._audit_atoms(),
                "mcp_tools": await self._audit_mcp_tools(),
                "memory_systems": await self._audit_memory_systems(),
                "reasoning_engines": await self._audit_reasoning_engines(),
            },
            "summary": {},
            "issues": [],
            "recommendations": [],
        }

        # Generate summary
        audit_results["summary"] = self._generate_audit_summary(
            audit_results["audit_results"]
        )

        # Identify issues and recommendations
        audit_results["issues"] = self._identify_issues(audit_results["audit_results"])
        audit_results["recommendations"] = self._generate_recommendations(audit_results)

        # Store audit in history
        self.registry.audit_history.append(audit_results)

        logger.info(
            f"Audit complete. Found {audit_results['summary']['total_capabilities']} capabilities"
        )
        return audit_results

    async def _audit_plugins(self) -> dict[str, Any]:
        """Audit plugin capabilities"""
        plugin_results = {"discovered": [], "errors": [], "total": 0}

        try:
            if not self.plugins_path.exists():
                plugin_results["errors"].append("Plugins directory not found")
                return plugin_results

            for plugin_file in self.plugins_path.glob("*.py"):
                if (
                    plugin_file.name.startswith("_")
                    or plugin_file.name == "plugin_interface.py"
                ):
                    continue

                try:
                    plugin_info = await self._analyze_plugin_file(plugin_file)
                    if plugin_info:
                        plugin_results["discovered"].append(plugin_info)

                        # Register in capability registry
                        metadata = CapabilityMetadata(
                            name=plugin_info["name"],
                            capability_type=CapabilityType.PLUGIN,
                            description=plugin_info["description"],
                            tags=plugin_info["tags"],
                            use_cases=plugin_info["use_cases"],
                            file_path=str(plugin_file),
                            status=(
                                CapabilityStatus.ACTIVE
                                if plugin_info["valid"]
                                else CapabilityStatus.ERROR
                            ),
                            error_message=plugin_info.get("error"),
                        )

                        interface = CapabilityInterface(
                            name=plugin_info["name"],
                            methods=plugin_info["methods"],
                            properties=plugin_info["properties"],
                            events_emitted=plugin_info.get("events_emitted", []),
                            events_consumed=plugin_info.get("events_consumed", []),
                        )

                        self.registry.register_capability(metadata, interface)

                except Exception as e:
                    error = f"Error analyzing {plugin_file.name}: {e}"
                    plugin_results["errors"].append(error)
                    logger.warning(error)

            plugin_results["total"] = len(plugin_results["discovered"])

        except Exception as e:
            plugin_results["errors"].append(f"Plugin audit failed: {e}")

        return plugin_results

    async def _analyze_plugin_file(self, plugin_file: Path) -> dict[str, Any] | None:
        """Analyze a single plugin file"""
        try:
            # Read file content
            content = plugin_file.read_text(encoding="utf-8")

            plugin_info = {
                "name": plugin_file.stem,
                "file_path": str(plugin_file),
                "description": "Plugin description not found",
                "methods": [],
                "properties": [],
                "tags": [],
                "use_cases": [],
                "valid": False,
                "has_plugin_interface": False,
                "class_names": [],
            }

            # Basic analysis - look for plugin class patterns
            lines = content.split("\n")
            in_class = False
            current_class = None

            for line in lines:
                line_clean = line.strip()

                # Find class definitions
                if line_clean.startswith("class ") and "Plugin" in line_clean:
                    class_name = line_clean.split()[1].split("(")[0]
                    plugin_info["class_names"].append(class_name)
                    current_class = class_name
                    in_class = True

                    # Check if extends PluginInterface
                    if "PluginInterface" in line_clean:
                        plugin_info["has_plugin_interface"] = True
                        plugin_info["valid"] = True

                # Find methods
                elif in_class and line_clean.startswith("def "):
                    method_name = line_clean.split()[1].split("(")[0]
                    if not method_name.startswith("_"):  # Public methods only
                        plugin_info["methods"].append(method_name)

                # Find properties
                elif in_class and "@property" in line_clean:
                    plugin_info["properties"].append("property_found")

                # Extract docstring for description
                elif '"""' in line_clean and "description" not in plugin_info:
                    # Simple docstring extraction
                    if (
                        current_class
                        and plugin_info["description"] == "Plugin description not found"
                    ):
                        plugin_info["description"] = line_clean.replace(
                            '"""', ""
                        ).strip()

                # End of class
                elif (
                    in_class
                    and line.startswith("class ")
                    and "Plugin" not in line_clean
                ):
                    in_class = False
                    current_class = None

            # Determine tags based on name and content
            name_lower = plugin_info["name"].lower()
            if "memory" in name_lower:
                plugin_info["tags"].extend(["memory", "storage"])
                plugin_info["use_cases"].append("Memory management and retrieval")
            if "conversation" in name_lower:
                plugin_info["tags"].extend(["conversation", "chat", "nlp"])
                plugin_info["use_cases"].append("Natural language conversation")
            if "tool" in name_lower or "creator" in name_lower:
                plugin_info["tags"].extend(["tools", "creation"])
                plugin_info["use_cases"].append("Tool creation and management")
            if "planner" in name_lower:
                plugin_info["tags"].extend(["planning", "reasoning"])
                plugin_info["use_cases"].append("Task planning and orchestration")
            if "atom" in name_lower:
                plugin_info["tags"].extend(["atoms", "execution"])
                plugin_info["use_cases"].append("Atomic task execution")

            return plugin_info

        except Exception as e:
            logger.error(f"Error analyzing plugin file {plugin_file}: {e}")
            return None

    async def _audit_dynamic_tools(self) -> dict[str, Any]:
        """Audit dynamic tools from registry"""
        tool_results = {"discovered": [], "errors": [], "total": 0}

        try:
            # Get tools from dynamic registry
            tool_names = dynamic_tool_registry.list_tools()

            for tool_name in tool_names:
                try:
                    tool = dynamic_tool_registry.get_tool(tool_name)
                    schema = dynamic_tool_registry.get_tool_schema(tool_name)

                    if tool and schema:
                        tool_info = {
                            "name": tool_name,
                            "description": schema.description,
                            "version": schema.version,
                            "parameters": [
                                {
                                    "name": param.name,
                                    "type": param.type.value,
                                    "description": param.description,
                                    "required": param.required,
                                }
                                for param in schema.parameters
                            ],
                            "return_type": schema.return_type,
                            "usage_count": tool.call_count,
                            "last_used": (
                                tool.last_called.isoformat()
                                if tool.last_called
                                else None
                            ),
                            "examples": schema.examples,
                        }

                        tool_results["discovered"].append(tool_info)

                        # Register in capability registry
                        metadata = CapabilityMetadata(
                            name=tool_name,
                            capability_type=CapabilityType.DYNAMIC_TOOL,
                            description=schema.description,
                            version=schema.version,
                            tags=["dynamic", "tool"],
                            usage_count=tool.call_count,
                            last_used=tool.last_called,
                            status=CapabilityStatus.ACTIVE,
                        )

                        interface = CapabilityInterface(
                            name=tool_name,
                            methods=["execute"],
                            parameters={
                                param.name: param.type.value
                                for param in schema.parameters
                            },
                            return_types={"execute": schema.return_type},
                        )

                        self.registry.register_capability(metadata, interface)

                except Exception as e:
                    error = f"Error analyzing dynamic tool {tool_name}: {e}"
                    tool_results["errors"].append(error)
                    logger.warning(error)

            tool_results["total"] = len(tool_results["discovered"])

        except Exception as e:
            tool_results["errors"].append(f"Dynamic tools audit failed: {e}")

        return tool_results

    async def _audit_atoms(self) -> dict[str, Any]:
        """Audit atom files"""
        atom_results = {"discovered": [], "errors": [], "total": 0}

        try:
            # Look for atom files in plugins directory
            atom_files = list(self.plugins_path.glob("*_atom.py"))

            for atom_file in atom_files:
                try:
                    atom_info = {
                        "name": atom_file.stem,
                        "file_path": str(atom_file),
                        "description": f"Atom: {atom_file.stem.replace('_atom', '').replace('_', ' ').title()}",
                        "size_bytes": atom_file.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            atom_file.stat().st_mtime, UTC
                        ).isoformat(),
                    }

                    # Try to extract more info from file content
                    try:
                        content = atom_file.read_text(encoding="utf-8")
                        if '"""' in content:
                            # Extract docstring
                            docstring_start = content.find('"""')
                            docstring_end = content.find('"""', docstring_start + 3)
                            if docstring_end > docstring_start:
                                atom_info["description"] = content[
                                    docstring_start + 3 : docstring_end
                                ].strip()
                    except:
                        pass  # Keep default description

                    atom_results["discovered"].append(atom_info)

                    # Register in capability registry
                    metadata = CapabilityMetadata(
                        name=atom_info["name"],
                        capability_type=CapabilityType.ATOM,
                        description=atom_info["description"],
                        tags=["atom", "execution"],
                        use_cases=["Atomic task execution"],
                        file_path=str(atom_file),
                        status=CapabilityStatus.ACTIVE,
                    )

                    self.registry.register_capability(metadata)

                except Exception as e:
                    error = f"Error analyzing atom {atom_file.name}: {e}"
                    atom_results["errors"].append(error)
                    logger.warning(error)

            atom_results["total"] = len(atom_results["discovered"])

        except Exception as e:
            atom_results["errors"].append(f"Atoms audit failed: {e}")

        return atom_results

    async def _audit_mcp_tools(self) -> dict[str, Any]:
        """Audit MCP tools"""
        mcp_results = {"discovered": [], "errors": [], "total": 0}

        try:
            # Check for MCP server/tools
            mcp_paths = [Path("mcp_server"), Path("src/mcp"), Path("mcp")]

            for mcp_path in mcp_paths:
                if mcp_path.exists():
                    for tool_file in mcp_path.rglob("*.py"):
                        if "tool" in tool_file.name.lower():
                            try:
                                mcp_info = {
                                    "name": tool_file.stem,
                                    "file_path": str(tool_file),
                                    "description": f"MCP Tool: {tool_file.stem}",
                                    "size_bytes": tool_file.stat().st_size,
                                }

                                mcp_results["discovered"].append(mcp_info)

                                # Register in capability registry
                                metadata = CapabilityMetadata(
                                    name=mcp_info["name"],
                                    capability_type=CapabilityType.MCP_TOOL,
                                    description=mcp_info["description"],
                                    tags=["mcp", "external", "tool"],
                                    file_path=str(tool_file),
                                    status=CapabilityStatus.ACTIVE,
                                )

                                self.registry.register_capability(metadata)

                            except Exception as e:
                                error = (
                                    f"Error analyzing MCP tool {tool_file.name}: {e}"
                                )
                                mcp_results["errors"].append(error)

            mcp_results["total"] = len(mcp_results["discovered"])

        except Exception as e:
            mcp_results["errors"].append(f"MCP tools audit failed: {e}")

        return mcp_results

    async def _audit_memory_systems(self) -> dict[str, Any]:
        """Audit memory and storage systems"""
        memory_results = {"discovered": [], "errors": [], "total": 0}

        try:
            # Look for memory-related components
            memory_indicators = [
                "memory",
                "store",
                "semantic",
                "embedding",
                "vector",
                "chroma",
            ]

            # Check plugins for memory systems
            for plugin_file in self.plugins_path.glob("*.py"):
                name_lower = plugin_file.stem.lower()
                if any(indicator in name_lower for indicator in memory_indicators):
                    memory_info = {
                        "name": plugin_file.stem,
                        "type": "plugin_based",
                        "file_path": str(plugin_file),
                        "description": f"Memory system: {plugin_file.stem}",
                    }

                    memory_results["discovered"].append(memory_info)

                    # Register in capability registry
                    metadata = CapabilityMetadata(
                        name=memory_info["name"],
                        capability_type=CapabilityType.MEMORY_SYSTEM,
                        description=memory_info["description"],
                        tags=["memory", "storage", "persistence"],
                        use_cases=["Data storage and retrieval", "Context management"],
                        file_path=str(plugin_file),
                        status=CapabilityStatus.ACTIVE,
                    )

                    self.registry.register_capability(metadata)

            # Check for core memory systems
            core_memory_paths = [
                Path("src/core/neural_store.py"),
                Path("src/core/memory.py"),
                Path("src/storage"),
            ]

            for path in core_memory_paths:
                if path.exists():
                    memory_info = {
                        "name": path.stem if path.is_file() else path.name,
                        "type": "core_system",
                        "file_path": str(path),
                        "description": f"Core memory system: {path.name}",
                    }

                    memory_results["discovered"].append(memory_info)

                    # Register in capability registry
                    metadata = CapabilityMetadata(
                        name=memory_info["name"],
                        capability_type=CapabilityType.MEMORY_SYSTEM,
                        description=memory_info["description"],
                        tags=["memory", "core", "storage"],
                        use_cases=["Core data management"],
                        file_path=str(path),
                        status=CapabilityStatus.ACTIVE,
                    )

                    self.registry.register_capability(metadata)

            memory_results["total"] = len(memory_results["discovered"])

        except Exception as e:
            memory_results["errors"].append(f"Memory systems audit failed: {e}")

        return memory_results

    async def _audit_reasoning_engines(self) -> dict[str, Any]:
        """Audit reasoning and planning engines"""
        reasoning_results = {"discovered": [], "errors": [], "total": 0}

        try:
            # Look for reasoning-related components
            reasoning_indicators = [
                "planner",
                "reasoning",
                "logic",
                "inference",
                "decision",
                "ladder",
                "aog",
                "execution_flow",
                "fsm",
                "state",
            ]

            # Check for reasoning engines
            search_paths = [
                self.plugins_path,
                Path("src/core"),
                Path("src/orchestration"),
                Path("src/script_of_thought"),
            ]

            for search_path in search_paths:
                if not search_path.exists():
                    continue

                for file in search_path.rglob("*.py"):
                    name_lower = file.stem.lower()
                    if any(
                        indicator in name_lower for indicator in reasoning_indicators
                    ):
                        reasoning_info = {
                            "name": file.stem,
                            "type": "reasoning_engine",
                            "file_path": str(file),
                            "description": f"Reasoning engine: {file.stem}",
                            "category": self._categorize_reasoning_engine(file.stem),
                        }

                        reasoning_results["discovered"].append(reasoning_info)

                        # Register in capability registry
                        metadata = CapabilityMetadata(
                            name=reasoning_info["name"],
                            capability_type=CapabilityType.REASONING_ENGINE,
                            description=reasoning_info["description"],
                            tags=["reasoning", "planning", reasoning_info["category"]],
                            use_cases=[
                                "Task planning",
                                "Decision making",
                                "Logical reasoning",
                            ],
                            file_path=str(file),
                            status=CapabilityStatus.ACTIVE,
                        )

                        self.registry.register_capability(metadata)

            reasoning_results["total"] = len(reasoning_results["discovered"])

        except Exception as e:
            reasoning_results["errors"].append(f"Reasoning engines audit failed: {e}")

        return reasoning_results

    def _categorize_reasoning_engine(self, name: str) -> str:
        """Categorize reasoning engine by name"""
        name_lower = name.lower()

        if "planner" in name_lower:
            return "planning"
        elif "fsm" in name_lower or "state" in name_lower:
            return "state_machine"
        elif "execution" in name_lower or "flow" in name_lower:
            return "execution"
        elif "ladder" in name_lower or "aog" in name_lower:
            return "hierarchical"
        elif "sot" in name_lower or "script" in name_lower:
            return "script_based"
        else:
            return "general"

    def _generate_audit_summary(self, audit_results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of audit results"""
        total_capabilities = 0
        total_errors = 0

        for category, results in audit_results.items():
            total_capabilities += results.get("total", 0)
            total_errors += len(results.get("errors", []))

        return {
            "total_capabilities": total_capabilities,
            "total_errors": total_errors,
            "categories": {
                category: results.get("total", 0)
                for category, results in audit_results.items()
            },
            "health_score": (
                max(0, 100 - (total_errors * 10)) if total_capabilities > 0 else 0
            ),
        }

    def _identify_issues(self, audit_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify issues from audit results"""
        issues = []

        # Collect all errors
        for category, results in audit_results.items():
            for error in results.get("errors", []):
                issues.append(
                    {
                        "type": "error",
                        "category": category,
                        "description": error,
                        "severity": "high",
                    }
                )

        # Check for missing critical capabilities
        critical_capabilities = ["conversation", "memory", "planner", "tool_creator"]
        discovered_names = set()

        for category, results in audit_results.items():
            for item in results.get("discovered", []):
                discovered_names.add(item["name"].lower())

        for critical in critical_capabilities:
            if not any(critical in name for name in discovered_names):
                issues.append(
                    {
                        "type": "missing_capability",
                        "category": "critical",
                        "description": f"Critical capability '{critical}' not found",
                        "severity": "high",
                    }
                )

        return issues

    def _generate_recommendations(self, audit_results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on audit"""
        recommendations = []

        summary = audit_results["summary"]
        issues = audit_results["issues"]

        # Recommendations based on issues
        if len(issues) > 0:
            error_count = len([i for i in issues if i["type"] == "error"])
            if error_count > 0:
                recommendations.append(
                    f"Fix {error_count} capability errors to improve system stability"
                )

        # Recommendations based on capability coverage
        if summary["total_capabilities"] < 10:
            recommendations.append(
                "Consider expanding capability coverage - current count is low"
            )

        if summary.get("health_score", 0) < 80:
            recommendations.append(
                "System health score is below 80% - investigate and fix issues"
            )

        # Category-specific recommendations
        categories = summary.get("categories", {})

        if categories.get("dynamic_tools", 0) == 0:
            recommendations.append(
                "No dynamic tools found - consider implementing tool creation capabilities"
            )

        if categories.get("memory_systems", 0) == 0:
            recommendations.append(
                "No memory systems found - implement persistence and context management"
            )

        if categories.get("reasoning_engines", 0) < 2:
            recommendations.append(
                "Limited reasoning capabilities - consider adding more planning engines"
            )

        return recommendations


# Global capability registry instance
capability_registry = CapabilityRegistry()


async def run_capability_audit() -> dict[str, Any]:
    """Convenience function to run a complete capability audit"""
    auditor = CapabilityAuditor(capability_registry)
    return await auditor.audit_all_capabilities()


if __name__ == "__main__":

    async def main():
        print("🔍 Super Alita Capability Audit System")
        print("=" * 50)

        # Run comprehensive audit
        audit_results = await run_capability_audit()

        print("\n📊 Audit Summary:")
        summary = audit_results["summary"]
        print(f"  Total capabilities: {summary['total_capabilities']}")
        print(f"  Total errors: {summary['total_errors']}")
        print(f"  Health score: {summary['health_score']}%")

        print("\n📋 By Category:")
        for category, count in summary["categories"].items():
            print(f"  {category}: {count}")

        if audit_results["issues"]:
            print(f"\n⚠️  Issues Found ({len(audit_results['issues'])}):")
            for issue in audit_results["issues"][:5]:  # Show first 5
                print(f"  - {issue['description']}")

        if audit_results["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in audit_results["recommendations"]:
                print(f"  - {rec}")

        # Export results
        export_path = "capability_audit_report.json"
        if capability_registry.export_capabilities(export_path):
            print(f"\n📄 Full report exported to {export_path}")

    asyncio.run(main())
