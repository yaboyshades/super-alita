# scripts/knowledge_base/neural_indexer.py
"""
Neural Code Indexer using LibCST for precise AST analysis.
Extracts code entities, relationships, and neural atoms from Python codebases.
"""

import asyncio
from pathlib import Path

import libcst as cst
from libcst.metadata import PositionProvider

from ..common import Project
from ..sdd_models import (
    ClassEntity,
    CodeEntity,
    CodeEntityType,
    FunctionEntity,
    NeuralAtom,
)


class LibCSTEntityExtractor(cst.CSTVisitor):
    """
    LibCST visitor that extracts code entities with precise metadata.
    Preserves whitespace, comments, and exact source locations.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_code = source_code
        self.entities: list[CodeEntity] = []
        self.scope_stack: list[str] = []
        self.call_graph: dict[str, set[str]] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Extract class definitions with inheritance and metadata."""
        pos = self.get_metadata(PositionProvider, node)
        class_name = node.name.value

        # Build unique ID from current scope
        full_name = ".".join(self.scope_stack + [class_name])
        entity_id = f"{self.file_path}::{CodeEntityType.CLASS}::{full_name}"

        # Extract inheritance using LibCST's precise parsing
        inheritance = []
        for base in node.bases:
            if isinstance(base.value, cst.Name):
                inheritance.append(base.value.value)
            elif isinstance(base.value, cst.Attribute):
                # Handle qualified names like module.Class
                inheritance.append(self._extract_attribute_name(base.value))

        class_entity = ClassEntity(
            id=entity_id,
            name=class_name,
            file_path=self.file_path,
            start_line=pos.start.line,
            end_line=pos.end.line,
            docstring=self._get_docstring(node),
            inheritance=inheritance,
        )
        self.entities.append(class_entity)

        # Manage scope for nested classes/methods
        self.scope_stack.append(class_name)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:  # noqa: ARG002
        """Exit class scope."""
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """Extract function/method definitions with signatures and decorators."""
        pos = self.get_metadata(PositionProvider, node)
        func_name = node.name.value

        is_method = bool(self.scope_stack)
        node_type = CodeEntityType.METHOD if is_method else CodeEntityType.FUNCTION

        full_name = ".".join(self.scope_stack + [func_name])
        entity_id = f"{self.file_path}::{node_type}::{full_name}"

        # Precisely extract function signature using LibCST
        signature = self._extract_signature(node)

        # Extract decorators
        decorators = []
        for dec in node.decorators:
            if isinstance(dec.decorator, cst.Name):
                decorators.append(f"@{dec.decorator.value}")
            elif isinstance(dec.decorator, cst.Attribute):
                decorators.append(f"@{self._extract_attribute_name(dec.decorator)}")

        func_entity = FunctionEntity(
            id=entity_id,
            name=func_name,
            file_path=self.file_path,
            start_line=pos.start.line,
            end_line=pos.end.line,
            docstring=self._get_docstring(node),
            is_async=node.asynchronous is not None,
            signature=signature,
            decorators=decorators,
        )
        self.entities.append(func_entity)

        # Return False to prevent visiting function body for now
        # We'll do separate passes for call graph analysis
        return False

    def visit_Call(self, node: cst.Call) -> None:
        """Extract function calls for building call graph."""
        if not self.scope_stack:
            return  # Only track calls within functions/methods

        caller_name = ".".join(self.scope_stack)
        caller_id = f"{self.file_path}::function::{caller_name}"

        # Extract callee name
        if isinstance(node.func, cst.Name):
            callee_name = node.func.value
            callee_id = f"{self.file_path}::function::{callee_name}"
            self.call_graph.setdefault(caller_id, set()).add(callee_id)
        elif isinstance(node.func, cst.Attribute):
            # Handle method calls like obj.method()
            attr_name = self._extract_attribute_name(node.func)
            # For simplicity, we'll track the full attribute name
            # In a full implementation, you'd resolve the actual function
            callee_id = f"{self.file_path}::function::{attr_name}"
            self.call_graph.setdefault(caller_id, set()).add(callee_id)

    def _extract_signature(self, node: cst.FunctionDef) -> str:
        """Extract function signature as string."""
        # Use LibCST's precise source code extraction
        params_start = node.params.get_start_offset()
        params_end = node.params.get_end_offset()
        return self.source_code[params_start:params_end].strip()

    def _extract_attribute_name(self, node: cst.Attribute) -> str:
        """Extract full attribute name (e.g., 'module.Class.method')."""
        parts = []
        current = node
        while isinstance(current, cst.Attribute):
            parts.insert(0, current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.insert(0, current.value)
        return ".".join(parts)

    def _get_docstring(self, node: cst.ClassDef | cst.FunctionDef) -> str | None:
        """Extract docstring using LibCST's docstring detection."""
        docstring_node = cst.parse_expression(node.get_docstring() or "None")
        if isinstance(docstring_node, cst.SimpleString):
            return docstring_node.value.strip("\"'")
        return None


class NeuralCodeIndexer:
    """
    Main indexer that orchestrates the neural indexing process.
    Uses LibCST for precise code analysis and builds knowledge graphs.
    """

    def __init__(self, db_client):
        """Initialize with database client for persistence."""
        self.db_client = db_client
        self.project: Project | None = None

    async def index_project_with_evolution(self, project_path: Path) -> dict[str, any]:
        """
        Main indexing entry point with evolutionary capabilities.
        Returns metrics and generated artifacts.
        """
        self.project = Project(root=project_path)

        # Phase 1: Extract code entities using LibCST
        entities = await self._extract_code_entities(project_path)

        # Phase 2: Build relationships and call graphs
        await self._build_relationships(entities)

        # Phase 3: Generate neural atoms from code analysis
        neural_atoms = await self._generate_neural_atoms(entities)

        # Phase 4: Persist to knowledge graph
        await self._persist_to_graph(entities)

        # Phase 5: Evolutionary analysis (generate new capabilities)
        generated_mcps = await self._evolutionary_analysis(entities)

        return {
            "metrics": {
                "files_processed": len({e.file_path for e in entities}),
                "entities_created": len(entities),
                "neural_atoms_generated": len(neural_atoms),
            },
            "entities": entities,
            "neural_atoms": neural_atoms,
            "generated_mcps": generated_mcps,
        }

    async def _extract_code_entities(self, project_path: Path) -> list[CodeEntity]:
        """Extract all code entities from Python files using LibCST."""
        entities = []

        # Find all Python files
        python_files = list(project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                # Read source code
                source_code = file_path.read_text(encoding="utf-8")

                # Parse with LibCST
                tree = cst.parse_module(source_code)

                # Extract entities
                extractor = LibCSTEntityExtractor(str(file_path), source_code)
                tree.visit(extractor)

                entities.extend(extractor.entities)

            except Exception as e:
                # Log parsing errors but continue
                print(f"Error parsing {file_path}: {e}")
                continue

        return entities

    async def _build_relationships(self, entities: list[CodeEntity]) -> None:
        """Build relationships between code entities."""
        # This would analyze imports, inheritance, calls, etc.
        # For now, we'll focus on the basic structure
        pass

    async def _generate_neural_atoms(
        self, entities: list[CodeEntity]
    ) -> list[NeuralAtom]:
        """Generate neural atoms from code analysis."""
        atoms = []

        for entity in entities:
            # Generate atoms based on entity characteristics
            if isinstance(entity, FunctionEntity):
                # Analyze function complexity, patterns, etc.
                atom = NeuralAtom(
                    id=f"atom_{entity.id}_complexity",
                    content=f"Function {entity.name} has "
                    f"{len(entity.parameters)} parameters",
                    confidence=0.8,
                    source="code_analysis",
                )
                atoms.append(atom)

            elif isinstance(entity, ClassEntity):
                # Analyze class design patterns
                atom = NeuralAtom(
                    id=f"atom_{entity.id}_design",
                    content=f"Class {entity.name} inherits from "
                    f"{entity.inheritance}",
                    confidence=0.9,
                    source="code_analysis",
                )
                atoms.append(atom)

        return atoms

    async def _persist_to_graph(self, entities: list[CodeEntity]) -> None:
        """Persist entities and atoms to the knowledge graph."""
        # Group entities by file for efficient batch operations
        file_groups = {}
        for entity in entities:
            file_groups.setdefault(entity.file_path, []).append(entity)

        # Persist each file's entities
        for file_path, file_entities in file_groups.items():
            await asyncio.to_thread(
                self.db_client.update_file_graph, file_path, file_entities
            )

        # Persist neural atoms (would need additional DB methods)
        # await self._persist_neural_atoms(neural_atoms)

    async def _evolutionary_analysis(
        self, entities: list[CodeEntity]
    ) -> list[dict[str, any]]:
        """
        Analyze codebase for potential new MCP capabilities.
        This is where Super Alita evolves new tools based on code patterns.
        """
        generated_mcps = []

        # Example: Detect common patterns that could become tools
        # This is a simplified version - real implementation would be
        # much more sophisticated

        # Look for repeated utility functions
        utility_functions = [
            e
            for e in entities
            if isinstance(e, FunctionEntity)
            and not e.decorators  # Not already a tool
            and len(e.calls) > 3  # Called from multiple places
        ]

        for func in utility_functions[:3]:  # Limit for demo
            mcp_spec = {
                "name": f"auto_{func.name}",
                "description": f"Auto-generated tool from {func.name}",
                "function_id": func.id,
                "confidence": 0.7,
            }
            generated_mcps.append(mcp_spec)

        return generated_mcps

    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during indexing."""
        # Skip common directories
        skip_dirs = {".git", "__pycache__", ".venv", "node_modules", "build", "dist"}
        return any(part in skip_dirs for part in file_path.parts)
