# scripts/knowledge_base/graph_db_client.py
"""
Neo4j Graph Database Client for Knowledge Graph Operations.
Handles transactional persistence of code entities and relationships.
"""

from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction
from neo4j.exceptions import Neo4jError

from ..sdd_models import ClassEntity, CodeEntity, FunctionEntity, ModuleEntity


class GraphDatabaseClient:
    """
    Async Neo4j client for knowledge graph operations.
    Manages connections, transactions, and Cypher queries.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection parameters."""
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.user, self.password), database=self.database
            )
            # Test connection
            await self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        """Context manager for database sessions."""
        if not self.driver:
            raise ConnectionError("Database not connected")

        session = self.driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query with parameters.
        Returns list of result records as dictionaries.
        """
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> None:
        """Execute a write query within a transaction."""
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                await tx.run(query, parameters or {})
                await tx.commit()

    async def create_schema(self) -> None:
        """
        Create database schema with constraints and indexes.
        This should be called once during initialization.
        """
        schema_queries = [
            # Create constraints for unique IDs
            "CREATE CONSTRAINT code_entity_id IF NOT EXISTS "
            "FOR (ce:CodeEntity) REQUIRE ce.id IS UNIQUE",
            # Create indexes for performance
            "CREATE INDEX code_entity_file IF NOT EXISTS "
            "FOR (ce:CodeEntity) ON (ce.file_path)",
            "CREATE INDEX code_entity_type IF NOT EXISTS "
            "FOR (ce:CodeEntity) ON (ce.entity_type)",
            # Create relationship indexes
            "CREATE INDEX calls_relationship IF NOT EXISTS "
            "FOR ()-[r:CALLS]-() ON (r.line_number)",
            "CREATE INDEX inherits_relationship IF NOT EXISTS "
            "FOR ()-[r:INHERITS_FROM]-() ON (r.inheritance_order)",
        ]

        for query in schema_queries:
            try:
                await self.execute_write_query(query)
            except Neo4jError as e:
                # Constraint/index might already exist
                print(f"Schema creation note: {e}")

    async def clear_database(self) -> None:
        """Clear all data from the database (for testing/reset)."""
        await self.execute_write_query("MATCH (n) DETACH DELETE n")

    async def update_file_graph(
        self, file_path: str, entities: list[CodeEntity]
    ) -> None:
        """
        Update the knowledge graph for a specific file.
        Uses transactions for atomicity and performance.
        """
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                # Remove existing entities for this file
                await tx.run(
                    "MATCH (ce:CodeEntity {file_path: $file_path}) " "DETACH DELETE ce",
                    file_path=file_path,
                )

                # Create new entities
                for entity in entities:
                    await self._create_entity_in_tx(tx, entity)

                # Create relationships within this file
                await self._create_file_relationships_in_tx(tx, entities)

                await tx.commit()

    async def _create_entity_in_tx(
        self, tx: AsyncTransaction, entity: CodeEntity
    ) -> None:
        """Create a code entity node within a transaction."""
        if isinstance(entity, FunctionEntity):
            await tx.run(
                """
                CREATE (ce:CodeEntity:Function {
                    id: $id,
                    name: $name,
                    file_path: $file_path,
                    start_line: $start_line,
                    end_line: $end_line,
                    docstring: $docstring,
                    is_async: $is_async,
                    signature: $signature,
                    entity_type: 'function'
                })
                """,
                id=entity.id,
                name=entity.name,
                file_path=entity.file_path,
                start_line=entity.start_line,
                end_line=entity.end_line,
                docstring=entity.docstring,
                is_async=entity.is_async,
                signature=entity.signature,
            )

            # Add decorators as separate nodes
            for decorator in entity.decorators:
                await tx.run(
                    """
                    MATCH (ce:CodeEntity {id: $entity_id})
                    MERGE (d:Decorator {name: $decorator_name})
                    CREATE (ce)-[:HAS_DECORATOR]->(d)
                    """,
                    entity_id=entity.id,
                    decorator_name=decorator,
                )

        elif isinstance(entity, ClassEntity):
            await tx.run(
                """
                CREATE (ce:CodeEntity:Class {
                    id: $id,
                    name: $name,
                    file_path: $file_path,
                    start_line: $start_line,
                    end_line: $end_line,
                    docstring: $docstring,
                    entity_type: 'class'
                })
                """,
                id=entity.id,
                name=entity.name,
                file_path=entity.file_path,
                start_line=entity.start_line,
                end_line=entity.end_line,
                docstring=entity.docstring,
            )

            # Add inheritance relationships
            for base_class in entity.inheritance:
                await tx.run(
                    """
                    MATCH (ce:CodeEntity {id: $entity_id})
                    MERGE (base:Class {name: $base_name})
                    CREATE (ce)-[:INHERITS_FROM {inheritance_order: $order}]->(base)
                    """,
                    entity_id=entity.id,
                    base_name=base_class,
                    order=entity.inheritance.index(base_class),
                )

        elif isinstance(entity, ModuleEntity):
            await tx.run(
                """
                CREATE (ce:CodeEntity:Module {
                    id: $id,
                    name: $name,
                    file_path: $file_path,
                    docstring: $docstring,
                    entity_type: 'module'
                })
                """,
                id=entity.id,
                name=entity.name,
                file_path=entity.file_path,
                docstring=entity.docstring,
            )

    async def _create_file_relationships_in_tx(
        self, tx: AsyncTransaction, entities: list[CodeEntity]
    ) -> None:
        """Create relationships between entities within a file."""
        # Create CONTAINS relationships (classes contain methods)
        for entity in entities:
            if isinstance(entity, FunctionEntity) and entity.id.count("::") > 2:
                # This is a method (has class scope)
                class_id = (
                    "::".join(entity.id.split("::")[:-2])
                    + "::class::"
                    + "::".join(entity.id.split("::")[-3].split("."))
                )

                await tx.run(
                    """
                    MATCH (class:CodeEntity {id: $class_id})
                    MATCH (method:CodeEntity {id: $method_id})
                    CREATE (class)-[:CONTAINS]->(method)
                    """,
                    class_id=class_id,
                    method_id=entity.id,
                )

    async def create_cross_file_relationships(
        self, relationships: list[dict[str, Any]]
    ) -> None:
        """
        Create relationships that span multiple files (imports, calls).
        Expects list of relationship dictionaries with keys:
        - from_id: source entity ID
        - to_id: target entity ID
        - type: relationship type (CALLS, IMPORTS, etc.)
        - metadata: additional properties
        """
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                for rel in relationships:
                    await tx.run(
                        f"""
                        MATCH (from_entity:CodeEntity {{id: $from_id}})
                        MATCH (to_entity:CodeEntity {{id: $to_id}})
                        CREATE (from_entity)-[:{rel['type']} $metadata]->(to_entity)
                        """,
                        from_id=rel["from_id"],
                        to_id=rel["to_id"],
                        metadata=rel.get("metadata", {}),
                    )
                await tx.commit()

    async def query_entities_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Query all entities in a specific file."""
        return await self.execute_query(
            """
            MATCH (ce:CodeEntity {file_path: $file_path})
            RETURN ce
            ORDER BY ce.start_line
            """,
            {"file_path": file_path},
        )

    async def query_entity_relationships(
        self, entity_id: str, relationship_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Query relationships for a specific entity."""
        if relationship_types:
            types_clause = f"r:{'|'.join(relationship_types)}"
        else:
            types_clause = "r"

        return await self.execute_query(
            f"""
            MATCH (ce:CodeEntity {{id: $entity_id}})-[{types_clause}]-(other)
            RETURN type(r) as relationship_type, other, r
            """,
            {"entity_id": entity_id},
        )

    async def find_similar_functions(
        self, signature_pattern: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find functions with similar signatures."""
        return await self.execute_query(
            """
            MATCH (f:Function)
            WHERE f.signature =~ $pattern
            RETURN f
            ORDER BY f.name
            LIMIT $limit
            """,
            {"pattern": signature_pattern, "limit": limit},
        )

    async def get_code_metrics(self) -> dict[str, Any]:
        """Get overall codebase metrics."""
        queries = {
            "total_entities": "MATCH (ce:CodeEntity) RETURN count(ce) as count",
            "entity_types": """
                MATCH (ce:CodeEntity)
                RETURN ce.entity_type as type, count(*) as count
                ORDER BY count DESC
            """,
            "relationship_count": "MATCH ()-[r]-() RETURN count(r) as count",
            "files_count": """
                MATCH (ce:CodeEntity)
                RETURN count(DISTINCT ce.file_path) as count
            """,
        }

        metrics = {}
        for key, query in queries.items():
            result = await self.execute_query(query)
            if result:
                if key in ["total_entities", "relationship_count", "files_count"]:
                    metrics[key] = result[0]["count"]
                else:
                    metrics[key] = result

        return metrics

    async def search_entities(
        self, query: str, entity_types: list[str] | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Search entities by name or content."""
        type_filter = ""
        if entity_types:
            type_filter = f"AND ce.entity_type IN {entity_types}"

        return await self.execute_query(
            f"""
            MATCH (ce:CodeEntity)
            WHERE (ce.name =~ $query OR ce.docstring =~ $query) {type_filter}
            RETURN ce
            ORDER BY ce.name
            LIMIT $limit
            """,
            {"query": f".*{query}.*", "limit": limit},
        )
