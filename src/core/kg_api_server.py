# FastAPI endpoints for Super Alita Knowledge Graph API
# Provides REST endpoints for policy, personality, and consolidation insights

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, Field


# Simple atom creation function for API usage
def create_atom(
    atom_type: str, content: str, confidence: float = 0.8, metadata: dict | None = None
):
    """Create a simple atom structure for knowledge graph storage"""
    return {
        "id": str(uuid.uuid4()),
        "type": atom_type,
        "content": content,
        "confidence": confidence,
        "metadata": metadata or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }


logger = logging.getLogger(__name__)

# FastAPI instance for kg-api endpoints
app = FastAPI(title="Super Alita KG API", version="1.0.0")


# Pydantic models for API requests/responses
class PolicyRequest(BaseModel):
    content: str = Field(..., description="Policy content")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score"
    )
    category: str = Field(default="general", description="Policy category")


class PersonalityRequest(BaseModel):
    risk_tolerance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Risk tolerance"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence threshold"
    )
    learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Learning rate"
    )


class ConsolidationRequest(BaseModel):
    content: str = Field(..., description="Insight content")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score"
    )
    session_count: int = Field(default=1, ge=1, description="Number of sessions")
    insight_strength: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Insight strength"
    )
    patterns: list[str] = Field(default_factory=list, description="Identified patterns")
    related_atom_ids: list[str] = Field(
        default_factory=list, description="Related atom IDs"
    )


class PolicyResponse(BaseModel):
    policies: list[dict[str, Any]]
    count: int


class PersonalityResponse(BaseModel):
    latest: dict[str, Any] | None
    recent: list[dict[str, Any]]
    trend: list[dict[str, Any]]


class ConsolidationResponse(BaseModel):
    consolidations: list[dict[str, Any]]
    count: int


# Global Neo4j driver (would be injected in real implementation)
neo4j_driver = None


async def init_neo4j_driver(uri: str, user: str, password: str):
    """Initialize Neo4j driver"""
    global neo4j_driver
    neo4j_driver = AsyncGraphDatabase.driver(uri, auth=(user, password))


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    # In real implementation, this would read from config
    # await init_neo4j_driver("bolt://localhost:7687", "neo4j", "password")
    logger.info("KG API server starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    global neo4j_driver
    if neo4j_driver:
        await neo4j_driver.close()
    logger.info("KG API server shutting down")


@app.get("/api/kg/policy", response_model=PolicyResponse)
async def get_policy_atoms():
    """Get current policy atoms from knowledge graph"""
    try:
        if not neo4j_driver:
            # Return mock data for testing
            return PolicyResponse(
                policies=[
                    {
                        "id": "policy_001",
                        "content": "Prefer high-confidence decisions in critical situations",
                        "confidence": 0.9,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "category": "decision_making",
                    },
                    {
                        "id": "policy_002",
                        "content": "Learn from hypothesis confirmations and update strategies",
                        "confidence": 0.85,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "category": "learning",
                    },
                ],
                count=2,
            )

        # Query policy atoms from Neo4j
        query = """
        MATCH (a:Atom {type: 'policy'})
        RETURN a.id as id, a.content as content, a.confidence as confidence,
               a.timestamp as timestamp, a.metadata.category as category
        ORDER BY a.timestamp DESC
        LIMIT 50
        """

        result = await neo4j_driver.execute_query(query)
        policy_atoms = [dict(record) for record in result.records]

        return PolicyResponse(policies=policy_atoms, count=len(policy_atoms))

    except Exception as e:
        logger.error(f"Failed to fetch policy atoms: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch policy atoms: {str(e)}"
        ) from e


@app.post("/api/kg/policy")
async def apply_policy_change(policy_data: PolicyRequest):
    """Apply a new policy change to the knowledge graph"""
    try:
        # Create new policy atom
        policy_atom = create_atom(
            atom_type="policy",
            content=policy_data.content,
            confidence=policy_data.confidence,
            metadata={"source": "api", "category": policy_data.category},
        )

        if neo4j_driver:
            # Store in Neo4j
            query = """
            CREATE (a:Atom {
                id: $id,
                type: 'policy',
                content: $content,
                confidence: $confidence,
                timestamp: datetime(),
                metadata: $metadata
            })
            RETURN a.id as id
            """
            await neo4j_driver.execute_query(
                query,
                id=policy_atom.id,
                content=policy_atom.content,
                confidence=policy_atom.confidence,
                metadata=policy_atom.metadata,
            )

        logger.info(f"Applied policy change: {policy_atom.id}")
        return {"success": True, "policy_id": policy_atom.id}

    except Exception as e:
        logger.error(f"Failed to apply policy: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to apply policy: {str(e)}"
        ) from e


@app.get("/api/kg/personality", response_model=PersonalityResponse)
async def get_personality_metrics():
    """Get current personality metrics and trends"""
    try:
        if not neo4j_driver:
            # Return mock data for testing
            return PersonalityResponse(
                latest={
                    "id": "personality_001",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "risk_tolerance": 0.6,
                    "confidence_threshold": 0.75,
                    "learning_rate": 0.12,
                },
                recent=[
                    {
                        "id": "personality_001",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "risk_tolerance": 0.6,
                        "confidence_threshold": 0.75,
                        "learning_rate": 0.12,
                    }
                ],
                trend=[
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "risk_tolerance": 0.6,
                        "confidence_threshold": 0.75,
                        "learning_rate": 0.12,
                    }
                ],
            )

        # Query personality atoms from Neo4j
        query = """
        MATCH (a:Atom {type: 'personality'})
        WITH a
        ORDER BY a.timestamp DESC
        LIMIT 100
        WITH collect(a) as atoms
        RETURN {
            latest: atoms[0],
            recent: atoms[0..10],
            trend: [atom in atoms | {
                timestamp: atom.timestamp,
                risk_tolerance: atom.metadata.risk_tolerance,
                confidence_threshold: atom.metadata.confidence_threshold,
                learning_rate: atom.metadata.learning_rate
            }]
        } as personality_data
        """

        result = await neo4j_driver.execute_query(query)

        if result.records:
            personality_data = result.records[0]["personality_data"]
            return PersonalityResponse(**personality_data)
        else:
            return PersonalityResponse(latest=None, recent=[], trend=[])

    except Exception as e:
        logger.error(f"Failed to fetch personality metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch personality metrics: {str(e)}"
        ) from e


@app.post("/api/kg/personality")
async def update_personality_traits(personality_data: PersonalityRequest):
    """Update agent personality traits"""
    try:
        # Create new personality atom
        personality_atom = create_atom(
            atom_type="personality",
            content=f"Personality update: risk_tolerance={personality_data.risk_tolerance}, confidence_threshold={personality_data.confidence_threshold}, learning_rate={personality_data.learning_rate}",
            confidence=0.9,
            metadata={
                "risk_tolerance": personality_data.risk_tolerance,
                "confidence_threshold": personality_data.confidence_threshold,
                "learning_rate": personality_data.learning_rate,
                "source": "api",
            },
        )

        if neo4j_driver:
            # Store in Neo4j
            query = """
            CREATE (a:Atom {
                id: $id,
                type: 'personality',
                content: $content,
                confidence: $confidence,
                timestamp: datetime(),
                metadata: $metadata
            })
            RETURN a.id as id
            """
            await neo4j_driver.execute_query(
                query,
                id=personality_atom.id,
                content=personality_atom.content,
                confidence=personality_atom.confidence,
                metadata=personality_atom.metadata,
            )

        logger.info(f"Updated personality traits: {personality_atom.id}")
        return {"success": True, "personality_id": personality_atom.id}

    except Exception as e:
        logger.error(f"Failed to update personality: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update personality: {str(e)}"
        )


@app.get("/api/kg/consolidation", response_model=ConsolidationResponse)
async def get_consolidation_insights():
    """Get cross-session insight consolidation data"""
    try:
        if not neo4j_driver:
            # Return mock data for testing
            return ConsolidationResponse(
                consolidations=[
                    {
                        "id": "consolidation_001",
                        "content": "High confidence decisions lead to better outcomes in critical situations",
                        "confidence": 0.88,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "session_count": 5,
                        "insight_strength": 0.75,
                        "related": [],
                    }
                ],
                count=1,
            )

        # Query consolidation atoms and bonds
        query = """
        MATCH (a:Atom {type: 'consolidation'})
        OPTIONAL MATCH (a)-[b:Bond]->(related:Atom)
        WITH a, collect({
            related_id: related.id,
            related_type: related.type,
            bond_strength: b.strength,
            bond_type: b.bond_type
        }) as related_atoms
        RETURN {
            id: a.id,
            content: a.content,
            confidence: a.confidence,
            timestamp: a.timestamp,
            session_count: a.metadata.session_count,
            insight_strength: a.metadata.insight_strength,
            related: related_atoms
        } as consolidation
        ORDER BY a.timestamp DESC
        LIMIT 20
        """

        result = await neo4j_driver.execute_query(query)
        consolidations = [dict(record["consolidation"]) for record in result.records]

        return ConsolidationResponse(
            consolidations=consolidations, count=len(consolidations)
        )

    except Exception as e:
        logger.error(f"Failed to fetch consolidation insights: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch consolidation insights: {str(e)}"
        )


@app.post("/api/kg/consolidation")
async def create_consolidation_insight(insight_data: ConsolidationRequest):
    """Create a new cross-session consolidation insight"""
    try:
        # Create consolidation atom
        consolidation_atom = create_atom(
            atom_type="consolidation",
            content=insight_data.content,
            confidence=insight_data.confidence,
            metadata={
                "session_count": insight_data.session_count,
                "insight_strength": insight_data.insight_strength,
                "patterns": insight_data.patterns,
                "source": "api",
            },
        )

        if neo4j_driver:
            # Store in Neo4j with potential bonds to related atoms
            query = """
            CREATE (a:Atom {
                id: $id,
                type: 'consolidation',
                content: $content,
                confidence: $confidence,
                timestamp: datetime(),
                metadata: $metadata
            })
            RETURN a.id as id
            """
            await neo4j_driver.execute_query(
                query,
                id=consolidation_atom.id,
                content=consolidation_atom.content,
                confidence=consolidation_atom.confidence,
                metadata=consolidation_atom.metadata,
            )

            # Create bonds to related atoms if specified
            for related_id in insight_data.related_atom_ids:
                bond_query = """
                MATCH (a:Atom {id: $consolidation_id})
                MATCH (r:Atom {id: $related_id})
                CREATE (a)-[:Bond {
                    bond_type: 'consolidation',
                    strength: 0.8,
                    timestamp: datetime()
                }]->(r)
                """
                await neo4j_driver.execute_query(
                    bond_query,
                    consolidation_id=consolidation_atom.id,
                    related_id=related_id,
                )

        logger.info(f"Created consolidation insight: {consolidation_atom.id}")
        return {"success": True, "consolidation_id": consolidation_atom.id}

    except Exception as e:
        logger.error(f"Failed to create consolidation insight: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create consolidation insight: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for kg-api service"""
    try:
        if neo4j_driver:
            # Quick Neo4j connectivity test
            query = "RETURN 1 as test"
            await neo4j_driver.execute_query(query)

        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "neo4j_connected": neo4j_driver is not None,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Return metrics in Prometheus format
    metrics = []

    # Example metrics (would be collected from actual system)
    metrics.append("# HELP sa_api_requests_total Total API requests")
    metrics.append("# TYPE sa_api_requests_total counter")
    metrics.append('sa_api_requests_total{endpoint="policy"} 42')
    metrics.append('sa_api_requests_total{endpoint="personality"} 15')
    metrics.append('sa_api_requests_total{endpoint="consolidation"} 8')

    metrics.append("# HELP sa_api_response_time_seconds API response time")
    metrics.append("# TYPE sa_api_response_time_seconds histogram")
    metrics.append('sa_api_response_time_seconds_bucket{endpoint="policy",le="0.1"} 35')
    metrics.append('sa_api_response_time_seconds_bucket{endpoint="policy",le="0.5"} 42')
    metrics.append(
        'sa_api_response_time_seconds_bucket{endpoint="policy",le="+Inf"} 42'
    )

    # **NEW: Concurrency safety metrics**
    metrics.append(
        "# HELP sa_fsm_ignored_triggers_total FSM ignored triggers (during non-accepting states)"
    )
    metrics.append("# TYPE sa_fsm_ignored_triggers_total counter")
    metrics.append("sa_fsm_ignored_triggers_total 23")

    metrics.append(
        "# HELP sa_fsm_stale_completions_total FSM stale completions (operation ID mismatch)"
    )
    metrics.append("# TYPE sa_fsm_stale_completions_total counter")
    metrics.append("sa_fsm_stale_completions_total 7")

    metrics.append("# HELP sa_fsm_mailbox_size Current FSM mailbox size")
    metrics.append("# TYPE sa_fsm_mailbox_size gauge")
    metrics.append("sa_fsm_mailbox_size 2")

    metrics.append("# HELP sa_fsm_mailbox_size_max Maximum FSM mailbox size reached")
    metrics.append("# TYPE sa_fsm_mailbox_size_max gauge")
    metrics.append("sa_fsm_mailbox_size_max 12")

    metrics.append("# HELP sa_fsm_active_operations Current active operations")
    metrics.append("# TYPE sa_fsm_active_operations gauge")
    metrics.append("sa_fsm_active_operations 1")

    metrics.append("# HELP sa_fsm_operations_total Total operations registered")
    metrics.append("# TYPE sa_fsm_operations_total counter")
    metrics.append("sa_fsm_operations_total 145")

    return "\n".join(metrics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
