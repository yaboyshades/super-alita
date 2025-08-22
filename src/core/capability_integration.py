#!/usr/bin/env python3
"""
Integration module to wire Capability Audit into Super Alita main system
"""

import logging

from src.core.capability_audit import capability_registry, run_capability_audit

logger = logging.getLogger(__name__)


async def initialize_capability_system():
    """Initialize the capability system as part of Super Alita startup"""
    logger.info("Initializing Super Alita Capability Audit System...")

    try:
        # Run initial audit
        audit_results = await run_capability_audit()

        summary = audit_results["summary"]
        logger.info(
            f"Capability audit complete: {summary['total_capabilities']} capabilities, "
            f"{summary['total_errors']} errors, {summary['health_score']}% health"
        )

        # Log issues if any
        if audit_results["issues"]:
            logger.warning(f"Found {len(audit_results['issues'])} capability issues")
            for issue in audit_results["issues"][:3]:  # Log first 3
                logger.warning(f"Issue: {issue['description']}")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize capability system: {e}")
        return False


def get_capability_summary() -> dict:
    """Get a quick summary of system capabilities"""
    stats = capability_registry.get_capability_stats()
    return {
        "total_capabilities": stats["total_capabilities"],
        "by_type": stats["by_type"],
        "health_score": stats.get("health_score", 0),
    }


def search_capabilities_for_task(task_description: str) -> list:
    """Search for capabilities that might help with a specific task"""
    # Split task description into keywords
    keywords = task_description.lower().split()

    # Search for each keyword
    all_matches = set()
    for keyword in keywords:
        matches = capability_registry.search_capabilities(keyword)
        all_matches.update(cap.name for cap in matches)

    # Return capability details
    return [
        {"name": cap_name, "capability": capability_registry.get_capability(cap_name)}
        for cap_name in all_matches
        if capability_registry.get_capability(cap_name)
    ]


def get_missing_capabilities() -> list:
    """Get list of critical capabilities that are missing"""
    # Critical capabilities every agent should have
    critical_capabilities = [
        "conversation",
        "memory",
        "planner",
        "tool_creator",
        "reasoning",
        "execution",
    ]

    # Get all capability names (lowercase for comparison)
    existing_names = [
        cap.name.lower() for cap in capability_registry.list_capabilities()
    ]

    # Find missing capabilities
    missing = []
    for critical in critical_capabilities:
        if not any(critical in name for name in existing_names):
            missing.append(critical)

    return missing


def recommend_capability_improvements() -> list:
    """Get recommendations for improving the capability ecosystem"""
    recommendations = []
    stats = capability_registry.get_capability_stats()

    # Check capability counts
    total = stats["total_capabilities"]
    by_type = stats.get("by_type", {})

    if total < 50:
        recommendations.append(
            "Low capability count - consider expanding plugin ecosystem"
        )

    if by_type.get("dynamic_tool", 0) < 5:
        recommendations.append(
            "Few dynamic tools - implement more runtime tool creation"
        )

    if by_type.get("memory_system", 0) < 2:
        recommendations.append(
            "Limited memory systems - consider adding more persistence layers"
        )

    if by_type.get("reasoning_engine", 0) < 3:
        recommendations.append(
            "Limited reasoning engines - add more planning and inference systems"
        )

    # Check for error capabilities
    error_caps = [
        cap for cap in capability_registry.list_capabilities() if cap.error_message
    ]

    if error_caps:
        recommendations.append(
            f"Fix {len(error_caps)} capabilities with errors to improve stability"
        )

    return recommendations


async def audit_and_report() -> dict:
    """Convenience function to audit capabilities and return a formatted report"""
    try:
        # Run audit
        audit_results = await run_capability_audit()

        # Get additional insights
        missing = get_missing_capabilities()
        improvements = recommend_capability_improvements()

        return {
            "audit_results": audit_results,
            "missing_capabilities": missing,
            "improvement_recommendations": improvements,
            "quick_stats": get_capability_summary(),
        }

    except Exception as e:
        logger.error(f"Audit and report failed: {e}")
        return {
            "error": str(e),
            "audit_results": None,
            "missing_capabilities": [],
            "improvement_recommendations": ["Fix audit system errors"],
            "quick_stats": {"total_capabilities": 0, "by_type": {}, "health_score": 0},
        }


# Export the main integration functions
__all__ = [
    "initialize_capability_system",
    "get_capability_summary",
    "search_capabilities_for_task",
    "get_missing_capabilities",
    "recommend_capability_improvements",
    "audit_and_report",
    "capability_registry",  # Direct access to registry if needed
]
