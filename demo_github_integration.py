#!/usr/bin/env python3
"""
Demonstration of GitHub integration capabilities in Super Alita cognitive agent.

This script shows how the enhanced GitHub integration features work,
including event processing, priority calculation, and API integration.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Import our GitHub integration components
import sys
sys.path.append('./src')

from src.core.schemas import (
    GitHubEventSchema,
    GitHubEventType,
    GitHubPriorityMetrics,
    AttentionLevel
)
from src.tools.github_cli_tool import GitHubCliTool, GitHubCliInput
from src.integration.github_api import GitHubApiClient
from src.core.github_priority_calculator import GitHubPriorityCalculator, EnhancedPriorityMetrics


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "="*60)
    print(f"🧠 {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * 40)


async def demo_github_event_processing():
    """Demonstrate GitHub event processing."""
    print_banner("GitHub Event Processing Demo")
    
    # Create sample GitHub events
    events = [
        GitHubEventSchema(
            event_type=GitHubEventType.ISSUE_CREATED,
            repository="super-alita/cognitive-agent",
            actor="developer",
            payload={
                "issue_number": 123,
                "title": "Critical security vulnerability",
                "labels": [{"name": "security"}, {"name": "critical"}],
                "body": "Found a critical security issue that needs @team review ASAP"
            },
            event_id="issue-123-created"
        ),
        GitHubEventSchema(
            event_type=GitHubEventType.PR_OPENED,
            repository="super-alita/cognitive-agent",
            actor="contributor",
            payload={
                "pr_number": 456,
                "title": "Add GitHub integration features",
                "labels": [{"name": "enhancement"}],
                "comments": 3,
                "changed_files": 8,
                "additions": 150,
                "deletions": 20
            },
            event_id="pr-456-opened"
        ),
        GitHubEventSchema(
            event_type=GitHubEventType.WORKFLOW_RUN,
            repository="super-alita/cognitive-agent",
            actor="github-actions[bot]",
            payload={
                "workflow_name": "CI/CD Pipeline",
                "status": "failure", 
                "conclusion": "failure"
            },
            event_id="workflow-run-789",
            attention_level=AttentionLevel.HIGH
        )
    ]
    
    print_section("Processing GitHub Events")
    
    for i, event in enumerate(events, 1):
        print(f"\n{i}. {event.event_type.value.replace('_', ' ').title()}")
        print(f"   Repository: {event.repository}")
        print(f"   Actor: {event.actor}")
        print(f"   Attention Level: {event.attention_level.value}")
        print(f"   Event ID: {event.event_id}")
        
        # Show key payload data
        if event.event_type == GitHubEventType.ISSUE_CREATED:
            print(f"   Issue #{event.payload['issue_number']}: {event.payload['title']}")
            labels = [label['name'] for label in event.payload['labels']]
            print(f"   Labels: {', '.join(labels)}")
        elif event.event_type == GitHubEventType.PR_OPENED:
            print(f"   PR #{event.payload['pr_number']}: {event.payload['title']}")
            print(f"   Changes: +{event.payload['additions']} -{event.payload['deletions']} lines")
        elif event.event_type == GitHubEventType.WORKFLOW_RUN:
            print(f"   Workflow: {event.payload['workflow_name']}")
            print(f"   Status: {event.payload['status']} - {event.payload['conclusion']}")
    
    return events


async def demo_priority_calculation():
    """Demonstrate GitHub-enhanced priority calculation."""
    print_banner("GitHub-Enhanced Priority Calculation Demo")
    
    calculator = GitHubPriorityCalculator()
    
    # Create sample scenarios
    scenarios = [
        {
            "name": "Critical Security Issue",
            "metrics": EnhancedPriorityMetrics(
                impact=9.0,
                urgency=9.0,
                effort=3.0,
                github_metrics=GitHubPriorityMetrics(
                    has_security_alert=True,
                    has_stakeholder_mention=True,
                    issue_labels=["security", "critical"],
                    comment_count=5
                ),
                age_hours=2.0,
                deadline_hours=12.0
            )
        },
        {
            "name": "Blocking Pull Request",
            "metrics": EnhancedPriorityMetrics(
                impact=7.0,
                urgency=6.0,
                effort=4.0,
                github_metrics=GitHubPriorityMetrics(
                    blocks_other_prs=True,
                    merge_conflicts=True,
                    file_changes_count=15,
                    lines_changed=300,
                    ci_status="failure"
                ),
                age_hours=24.0
            )
        },
        {
            "name": "Regular Enhancement",
            "metrics": EnhancedPriorityMetrics(
                impact=5.0,
                urgency=4.0,
                effort=6.0,
                github_metrics=GitHubPriorityMetrics(
                    comment_count=2,
                    issue_labels=["enhancement"],
                    file_changes_count=3,
                    lines_changed=50
                ),
                age_hours=48.0
            )
        }
    ]
    
    print_section("Priority Calculation Results")
    
    for i, scenario in enumerate(scenarios, 1):
        metrics = scenario["metrics"]
        priority = calculator.calculate_priority(metrics)
        explanation = calculator.get_priority_explanation(metrics, priority)
        
        print(f"\n{i}. {scenario['name']}")
        print(f"   Priority Score: {priority:.2f}")
        print(f"   Category: {explanation['priority_category']}")
        print(f"   Base Priority: {explanation['base_priority']:.2f}")
        print(f"   GitHub Adjustment: {explanation['github_adjustment']:.2f}x")
        print(f"   Temporal Adjustment: {explanation['temporal_adjustment']:.2f}x")
        
        if explanation['contributing_factors']:
            print("   Contributing Factors:")
            for factor in explanation['contributing_factors']:
                print(f"     • {factor}")
        
        if metrics.github_metrics:
            print("   GitHub Metrics:")
            if metrics.github_metrics.has_security_alert:
                print("     • Security alert detected")
            if metrics.github_metrics.blocks_other_prs:
                print("     • Blocks other PRs")
            if metrics.github_metrics.merge_conflicts:
                print("     • Has merge conflicts")
            if metrics.github_metrics.ci_status == "failure":
                print("     • CI pipeline failing")


async def demo_github_cli_tool():
    """Demonstrate GitHub CLI tool integration."""
    print_banner("GitHub CLI Tool Integration Demo")
    
    tool = GitHubCliTool()
    
    # Test various GitHub CLI commands in dry-run mode
    commands = [
        "gh issue create --title 'New bug report' --body 'Found an issue' --label bug",
        "gh pr create --title 'Fix critical bug' --body 'This fixes the critical bug' --head feature/fix",
        "gh pr merge 123 --squash",
        "gh issue list --state open --label critical",
        "gh workflow run ci.yml",
        "invalid command",  # This should fail validation
        "gh issue list; rm -rf /"  # This should fail security validation
    ]
    
    print_section("GitHub CLI Command Execution (Dry-Run Mode)")
    
    for i, command in enumerate(commands, 1):
        print(f"\n{i}. Command: {command}")
        
        input_data = GitHubCliInput(
            command=command,
            dry_run=True,
            repository="super-alita/cognitive-agent"
        )
        
        result = await tool.execute(input_data)
        
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Output: {result.output}")
            if result.github_event:
                print(f"   Generated Event: {result.github_event.event_type.value}")
        else:
            print(f"   Error: {result.error}")
        print(f"   Execution Time: {result.execution_time:.4f}s")


async def demo_github_api_integration():
    """Demonstrate GitHub API integration (mock mode)."""
    print_banner("GitHub API Integration Demo")
    
    print_section("GitHub API Client Features")
    
    # Initialize client (without token for demo)
    client = GitHubApiClient(token=None)
    
    print("✅ GitHub API Client initialized")
    print(f"   Base URL: {client.base_url}")
    print(f"   Rate Limit Buffer: {client.rate_limit_buffer}")
    print(f"   Timeout: {client.timeout}s")
    
    # Show API request structure (won't actually make requests without token)
    from src.core.schemas import GitHubApiRequest
    
    sample_requests = [
        GitHubApiRequest(
            endpoint="repos/super-alita/cognitive-agent/issues",
            method="GET",
            parameters={"state": "open", "labels": "bug,critical"}
        ),
        GitHubApiRequest(
            endpoint="repos/super-alita/cognitive-agent/pulls/123",
            method="GET"
        ),
        GitHubApiRequest(
            endpoint="repos/super-alita/cognitive-agent/actions/runs",
            method="GET",
            parameters={"status": "failure"}
        )
    ]
    
    print("\n📋 Sample API Requests:")
    for i, request in enumerate(sample_requests, 1):
        print(f"\n{i}. {request.method} {request.endpoint}")
        if request.parameters:
            print(f"   Parameters: {json.dumps(request.parameters, indent=6)}")
        print(f"   Rate Limit Aware: {request.rate_limit_aware}")
    
    # Demonstrate priority metrics extraction (mock)
    print_section("Priority Metrics Extraction")
    
    mock_metrics = GitHubPriorityMetrics(
        has_security_alert=True,
        blocks_other_prs=False,
        has_stakeholder_mention=True,
        ci_status="failure",
        review_count=3,
        comment_count=8,
        file_changes_count=12,
        lines_changed=245,
        issue_labels=["bug", "critical", "security"],
        merge_conflicts=True
    )
    
    print("📊 Sample Priority Metrics (extracted from GitHub API):")
    print(f"   Security Alert: {mock_metrics.has_security_alert}")
    print(f"   Blocks Other PRs: {mock_metrics.blocks_other_prs}")
    print(f"   Stakeholder Mention: {mock_metrics.has_stakeholder_mention}")
    print(f"   CI Status: {mock_metrics.ci_status}")
    print(f"   Comments: {mock_metrics.comment_count}")
    print(f"   Reviews: {mock_metrics.review_count}")
    print(f"   Files Changed: {mock_metrics.file_changes_count}")
    print(f"   Lines Changed: {mock_metrics.lines_changed}")
    print(f"   Labels: {', '.join(mock_metrics.issue_labels)}")
    print(f"   Merge Conflicts: {mock_metrics.merge_conflicts}")


async def demo_integration_workflow():
    """Demonstrate end-to-end GitHub integration workflow."""
    print_banner("End-to-End GitHub Integration Workflow")
    
    print_section("Simulated GitHub Workflow")
    
    # Step 1: GitHub event occurs
    print("1. 📨 GitHub Event: Pull Request Opened")
    
    event = GitHubEventSchema(
        event_type=GitHubEventType.PR_OPENED,
        repository="super-alita/cognitive-agent",
        actor="developer",
        payload={
            "pr_number": 789,
            "title": "Critical security fix",
            "labels": [{"name": "security"}, {"name": "hotfix"}],
            "body": "Fixes critical vulnerability. Needs @security-team review urgently.",
            "changed_files": 5,
            "additions": 50,
            "deletions": 10,
            "comments": 0
        },
        event_id="workflow-demo-pr-789"
    )
    
    print(f"   PR #{event.payload['pr_number']}: {event.payload['title']}")
    print(f"   Actor: {event.actor}")
    
    # Step 2: Extract priority metrics from event
    print("\n2. 🔍 Extracting Priority Metrics")
    
    calculator = GitHubPriorityCalculator()
    priority_metrics = calculator.create_priority_metrics_from_github_event(event)
    
    print(f"   Base Impact: {priority_metrics.impact}")
    print(f"   Base Urgency: {priority_metrics.urgency}")
    print(f"   GitHub Metrics: {'✅ Available' if priority_metrics.github_metrics else '❌ None'}")
    
    # Step 3: Calculate enhanced priority
    print("\n3. 📊 Calculating Enhanced Priority")
    
    # Enhance with additional GitHub metrics
    if priority_metrics.github_metrics:
        priority_metrics.github_metrics.has_security_alert = True
        priority_metrics.github_metrics.has_stakeholder_mention = True
        priority_metrics.github_metrics.file_changes_count = event.payload["changed_files"]
        priority_metrics.github_metrics.lines_changed = (
            event.payload["additions"] + event.payload["deletions"]
        )
    
    final_priority = calculator.calculate_priority(priority_metrics)
    explanation = calculator.get_priority_explanation(priority_metrics, final_priority)
    
    print(f"   Final Priority Score: {final_priority:.2f}")
    print(f"   Priority Category: {explanation['priority_category']}")
    
    # Step 4: Generate cognitive response (simulated)
    print("\n4. 🧠 Cognitive Agent Response")
    
    if final_priority >= 15.0:
        response_urgency = "IMMEDIATE"
        actions = [
            "Notify security team immediately",
            "Schedule urgent code review",
            "Block other deployments until resolved",
            "Set up monitoring for this change"
        ]
    elif final_priority >= 10.0:
        response_urgency = "HIGH"
        actions = [
            "Schedule expedited review",
            "Notify relevant stakeholders",
            "Run additional security checks"
        ]
    else:
        response_urgency = "NORMAL"
        actions = [
            "Add to review queue",
            "Run standard CI/CD pipeline"
        ]
    
    print(f"   Response Urgency: {response_urgency}")
    print("   Recommended Actions:")
    for action in actions:
        print(f"     • {action}")
    
    # Step 5: Execute actions using GitHub CLI (dry-run)
    print("\n5. 🛠️ Executing Actions (Dry-Run)")
    
    cli_tool = GitHubCliTool()
    
    if final_priority >= 15.0:
        # High priority actions
        cli_commands = [
            "gh issue create --title '[URGENT] Security review needed for PR #789' --body 'Critical security PR requires immediate review' --label urgent,security",
            "gh pr edit 789 --add-label priority-critical"
        ]
    else:
        cli_commands = [
            "gh pr edit 789 --add-label needs-review"
        ]
    
    for command in cli_commands:
        input_data = GitHubCliInput(
            command=command,
            dry_run=True,
            repository=event.repository
        )
        
        result = await cli_tool.execute(input_data)
        print(f"   ✅ {command}")
        print(f"      Result: {result.output}")
    
    print("\n6. 📈 Telemetry & Learning")
    print("   • Event processed and logged")
    print("   • Priority calculation recorded") 
    print("   • Action effectiveness will be monitored")
    print("   • Model will learn from outcome")


async def main():
    """Main demonstration function."""
    print("🚀 Super Alita GitHub Integration Demonstration")
    print("=" * 60)
    print("\nThis demo showcases the enhanced GitHub integration capabilities")
    print("added to the Super Alita cognitive agent architecture.")
    
    try:
        # Run all demonstration sections
        await demo_github_event_processing()
        await demo_priority_calculation() 
        await demo_github_cli_tool()
        await demo_github_api_integration()
        await demo_integration_workflow()
        
        print_banner("Demo Complete!")
        print("\n✅ All GitHub integration features demonstrated successfully!")
        print("\n🎯 Key Capabilities Shown:")
        print("   • GitHub event schema processing")
        print("   • Enhanced priority calculation with GitHub metrics")
        print("   • GitHub CLI tool integration with dry-run support")
        print("   • GitHub API client with rate limiting")
        print("   • End-to-end cognitive workflow for GitHub events")
        print("\n🔗 Next Steps:")
        print("   • Configure GitHub token for live API access")
        print("   • Set up GitHub webhooks for real-time events")
        print("   • Deploy cognitive agent workflow in shadow mode")
        print("   • Monitor and tune priority calculation parameters")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())