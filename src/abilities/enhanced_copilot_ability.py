#!/usr/bin/env python3
"""
Enhanced Copilot Ability - Integrates DeepCode analysis with automated GitHub repository discovery
for comprehensive problem-solving workflows.

This ability combines:
- DeepCode analysis capabilities
- GitHub repository search and analysis
- Automated problem-solving workflows
- End-to-end solution implementation
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

from src.abilities.deepcode_analysis_ability import DeepCodeAnalysisAbility
from src.abilities.deepcode_integration_ability import DeepCodeIntegrationAbility
from src.atoms.web_agent_atom import WebAgentAtom
from src.core.events import create_event
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class EnhancedCopilotAbility(PluginInterface):
    """Enhanced Copilot ability with DeepCode integration and automated GitHub discovery"""

    def __init__(self) -> None:
        super().__init__()
        self.enabled = os.getenv("ENHANCED_COPILOT_ENABLED", "true").lower() == "true"
        
        # Initialize sub-abilities
        self.deepcode_analysis = DeepCodeAnalysisAbility()
        self.deepcode_integration = DeepCodeIntegrationAbility()
        self.web_agent = WebAgentAtom()
        
        # GitHub token for enhanced repository access
        self.github_token = os.getenv("GITHUB_TOKEN", "")

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        
        # Setup sub-abilities
        await self.deepcode_analysis.setup(event_bus, store, config)
        await self.deepcode_integration.setup(event_bus, store, config)
        await self.web_agent.setup(event_bus, store, config)
        
        logger.info("EnhancedCopilotAbility setup complete")

    @property
    def name(self) -> str:
        return "enhanced_copilot_ability"

    async def start(self) -> None:
        await super().start()
        if not self.enabled:
            logger.info("EnhancedCopilotAbility disabled; not starting.")
            return
            
        # Start sub-abilities
        await self.deepcode_analysis.start()
        await self.deepcode_integration.start()
        await self.web_agent.start()
        
        # Register as a tool provider
        await self.subscribe("tool_execution_request", self._handle_tool_request)
        logger.info("EnhancedCopilotAbility started")

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return list of available enhanced copilot tools"""
        return [
            {
                "name": "analyze_and_suggest_repos",
                "description": "Analyze code problems and suggest GitHub repositories that can help solve them",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem_description": {
                            "type": "string",
                            "description": "Description of the coding problem or requirement"
                        },
                        "code_context": {
                            "type": "string",
                            "description": "Optional existing code context for analysis",
                            "default": ""
                        },
                        "language_preference": {
                            "type": "string",
                            "description": "Preferred programming language (python, javascript, etc.)",
                            "default": "python"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of repository suggestions",
                            "default": 5
                        }
                    },
                    "required": ["problem_description"]
                }
            },
            {
                "name": "automated_problem_solver",
                "description": "End-to-end automated problem solver that finds repos, analyzes code, and provides implementation guidance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Detailed description of the task to solve"
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to the workspace directory",
                            "default": "."
                        },
                        "include_code_generation": {
                            "type": "boolean",
                            "description": "Whether to include code generation suggestions",
                            "default": True
                        },
                        "analyze_existing_code": {
                            "type": "boolean",
                            "description": "Whether to analyze existing code in workspace",
                            "default": True
                        }
                    },
                    "required": ["task_description"]
                }
            },
            {
                "name": "repository_deep_analysis",
                "description": "Perform deep analysis on a specific GitHub repository to understand its capabilities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "GitHub repository URL"
                        },
                        "analysis_focus": {
                            "type": "string",
                            "enum": ["architecture", "security", "performance", "usability", "all"],
                            "description": "Focus area for the analysis",
                            "default": "all"
                        },
                        "include_dependencies": {
                            "type": "boolean",
                            "description": "Whether to analyze dependencies",
                            "default": True
                        }
                    },
                    "required": ["repo_url"]
                }
            },
            {
                "name": "enhanced_code_review",
                "description": "Comprehensive code review with GitHub repository context and DeepCode analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_path": {
                            "type": "string",
                            "description": "Path to code file or directory to review"
                        },
                        "review_type": {
                            "type": "string",
                            "enum": ["security", "performance", "best_practices", "comprehensive"],
                            "description": "Type of review to perform",
                            "default": "comprehensive"
                        },
                        "suggest_improvements": {
                            "type": "boolean",
                            "description": "Whether to suggest specific improvements with GitHub examples",
                            "default": True
                        }
                    },
                    "required": ["code_path"]
                }
            }
        ]

    async def _handle_tool_request(self, event: dict[str, Any]) -> None:
        """Handle tool execution requests for enhanced copilot"""
        try:
            tool_name = event.get("tool_name")
            args = event.get("args", {})
            session_id = event.get("session_id", "unknown")

            result = await self._execute_tool(tool_name, args)

            # Emit result event
            result_event = create_event(
                "tool_execution_result",
                tool_name=tool_name,
                result=result,
                session_id=session_id,
                source_plugin=self.name,
                success=not result.get("error"),
                timestamp=_utcnow()
            )
            
            if self.event_bus:
                await self.event_bus.publish(result_event)

        except Exception as e:
            logger.exception(f"Enhanced copilot tool execution failed: {e}")
            error_event = create_event(
                "tool_execution_error",
                tool_name=event.get("tool_name"),
                error=str(e),
                session_id=event.get("session_id", "unknown"),
                source_plugin=self.name,
                timestamp=_utcnow()
            )
            
            if self.event_bus:
                await self.event_bus.publish(error_event)

    async def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the specified enhanced copilot tool"""
        
        if tool_name == "analyze_and_suggest_repos":
            return await self._analyze_and_suggest_repos(args)
        elif tool_name == "automated_problem_solver":
            return await self._automated_problem_solver(args)
        elif tool_name == "repository_deep_analysis":
            return await self._repository_deep_analysis(args)
        elif tool_name == "enhanced_code_review":
            return await self._enhanced_code_review(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _analyze_and_suggest_repos(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze problem and suggest relevant GitHub repositories"""
        try:
            problem_description = args.get("problem_description", "")
            code_context = args.get("code_context", "")
            language_preference = args.get("language_preference", "python")
            max_results = args.get("max_results", 5)

            # Step 1: Analyze code context if provided
            code_analysis = None
            if code_context.strip():
                code_analysis = await self._analyze_code_context(code_context)

            # Step 2: Generate search query based on problem description and analysis
            search_query = await self._generate_search_query(
                problem_description, code_analysis, language_preference
            )

            # Step 3: Search GitHub repositories
            repo_suggestions = await self._search_github_repos(search_query, max_results)

            # Step 4: Analyze each suggested repository
            analyzed_repos = []
            for repo in repo_suggestions[:3]:  # Analyze top 3 repositories
                repo_analysis = await self._analyze_repository_relevance(
                    repo, problem_description
                )
                analyzed_repos.append({
                    **repo,
                    "relevance_analysis": repo_analysis
                })

            return {
                "problem_description": problem_description,
                "search_query": search_query,
                "code_analysis": code_analysis,
                "repository_suggestions": analyzed_repos,
                "total_found": len(repo_suggestions),
                "timestamp": _utcnow()
            }

        except Exception as e:
            logger.error(f"Error in analyze_and_suggest_repos: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    async def _automated_problem_solver(self, args: dict[str, Any]) -> dict[str, Any]:
        """End-to-end automated problem solver"""
        try:
            task_description = args.get("task_description", "")
            workspace_path = args.get("workspace_path", ".")
            include_code_generation = args.get("include_code_generation", True)
            analyze_existing_code = args.get("analyze_existing_code", True)

            solution_steps = []

            # Step 1: Analyze existing workspace if requested
            workspace_analysis = None
            if analyze_existing_code and Path(workspace_path).exists():
                workspace_analysis = await self.deepcode_integration._analyze_workspace_context({
                    "workspace_path": workspace_path
                })
                solution_steps.append({
                    "step": "workspace_analysis",
                    "description": "Analyzed existing workspace code",
                    "result": workspace_analysis
                })

            # Step 2: Find relevant repositories
            repo_analysis = await self._analyze_and_suggest_repos({
                "problem_description": task_description,
                "code_context": json.dumps(workspace_analysis) if workspace_analysis else "",
                "max_results": 5
            })
            solution_steps.append({
                "step": "repository_discovery",
                "description": "Found relevant GitHub repositories",
                "result": repo_analysis
            })

            # Step 3: Generate implementation plan
            implementation_plan = await self._generate_implementation_plan(
                task_description, workspace_analysis, repo_analysis
            )
            solution_steps.append({
                "step": "implementation_planning",
                "description": "Generated implementation plan",
                "result": implementation_plan
            })

            # Step 4: Code generation (if requested)
            code_suggestions = None
            if include_code_generation:
                code_suggestions = await self._generate_code_suggestions(
                    task_description, implementation_plan, repo_analysis
                )
                solution_steps.append({
                    "step": "code_generation",
                    "description": "Generated code suggestions",
                    "result": code_suggestions
                })

            return {
                "task_description": task_description,
                "workspace_analysis": workspace_analysis,
                "solution_steps": solution_steps,
                "implementation_plan": implementation_plan,
                "code_suggestions": code_suggestions,
                "success": True,
                "timestamp": _utcnow()
            }

        except Exception as e:
            logger.error(f"Error in automated_problem_solver: {e}")
            return {"error": f"Problem solving failed: {str(e)}"}

    async def _repository_deep_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        """Perform deep analysis on a specific GitHub repository"""
        try:
            repo_url = args.get("repo_url", "")
            analysis_focus = args.get("analysis_focus", "all")
            include_dependencies = args.get("include_dependencies", True)

            # Extract owner and repo from URL
            owner, repo = self._extract_repo_info(repo_url)
            if not owner or not repo:
                return {"error": "Invalid repository URL"}

            # Fetch repository metadata
            repo_metadata = await self._fetch_repo_metadata(owner, repo)

            # Analyze repository structure
            structure_analysis = await self._analyze_repo_structure(owner, repo)

            # Perform focused analysis based on focus parameter
            focused_analysis = await self._perform_focused_analysis(
                owner, repo, analysis_focus, structure_analysis
            )

            # Analyze dependencies if requested
            dependency_analysis = None
            if include_dependencies:
                dependency_analysis = await self._analyze_dependencies(owner, repo)

            return {
                "repository": f"{owner}/{repo}",
                "metadata": repo_metadata,
                "structure_analysis": structure_analysis,
                "focused_analysis": focused_analysis,
                "dependency_analysis": dependency_analysis,
                "analysis_focus": analysis_focus,
                "timestamp": _utcnow()
            }

        except Exception as e:
            logger.error(f"Error in repository_deep_analysis: {e}")
            return {"error": f"Repository analysis failed: {str(e)}"}

    async def _enhanced_code_review(self, args: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive code review with GitHub context"""
        try:
            code_path = args.get("code_path", "")
            review_type = args.get("review_type", "comprehensive")
            suggest_improvements = args.get("suggest_improvements", True)

            # Perform DeepCode analysis
            if Path(code_path).is_file():
                deepcode_result = await self.deepcode_analysis._analyze_file({
                    "file_path": code_path,
                    "analysis_level": "deep"
                })
            else:
                deepcode_result = await self.deepcode_analysis._analyze_directory({
                    "directory_path": code_path,
                    "analysis_level": "deep"
                })

            # Generate improvement suggestions with GitHub examples
            improvement_suggestions = []
            if suggest_improvements and deepcode_result.get("issues"):
                improvement_suggestions = await self._generate_improvement_suggestions(
                    deepcode_result, review_type
                )

            return {
                "code_path": code_path,
                "review_type": review_type,
                "deepcode_analysis": deepcode_result,
                "improvement_suggestions": improvement_suggestions,
                "timestamp": _utcnow()
            }

        except Exception as e:
            logger.error(f"Error in enhanced_code_review: {e}")
            return {"error": f"Code review failed: {str(e)}"}

    # Helper methods

    async def _analyze_code_context(self, code_context: str) -> dict[str, Any]:
        """Analyze provided code context using DeepCode"""
        try:
            # Create temporary file for analysis
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code_context)
                temp_file_path = temp_file.name

            try:
                result = await self.deepcode_analysis._analyze_file({
                    "file_path": temp_file_path,
                    "analysis_level": "semantic"
                })
                return result
            finally:
                Path(temp_file_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error analyzing code context: {e}")
            return {"error": f"Code analysis failed: {str(e)}"}

    async def _generate_search_query(
        self, problem_description: str, code_analysis: dict[str, Any] | None, language: str
    ) -> str:
        """Generate optimized search query for GitHub repositories"""
        query_parts = []
        
        # Add language filter
        query_parts.append(f"language:{language}")
        
        # Extract keywords from problem description
        problem_keywords = self._extract_keywords(problem_description)
        query_parts.extend(problem_keywords[:3])  # Limit to top 3 keywords
        
        # Add technology keywords from code analysis
        if code_analysis and "technologies" in code_analysis:
            tech_keywords = code_analysis["technologies"][:2]  # Limit to top 2
            query_parts.extend(tech_keywords)
        
        # Add common quality filters
        query_parts.append("stars:>10")
        
        return " ".join(query_parts)

    async def _search_github_repos(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search GitHub repositories using the web agent"""
        try:
            # Use the web agent's call method directly
            search_result = await self.web_agent.call(query, web_k=0, github_k=max_results)
            
            # Extract GitHub repositories from results
            github_repos = search_result.get("github", [])
            
            return github_repos[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching GitHub repos: {e}")
            return []

    async def _analyze_repository_relevance(
        self, repo: dict[str, Any], problem_description: str
    ) -> dict[str, Any]:
        """Analyze how relevant a repository is to the problem"""
        try:
            # Simple relevance scoring based on description matching
            repo_description = repo.get("snippet", "").lower()
            problem_words = set(problem_description.lower().split())
            repo_words = set(repo_description.split())
            
            # Calculate word overlap
            common_words = problem_words.intersection(repo_words)
            relevance_score = len(common_words) / max(len(problem_words), 1)
            
            return {
                "relevance_score": min(relevance_score, 1.0),
                "matching_keywords": list(common_words),
                "analysis_summary": f"Repository shows {relevance_score:.2%} relevance to the problem",
            }
            
        except Exception as e:
            logger.error(f"Error analyzing repository relevance: {e}")
            return {"error": f"Relevance analysis failed: {str(e)}"}

    async def _generate_implementation_plan(
        self, 
        task_description: str,
        workspace_analysis: dict[str, Any] | None,
        repo_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate implementation plan based on analysis"""
        try:
            plan_steps = []
            
            # Step 1: Environment setup
            plan_steps.append({
                "step": 1,
                "title": "Environment Setup",
                "description": "Set up development environment and dependencies",
                "actions": ["Create virtual environment", "Install required packages"]
            })
            
            # Step 2: Repository integration
            if repo_analysis.get("repository_suggestions"):
                top_repo = repo_analysis["repository_suggestions"][0]
                plan_steps.append({
                    "step": 2,
                    "title": "Repository Integration",
                    "description": f"Integrate patterns from {top_repo.get('title', 'N/A')}",
                    "actions": [
                        f"Study repository: {top_repo.get('url', 'N/A')}",
                        "Adapt relevant patterns to current project"
                    ]
                })
            
            # Step 3: Implementation
            plan_steps.append({
                "step": 3,
                "title": "Core Implementation",
                "description": "Implement the main functionality",
                "actions": ["Write core logic", "Add error handling", "Implement tests"]
            })
            
            # Step 4: Testing and validation
            plan_steps.append({
                "step": 4,
                "title": "Testing and Validation",
                "description": "Test and validate the implementation",
                "actions": ["Write unit tests", "Perform integration testing", "Validate against requirements"]
            })
            
            return {
                "task_description": task_description,
                "plan_steps": plan_steps,
                "estimated_complexity": "medium",  # Could be enhanced with ML
                "recommendations": [
                    "Follow existing code patterns in the workspace",
                    "Use proven solutions from suggested repositories",
                    "Implement comprehensive error handling"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating implementation plan: {e}")
            return {"error": f"Plan generation failed: {str(e)}"}

    async def _generate_code_suggestions(
        self,
        task_description: str,
        implementation_plan: dict[str, Any],
        repo_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate code suggestions based on analysis"""
        try:
            suggestions = []
            
            # Generate basic code structure
            suggestions.append({
                "type": "structure",
                "title": "Basic Project Structure",
                "code": self._generate_basic_structure(task_description),
                "description": "Basic project structure to get started"
            })
            
            # Generate implementation template
            suggestions.append({
                "type": "implementation",
                "title": "Implementation Template",
                "code": self._generate_implementation_template(task_description),
                "description": "Template code based on task requirements"
            })
            
            # Generate test template
            suggestions.append({
                "type": "tests",
                "title": "Test Template",
                "code": self._generate_test_template(task_description),
                "description": "Unit test template for the implementation"
            })
            
            return {
                "task_description": task_description,
                "code_suggestions": suggestions,
                "usage_notes": [
                    "Adapt the templates to your specific requirements",
                    "Add proper error handling and logging",
                    "Follow your project's coding standards"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating code suggestions: {e}")
            return {"error": f"Code generation failed: {str(e)}"}

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction (could be enhanced with NLP)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words = [word.lower().strip('.,!?();') for word in text.split()]
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        return list(set(keywords))[:10]  # Return unique keywords, limited to 10

    def _extract_repo_info(self, repo_url: str) -> tuple[str, str]:
        """Extract owner and repository name from GitHub URL"""
        try:
            # Handle various GitHub URL formats
            if "github.com/" in repo_url:
                parts = repo_url.split("github.com/")[1].split("/")
                if len(parts) >= 2:
                    return parts[0], parts[1].replace(".git", "")
            return "", ""
        except Exception:
            return "", ""

    async def _fetch_repo_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        """Fetch repository metadata from GitHub API"""
        if not self.github_token:
            return {"error": "GitHub token not configured"}
            
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.github.com/repos/{owner}/{repo}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"Failed to fetch metadata: {response.status}"}
                        
        except Exception as e:
            logger.error(f"Error fetching repo metadata: {e}")
            return {"error": f"Metadata fetch failed: {str(e)}"}

    async def _analyze_repo_structure(self, owner: str, repo: str) -> dict[str, Any]:
        """Analyze repository structure"""
        # Placeholder implementation - could be enhanced with actual GitHub API calls
        return {
            "has_readme": True,
            "has_tests": True,
            "has_ci": True,
            "main_language": "python",
            "structure_score": 0.8
        }

    async def _perform_focused_analysis(
        self, owner: str, repo: str, focus: str, structure: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform focused analysis based on focus parameter"""
        # Placeholder implementation
        return {
            "focus_area": focus,
            "analysis_results": f"Focused analysis on {focus} completed",
            "recommendations": [f"Consider improving {focus} aspects"]
        }

    async def _analyze_dependencies(self, owner: str, repo: str) -> dict[str, Any]:
        """Analyze repository dependencies"""
        # Placeholder implementation
        return {
            "dependency_count": 15,
            "outdated_dependencies": 2,
            "security_issues": 0,
            "recommendations": ["Update outdated dependencies"]
        }

    async def _generate_improvement_suggestions(
        self, deepcode_result: dict[str, Any], review_type: str
    ) -> list[dict[str, Any]]:
        """Generate improvement suggestions based on DeepCode analysis"""
        suggestions = []
        
        issues = deepcode_result.get("issues", [])
        for issue in issues[:5]:  # Limit to top 5 issues
            suggestions.append({
                "issue": issue.get("message", "Unknown issue"),
                "severity": issue.get("severity", "medium"),
                "line": issue.get("line", 0),
                "suggestion": f"Consider addressing this {issue.get('severity', 'medium')} severity issue",
                "example_repos": []  # Could be populated with relevant GitHub examples
            })
        
        return suggestions

    def _generate_basic_structure(self, task_description: str) -> str:
        """Generate basic project structure code"""
        return '''# Basic project structure
"""
Project: {task}
Generated by Enhanced Copilot
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProjectCore:
    """Core functionality for the project"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup project logging"""
        logging.basicConfig(level=logging.INFO)
        logger.info("Project initialized")
    
    def main_function(self):
        """Main functionality goes here"""
        # TODO: Implement main logic
        pass

if __name__ == "__main__":
    project = ProjectCore()
    project.main_function()
'''.format(task=task_description[:50] + "..." if len(task_description) > 50 else task_description)

    def _generate_implementation_template(self, task_description: str) -> str:
        """Generate implementation template code"""
        return '''# Implementation template
"""
Implementation for: {task}
"""

async def solve_problem():
    """
    Main implementation function
    
    TODO: Replace this with actual implementation logic
    """
    try:
        # Step 1: Setup
        logger.info("Starting problem solving process")
        
        # Step 2: Main logic
        result = await process_task()
        
        # Step 3: Return result
        return {{"success": True, "result": result}}
        
    except Exception as e:
        logger.error(f"Error solving problem: {{e}}")
        return {{"success": False, "error": str(e)}}

async def process_task():
    """Process the main task logic"""
    # TODO: Implement task processing
    return "Task completed"
'''.format(task=task_description[:50] + "..." if len(task_description) > 50 else task_description)

    def _generate_test_template(self, task_description: str) -> str:
        """Generate test template code"""
        return '''# Test template
"""
Tests for: {task}
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

# Import your implementation here
# from your_module import solve_problem

@pytest.mark.asyncio
async def test_solve_problem_success():
    """Test successful problem solving"""
    # TODO: Implement test logic
    # result = await solve_problem()
    # assert result["success"] is True
    pass

@pytest.mark.asyncio  
async def test_solve_problem_error():
    """Test error handling"""
    # TODO: Implement error test logic
    pass

def test_basic_functionality():
    """Test basic functionality"""
    # TODO: Implement basic tests
    assert True  # Placeholder
'''.format(task=task_description[:50] + "..." if len(task_description) > 50 else task_description)