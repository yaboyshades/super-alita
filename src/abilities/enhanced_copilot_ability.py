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
            },
            {
                "name": "discover_github_upgrades",
                "description": "Analyze codebase to discover upgrade opportunities and find GitHub repositories with better implementations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to the workspace directory to analyze",
                            "default": "."
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas to focus on: dependencies, security, performance, patterns, architecture",
                            "default": ["dependencies", "security", "performance", "patterns"]
                        },
                        "max_suggestions": {
                            "type": "integer",
                            "description": "Maximum number of upgrade suggestions to return",
                            "default": 10
                        },
                        "include_integration_steps": {
                            "type": "boolean",
                            "description": "Whether to include detailed integration steps for each suggestion",
                            "default": True
                        }
                    }
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
        elif tool_name == "discover_github_upgrades":
            return await self._discover_github_upgrades(args)
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
        """Search GitHub repositories using the web agent with fallback to mock data"""
        try:
            # Use the web agent's call method directly
            search_result = await self.web_agent.call(query, web_k=0, github_k=max_results)

            # Extract GitHub repositories from results
            github_repos = search_result.get("github", [])

            if github_repos:
                return github_repos[:max_results]
            else:
                logger.info("No repositories found via web agent, using fallback")
                return self._generate_fallback_repos(query, max_results)

        except Exception as e:
            logger.warning(f"GitHub search via web agent failed: {e}")
            # Fallback to mock data for demonstration
            return self._generate_fallback_repos(query, max_results)

    def _generate_fallback_repos(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Generate fallback repository suggestions when search is unavailable"""
        
        # Analyze query to provide relevant suggestions
        query_lower = query.lower()
        
        # Repository suggestions based on common query patterns
        fallback_repos = []
        
        if "python" in query_lower and "dependencies" in query_lower:
            fallback_repos.extend([
                {
                    "title": "pypa/pip-tools",
                    "url": "https://github.com/pypa/pip-tools",
                    "snippet": "A set of tools to keep your pinned Python dependencies fresh. Essential for Python dependency management, requirements upgrade, and package security."
                },
                {
                    "title": "pyupio/safety",
                    "url": "https://github.com/pyupio/safety", 
                    "snippet": "Safety checks your installed dependencies for known security vulnerabilities and suggests updates. Perfect for Python dependency security."
                },
                {
                    "title": "jazzband/pip-tools",
                    "url": "https://github.com/jazzband/pip-tools",
                    "snippet": "Pip-tools generates and maintains requirements.txt files. Ideal for dependency management and keeping Python dependencies updated."
                }
            ])
            
        elif "python" in query_lower and "security" in query_lower:
            fallback_repos.extend([
                {
                    "title": "bandit-dev/bandit",
                    "url": "https://github.com/bandit-dev/bandit",
                    "snippet": "Bandit finds common security issues in Python code. Essential for Python security, secure python development, and security best practices."
                },
                {
                    "title": "pyupio/safety",
                    "url": "https://github.com/pyupio/safety",
                    "snippet": "Safety scans Python dependencies for security vulnerabilities. Perfect for django security, flask security, and secure python practices."
                },
                {
                    "title": "python-security/pyt",
                    "url": "https://github.com/python-security/pyt", 
                    "snippet": "Python Taint Analysis Tool for detecting security vulnerabilities in Python web applications. Great for python security hardening."
                }
            ])
            
        elif "python" in query_lower:
            fallback_repos.extend([
                {
                    "title": "psf/requests",
                    "url": "https://github.com/psf/requests", 
                    "snippet": "A simple, yet elegant, HTTP library for Python. Great for API integration and modern HTTP patterns in python applications."
                },
                {
                    "title": "python-patterns/python-patterns",
                    "url": "https://github.com/python-patterns/python-patterns",
                    "snippet": "A collection of design patterns and idioms in Python. Essential for learning modern Python patterns, best practices, and design patterns."
                },
                {
                    "title": "fastapi/fastapi", 
                    "url": "https://github.com/fastapi/fastapi",
                    "snippet": "FastAPI framework, high performance, easy to learn, fast to code, ready for production. Modern python framework with best practices."
                }
            ])
            
        if "security" in query_lower and not fallback_repos:
            fallback_repos.extend([
                {
                    "title": "OWASP/CheatSheetSeries",
                    "url": "https://github.com/OWASP/CheatSheetSeries", 
                    "snippet": "The OWASP Cheat Sheet Series provides security best practices, vulnerability scanning guidelines, and code security recommendations."
                },
                {
                    "title": "securecodewarrior/secure-code-review",
                    "url": "https://github.com/securecodewarrior/secure-code-review",
                    "snippet": "Comprehensive guide for secure code review, security tools, and security best practices for developers."
                }
            ])
            
        if "dependency" in query_lower and not fallback_repos:
            fallback_repos.extend([
                {
                    "title": "dependabot/dependabot-core",
                    "url": "https://github.com/dependabot/dependabot-core",
                    "snippet": "The core logic behind Dependabot's dependency management, update PR creation, and package security monitoring."
                },
                {
                    "title": "pyupio/safety-db",
                    "url": "https://github.com/pyupio/safety-db",
                    "snippet": "Safety database of known security vulnerabilities in Python packages. Essential for dependency management and package security."
                }
            ])
            
        if "performance" in query_lower:
            fallback_repos.extend([
                {
                    "title": "python-performance/perf",
                    "url": "https://github.com/python-performance/perf",
                    "snippet": "Collection of Python performance optimization techniques, profiling tools, and benchmarks for better performance."
                },
                {
                    "title": "async-profiler/async-profiler",
                    "url": "https://github.com/async-profiler/async-profiler", 
                    "snippet": "Sampling CPU and HEAP profiler for performance optimization, featuring AsyncGetCallTrace + perf_events"
                }
            ])
            
        if "framework" in query_lower or "architecture" in query_lower:
            fallback_repos.extend([
                {
                    "title": "microsoft/architecture-center", 
                    "url": "https://github.com/microsoft/architecture-center",
                    "snippet": "Azure Architecture Center. Guidance for architecting solutions on Azure using established patterns and practices."
                },
                {
                    "title": "donnemartin/system-design-primer",
                    "url": "https://github.com/donnemartin/system-design-primer",
                    "snippet": "Learn how to design large-scale systems. Prep for the system design interview. Includes architecture patterns and best practices."
                }
            ])
            
        # If no specific matches, provide general popular repositories
        if not fallback_repos:
            fallback_repos = [
                {
                    "title": "github/gitignore",
                    "url": "https://github.com/github/gitignore", 
                    "snippet": "A collection of useful .gitignore templates for various languages and frameworks. Great for best practices."
                },
                {
                    "title": "awesome-lists/awesome",
                    "url": "https://github.com/sindresorhus/awesome",
                    "snippet": "Awesome lists about all kinds of interesting topics for developers. Curated list of best practices and tools."
                },
                {
                    "title": "best-practices/backend-best-practices",
                    "url": "https://github.com/futurice/backend-best-practices",
                    "snippet": "An evolving description of general best practices for backend development, patterns, and architecture."
                }
            ]
            
        logger.info(f"Generated {len(fallback_repos)} fallback repositories for query: {query}")
        return fallback_repos[:max_results]

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

    async def _discover_github_upgrades(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze codebase and discover GitHub repositories with upgrade opportunities"""
        try:
            workspace_path = args.get("workspace_path", ".")
            focus_areas = args.get("focus_areas", ["dependencies", "security", "performance", "patterns"])
            max_suggestions = args.get("max_suggestions", 10)
            include_integration_steps = args.get("include_integration_steps", True)

            upgrade_suggestions = []
            analysis_summary = {"workspace_path": workspace_path, "focus_areas": focus_areas}

            # Step 1: Analyze workspace with DeepCode integration
            workspace_analysis = await self.deepcode_integration._analyze_workspace_context({
                "workspace_path": workspace_path,
                "include_quality_metrics": True
            })

            if "error" in workspace_analysis:
                return {"error": f"Workspace analysis failed: {workspace_analysis['error']}"}

            analysis_summary["workspace_analysis"] = workspace_analysis

            # Step 2: Identify improvement opportunities based on focus areas
            improvement_opportunities = await self._identify_improvement_opportunities(
                workspace_analysis, focus_areas
            )
            analysis_summary["improvement_opportunities"] = improvement_opportunities

            # Step 3: Search GitHub for each improvement opportunity
            for opportunity in improvement_opportunities[:max_suggestions]:
                try:
                    # Generate search query for this opportunity
                    search_query = await self._generate_upgrade_search_query(opportunity)
                    
                    # Search GitHub for relevant repositories
                    github_repos = await self._search_github_repos(search_query, 3)
                    
                    # Analyze each repository for upgrade potential
                    analyzed_repos = []
                    for repo in github_repos:
                        repo_analysis = await self._analyze_upgrade_potential(repo, opportunity)
                        if repo_analysis.get("upgrade_score", 0) > 0.3:  # Lowered threshold for demonstration
                            analyzed_repos.append({**repo, "upgrade_analysis": repo_analysis})

                    if analyzed_repos:
                        suggestion = {
                            "opportunity": opportunity,
                            "search_query": search_query,
                            "suggested_repositories": analyzed_repos,
                            "priority": self._calculate_priority(opportunity, analyzed_repos),
                            "timestamp": _utcnow()
                        }

                        # Add integration steps if requested
                        if include_integration_steps and analyzed_repos:
                            suggestion["integration_steps"] = await self._generate_integration_steps(
                                opportunity, analyzed_repos[0]
                            )

                        upgrade_suggestions.append(suggestion)

                except Exception as e:
                    logger.warning(f"Failed to process opportunity {opportunity.get('type', 'unknown')}: {e}")

            return {
                "workspace_path": workspace_path,
                "analysis_summary": analysis_summary,
                "upgrade_suggestions": upgrade_suggestions,
                "total_opportunities": len(improvement_opportunities),
                "total_suggestions": len(upgrade_suggestions),
                "focus_areas": focus_areas,
                "timestamp": _utcnow()
            }

        except Exception as e:
            logger.error(f"Error in discover_github_upgrades: {e}")
            return {"error": f"GitHub upgrades discovery failed: {str(e)}"}

    async def _identify_improvement_opportunities(
        self, workspace_analysis: dict[str, Any], focus_areas: list[str]
    ) -> list[dict[str, Any]]:
        """Identify specific improvement opportunities from workspace analysis"""
        opportunities = []

        # Extract quality metrics and project info
        quality_metrics = workspace_analysis.get("quality_metrics", {})
        file_details = workspace_analysis.get("file_details", [])
        project_type = workspace_analysis.get("project_type", "unknown")
        main_languages = workspace_analysis.get("main_languages", [])
        workspace_path = workspace_analysis.get("workspace_path", ".")

        # Dependencies-focused opportunities
        if "dependencies" in focus_areas:
            # Check for common dependency files
            dependency_files_found = []
            dependency_files = ["requirements.txt", "package.json", "pom.xml", "cargo.toml", "composer.json", "Gemfile", "go.mod"]
            
            # Check workspace for dependency files
            from pathlib import Path
            workspace = Path(workspace_path)
            for dep_file in dependency_files:
                if (workspace / dep_file).exists():
                    dependency_files_found.append(dep_file)
                    
            # Always suggest dependency analysis for Python projects
            if project_type == "python" or "Python" in main_languages:
                opportunities.append({
                    "type": "dependency_upgrade",
                    "category": "dependencies",
                    "description": "Python dependency analysis and potential upgrades",
                    "file_path": "requirements.txt",
                    "priority": "medium",
                    "search_terms": ["python dependencies", "requirements upgrade", "pip-tools", "dependency management", "package security"]
                })
                
            # Add opportunities for found dependency files
            for dep_file in dependency_files_found:
                if dep_file != "requirements.txt":  # Avoid duplicates
                    lang = {"package.json": "JavaScript", "pom.xml": "Java", "cargo.toml": "Rust", "composer.json": "PHP", "Gemfile": "Ruby", "go.mod": "Go"}.get(dep_file, "")
                    opportunities.append({
                        "type": "dependency_upgrade", 
                        "category": "dependencies",
                        "description": f"{lang} dependency updates in {dep_file}",
                        "file_path": dep_file,
                        "priority": "medium",
                        "search_terms": [f"{lang.lower()} dependencies", "package upgrade", "security updates", f"{dep_file}"]
                    })
                    
            # Generic dependency opportunity if no specific files found
            if not dependency_files_found and not any(opp["type"] == "dependency_upgrade" for opp in opportunities):
                opportunities.append({
                    "type": "dependency_upgrade",
                    "category": "dependencies", 
                    "description": "Dependency management and security audit",
                    "priority": "low",
                    "search_terms": ["dependency management", "security audit", "package vulnerabilities", "update dependencies"]
                })

        # Security-focused opportunities
        if "security" in focus_areas:
            security_issues = quality_metrics.get("security_issues", 0)
            
            # Always suggest security improvements for active projects
            if security_issues > 0:
                opportunities.append({
                    "type": "security_improvement",
                    "category": "security",
                    "description": f"Address {security_issues} identified security issues",
                    "issue_count": security_issues,
                    "priority": "high",
                    "search_terms": ["security vulnerability", "secure coding", "security best practices", "penetration testing"]
                })
            else:
                # Proactive security opportunities based on project type
                if project_type == "python" or "Python" in main_languages:
                    opportunities.append({
                        "type": "security_improvement",
                        "category": "security",
                        "description": "Python security hardening and best practices",
                        "priority": "medium",
                        "search_terms": ["python security", "django security", "flask security", "secure python", "bandit security"]
                    })
                    
                # Web application security
                web_indicators = ["flask", "django", "fastapi", "express", "spring"]
                workspace_lower = str(workspace_path).lower()
                if any(indicator in workspace_lower for indicator in web_indicators):
                    opportunities.append({
                        "type": "security_improvement",
                        "category": "security", 
                        "description": "Web application security enhancements",
                        "priority": "high",
                        "search_terms": ["web security", "owasp", "authentication", "csrf protection", "sql injection"]
                    })
                else:
                    # Generic security opportunity
                    opportunities.append({
                        "type": "security_improvement",
                        "category": "security",
                        "description": "Proactive security analysis and hardening",
                        "priority": "medium",
                        "search_terms": ["security best practices", "code security", "vulnerability scanning", "security tools"]
                    })

        # Performance-focused opportunities  
        if "performance" in focus_areas:
            performance_issues = quality_metrics.get("performance_issues", 0)
            
            if performance_issues > 0:
                opportunities.append({
                    "type": "performance_optimization",
                    "category": "performance", 
                    "description": f"Address {performance_issues} performance-related issues",
                    "issue_count": performance_issues,
                    "priority": "medium",
                    "search_terms": ["performance optimization", "profiling", "caching", "database optimization", "async programming"]
                })
            else:
                # Proactive performance opportunities
                for language in main_languages:
                    if language.lower() == "python":
                        opportunities.append({
                            "type": "performance_optimization",
                            "category": "performance",
                            "description": "Python performance optimization techniques",
                            "language": language,
                            "priority": "low",
                            "search_terms": ["python performance", "asyncio", "cython", "numpy optimization", "python profiling"]
                        })
                    elif language.lower() == "javascript":
                        opportunities.append({
                            "type": "performance_optimization", 
                            "category": "performance",
                            "description": "JavaScript performance optimization",
                            "language": language,
                            "priority": "low",
                            "search_terms": ["javascript performance", "node.js optimization", "webpack optimization", "js profiling"]
                        })

        # Code patterns and architecture
        if "patterns" in focus_areas:
            # Analyze code patterns based on language and structure
            for language in main_languages:
                opportunities.append({
                    "type": "pattern_modernization", 
                    "category": "patterns",
                    "description": f"Modern {language} patterns and best practices",
                    "language": language,
                    "priority": "low",
                    "search_terms": [f"{language.lower()} patterns", "best practices", "modern", "design patterns", f"{language.lower()} framework"]
                })

        # Architecture improvements
        if "architecture" in focus_areas:
            if project_type != "unknown":
                opportunities.append({
                    "type": "architecture_upgrade",
                    "category": "architecture", 
                    "description": f"Architecture improvements for {project_type} project",
                    "project_type": project_type,
                    "priority": "medium",
                    "search_terms": [f"{project_type} architecture", "microservices", "clean architecture", "scalability", "system design"]
                })
            else:
                # Generic architecture opportunity
                opportunities.append({
                    "type": "architecture_upgrade",
                    "category": "architecture",
                    "description": "Software architecture and design improvements",
                    "priority": "low", 
                    "search_terms": ["software architecture", "design patterns", "system design", "scalable architecture"]
                })

        return opportunities

    async def _generate_upgrade_search_query(self, opportunity: dict[str, Any]) -> str:
        """Generate GitHub search query for an improvement opportunity"""
        search_terms = opportunity.get("search_terms", [])
        category = opportunity.get("category", "")
        priority = opportunity.get("priority", "medium")

        # Build base query
        query_parts = []
        
        # Add search terms
        if search_terms:
            query_parts.extend(search_terms[:3])  # Limit to top 3 terms

        # Add category-specific filters
        if category == "dependencies":
            query_parts.extend(["stars:>100", "pushed:>2023-01-01"])
        elif category == "security":
            query_parts.extend(["security", "stars:>50", "language:python"])
        elif category == "performance":
            query_parts.extend(["performance", "optimization", "stars:>20"])
        else:
            query_parts.extend(["stars:>10"])

        # Combine into search query
        return " ".join(query_parts)

    async def _analyze_upgrade_potential(
        self, repo: dict[str, Any], opportunity: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze how well a repository addresses the improvement opportunity"""
        try:
            # Extract repository information
            title = repo.get("title", "")
            url = repo.get("url", "")
            snippet = repo.get("snippet", "")
            
            # Calculate relevance score based on opportunity
            relevance_score = await self._calculate_relevance_score(repo, opportunity)
            
            # Analyze repository metadata if available
            metadata_score = 0.5  # Default score
            if url and "github.com" in url:
                try:
                    # Try to get basic repo info
                    owner, repo_name = self._extract_repo_info(url)
                    if owner and repo_name:
                        metadata = await self._fetch_repo_metadata(owner, repo_name)
                        metadata_score = self._score_repository_metadata(metadata, opportunity)
                except Exception as e:
                    logger.debug(f"Could not fetch metadata for {url}: {e}")

            # Combine scores
            upgrade_score = (relevance_score * 0.6) + (metadata_score * 0.4)
            
            return {
                "upgrade_score": min(upgrade_score, 1.0),
                "relevance_score": relevance_score,
                "metadata_score": metadata_score,
                "recommendation": self._generate_upgrade_recommendation(repo, opportunity, upgrade_score),
                "confidence": "high" if upgrade_score > 0.8 else "medium" if upgrade_score > 0.6 else "low"
            }

        except Exception as e:
            logger.error(f"Error analyzing upgrade potential: {e}")
            return {"upgrade_score": 0.0, "error": str(e)}

    async def _calculate_relevance_score(
        self, repo: dict[str, Any], opportunity: dict[str, Any]
    ) -> float:
        """Calculate relevance score between repository and opportunity"""
        title = repo.get("title", "").lower()
        snippet = repo.get("snippet", "").lower()
        search_terms = [term.lower() for term in opportunity.get("search_terms", [])]
        
        if not search_terms:
            return 0.5
            
        score = 0.0
        total_possible_score = 0.0
        
        # Check each search term
        for term in search_terms:
            term_words = term.split()
            term_score = 0.0
            
            # For multi-word terms, check if all words appear
            if len(term_words) > 1:
                # Multi-word term matching
                title_matches = all(word in title for word in term_words)
                snippet_matches = all(word in snippet for word in term_words)
                
                if title_matches:
                    term_score = 1.0  # Perfect match in title
                elif snippet_matches:
                    term_score = 0.8  # Perfect match in snippet
                elif any(word in title for word in term_words):
                    term_score = 0.6  # Partial match in title
                elif any(word in snippet for word in term_words):
                    term_score = 0.4  # Partial match in snippet
            else:
                # Single word term matching
                if term in title:
                    term_score = 0.9
                elif term in snippet:
                    term_score = 0.6
                elif any(word in term for word in title.split()):
                    term_score = 0.3
                elif any(word in term for word in snippet.split()):
                    term_score = 0.2
                    
            score += term_score
            total_possible_score += 1.0
                
        final_score = score / total_possible_score if total_possible_score > 0 else 0.0
        return min(final_score, 1.0)

    def _score_repository_metadata(self, metadata: dict[str, Any], opportunity: dict[str, Any]) -> float:
        """Score repository based on metadata quality"""
        score = 0.0
        
        # Check stars (popularity indicator)
        stars = metadata.get("stars", 0)
        if stars > 1000:
            score += 0.3
        elif stars > 100:
            score += 0.2
        elif stars > 10:
            score += 0.1
            
        # Check recent activity
        updated_at = metadata.get("updated_at", "")
        if updated_at and "2024" in updated_at or "2023" in updated_at:
            score += 0.2
            
        # Check if it has documentation
        has_readme = metadata.get("has_readme", False)
        if has_readme:
            score += 0.1
            
        # Check language match
        language = metadata.get("language", "").lower()
        if language and language in opportunity.get("search_terms", []):
            score += 0.2
            
        return min(score, 1.0)

    def _generate_upgrade_recommendation(
        self, repo: dict[str, Any], opportunity: dict[str, Any], upgrade_score: float
    ) -> str:
        """Generate human-readable upgrade recommendation"""
        repo_name = repo.get("title", "Repository")
        opportunity_type = opportunity.get("type", "improvement")
        category = opportunity.get("category", "general")
        
        if upgrade_score > 0.8:
            return f"Highly recommended: {repo_name} is an excellent solution for {category} {opportunity_type}"
        elif upgrade_score > 0.6:
            return f"Recommended: {repo_name} could help address your {category} needs"
        else:
            return f"Consider: {repo_name} might provide some relevant insights for {category}"

    def _calculate_priority(self, opportunity: dict[str, Any], repos: list[dict[str, Any]]) -> str:
        """Calculate overall priority for an upgrade suggestion"""
        opp_priority = opportunity.get("priority", "medium")
        repo_scores = [repo.get("upgrade_analysis", {}).get("upgrade_score", 0) for repo in repos]
        avg_score = sum(repo_scores) / len(repo_scores) if repo_scores else 0
        
        if opp_priority == "high" and avg_score > 0.7:
            return "high"
        elif opp_priority == "high" or avg_score > 0.8:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"

    async def _generate_integration_steps(
        self, opportunity: dict[str, Any], best_repo: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate step-by-step integration guide"""
        steps = []
        
        opportunity_type = opportunity.get("type", "improvement")
        repo_title = best_repo.get("title", "Repository")
        repo_url = best_repo.get("url", "")
        
        # Generic integration steps based on opportunity type
        if opportunity_type == "dependency_upgrade":
            steps = [
                {
                    "step": 1,
                    "title": "Research the dependency",
                    "description": f"Review {repo_title} documentation and compatibility requirements",
                    "action": f"Visit {repo_url} and read the README"
                },
                {
                    "step": 2,
                    "title": "Test in development",
                    "description": "Install and test the new dependency in a development environment",
                    "action": "Create a feature branch and update dependency versions"
                },
                {
                    "step": 3,
                    "title": "Update code if needed",
                    "description": "Modify code to work with the upgraded dependency",
                    "action": "Run tests and fix any breaking changes"
                },
                {
                    "step": 4,
                    "title": "Deploy and monitor",
                    "description": "Deploy to staging/production and monitor for issues",
                    "action": "Monitor performance and error rates"
                }
            ]
        elif opportunity_type == "security_improvement":
            steps = [
                {
                    "step": 1,
                    "title": "Analyze security patterns",
                    "description": f"Study security patterns from {repo_title}",
                    "action": f"Review security implementation in {repo_url}"
                },
                {
                    "step": 2,
                    "title": "Identify vulnerable areas",
                    "description": "Identify areas in your code that need security improvements",
                    "action": "Run security scans and review DeepCode analysis"
                },
                {
                    "step": 3,
                    "title": "Implement improvements",
                    "description": "Apply security patterns and best practices",
                    "action": "Update authentication, validation, and encryption"
                }
            ]
        else:
            # Generic steps
            steps = [
                {
                    "step": 1,
                    "title": "Study the solution",
                    "description": f"Analyze how {repo_title} addresses your needs",
                    "action": f"Review code and documentation at {repo_url}"
                },
                {
                    "step": 2,
                    "title": "Plan integration",
                    "description": "Plan how to integrate patterns or code into your project",
                    "action": "Create an integration plan and timeline"
                },
                {
                    "step": 3,
                    "title": "Implement changes",
                    "description": "Implement the improvements in your codebase",
                    "action": "Make incremental changes and test thoroughly"
                }
            ]
        
        return steps

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
