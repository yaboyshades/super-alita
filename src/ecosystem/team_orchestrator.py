# src/ecosystem/team_orchestrator.py
"""
Orchestrates AI development tools at the team level by aggregating
metrics and identifying organization-wide patterns and bottlenecks.
"""
from collections import Counter
from typing import Dict, Any, List
from datetime import datetime, timezone


class TeamProductivityOrchestrator:
    """Aggregates developer metrics to provide team-level insights."""

    def __init__(self):
        # In-memory storage for aggregated data. In production, this would be
        # a database or a time-series metrics store (e.g., Prometheus, InfluxDB).
        self.todo_keyword_counts: Counter = Counter()
        self.naming_inconsistencies: Counter = Counter()
        self.workflow_execution_times: Dict[str, List[float]] = {
            "todo_resolution": []
        }
        self.total_workflows_processed = 0

    def consume_event(self, topic: str, payload: Dict[str, Any]):
        """Consumes an event from the event bus to update team metrics."""
        # This method would be called by a subscriber to the main event bus.
        if topic == "workflow.todo_resolution.completed":
            self._process_todo_completion(payload)
        
        # In a real system, you would handle other events like 'naming.inconsistency.detected',
        # 'snippet.used', 'code_review.completed', etc.

    def _process_todo_completion(self, payload: Dict[str, Any]):
        """Processes data from a completed TODO workflow."""
        self.total_workflows_processed += 1
        
        # Analyze TODO text for common keywords to suggest shared snippets
        todo_text = payload.get("context", {}).get("todo_text", "")
        keywords = self._extract_keywords(todo_text)
        self.todo_keyword_counts.update(keywords)
        
        # You would also record execution times, complexity scores, etc.
        # execution_time = payload.get("execution_time_ms")
        # if execution_time:
        #     self.workflow_execution_times["todo_resolution"].append(execution_time)

    def _extract_keywords(self, text: str) -> List[str]:
        """A simple keyword extractor to find common themes in TODOs."""
        import re
        words = re.findall(r'\b\w{4,15}\b', text.lower()) # Find words 4-15 chars long
        # Filter out common stop words
        stop_words = {"this", "that", "with", "from", "implement", "should", "todo"}
        return [word for word in words if word not in stop_words]

    def generate_team_health_summary(self) -> Dict[str, Any]:
        """Generates a summary of team productivity patterns."""
        
        # Suggest shared snippets based on most common TODO keywords
        suggested_snippets = []
        for keyword, count in self.todo_keyword_counts.most_common(5):
            if count >= 2: # Lower threshold for demo/testing purposes
                suggested_snippets.append({
                    "keyword": keyword,
                    "occurrences": count,
                    "suggestion": f"Create a shared snippet library for tasks related to '{keyword}'."
                })
        
        # In the future, this would also analyze naming, review velocity, etc.
        
        return {
            "summary_generated_at": datetime.now(timezone.utc).isoformat(),
            "total_workflows_analyzed": self.total_workflows_processed,
            "optimizations": {
                "suggested_snippet_libraries": suggested_snippets,
                "identified_naming_baselines": [
                    # This would be populated by analyzing naming convention events
                    {"pattern": "get_*_by_id", "consistency": 0.92, "suggestion": "Standardize data-fetching function names."}
                ]
            }
        }