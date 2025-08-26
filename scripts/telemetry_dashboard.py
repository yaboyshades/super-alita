"""
Real-time telemetry dashboard for monitoring Copilot integration
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

class TelemetryDashboard:
    """Monitor Copilot's adherence to architectural guidelines"""
    
    def __init__(self):
        self.metrics = {
            "total_interactions": 0,
            "guideline_references": {
                "plugin_architecture": 0,
                "registry_management": 0,
                "state_machine": 0,
                "event_bus": 0,
                "integration": 0
            },
            "mode_usage": {
                "guardian": 0,
                "refactor": 0,
                "generator": 0,
                "audit": 0
            },
            "compliance_score": 0.0
        }
        
    def track_interaction(self, response_text: str) -> Dict:
        """Analyze Copilot response for telemetry markers"""
        
        markers = {
            "persona_acknowledged": False,
            "version_mentioned": False,
            "guidelines_referenced": [],
            "mode_detected": None,
            "compliance_patterns": []
        }
        
        # Check for persona acknowledgment
        if "Super Alita Architectural Guardian" in response_text:
            markers["persona_acknowledged"] = True
            
        # Check version
        if "v2.0" in response_text or "version 2" in response_text:
            markers["version_mentioned"] = True
            
        # Check guideline references
        guidelines = [
            ("Plugin Architecture", "plugin_architecture"),
            ("Tool Registry", "registry_management"),
            ("State Machine", "state_machine"),
            ("Event Bus", "event_bus"),
            ("Component Integration", "integration")
        ]
        
        for guideline_name, metric_key in guidelines:
            if guideline_name.lower() in response_text.lower():
                markers["guidelines_referenced"].append(guideline_name)
                self.metrics["guideline_references"][metric_key] += 1
                
        # Detect operational mode
        if "reviewing" in response_text.lower() or "audit" in response_text.lower():
            markers["mode_detected"] = "audit"
            self.metrics["mode_usage"]["audit"] += 1
        elif "refactor" in response_text.lower():
            markers["mode_detected"] = "refactor"
            self.metrics["mode_usage"]["refactor"] += 1
        elif "generat" in response_text.lower():
            markers["mode_detected"] = "generator"
            self.metrics["mode_usage"]["generator"] += 1
        else:
            markers["mode_detected"] = "guardian"
            self.metrics["mode_usage"]["guardian"] += 1
            
        # Check for compliance patterns
        compliance_patterns = [
            "async def",
            "PluginInterface",
            "DecisionPolicyEngine",
            "create_event",
            "TransitionTrigger"
        ]
        
        for pattern in compliance_patterns:
            if pattern in response_text:
                markers["compliance_patterns"].append(pattern)
                
        # Update metrics
        self.metrics["total_interactions"] += 1
        self._calculate_compliance_score(markers)
        
        return markers
        
    def _calculate_compliance_score(self, markers: Dict) -> None:
        """Calculate overall compliance score"""
        score = 0
        max_score = 5
        
        if markers["persona_acknowledged"]:
            score += 1
        if markers["version_mentioned"]:
            score += 1
        if len(markers["guidelines_referenced"]) > 0:
            score += 1
        if markers["mode_detected"]:
            score += 1
        if len(markers["compliance_patterns"]) > 0:
            score += 1
            
        self.metrics["compliance_score"] = (score / max_score) * 100
        
    def generate_report(self) -> str:
        """Generate telemetry report"""
        return f"""
# 📈 Copilot Telemetry Dashboard
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**User:** yaboyshades

## Metrics Summary
- **Total Interactions:** {self.metrics['total_interactions']}
- **Compliance Score:** {self.metrics['compliance_score']:.1f}%

## Guideline References
| Guideline | Count | Percentage |
|-----------|-------|------------|
| Plugin Architecture | {self.metrics['guideline_references']['plugin_architecture']} | {self._calc_percentage('plugin_architecture')}% |
| Registry Management | {self.metrics['guideline_references']['registry_management']} | {self._calc_percentage('registry_management')}% |
| State Machine | {self.metrics['guideline_references']['state_machine']} | {self._calc_percentage('state_machine')}% |
| Event Bus | {self.metrics['guideline_references']['event_bus']} | {self._calc_percentage('event_bus')}% |
| Integration | {self.metrics['guideline_references']['integration']} | {self._calc_percentage('integration')}% |

## Mode Usage
| Mode | Count | Percentage |
|------|-------|------------|
| Guardian | {self.metrics['mode_usage']['guardian']} | {self._calc_mode_percentage('guardian')}% |
| Refactor | {self.metrics['mode_usage']['refactor']} | {self._calc_mode_percentage('refactor')}% |
| Generator | {self.metrics['mode_usage']['generator']} | {self._calc_mode_percentage('generator')}% |
| Audit | {self.metrics['mode_usage']['audit']} | {self._calc_mode_percentage('audit')}% |
"""
        
    def _calc_percentage(self, guideline: str) -> float:
        """Calculate guideline reference percentage"""
        total = sum(self.metrics['guideline_references'].values())
        if total == 0:
            return 0.0
        return (self.metrics['guideline_references'][guideline] / total) * 100
        
    def _calc_mode_percentage(self, mode: str) -> float:
        """Calculate mode usage percentage"""
        total = sum(self.metrics['mode_usage'].values())
        if total == 0:
            return 0.0
        return (self.metrics['mode_usage'][mode] / total) * 100

if __name__ == "__main__":
    # Example usage
    dashboard = TelemetryDashboard()
    
    # Simulate some interactions
    sample_responses = [
        "I am the Super Alita Architectural Guardian v2.0. This code should inherit from PluginInterface.",
        "This violates guideline #2 - Tool Registry Management. Use DecisionPolicyEngine instead.",
        "Let me refactor this to use async def and create_event properly.",
        "Generating a new plugin with proper Event Bus patterns..."
    ]
    
    for response in sample_responses:
        dashboard.track_interaction(response)
    
    print(dashboard.generate_report())