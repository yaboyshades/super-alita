#!/usr/bin/env python3
"""
Bidirectional Telemetry Integration for Super Alita Guardian
Captures feedback about both @alita participant and Copilot IDE performance
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import sys

class BidirectionalTelemetryMonitor:
    """Monitor and analyze bidirectional telemetry data"""
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.telemetry_dir = self.workspace_root / ".vscode"
        self.telemetry_file = self.telemetry_dir / "telemetry.json"
        self.export_dir = self.telemetry_dir / "telemetry-exports"
        self.export_dir.mkdir(exist_ok=True)
        
    def read_telemetry_data(self) -> Dict[str, Any]:
        """Read current telemetry data"""
        if self.telemetry_file.exists():
            try:
                with open(self.telemetry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return self.get_default_metrics()
    
    def get_default_metrics(self) -> Dict[str, Any]:
        """Get default telemetry structure"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "totalInteractions": 0,
                "complianceScore": 0,
                "guidelineReferences": {},
                "modeUsage": {},
                "copilotPerformance": {
                    "responsesGenerated": 0,
                    "userRatings": [],
                    "responseAcceptanceRate": 0,
                    "averageResponseTime": 0,
                    "architecturalComplianceRate": 0,
                    "feedbackPatterns": {},
                    "contextualAccuracy": 0,
                    "codeQualityScore": 0
                },
                "ideInteraction": {
                    "editsTracked": 0,
                    "filesModified": [],
                    "commandsUsed": {},
                    "errorPatterns": {},
                    "sessionDuration": 0,
                    "productivityMetrics": {
                        "linesGenerated": 0,
                        "functionsCreated": 0,
                        "bugsIntroduced": 0,
                        "testsCovered": 0
                    }
                }
            },
            "version": "2.0.0"
        }
    
    def analyze_copilot_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Copilot performance metrics"""
        metrics = data.get("metrics", {})
        copilot = metrics.get("copilotPerformance", {})
        ide = metrics.get("ideInteraction", {})
        
        analysis = {
            "overall_score": 0,
            "strengths": [],
            "areas_for_improvement": [],
            "recommendations": [],
            "performance_trends": {}
        }
        
        # Calculate overall performance score
        factors = []
        
        # Response quality (user ratings)
        ratings = copilot.get("userRatings", [])
        if ratings:
            avg_rating = sum(r.get("rating", 0) for r in ratings) / len(ratings)
            quality_score = (avg_rating / 5.0) * 100
            factors.append(quality_score)
            
            if avg_rating >= 4.0:
                analysis["strengths"].append(f"High user satisfaction ({avg_rating:.1f}/5.0)")
            else:
                analysis["areas_for_improvement"].append(f"User satisfaction below target ({avg_rating:.1f}/5.0)")
        
        # Architectural compliance
        compliance_rate = copilot.get("architecturalComplianceRate", 0)
        factors.append(compliance_rate)
        
        if compliance_rate >= 70:
            analysis["strengths"].append(f"Good architectural compliance ({compliance_rate:.1f}%)")
        else:
            analysis["areas_for_improvement"].append(f"Architectural compliance needs improvement ({compliance_rate:.1f}%)")
            analysis["recommendations"].append("Review REUG v9.1 guidelines more thoroughly")
        
        # Code quality (error rate)
        bugs_introduced = ide.get("productivityMetrics", {}).get("bugsIntroduced", 0)
        edits_tracked = max(1, ide.get("editsTracked", 0))  # Avoid division by zero
        error_rate = (bugs_introduced / edits_tracked) * 100
        quality_score = max(0, 100 - error_rate)
        factors.append(quality_score)
        
        if error_rate < 5:
            analysis["strengths"].append(f"Low error rate ({error_rate:.1f}%)")
        else:
            analysis["areas_for_improvement"].append(f"High error rate ({error_rate:.1f}%)")
            analysis["recommendations"].append("Focus on code validation and testing patterns")
        
        # Productivity impact
        productivity = ide.get("productivityMetrics", {})
        lines_generated = productivity.get("linesGenerated", 0)
        functions_created = productivity.get("functionsCreated", 0)
        
        if lines_generated > 100:
            analysis["strengths"].append(f"High productivity ({lines_generated} lines generated)")
        
        if functions_created > 5:
            analysis["strengths"].append(f"Good function creation rate ({functions_created} functions)")
        
        # Calculate overall score
        if factors:
            analysis["overall_score"] = sum(factors) / len(factors)
        
        # Generate recommendations
        if analysis["overall_score"] < 70:
            analysis["recommendations"].extend([
                "Consider additional architectural training",
                "Review error patterns and improve validation",
                "Gather more user feedback to identify pain points"
            ])
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        data = self.read_telemetry_data()
        analysis = self.analyze_copilot_performance(data)
        
        metrics = data.get("metrics", {})
        copilot = metrics.get("copilotPerformance", {})
        ide = metrics.get("ideInteraction", {})
        
        # Calculate session duration
        session_start = data.get("timestamp", "")
        session_duration = "Unknown"
        if session_start:
            try:
                start_time = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
                duration = datetime.now(timezone.utc) - start_time
                session_duration = f"{duration.total_seconds() / 3600:.1f} hours"
            except:
                pass
        
        report = f"""
# 🤖 Bidirectional Telemetry Report
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Session Duration:** {session_duration}
**Overall Performance Score:** {analysis['overall_score']:.1f}/100

## 📊 Performance Summary

### Copilot Response Quality
- **Total Responses:** {copilot.get('responsesGenerated', 0)}
- **User Ratings:** {len(copilot.get('userRatings', []))} ratings provided
- **Acceptance Rate:** {copilot.get('responseAcceptanceRate', 0):.1f}%
- **Average Response Time:** {copilot.get('averageResponseTime', 0):.0f}ms
- **Architectural Compliance:** {copilot.get('architecturalComplianceRate', 0):.1f}%

### IDE Interaction Metrics
- **Edits Tracked:** {ide.get('editsTracked', 0)}
- **Files Modified:** {len(ide.get('filesModified', []))}
- **Lines Generated:** {ide.get('productivityMetrics', {}).get('linesGenerated', 0)}
- **Functions Created:** {ide.get('productivityMetrics', {}).get('functionsCreated', 0)}
- **Bugs Introduced:** {ide.get('productivityMetrics', {}).get('bugsIntroduced', 0)}

## 🎯 Performance Analysis

### Strengths
{chr(10).join(f"- {strength}" for strength in analysis['strengths']) if analysis['strengths'] else "- No specific strengths identified yet"}

### Areas for Improvement
{chr(10).join(f"- {area}" for area in analysis['areas_for_improvement']) if analysis['areas_for_improvement'] else "- Performance is meeting expectations"}

### Recommendations
{chr(10).join(f"- {rec}" for rec in analysis['recommendations']) if analysis['recommendations'] else "- Continue current approach"}

## 📈 User Feedback Details

### Recent Ratings
"""
        
        # Add recent ratings
        ratings = copilot.get("userRatings", [])
        if ratings:
            for rating in ratings[-5:]:  # Last 5 ratings
                stars = "⭐" * rating.get("rating", 0) + "☆" * (5 - rating.get("rating", 0))
                timestamp = rating.get("timestamp", "")
                context = rating.get("context", "")[:100] + "..." if len(rating.get("context", "")) > 100 else rating.get("context", "")
                report += f"- {stars} | {timestamp} | {context}\n"
        else:
            report += "- No ratings provided yet\n"
        
        report += f"""

## 🐛 Error Pattern Analysis

### Most Common Errors
"""
        
        # Add error patterns
        error_patterns = ide.get("errorPatterns", {})
        if error_patterns:
            sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            for error_type, count in sorted_errors[:5]:
                report += f"- {error_type.replace('_', ' ').title()}: {count} occurrences\n"
        else:
            report += "- No error patterns detected\n"
        
        report += f"""

## 🔄 Feedback Loop Insights

This telemetry provides bidirectional feedback:
1. **@alita participant performance** - How well the guardian provides architectural guidance
2. **Copilot IDE integration** - How well general Copilot assists with Super Alita patterns
3. **User satisfaction** - Direct feedback on response quality and usefulness
4. **Code quality impact** - Real measurements of introduced errors and productivity

### Next Steps
1. Review areas for improvement regularly
2. Use rating feedback to adjust response patterns  
3. Monitor error trends to improve code generation
4. Track compliance scores to ensure architectural adherence

---
*This report helps both the @alita guardian and general Copilot improve their performance in the Super Alita ecosystem.*
"""
        
        return report
    
    def export_analysis(self) -> str:
        """Export detailed analysis to file"""
        report = self.generate_performance_report()
        
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        export_file = self.export_dir / f"performance_analysis_{timestamp}.md"
        
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 Performance analysis exported to: {export_file}")
        return str(export_file)
    
    def monitor_live(self, interval: int = 30):
        """Monitor telemetry data in real-time"""
        print("🔄 Starting live telemetry monitoring...")
        print(f"📂 Monitoring: {self.telemetry_file}")
        print(f"⏱️  Refresh interval: {interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                data = self.read_telemetry_data()
                analysis = self.analyze_copilot_performance(data)
                
                metrics = data.get("metrics", {})
                copilot = metrics.get("copilotPerformance", {})
                
                print(f"\\n{datetime.now().strftime('%H:%M:%S')} | " +
                      f"Score: {analysis['overall_score']:.1f} | " +
                      f"Responses: {copilot.get('responsesGenerated', 0)} | " +
                      f"Acceptance: {copilot.get('responseAcceptanceRate', 0):.1f}%")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\\n🛑 Monitoring stopped")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        workspace = sys.argv[2] if len(sys.argv) > 2 else None
        
        monitor = BidirectionalTelemetryMonitor(workspace)
        
        if command == "report":
            print(monitor.generate_performance_report())
        elif command == "export":
            monitor.export_analysis()
        elif command == "monitor":
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            monitor.monitor_live(interval)
        else:
            print("Usage: python bidirectional_telemetry.py [report|export|monitor] [workspace_path] [interval]")
    else:
        # Default: generate report
        monitor = BidirectionalTelemetryMonitor()
        print(monitor.generate_performance_report())

if __name__ == "__main__":
    main()