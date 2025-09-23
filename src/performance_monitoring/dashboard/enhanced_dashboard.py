"""
Enhanced Performance Monitoring Dashboard with Predictive Analysis

Advanced dashboard with machine learning predictions and anomaly detection.
"""

import asyncio
import statistics
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformancePrediction:
    """Performance prediction data."""
    metric_name: str
    current_value: float
    predicted_value: float
    confidence: float
    trend_direction: str  # up, down, stable
    risk_level: str  # low, medium, high, critical
    recommendation: Optional[str] = None


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    severity: str  # low, medium, high, critical
    description: str
    first_detected: datetime


@dataclass
class PerformanceInsight:
    """Performance insight and recommendation."""
    title: str
    description: str
    impact: str  # performance, reliability, cost, user_experience
    priority: str  # low, medium, high, critical
    actionable_steps: List[str]
    expected_improvement: Optional[str] = None


class EnhancedPerformanceDashboard:
    """
    Enhanced performance dashboard with predictive analytics.
    
    Implements Article IV: Integration-First through comprehensive integration.
    Implements Article V: Clarity through clear visualization and insights.
    """
    
    def __init__(self, performance_monitor, telemetry_bridge, constitutional_engine):
        self.performance_monitor = performance_monitor
        self.telemetry_bridge = telemetry_bridge
        self.constitutional_engine = constitutional_engine
        
        # Analytics data
        self.metric_history: Dict[str, deque] = {}
        self.predictions: Dict[str, PerformancePrediction] = {}
        self.anomalies: Dict[str, AnomalyDetection] = {}
        self.insights: List[PerformanceInsight] = []
        
        # Monitoring state
        self._prediction_active = False
        self._prediction_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced Performance Dashboard initialized")

    async def start_predictive_monitoring(self) -> None:
        """Start predictive monitoring."""
        if self._prediction_active:
            return
            
        self._prediction_active = True
        self._prediction_task = asyncio.create_task(self._analysis_loop())
        logger.info("Predictive monitoring started")

    async def stop_predictive_monitoring(self) -> None:
        """Stop predictive monitoring."""
        if not self._prediction_active:
            return
            
        self._prediction_active = False
        if self._prediction_task:
            self._prediction_task.cancel()
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                pass
        logger.info("Predictive monitoring stopped")

    async def generate_predictions(self) -> Dict[str, PerformancePrediction]:
        """Generate performance predictions."""
        current_data = self._collect_current_metrics()
        predictions = {}
        
        for metric_name, current_value in current_data.items():
            prediction = await self._predict_metric(metric_name, current_value)
            if prediction:
                predictions[metric_name] = prediction
        
        self.predictions = predictions
        return predictions

    async def detect_anomalies(self) -> Dict[str, AnomalyDetection]:
        """Detect performance anomalies."""
        current_data = self._collect_current_metrics()
        anomalies = {}
        
        for metric_name, current_value in current_data.items():
            anomaly = await self._detect_anomaly(metric_name, current_value)
            if anomaly:
                anomalies[metric_name] = anomaly
        
        self.anomalies = anomalies
        return anomalies

    async def generate_insights(self) -> List[PerformanceInsight]:
        """Generate actionable performance insights."""
        insights = []
        
        # Analyze predictions
        for prediction in self.predictions.values():
            if prediction.risk_level in ["high", "critical"]:
                insights.append(PerformanceInsight(
                    title=f"Predicted {prediction.metric_name} degradation",
                    description=f"Risk level: {prediction.risk_level}",
                    impact="performance",
                    priority=prediction.risk_level,
                    actionable_steps=[
                        prediction.recommendation or "Monitor closely",
                        "Review recent changes",
                        "Consider optimization"
                    ],
                    expected_improvement="Prevent degradation"
                ))
        
        # Analyze anomalies
        for anomaly in self.anomalies.values():
            if anomaly.severity in ["medium", "high", "critical"]:
                insights.append(PerformanceInsight(
                    title=f"Anomaly in {anomaly.metric_name}",
                    description=anomaly.description,
                    impact="reliability",
                    priority=anomaly.severity,
                    actionable_steps=[
                        "Investigate root cause",
                        "Check recent changes",
                        "Review error logs"
                    ],
                    expected_improvement="Restore normal levels"
                ))
        
        # Constitutional compliance insights
        constitutional_insights = await self._analyze_constitutional_insights()
        insights.extend(constitutional_insights)
        
        # Sort by priority
        insights.sort(key=lambda x: self._get_priority_value(x.priority), reverse=True)
        self.insights = insights[:10]
        return self.insights

    def get_enhanced_dashboard_state(self) -> Dict[str, Any]:
        """Get enhanced dashboard state."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": {name: self._prediction_to_dict(pred) 
                          for name, pred in self.predictions.items()},
            "anomalies": {name: self._anomaly_to_dict(anomaly) 
                         for name, anomaly in self.anomalies.items()},
            "insights": [self._insight_to_dict(insight) for insight in self.insights],
            "performance_scores": self._calculate_performance_scores(),
            "system_health": self._predict_system_health()
        }

    async def _analysis_loop(self) -> None:
        """Background analysis loop."""
        while self._prediction_active:
            try:
                await self._update_metric_history()
                await self.generate_predictions()
                await self.detect_anomalies()
                await self.generate_insights()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(60)

    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metric values."""
        metrics = {}
        
        try:
            # Performance metrics
            summary = self.performance_monitor.get_performance_summary()
            metrics["response_time"] = summary.get("average_response_time_ms", 0)
            metrics["success_rate"] = summary.get("success_rate", 1.0)
            
            # Telemetry metrics
            telemetry = self.telemetry_bridge.get_telemetry_summary()
            metrics["event_rate"] = telemetry.get("event_rate_per_minute", 0)
            
            # Constitutional metrics
            compliance = self.constitutional_engine.get_compliance_trend()
            if compliance.get("status") != "no_data":
                metrics["constitutional_score"] = compliance.get("average_score", 0.75)
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics

    async def _predict_metric(self, metric_name: str, current_value: float) -> Optional[PerformancePrediction]:
        """Predict future metric value."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 5:
            return None
        
        try:
            history = list(self.metric_history[metric_name])
            
            # Simple linear regression
            x = list(range(len(history)))
            slope, intercept = self._linear_regression(x, history)
            predicted_value = slope * len(history) + intercept
            
            # Calculate confidence and risk
            variance = statistics.variance(history) if len(history) > 1 else 0
            confidence = max(0.1, min(0.95, 1.0 - (variance / max(abs(current_value), 1))))
            
            trend = "stable"
            if abs(slope) > 0.01:
                trend = "up" if slope > 0 else "down"
            
            risk_level = self._assess_risk(metric_name, current_value, predicted_value)
            recommendation = self._get_recommendation(metric_name, risk_level)
            
            return PerformancePrediction(
                metric_name=metric_name,
                current_value=current_value,
                predicted_value=predicted_value,
                confidence=confidence,
                trend_direction=trend,
                risk_level=risk_level,
                recommendation=recommendation
            )
        except Exception as e:
            logger.error(f"Prediction error for {metric_name}: {e}")
            return None

    async def _detect_anomaly(self, metric_name: str, current_value: float) -> Optional[AnomalyDetection]:
        """Detect metric anomaly."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 10:
            return None
        
        try:
            history = list(self.metric_history[metric_name])
            mean = statistics.mean(history)
            std_dev = statistics.stdev(history) if len(history) > 1 else 0
            
            # 2-sigma bounds
            lower_bound = mean - (2 * std_dev)
            upper_bound = mean + (2 * std_dev)
            
            if current_value < lower_bound or current_value > upper_bound:
                anomaly_score = abs(current_value - mean) / max(std_dev, 0.01)
                
                severity = "low"
                if anomaly_score > 3:
                    severity = "critical"
                elif anomaly_score > 2:
                    severity = "high"
                elif anomaly_score > 1:
                    severity = "medium"
                
                direction = "below" if current_value < lower_bound else "above"
                description = f"{metric_name} is {direction} normal range"
                
                return AnomalyDetection(
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_range=(lower_bound, upper_bound),
                    severity=severity,
                    description=description,
                    first_detected=datetime.now(timezone.utc)
                )
        except Exception as e:
            logger.error(f"Anomaly detection error for {metric_name}: {e}")
        
        return None

    async def _update_metric_history(self) -> None:
        """Update metric history."""
        current_metrics = self._collect_current_metrics()
        
        for metric_name, value in current_metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=100)
            self.metric_history[metric_name].append(value)

    async def _analyze_constitutional_insights(self) -> List[PerformanceInsight]:
        """Analyze constitutional compliance for insights."""
        insights = []
        
        try:
            compliance = self.constitutional_engine.get_compliance_trend()
            if compliance.get("status") != "no_data":
                avg_score = compliance.get("average_score", 0.75)
                
                if avg_score < 0.75:
                    insights.append(PerformanceInsight(
                        title="Constitutional compliance below threshold",
                        description=f"Score: {avg_score:.3f} (required: 0.75)",
                        impact="reliability",
                        priority="high",
                        actionable_steps=[
                            "Review violations",
                            "Update workflows",
                            "Provide training"
                        ],
                        expected_improvement="Achieve >0.75 compliance"
                    ))
        except Exception as e:
            logger.error(f"Constitutional analysis error: {e}")
        
        return insights

    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Calculate linear regression."""
        n = len(x)
        if n == 0:
            return 0.0, 0.0
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0, sum_y / n if n > 0 else 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept

    def _assess_risk(self, metric_name: str, current: float, predicted: float) -> str:
        """Assess prediction risk level."""
        change_percent = abs(predicted - current) / max(abs(current), 0.01)
        
        if metric_name == "response_time":
            if predicted > 2000:
                return "critical"
            elif predicted > 1000:
                return "high"
        elif metric_name == "success_rate":
            if predicted < 0.8:
                return "critical"
            elif predicted < 0.9:
                return "high"
        elif metric_name == "constitutional_score":
            if predicted < 0.6:
                return "critical"
            elif predicted < 0.75:
                return "high"
        
        return "medium" if change_percent > 0.2 else "low"

    def _get_recommendation(self, metric_name: str, risk_level: str) -> Optional[str]:
        """Get recommendation for metric."""
        if risk_level in ["high", "critical"]:
            recommendations = {
                "response_time": "Optimize performance or scale resources",
                "success_rate": "Review error patterns and implement fixes",
                "constitutional_score": "Review validation processes"
            }
            return recommendations.get(metric_name)
        return None

    def _calculate_performance_scores(self) -> Dict[str, float]:
        """Calculate performance scores."""
        scores = {}
        
        if "response_time" in self.predictions:
            rt = self.predictions["response_time"].predicted_value
            scores["response_score"] = 1.0 if rt < 500 else max(0.1, 1000/rt)
        
        if "success_rate" in self.predictions:
            scores["reliability_score"] = self.predictions["success_rate"].predicted_value
        
        if "constitutional_score" in self.predictions:
            scores["constitutional_score"] = self.predictions["constitutional_score"].predicted_value
        
        if scores:
            scores["overall_score"] = statistics.mean(scores.values())
        
        return scores

    def _predict_system_health(self) -> Dict[str, Any]:
        """Predict system health."""
        high_risk = len([p for p in self.predictions.values() if p.risk_level in ["high", "critical"]])
        critical_anomalies = len([a for a in self.anomalies.values() if a.severity in ["high", "critical"]])
        
        if high_risk > 2 or critical_anomalies > 1:
            status = "declining"
        elif high_risk > 0 or critical_anomalies > 0:
            status = "at_risk"
        else:
            status = "healthy"
        
        return {"status": status, "risk_factors": high_risk + critical_anomalies}

    def _prediction_to_dict(self, prediction: PerformancePrediction) -> Dict[str, Any]:
        """Convert prediction to dict."""
        return {
            "current_value": prediction.current_value,
            "predicted_value": prediction.predicted_value,
            "confidence": prediction.confidence,
            "trend": prediction.trend_direction,
            "risk_level": prediction.risk_level,
            "recommendation": prediction.recommendation
        }

    def _anomaly_to_dict(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """Convert anomaly to dict."""
        return {
            "current_value": anomaly.current_value,
            "expected_range": anomaly.expected_range,
            "severity": anomaly.severity,
            "description": anomaly.description,
            "detected_at": anomaly.first_detected.isoformat()
        }

    def _insight_to_dict(self, insight: PerformanceInsight) -> Dict[str, Any]:
        """Convert insight to dict."""
        return {
            "title": insight.title,
            "description": insight.description,
            "impact": insight.impact,
            "priority": insight.priority,
            "actions": insight.actionable_steps,
            "improvement": insight.expected_improvement
        }

    def _get_priority_value(self, priority: str) -> int:
        """Get priority numeric value."""
        values = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return values.get(priority.lower(), 2)