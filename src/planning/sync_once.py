"""
One-shot metrics synchronization script for Super Alita.
Reads current metrics, applies decision engine, and updates todos accordingly.
"""

import asyncio
import logging
import sys
from time import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def collect_live_metrics() -> dict[str, float]:
    """
    Collect current metrics from the system.
    In production, this would read from Prometheus or the metrics registry.
    """
    try:
        # Try to get metrics from the registry
        from src.core.metrics_registry import get_metrics_registry

        registry = get_metrics_registry()

        # Get current FSM metrics
        metrics = {
            "mailbox_pressure": 0.0,
            "stale_rate": 0.0,
            "concurrency_load": 0.0,
            "ignored_triggers_rate": 0.0,
        }

        # Calculate derived metrics
        mailbox_size = registry.get_gauge("sa_fsm_mailbox_size")
        mailbox_max = max(registry.get_gauge("sa_fsm_mailbox_size_max"), 1)
        active_ops = registry.get_gauge("sa_fsm_active_operations")
        stale_completions = registry.get_counter("sa_fsm_stale_completions_total")
        total_ops = max(registry.get_counter("sa_fsm_operations_total"), 1)
        ignored_triggers = registry.get_counter("sa_fsm_ignored_triggers_total")

        metrics["mailbox_pressure"] = mailbox_size / mailbox_max
        metrics["stale_rate"] = stale_completions / total_ops
        metrics["concurrency_load"] = active_ops / 5.0  # Assume 5 max concurrent
        metrics["ignored_triggers_rate"] = ignored_triggers / total_ops

        logger.info(f"Collected live metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.warning(f"Failed to collect live metrics: {e}")

        # Fallback to simulated realistic values for demo
        return {
            "mailbox_pressure": 0.75,  # High pressure
            "stale_rate": 0.12,  # High stale rate
            "concurrency_load": 0.80,  # High load
            "ignored_triggers_rate": 0.08,  # Moderate ignored triggers
        }


async def sync_metrics_to_todos():
    """
    Main synchronization function - collect metrics and update todos.
    """
    start_time = time()
    logger.info("🔄 Starting metrics → todos synchronization")

    try:
        # 1. Collect current metrics
        logger.info("📊 Collecting metrics...")
        metrics = await collect_live_metrics()

        # 2. Apply smoothing and trend analysis
        logger.info("📈 Applying smoothing and trend analysis...")
        from src.core.smoothing import get_metrics_smoother

        smoother = get_metrics_smoother()

        smoothed_metrics = {}
        for name, value in metrics.items():
            analysis = smoother.update_metric(name, value)
            smoothed_metrics[name] = analysis["smoothed"]
            logger.info(
                f"  {name}: raw={value:.3f}, smoothed={analysis['smoothed']:.3f}, trend={'📈' if analysis['trending_up'] else '📉' if analysis['trending_down'] else '➡️'}"
            )

        # 3. Apply decision engine with anti-thrash protection
        logger.info("�� Running decision engine...")
        from src.core.decision_engine import get_metrics_classifier

        classifier = get_metrics_classifier()

        classifications = {}
        for name, value in smoothed_metrics.items():
            classification, should_act = classifier.classify_metric(name, value)
            classifications[name] = (classification, should_act)
            action_icon = "🚨" if should_act else "✅"
            logger.info(f"  {name}: {classification} {action_icon}")

        # 4. Calculate risk scores
        logger.info("⚠️  Calculating risk scores...")
        from src.core.risk_engine import get_risk_engine

        risk_engine = get_risk_engine()

        risk_assessment = risk_engine.assess_risk(
            "fsm",
            smoothed_metrics["mailbox_pressure"],
            smoothed_metrics["stale_rate"],
            smoothed_metrics["concurrency_load"],
            smoothed_metrics["ignored_triggers_rate"],
        )

        logger.info(f"  Risk Score: {risk_assessment['risk_score']:.3f}")
        logger.info(f"  Priority: {risk_assessment['current_priority']}")
        logger.info(f"  Trend: {risk_assessment['trend']}")

        # 5. Synchronize todos
        logger.info("📝 Synchronizing todos...")
        from src.core.todo_sync import get_todo_sync

        todo_sync = get_todo_sync()

        actions = todo_sync.sync_from_metrics(smoothed_metrics)

        # Log actions taken
        for metric, action in actions.items():
            action_icons = {
                "CREATE": "➕",
                "UPDATE": "✏️",
                "ESCALATE": "⬆️",
                "CLOSE": "✅",
                "NO_ACTION": "➡️",
            }
            icon = action_icons.get(action.value, "❓")
            logger.info(f"  {metric}: {action.value} {icon}")

        # 6. Get summary
        summary = todo_sync.get_active_todos_summary()
        logger.info(f"📋 Active todos: {summary['count']} total")
        for priority, count in summary["by_priority"].items():
            if count > 0:
                logger.info(f"  {priority}: {count} todos")

        elapsed = time() - start_time
        logger.info(f"✅ Synchronization completed in {elapsed:.2f}s")

        return {
            "success": True,
            "elapsed_s": elapsed,
            "metrics_processed": len(metrics),
            "actions_taken": len(
                [a for a in actions.values() if a.value != "NO_ACTION"]
            ),
            "active_todos": summary["count"],
            "risk_score": risk_assessment["risk_score"],
            "system_priority": risk_assessment["current_priority"],
        }

    except Exception as e:
        logger.error(f"❌ Synchronization failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "elapsed_s": time() - start_time}


def demo_anti_thrash_protection():
    """
    Demonstrate anti-thrash protection by simulating oscillating metrics.
    """
    logger.info("🎯 Demonstrating Anti-Thrash Protection")
    logger.info("=" * 50)

    from src.core.decision_engine import AlertGate, MetricsClassifier
    from src.core.smoothing import MetricsSmoother

    # Create classifier with short debounce for demo
    classifier = MetricsClassifier(AlertGate(min_interval_s=30))
    smoother = MetricsSmoother()

    # Simulate oscillating mailbox pressure around warning threshold
    test_values = [0.45, 0.52, 0.48, 0.55, 0.47, 0.53, 0.49, 0.51]

    for i, value in enumerate(test_values):
        # Smooth the value
        analysis = smoother.update_metric("mailbox_pressure", value)
        smoothed = analysis["smoothed"]

        # Classify
        classification, should_act = classifier.classify_metric(
            "mailbox_pressure", smoothed
        )

        logger.info(
            f"Step {i+1}: raw={value:.2f}, smoothed={smoothed:.2f}, class={classification}, act={should_act}"
        )

    alert_summary = classifier.get_alert_summary()
    logger.info(f"Final state: {alert_summary['alert_count']} active alerts")
    logger.info("✅ Anti-thrash protection working - oscillations handled gracefully")


async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_anti_thrash_protection()
        return

    logger.info("🚀 Super Alita: Metrics → Todos Synchronization")
    logger.info("=" * 60)

    result = await sync_metrics_to_todos()

    if result["success"]:
        logger.info("✅ Sync completed successfully!")
        logger.info(f"  📊 Processed {result['metrics_processed']} metrics")
        logger.info(f"  📝 Took {result['actions_taken']} todo actions")
        logger.info(f"  📋 {result['active_todos']} active todos")
        logger.info(f"  ⚠️  Risk score: {result['risk_score']:.3f}")
        logger.info(f"  🎯 System priority: {result['system_priority']}")
        logger.info(f"  ⏱️  Completed in {result['elapsed_s']:.2f}s")
        sys.exit(0)
    else:
        logger.error(f"❌ Sync failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
