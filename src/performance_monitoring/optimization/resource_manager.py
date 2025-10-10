"""
Resource Management Module

Provides comprehensive resource management and optimization:
- Memory usage monitoring and optimization
- CPU and I/O resource tracking
- Garbage collection optimization
- Resource pooling and lifecycle management
- Memory leak detection
- Resource usage analytics and reporting
"""

import asyncio
import gc
import logging
import sys
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psutil

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamp: datetime
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_file_descriptors: int
    thread_count: int
    gc_collections: dict[int, int] = field(default_factory=dict)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    total_memory_mb: float
    heap_size_mb: float
    stack_size_mb: float
    gc_objects: int
    top_memory_consumers: list[tuple[str, int]] = field(default_factory=list)

@dataclass
class ResourcePool:
    """Generic resource pool"""
    name: str
    resource_type: str
    max_size: int
    current_size: int = 0
    in_use: int = 0
    created_count: int = 0
    destroyed_count: int = 0
    peak_usage: int = 0
    created_at: datetime = field(default_factory=datetime.now)

class MemoryTracker:
    """Advanced memory usage tracking"""
    
    def __init__(self):
        self.snapshots: deque = deque(maxlen=1000)
        self.memory_alerts: list[dict[str, Any]] = []
        self.tracemalloc_enabled = False
        
        # Memory thresholds
        self.warning_threshold_mb = 1000  # 1GB
        self.critical_threshold_mb = 2000  # 2GB
        
        # Track objects for leak detection
        self.object_refs: weakref.WeakSet = weakref.WeakSet()
        self.object_counts: dict[str, int] = defaultdict(int)
    
    def start_tracking(self):
        """Start memory tracking"""
        try:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            logger.info("Memory tracking started")
        except Exception as e:
            logger.warning(f"Could not start tracemalloc: {e}")
    
    def stop_tracking(self):
        """Stop memory tracking"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            logger.info("Memory tracking stopped")
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take memory usage snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get GC stats
        gc_objects = len(gc.get_objects())
        
        # Get top memory consumers if tracemalloc is enabled
        top_consumers = []
        if self.tracemalloc_enabled:
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                for stat in top_stats[:10]:
                    top_consumers.append((str(stat.traceback), stat.size))
            except Exception as e:
                logger.debug(f"Error getting memory traceback: {e}")
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            total_memory_mb=memory_info.rss / (1024 * 1024),
            heap_size_mb=memory_info.vms / (1024 * 1024),
            stack_size_mb=0,  # Not easily available in Python
            gc_objects=gc_objects,
            top_memory_consumers=top_consumers
        )
        
        self.snapshots.append(snapshot)
        
        # Check for memory alerts
        self._check_memory_alerts(snapshot)
        
        return snapshot
    
    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check for memory usage alerts"""
        memory_mb = snapshot.total_memory_mb
        
        if memory_mb > self.critical_threshold_mb:
            alert = {
                'level': 'critical',
                'message': f'Critical memory usage: {memory_mb:.2f}MB',
                'timestamp': snapshot.timestamp,
                'memory_mb': memory_mb
            }
            self.memory_alerts.append(alert)
            logger.critical(alert['message'])
            
        elif memory_mb > self.warning_threshold_mb:
            alert = {
                'level': 'warning',
                'message': f'High memory usage: {memory_mb:.2f}MB',
                'timestamp': snapshot.timestamp,
                'memory_mb': memory_mb
            }
            self.memory_alerts.append(alert)
            logger.warning(alert['message'])
    
    def detect_memory_leaks(self) -> list[dict[str, Any]]:
        """Detect potential memory leaks"""
        leaks = []
        
        if len(self.snapshots) < 2:
            return leaks
        
        # Compare recent snapshots
        recent = list(self.snapshots)[-10:]  # Last 10 snapshots
        
        if len(recent) < 10:
            return leaks
        
        # Calculate memory growth trend
        memory_values = [s.total_memory_mb for s in recent]
        
        # Simple linear regression to detect trend
        n = len(memory_values)
        sum_x = sum(range(n))
        sum_y = sum(memory_values)
        sum_xy = sum(i * memory_values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # If memory is consistently growing
        if slope > 5:  # More than 5MB per snapshot
            leaks.append({
                'type': 'memory_growth',
                'slope_mb_per_snapshot': slope,
                'current_memory_mb': memory_values[-1],
                'growth_rate': 'high' if slope > 20 else 'moderate'
            })
        
        # Check for object count growth
        current_objects = recent[-1].gc_objects
        previous_objects = recent[0].gc_objects
        
        if current_objects > previous_objects * 1.5:  # 50% increase in objects
            leaks.append({
                'type': 'object_growth',
                'object_increase': current_objects - previous_objects,
                'current_objects': current_objects,
                'growth_rate': 'high'
            })
        
        return leaks
    
    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.snapshots:
            return {}
        
        recent_snapshots = list(self.snapshots)[-10:]
        current = recent_snapshots[-1]
        
        # Calculate trends
        memory_values = [s.total_memory_mb for s in recent_snapshots]
        avg_memory = sum(memory_values) / len(memory_values)
        peak_memory = max(memory_values)
        
        return {
            'current_memory_mb': current.total_memory_mb,
            'peak_memory_mb': peak_memory,
            'average_memory_mb': avg_memory,
            'gc_objects': current.gc_objects,
            'memory_alerts': len(self.memory_alerts),
            'recent_alerts': [a for a in self.memory_alerts if (datetime.now() - a['timestamp']).total_seconds() < 3600],
            'potential_leaks': self.detect_memory_leaks()
        }

class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history: deque = deque(maxlen=2000)
        self.resource_pools: dict[str, ResourcePool] = {}
        
        # Resource thresholds
        self.cpu_threshold = 80.0  # 80%
        self.memory_threshold = 85.0  # 85%
        self.disk_threshold = 90.0  # 90%
        
        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._running = False
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Resource monitoring stopped")
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        try:
            process = psutil.Process()
            
            # Memory metrics
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # CPU metrics
            cpu_percent = process.cpu_percent()
            
            # I/O metrics
            io_counters = process.io_counters()
            disk_read_mb = io_counters.read_bytes / (1024 * 1024)
            disk_write_mb = io_counters.write_bytes / (1024 * 1024)
            
            # Network metrics (system-wide)
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024)
            net_recv_mb = net_io.bytes_recv / (1024 * 1024)
            
            # File descriptors
            try:
                open_fds = process.num_fds()
            except AttributeError:
                # Windows doesn't have num_fds
                open_fds = len(process.open_files())
            
            # Thread count
            thread_count = process.num_threads()
            
            # GC collections
            gc_stats = {}
            for i in range(3):  # GC generations 0, 1, 2
                gc_stats[i] = gc.get_count()[i]
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                memory_usage_mb=memory_info.rss / (1024 * 1024),
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_sent_mb=net_sent_mb,
                network_io_recv_mb=net_recv_mb,
                open_file_descriptors=open_fds,
                thread_count=thread_count,
                gc_collections=gc_stats
            )
            
            self.metrics_history.append(metrics)
            
            # Check thresholds
            self._check_resource_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def _check_resource_thresholds(self, metrics: ResourceMetrics):
        """Check if resource usage exceeds thresholds"""
        if metrics.cpu_percent > self.cpu_threshold:
            logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.memory_threshold:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Check disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        if disk_percent > self.disk_threshold:
            logger.warning(f"High disk usage: {disk_percent:.1f}%")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def register_resource_pool(self, name: str, resource_type: str, max_size: int):
        """Register a resource pool for monitoring"""
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            max_size=max_size
        )
        self.resource_pools[name] = pool
        logger.info(f"Registered resource pool: {name} ({resource_type}, max: {max_size})")
    
    def update_pool_stats(self, name: str, current_size: int, in_use: int,
                         created_count: int = None, destroyed_count: int = None):
        """Update resource pool statistics"""
        if name not in self.resource_pools:
            return
        
        pool = self.resource_pools[name]
        pool.current_size = current_size
        pool.in_use = in_use
        
        if created_count is not None:
            pool.created_count = created_count
        
        if destroyed_count is not None:
            pool.destroyed_count = destroyed_count
        
        # Update peak usage
        pool.peak_usage = max(pool.peak_usage, in_use)
    
    def get_resource_summary(self) -> dict[str, Any]:
        """Get comprehensive resource summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]
        current = recent_metrics[-1]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Pool statistics
        pool_stats = {}
        for name, pool in self.resource_pools.items():
            pool_stats[name] = {
                'current_size': pool.current_size,
                'in_use': pool.in_use,
                'utilization': pool.in_use / pool.max_size if pool.max_size > 0 else 0,
                'peak_usage': pool.peak_usage,
                'created_count': pool.created_count,
                'destroyed_count': pool.destroyed_count
            }
        
        return {
            'current': {
                'memory_usage_mb': current.memory_usage_mb,
                'memory_percent': current.memory_percent,
                'cpu_percent': current.cpu_percent,
                'thread_count': current.thread_count,
                'open_file_descriptors': current.open_file_descriptors
            },
            'averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'resource_pools': pool_stats,
            'system_info': {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'platform': sys.platform
            }
        }

class GarbageCollectionOptimizer:
    """Garbage collection optimization and tuning"""
    
    def __init__(self):
        self.gc_stats: deque = deque(maxlen=1000)
        self.optimization_enabled = True
        
        # GC thresholds (generation 0, 1, 2)
        self.gc_thresholds = gc.get_threshold()
        self.optimal_thresholds = [700, 10, 10]  # Optimized values
    
    def enable_optimization(self):
        """Enable GC optimization"""
        if not self.optimization_enabled:
            # Set optimized thresholds
            gc.set_threshold(*self.optimal_thresholds)
            self.optimization_enabled = True
            logger.info("GC optimization enabled")
    
    def disable_optimization(self):
        """Disable GC optimization"""
        if self.optimization_enabled:
            # Restore default thresholds
            gc.set_threshold(*self.gc_thresholds)
            self.optimization_enabled = False
            logger.info("GC optimization disabled")
    
    def force_collection(self) -> dict[str, int]:
        """Force garbage collection and return stats"""
        start_time = time.time()
        
        # Collect stats before
        before_objects = len(gc.get_objects())
        
        # Force collection
        collected = {
            'gen0': gc.collect(0),
            'gen1': gc.collect(1),
            'gen2': gc.collect(2)
        }
        
        # Collect stats after
        after_objects = len(gc.get_objects())
        collection_time = time.time() - start_time
        
        stats = {
            'timestamp': datetime.now(),
            'collected_objects': collected,
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_freed': before_objects - after_objects,
            'collection_time_ms': collection_time * 1000
        }
        
        self.gc_stats.append(stats)
        
        logger.info(f"GC collection freed {stats['objects_freed']} objects in {stats['collection_time_ms']:.2f}ms")
        
        return stats
    
    def get_gc_stats(self) -> dict[str, Any]:
        """Get garbage collection statistics"""
        if not self.gc_stats:
            return {}
        
        recent_stats = list(self.gc_stats)[-10:]
        
        total_freed = sum(s['objects_freed'] for s in recent_stats)
        avg_time = sum(s['collection_time_ms'] for s in recent_stats) / len(recent_stats)
        
        return {
            'optimization_enabled': self.optimization_enabled,
            'current_thresholds': gc.get_threshold(),
            'recent_collections': len(recent_stats),
            'total_objects_freed': total_freed,
            'avg_collection_time_ms': avg_time,
            'current_counts': gc.get_count(),
            'total_objects': len(gc.get_objects())
        }

class ResourceManager:
    """Main resource management coordinator"""
    
    def __init__(self):
        self.memory_tracker = MemoryTracker()
        self.resource_monitor = ResourceMonitor()
        self.gc_optimizer = GarbageCollectionOptimizer()
        
        # Cleanup tasks
        self._cleanup_tasks: list[asyncio.Task] = []
        self._running = False
        
        # Resource cleanup callbacks
        self.cleanup_callbacks: list[Callable] = []
    
    async def start(self):
        """Start resource management"""
        self._running = True
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Enable GC optimization
        self.gc_optimizer.enable_optimization()
        
        # Start cleanup tasks
        self._cleanup_tasks.append(
            asyncio.create_task(self._periodic_cleanup_loop())
        )
        
        self._cleanup_tasks.append(
            asyncio.create_task(self._memory_snapshot_loop())
        )
        
        logger.info("Resource management started")
    
    async def stop(self):
        """Stop resource management"""
        self._running = False
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Stop memory tracking
        self.memory_tracker.stop_tracking()
        
        # Cancel cleanup tasks
        for task in self._cleanup_tasks:
            task.cancel()
        
        # Run final cleanup
        await self.run_cleanup()
        
        logger.info("Resource management stopped")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    async def run_cleanup(self):
        """Run resource cleanup"""
        logger.info("Running resource cleanup")
        
        # Run registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
        
        # Force garbage collection
        self.gc_optimizer.force_collection()
        
        # Clear weak references
        self.memory_tracker.object_refs.clear()
    
    async def optimize_memory(self):
        """Perform memory optimization"""
        logger.info("Performing memory optimization")
        
        # Take memory snapshot
        snapshot = self.memory_tracker.take_snapshot()
        
        # Detect memory leaks
        leaks = self.memory_tracker.detect_memory_leaks()
        
        if leaks:
            logger.warning(f"Detected {len(leaks)} potential memory leaks")
            for leak in leaks:
                logger.warning(f"Memory leak: {leak}")
        
        # Force garbage collection
        gc_stats = self.gc_optimizer.force_collection()
        
        # Run cleanup
        await self.run_cleanup()
        
        return {
            'snapshot': snapshot,
            'leaks_detected': leaks,
            'gc_stats': gc_stats
        }
    
    async def _periodic_cleanup_loop(self):
        """Periodic cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.run_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _memory_snapshot_loop(self):
        """Periodic memory snapshot loop"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                self.memory_tracker.take_snapshot()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory snapshot loop: {e}")
    
    def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            'memory': self.memory_tracker.get_memory_stats(),
            'resources': self.resource_monitor.get_resource_summary(),
            'garbage_collection': self.gc_optimizer.get_gc_stats(),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> dict[str, Any]:
        """Assess overall system health"""
        memory_stats = self.memory_tracker.get_memory_stats()
        resource_summary = self.resource_monitor.get_resource_summary()
        
        health_score = 100.0
        issues = []
        
        # Check memory health
        if memory_stats.get('potential_leaks'):
            health_score -= 20
            issues.append("Potential memory leaks detected")
        
        if memory_stats.get('current_memory_mb', 0) > 1000:
            health_score -= 10
            issues.append("High memory usage")
        
        # Check resource health
        current = resource_summary.get('current', {})
        
        if current.get('cpu_percent', 0) > 80:
            health_score -= 15
            issues.append("High CPU usage")
        
        if current.get('memory_percent', 0) > 85:
            health_score -= 15
            issues.append("High memory percentage")
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 70:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            'health_score': max(health_score, 0),
            'status': status,
            'issues': issues,
            'recommendations': self._get_optimization_recommendations(issues)
        }
    
    def _get_optimization_recommendations(self, issues: list[str]) -> list[str]:
        """Get optimization recommendations based on issues"""
        recommendations = []
        
        if "Potential memory leaks detected" in issues:
            recommendations.append("Run memory profiling to identify leak sources")
            recommendations.append("Review object lifecycle management")
        
        if "High memory usage" in issues:
            recommendations.append("Consider increasing garbage collection frequency")
            recommendations.append("Review data structures for memory efficiency")
        
        if "High CPU usage" in issues:
            recommendations.append("Profile CPU-intensive operations")
            recommendations.append("Consider implementing connection pooling")
        
        if not recommendations:
            recommendations.append("System is running optimally")
        
        return recommendations