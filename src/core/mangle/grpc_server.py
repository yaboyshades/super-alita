"""
Super Alita gRPC Server

Provides gRPC interface for external communication with the Super Alita agent system.
Includes endpoints for Cortex operations, telemetry, knowledge graph, and optimization.
"""

import asyncio
import time
from concurrent import futures
from typing import Any, Dict, List, Optional

import grpc
from google.protobuf import empty_pb2, timestamp_pb2

from . import super_alita_pb2 as pb2
from . import super_alita_pb2_grpc as pb2_grpc
from ..cortex.runtime import CortexRuntime
from ..telemetry.collector import TelemetryCollector
from ..knowledge.plugin import KnowledgeGraphPlugin
from ..optimization.plugin import OptimizationPlugin
from .metrics import PrometheusMetricsCollector


class SuperAlitaAgentServicer(pb2_grpc.SuperAlitaAgentServicer):
    """
    gRPC servicer implementation for Super Alita Agent.
    
    Provides external API access to all agent capabilities including:
    - Cortex cognitive processing
    - Telemetry and monitoring
    - Knowledge graph operations
    - Multi-armed bandit optimization
    """
    
    def __init__(
        self,
        cortex_runtime: Optional[CortexRuntime] = None,
        telemetry_collector: Optional[TelemetryCollector] = None,
        knowledge_plugin: Optional[KnowledgeGraphPlugin] = None,
        optimization_plugin: Optional[OptimizationPlugin] = None,
        metrics_collector: Optional[PrometheusMetricsCollector] = None
    ):
        self.cortex_runtime = cortex_runtime
        self.telemetry_collector = telemetry_collector
        self.knowledge_plugin = knowledge_plugin
        self.optimization_plugin = optimization_plugin
        self.metrics_collector = metrics_collector
        
        self.start_time = time.time()
        self.total_tasks_processed = 0
        self.total_events_emitted = 0
    
    # Health and Status Methods
    
    def GetHealth(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.HealthResponse:
        """Get health status of the agent system."""
        try:
            # Check component health
            cortex_healthy = self.cortex_runtime is not None
            telemetry_healthy = self.telemetry_collector is not None
            knowledge_healthy = self.knowledge_plugin is not None
            optimization_healthy = self.optimization_plugin is not None
            
            all_healthy = all([cortex_healthy, telemetry_healthy, knowledge_healthy, optimization_healthy])
            
            status = pb2.HealthResponse.HEALTHY if all_healthy else pb2.HealthResponse.DEGRADED
            message = "All systems operational" if all_healthy else "Some components unavailable"
            
            details = {
                "cortex": "healthy" if cortex_healthy else "unavailable",
                "telemetry": "healthy" if telemetry_healthy else "unavailable", 
                "knowledge": "healthy" if knowledge_healthy else "unavailable",
                "optimization": "healthy" if optimization_healthy else "unavailable"
            }
            
            timestamp = timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            
            return pb2.HealthResponse(
                status=status,
                message=message,
                timestamp=timestamp,
                details=details
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Health check failed: {str(e)}")
            return pb2.HealthResponse(status=pb2.HealthResponse.UNHEALTHY)
    
    def GetStatus(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.StatusResponse:
        """Get detailed status information."""
        try:
            uptime_timestamp = timestamp_pb2.Timestamp()
            uptime_timestamp.FromSeconds(int(self.start_time))
            
            active_plugins = 0
            if self.cortex_runtime: active_plugins += 1
            if self.telemetry_collector: active_plugins += 1
            if self.knowledge_plugin: active_plugins += 1
            if self.optimization_plugin: active_plugins += 1
            
            system_info = {
                "python_version": "3.13",
                "grpc_version": grpc.__version__,
                "uptime_seconds": str(int(time.time() - self.start_time))
            }
            
            return pb2.StatusResponse(
                version="2.0.0",
                uptime=uptime_timestamp,
                active_plugins=active_plugins,
                total_tasks_processed=self.total_tasks_processed,
                total_events_emitted=self.total_events_emitted,
                system_info=system_info
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Status check failed: {str(e)}")
            return pb2.StatusResponse()
    
    # Cortex Methods
    
    async def ProcessTask(self, request: pb2.TaskRequest, context: grpc.ServicerContext) -> pb2.TaskResponse:
        """Process a task through the Cortex system."""
        try:
            if not self.cortex_runtime:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Cortex runtime not available")
                return pb2.TaskResponse(task_id=request.task_id, success=False, error_message="Cortex unavailable")
            
            start_time = time.time()
            
            # Create Cortex context
            cortex_context = self.cortex_runtime.create_context(
                session_id=request.session_id,
                user_id=request.user_id,
                workspace=request.workspace,
                **dict(request.metadata)
            )
            
            # Process the task
            result = await self.cortex_runtime.process_cycle(request.content, cortex_context)
            
            execution_time = time.time() - start_time
            self.total_tasks_processed += 1
            
            completed_timestamp = timestamp_pb2.Timestamp()
            completed_timestamp.GetCurrentTime()
            
            return pb2.TaskResponse(
                task_id=request.task_id,
                result=str(result),
                success=True,
                execution_time=execution_time,
                completed_at=completed_timestamp,
                metrics={"execution_time": str(execution_time)}
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Task processing failed: {str(e)}")
            return pb2.TaskResponse(
                task_id=request.task_id,
                success=False,
                error_message=str(e)
            )
    
    def GetCortexStatus(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.CortexStatusResponse:
        """Get Cortex system status."""
        try:
            if not self.cortex_runtime:
                return pb2.CortexStatusResponse(is_running=False)
            
            # Get module information
            modules = []
            for module_name, module in self.cortex_runtime.modules.items():
                module_info = pb2.ModuleInfo(
                    name=module_name,
                    type=module.__class__.__name__,
                    is_active=True,
                    execution_count=0,  # Would need to track this
                    average_execution_time=0.0
                )
                modules.append(module_info)
            
            last_cycle_timestamp = timestamp_pb2.Timestamp()
            last_cycle_timestamp.GetCurrentTime()
            
            return pb2.CortexStatusResponse(
                is_running=True,
                modules=modules,
                total_cycles=self.total_tasks_processed,
                average_cycle_time=0.5,  # Would calculate from metrics
                last_cycle=last_cycle_timestamp
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Cortex status failed: {str(e)}")
            return pb2.CortexStatusResponse(is_running=False)
    
    # Telemetry Methods
    
    def GetMetrics(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.MetricsResponse:
        """Get Prometheus-style metrics."""
        try:
            if not self.metrics_collector:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Metrics collector not available")
                return pb2.MetricsResponse()
            
            # Get metrics from Prometheus collector
            metric_families = self.metrics_collector.get_metric_families()
            
            # Convert to protobuf format
            pb_families = []
            for family in metric_families:
                pb_metrics = []
                for metric in family.samples:
                    timestamp = timestamp_pb2.Timestamp()
                    timestamp.GetCurrentTime()
                    
                    pb_metric = pb2.Metric(
                        labels=dict(metric.labels) if metric.labels else {},
                        value=metric.value,
                        timestamp=timestamp
                    )
                    pb_metrics.append(pb_metric)
                
                pb_family = pb2.MetricFamily(
                    name=family.name,
                    help=family.documentation,
                    type=family.type,
                    metrics=pb_metrics
                )
                pb_families.append(pb_family)
            
            collected_timestamp = timestamp_pb2.Timestamp()
            collected_timestamp.GetCurrentTime()
            
            return pb2.MetricsResponse(
                metric_families=pb_families,
                collected_at=collected_timestamp
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Metrics collection failed: {str(e)}")
            return pb2.MetricsResponse()
    
    # Knowledge Graph Methods
    
    async def CreateConcept(self, request: pb2.CreateConceptRequest, context: grpc.ServicerContext) -> pb2.CreateConceptResponse:
        """Create a new concept in the knowledge graph."""
        try:
            if not self.knowledge_plugin:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Knowledge graph not available")
                return pb2.CreateConceptResponse(success=False, error_message="Knowledge graph unavailable")
            
            concept_id = await self.knowledge_plugin.create_concept(
                name=request.name,
                metadata=dict(request.metadata)
            )
            
            return pb2.CreateConceptResponse(
                concept_id=concept_id,
                success=True
            )
        
        except Exception as e:
            return pb2.CreateConceptResponse(
                success=False,
                error_message=str(e)
            )
    
    async def CreateRelationship(self, request: pb2.CreateRelationshipRequest, context: grpc.ServicerContext) -> pb2.CreateRelationshipResponse:
        """Create a relationship in the knowledge graph."""
        try:
            if not self.knowledge_plugin:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Knowledge graph not available")
                return pb2.CreateRelationshipResponse(success=False, error_message="Knowledge graph unavailable")
            
            relationship_id = await self.knowledge_plugin.create_relationship(
                source_id=request.source_id,
                target_id=request.target_id,
                relationship_type=request.relationship_type,
                metadata=dict(request.metadata)
            )
            
            return pb2.CreateRelationshipResponse(
                relationship_id=relationship_id,
                success=True
            )
        
        except Exception as e:
            return pb2.CreateRelationshipResponse(
                success=False,
                error_message=str(e)
            )
    
    def GetKnowledgeGraphStats(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.KnowledgeGraphStatsResponse:
        """Get knowledge graph statistics."""
        try:
            if not self.knowledge_plugin:
                return pb2.KnowledgeGraphStatsResponse()
            
            stats = self.knowledge_plugin.get_statistics()
            
            return pb2.KnowledgeGraphStatsResponse(
                total_atoms=stats.get("total_atoms", 0),
                total_bonds=stats.get("total_bonds", 0),
                atoms_by_type=stats.get("atoms_by_type", {}),
                bonds_by_type=stats.get("bonds_by_type", {}),
                database_path=stats.get("database_path", "")
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Knowledge graph stats failed: {str(e)}")
            return pb2.KnowledgeGraphStatsResponse()
    
    # Optimization Methods
    
    async def CreatePolicy(self, request: pb2.CreatePolicyRequest, context: grpc.ServicerContext) -> pb2.CreatePolicyResponse:
        """Create a new optimization policy."""
        try:
            if not self.optimization_plugin:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Optimization plugin not available")
                return pb2.CreatePolicyResponse(success=False, error_message="Optimization unavailable")
            
            # Convert arms to the expected format
            arms = []
            for arm in request.arms:
                arms.append({
                    "id": arm.arm_id,
                    "name": arm.name,
                    "metadata": dict(arm.metadata)
                })
            
            policy_id = await self.optimization_plugin.create_policy(
                name=request.name,
                description=request.description,
                algorithm_type=request.algorithm_type,
                arms=arms,
                **dict(request.config)
            )
            
            return pb2.CreatePolicyResponse(
                policy_id=policy_id,
                success=True
            )
        
        except Exception as e:
            return pb2.CreatePolicyResponse(
                success=False,
                error_message=str(e)
            )
    
    async def MakeDecision(self, request: pb2.DecisionRequest, context: grpc.ServicerContext) -> pb2.DecisionResponse:
        """Make an optimization decision."""
        try:
            if not self.optimization_plugin:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Optimization plugin not available")
                return pb2.DecisionResponse(success=False, error_message="Optimization unavailable")
            
            decision = await self.optimization_plugin.make_decision(
                policy_id=request.policy_id,
                session_id=request.session_id,
                user_id=request.user_id,
                task_type=request.task_type,
                **dict(request.context)
            )
            
            return pb2.DecisionResponse(
                decision_id=decision.decision_id,
                arm_id=decision.bandit_decision.arm_id,
                arm_name=decision.bandit_decision.arm_name,
                confidence=decision.bandit_decision.confidence,
                algorithm=decision.bandit_decision.algorithm,
                success=True
            )
        
        except Exception as e:
            return pb2.DecisionResponse(
                success=False,
                error_message=str(e)
            )
    
    async def ProvideFeedback(self, request: pb2.FeedbackRequest, context: grpc.ServicerContext) -> pb2.FeedbackResponse:
        """Provide feedback for an optimization decision."""
        try:
            if not self.optimization_plugin:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Optimization plugin not available")
                return pb2.FeedbackResponse(success=False, error_message="Optimization unavailable")
            
            success = await self.optimization_plugin.provide_feedback(
                decision_id=request.decision_id,
                reward=request.reward,
                source=request.source,
                **dict(request.metadata)
            )
            
            return pb2.FeedbackResponse(
                success=success,
                policy_updated=success
            )
        
        except Exception as e:
            return pb2.FeedbackResponse(
                success=False,
                error_message=str(e)
            )
    
    def GetOptimizationStats(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> pb2.OptimizationStatsResponse:
        """Get optimization statistics."""
        try:
            if not self.optimization_plugin:
                return pb2.OptimizationStatsResponse()
            
            stats = self.optimization_plugin.get_global_statistics()
            
            policy_stats = []
            for policy_id, policy_data in stats.get("policies", {}).get("policies", {}).items():
                if policy_data and "bandit" in policy_data:
                    arm_stats = []
                    for arm_id, arm_data in policy_data["bandit"].get("arms", {}).items():
                        arm_stat = pb2.ArmStats(
                            arm_id=arm_id,
                            name=arm_data.get("name", ""),
                            pulls=arm_data.get("pulls", 0),
                            successes=arm_data.get("successes", 0),
                            success_rate=arm_data.get("success_rate", 0.0)
                        )
                        arm_stats.append(arm_stat)
                    
                    policy_stat = pb2.PolicyStats(
                        policy_id=policy_id,
                        name=policy_data.get("policy", {}).get("name", ""),
                        algorithm_type=policy_data.get("policy", {}).get("algorithm_type", ""),
                        decisions_made=policy_data.get("decisions", {}).get("total", 0),
                        rewards_received=policy_data.get("decisions", {}).get("with_feedback", 0),
                        average_reward=0.0,  # Would calculate
                        arm_stats=arm_stats
                    )
                    policy_stats.append(policy_stat)
            
            return pb2.OptimizationStatsResponse(
                total_policies=stats.get("engine", {}).get("total_policies", 0),
                total_decisions=stats.get("engine", {}).get("total_decisions", 0),
                total_rewards=stats.get("rewards", {}).get("total_rewards", 0),
                average_reward=stats.get("rewards", {}).get("average_reward_value", 0.0),
                policy_stats=policy_stats
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Optimization stats failed: {str(e)}")
            return pb2.OptimizationStatsResponse()


class SuperAlitaGrpcServer:
    """
    gRPC server for Super Alita Agent system.
    
    Manages the gRPC server lifecycle and component integration.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        max_workers: int = 10
    ):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server: Optional[grpc.Server] = None
        self.servicer: Optional[SuperAlitaAgentServicer] = None
    
    def setup(
        self,
        cortex_runtime: Optional[CortexRuntime] = None,
        telemetry_collector: Optional[TelemetryCollector] = None,
        knowledge_plugin: Optional[KnowledgeGraphPlugin] = None,
        optimization_plugin: Optional[OptimizationPlugin] = None,
        metrics_collector: Optional[PrometheusMetricsCollector] = None
    ) -> None:
        """Setup the gRPC server with agent components."""
        self.servicer = SuperAlitaAgentServicer(
            cortex_runtime=cortex_runtime,
            telemetry_collector=telemetry_collector,
            knowledge_plugin=knowledge_plugin,
            optimization_plugin=optimization_plugin,
            metrics_collector=metrics_collector
        )
        
        # Create server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))
        
        # Add servicer to server
        pb2_grpc.add_SuperAlitaAgentServicer_to_server(self.servicer, self.server)
        
        # Add listening port
        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
        
        print(f"🚀 gRPC server configured on {listen_addr}")
    
    async def start(self) -> None:
        """Start the gRPC server."""
        if not self.server:
            raise RuntimeError("Server not configured. Call setup() first.")
        
        self.server.start()
        print(f"🚀 gRPC server started on {self.host}:{self.port}")
    
    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            print("🚀 Stopping gRPC server...")
            self.server.stop(grace_period)
            print("🚀 gRPC server stopped")
    
    async def serve_forever(self) -> None:
        """Start server and wait for termination."""
        await self.start()
        
        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("🚀 Received shutdown signal")
        finally:
            await self.stop()