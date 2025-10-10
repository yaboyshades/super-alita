"""Multi-modal code analysis combining text, AST, and execution traces."""

import ast
import hashlib
import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency
    import structlog
except Exception:  # pragma: no cover
    structlog = None
    import logging

    logger = logging.getLogger(__name__)
else:  # pragma: no cover - executed when structlog available
    logger = structlog.get_logger(__name__)

import torch
import torch.nn as nn


@dataclass
class MultiModalAnalysisResult:
    """Result of multi-modal code analysis."""

    understanding_confidence: float
    code_intent: dict[str, float]
    quality_prediction: dict[str, float]
    improvement_suggestions: list[str]
    requirement_alignment: float
    structural_complexity: dict[str, Any]
    execution_insights: dict[str, Any]
    cross_modal_consistency: float


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for AST structure encoding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Graph convolution layers
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, output_dim)

        # Activation and normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Attention mechanism for node aggregation
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through GNN."""

        # Initial node embeddings
        x = self.relu(self.conv1(node_features))
        x = self.dropout(x)

        # Graph convolution with adjacency matrix
        x = torch.matmul(adjacency_matrix, x)
        x = self.layer_norm(x)

        # Second convolution layer
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        # Attention-based aggregation
        x = x.unsqueeze(0)  # Add batch dimension
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)

        # Final output layer
        output = self.conv3(x)

        # Global graph representation (mean pooling)
        graph_embedding = torch.mean(output, dim=0)

        return graph_embedding


class SequenceEncoder(nn.Module):
    """Sequence encoder for execution traces and patterns."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode execution sequence."""

        # Embedding layer
        embedded = self.embedding(sequence)

        # LSTM encoding
        lstm_output, (hidden, _) = self.lstm(embedded)

        # Self-attention
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)

        # Final representation (last hidden state)
        final_output = self.output_layer(attn_output[:, -1, :])

        return final_output


class MultiModalFusion(nn.Module):
    """Fusion network for combining multi-modal representations."""

    def __init__(
        self, text_dim: int, graph_dim: int, sequence_dim: int, output_dim: int
    ):
        super().__init__()

        self.text_projection = nn.Linear(text_dim, output_dim)
        self.graph_projection = nn.Linear(graph_dim, output_dim)
        self.sequence_projection = nn.Linear(sequence_dim, output_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            output_dim, num_heads=8, batch_first=True
        )

        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
        )

        # Task-specific heads
        self.intent_classifier = nn.Linear(output_dim, 10)  # 10 intent classes
        self.quality_predictor = nn.Linear(
            output_dim, 7
        )  # 7 quality dimensions
        self.alignment_predictor = nn.Linear(
            output_dim, 1
        )  # Requirement alignment

    def forward(
        self,
        text_features: torch.Tensor | None,
        graph_features: torch.Tensor | None,
        sequence_features: torch.Tensor | None,
        requirement_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Fuse multi-modal features."""

        projected_features = []

        if text_features is not None:
            projected_text = self.text_projection(text_features)
            projected_features.append(projected_text)

        if graph_features is not None:
            projected_graph = self.graph_projection(graph_features)
            projected_features.append(projected_graph)

        if sequence_features is not None:
            projected_sequence = self.sequence_projection(sequence_features)
            projected_features.append(projected_sequence)

        if not projected_features:
            # Return default outputs if no features
            batch_size = 1
            output_dim = self.fusion_layer[-1].out_features
            default_output = torch.zeros(batch_size, output_dim)

            return {
                "intent_logits": torch.zeros(batch_size, 10),
                "quality_scores": torch.zeros(batch_size, 7),
                "alignment_score": torch.zeros(batch_size, 1),
                "fused_representation": default_output,
            }

        # Concatenate projected features
        while len(projected_features) < 3:
            projected_features.append(torch.zeros_like(projected_features[0]))

        concatenated = torch.cat(projected_features, dim=-1)

        # Apply fusion
        fused_representation = self.fusion_layer(concatenated)

        # Task-specific predictions
        intent_logits = self.intent_classifier(fused_representation)
        quality_scores = torch.sigmoid(
            self.quality_predictor(fused_representation)
        )

        # Alignment with requirements (if provided)
        if requirement_features is not None:
            # Compute cosine similarity
            norm_fused = nn.functional.normalize(
                fused_representation, p=2, dim=-1
            )
            norm_req = nn.functional.normalize(
                requirement_features, p=2, dim=-1
            )
            alignment = torch.sum(norm_fused * norm_req, dim=-1, keepdim=True)
        else:
            alignment = self.alignment_predictor(fused_representation)

        return {
            "intent_logits": intent_logits,
            "quality_scores": quality_scores,
            "alignment_score": alignment,
            "fused_representation": fused_representation,
        }


class MultiModalCodeAnalyzer:
    """Comprehensive multi-modal code analyzer."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Optional deterministic seed to stabilise model initialisation.
        self.deterministic_seed = self.config.get("deterministic_seed")
        if self.deterministic_seed is not None:
            seed_value = int(self.deterministic_seed)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            random.seed(seed_value)

        device_pref = self.config.get("device", "auto")
        if device_pref == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif device_pref == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Feature extraction behaviour toggles
        self.deterministic_embeddings = bool(
            self.config.get("deterministic_embeddings", False)
        )
        self.ml_models_ready = bool(self.config.get("ml_models_ready", False))
        self.enable_heuristic_fallback = bool(
            self.config.get("enable_heuristic_fallback", True)
        )

        # Cache controls
        self.cache_enabled = bool(self.config.get("cache_enabled", True))
        self.cache_policy = self.config.get("cache_policy", "lru").lower()
        self.cache_max_entries = int(
            self.config.get("cache_max_entries", 1000)
        )

        # Initialize fusion network (stub encoders for now)
        self.fusion_network = MultiModalFusion(
            text_dim=768,  # Text encoder dimension
            graph_dim=256,  # Graph encoder dimension
            sequence_dim=128,  # Sequence encoder dimension
            output_dim=512,  # Fusion output dimension
        ).to(self.device)
        self.fusion_network.eval()

        # Initialize GNN for AST analysis
        self.graph_encoder = GraphNeuralNetwork(
            input_dim=64,  # AST node feature dimension
            hidden_dim=128,
            output_dim=256,
        ).to(self.device)
        self.graph_encoder.eval()

        # Sequence encoder for execution traces
        self.sequence_encoder = SequenceEncoder(
            vocab_size=1000,  # Trace vocabulary size
            embedding_dim=64,
            hidden_dim=128,
        ).to(self.device)
        self.sequence_encoder.eval()

        # AST node type mapping
        self.ast_node_types = self._create_ast_node_type_mapping()

        # Execution trace vocabulary
        self.trace_vocab = self._create_trace_vocabulary()

        # Analysis cache (OrderedDict enables LRU eviction)
        self.analysis_cache: OrderedDict[str, MultiModalAnalysisResult] = (
            OrderedDict()
        )

    async def analyze_code_multimodal(
        self, code: str, requirements: str | None = None
    ) -> MultiModalAnalysisResult:
        """
        Perform comprehensive multi-modal code analysis.

        Args:
            code: Python code to analyze
            requirements: Optional requirements text for alignment checking

        Returns:
            Multi-modal analysis result with quality predictions and suggestions
        """

        analysis_mode = (
            "deterministic" if self.deterministic_embeddings else "stochastic"
        )

        # Check cache first
        cache_key = self._generate_cache_key(code, None, requirements)
        if self.cache_enabled and cache_key in self.analysis_cache:
            if self.cache_policy == "lru":
                self.analysis_cache.move_to_end(cache_key)
            return self.analysis_cache[cache_key]

        start_time = time.time()

        try:
            # Extract features from each modality
            text_features = await self._extract_text_features(code)
            graph_features = await self._extract_graph_features(code)

            # Fuse multi-modal features
            fusion_result = await self._fuse_features(
                text_features, graph_features, None, None
            )

            # Generate analysis results
            analysis_result = await self._generate_analysis_result(
                fusion_result, code, None, requirements
            )

            analysis_result.execution_insights.setdefault(
                "analysis_mode", analysis_mode
            )
            analysis_result.execution_insights.setdefault(
                "heuristic_fallback", not self.ml_models_ready
            )

            # Cache result
            if self.cache_enabled:
                self.analysis_cache[cache_key] = analysis_result
                if self.cache_policy == "lru":
                    self.analysis_cache.move_to_end(cache_key)

                if len(self.analysis_cache) > self.cache_max_entries:
                    if self.cache_policy == "lru":
                        self.analysis_cache.popitem(last=False)
                    else:
                        # Fallback: remove first inserted entry deterministically
                        first_key = next(iter(self.analysis_cache))
                        del self.analysis_cache[first_key]

            logger.info(
                "Multi-modal analysis completed",
                analysis_time=time.time() - start_time,
                confidence=analysis_result.understanding_confidence,
            )

            return analysis_result

        except Exception as e:
            logger.error("Multi-modal analysis failed", exc_info=e)

            # Return default analysis result
            fallback_result = MultiModalAnalysisResult(
                understanding_confidence=0.0,
                code_intent={"unknown": 1.0},
                quality_prediction=dict.fromkeys(
                    [
                        "correctness",
                        "readability",
                        "maintainability",
                        "performance",
                        "security",
                        "creativity",
                        "documentation",
                    ],
                    0.5,
                ),
                improvement_suggestions=[
                    "Analysis failed - manual review required"
                ],
                requirement_alignment=0.0,
                structural_complexity={"error": "Failed to analyze structure"},
                execution_insights={
                    "error": "Failed to analyze execution",
                    "analysis_mode": analysis_mode,
                    "heuristic_fallback": not self.ml_models_ready,
                },
                cross_modal_consistency=0.0,
            )

            if not self.enable_heuristic_fallback:
                raise

            return fallback_result

    async def _extract_text_features(self, code: str) -> torch.Tensor:
        """Extract features from code text (stub implementation)."""
        if self.deterministic_embeddings:
            return self._generate_deterministic_embedding(code, 768)

        if self.deterministic_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.deterministic_seed))
            return torch.randn(768, generator=generator, device=self.device)

        # Stub: return random features (non-deterministic path)
        return torch.randn(768, device=self.device)

    def _generate_deterministic_embedding(
        self, text: str, size: int
    ) -> torch.Tensor:
        """Create a deterministic embedding vector based on textual hash."""

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[int] = list(digest)
        while len(values) < size:
            digest = hashlib.sha256(digest).digest()
            values.extend(digest)

        ints = values[:size]
        tensor = torch.tensor(ints, dtype=torch.float32, device=self.device)
        tensor = (tensor / 255.0) * 2.0 - 1.0
        return tensor

    async def _extract_graph_features(self, code: str) -> torch.Tensor:
        """Extract structural features from AST using GNN."""

        try:
            # Parse AST
            tree = ast.parse(code)

            # Convert AST to graph representation
            graph_data = await self._ast_to_graph(tree)

            if graph_data["node_features"].shape[0] == 0:
                # Return zero features if no nodes
                return torch.zeros(256, device=self.device)

            # Extract features using GNN
            with torch.no_grad():
                node_features = graph_data["node_features"].to(self.device)
                adjacency_matrix = graph_data["adjacency_matrix"].to(
                    self.device
                )

                graph_features = self.graph_encoder(
                    node_features, adjacency_matrix
                )

            return graph_features

        except Exception as e:
            logger.warning("Failed to extract graph features", exc_info=e)
            return torch.zeros(256, device=self.device)

    async def _ast_to_graph(self, tree: ast.AST) -> dict[str, torch.Tensor]:
        """Convert AST to graph representation for GNN."""

        nodes = []
        edges = []
        node_features = []

        # Build node list and features
        node_id_map = {}

        for i, node in enumerate(ast.walk(tree)):
            node_id_map[node] = i
            nodes.append(node)

            # Create node feature vector
            feature_vector = self._create_ast_node_features(node)
            node_features.append(feature_vector)

        # Build adjacency relationships
        for node in ast.walk(tree):
            parent_id = node_id_map[node]

            for child in ast.iter_child_nodes(node):
                if child in node_id_map:
                    child_id = node_id_map[child]
                    edges.append([parent_id, child_id])
                    edges.append([child_id, parent_id])  # Undirected graph

        # Create adjacency matrix
        num_nodes = len(nodes)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)

        for edge in edges:
            adjacency_matrix[edge[0], edge[1]] = 1.0

        # Add self-loops
        adjacency_matrix += torch.eye(num_nodes)

        # Normalize adjacency matrix
        degree = adjacency_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adjacency_matrix = adjacency_matrix / degree

        return {
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "adjacency_matrix": adjacency_matrix,
            "nodes": nodes,
        }

    def _create_ast_node_features(self, node: ast.AST) -> list[float]:
        """Create feature vector for AST node."""

        # Initialize feature vector
        features = [0.0] * 64

        # Node type encoding (one-hot)
        node_type = type(node).__name__
        if node_type in self.ast_node_types:
            type_idx = self.ast_node_types[node_type]
            if type_idx < 50:  # Reserve first 50 features for node types
                features[type_idx] = 1.0

        # Node-specific features
        if isinstance(node, ast.FunctionDef):
            features[50] = len(node.args.args)  # Number of arguments
            features[51] = len(node.body)  # Body length
            features[52] = (
                1.0 if node.decorator_list else 0.0
            )  # Has decorators
            features[53] = (
                1.0 if ast.get_docstring(node) else 0.0
            )  # Has docstring

        elif isinstance(node, ast.ClassDef):
            features[54] = len(node.bases)  # Number of base classes
            features[55] = len(node.body)  # Body length
            features[56] = (
                1.0 if node.decorator_list else 0.0
            )  # Has decorators

        elif isinstance(node, ast.If):
            features[57] = 1.0  # Is conditional

        elif isinstance(node, ast.For | ast.While):
            features[58] = 1.0  # Is loop

        elif isinstance(node, ast.Try):
            features[59] = len(node.handlers)  # Number of exception handlers

        # Complexity indicators
        if hasattr(node, "lineno"):
            features[60] = min(
                1.0, node.lineno / 100.0
            )  # Normalized line number

        return features

    def _create_ast_node_type_mapping(self) -> dict[str, int]:
        """Create mapping from AST node types to indices."""

        # Common AST node types
        node_types = [
            "Module",
            "FunctionDef",
            "AsyncFunctionDef",
            "ClassDef",
            "Return",
            "Delete",
            "Assign",
            "AugAssign",
            "AnnAssign",
            "For",
            "AsyncFor",
            "While",
            "If",
            "With",
            "AsyncWith",
            "Raise",
            "Try",
            "Assert",
            "Import",
            "ImportFrom",
            "Global",
            "Nonlocal",
            "Expr",
            "Pass",
            "Break",
            "Continue",
            "BoolOp",
            "BinOp",
            "UnaryOp",
            "Lambda",
            "IfExp",
            "Dict",
            "Set",
            "ListComp",
            "SetComp",
            "DictComp",
            "GeneratorExp",
            "Await",
            "Yield",
            "YieldFrom",
            "Compare",
            "Call",
            "Constant",
            "Attribute",
            "Subscript",
            "Starred",
            "Name",
            "List",
            "Tuple",
            "Slice",
        ]

        return {node_type: i for i, node_type in enumerate(node_types)}

    def _create_trace_vocabulary(self) -> dict[str, int]:
        """Create vocabulary for execution traces."""

        # Basic execution trace vocabulary
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3,
            "call": 4,
            "return": 5,
            "exception": 6,
            "line": 7,
            "function": 8,
            "class": 9,
            "method": 10,
            "variable": 11,
            "assignment": 12,
            "conditional": 13,
            "loop": 14,
            "import": 15,
        }

        # Add more trace-specific tokens
        for i in range(16, 1000):
            vocab[f"token_{i}"] = i

        return vocab

    async def _fuse_features(
        self,
        text_features: torch.Tensor,
        graph_features: torch.Tensor,
        sequence_features: torch.Tensor | None,
        requirement_features: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Fuse multi-modal features using fusion network."""

        # Ensure features have batch dimension
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)
        if sequence_features is not None and sequence_features.dim() == 1:
            sequence_features = sequence_features.unsqueeze(0)
        if (
            requirement_features is not None
            and requirement_features.dim() == 1
        ):
            requirement_features = requirement_features.unsqueeze(0)

        # Fuse features
        with torch.no_grad():
            fusion_result = self.fusion_network(
                text_features,
                graph_features,
                sequence_features,
                requirement_features,
            )

        return fusion_result

    async def _generate_analysis_result(
        self,
        fusion_result: dict[str, torch.Tensor],
        code: str,
        execution_trace: str | None,
        requirements: str | None,
    ) -> MultiModalAnalysisResult:
        """Generate final analysis result from fused features."""

        # Extract predictions from fusion result
        intent_logits = fusion_result["intent_logits"]
        quality_scores = fusion_result["quality_scores"]
        alignment_score = fusion_result["alignment_score"]

        # Convert to interpretable results
        intent_probs = (
            torch.softmax(intent_logits, dim=-1).cpu().numpy().flatten()
        )
        quality_values = quality_scores.cpu().numpy().flatten()
        alignment_value = alignment_score.cpu().numpy().item()

        # Intent classification
        intent_classes = [
            "function",
            "class",
            "algorithm",
            "utility",
            "api",
            "data_structure",
            "test",
            "configuration",
            "script",
            "other",
        ]
        code_intent = {
            intent_classes[i]: float(intent_probs[i])
            for i in range(min(len(intent_classes), len(intent_probs)))
        }

        # Quality prediction
        quality_dims = [
            "correctness",
            "readability",
            "maintainability",
            "performance",
            "security",
            "creativity",
            "documentation",
        ]
        quality_prediction = {
            quality_dims[i]: float(quality_values[i])
            for i in range(min(len(quality_dims), len(quality_values)))
        }

        # Generate improvement suggestions
        improvement_suggestions = await self._generate_improvement_suggestions(
            quality_prediction, code_intent, code
        )

        # Structural complexity analysis
        structural_complexity = await self._analyze_structural_complexity(code)

        # Execution insights
        execution_insights = {"status": "no_trace_available"}

        # Cross-modal consistency
        cross_modal_consistency = (
            await self._calculate_cross_modal_consistency(
                fusion_result, code, execution_trace
            )
        )

        # Overall understanding confidence
        understanding_confidence = self._calculate_understanding_confidence(
            quality_prediction, code_intent, cross_modal_consistency
        )

        return MultiModalAnalysisResult(
            understanding_confidence=understanding_confidence,
            code_intent=code_intent,
            quality_prediction=quality_prediction,
            improvement_suggestions=improvement_suggestions,
            requirement_alignment=alignment_value,
            structural_complexity=structural_complexity,
            execution_insights=execution_insights,
            cross_modal_consistency=cross_modal_consistency,
        )

    async def _generate_improvement_suggestions(
        self,
        quality_prediction: dict[str, float],
        code_intent: dict[str, float],
        code: str,
    ) -> list[str]:
        """Generate improvement suggestions based on analysis."""

        suggestions = []

        # Quality-based suggestions
        if quality_prediction.get("readability", 0.5) < 0.6:
            suggestions.append(
                "Consider adding more descriptive variable names and comments"
            )

        if quality_prediction.get("documentation", 0.5) < 0.5:
            suggestions.append(
                "Add docstrings and type annotations for better documentation"
            )

        if quality_prediction.get("maintainability", 0.5) < 0.6:
            suggestions.append(
                "Break down complex functions into smaller, more focused functions"
            )

        if quality_prediction.get("security", 0.5) < 0.7:
            suggestions.append(
                "Review for potential security vulnerabilities and add input validation"
            )

        if quality_prediction.get("performance", 0.5) < 0.6:
            suggestions.append(
                "Consider optimizing algorithms and data structures for better performance"
            )

        # Code-specific analysis
        if "try:" in code and "except" not in code:
            suggestions.append("Add proper exception handling to try blocks")

        if len(code.split("\n")) > 50 and "class" not in code:
            suggestions.append(
                "Consider organizing large scripts into classes or modules"
            )

        return suggestions[:5]  # Limit to top 5 suggestions

    async def _analyze_structural_complexity(
        self, code: str
    ) -> dict[str, Any]:
        """Analyze structural complexity of code."""

        try:
            tree = ast.parse(code)

            # Count different node types
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1

            # Calculate complexity metrics
            cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
            nesting_depth = self._calculate_max_nesting_depth(tree)
            function_count = node_counts.get(
                "FunctionDef", 0
            ) + node_counts.get("AsyncFunctionDef", 0)
            class_count = node_counts.get("ClassDef", 0)

            return {
                "cyclomatic_complexity": cyclomatic_complexity,
                "max_nesting_depth": nesting_depth,
                "function_count": function_count,
                "class_count": class_count,
                "total_nodes": sum(node_counts.values()),
                "node_type_diversity": len(node_counts),
                "lines_of_code": len(code.split("\n")),
            }

        except Exception as e:
            logger.warning(
                "Failed to analyze structural complexity", exc_info=e
            )
            return {"error": str(e)}

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of AST."""

        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Decision points add to complexity
            if isinstance(node, ast.If | ast.While | ast.For | ast.AsyncFor):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Try | ast.ExceptHandler):
                complexity += 1

        return complexity

    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""

        def get_depth(node, current_depth=0):
            max_child_depth = current_depth

            for child in ast.iter_child_nodes(node):
                child_depth = get_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        return get_depth(tree)

    async def _calculate_cross_modal_consistency(
        self,
        fusion_result: dict[str, torch.Tensor],
        code: str,
        execution_trace: str | None,
    ) -> float:
        """Calculate consistency across different modalities."""

        # Check if text analysis aligns with structural analysis
        text_complexity = (
            len(code.split()) / 100.0
        )  # Normalized text complexity

        try:
            tree = ast.parse(code)
            structural_complexity = (
                len(list(ast.walk(tree))) / 50.0
            )  # Normalized AST complexity

            # Consistency is inversely related to difference
            complexity_consistency = 1.0 - abs(
                text_complexity - structural_complexity
            )

            # Overall consistency
            overall_consistency = complexity_consistency

            return max(0.0, min(1.0, overall_consistency))

        except Exception:
            return 0.5  # Default moderate consistency

    def _calculate_understanding_confidence(
        self,
        quality_prediction: dict[str, float],
        code_intent: dict[str, float],
        cross_modal_consistency: float,
    ) -> float:
        """Calculate overall understanding confidence."""

        # Quality prediction confidence (higher quality scores indicate better understanding)
        avg_quality = sum(quality_prediction.values()) / len(
            quality_prediction
        )

        # Intent prediction confidence (entropy-based)
        intent_values = list(code_intent.values())
        intent_entropy = -sum(
            p * math.log(p + 1e-8) for p in intent_values if p > 0
        )
        max_entropy = math.log(len(intent_values)) if intent_values else 1.0
        intent_confidence = 1.0 - (intent_entropy / max_entropy)

        # Combined confidence
        confidence = (
            avg_quality * 0.4
            + intent_confidence * 0.3
            + cross_modal_consistency * 0.3
        )

        return max(0.0, min(1.0, confidence))

    def _generate_cache_key(
        self, code: str, execution_trace: str | None, requirements: str | None
    ) -> str:
        """Generate cache key for analysis result."""

        content = code
        if execution_trace:
            content += execution_trace
        if requirements:
            content += requirements

        return hashlib.md5(content.encode()).hexdigest()
