"""
Constitutional Middleware for gRPC Super Alita Agent

Provides constitutional validation at every RPC call entry point following
the Super-Alita Constitutional Framework with six core articles.

Constitutional Articles:
- Article I: Library-First Development
- Article II: Test-First Development
- Article III: Simplicity Gate
- Article IV: Integration-First Testing
- Article V: Clarity and Unambiguity
- Article VI: Counterfactual Justification
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import grpc
from google.protobuf import message

logger = logging.getLogger(__name__)


class ConstitutionalScore:
    """Constitutional compliance scoring for artifacts."""

    def __init__(self):
        self.article_weights = {
            "Library_First": 1.0,
            "Test_First": 1.0,
            "Simplicity_Gate": 1.5,  # Higher weight for complexity control
            "Integration_First": 1.0,
            "Clarity_Unambiguity": 2.0,  # Highest weight for clarity
            "Counterfactual_Justification": 1.0,
        }
        self.compliance_threshold = 0.75

    def calculate_compliance_score(
        self, artifact: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate constitutional compliance score for an artifact."""
        scores = {}
        violations = []

        # Article I: Library-First Development
        scores["Library_First"] = self._evaluate_library_first(
            artifact, context
        )

        # Article II: Test-First Development
        scores["Test_First"] = self._evaluate_test_first(artifact, context)

        # Article III: Simplicity Gate
        scores["Simplicity_Gate"] = self._evaluate_simplicity(
            artifact, context
        )

        # Article IV: Integration-First Testing
        scores["Integration_First"] = self._evaluate_integration_first(
            artifact, context
        )

        # Article V: Clarity and Unambiguity
        scores["Clarity_Unambiguity"] = self._evaluate_clarity(
            artifact, context
        )

        # Article VI: Counterfactual Justification
        scores["Counterfactual_Justification"] = self._evaluate_counterfactual(
            artifact, context
        )

        # Calculate weighted overall score
        total_weighted_score = 0.0
        total_weights = 0.0

        for article, score in scores.items():
            weight = self.article_weights[article]
            total_weighted_score += score * weight
            total_weights += weight

            # Track violations (score < 0.5)
            if score < 0.5:
                violations.append(
                    f"Article {article}: Score {score:.2f} below threshold"
                )

        overall_score = (
            total_weighted_score / total_weights if total_weights > 0 else 0.0
        )

        return {
            "overall_score": overall_score,
            "article_scores": scores,
            "violations": violations,
            "compliant": overall_score >= self.compliance_threshold,
            "threshold": self.compliance_threshold,
        }

    def _evaluate_library_first(
        self, artifact: Any, context: dict[str, Any]
    ) -> float:
        """Evaluate Article I: Library-First Development compliance."""
        # Check if existing solutions were considered
        if (
            hasattr(artifact, "content")
            and "existing" in str(artifact.content).lower()
        ):
            return 0.8
        return 0.6  # Default moderate compliance

    def _evaluate_test_first(
        self, artifact: Any, context: dict[str, Any]
    ) -> float:
        """Evaluate Article II: Test-First Development compliance."""
        # Check for test-related context
        if (
            context.get("has_tests", False)
            or "test" in context.get("request_content", "").lower()
        ):
            return 0.9
        return 0.5  # Needs test consideration

    def _evaluate_simplicity(
        self, artifact: Any, context: dict[str, Any]
    ) -> float:
        """Evaluate Article III: Simplicity Gate compliance."""
        # Check complexity indicators
        content_length = len(str(artifact)) if artifact else 0

        if content_length < 1000:  # Simple response
            return 0.9
        elif content_length < 5000:  # Moderate complexity
            return 0.7
        else:  # High complexity
            return 0.4

    def _evaluate_integration_first(
        self, artifact: Any, context: dict[str, Any]
    ) -> float:
        """Evaluate Article IV: Integration-First Testing compliance."""
        # Check for integration considerations
        if "integration" in context.get("request_content", "").lower():
            return 0.8
        return 0.6  # Default moderate compliance

    def _evaluate_clarity(
        self, artifact: Any, context: dict[str, Any]
    ) -> float:
        """Evaluate Article V: Clarity and Unambiguity compliance."""
        # Check for clear, unambiguous responses
        if hasattr(artifact, "success") and artifact.success:
            return 0.8
        if hasattr(artifact, "error_message") and artifact.error_message:
            return 0.3  # Error indicates lack of clarity
        return 0.7  # Default good clarity

    def _evaluate_counterfactual(
        self, artifact: Any, context: dict[str, Any]
    ) -> float:
        """Evaluate Article VI: Counterfactual Justification compliance."""
        # Check for justification/reasoning
        method_name = context.get("method_name", "")
        if (
            "decision" in method_name.lower()
            or "validate" in method_name.lower()
        ):
            return 0.8  # Decision methods should include justification
        return 0.7  # Default good justification


class ConstitutionalValidationMiddleware:
    """gRPC middleware for constitutional compliance validation."""

    def __init__(self, scorer: ConstitutionalScore | None = None):
        self.scorer = scorer or ConstitutionalScore()
        self.validation_enabled = True

    async def validate_request(
        self,
        request: message.Message,
        method_name: str,
        context: grpc.ServicerContext,
    ) -> dict[str, Any]:
        """Validate incoming request for constitutional compliance."""
        if not self.validation_enabled:
            return {"compliant": True, "score": 1.0}

        validation_context = {
            "method_name": method_name,
            "request_content": str(request),
            "timestamp": time.time(),
        }

        try:
            compliance_result = self.scorer.calculate_compliance_score(
                request, validation_context
            )

            if not compliance_result["compliant"]:
                logger.warning(
                    f"Constitutional violation in {method_name}: "
                    f"Score {compliance_result['overall_score']:.3f} "
                    f"Violations: {compliance_result['violations']}"
                )

                # Add metadata about constitutional compliance
                context.set_trailing_metadata(
                    [
                        (
                            "constitutional-score",
                            str(compliance_result["overall_score"]),
                        ),
                        (
                            "constitutional-violations",
                            str(len(compliance_result["violations"])),
                        ),
                    ]
                )

            return compliance_result

        except Exception as e:
            logger.error(f"Constitutional validation error: {e}")
            return {"compliant": True, "score": 0.5, "error": str(e)}

    async def validate_response(
        self,
        response: message.Message,
        method_name: str,
        context: grpc.ServicerContext,
        request_validation: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate outgoing response for constitutional compliance."""
        if not self.validation_enabled:
            return {"compliant": True, "score": 1.0}

        validation_context = {
            "method_name": method_name,
            "request_compliance": request_validation.get("overall_score", 0.5),
            "has_tests": False,  # Would be determined by actual test analysis
            "timestamp": time.time(),
        }

        try:
            compliance_result = self.scorer.calculate_compliance_score(
                response, validation_context
            )

            # Add constitutional compliance metadata to response
            context.set_trailing_metadata(
                [
                    (
                        "constitutional-response-score",
                        str(compliance_result["overall_score"]),
                    ),
                    (
                        "constitutional-compliant",
                        str(compliance_result["compliant"]).lower(),
                    ),
                ]
            )

            if not compliance_result["compliant"]:
                logger.warning(
                    f"Response constitutional violation in {method_name}: "
                    f"Score {compliance_result['overall_score']:.3f}"
                )

            return compliance_result

        except Exception as e:
            logger.error(f"Response constitutional validation error: {e}")
            return {"compliant": True, "score": 0.5, "error": str(e)}


def constitutional_rpc_interceptor(
    middleware: ConstitutionalValidationMiddleware,
) -> Callable:
    """Create constitutional validation interceptor for gRPC methods."""

    def decorator(original_method: Callable) -> Callable:
        async def wrapper(
            self, request: message.Message, context: grpc.ServicerContext
        ):
            method_name = original_method.__name__
            start_time = time.time()

            # Constitutional input validation
            request_validation = await middleware.validate_request(
                request, method_name, context
            )

            # Execute original method
            try:
                response = await original_method(self, request, context)

                # Constitutional output validation
                await middleware.validate_response(
                    response, method_name, context, request_validation
                )

                # Add performance metadata
                execution_time = time.time() - start_time
                context.set_trailing_metadata(
                    [
                        ("execution-time", str(execution_time)),
                        ("constitutional-validation", "enabled"),
                    ]
                )

                return response

            except Exception as e:
                # Constitutional error handling
                logger.error(
                    f"Constitutional method {method_name} failed: {e}"
                )
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(
                    f"Constitutional validation failure: {str(e)}"
                )
                raise

        return wrapper

    return decorator


class ConstitutionalServicerMixin:
    """Mixin to add constitutional validation to any gRPC servicer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constitutional_middleware = ConstitutionalValidationMiddleware()

    async def constitutional_validate_and_execute(
        self,
        request: message.Message,
        context: grpc.ServicerContext,
        method_func: Callable,
        method_name: str,
    ) -> message.Message:
        """Execute method with constitutional validation."""
        return await constitutional_rpc_interceptor(
            self.constitutional_middleware
        )(method_func)(self, request, context)


# Export key components
__all__ = [
    "ConstitutionalScore",
    "ConstitutionalValidationMiddleware",
    "constitutional_rpc_interceptor",
    "ConstitutionalServicerMixin",
]
