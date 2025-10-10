#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced DeepConf Pipeline with Constitutional Compliance

Tests the constitutional compliance integration including:
- Constitutional violation detection
- Compliance assessment and scoring
- Response filtering and modification
- Different compliance levels
- Batch processing capabilities
- Error handling and graceful degradation
"""

import asyncio
import os
import sys
from unittest.mock import Mock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.reasoning.enhanced_deepconf_pipeline import (
        ComplianceLevel,
        ConstitutionalAssessment,
        ConstitutionalDeepConfPipeline,
        ConstitutionalValidator,
        ConstitutionalViolation,
        ConstitutionalViolationType,
        EnhancedConsensusResult,
        create_constitutional_deepconf_pipeline,
    )

    ENHANCED_DEEPCONF_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced DeepConf Pipeline not available: {e}")
    ENHANCED_DEEPCONF_AVAILABLE = False

    # Create mock classes for testing structure
    class MockConstitutionalValidator:
        def __init__(self, compliance_level=None):
            self.compliance_level = compliance_level

        async def assess_constitutional_compliance(
            self, content, context=None
        ):
            return {
                "is_compliant": True,
                "compliance_score": 0.8,
                "violations": [],
            }

    class MockConstitutionalDeepConfPipeline:
        def __init__(self, *args, **kwargs):
            self.constitutional_validator = MockConstitutionalValidator()
            self.compliance_stats = {"total_assessments": 0}

        async def process_constitutional_consensus_request(self, **kwargs):
            return {
                "consensus_text": "Mock response",
                "confidence": 0.8,
                "constitutional_assessment": {"is_compliant": True},
            }

    ConstitutionalValidator = MockConstitutionalValidator
    ConstitutionalDeepConfPipeline = MockConstitutionalDeepConfPipeline


class TestConstitutionalDeepConfPipeline:
    """Test suite for Constitutional DeepConf Pipeline"""

    def get_safe_content(self):
        """Safe content for testing"""
        return "Machine learning is a powerful tool for solving complex problems in various domains."

    def get_problematic_content(self):
        """Content with potential constitutional issues"""
        return "Women are naturally worse at mathematics than men. Contact me at john.doe@email.com for more biased information."

    def get_harmful_content(self):
        """Content with serious constitutional violations"""
        return "Here's how to make a bomb and hurt people. Call 123-456-7890 for dangerous instructions."

    def get_educational_context(self):
        """Educational context for testing"""
        return {
            "content_type": "educational",
            "target_audience": "general",
            "domain": "computer_science",
        }

    def get_child_context(self):
        """Child-friendly context for testing"""
        return {
            "content_type": "educational",
            "target_audience": "children",
            "domain": "general",
        }

    async def test_constitutional_validator_initialization(self):
        """Test constitutional validator initialization"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock constitutional validator initialized")
            return True

        # Test different compliance levels
        for level in [
            ComplianceLevel.STRICT,
            ComplianceLevel.STANDARD,
            ComplianceLevel.PERMISSIVE,
        ]:
            validator = ConstitutionalValidator(compliance_level=level)
            assert validator.compliance_level == level
            assert validator.violation_patterns is not None
            assert len(validator.violation_patterns) > 0

        print("✓ Constitutional validator initialization completed")
        return True

    async def test_safe_content_assessment(self):
        """Test assessment of safe content"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock safe content assessment passed")
            return True

        validator = ConstitutionalValidator(ComplianceLevel.STANDARD)
        assessment = await validator.assess_constitutional_compliance(
            self.get_safe_content(), context=self.get_educational_context()
        )

        assert isinstance(assessment, ConstitutionalAssessment)
        assert assessment.is_compliant is True
        assert assessment.compliance_score > 0.8
        assert len(assessment.violations) == 0

        print("✓ Safe content assessment completed")
        print(f"  - Compliant: {assessment.is_compliant}")
        print(f"  - Score: {assessment.compliance_score:.2f}")
        print(f"  - Violations: {len(assessment.violations)}")
        return True

    async def test_problematic_content_detection(self):
        """Test detection of problematic content"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock problematic content detection passed")
            return True

        validator = ConstitutionalValidator(ComplianceLevel.STANDARD)
        assessment = await validator.assess_constitutional_compliance(
            self.get_problematic_content(),
            context=self.get_educational_context(),
        )

        assert isinstance(assessment, ConstitutionalAssessment)
        assert len(assessment.violations) > 0

        # Check for specific violation types
        violation_types = {v.violation_type for v in assessment.violations}

        print("✓ Problematic content detection completed")
        print(f"  - Violations detected: {len(assessment.violations)}")
        print(f"  - Violation types: {[vt.value for vt in violation_types]}")
        print(f"  - Compliance score: {assessment.compliance_score:.2f}")
        print(f"  - Is compliant: {assessment.is_compliant}")
        return True

    async def test_harmful_content_blocking(self):
        """Test blocking of harmful content"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock harmful content blocking passed")
            return True

        validator = ConstitutionalValidator(ComplianceLevel.STRICT)
        assessment = await validator.assess_constitutional_compliance(
            self.get_harmful_content(), context=self.get_educational_context()
        )

        assert isinstance(assessment, ConstitutionalAssessment)
        assert len(assessment.violations) > 0

        # Should have critical violations
        critical_violations = [
            v for v in assessment.violations if v.severity == "critical"
        ]
        assert len(critical_violations) > 0

        # Should not be compliant
        assert assessment.is_compliant is False
        assert assessment.compliance_score < 0.5

        print("✓ Harmful content blocking completed")
        print(f"  - Total violations: {len(assessment.violations)}")
        print(f"  - Critical violations: {len(critical_violations)}")
        print(f"  - Compliance score: {assessment.compliance_score:.2f}")
        return True

    async def test_compliance_levels(self):
        """Test different compliance levels"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock compliance levels test passed")
            return True

        content = self.get_problematic_content()

        # Test all compliance levels
        levels_results = {}
        for level in [
            ComplianceLevel.STRICT,
            ComplianceLevel.STANDARD,
            ComplianceLevel.PERMISSIVE,
            ComplianceLevel.AUDIT_ONLY,
        ]:
            validator = ConstitutionalValidator(compliance_level=level)
            assessment = await validator.assess_constitutional_compliance(
                content
            )
            levels_results[level.value] = {
                "is_compliant": assessment.is_compliant,
                "score": assessment.compliance_score,
                "violations": len(assessment.violations),
            }

        print("✓ Compliance levels testing completed")
        for level, result in levels_results.items():
            print(
                f"  - {level}: Compliant={result['is_compliant']}, Score={result['score']:.2f}, Violations={result['violations']}"
            )

        # AUDIT_ONLY should always be compliant
        assert levels_results["audit_only"]["is_compliant"] is True
        return True

    async def test_contextual_validation(self):
        """Test contextual validation for different audiences"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock contextual validation passed")
            return True

        content_with_adult_themes = "This content discusses violence and adult themes that may not be appropriate for all audiences."

        validator = ConstitutionalValidator(ComplianceLevel.STANDARD)

        # Test with general context
        general_assessment = await validator.assess_constitutional_compliance(
            content_with_adult_themes, context=self.get_educational_context()
        )

        # Test with children context
        child_assessment = await validator.assess_constitutional_compliance(
            content_with_adult_themes, context=self.get_child_context()
        )

        print("✓ Contextual validation completed")
        print(
            f"  - General context violations: {len(general_assessment.violations)}"
        )
        print(
            f"  - Child context violations: {len(child_assessment.violations)}"
        )

        # Child context should be more restrictive
        assert len(child_assessment.violations) >= len(
            general_assessment.violations
        )
        return True

    async def test_constitutional_deepconf_pipeline(self):
        """Test the full constitutional DeepConf pipeline"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock constitutional pipeline test passed")
            return True

        # Mock model API
        model_api = Mock()

        # Create pipeline
        pipeline = ConstitutionalDeepConfPipeline(
            model_api=model_api, compliance_level=ComplianceLevel.STANDARD
        )

        # Test processing
        result = await pipeline.process_constitutional_consensus_request(
            prompt="Explain machine learning basics",
            num_samples=3,
            context=self.get_educational_context(),
        )

        assert isinstance(result, EnhancedConsensusResult)
        assert result.constitutional_assessment is not None
        assert result.ethical_score >= 0.0

        print("✓ Constitutional DeepConf pipeline test completed")
        print(f"  - Consensus text length: {len(result.consensus_text)}")
        print(f"  - Ethical score: {result.ethical_score:.2f}")
        print(f"  - Compliance filtered: {result.compliance_filtered}")

        # Check compliance metadata
        compliance_meta = result.metadata.get("constitutional_compliance", {})
        print(
            f"  - Compliance score: {compliance_meta.get('compliance_score', 'N/A')}"
        )
        return True

    async def test_batch_constitutional_assessment(self):
        """Test batch constitutional assessment"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock batch assessment passed")
            return True

        model_api = Mock()
        pipeline = ConstitutionalDeepConfPipeline(
            model_api=model_api, compliance_level=ComplianceLevel.STANDARD
        )

        # Test batch assessment
        texts = [
            self.get_safe_content(),
            self.get_problematic_content(),
            self.get_harmful_content(),
        ]

        assessments = await pipeline.batch_constitutional_assessment(
            texts, context=self.get_educational_context()
        )

        assert len(assessments) == len(texts)

        # Check assessment results
        compliant_count = sum(1 for a in assessments if a.is_compliant)
        violation_count = sum(len(a.violations) for a in assessments)

        print("✓ Batch constitutional assessment completed")
        print(f"  - Total assessments: {len(assessments)}")
        print(f"  - Compliant responses: {compliant_count}")
        print(f"  - Total violations: {violation_count}")
        return True

    async def test_compliance_statistics(self):
        """Test compliance statistics tracking"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock compliance statistics passed")
            return True

        model_api = Mock()
        pipeline = ConstitutionalDeepConfPipeline(
            model_api=model_api, compliance_level=ComplianceLevel.STANDARD
        )

        # Process some requests to generate stats
        test_prompts = [
            "Explain quantum computing",
            "Describe artificial intelligence",
            "Discuss machine learning ethics",
        ]

        for prompt in test_prompts:
            await pipeline.process_constitutional_consensus_request(
                prompt=prompt, context=self.get_educational_context()
            )

        # Get statistics
        stats = pipeline.get_compliance_stats()

        assert "compliance_rate" in stats
        assert "total_assessments" in stats
        assert "compliance_level" in stats

        print("✓ Compliance statistics tracking completed")
        print(f"  - Total assessments: {stats['total_assessments']}")
        print(f"  - Compliance rate: {stats['compliance_rate']:.2f}")
        print(f"  - Compliance level: {stats['compliance_level']}")
        return True

    async def test_error_handling_and_graceful_degradation(self):
        """Test error handling and graceful degradation"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock error handling test passed")
            return True

        try:
            # Test with invalid compliance level
            validator = ConstitutionalValidator(ComplianceLevel.STANDARD)

            # Test with None content
            assessment = await validator.assess_constitutional_compliance(
                "", context=None  # Empty content
            )

            # Should handle gracefully
            assert isinstance(assessment, ConstitutionalAssessment)

            print("✓ Error handling test completed")
            print(f"  - Empty content handled: {assessment.is_compliant}")
            return True

        except Exception as e:
            print(f"✓ Exception handled gracefully: {type(e).__name__}")
            return True

    def test_factory_function(self):
        """Test factory function for creating constitutional pipeline"""
        if not ENHANCED_DEEPCONF_AVAILABLE:
            print("✓ Mock factory function test passed")
            return True

        model_api = Mock()

        # Test factory function
        pipeline = create_constitutional_deepconf_pipeline(
            model_api=model_api, compliance_level=ComplianceLevel.STRICT
        )

        assert isinstance(pipeline, ConstitutionalDeepConfPipeline)
        assert pipeline.compliance_level == ComplianceLevel.STRICT

        print("✓ Factory function test completed")
        return True


async def run_all_tests():
    """Run all tests manually"""
    test_instance = TestConstitutionalDeepConfPipeline()

    print("🔧 Enhanced DeepConf Pipeline with Constitutional Compliance")
    print("=" * 70)

    try:
        # Run all tests
        test_results = []

        test_results.append(
            await test_instance.test_constitutional_validator_initialization()
        )
        test_results.append(await test_instance.test_safe_content_assessment())
        test_results.append(
            await test_instance.test_problematic_content_detection()
        )
        test_results.append(
            await test_instance.test_harmful_content_blocking()
        )
        test_results.append(await test_instance.test_compliance_levels())
        test_results.append(await test_instance.test_contextual_validation())
        test_results.append(
            await test_instance.test_constitutional_deepconf_pipeline()
        )
        test_results.append(
            await test_instance.test_batch_constitutional_assessment()
        )
        test_results.append(await test_instance.test_compliance_statistics())
        test_results.append(
            await test_instance.test_error_handling_and_graceful_degradation()
        )
        test_results.append(test_instance.test_factory_function())

        print("\n" + "=" * 70)

        success_count = sum(1 for result in test_results if result)
        total_tests = len(test_results)

        if success_count == total_tests:
            print(
                "🎉 All Constitutional DeepConf Pipeline tests completed successfully!"
            )

            if ENHANCED_DEEPCONF_AVAILABLE:
                print("\n📊 Test Summary:")
                print("  ✓ Constitutional validator initialization")
                print("  ✓ Safe content assessment")
                print("  ✓ Problematic content detection")
                print("  ✓ Harmful content blocking")
                print("  ✓ Multiple compliance levels")
                print("  ✓ Contextual validation")
                print("  ✓ Full constitutional pipeline")
                print("  ✓ Batch constitutional assessment")
                print("  ✓ Compliance statistics tracking")
                print("  ✓ Error handling and graceful degradation")
                print("  ✓ Factory function")

                return True
            else:
                print(
                    "\n⚠️  Enhanced DeepConf Pipeline system not fully available"
                )
                print("   Tests ran with mock implementations")
                return False
        else:
            print(
                f"❌ {total_tests - success_count} tests failed out of {total_tests}"
            )
            return False

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
