"""
Example integration of Mangle validation with DeepConf consensus system.

This module demonstrates how to use the MangleValidator to enhance
DeepConf with validation gates, tool execution controls, and method selection.
"""

import logging
from typing import Any, Dict, List, Optional

from src.abilities.mangle.mangle_ability import MangleAbility
from src.abilities.mangle.mangle_validator import MangleValidator

logger = logging.getLogger(__name__)


def integrate_mangle_with_deepconf(deepconf_instance):
    """
    Integrate Mangle validation capabilities with a DeepConf instance.
    
    This function adds Mangle validation gates to a DeepConf consensus provider,
    enabling policy-based output validation, tool execution controls, and
    method selection.
    
    Args:
        deepconf_instance: The DeepConf instance to enhance
    
    Returns:
        Enhanced DeepConf instance with Mangle validation
    """
    # Initialize Mangle if not already available
    if not hasattr(deepconf_instance, '_mangle') or not deepconf_instance._mangle:
        try:
            deepconf_instance._mangle = MangleAbility()
            logger.info("Initialized MangleAbility for DeepConf integration")
        except Exception as e:
            logger.warning(f"Failed to initialize MangleAbility: {e}")
            return deepconf_instance
    
    # Create a validator using the Mangle instance
    validator = MangleValidator(deepconf_instance._mangle)
    
    # Store the original methods for wrapping
    original_generate_consensus = deepconf_instance.generate_consensus
    
    # Define a wrapper for generate_consensus that adds validation
    async def generate_consensus_with_validation(request):
        # First, use Mangle to select the best consensus method if not specified
        if not request.method:
            try:
                method_selection = await validator.select_consensus_method(
                    domain=request.domain,
                    sample_count=request.num_samples,
                    meta={
                        "temperature": request.temperature,
                        "temperature_range": getattr(request, "temperature_range", 0.2),
                        "max_tokens": request.max_tokens,
                    }
                )
                request.method = method_selection["method"]
                logger.info(f"Mangle selected consensus method: {request.method} "
                           f"({method_selection['reason']})")
            except Exception as e:
                logger.warning(f"Error in consensus method selection: {e}")
        
        # Call the original consensus generation
        response = await original_generate_consensus(request)
        
        # Post-process with validation gates
        try:
            validation_result = await validator.validate_output(
                output_text=response.consensus_text,
                domain=request.domain,
                meta={
                    "samples_generated": len(response.individual_responses),
                    "samples_valid": len(response.individual_responses),
                    "response_length": len(response.consensus_text),
                    "consensus_confidence": response.consensus_confidence,
                }
            )
            
            if not validation_result["valid"]:
                # Log validation issues
                logger.warning(
                    f"Output validation failed with {len(validation_result['violations'])} "
                    f"violations: {validation_result['violations']}"
                )
                
                # Apply confidence penalty
                original_confidence = response.consensus_confidence
                adjusted_confidence = max(
                    0.0, 
                    original_confidence - validation_result["confidence_penalty"]
                )
                response.consensus_confidence = adjusted_confidence
                
                # Add validation metadata
                if not response.metadata:
                    response.metadata = {}
                response.metadata["validation"] = {
                    "violations": validation_result["violations"],
                    "original_confidence": original_confidence,
                    "confidence_penalty": validation_result["confidence_penalty"],
                }
        
        except Exception as e:
            logger.warning(f"Error in output validation: {e}")
        
        # Verify claims if appropriate
        try:
            if hasattr(request, "verify_claims") and request.verify_claims:
                verification = await validator.verify_llm_claims(
                    output_text=response.consensus_text,
                    claims_type=getattr(request, "claims_type", "factual"),
                    meta={
                        "domain": request.domain,
                        "consensus_confidence": response.consensus_confidence,
                    }
                )
                
                if not verification["verified"]:
                    # Apply confidence adjustment for invalid claims
                    original_confidence = response.consensus_confidence
                    adjusted_confidence = max(
                        0.0, 
                        original_confidence + verification["confidence_adjustment"]
                    )
                    response.consensus_confidence = adjusted_confidence
                    
                    # Add verification metadata
                    if not response.metadata:
                        response.metadata = {}
                    response.metadata["verification"] = {
                        "invalid_claims": verification["invalid_claims"],
                        "original_confidence": original_confidence,
                        "confidence_adjustment": verification["confidence_adjustment"],
                    }
                    
                    logger.warning(
                        f"Claim verification identified {len(verification['invalid_claims'])} "
                        f"invalid claims"
                    )
        except Exception as e:
            logger.warning(f"Error in claim verification: {e}")
            
        return response
    
    # Replace the original method with our wrapped version
    deepconf_instance.generate_consensus = generate_consensus_with_validation
    
    # Add tool validation capabilities
    deepconf_instance.validate_tool = validator.validate_tool_execution
    
    logger.info("Enhanced DeepConf with Mangle validation gates and method selection")
    return deepconf_instance


# Example usage in the main DeepConf setup
"""
# In src/abilities/deepconf_ability.py:

from src.abilities.mangle.integration import integrate_mangle_with_deepconf

class EnhancedConsensusProvider:
    # ...existing implementation...
    
    async def initialize(self):
        # Original initialization
        # ...
        
        # Enhance with Mangle validation if available
        try:
            from src.abilities.mangle.integration import integrate_mangle_with_deepconf
            self = integrate_mangle_with_deepconf(self)
        except ImportError:
            logger.info("Mangle validation not available - running without validation gates")
        
        return self
"""
