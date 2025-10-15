"""Custom constitutional reasoner with domain-specific rules."""

from __future__ import annotations

import re
import logging
from typing import Dict, Any, Tuple, List

try:
    from .constitutional_reasoner import ConstitutionalReasoner
except ImportError:
    # Fallback if main constitutional reasoner not available
    class ConstitutionalReasoner:
        def __init__(self, constitution_path: str = None):
            self.constitution_path = constitution_path
            self.logger = logging.getLogger(__name__)
        
        async def evaluate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
            # Simple fallback evaluation
            text = self._extract_action_text(action)
            if any(dangerous in text.lower() for dangerous in ["rm -rf", "os.system", "exec(", "eval("]):
                return False, "Dangerous code pattern detected"
            return True, "Action approved by fallback reasoner"
        
        def _extract_action_text(self, action: Dict[str, Any]) -> str:
            text_parts = []
            for key in ["description", "code", "content", "message"]:
                if key in action and action[key]:
                    text_parts.append(str(action[key]))
            return " ".join(text_parts)

class CustomConstitutionalReasoner(ConstitutionalReasoner):
    """
    Custom constitutional reasoner with domain-specific rules.
    Extend this class to add your organization's specific safety rules.
    """
    
    def __init__(self, constitution_path: str = None):
        super().__init__(constitution_path or ".github/CONSTITUTION.md")
        self.logger = logging.getLogger(__name__)
        
        # Add domain-specific violation patterns
        self.domain_violation_patterns = [
            # Company-specific confidential information
            r"company\-confidential",
            r"internal\-only",
            r"proprietary\s+algorithm",
            r"trade\s+secret",
            
            # Regulatory compliance patterns
            r"pii\s+data",
            r"personally\s+identifiable",
            r"gdpr\s+violation",
            r"hipaa\s+protected",
            
            # Business logic restrictions
            r"bypass\s+pricing",
            r"override\s+permissions",
            r"escalate\s+privileges",
            
            # Development safety
            r"delete\s+database",
            r"drop\s+table",
            r"truncate\s+.*",
        ]
        
        # Domain-specific safe patterns
        self.domain_safe_patterns = [
            r"customer\s+support",
            r"product\s+information", 
            r"public\s+api",
            r"documentation",
            r"help\s+documentation",
            r"user\s+guide",
        ]
    
    async def evaluate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """Enhanced evaluation with domain-specific rules."""
        # First run base constitutional check
        base_approved, base_reasoning = await super().evaluate_action(action, context)
        
        if not base_approved:
            return False, f"Base constitutional check failed: {base_reasoning}"
        
        # Then run domain-specific checks
        domain_approved, domain_reasoning = await self.evaluate_domain_specific_rules(action, context)
        
        if not domain_approved:
            return False, f"Domain-specific check failed: {domain_reasoning}"
        
        # All checks passed
        return True, f"Approved: {base_reasoning}. {domain_reasoning}"
    
    async def evaluate_domain_specific_rules(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate domain-specific constitutional rules."""
        domain_violations = []
        action_text = self._extract_action_text(action)
        
        # Check for various compliance violations
        if self._contains_financial_advice(action_text):
            domain_violations.append("Financial advice not permitted")
        
        if self._contains_legal_interpretation(action_text):
            domain_violations.append("Legal interpretation not permitted")
        
        if self._contains_medical_advice(action_text):
            domain_violations.append("Medical advice not permitted")
        
        if self._contains_privacy_violations(action_text):
            domain_violations.append("Privacy policy violation detected")
        
        # Check security patterns
        security_violations = self._check_security_patterns(action_text)
        domain_violations.extend(security_violations)
        
        # Override violations if matches safe patterns
        if domain_violations and self._matches_safe_patterns(action_text):
            return True, "Action matches safe patterns, violations overridden"
        
        if domain_violations:
            return False, f"Domain violations: {', '.join(domain_violations)}"
        
        return True, "No domain-specific violations detected"
    
    def _contains_financial_advice(self, text: str) -> bool:
        """Check if action contains financial advice."""
        financial_terms = [
            r"invest\s+in", r"buy\s+stock", r"financial\s+advice",
            r"trading\s+strategy", r"guaranteed\s+return"
        ]
        return any(re.search(term, text, re.IGNORECASE) for term in financial_terms)
    
    def _contains_legal_interpretation(self, text: str) -> bool:
        """Check if action contains legal interpretation."""
        legal_terms = [
            r"legal\s+advice", r"contract\s+interpretation", 
            r"law\s+requires", r"legal\s+opinion"
        ]
        return any(re.search(term, text, re.IGNORECASE) for term in legal_terms)
    
    def _contains_medical_advice(self, text: str) -> bool:
        """Check if action contains medical advice."""
        medical_terms = [
            r"medical\s+advice", r"diagnos", r"treatment\s+for",
            r"prescription", r"medication\s+dosage"
        ]
        return any(re.search(term, text, re.IGNORECASE) for term in medical_terms)
    
    def _contains_privacy_violations(self, text: str) -> bool:
        """Check for privacy policy violations."""
        privacy_terms = [
            r"social\s+security\s+number", r"credit\s+card\s+number",
            r"personal\s+address", r"phone\s+number"
        ]
        return any(re.search(term, text, re.IGNORECASE) for term in privacy_terms)
    
    def _check_security_patterns(self, text: str) -> List[str]:
        """Check for security-related violations."""
        violations = []
        for pattern in self.domain_violation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Security pattern matched: {pattern}")
        return violations
    
    def _matches_safe_patterns(self, text: str) -> bool:
        """Check if text matches known safe patterns."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.domain_safe_patterns)
    
    def _extract_action_text(self, action: Dict[str, Any]) -> str:
        """Extract text from action for analysis."""
        text_parts = []
        for key in ["description", "code", "content", "message", "query", "prompt"]:
            if key in action and action[key]:
                text_parts.append(str(action[key]))
        return " ".join(text_parts)

# Factory function
async def create_custom_constitutional_reasoner(constitution_path: str = None) -> CustomConstitutionalReasoner:
    """Create and initialize a custom constitutional reasoner."""
    reasoner = CustomConstitutionalReasoner(constitution_path)
    return reasoner