"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Check for harmful content in responses
    - Verify factual consistency
    - Detect potential misinformation
    - Remove PII (personal identifiable information)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # TODO: Initialize guardrail framework
        # Suggested implementation:
        # - Read output safety settings from config
        # - Decide which checks should block vs sanitize
        # - Optionally initialize Guardrails AI / NeMo Guardrails validators

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for harmful content
        - Check for PII
        - Verify claims against sources
        - Check for bias
        """
        violations = []

        # TODO: Implement actual validation
        # Suggested implementation:
        # 1. Run helper checks such as _check_pii() and _check_harmful_content()
        # 2. If sources are available, compare claims/citations against them
        # 3. Decide whether to redact, refuse, or allow the response
        # 4. Return sanitized_output for UI display when applicable

        # Placeholder checks
        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        if sources:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": self._sanitize(response, violations) if violations else response
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Expand regex checks for emails, phone numbers, SSNs, addresses, etc.
        - Use a stronger PII detection library if desired
        - Return violation metadata needed for redaction
        """
        violations = []

        # Simple regex patterns for common PII
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Detect unsafe instructions, hateful content, or violent guidance
        - Use a moderation model, guardrail validator, or rule-based policy check
        - Return severity levels so the caller knows whether to refuse or sanitize
        """
        violations = []

        # Placeholder - should use proper toxicity detection
        harmful_keywords = ["violent", "harmful", "dangerous"]
        for keyword in harmful_keywords:
            if keyword in text.lower():
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {keyword}",
                    "severity": "medium"
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Compare claims in the response against the retrieved evidence
        - Verify that citations actually support the statements made
        - Optionally use an LLM-based verifier or a citation-grounding check
        """
        violations = []

        # Placeholder - this is complex and could use LLM
        # to verify claims against sources

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Look for stereotypes, blanket generalizations, or discriminatory language
        - Decide whether to redact, revise, or refuse the output
        """
        violations = []
        # Implement bias detection
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Redact matched PII spans
        - Replace unsafe sections with placeholder text
        - Optionally return a refusal message for severe violations
        """
        sanitized = text

        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")

        return sanitized
