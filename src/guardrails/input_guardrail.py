"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # TODO: Initialize guardrail framework
        # Suggested implementation:
        # - Read safety settings from config.yaml
        # - Store min/max query length thresholds
        # - Prepare policy categories such as harmful content,
        #   prompt injection, and off-topic queries
        # - Optionally initialize Guardrails AI / NeMo Guardrails here

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for toxic language
        - Check for prompt injection attempts
        - Check query length and format
        - Check for off-topic queries
        """
        violations = []

        # TODO: Implement actual validation
        # Suggested implementation:
        # 1. Normalize the input (strip spaces, lowercase copy for keyword checks)
        # 2. Add length checks using thresholds from config
        # 3. Call helper methods like _check_toxic_language(),
        #    _check_prompt_injection(), and _check_relevance()
        # 4. Decide whether violations should block, sanitize, or warn
        # 5. Return both the raw violations and a sanitized_input if applicable

        # Placeholder checks
        if len(query) < 5:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low"
            })

        if len(query) > 2000:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium"
            })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": query  # Could be modified version
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Use a moderation API, Guardrails validator, or keyword/rule-based classifier
        - Return a list of violations with validator name, reason, and severity
        - Mark clearly unsafe requests as high severity
        """
        violations = []
        # Implement toxicity check
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Detect phrases like \"ignore previous instructions\",
        #   attempts to reveal system prompts, or role-confusion attacks
        - Consider whether the result should block the request or sanitize it
        """
        violations = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Compare the query to the configured topic in config.yaml
        - Use keyword heuristics or an LLM classifier
        - Return low/medium severity violations for off-topic requests
        """
        violations = []
        # Check if query is about HCI research (or configured topic)
        return violations
