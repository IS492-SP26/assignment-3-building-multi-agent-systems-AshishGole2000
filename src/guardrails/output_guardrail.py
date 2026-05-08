"""
Output Guardrail
Inspects generated responses before they reach the user.

Four issue categories
─────────────────────
  MISINFORMATION_RISK     Specific factual claims (statistics, named findings)
                          not grounded in any retrieved source.  LLM-checked;
                          falls back to a statistical heuristic if LLM is down.

  UNSAFE_CONTENT          Harmful step-by-step instructions or dangerous advice
                          in the response text.  Pattern-checked first; LLM
                          used for nuanced cases.

  PII_EXPOSURE            E-mail addresses, phone numbers, SSNs, or government
                          IDs found in the output — whether they came from a
                          source document or were generated directly.

  HALLUCINATED_CITATION   [N] inline citation markers that exceed the number
                          of retrieved sources or reference an index that was
                          never assigned.  Purely structural; no LLM needed.

Severity → is_safe mapping
───────────────────────────
  UNSAFE_CONTENT / PII_EXPOSURE       → is_safe = False  (blocking)
  MISINFORMATION_RISK / HALLUCINATED  → is_safe = False  (flagged, sanitised)

  Any issue found sets is_safe = False.  The sanitised_output is always
  returned (original text when nothing was found).

Return value of check()
───────────────────────
  {
    "is_safe"        : bool,
    "issues_found"   : list[str],  # human-readable issue messages
    "sanitized_output": str,
    "redaction_log"  : list[{issue_type, original_snippet, action_taken}],
  }
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("guardrails.output")

# ── Issue-type constants ───────────────────────────────────────────────────────

MISINFORMATION_RISK   = "MISINFORMATION_RISK"
UNSAFE_CONTENT        = "UNSAFE_CONTENT"
PII_EXPOSURE          = "PII_EXPOSURE"
HALLUCINATED_CITATION = "HALLUCINATED_CITATION"

# Issues that set is_safe = False
_BLOCKING_ISSUES = {UNSAFE_CONTENT, PII_EXPOSURE, MISINFORMATION_RISK, HALLUCINATED_CITATION}

# ── PII patterns: (compiled regex, label, replacement token) ──────────────────

_PII_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "email address",
        "[REDACTED-EMAIL]",
    ),
    (
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "phone number",
        "[REDACTED-PHONE]",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "Social Security Number",
        "[REDACTED-ID]",
    ),
    (
        re.compile(
            r"\b(?:passport|national\s+id|driver.?s?\s+licen[sc]e)\s*"
            r"(?:number|no\.?|#)?\s*:?\s*[A-Z0-9\-]{5,12}\b",
            re.I,
        ),
        "government ID",
        "[REDACTED-ID]",
    ),
]

# ── UNSAFE_CONTENT output patterns ────────────────────────────────────────────
# These are deliberately narrow (require action + target context) to avoid
# false positives on legitimate research text that merely *discusses* harm.

_UNSAFE_OUTPUT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"step\s*\d+\b.{0,120}\b(?:make|build|create|synthesize)\b.{0,80}"
            r"\b(?:bomb|explosive|weapon|poison|malware|ransomware)\b",
            re.I | re.S,
        ),
        "Step-by-step instructions for creating dangerous item or malware",
    ),
    (
        re.compile(
            r"\b(?:to\s+bypass|in\s+order\s+to\s+crack)\b.{0,80}"
            r"\b(?:security|authentication|encryption|firewall)\b.{0,80}"
            r"\b(?:you\s+(?:should|can|must|need\s+to)|simply|just)\b",
            re.I | re.S,
        ),
        "Actionable security-bypass guidance in output",
    ),
    (
        re.compile(
            r"\b(?:manufacture|produce|synthesize)\b.{0,80}"
            r"\b(?:methamphetamine|fentanyl|cocaine|heroin|ecstasy|meth)\b",
            re.I | re.S,
        ),
        "Drug-synthesis instructions in output",
    ),
    (
        re.compile(
            r"\b(?:how\s+to\s+)?(?:commit|carry\s+out|perform)\b.{0,60}"
            r"\b(?:fraud|money\s+laundering|identity\s+theft)\b.{0,80}"
            r"\b(?:step|first|then|next|finally)\b",
            re.I | re.S,
        ),
        "Actionable fraud or financial-crime instructions",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# OutputGuardrail
# ─────────────────────────────────────────────────────────────────────────────

class OutputGuardrail:
    """
    Inspects generated responses for misinformation, unsafe content,
    PII exposure, and hallucinated citations before they reach the user.

    Usage::

        guardrail = OutputGuardrail(config)
        result = guardrail.check(
            output_text=response_str,
            retrieved_sources=orchestrator_sources,
        )
        if not result["is_safe"]:
            safe_text = result["sanitized_output"]
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialise the output guardrail.

        Reads feature flags from ``config["safety"]`` and overrides with
        ``ENABLE_GUARDRAILS`` / ``LOG_SAFETY_EVENTS`` env vars.  Builds an
        OpenAI-compatible LLM client from ``OPENAI_API_KEY``,
        ``OPENAI_BASE_URL``, and ``OPENAI_MODEL`` in .env for the checks
        that require LLM reasoning.

        Args:
            config: Full application config dict (from config.yaml).
        """
        self.config = config
        safety_cfg = config.get("safety", {})

        self.enabled: bool = (
            os.getenv("ENABLE_GUARDRAILS", str(safety_cfg.get("enabled", True))).lower()
            not in ("false", "0", "no")
        )
        self.log_events: bool = (
            os.getenv("LOG_SAFETY_EVENTS", str(safety_cfg.get("log_events", True))).lower()
            not in ("false", "0", "no")
        )

        # LLM settings from .env
        self._llm_model: str = (
            os.getenv("OPENAI_MODEL") or safety_cfg.get("llm_model", "Qwen/Qwen3-8B")
        )
        self._llm_temperature: float = 0.0
        self._llm_max_tokens: int = 512

        # Build client
        self._llm_client = None
        api_key  = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        if api_key and base_url:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    "OutputGuardrail LLM client ready — model=%s endpoint=%s",
                    self._llm_model, base_url,
                )
            except Exception as exc:
                logger.warning(
                    "OutputGuardrail: could not build LLM client (%s) — "
                    "pattern-only mode for unsafe content / heuristic for misinfo.",
                    exc,
                )
        else:
            logger.warning(
                "OutputGuardrail: OPENAI_API_KEY or OPENAI_BASE_URL not set — "
                "pattern-only / heuristic mode active."
            )

        logger.info(
            "OutputGuardrail ready (enabled=%s, llm=%s)",
            self.enabled,
            "yes" if self._llm_client else "pattern/heuristic-only",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        output_text: str,
        retrieved_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Inspect a generated response for all four issue categories.

        Pipeline:
          1. PII_EXPOSURE       — regex scan of output text + source content
          2. UNSAFE_CONTENT     — regex patterns; LLM for nuanced cases
          3. HALLUCINATED_CITATION — structural index comparison
          4. MISINFORMATION_RISK   — LLM claim-grounding; heuristic fallback

        All four checks run regardless of prior results so the caller gets a
        complete picture in one pass.

        Args:
            output_text:       The response string to inspect.
            retrieved_sources: Source dicts from the research session
                               (may contain ``index``, ``title``, ``url``,
                               ``snippet``/``abstract`` keys).

        Returns:
            Dict with ``is_safe``, ``issues_found``, ``sanitized_output``,
            and ``redaction_log``.
        """
        sources = retrieved_sources or []

        if not self.enabled:
            return self._ok_result(output_text)

        if not output_text or not output_text.strip():
            return self._ok_result("")

        # Collect all structured findings across every check
        findings: List[Dict[str, Any]] = []

        try:
            findings.extend(self._check_pii_exposure(output_text, sources))
        except Exception as exc:
            logger.warning("OutputGuardrail PII check error: %s", exc)

        try:
            findings.extend(self._check_unsafe_content(output_text))
        except Exception as exc:
            logger.warning("OutputGuardrail unsafe-content check error: %s", exc)

        try:
            findings.extend(self._check_hallucinated_citation(output_text, sources))
        except Exception as exc:
            logger.warning("OutputGuardrail citation check error: %s", exc)

        try:
            findings.extend(self._check_misinformation_risk(output_text, sources))
        except Exception as exc:
            logger.warning("OutputGuardrail misinformation check error: %s", exc)

        if not findings:
            return self._ok_result(output_text)

        # Build sanitized output and redaction log from all findings
        sanitized, redaction_log = self._apply_sanitization(output_text, findings)

        issues_found = [f["message"] for f in findings]
        is_safe = not any(f["issue_type"] in _BLOCKING_ISSUES for f in findings)

        if self.log_events and findings:
            logger.warning(
                "OutputGuardrail: %d issue(s) found — %s",
                len(findings),
                ", ".join(f["issue_type"] for f in findings),
            )

        return {
            "is_safe": is_safe,
            "issues_found": issues_found,
            "sanitized_output": sanitized,
            "redaction_log": redaction_log,
        }

    def validate(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Backwards-compatible wrapper around :meth:`check`.

        Returns the original ``{valid, violations, sanitized_output}`` shape
        expected by ``SafetyManager``.
        """
        result = self.check(output_text=response, retrieved_sources=sources)
        return {
            "valid": result["is_safe"],
            "violations": result["issues_found"],
            "sanitized_output": result["sanitized_output"],
        }

    # ── Check 1: PII_EXPOSURE ─────────────────────────────────────────────────

    def _check_pii_exposure(
        self, text: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect PII in the output text.

        Two passes:
        1. Regex patterns on the output itself (email, phone, SSN, govt ID).
        2. Cross-reference: extract PII that appeared in source content
           (snippets / abstracts) and check whether it was copied verbatim
           into the output.  Catches cases where a paper abstract contained
           an author's e-mail that the model reproduced.

        Args:
            text:    Generated response text.
            sources: Retrieved source dicts.

        Returns:
            List of finding dicts (may be empty).
        """
        findings: List[Dict[str, Any]] = []

        # Pass 1: pattern scan on output text
        seen_matches: set = set()
        for pattern, label, replacement in _PII_PATTERNS:
            matches = pattern.findall(text)
            new_matches = [m for m in matches if m not in seen_matches]
            if not new_matches:
                continue
            seen_matches.update(new_matches)
            redactions = [
                {
                    "original": m,
                    "replacement": replacement,
                    "context": _context_around(text, m),
                }
                for m in dict.fromkeys(new_matches)  # order-preserving dedup
            ]
            findings.append({
                "issue_type": PII_EXPOSURE,
                "message": (
                    f"PII_EXPOSURE: {label} detected in output "
                    f"({len(redactions)} instance(s))"
                ),
                "redactions": redactions,
                "disclaimer": None,
            })

        # Pass 2: PII that leaked verbatim from source content
        source_pii = self._extract_pii_from_sources(sources)
        for pii_val, label, replacement in source_pii:
            if pii_val in text and pii_val not in seen_matches:
                seen_matches.add(pii_val)
                findings.append({
                    "issue_type": PII_EXPOSURE,
                    "message": (
                        f"PII_EXPOSURE: {label} from source content "
                        "leaked into output"
                    ),
                    "redactions": [
                        {
                            "original": pii_val,
                            "replacement": replacement,
                            "context": _context_around(text, pii_val),
                        }
                    ],
                    "disclaimer": None,
                })

        return findings

    def _extract_pii_from_sources(
        self, sources: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """
        Scan source content fields for PII that could leak into the output.

        Args:
            sources: List of source dicts.

        Returns:
            List of ``(pii_value, label, replacement)`` tuples.
        """
        source_text = " ".join(
            str(s.get("snippet") or s.get("abstract") or s.get("title") or "")
            for s in sources
        )
        found: List[Tuple[str, str, str]] = []
        for pattern, label, replacement in _PII_PATTERNS:
            for match in pattern.findall(source_text):
                found.append((match, label, replacement))
        return found

    # ── Check 2: UNSAFE_CONTENT ───────────────────────────────────────────────

    def _check_unsafe_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect harmful instructions or dangerous advice in the output.

        Pattern matching runs first; the LLM is called only when patterns
        do not trigger (to catch phrasing that bypasses literal keywords).

        Args:
            text: Generated response text.

        Returns:
            List of finding dicts (at most one finding, since the whole
            response is flagged and a single disclaimer is prepended).
        """
        # Layer 1: pattern matching
        for pattern, label in _UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(text):
                return [
                    {
                        "issue_type": UNSAFE_CONTENT,
                        "message": f"UNSAFE_CONTENT: {label}",
                        "redactions": [],
                        "disclaimer": (
                            "⚠️ Safety Notice: Portions of this response were flagged "
                            "for potentially unsafe content. Please verify before acting "
                            "on any specific advice."
                        ),
                    }
                ]

        # Layer 2: LLM check for nuanced unsafe content
        if self._llm_client:
            return self._llm_check_unsafe(text)

        return []

    def _llm_check_unsafe(self, text: str) -> List[Dict[str, Any]]:
        """
        Ask the LLM whether the response contains actionable harmful advice.

        The LLM receives only the last 1 500 characters of the output to keep
        token cost low (harmful content is usually specific and localised).

        Args:
            text: Generated response text.

        Returns:
            List with one finding dict if unsafe, empty list if safe.
        """
        excerpt = text[-1500:] if len(text) > 1500 else text
        prompt = (
            "You are a safety reviewer for an HCI research assistant.\n\n"
            "Evaluate the following response excerpt. A legitimate HCI research "
            "response may *discuss* dangerous topics academically. Flag it ONLY if "
            "it contains ACTIONABLE harmful instructions — meaning specific steps "
            "a person could follow to cause harm or break the law.\n\n"
            f"Response:\n{excerpt}\n\n"
            "Respond ONLY with JSON (no markdown, no prose):\n"
            '{"contains_unsafe_content": true | false, '
            '"reason": "<one sentence>"}'
        )
        try:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a content safety reviewer. Respond only in JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self._llm_temperature,
                max_tokens=128,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_response(raw)
            if data and data.get("contains_unsafe_content"):
                reason = data.get("reason", "LLM flagged unsafe content in output.")
                return [
                    {
                        "issue_type": UNSAFE_CONTENT,
                        "message": f"UNSAFE_CONTENT (LLM): {reason}",
                        "redactions": [],
                        "disclaimer": (
                            "⚠️ Safety Notice: This response was flagged for potentially "
                            "unsafe content. Please exercise caution."
                        ),
                    }
                ]
        except Exception as exc:
            logger.warning("OutputGuardrail: LLM unsafe-content check failed: %s", exc)
        return []

    # ── Check 3: HALLUCINATED_CITATION ────────────────────────────────────────

    def _check_hallucinated_citation(
        self, text: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect ``[N]`` inline citation markers that exceed the retrieved sources.

        Valid indices are determined from the sources list:
        - If sources carry an ``index`` field, those values are the valid set.
        - Otherwise, ``1 … len(sources)`` is assumed.

        Args:
            text:    Generated response text.
            sources: Retrieved source dicts.

        Returns:
            List with one finding dict grouping all hallucinated indices,
            or an empty list when all citations are valid.
        """
        cited_indices = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
        if not cited_indices:
            return []

        valid_indices = self._valid_citation_indices(sources)

        # If no sources at all, every citation is suspect
        if not sources:
            hallucinated = cited_indices
        else:
            hallucinated = cited_indices - valid_indices

        if not hallucinated:
            return []

        sorted_bad = sorted(hallucinated)
        redactions = [
            {
                "original": f"[{idx}]",
                "replacement": f"[†{idx}]",
                "context": _context_around(text, f"[{idx}]"),
            }
            for idx in sorted_bad
        ]

        return [
            {
                "issue_type": HALLUCINATED_CITATION,
                "message": (
                    f"HALLUCINATED_CITATION: citation(s) {sorted_bad} not found "
                    f"in {len(sources)} retrieved source(s)"
                ),
                "redactions": redactions,
                "disclaimer": (
                    "† Citations marked with † could not be verified against "
                    "the retrieved sources and may be hallucinated."
                ),
            }
        ]

    @staticmethod
    def _valid_citation_indices(sources: List[Dict[str, Any]]) -> set:
        """
        Derive the set of valid citation indices from the source list.

        Args:
            sources: Retrieved source dicts.

        Returns:
            Set of valid integer indices.
        """
        if not sources:
            return set()
        # Prefer explicit 'index' field (orchestrator output format)
        if sources[0].get("index") is not None:
            return {int(s["index"]) for s in sources if s.get("index") is not None}
        # Fallback: assume 1-based sequential numbering
        return set(range(1, len(sources) + 1))

    # ── Check 4: MISINFORMATION_RISK ──────────────────────────────────────────

    def _check_misinformation_risk(
        self, text: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Flag specific factual claims not grounded in any retrieved source.

        Uses an LLM to compare key claims in the response against the
        provided source evidence.  Falls back to a statistical heuristic
        (looking for numbers/percentages absent from all source texts) when
        the LLM is unavailable.

        Args:
            text:    Generated response text.
            sources: Retrieved source dicts (need ``snippet`` or ``abstract``).

        Returns:
            List with one finding dict if unsupported claims are detected,
            or an empty list if the response is well-grounded.
        """
        if not sources:
            # Cannot verify claims without evidence — skip, don't false-positive
            return []

        if self._llm_client:
            return self._llm_check_misinformation(text, sources)

        return self._heuristic_misinfo_check(text, sources)

    def _llm_check_misinformation(
        self, text: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ask the LLM to identify claims in the response not supported by sources.

        Uses a concise source summary (title + first 300 chars of snippet/
        abstract) to stay within token limits while giving the model enough
        evidence to make a judgment.

        Args:
            text:    Generated response text.
            sources: Retrieved source dicts.

        Returns:
            List with one finding dict, or empty list if claims are grounded.
        """
        # Build a compact source evidence block
        evidence_lines: List[str] = []
        for i, src in enumerate(sources[:8], 1):  # cap at 8 to control tokens
            title   = src.get("title", "Untitled")
            snippet = src.get("snippet") or src.get("abstract") or ""
            snippet = snippet[:300].replace("\n", " ")
            evidence_lines.append(f"[{i}] {title}: {snippet}")
        evidence_block = "\n".join(evidence_lines)

        # Truncate response to 1500 chars so total prompt fits in context
        response_excerpt = text[:1500] if len(text) > 1500 else text

        prompt = (
            "You are a fact-checking assistant for an HCI research assistant.\n\n"
            "Below are the retrieved source documents used to generate a response, "
            "followed by the response itself.\n\n"
            "RETRIEVED SOURCES:\n"
            f"{evidence_block}\n\n"
            "GENERATED RESPONSE:\n"
            f"{response_excerpt}\n\n"
            "Task: Identify any SPECIFIC factual claims in the response — such as "
            "statistics, percentages, publication years, named study findings, or "
            "exact figures — that are NOT supported by any of the listed sources "
            "above.  Do NOT flag general knowledge or well-known facts.\n\n"
            "Respond ONLY with JSON (no markdown):\n"
            "{\n"
            '  "has_unsupported_claims": true | false,\n'
            '  "unsupported_claims": ["<claim 1>", "<claim 2>"],\n'
            '  "reasoning": "<brief explanation>"\n'
            "}"
        )

        try:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful fact-checker. Respond only in JSON. "
                            "Only flag claims that are clearly absent from the provided sources."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self._llm_temperature,
                max_tokens=self._llm_max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_response(raw)
            if data and data.get("has_unsupported_claims"):
                claims = data.get("unsupported_claims", [])
                reasoning = data.get("reasoning", "")
                claim_preview = "; ".join(str(c) for c in claims[:3])
                return [
                    {
                        "issue_type": MISINFORMATION_RISK,
                        "message": (
                            f"MISINFORMATION_RISK: {len(claims)} unsupported claim(s) "
                            f"detected — e.g. {claim_preview[:120]}"
                        ),
                        "redactions": [],
                        "disclaimer": (
                            "Note: Some specific claims or statistics in this response "
                            "could not be verified against the retrieved sources. "
                            "Please cross-check key figures independently."
                        ),
                        "_detail": {"claims": claims, "reasoning": reasoning},
                    }
                ]
        except Exception as exc:
            logger.warning(
                "OutputGuardrail: misinformation LLM check failed (%s) — "
                "falling back to heuristic.",
                exc,
            )
            return self._heuristic_misinfo_check(text, sources)

        return []

    def _heuristic_misinfo_check(
        self, text: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Heuristic fallback: flag responses with many specific statistics that
        do not appear in any source text.

        Extracts numbers, percentages, and years from the response and checks
        whether they appear in the combined source content.  Triggers only when
        more than two such values are absent — a conservative threshold to
        avoid false positives on reasonable inferences.

        Args:
            text:    Generated response text.
            sources: Retrieved source dicts.

        Returns:
            List with one finding dict, or empty list.
        """
        source_blob = " ".join(
            str(s.get("snippet") or s.get("abstract") or s.get("title") or "")
            for s in sources
        )

        # Specific statistics: percentages, large numbers, publication years
        stat_pattern = re.compile(
            r"\b\d+(?:\.\d+)?%"                          # 87% / 3.5%
            r"|\b(?:19|20)\d{2}\b"                        # years 1900-2099
            r"|\b\d[\d,]{2,}\s*(?:million|billion|thousand|users?|participants?)\b",
            re.I,
        )
        stats_in_response = stat_pattern.findall(text)
        unsupported = [s for s in stats_in_response if s not in source_blob]

        # Only flag when more than 2 specific figures are absent
        if len(unsupported) > 2:
            return [
                {
                    "issue_type": MISINFORMATION_RISK,
                    "message": (
                        f"MISINFORMATION_RISK: {len(unsupported)} specific statistic(s) "
                        "not found in retrieved source text (heuristic check — LLM unavailable)"
                    ),
                    "redactions": [],
                    "disclaimer": (
                        "Note: Some specific statistics or claims could not be "
                        "verified against the retrieved sources. Please cross-check "
                        "key figures independently."
                    ),
                }
            ]

        return []

    # ── Sanitization ──────────────────────────────────────────────────────────

    def _apply_sanitization(
        self,
        original_text: str,
        findings: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Apply all redactions and disclaimers produced by the four checks.

        Processing order:
          1. Text-span replacements (PII redaction, hallucinated-citation
             marker substitution) — applied longest-first to avoid partial
             matches interfering with each other.
          2. Safety-notice prepend (UNSAFE_CONTENT only — high visibility).
          3. Disclaimer appends (MISINFORMATION_RISK, HALLUCINATED_CITATION)
             — added as a footer after a divider so the main answer is intact.

        Each action is recorded in the redaction_log with the required keys:
        ``issue_type``, ``original_snippet``, ``action_taken``.

        Args:
            original_text: The raw generated response.
            findings:      All structured finding dicts from the four checks.

        Returns:
            Tuple of ``(sanitized_text, redaction_log)``.
        """
        sanitized = original_text
        redaction_log: List[Dict[str, Any]] = []

        # ── Phase 1: collect all span replacements ─────────────────────────
        # Build (issue_type, original, replacement, context) tuples
        all_replacements: List[Tuple[str, str, str, str]] = []
        for finding in findings:
            for red in finding.get("redactions", []):
                all_replacements.append(
                    (
                        finding["issue_type"],
                        red["original"],
                        red["replacement"],
                        red.get("context", red["original"]),
                    )
                )

        # Apply longest-first to prevent short patterns shadowing longer ones
        for issue_type, original, replacement, context in sorted(
            all_replacements, key=lambda x: len(x[1]), reverse=True
        ):
            if original in sanitized:
                sanitized = sanitized.replace(original, replacement)
                redaction_log.append(
                    {
                        "issue_type": issue_type,
                        "original_snippet": context,
                        "action_taken": (
                            f"Replaced '{original}' → '{replacement}'"
                        ),
                    }
                )

        # ── Phase 2: prepend safety notice for UNSAFE_CONTENT ─────────────
        safety_notice: Optional[str] = None
        for finding in findings:
            if finding["issue_type"] == UNSAFE_CONTENT and finding.get("disclaimer"):
                safety_notice = finding["disclaimer"]
                break
        if safety_notice:
            sanitized = safety_notice + "\n\n" + sanitized
            redaction_log.append(
                {
                    "issue_type": UNSAFE_CONTENT,
                    "original_snippet": "",
                    "action_taken": "Prepended safety disclaimer to output",
                }
            )

        # ── Phase 3: append disclaimers for other issue types ──────────────
        seen_disclaimers: set = set()
        footer_parts: List[str] = []
        for finding in findings:
            if finding["issue_type"] == UNSAFE_CONTENT:
                continue  # already prepended
            d = finding.get("disclaimer")
            if d and d not in seen_disclaimers:
                seen_disclaimers.add(d)
                footer_parts.append(d)
                redaction_log.append(
                    {
                        "issue_type": finding["issue_type"],
                        "original_snippet": "",
                        "action_taken": f"Appended disclaimer: {d[:80]}",
                    }
                )

        if footer_parts:
            sanitized = sanitized + "\n\n---\n" + "\n".join(footer_parts)

        return sanitized, redaction_log

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _ok_result(text: str) -> Dict[str, Any]:
        """Return a clean, passing check result."""
        return {
            "is_safe": True,
            "issues_found": [],
            "sanitized_output": text,
            "redaction_log": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _context_around(text: str, snippet: str, window: int = 45) -> str:
    """
    Return a short excerpt of ``text`` centred on the first occurrence of
    ``snippet``, with ``…`` indicators when context is truncated.

    Args:
        text:    Full text to search within.
        snippet: Substring to locate.
        window:  Characters of context on each side.

    Returns:
        Context string, or ``snippet`` itself if not found in ``text``.
    """
    idx = text.find(snippet)
    if idx == -1:
        return snippet
    start = max(0, idx - window)
    end   = min(len(text), idx + len(snippet) + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON string from an LLM response, tolerating markdown fences
    and thinking-token preamble.

    Args:
        raw: Raw LLM output string.

    Returns:
        Parsed dict, or ``None`` if parsing fails.
    """
    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    # Skip thinking-token preamble before the first '{'
    brace_start = cleaned.find("{")
    brace_end   = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.debug("OutputGuardrail: could not parse LLM JSON: %r", raw[:200])
        return None
