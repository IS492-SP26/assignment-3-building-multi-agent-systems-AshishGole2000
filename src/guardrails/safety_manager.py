"""
Safety Manager
Coordinates InputGuardrail and OutputGuardrail, enforces per-category policies,
and writes a structured audit log to outputs/safety_log.jsonl.

Policy table
────────────
Input categories (from InputGuardrail):
  HARMFUL          → REFUSE   – block query, return refusal message
  PROMPT_INJECTION → REFUSE   – block query, return refusal message
  PII              → SANITIZE – strip PII from query, continue with clean text
  OFF_TOPIC        → WARN     – allow query, prepend an advisory note

Output issue types (from OutputGuardrail):
  UNSAFE_CONTENT          → REFUSE   – discard response, return refusal message
  PII_EXPOSURE            → SANITIZE – return redacted response
  HALLUCINATED_CITATION   → SANITIZE – return response with [†N] markers + footnote
  MISINFORMATION_RISK     → WARN     – return response with verification disclaimer

When multiple output issues are present the strictest policy wins:
  REFUSE (3) > SANITIZE (2) > WARN (1) > ALLOW (0)

Safety-event schema
───────────────────
Each entry in ``safety_events`` (and each JSON line in the log file) has:
  timestamp    – ISO-8601 string
  event_type   – "input" | "output"
  category     – violated category/issue string, or "NONE" when safe
  query_snippet – first 120 characters of the checked content
  action_taken – "REFUSE" | "SANITIZE" | "WARN" | "ALLOW"
  details      – full guardrail result dict for debugging
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .input_guardrail  import InputGuardrail
from .output_guardrail import OutputGuardrail

logger = logging.getLogger("safety")

# ── Action constants ───────────────────────────────────────────────────────────

REFUSE   = "REFUSE"
SANITIZE = "SANITIZE"
WARN     = "WARN"
ALLOW    = "ALLOW"

# ── Per-category policy tables ─────────────────────────────────────────────────

_INPUT_POLICY: Dict[str, str] = {
    "HARMFUL":          REFUSE,
    "PROMPT_INJECTION": REFUSE,
    "PII":              SANITIZE,
    "OFF_TOPIC":        WARN,
}

_OUTPUT_POLICY: Dict[str, str] = {
    "UNSAFE_CONTENT":        REFUSE,
    "PII_EXPOSURE":          SANITIZE,
    "HALLUCINATED_CITATION": SANITIZE,
    "MISINFORMATION_RISK":   WARN,
}

# Numeric severity so we can pick the strictest action when multiple fire
_SEVERITY: Dict[str, int] = {REFUSE: 3, SANITIZE: 2, WARN: 1, ALLOW: 0}

# ── Canned messages ────────────────────────────────────────────────────────────

_REFUSE_INPUT_MSG = (
    "I cannot process this request due to safety policy violations. "
    "Please rephrase your query and ensure it is appropriate for an "
    "HCI research assistant."
)
_REFUSE_OUTPUT_MSG = (
    "I cannot provide this response due to safety policy violations. "
    "Please try rephrasing your request."
)
_WARN_OFF_TOPIC_NOTE = (
    "Advisory: Your query may be outside the primary scope of this HCI "
    "research assistant. I'll do my best to help — consider refining your "
    "question to focus on user interface design, usability, accessibility, "
    "or related technology topics for better results.\n\n"
)

# Default log file path (relative to project root)
_DEFAULT_LOG_PATH = Path("outputs") / "safety_log.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# SafetyManager
# ─────────────────────────────────────────────────────────────────────────────

class SafetyManager:
    """
    Central coordinator for input and output safety checks.

    Instantiate once and reuse across queries; call :meth:`reset` between
    sessions to clear the in-memory event list (the JSONL file is never
    truncated — it accumulates across restarts for audit purposes).

    Usage::

        manager = SafetyManager(config["safety"])

        # Before processing a user query:
        input_result = manager.run_input_check(query)
        if not input_result["allowed"]:
            return input_result["response"]   # refusal message
        clean_query = input_result["query"]   # possibly sanitized

        # After generating a response:
        output_result = manager.run_output_check(response, sources)
        final_response = output_result["response"]
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialise the safety manager.

        Reads all settings from ``config`` (the ``safety`` sub-dict from
        config.yaml) and from environment variables:
          ``ENABLE_GUARDRAILS``  — overrides ``config["enabled"]``
          ``LOG_SAFETY_EVENTS``  — overrides ``config["log_events"]``

        Also instantiates :class:`InputGuardrail` and
        :class:`OutputGuardrail`, which in turn read LLM credentials from
        ``.env`` via ``load_dotenv()``.

        Args:
            config: The ``safety`` sub-dict from config.yaml, or the full
                    config dict — both are tolerated.
        """
        self.config = config

        # Support both full-config and safety-sub-dict callers
        safety_cfg = config.get("safety", config)

        self.enabled: bool = (
            os.getenv("ENABLE_GUARDRAILS", str(safety_cfg.get("enabled", True))).lower()
            not in ("false", "0", "no")
        )
        self.log_events: bool = (
            os.getenv("LOG_SAFETY_EVENTS", str(safety_cfg.get("log_events", True))).lower()
            not in ("false", "0", "no")
        )

        # JSONL audit log path — create parent directory on first write
        log_path_str = (
            safety_cfg.get("safety_log_file")
            or config.get("logging", {}).get("safety_log")
            or str(_DEFAULT_LOG_PATH)
        )
        self._log_path = Path(log_path_str)

        # In-memory event list for the current session
        self.safety_events: List[Dict[str, Any]] = []

        # Wire in the guardrails — pass full config so they can read all sections
        self._input_guardrail  = InputGuardrail(config)
        self._output_guardrail = OutputGuardrail(config)

        logger.info(
            "SafetyManager ready (enabled=%s, log=%s, log_path=%s)",
            self.enabled, self.log_events, self._log_path,
        )

    # ── Primary public API ─────────────────────────────────────────────────────

    def run_input_check(self, query: str) -> Dict[str, Any]:
        """
        Screen a user query before it enters the agent pipeline.

        Calls :class:`InputGuardrail`, maps the result to a policy action,
        logs a safety event, and returns a unified result dict.

        Policy (highest-severity category wins when multiple fire):
          HARMFUL / PROMPT_INJECTION → REFUSE
          PII                        → SANITIZE (PII stripped, clean query returned)
          OFF_TOPIC                  → WARN (advisory note added, query unchanged)

        Args:
            query: Raw user input string.

        Returns:
            Dict with:
              ``allowed``   – ``True`` when the pipeline may proceed.
              ``action``    – ``"REFUSE"`` | ``"SANITIZE"`` | ``"WARN"`` | ``"ALLOW"``
              ``query``     – Query to use downstream (may be sanitized).
              ``category``  – Violated category or ``None``.
              ``reason``    – Human-readable explanation.
              ``response``  – Refusal message (only meaningful when ``allowed=False``).
              ``warning``   – Advisory string to prepend (only for ``WARN``).
              ``event``     – The safety-event dict that was logged.
        """
        if not self.enabled:
            return self._pass_input(query, "Guardrails disabled.", None)

        guardrail_result = self._input_guardrail.validate(query)

        is_safe  = guardrail_result.get("is_safe", True)
        category = guardrail_result.get("category")
        reason   = guardrail_result.get("reason", "")
        suggested = guardrail_result.get("suggested_response", "")

        action = _INPUT_POLICY.get(category, ALLOW) if not is_safe else ALLOW

        # Build outcome
        if action == REFUSE:
            event = self._record_event(
                event_type="input",
                category=category,
                content=query,
                action=REFUSE,
                details=guardrail_result,
            )
            return {
                "allowed":  False,
                "action":   REFUSE,
                "query":    query,
                "category": category,
                "reason":   reason,
                # Use SafetyManager's canonical message so UI messaging is
                # consistent regardless of which guardrail pattern fired.
                "response": _REFUSE_INPUT_MSG,
                "warning":  None,
                "event":    event,
            }

        if action == SANITIZE:
            # For input PII: use sanitized_input if provided, else original
            clean_query = guardrail_result.get("sanitized_input", query) or query
            event = self._record_event(
                event_type="input",
                category=category,
                content=query,
                action=SANITIZE,
                details=guardrail_result,
            )
            return {
                "allowed":  True,
                "action":   SANITIZE,
                "query":    clean_query,
                "category": category,
                "reason":   reason,
                "response": "",
                "warning":  None,
                "event":    event,
            }

        if action == WARN:
            event = self._record_event(
                event_type="input",
                category=category,
                content=query,
                action=WARN,
                details=guardrail_result,
            )
            return {
                "allowed":  True,
                "action":   WARN,
                "query":    query,
                "category": category,
                "reason":   reason,
                "response": "",
                "warning":  suggested or _WARN_OFF_TOPIC_NOTE,
                "event":    event,
            }

        # ALLOW — safe query
        return self._pass_input(query, reason, guardrail_result)

    def run_output_check(
        self,
        output: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Inspect a generated response before it is returned to the user.

        Calls :class:`OutputGuardrail`, determines the strictest policy
        action across all issue types found, logs a safety event, and
        returns a unified result dict.

        Policy (strictest action across all issues wins):
          UNSAFE_CONTENT                      → REFUSE
          PII_EXPOSURE / HALLUCINATED_CITATION → SANITIZE (redacted response)
          MISINFORMATION_RISK                  → WARN (disclaimer appended)

        Args:
            output:  The generated response string.
            sources: Retrieved source dicts from the research session.

        Returns:
            Dict with:
              ``allowed``       – ``True`` when a usable response is available.
              ``action``        – ``"REFUSE"`` | ``"SANITIZE"`` | ``"WARN"`` | ``"ALLOW"``
              ``response``      – The response to send to the user.
              ``issues_found``  – List of issue-description strings.
              ``redaction_log`` – List of redaction-log entry dicts.
              ``event``         – The safety-event dict that was logged.
        """
        if not self.enabled:
            return self._pass_output(output, [], [])

        guardrail_result = self._output_guardrail.check(
            output_text=output,
            retrieved_sources=sources or [],
        )

        issues_found  = guardrail_result.get("issues_found", [])
        sanitized_out = guardrail_result.get("sanitized_output", output)
        redaction_log = guardrail_result.get("redaction_log", [])

        # Determine the strictest action across all issue types detected
        action, triggered_issues = self._resolve_output_action(issues_found)

        if action == ALLOW:
            return self._pass_output(output, issues_found, redaction_log)

        # Build the response the user will actually see
        if action == REFUSE:
            final_response = _REFUSE_OUTPUT_MSG
        else:
            # SANITIZE or WARN — use the guardrail's sanitized version
            final_response = sanitized_out

        event = self._record_event(
            event_type="output",
            category=", ".join(triggered_issues) if triggered_issues else action,
            content=output,
            action=action,
            details=guardrail_result,
        )

        return {
            "allowed":       action != REFUSE,
            "action":        action,
            "response":      final_response,
            "issues_found":  issues_found,
            "redaction_log": redaction_log,
            "event":         event,
        }

    # ── Session management ─────────────────────────────────────────────────────

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """
        Return all safety events recorded in the current session.

        Each entry follows the schema:
          timestamp, event_type, category, query_snippet, action_taken, details

        Returns:
            List of event dicts in chronological order.
        """
        return list(self.safety_events)

    def reset(self) -> None:
        """
        Clear the in-memory safety-event list for the next session.

        The JSONL log file on disk is never truncated — it is a persistent
        audit trail.  Only the in-memory list is cleared.
        """
        self.safety_events.clear()
        logger.info("SafetyManager: session state reset.")

    # ── Backwards-compatible aliases ───────────────────────────────────────────

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Backwards-compatible wrapper for code that calls the old API.

        Delegates to :meth:`run_input_check` and translates the result to
        the original ``{safe, violations, query}`` shape.

        Args:
            query: Raw user input string.

        Returns:
            Dict with ``safe`` (bool), ``violations`` (list), and ``query``.
        """
        result = self.run_input_check(query)
        return {
            "safe":       result["allowed"],
            "violations": (
                [{"category": result["category"], "reason": result["reason"]}]
                if result["category"]
                else []
            ),
            "query":      result["query"],
            "action":     result["action"],
            "response":   result.get("response", ""),
        }

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Backwards-compatible wrapper for code that calls the old API.

        Delegates to :meth:`run_output_check` and translates the result to
        the original ``{safe, violations, response}`` shape.

        Note on ``safe``: this reflects whether any issues were found
        (``False`` when issues_found is non-empty), which differs from
        ``allowed`` in :meth:`run_output_check` that tracks whether a
        usable response exists.  Old callers that checked ``safe`` to
        decide whether to show the response should now check ``allowed``.

        Args:
            response: Generated response string.
            sources:  Optional source dicts.

        Returns:
            Dict with ``safe`` (bool), ``violations`` (list), and ``response``.
        """
        result = self.run_output_check(output=response, sources=sources)
        return {
            # safe = no issues found (backwards compat expectation)
            "safe":       not bool(result["issues_found"]),
            "violations": result["issues_found"],
            "response":   result["response"],
            "action":     result["action"],
        }

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Summarise safety events for the current session.

        Returns:
            Dict with total_events, input_checks, output_checks, violations,
            violation_rate, and per-action counts.
        """
        total   = len(self.safety_events)
        inputs  = sum(1 for e in self.safety_events if e["event_type"] == "input")
        outputs = sum(1 for e in self.safety_events if e["event_type"] == "output")
        refused   = sum(1 for e in self.safety_events if e["action_taken"] == REFUSE)
        sanitized = sum(1 for e in self.safety_events if e["action_taken"] == SANITIZE)
        warned    = sum(1 for e in self.safety_events if e["action_taken"] == WARN)
        allowed   = sum(1 for e in self.safety_events if e["action_taken"] == ALLOW)
        violations = refused + sanitized + warned

        return {
            "total_events":   total,
            "input_checks":   inputs,
            "output_checks":  outputs,
            "violations":     violations,
            "violation_rate": violations / total if total > 0 else 0.0,
            "by_action": {
                REFUSE:   refused,
                SANITIZE: sanitized,
                WARN:     warned,
                ALLOW:    allowed,
            },
        }

    # Keep old name as an alias
    def clear_events(self) -> None:
        """Alias for :meth:`reset` (backwards compat)."""
        self.reset()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_output_action(
        self, issues_found: List[str]
    ) -> tuple[str, List[str]]:
        """
        Determine the strictest policy action across all detected issue types.

        Scans each issue string for known issue-type prefixes and maps them
        to their policy action.  The action with the highest severity wins.

        Args:
            issues_found: List of issue-description strings from OutputGuardrail,
                          e.g. ``["HALLUCINATED_CITATION: [5] not found …"]``.

        Returns:
            Tuple of ``(action, triggered_issue_types)`` where
            ``triggered_issue_types`` lists the issue-type strings that fired.
        """
        best_action    = ALLOW
        best_severity  = _SEVERITY[ALLOW]
        triggered: List[str] = []

        for issue_str in issues_found:
            # issue strings are prefixed with the issue type: "ISSUE_TYPE: …"
            issue_type = issue_str.split(":")[0].strip()
            action = _OUTPUT_POLICY.get(issue_type, WARN)
            severity = _SEVERITY.get(action, 0)

            triggered.append(issue_type)
            if severity > best_severity:
                best_severity = severity
                best_action   = action

        return best_action, triggered

    def _record_event(
        self,
        event_type: str,
        category: Optional[str],
        content: str,
        action: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a safety-event dict, append it to the session list, and
        write it as a JSON line to the audit log file.

        Args:
            event_type: ``"input"`` or ``"output"``.
            category:   Violated category/issue string (or ``None`` / ``"NONE"``).
            content:    The content that was checked (query or response text).
            action:     Policy action taken (``REFUSE`` / ``SANITIZE`` / ``WARN`` / ``ALLOW``).
            details:    Full guardrail result dict for debugging.

        Returns:
            The event dict that was recorded.
        """
        event: Dict[str, Any] = {
            "timestamp":     datetime.now().isoformat(),
            "event_type":    event_type,
            "category":      category or "NONE",
            "query_snippet": content[:120] + ("…" if len(content) > 120 else ""),
            "action_taken":  action,
            "details":       details,
        }

        self.safety_events.append(event)

        if self.log_events:
            logger.warning(
                "SafetyManager [%s] %s → %s | category=%s",
                event_type.upper(), event["query_snippet"][:60],
                action, event["category"],
            )

        self._write_log_entry(event)
        return event

    def _write_log_entry(self, event: Dict[str, Any]) -> None:
        """
        Append one JSON line to the JSONL audit log.

        Creates the ``outputs/`` directory and the log file if they do not
        exist yet.  Failures are logged as warnings so a disk error does not
        crash the pipeline.

        Args:
            event: Safety-event dict to serialise.
        """
        if not self.log_events:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            # Write a "details"-stripped copy to keep the log readable
            log_entry = {k: v for k, v in event.items() if k != "details"}
            log_entry["details_summary"] = {
                "issues_found": event.get("details", {}).get("issues_found", []),
                "is_safe":      event.get("details", {}).get("is_safe",
                                event.get("details", {}).get("valid", True)),
            }
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("SafetyManager: failed to write audit log: %s", exc)

    # ── Clean-pass helpers ─────────────────────────────────────────────────────

    def _pass_input(
        self,
        query: str,
        reason: str,
        details: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a clean ALLOW result for input (no event recorded)."""
        return {
            "allowed":  True,
            "action":   ALLOW,
            "query":    query,
            "category": None,
            "reason":   reason,
            "response": "",
            "warning":  None,
            "event":    {},
        }

    def _pass_output(
        self,
        output: str,
        issues_found: List[str],
        redaction_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a clean ALLOW result for output (no event recorded)."""
        return {
            "allowed":       True,
            "action":        ALLOW,
            "response":      output,
            "issues_found":  issues_found,
            "redaction_log": redaction_log,
            "event":         {},
        }
