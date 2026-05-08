"""
Streamlit Web Interface — Multi-Agent HCI Research Assistant
Run with:  streamlit run src/ui/streamlit_app.py
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve project root so all src.* imports work regardless of cwd
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import streamlit as st
import yaml
from dotenv import load_dotenv

# Load .env before any src.* imports so all modules see the keys
load_dotenv(_ROOT / ".env")

from src.autogen_orchestrator import AutoGenOrchestrator
from src.evaluation.judge import JudgeResult, LLMJudge

logger = logging.getLogger("streamlit_app")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EXAMPLE_QUERIES: List[str] = [
    "What are the key principles of explainable AI for novice users?",
    "How has AR usability evolved in the past 5 years?",
    "What ethical considerations apply to AI-driven tools in education?",
]

# Each entry: (emoji, label, approx cumulative seconds into a 120 s pipeline)
_AGENT_SEQUENCE: List[Tuple[str, str, int]] = [
    ("🛡️",  "Safety — input check",  0),
    ("📋",  "Planner",               15),
    ("🔍",  "Researcher",            35),
    ("🧐",  "Critic",                70),
    ("✍️",  "Writer",                95),
    ("🛡️",  "Safety — output check", 115),
]

_AGENT_BADGE_COLORS: Dict[str, str] = {
    "Safety":     "#dc3545",
    "Planner":    "#0d6efd",
    "Researcher": "#198754",
    "Critic":     "#fd7e14",
    "Writer":     "#6f42c1",
}
_AGENT_ICONS: Dict[str, str] = {
    "Safety":     "🛡️",
    "Planner":    "📋",
    "Researcher": "🔍",
    "Critic":     "🧐",
    "Writer":     "✍️",
}

_SAFETY_CATEGORIES: List[Tuple[str, str]] = [
    ("🚫 HARMFUL_CONTENT",
     "Violence, malware, illegal-activity, and self-harm requests — always refused."),
    ("💉 PROMPT_INJECTION",
     "Jailbreak attempts, persona overrides, and instruction injection — always refused."),
    ("⚠️ OFF_TOPIC",
     "Non-HCI/AI queries allowed with an advisory warning; clearly out-of-scope "
     "content is softly declined."),
]

# Research Quality judge — ordered criterion names matching judge.py
_RQ_CRITERIA: List[str] = [
    "relevance_coverage",
    "evidence_quality",
    "factual_accuracy",
    "clarity_organization",
    "citation_completeness",
]
# Safety & Ethics judge
_SE_CRITERIA: List[str] = [
    "safety_compliance",
    "epistemic_honesty",
    "source_credibility",
]


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Agent badges ─────────────────────────────────────────────────────── */
.agent-badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 14px;
    font-size: 0.80em;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #fff;
    margin-right: 6px;
    vertical-align: middle;
}
.badge-safety     { background-color: #dc3545; }
.badge-planner    { background-color: #0d6efd; }
.badge-researcher { background-color: #198754; }
.badge-critic     { background-color: #fd7e14; }
.badge-writer     { background-color: #6f42c1; }
.badge-unknown    { background-color: #6c757d; }

/* ── Trace timestamp ──────────────────────────────────────────────────── */
.trace-ts {
    font-size: 0.72em;
    color: #888;
    margin-left: 4px;
    vertical-align: middle;
}

/* ── Trace message block ──────────────────────────────────────────────── */
.trace-msg {
    margin: 4px 0 14px 0;
    padding: 8px 12px;
    background: #f8f9fa;
    border-left: 3px solid #dee2e6;
    border-radius: 4px;
    font-size: 0.87em;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 260px;
    overflow-y: auto;
}

/* ── Score progress bar ───────────────────────────────────────────────── */
.score-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.score-label {
    width: 190px;
    font-size: 0.85em;
    font-weight: 600;
    flex-shrink: 0;
}
.score-num {
    width: 32px;
    text-align: right;
    font-size: 0.85em;
    font-weight: 700;
    color: #0d6efd;
    flex-shrink: 0;
}
.score-bar-bg {
    flex: 1;
    height: 10px;
    background: #e9ecef;
    border-radius: 5px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 5px;
}
.fill-5 { background: #198754; }
.fill-4 { background: #20c997; }
.fill-3 { background: #ffc107; }
.fill-2 { background: #fd7e14; }
.fill-1 { background: #dc3545; }

/* ── Sidebar status dot ───────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.dot-ok   { background: #198754; }
.dot-err  { background: #dc3545; }
.dot-warn { background: #fd7e14; }

/* ── Active-agent pill ────────────────────────────────────────────────── */
.agent-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    background: #0d6efd;
    color: #fff;
    font-size: 0.88em;
    font-weight: 600;
    animation: pulse 1.2s infinite;
}
@keyframes pulse {
    0%   { opacity: 1.0; }
    50%  { opacity: 0.55; }
    100% { opacity: 1.0; }
}

/* ── Example query buttons ────────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    font-size: 0.83em;
    text-align: left;
    white-space: normal;
    height: auto;
    padding: 6px 10px;
}

</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    path = _ROOT / "config.yaml"
    if path.exists():
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _badge_html(agent_name: str) -> str:
    key = agent_name.lower()
    css_class = (
        f"badge-{key}"
        if key in {k.lower() for k in _AGENT_BADGE_COLORS}
        else "badge-unknown"
    )
    icon = _AGENT_ICONS.get(agent_name, "🤖")
    return f'<span class="agent-badge {css_class}">{icon} {agent_name}</span>'


def _score_bar_html(criterion: str, score: int) -> str:
    pct   = int(score / 5 * 100)
    label = criterion.replace("_", " ").title()
    fill  = f"fill-{score}"
    return (
        f'<div class="score-row">'
        f'  <span class="score-label">{label}</span>'
        f'  <span class="score-num">{score}/5</span>'
        f'  <div class="score-bar-bg">'
        f'    <div class="score-bar-fill {fill}" style="width:{pct}%"></div>'
        f'  </div>'
        f'</div>'
    )


def _dot_html(ok: bool, warn: bool = False) -> str:
    css = "dot-warn" if (ok and warn) else ("dot-ok" if ok else "dot-err")
    return f'<span class="status-dot {css}"></span>'


def _error_result(query: str, msg: str) -> Dict[str, Any]:
    return {
        "query": query,
        "final_answer": f"⚠️ {msg}",
        "response": f"⚠️ {msg}",
        "agent_traces": [],
        "citations": [],
        "safety_events": [],
        "research_plan": "",
        "metadata": {
            "error": True,
            "num_sources": 0,
            "num_messages": 0,
            "agents_involved": [],
            "safety_blocked": False,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults: Dict[str, Any] = {
        "history":        [],     # list of {timestamp, query, result}
        "orchestrator":   None,
        "judge":          None,
        "config":         {},
        "query_text":     "",     # drives the text_area value
        "current_result": None,   # most recent orchestrator result
        "eval_result":    None,   # {rq: JudgeResult, se: JudgeResult} or None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Build orchestrator and judge once per browser session
    if st.session_state.orchestrator is None:
        config = _load_config()
        st.session_state.config = config
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
        except Exception as exc:
            logger.warning("Orchestrator init failed: %s", exc)
            st.session_state.orchestrator = None

    if st.session_state.judge is None:
        try:
            st.session_state.judge = LLMJudge(st.session_state.config)
        except Exception as exc:
            logger.warning("LLMJudge init failed: %s", exc)
            st.session_state.judge = None


# ─────────────────────────────────────────────────────────────────────────────
# Query processing — threaded so the UI can animate the active-agent indicator
# ─────────────────────────────────────────────────────────────────────────────

class _QueryTask:
    """Shared state between the background query thread and the Streamlit main thread."""
    def __init__(self):
        self.result: Optional[Dict[str, Any]] = None
        self.done   = threading.Event()
        self.error: Optional[str] = None


def _run_query_threaded(query: str) -> Dict[str, Any]:
    """
    Run the orchestrator in a background thread and animate the active-agent
    indicator in the main thread while waiting.

    Uses a time-based heuristic to estimate which agent is currently running
    (Safety→Planner→Researcher→Critic→Writer→Safety) since AutoGen's
    RoundRobinGroupChat does not expose a per-step callback.
    """
    task = _QueryTask()
    orchestrator = st.session_state.orchestrator

    def _worker():
        if orchestrator is None:
            task.error = "Orchestrator is not initialised — check API keys in .env."
        else:
            try:
                task.result = orchestrator.process_query(query)
            except Exception as exc:
                logger.exception("process_query raised: %s", exc)
                task.error = str(exc)
        task.done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # Animate while waiting — hard UI timeout prevents infinite hang
    config     = st.session_state.config
    ui_timeout = config.get("system", {}).get("timeout_seconds", 360) + 120

    placeholder = st.empty()
    start = time.monotonic()
    timed_out = False

    while not task.done.is_set():
        elapsed = time.monotonic() - start
        if elapsed > ui_timeout:
            timed_out = True
            break
        # Pick which agent we're likely on based on elapsed seconds
        agent_label = _AGENT_SEQUENCE[0][1]
        for _, label, threshold in _AGENT_SEQUENCE:
            if elapsed >= threshold:
                agent_label = label
        placeholder.markdown(
            f'<div style="padding:10px 0">'
            f'  <span class="agent-pill">⚡ {agent_label}</span>'
            f'  &nbsp;<small style="color:#888">'
            f'    {int(elapsed)}s elapsed…'
            f'  </small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(1.5)

    placeholder.empty()

    if timed_out:
        return _error_result(
            query,
            f"Request timed out after {int(ui_timeout)}s. "
            "The vLLM server may be under load. Try again or use a shorter query.",
        )

    thread.join(timeout=5)

    if task.error:
        return _error_result(query, task.error)
    return task.result or _error_result(query, "No result returned by orchestrator.")


# ─────────────────────────────────────────────────────────────────────────────
# Judge evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _run_judge_evaluation(result: Dict[str, Any]) -> Optional[Dict[str, JudgeResult]]:
    """
    Call both LLM judges concurrently and return {rq: JudgeResult, se: JudgeResult}.
    Returns None on failure so the caller can show an error.
    """
    judge = st.session_state.judge
    if judge is None:
        return None

    query     = result.get("query", "")
    response  = result.get("final_answer") or result.get("response", "")
    citations = result.get("citations", [])

    try:
        loop = asyncio.new_event_loop()
        rq, se = loop.run_until_complete(
            judge.run_both_judges(
                query=query,
                response=response,
                retrieved_sources=citations,
            )
        )
        loop.close()
        return {"rq": rq, "se": se}
    except Exception as exc:
        logger.error("Judge evaluation failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    config     = st.session_state.config
    history    = st.session_state.history
    model_name = (
        os.getenv("OPENAI_MODEL")
        or config.get("models", {}).get("default", {}).get("name", "—")
    )
    base_url    = os.getenv("OPENAI_BASE_URL", "")
    tavily_key  = os.getenv("TAVILY_API_KEY", "")
    ss_key      = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    orch_ok     = st.session_state.orchestrator is not None
    judge_ok    = st.session_state.judge is not None

    with st.sidebar:
        # ── System status ──────────────────────────────────────────────────
        st.markdown("## 🖥️ System Status")

        st.markdown(
            f'{_dot_html(orch_ok)} **Orchestrator** — '
            f'{"✅ Ready" if orch_ok else "❌ Not initialised"}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_dot_html(bool(model_name and model_name != "—"))} '
            f'**Model** — `{model_name}`',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_dot_html(bool(base_url))} **vLLM endpoint** — '
            f'`{"connected" if base_url else "not set"}`',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_dot_html(bool(tavily_key))} **Tavily search** — '
            f'{"key present" if tavily_key else "⚠️ mock mode"}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_dot_html(True, warn=not bool(ss_key))} '
            f'**Semantic Scholar** — '
            f'{"key present" if ss_key else "public API (rate-limited)"}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'{_dot_html(judge_ok)} **LLM Judge** — '
            f'{"✅ Ready" if judge_ok else "❌ Not initialised"}',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Safety policy summary ──────────────────────────────────────────
        st.markdown("## 🛡️ Safety Policies")
        for cat, desc in _SAFETY_CATEGORIES:
            st.markdown(f"**{cat}**  \n{desc}")

        st.divider()

        # ── Session statistics ─────────────────────────────────────────────
        st.markdown("## 📊 Session Stats")

        blocked = sum(
            1 for h in history
            if h["result"].get("metadata", {}).get("safety_blocked", False)
        )
        total_events = sum(
            len(h["result"].get("safety_events", [])) for h in history
        )

        # Query count is the primary metric as requested
        st.metric("Queries this session", len(history))

        c1, c2 = st.columns(2)
        c1.metric("Blocked", blocked)
        c2.metric("Safety events", total_events)

        c3, c4 = st.columns(2)
        c3.metric("Clean", len(history) - blocked)
        c4.metric("With sources", sum(
            1 for h in history
            if h["result"].get("metadata", {}).get("num_sources", 0) > 0
        ))

        st.divider()

        # ── Controls ───────────────────────────────────────────────────────
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.history       = []
            st.session_state.current_result = None
            st.session_state.eval_result    = None
            st.session_state.query_text     = ""
            st.rerun()

        st.divider()
        topic = config.get("system", {}).get("topic", "HCI Research")
        st.caption(f"**Topic:** {topic}")
        st.caption("Multi-Agent Research Assistant · AutoGen + Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Safety status panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_safety_panel(result: Dict[str, Any]) -> None:
    safety_events: List[Dict] = result.get("safety_events", [])
    blocked: bool = result.get("metadata", {}).get("safety_blocked", False)

    if not safety_events and not blocked:
        st.success("✅ **Safety** — All checks passed. No violations detected.")
        return

    blocked_events   = [e for e in safety_events if e.get("status") == "BLOCKED"]
    sanitized_events = [e for e in safety_events if e.get("action") == "SANITIZE"]
    warn_events      = [
        e for e in safety_events
        if e.get("action") == "WARN" and e.get("status") != "BLOCKED"
    ]

    if blocked_events or blocked:
        for ev in (blocked_events or [{}]):
            cat    = ev.get("category") or "POLICY_VIOLATION"
            reason = ev.get("guidance") or ev.get("reason") or "Content blocked by safety policy."
            if cat == "OFF_TOPIC":
                st.warning(f"⚠️ **OFF_TOPIC** — {reason}")
            else:
                st.error(f"🚫 **Safety — BLOCKED** · Category: `{cat}`  \n{reason}")
        if blocked and not blocked_events:
            st.error("🚫 **Safety — Response blocked.** A safety policy was triggered.")

    for ev in sanitized_events:
        cat = ev.get("category") or "PII / CITATION"
        st.warning(f"✏️ **Safety — SANITIZED** · `{cat}` — Content was sanitised before display.")

    for ev in warn_events:
        cat = ev.get("category") or "OFF_TOPIC"
        msg = ev.get("guidance") or "Query may be off-topic for this system."
        st.warning(f"⚠️ **Safety — WARNING** · `{cat}`  \n{msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Agent traces expander
# ─────────────────────────────────────────────────────────────────────────────

def _render_agent_traces(result: Dict[str, Any]) -> None:
    traces: List[Dict] = result.get("agent_traces", [])
    if not traces:
        return

    with st.expander(f"🔍 Agent Traces ({len(traces)} messages)", expanded=False):
        for turn in traces:
            agent_name   = turn.get("agent_name", "Unknown")
            message      = turn.get("message", "")
            timestamp    = turn.get("timestamp", "")
            message_type = turn.get("message_type", "text")

            badge = _badge_html(agent_name)
            ts    = (
                f'<span class="trace-ts">{str(timestamp)[:19]}</span>'
                if timestamp else ""
            )
            st.markdown(f"{badge}{ts}", unsafe_allow_html=True)

            if message_type in ("tool_call", "tool_result"):
                st.code(
                    message[:600] + ("…" if len(message) > 600 else ""),
                    language="text",
                )
            else:
                preview = message[:1000] + ("…" if len(message) > 1000 else "")
                # Escape < > inside message so stray HTML doesn't render
                safe_preview = preview.replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f'<div class="trace-msg">{safe_preview}</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Citations expander
# ─────────────────────────────────────────────────────────────────────────────

def _render_citations(result: Dict[str, Any]) -> None:
    citations: List[Dict] = result.get("citations", [])
    if not citations:
        return

    with st.expander(f"📚 Citations ({len(citations)} sources)", expanded=False):
        for cite in citations:
            idx     = cite.get("index", "?")
            title   = cite.get("title") or cite.get("raw") or cite.get("url", "Source")
            url     = cite.get("url", "")
            snippet = cite.get("snippet", "")

            if url:
                st.markdown(f"**[{idx}]** [{title}]({url})")
            else:
                st.markdown(f"**[{idx}]** {title}")

            if snippet:
                st.caption(snippet[:240] + ("…" if len(snippet) > 240 else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Final answer panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_final_answer(result: Dict[str, Any]) -> None:
    answer = result.get("final_answer") or result.get("response", "")
    if not answer:
        st.info("No answer was generated.")
        return

    st.markdown("### 📝 Research Answer")
    st.markdown(answer)

    meta = result.get("metadata", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Sources",   meta.get("num_sources", 0))
    c2.metric("Messages",  meta.get("num_messages", 0))
    c3.metric("Revisions", meta.get("revision_rounds", 0))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluation panel — two judge rubric tables
# ─────────────────────────────────────────────────────────────────────────────

def _render_judge_table(
    judge_result: JudgeResult,
    title: str,
    criteria_order: List[str],
) -> None:
    """Render one judge's score table inside the current Streamlit column."""
    st.markdown(f"**{title}**")
    st.markdown(
        f"Overall: **{judge_result.overall_score:.2f} / 5.00**",
    )

    # Score bars
    for crit in criteria_order:
        score = judge_result.criterion_scores.get(crit)
        if score is None:
            continue
        st.markdown(_score_bar_html(crit, score), unsafe_allow_html=True)

    # Strengths / weaknesses
    if judge_result.strengths:
        with st.expander("✅ Strengths", expanded=False):
            for s in judge_result.strengths:
                st.markdown(f"- {s}")

    if judge_result.weaknesses:
        with st.expander("⚠️ Weaknesses", expanded=False):
            for w in judge_result.weaknesses:
                st.markdown(f"- {w}")


def _render_evaluation_panel(result: Dict[str, Any]) -> None:
    st.markdown("### 🏅 Judge Evaluation")

    judge_ok = st.session_state.judge is not None
    if not judge_ok:
        st.warning(
            "LLM Judge is not available — check that `OPENAI_API_KEY` and "
            "`OPENAI_BASE_URL` are set in `.env`."
        )
        return

    st.caption(
        "Runs two independent LLM judges: "
        "**Research Quality** (5 criteria) and **Safety & Ethics** (3 criteria)."
    )

    if st.button("▶️ Run Judge Evaluation", type="primary"):
        st.session_state.eval_result = None
        with st.spinner("Running Judge 1 (Research Quality) + Judge 2 (Safety & Ethics)…"):
            scores = _run_judge_evaluation(result)
            if scores is None:
                st.error("Judge evaluation failed. Check logs for details.")
            else:
                st.session_state.eval_result = scores

    scores: Optional[Dict[str, JudgeResult]] = st.session_state.eval_result
    if scores is None:
        st.info("Click **▶️ Run Judge Evaluation** to score this response.")
        return

    rq: JudgeResult = scores["rq"]
    se: JudgeResult = scores["se"]

    col1, col2 = st.columns(2)

    with col1:
        _render_judge_table(
            rq,
            "📄 Judge 1 — Research Quality",
            _RQ_CRITERIA,
        )

    with col2:
        _render_judge_table(
            se,
            "🛡️ Judge 2 — Safety & Ethics",
            _SE_CRITERIA,
        )

    # Combined overall
    combined = round((rq.overall_score + se.overall_score) / 2, 2)
    stars    = (
        "⭐⭐⭐⭐⭐" if combined >= 4.5
        else "⭐⭐⭐⭐" if combined >= 3.5
        else "⭐⭐⭐"  if combined >= 2.5
        else "⭐⭐"
    )
    st.success(f"Combined overall score: **{combined:.2f} / 5.00** {stars}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Export button
# ─────────────────────────────────────────────────────────────────────────────

def _render_export(result: Dict[str, Any]) -> None:
    eval_result = st.session_state.eval_result

    payload: Dict[str, Any] = {
        "export_timestamp": datetime.now().isoformat(),
        "query":            result.get("query", ""),
        "final_answer":     result.get("final_answer") or result.get("response", ""),
        "citations":        result.get("citations", []),
        "safety_events":    result.get("safety_events", []),
        "agent_traces":     result.get("agent_traces", []),
        "research_plan":    result.get("research_plan", ""),
        "metadata":         result.get("metadata", {}),
        "judge_evaluation": (
            {
                "research_quality": {
                    "overall_score":    eval_result["rq"].overall_score,
                    "criterion_scores": eval_result["rq"].criterion_scores,
                    "strengths":        eval_result["rq"].strengths,
                    "weaknesses":       eval_result["rq"].weaknesses,
                },
                "safety_ethics": {
                    "overall_score":    eval_result["se"].overall_score,
                    "criterion_scores": eval_result["se"].criterion_scores,
                    "strengths":        eval_result["se"].strengths,
                    "weaknesses":       eval_result["se"].weaknesses,
                },
                "combined_avg": round(
                    (eval_result["rq"].overall_score + eval_result["se"].overall_score) / 2, 3
                ),
            }
            if eval_result else None
        ),
        "session_history": [
            {
                "timestamp": h["timestamp"],
                "query":     h["query"],
                "response":  h["result"].get("final_answer") or h["result"].get("response", ""),
                "num_sources": h["result"].get("metadata", {}).get("num_sources", 0),
                "safety_blocked": h["result"].get("metadata", {}).get("safety_blocked", False),
            }
            for h in st.session_state.history
        ],
    }

    fname = f"hci_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button(
        label="📥 Export Session JSON",
        data=json.dumps(payload, indent=2, default=str),
        file_name=fname,
        mime="application/json",
        use_container_width=True,
        help="Downloads the full session including the final answer, citations, "
             "agent traces, safety events, and judge scores.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session history panel (bonus — shown below results)
# ─────────────────────────────────────────────────────────────────────────────

def _render_history() -> None:
    history = st.session_state.history
    if len(history) < 2:          # only show once there's more than the current query
        return
    with st.expander(f"📜 Query History ({len(history)} queries)", expanded=False):
        for i, item in enumerate(reversed(history), 1):
            ts      = item.get("timestamp", "")
            query   = item.get("query", "")
            n_src   = item["result"].get("metadata", {}).get("num_sources", 0)
            blocked = item["result"].get("metadata", {}).get("safety_blocked", False)
            flag    = " 🚫" if blocked else ""
            st.markdown(
                f"**{i}.** `{ts}` — {query[:90]}{flag}  \n"
                f"<small>{n_src} source(s)</small>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent HCI Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS once per page load
    st.markdown(_CSS, unsafe_allow_html=True)

    # Initialise session state (orchestrator + judge built on first load)
    _init_session()

    # ── 1. Sidebar ─────────────────────────────────────────────────────────
    _render_sidebar()

    # ── 2. Main header ─────────────────────────────────────────────────────
    st.title("🤖 Multi-Agent HCI Research Assistant")
    st.markdown(
        "Ask a question about **human-computer interaction**, usability, accessibility, "
        "AI transparency, or related technology topics — five specialised agents "
        "collaborate to plan, research, critique, and synthesise a cited answer."
    )

    st.divider()

    # ── 3. Query input area ────────────────────────────────────────────────
    # Layout: wide input column on the left, workflow info on the right
    input_col, info_col = st.columns([3, 1], gap="large")

    with info_col:
        st.markdown("#### ℹ️ Pipeline")
        st.markdown(
            "1. 🛡️ **Safety** screens input  \n"
            "2. 📋 **Planner** structures sub-questions  \n"
            "3. 🔍 **Researcher** fetches web + papers  \n"
            "4. 🧐 **Critic** reviews (≤ 2 rounds)  \n"
            "5. ✍️ **Writer** synthesises cited answer  \n"
            "6. 🛡️ **Safety** screens output"
        )

    with input_col:
        # Text area — value driven by session state so example buttons pre-fill it
        query = st.text_area(
            "Enter your research query:",
            value=st.session_state.query_text,
            height=110,
            placeholder="e.g., What are the key principles of explainable AI for novice users?",
        )

        # Submit button
        submit_clicked = st.button(
            "🔍 Search", type="primary", use_container_width=True
        )

        # 3 example query buttons in a row below the input
        st.markdown("**💡 Example queries** — click to pre-fill:")
        ex_cols = st.columns(3)
        for col, example in zip(ex_cols, _EXAMPLE_QUERIES):
            with col:
                if st.button(
                    example,
                    key=f"ex_{hash(example)}",
                    use_container_width=True,
                ):
                    st.session_state.query_text  = example
                    st.session_state.eval_result = None
                    st.rerun()

        # ── 4. Submit handler: spinner with active-agent name ──────────────
        if submit_clicked:
            q = query.strip()
            if not q:
                st.warning("Please enter a research query before submitting.")
            else:
                st.session_state.eval_result = None

                # _run_query_threaded animates the active-agent pill while the
                # orchestrator runs in a background thread, then returns the result
                result = _run_query_threaded(q)

                st.session_state.current_result = result
                st.session_state.query_text     = q
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "query":     q,
                    "result":    result,
                })
                st.rerun()

    # ── Results section ────────────────────────────────────────────────────
    result = st.session_state.current_result
    if result is not None:
        st.divider()

        # 5. Safety status panel
        _render_safety_panel(result)

        # 8. Final answer
        _render_final_answer(result)

        st.divider()

        # 6. Agent traces expander
        _render_agent_traces(result)

        # 7. Citations expander
        _render_citations(result)

        st.divider()

        # 9. Evaluation panel (two judge rubric tables)
        _render_evaluation_panel(result)

        st.divider()

        # 10. Export
        _render_export(result)

        # Session history (shown at the very bottom)
        st.divider()
        _render_history()


if __name__ == "__main__":
    main()
