#!/usr/bin/env python3
"""
Single-query demo with per-agent timeout, retry, and LLM warmup.

Each agent is called individually via on_messages().  On a timeout the
pipeline waits 10 s and retries once with a fresh agent instance before
skipping and continuing.  A lightweight warmup ping is sent to the vLLM
endpoint before any agents run so cold-start latency does not eat into the
first agent's timeout budget.

Saves:
  outputs/demo_session.json
  outputs/demo_answer.md
  outputs/safety_log.jsonl   (appended)
"""

import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import yaml
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from src.agents.autogen_agents import (
    create_model_client,
    create_safety_agent,
    create_planner_agent,
    create_researcher_agent,
    create_critic_agent,
    create_writer_agent,
)
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search

# ── Constants ─────────────────────────────────────────────────────────────────

QUERY = "What are the key principles of explainable AI for novice users?"
AGENT_TIMEOUT = 120         # seconds per agent turn (120 s covers cold vLLM start)
RETRY_WAIT    = 10          # seconds to wait between first timeout and retry
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("run_demo")

# ── Regex helpers ─────────────────────────────────────────────────────────────

_SENTINEL_RE = re.compile(
    r"\b(TERMINATE|PLAN COMPLETE|PLAN STANDS|RESEARCH COMPLETE|"
    r"SAFETY CHECK COMPLETE|DRAFT COMPLETE|REVISION NEEDED)\b",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_URL_RE   = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")

# ANSI colours
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"
_RED   = "\033[31m"
_DIM   = "\033[2m"
_RST   = "\033[0m"

_AGENT_COLOR = {
    "Safety":     "\033[31m",
    "Planner":    "\033[34m",
    "Researcher": "\033[32m",
    "Critic":     "\033[33m",
    "Writer":     "\033[35m",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _SENTINEL_RE.sub("", text)
    return text.strip()


def _build_context(task: str, history: list) -> str:
    """Concatenate the task message with all prior agent turns."""
    parts = [task]
    for source, content in history:
        sep = "─" * 60
        parts.append(f"\n{sep}\n[{source}]:\n{content}")
    return "\n".join(parts)


async def _warmup_llm() -> None:
    """
    Send a minimal 'Hello' prompt to the vLLM endpoint and wait for any
    response.  This wakes the server so subsequent agent calls do not lose
    time to cold-start latency.  Failure is non-fatal — the pipeline continues.
    """
    import os
    from openai import AsyncOpenAI

    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model    = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")

    if not (api_key and base_url):
        log.warning("Warmup skipped — OPENAI_API_KEY or OPENAI_BASE_URL not set.")
        return

    print("  Warming up vLLM endpoint…", end=" ", flush=True)
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0,
            ),
            timeout=60,
        )
        print("done.\n")
        log.info("Warmup ping OK")
    except Exception as exc:
        print(f"failed ({exc}) — continuing anyway.\n")
        log.warning("Warmup failed (non-fatal): %s", exc)


async def _call_agent(make_agent, context: str, name: str) -> tuple:
    """
    Call agent.on_messages with AGENT_TIMEOUT.

    On first timeout: waits RETRY_WAIT seconds and retries once with a fresh
    agent instance (avoiding corrupted internal context from the cancelled
    coroutine).  If the retry also times out, returns a skip placeholder.

    Args:
        make_agent: Zero-argument callable that returns a fresh AssistantAgent.
        context:    Full accumulated conversation context string.
        name:       Agent display name for logging.

    Returns:
        (output_text: str, timed_out: bool)
    """
    for attempt in range(1, 3):   # attempt 1, then attempt 2
        agent = make_agent()       # fresh instance each attempt — no stale context
        try:
            response = await asyncio.wait_for(
                agent.on_messages(
                    [TextMessage(content=context, source="user")],
                    CancellationToken(),
                ),
                timeout=AGENT_TIMEOUT,
            )
            content = response.chat_message.content or ""
            log.info(
                "[%s] OK%s — %d chars",
                name,
                f" (attempt {attempt})" if attempt > 1 else "",
                len(content),
            )
            return content, False

        except asyncio.TimeoutError:
            if attempt < 2:
                log.warning(
                    "[%s] Timeout on attempt %d — waiting %ds before retry…",
                    name, attempt, RETRY_WAIT,
                )
                await asyncio.sleep(RETRY_WAIT)
                # loop continues → attempt 2
            else:
                msg = (
                    f"[TIMEOUT — {name} did not respond within {AGENT_TIMEOUT}s "
                    f"after 2 attempts — step skipped]"
                )
                log.warning(msg)
                return msg, True

        except Exception as exc:
            msg = f"[ERROR in {name}: {type(exc).__name__}: {exc}]"
            log.error(msg, exc_info=True)
            return msg, True

    return f"[TIMEOUT — {name} — skipped]", True   # unreachable but satisfies type checker


def _extract_citations(writer_output: str) -> list:
    """Parse ## References block from the Writer's output."""
    citations: dict = {}
    ref_m = re.search(
        r"##\s*References?\s*\n(.*?)(?=\n##|\Z)",
        writer_output,
        re.DOTALL | re.IGNORECASE,
    )
    if not ref_m:
        return []
    for entry in re.split(r"(?=\[\d+\])", ref_m.group(1).strip()):
        entry = entry.strip()
        if not entry:
            continue
        idx_m = re.match(r"\[(\d+)\]\s*(.*)", entry, re.DOTALL)
        if not idx_m:
            continue
        idx  = int(idx_m.group(1))
        rest = idx_m.group(2).strip()
        url_m = _URL_RE.search(rest)
        url   = url_m.group(0).rstrip(".,)") if url_m else ""
        title = re.sub(r"\s*https?://\S+", "", rest.split("\n")[0]).rstrip(".,").strip()
        citations[idx] = {
            "index": idx,
            "title": title or f"Source {idx}",
            "url":   url,
            "snippet": "",
            "raw":   entry,
        }
    return sorted(citations.values(), key=lambda c: c["index"])


def _write_demo_answer_md(path: Path, query: str, result: dict) -> None:
    answer    = result.get("final_answer", "")
    citations = result.get("citations", [])
    meta      = result.get("metadata", {})
    safety_events = result.get("safety_events", [])

    safety_note = ""
    if any(ev.get("status") == "BLOCKED" for ev in safety_events):
        safety_note = "\n> ⚠️ **Note:** One or more safety events were detected during this session.\n"

    lines = [
        "# Demo Answer",
        "",
        f"**Query:** {query}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        (
            f"**Sources:** {meta.get('num_sources', len(citations))}  |  "
            f"**Agent messages:** {meta.get('num_messages', 0)}  |  "
            f"**Revision rounds:** {meta.get('revision_rounds', 0)}"
        ),
        safety_note,
        "---",
        "",
        answer,
    ]

    if citations:
        lines += ["", "---", "", "## Bibliography", ""]
        for cite in citations:
            idx   = cite.get("index", "?")
            title = cite.get("title") or "Untitled"
            url   = cite.get("url", "")
            entry = f"[{idx}] **[{title}]({url})**" if url else f"[{idx}] **{title}**"
            lines.append(entry)
            if cite.get("snippet"):
                lines.append(f"> {cite['snippet'][:200]}")
            lines.append("")

    judge_r = result.get("demo_judge_results")
    if judge_r:
        lines += [
            "---",
            "",
            "## Judge Evaluation",
            "",
            f"**Judge 1 (Research Quality):** {judge_r['research_quality']['overall_score']:.2f}/5.00  ",
            f"**Judge 2 (Safety & Ethics):**  {judge_r['safety_ethics']['overall_score']:.2f}/5.00  ",
            f"**Combined:**                   {judge_r['combined_avg']:.2f}/5.00",
            "",
        ]

    lines += ["---", "_Generated by Multi-Agent HCI Research Assistant_"]
    path.write_text("\n".join(lines))


# ── Main demo coroutine ───────────────────────────────────────────────────────

async def run_demo() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    print(f"\n{'=' * 70}")
    print(f"{_BOLD}  Multi-Agent HCI Demo — {AGENT_TIMEOUT}s per-agent timeout (1 retry){_RST}")
    print(f"  Query: {_BOLD}{QUERY}{_RST}")
    print(f"{'=' * 70}\n")

    # ── Warmup the vLLM server ────────────────────────────────────────────────
    await _warmup_llm()

    # ── Pre-fetch search results ──────────────────────────────────────────────
    print("Pre-fetching search results…")
    try:
        web_str   = web_search(query=QUERY, max_results=5)
    except Exception as exc:
        web_str   = f"(web search unavailable: {exc})"
    try:
        paper_str = paper_search(query=QUERY, max_results=5)
    except Exception as exc:
        paper_str = f"(paper search unavailable: {exc})"

    search_ctx = (
        "=== PRE-FETCHED RESEARCH DATA ===\n\n"
        f"### Web Search Results\nQuery: {QUERY!r}\n\n{web_str}\n\n"
        f"### Academic Paper Results\nQuery: {QUERY!r}\n\n{paper_str}\n\n"
        "=== END PRE-FETCHED DATA ==="
    )
    print(f"  Web: {len(web_str)} chars | Paper: {len(paper_str)} chars\n")

    # ── Build task message ────────────────────────────────────────────────────
    task_message = (
        f"Research Query: {QUERY}\n\n{search_ctx}\n\n"
        "Work through the following workflow in order:\n\n"
        "Step 1 — Safety (INPUT CHECK): Screen this query for policy violations "
        "before any research begins. Report Status: SAFE or Status: BLOCKED.\n\n"
        "Step 2 — Planner: Decompose the query into 3–5 specific sub-questions. "
        "Generate exactly 3–5 targeted search queries (WEB or PAPER). "
        "End with 'PLAN COMPLETE'.\n\n"
        "Step 3 — Researcher: Review and organise the PRE-FETCHED RESEARCH DATA above. "
        "Assign sequential source numbers [1]…[N]. Do NOT call any tools — all data is "
        "already provided. End with 'RESEARCH COMPLETE'.\n\n"
        "Step 4 — Critic: Review the Researcher's findings for factual consistency, "
        "unsupported claims, and coverage gaps. When satisfied, approve and emit TERMINATE.\n\n"
        "Step 5 — Writer: Synthesise all evidence into a structured final answer with "
        "inline [N] citations and a ## References section. End with 'DRAFT COMPLETE'.\n\n"
        "Step 6 — Safety (OUTPUT CHECK): Screen the Writer's completed draft for "
        "output-side policy violations.\n\n"
        "IMPORTANT: Only the Critic should emit the TERMINATE signal."
    )

    # ── Shared model client + per-step agent factories ────────────────────────
    # Each factory is called fresh for every attempt so a cancelled coroutine
    # never leaves stale state in the agent's internal message context.
    print("Building model client…")
    model_client = create_model_client(config)
    print("  Model client ready\n")

    steps = [
        ("Safety",     "INPUT CHECK",     lambda: create_safety_agent(config, model_client)),
        ("Planner",    "Research plan",   lambda: create_planner_agent(config, model_client)),
        ("Researcher", "Evidence gather", lambda: create_researcher_agent(config, model_client)),
        ("Critic",     "Quality review",  lambda: create_critic_agent(config, model_client)),
        ("Writer",     "Draft synthesis", lambda: create_writer_agent(config, model_client)),
        ("Safety",     "OUTPUT CHECK",    lambda: create_safety_agent(config, model_client)),
    ]

    history: list = []       # [(agent_name, content), …]
    agent_traces: list = []
    any_timeout = False

    for agent_name, step_desc, make_agent in steps:
        color = _AGENT_COLOR.get(agent_name, "")
        print(f"  {color}{_BOLD}[{agent_name}]{_RST}  {step_desc}…", flush=True)

        context = _build_context(task_message, history)
        output, timed_out = await _call_agent(make_agent, context, agent_name)

        if timed_out:
            any_timeout = True
            print(f"    {_RED}⚠ TIMEOUT — skipped{_RST}")
        else:
            preview = _clean(output).strip().splitlines()
            for line in preview[:2]:
                if line.strip():
                    print(f"    {_DIM}↳ {line.strip()[:110]}{_RST}")

        print()
        history.append((agent_name, output))
        agent_traces.append({
            "agent_name": agent_name,
            "step":       step_desc,
            "message":    output,
            "timed_out":  timed_out,
            "timestamp":  datetime.utcnow().isoformat(),
        })

    # ── Extract structured outputs ────────────────────────────────────────────
    # history indices: 0=Safety-in, 1=Planner, 2=Researcher, 3=Critic, 4=Writer, 5=Safety-out
    writer_raw   = history[4][1] if len(history) > 4 else ""
    final_answer = _clean(writer_raw)
    citations    = _extract_citations(writer_raw)

    safety_events = []
    for i, (name, content) in enumerate(history):
        if name != "Safety":
            continue
        status   = "BLOCKED" if re.search(r"Status:\s*BLOCKED", content, re.IGNORECASE) else "SAFE"
        screened = "INPUT query" if i == 0 else "OUTPUT from Writer"
        safety_events.append({
            "status":    status,
            "screened":  screened,
            "timestamp": datetime.utcnow().isoformat(),
            "raw":       content,
        })

    research_plan = history[1][1] if len(history) > 1 else ""

    result = {
        "query":        QUERY,
        "final_answer": final_answer,
        "response":     final_answer,
        "agent_traces": agent_traces,
        "citations":    citations,
        "safety_events": safety_events,
        "research_plan": research_plan,
        "conversation_history": [
            {"agent_name": n, "content": c} for n, c in history
        ],
        "metadata": {
            "num_messages":     len(history),
            "num_sources":      len(citations),
            "revision_rounds":  0,
            "agents_involved":  list(dict.fromkeys(n for n, _ in history)),
            "safety_blocked":   any(ev["status"] == "BLOCKED" for ev in safety_events),
            "any_timeout":      any_timeout,
            "agent_timeout_s":  AGENT_TIMEOUT,
            "error":            False,
            "timeout":          False,
        },
    }

    # ── Run both LLM judges ───────────────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"{_BOLD}Running LLM judges…{_RST}\n")
    try:
        from src.evaluation.judge import LLMJudge
        judge = LLMJudge(config)
        rq, se = await judge.run_both_judges(
            query=QUERY,
            response=final_answer,
            retrieved_sources=citations,
        )
        combined = round((rq.overall_score + se.overall_score) / 2, 2)

        print(f"  Judge 1 — Research Quality : {_GREEN}{rq.overall_score:.2f}/5.00{_RST}")
        for crit, score in rq.criterion_scores.items():
            bar = "█" * score + "░" * (5 - score)
            print(f"    {crit:<28} {bar}  {score}/5")

        print(f"\n  Judge 2 — Safety & Ethics  : {_GREEN}{se.overall_score:.2f}/5.00{_RST}")
        for crit, score in se.criterion_scores.items():
            bar = "█" * score + "░" * (5 - score)
            print(f"    {crit:<28} {bar}  {score}/5")

        print(f"\n  {_BOLD}Combined: {combined:.2f}/5.00{_RST}\n")

        result["demo_judge_results"] = {
            "research_quality": {
                "overall_score":    rq.overall_score,
                "criterion_scores": rq.criterion_scores,
                "strengths":        rq.strengths,
                "weaknesses":       rq.weaknesses,
                "raw_prompt":       rq.raw_prompt,
                "raw_output":       rq.raw_output,
            },
            "safety_ethics": {
                "overall_score":    se.overall_score,
                "criterion_scores": se.criterion_scores,
                "strengths":        se.strengths,
                "weaknesses":       se.weaknesses,
                "raw_prompt":       se.raw_prompt,
                "raw_output":       se.raw_output,
            },
            "combined_avg":  combined,
            "evaluated_at":  datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        log.error("Judge error: %s", exc, exc_info=True)
        print(f"  {_RED}Judge failed: {exc}{_RST}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    session_path = OUTPUT_DIR / "demo_session.json"
    with open(session_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"{_GREEN}✅ demo_session.json → {session_path}{_RST}")

    answer_path = OUTPUT_DIR / "demo_answer.md"
    _write_demo_answer_md(answer_path, QUERY, result)
    print(f"{_GREEN}✅ demo_answer.md    → {answer_path}{_RST}")

    safety_log = OUTPUT_DIR / "safety_log.jsonl"
    with open(safety_log, "a") as fh:
        for ev in safety_events:
            fh.write(json.dumps({
                "timestamp":    ev["timestamp"],
                "source":       "demo",
                "query":        QUERY[:120],
                "status":       ev["status"],
                "action_taken": "REFUSE" if ev["status"] == "BLOCKED" else "ALLOW",
                "screened":     ev["screened"],
            }) + "\n")
        if not safety_events:
            fh.write(json.dumps({
                "timestamp":    datetime.now().isoformat(),
                "source":       "demo",
                "query":        QUERY[:120],
                "status":       "SAFE",
                "action_taken": "ALLOW",
                "screened":     "INPUT query",
            }) + "\n")
    print(f"{_GREEN}✅ safety_log.jsonl  → {safety_log}{_RST}")

    # ── Print final answer ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{_BOLD}FINAL ANSWER{_RST}")
    print(f"{'=' * 70}\n")
    preview = final_answer[:3000]
    print(preview)
    if len(final_answer) > 3000:
        print(f"\n{_DIM}[… {len(final_answer) - 3000} more chars — full text in demo_answer.md]{_RST}")

    print(f"\n{'=' * 70}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_demo())
