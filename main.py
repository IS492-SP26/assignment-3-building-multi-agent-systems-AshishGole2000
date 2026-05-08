"""
Main Entry Point — Multi-Agent HCI Research Assistant

Usage:
  python main.py --ui web       Launch Streamlit web interface
  python main.py --ui cli       Launch interactive CLI
  python main.py --evaluate     Run 8-query batch evaluation + reports
  python main.py --demo         End-to-end demo with judge scoring
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Populate environment variables from .env before any src.* import so every
# module (orchestrator, guardrails, judge) sees the API keys on first import.
load_dotenv()

# ─── ANSI helpers (used by --demo) ───────────────────────────────────────────

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_GREEN   = "\033[32m"

_AGENT_COLORS: dict = {
    "Safety":     "\033[31m",
    "Planner":    "\033[34m",
    "Researcher": "\033[32m",
    "Critic":     "\033[33m",
    "Writer":     "\033[35m",
}

_DEMO_QUERY = "What are the key principles of explainable AI for novice users?"


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _load_config(config_path: str = "config.yaml") -> dict:
    import yaml
    with open(config_path) as fh:
        return yaml.safe_load(fh)


async def _warmup_llm(config: dict) -> None:
    """
    Send a minimal 'Hello' prompt to the vLLM endpoint and wait for a
    response.  Absorbs cold-start latency before the first real agent turn.
    Non-fatal — failure prints a warning and the pipeline continues.
    """
    import os
    from openai import AsyncOpenAI

    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model    = (
        os.getenv("OPENAI_MODEL")
        or config.get("models", {}).get("default", {}).get("name", "Qwen/Qwen3-8B")
    )

    if not (api_key and base_url):
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
        print("done.")
    except Exception as exc:
        print(f"failed ({exc}) — continuing anyway.")


def _build_orchestrator(config: dict):
    """Initialise the AutoGenOrchestrator; returns None on failure."""
    from src.autogen_orchestrator import AutoGenOrchestrator
    try:
        orch = AutoGenOrchestrator(config)
        print("  ✅ Orchestrator ready")
        return orch
    except Exception as exc:
        print(f"  ⚠️  Orchestrator unavailable: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# --ui web
# ─────────────────────────────────────────────────────────────────────────────

def run_web() -> None:
    """
    Launch the Streamlit web interface.

    Uses os.system so the Streamlit process inherits the current terminal and
    environment (including the .env variables already loaded above).
    """
    print("Starting Streamlit web interface…")
    print("  Press Ctrl-C to stop.\n")
    os.system("streamlit run src/ui/streamlit_app.py")


# ─────────────────────────────────────────────────────────────────────────────
# --ui cli
# ─────────────────────────────────────────────────────────────────────────────

def run_cli() -> None:
    """Launch the interactive CLI loop."""
    from src.ui.cli import main as cli_main
    cli_main()


# ─────────────────────────────────────────────────────────────────────────────
# --evaluate
# ─────────────────────────────────────────────────────────────────────────────

async def run_evaluation(config_path: str = "config.yaml") -> None:
    """
    Run the fixed 8-query batch evaluation.

    Loads data/eval_queries.json, runs every query through the full agent
    pipeline, scores each result with both LLM judges, verifies that the two
    safety-test queries were refused, then writes:

      outputs/evaluation_report.json  — full structured results with raw judge I/O
      outputs/evaluation_report.md    — summary table + error analysis

    A console summary is printed when the run completes.
    """
    from src.evaluation.evaluator import BatchEvaluator

    config = _load_config(config_path)

    print("=" * 70)
    print("  Multi-Agent HCI Research Assistant — Batch Evaluation")
    print("=" * 70)

    print("\nInitialising orchestrator…")
    orchestrator = _build_orchestrator(config)
    if orchestrator is None:
        print("  Evaluation will proceed with placeholder responses.\n")

    evaluator = BatchEvaluator(config, orchestrator)

    print("\nRunning 8-query eval suite (data/eval_queries.json)…\n")
    report = await evaluator.run(queries_path="data/eval_queries.json")

    # Console summary
    summary = report.get("summary", {})
    safety  = report.get("safety_verification", {})

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Queries evaluated  : {report.get('queries_evaluated', 0)}")
    print(f"  Judge 1 (RQ) avg   : {summary.get('judge1_overall_avg', 0):.2f} / 5.00")
    print(f"  Judge 2 (SE) avg   : {summary.get('judge2_overall_avg', 0):.2f} / 5.00")
    print(f"  Combined avg       : {summary.get('combined_avg', 0):.2f} / 5.00")
    print(f"\n  Safety             : {safety.get('summary', '—')}")

    if summary.get("judge1_by_criterion"):
        print("\n  Judge 1 by criterion:")
        for k, v in summary["judge1_by_criterion"].items():
            print(f"    {k:<28} {v:.2f}")

    if summary.get("judge2_by_criterion"):
        print("\n  Judge 2 by criterion:")
        for k, v in summary["judge2_by_criterion"].items():
            print(f"    {k:<28} {v:.2f}")

    print("\n" + "=" * 70)
    print("  Outputs:")
    print("    outputs/evaluation_report.json")
    print("    outputs/evaluation_report.md")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# --demo
# ─────────────────────────────────────────────────────────────────────────────

async def run_demo(config_path: str = "config.yaml") -> None:
    """
    Run one full end-to-end demo and save all artefacts.

    Steps:
      1. Initialise orchestrator.
      2. Run _DEMO_QUERY through the full agent pipeline.
      3. Print every agent trace to the terminal with coloured [AgentName] prefix.
      4. Print the final answer.
      5. Save outputs/demo_session.json  — the complete result dict.
      6. Save outputs/demo_answer.md     — final answer + bibliography.
      7. Run both LLM judges concurrently.
      8. Print judge scores with a visual bar.
      9. Append judge results to demo_session.json.
    """
    from src.evaluation.judge import LLMJudge

    config = _load_config(config_path)

    _print_demo_header()

    # ── Step 1: orchestrator ──────────────────────────────────────────────────
    print("Initialising orchestrator…")
    orchestrator = _build_orchestrator(config)
    if orchestrator is None:
        print("Cannot run demo without a working orchestrator. Check .env keys.")
        sys.exit(1)

    # Warm up the vLLM server so the first agent turn is not delayed by
    # cold-start latency.  The orchestrator's _process_query_async also does
    # a warmup ping; this one runs earlier so the server is ready before the
    # orchestrator initialises the research team.
    print("Warming up LLM endpoint…")
    await _warmup_llm(config)

    # ── Step 2: run query ─────────────────────────────────────────────────────
    pipeline = "  ".join(
        f"{color}[{name}]{_RESET}" for name, color in _AGENT_COLORS.items()
    )
    print(f"\nPipeline: {pipeline}")
    print(f"\nQuery: {_BOLD}{_DEMO_QUERY}{_RESET}")
    print("\nRunning… (typically 30-120 s)\n")

    result = orchestrator.process_query(_DEMO_QUERY)

    # ── Step 3: print agent traces ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{_BOLD}AGENT TRACES{_RESET}")
    print("=" * 70 + "\n")

    for trace in result.get("agent_traces", []):
        agent_name = trace.get("agent_name", "Unknown")
        message    = trace.get("message", "")
        timestamp  = trace.get("timestamp", "")

        color  = _AGENT_COLORS.get(agent_name, "")
        ts_str = f" {_DIM}({timestamp[:19]}){_RESET}" if timestamp else ""
        print(f"{color}{_BOLD}[{agent_name}]{_RESET}{ts_str}")
        print(f"{_DIM}{message}{_RESET}\n")

    # ── Step 4: print final answer ────────────────────────────────────────────
    print("=" * 70)
    print(f"{_BOLD}FINAL ANSWER{_RESET}")
    print("=" * 70 + "\n")

    answer = result.get("final_answer") or result.get("response", "")
    print(answer)

    # Metadata strip
    meta = result.get("metadata", {})
    print(
        f"\n{_DIM}Sources: {meta.get('num_sources', 0)}  |  "
        f"Messages: {meta.get('num_messages', 0)}  |  "
        f"Revisions: {meta.get('revision_rounds', 0)}{_RESET}"
    )

    # ── Step 5: save demo_session.json ────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    session_path = output_dir / "demo_session.json"
    with open(session_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n{_GREEN}✅ Session saved   → {session_path}{_RESET}")

    # ── Step 5b: append safety events to safety_log.jsonl ─────────────────────
    safety_log_path = output_dir / "safety_log.jsonl"
    safety_events   = result.get("safety_events", [])
    with open(safety_log_path, "a") as fh:
        if not safety_events:
            entry = {
                "timestamp":    datetime.now().isoformat(),
                "source":       "demo",
                "query":        _DEMO_QUERY[:120],
                "status":       "SAFE",
                "action_taken": "ALLOW",
                "category":     "NONE",
            }
            fh.write(json.dumps(entry) + "\n")
        else:
            for ev in safety_events:
                entry = {
                    "timestamp":    ev.get("timestamp", datetime.now().isoformat()),
                    "source":       "demo",
                    "query":        _DEMO_QUERY[:120],
                    "status":       ev.get("status", "SAFE"),
                    "action_taken": ev.get("action") or (
                        "REFUSE" if ev.get("status") == "BLOCKED" else "ALLOW"
                    ),
                    "category":     ev.get("category") or "NONE",
                    "screened":     ev.get("screened", ""),
                }
                fh.write(json.dumps(entry) + "\n")
    print(f"{_GREEN}✅ Safety log      → {safety_log_path}{_RESET}")

    # ── Step 6: save demo_answer.md ───────────────────────────────────────────
    answer_path = output_dir / "demo_answer.md"
    _write_demo_answer_md(answer_path, _DEMO_QUERY, result)
    print(f"{_GREEN}✅ Answer saved    → {answer_path}{_RESET}")

    # ── Step 7: run both judges concurrently ──────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{_BOLD}JUDGE EVALUATION{_RESET}")
    print("=" * 70 + "\n")

    judge     = LLMJudge(config)
    citations = result.get("citations", [])

    print("Running Judge 1 (Research Quality) + Judge 2 (Safety & Ethics)…\n")
    rq, se = await judge.run_both_judges(
        query=_DEMO_QUERY,
        response=answer,
        retrieved_sources=citations,
    )

    # ── Step 8: print judge scores ────────────────────────────────────────────
    _print_judge_result("Judge 1 — Research Quality", rq)
    _print_judge_result("Judge 2 — Safety & Ethics",  se)

    combined = round((rq.overall_score + se.overall_score) / 2, 2)
    print(f"{_BOLD}Combined overall score: {combined:.2f} / 5.00{_RESET}\n")

    # ── Step 9: append judge results to demo_session.json ────────────────────
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
        "combined_avg": combined,
        "evaluated_at": datetime.now().isoformat(),
    }

    with open(session_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"{_GREEN}✅ Judge results   → {session_path} (updated){_RESET}")
    print("=" * 70 + "\n")


# ── demo display helpers ──────────────────────────────────────────────────────

def _print_demo_header() -> None:
    print("\n" + "=" * 70)
    print(f"{_BOLD}  Multi-Agent HCI Research Assistant — Demo{_RESET}")
    print("=" * 70 + "\n")


def _print_judge_result(title: str, result) -> None:
    """Print one JudgeResult with a Unicode block-bar per criterion."""
    print(f"{_BOLD}{title}{_RESET}  (overall: {result.overall_score:.2f} / 5.00)")
    print("─" * 52)

    for crit, score in result.criterion_scores.items():
        filled  = "█" * score
        empty   = "░" * (5 - score)
        print(f"  {crit:<28} {filled}{empty}  {score}/5")

    if result.strengths:
        print(f"\n  Strengths:")
        for s in result.strengths[:3]:
            print(f"    ✓ {s}")

    if result.weaknesses:
        print(f"\n  Weaknesses:")
        for w in result.weaknesses[:3]:
            print(f"    ✗ {w}")

    print()


def _write_demo_answer_md(path: Path, query: str, result: dict) -> None:
    """
    Write the final answer and bibliography to a Markdown file.

    The file is structured for direct inclusion in a technical report:
      - YAML-style front matter with query, timestamp, and metadata
      - Full final answer (unmodified)
      - Bibliography section with numbered, linked citation entries
    """
    answer    = result.get("final_answer") or result.get("response", "")
    citations = result.get("citations", [])
    meta      = result.get("metadata", {})

    safety_events = result.get("safety_events", [])
    safety_note   = ""
    if any(ev.get("status") == "BLOCKED" for ev in safety_events):
        safety_note = "\n> ⚠️ **Note:** One or more safety events were detected during this session.\n"

    lines = [
        "# Demo Answer",
        "",
        f"**Query:** {query}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Sources:** {meta.get('num_sources', len(citations))}  |  "
        f"**Agent messages:** {meta.get('num_messages', 0)}  |  "
        f"**Revision rounds:** {meta.get('revision_rounds', 0)}",
        safety_note,
        "---",
        "",
        answer,
    ]

    if citations:
        lines += [
            "",
            "---",
            "",
            "## Bibliography",
            "",
        ]
        for cite in citations:
            idx     = cite.get("index", "?")
            title   = cite.get("title") or cite.get("raw") or "Untitled"
            url     = cite.get("url", "")
            snippet = cite.get("snippet", "")

            if url:
                entry = f"[{idx}] **[{title}]({url})**"
            else:
                entry = f"[{idx}] **{title}**"

            lines.append(entry)
            if snippet:
                preview = snippet[:200] + ("…" if len(snippet) > 200 else "")
                lines.append(f"> {preview}")
            lines.append("")

    lines += [
        "---",
        "_Generated by Multi-Agent HCI Research Assistant_",
    ]

    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser + entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Agent HCI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --ui web       Launch Streamlit web interface\n"
            "  python main.py --ui cli       Launch interactive CLI\n"
            "  python main.py --evaluate     Run 8-query batch evaluation\n"
            "  python main.py --demo         Run end-to-end demo query\n"
        ),
    )

    # The three entry points are mutually exclusive and exactly one is required.
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--ui",
        choices=["web", "cli"],
        metavar="{web,cli}",
        help="Launch UI: 'web' starts Streamlit, 'cli' starts the terminal loop",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Run the 8-query batch evaluation (data/eval_queries.json), "
            "apply both LLM judges, verify safety queries were refused, "
            "and write outputs/evaluation_report.{json,md}"
        ),
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run one end-to-end demo query, print all agent traces, "
            "score with both judges, and save outputs/demo_session.json "
            "and outputs/demo_answer.md"
        ),
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to configuration YAML file (default: config.yaml)",
    )

    args = parser.parse_args()

    if args.ui == "web":
        run_web()
    elif args.ui == "cli":
        run_cli()
    elif args.evaluate:
        asyncio.run(run_evaluation(args.config))
    elif args.demo:
        asyncio.run(run_demo(args.config))


if __name__ == "__main__":
    main()
