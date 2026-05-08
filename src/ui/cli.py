"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import yaml
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour codes
# ─────────────────────────────────────────────────────────────────────────────

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_RED     = "\033[31m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_BLUE    = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN    = "\033[36m"

_AGENT_COLORS: Dict[str, str] = {
    "Safety":     _RED,
    "Planner":    _BLUE,
    "Researcher": _GREEN,
    "Critic":     _YELLOW,
    "Writer":     _MAGENTA,
}

_COMMANDS: Dict[str, str] = {
    "/help":    "Show this help message",
    "/history": "List all queries from this session",
    "/export":  "Save session to outputs/last_session.json",
    "/quit":    "Exit the application",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI class
# ─────────────────────────────────────────────────────────────────────────────

class CLI:
    """
    Command-line interface for the multi-agent research assistant.

    Supports:
    - Interactive query submission with a styled prompt symbol
    - Live agent-progress output (replayed from agent_traces after the call)
    - Final answer, numbered citations, and safety event display
    - /help, /history, /export, /quit slash commands
    - Graceful handling of KeyboardInterrupt (Ctrl-C)
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as fh:
            self.config = yaml.safe_load(fh)

        self._setup_logging()
        self.logger = logging.getLogger("cli")

        try:
            self.orchestrator = AutoGenOrchestrator(self.config)
            self.logger.info("AutoGen orchestrator initialised successfully")
        except Exception as exc:
            self.logger.error("Failed to initialise orchestrator: %s", exc)
            raise

        self.running       = True
        self.query_count   = 0
        self.history: List[Dict[str, Any]] = []   # {timestamp, query, result}
        self.last_result: Optional[Dict[str, Any]] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        log_config = self.config.get("logging", {})
        # Use WARNING in CLI mode so logging noise doesn't pollute the terminal
        level  = log_config.get("level", "WARNING")
        fmt    = log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.basicConfig(level=getattr(logging, level, logging.WARNING), format=fmt)

    async def run(self) -> None:
        """
        Main interactive loop.

        Reads one line at a time from stdin.  Slash-commands are dispatched to
        their handlers; everything else is treated as a research query.
        """
        self._print_welcome()

        while self.running:
            try:
                raw = input(f"\n{_BOLD}{_CYAN}▶{_RESET} ").strip()
            except KeyboardInterrupt:
                print(f"\n\n{_YELLOW}Interrupted — type /quit to exit gracefully.{_RESET}")
                continue
            except EOFError:
                # stdin closed (e.g. piped input exhausted)
                self._print_goodbye()
                break

            if not raw:
                continue

            cmd = raw.lower()

            if cmd in ("/quit", "/exit", "quit", "exit", "q"):
                self._print_goodbye()
                break
            elif cmd in ("/help", "help"):
                self._print_help()
            elif cmd == "/history":
                self._print_history()
            elif cmd == "/export":
                self._export_session()
            elif cmd == "clear":
                self._clear_screen()
            elif cmd == "stats":
                self._print_stats()
            else:
                self._process_and_display(raw)

    # ── query processing ──────────────────────────────────────────────────────

    def _process_and_display(self, query: str) -> None:
        """
        Run a research query through the orchestrator and display results.

        Shows a spinner while the agents are working, then replays each agent's
        turn with a coloured [AgentName] prefix, followed by the final answer,
        citations, and any safety events.
        """
        print(f"\n{'═' * 70}")
        print(f"{_BOLD}Processing query…{_RESET}")
        print(f"{'─' * 70}")
        pipeline = "  ".join(
            f"{color}[{name}]{_RESET}"
            for name, color in _AGENT_COLORS.items()
        )
        print(f"{_DIM}Pipeline:{_RESET} {pipeline}\n")

        # Spinner runs in a background thread while the orchestrator blocks
        stop_event = threading.Event()
        spinner_thread = threading.Thread(
            target=self._spinner_loop, args=(stop_event,), daemon=True
        )
        spinner_thread.start()

        try:
            result = self.orchestrator.process_query(query)
        except Exception as exc:
            stop_event.set()
            spinner_thread.join(timeout=0.5)
            print(f"\n{_RED}Error calling orchestrator: {exc}{_RESET}")
            logging.exception("process_query raised")
            return
        finally:
            stop_event.set()
            spinner_thread.join(timeout=0.5)

        self.query_count += 1
        self.last_result = result
        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query":     query,
            "result":    result,
        })

        # Replay agent traces with live-style [AgentName] prefix
        self._display_agent_traces(result)

        # Print answer, citations, safety events
        self._display_result(result)

    @staticmethod
    def _spinner_loop(stop_event: threading.Event) -> None:
        """Rotating-dot spinner printed to stdout while stop_event is clear."""
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not stop_event.is_set():
            print(
                f"\r{_DIM}  Agents working {frames[i % len(frames)]}{_RESET}",
                end="",
                flush=True,
            )
            time.sleep(0.1)
            i += 1
        # Erase the spinner line
        print("\r" + " " * 35 + "\r", end="", flush=True)

    # ── display helpers ───────────────────────────────────────────────────────

    def _display_agent_traces(self, result: Dict[str, Any]) -> None:
        """
        Print each agent turn from agent_traces with a coloured [AgentName] prefix.

        Shows a short preview of each message so the terminal doesn't flood.
        """
        traces: List[Dict[str, Any]] = result.get("agent_traces", [])
        if not traces:
            return

        print(f"{_BOLD}Agent Activity:{_RESET}")
        print(f"{'─' * 70}")

        for turn in traces:
            agent_name = turn.get("agent_name", "Unknown")
            message    = turn.get("message", "")
            timestamp  = turn.get("timestamp", "")

            color = _AGENT_COLORS.get(agent_name, "")
            label = f"{color}{_BOLD}[{agent_name}]{_RESET}"
            ts    = (
                f" {_DIM}({timestamp[:19]}){_RESET}"
                if timestamp else ""
            )
            print(f"{label}{ts}")

            # Single-line preview — truncate at 220 chars
            preview = message.replace("\n", " ")[:220]
            if len(message) > 220:
                preview += "…"
            print(f"  {_DIM}{preview}{_RESET}\n")

    def _display_result(self, result: Dict[str, Any]) -> None:
        """
        Display the full query result: safety events, final answer, citations,
        and a one-line metadata footer.
        """
        # Safety events first (before the answer so warnings are visible)
        self._display_safety_events(result)

        # Final answer
        print(f"\n{'═' * 70}")
        print(f"{_BOLD}RESEARCH ANSWER{_RESET}")
        print(f"{'═' * 70}")

        if result.get("metadata", {}).get("error"):
            print(f"\n{_RED}⚠ {result.get('error', 'An error occurred.')}{_RESET}")
        else:
            answer = result.get("final_answer") or result.get("response", "")
            if answer:
                print(f"\n{answer}\n")
            else:
                print(f"\n{_DIM}No answer was generated.{_RESET}\n")

        # Numbered citations list
        self._display_citations(result)

        # Metadata footer
        meta = result.get("metadata", {})
        print(
            f"\n{_DIM}{'─' * 70}\n"
            f"Sources: {meta.get('num_sources', 0)}  |  "
            f"Messages: {meta.get('num_messages', 0)}  |  "
            f"Revisions: {meta.get('revision_rounds', 0)}  |  "
            f"Query #{self.query_count}"
            f"{_RESET}"
        )

    def _display_safety_events(self, result: Dict[str, Any]) -> None:
        """
        Print any safety events returned by the orchestrator.

        BLOCKED → red  |  SANITIZE → yellow  |  WARN → yellow  |  SAFE → green
        """
        events: List[Dict[str, Any]] = result.get("safety_events", [])
        if not events:
            return

        print(f"\n{_BOLD}Safety Events:{_RESET}")
        print(f"{'─' * 70}")

        for ev in events:
            status   = ev.get("status", "SAFE")
            screened = ev.get("screened", "")
            category = ev.get("category", "")
            action   = ev.get("action", "")
            guidance = ev.get("guidance", "")

            if status == "BLOCKED":
                header = f"{_RED}{_BOLD}🚫 BLOCKED{_RESET}"
                body_lines = [
                    f"  Screened : {screened}",
                    f"  Category : {_BOLD}{category}{_RESET}" if category else "",
                    f"  Action   : {action}" if action else "",
                    f"  Guidance : {_DIM}{guidance[:240]}{_RESET}" if guidance else "",
                ]
            elif action == "SANITIZE":
                header = f"{_YELLOW}{_BOLD}✏️  SANITIZED{_RESET}"
                body_lines = [
                    f"  Screened : {screened}",
                    f"  Category : {category}" if category else "",
                ]
            elif action == "WARN":
                header = f"{_YELLOW}{_BOLD}⚠️  WARNING{_RESET}"
                body_lines = [
                    f"  Screened : {screened}",
                    f"  Category : {category}" if category else "",
                    f"  {_DIM}{guidance[:240]}{_RESET}" if guidance else "",
                ]
            else:
                header = f"{_GREEN}✅ SAFE{_RESET}"
                body_lines = [f"  Screened : {screened}"]

            print(header)
            for line in body_lines:
                if line:
                    print(line)
            print()

    def _display_citations(self, result: Dict[str, Any]) -> None:
        """
        Print a numbered citation list from result['citations'].

        Falls back to URL extraction from conversation_history when the
        orchestrator returns no structured citations.
        """
        citations: List[Dict[str, Any]] = result.get("citations", [])

        if not citations:
            return

        print(f"\n{_BOLD}📚 Citations:{_RESET}")
        print(f"{'─' * 70}")

        for cite in citations:
            idx     = cite.get("index", "?")
            title   = cite.get("title") or cite.get("raw") or "Source"
            url     = cite.get("url", "")
            snippet = cite.get("snippet", "")

            print(f"[{idx}] {_BOLD}{title}{_RESET}")
            if url and url != title:
                print(f"     {_DIM}{url}{_RESET}")
            if snippet:
                preview = snippet[:180] + ("…" if len(snippet) > 180 else "")
                print(f"     {_DIM}{preview}{_RESET}")

    # ── command handlers ──────────────────────────────────────────────────────

    def _print_welcome(self) -> None:
        system_name = self.config.get("system", {}).get("name", "Multi-Agent Research Assistant")
        topic       = self.config.get("system", {}).get("topic", "HCI Research")
        model_name  = os.getenv(
            "OPENAI_MODEL",
            self.config.get("models", {}).get("default", {}).get("name", "—"),
        )

        print(f"\n{_BOLD}{'═' * 70}{_RESET}")
        print(f"{_BOLD}{_CYAN}  {system_name}{_RESET}")
        print(f"  Topic: {topic}   Model: {model_name}")
        print(f"{_BOLD}{'═' * 70}{_RESET}")
        print(f"\nWelcome!  Ask me anything about {_BOLD}{topic}{_RESET}.")
        print(f"Type {_BOLD}/help{_RESET} for available commands.\n")

        # Show the agent pipeline
        print(f"{_DIM}Agent pipeline:{_RESET}", end="  ")
        for name, color in _AGENT_COLORS.items():
            print(f"{color}{_BOLD}[{name}]{_RESET}", end="  ")
        print(f"\n{_DIM}All five agents collaborate on every query.{_RESET}\n")

    def _print_help(self) -> None:
        print(f"\n{_BOLD}Available Commands:{_RESET}")
        print(f"{'─' * 40}")
        for cmd, desc in _COMMANDS.items():
            print(f"  {_CYAN}{cmd:<12}{_RESET} {desc}")
        print(f"\n  {_DIM}Or type any research question to begin.{_RESET}\n")

    def _print_history(self) -> None:
        if not self.history:
            print(f"\n{_DIM}No queries in this session yet.{_RESET}")
            return

        print(f"\n{_BOLD}Session History ({len(self.history)} queries):{_RESET}")
        print(f"{'─' * 70}")

        for i, item in enumerate(self.history, 1):
            ts      = item.get("timestamp", "")
            query   = item.get("query", "")
            meta    = item["result"].get("metadata", {})
            n_src   = meta.get("num_sources", 0)
            blocked = meta.get("safety_blocked", False)
            flag    = f" {_RED}[BLOCKED]{_RESET}" if blocked else ""

            print(f"  {_BOLD}{i}.{_RESET} [{ts}] {query[:72]}{flag}")
            print(f"     {_DIM}{n_src} source(s){_RESET}")

        print()

    def _export_session(self) -> None:
        if not self.history:
            print(f"\n{_YELLOW}No session data to export yet.{_RESET}")
            return

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "last_session.json"

        payload = {
            "export_timestamp": datetime.now().isoformat(),
            "query_count": self.query_count,
            "history": [
                {
                    "timestamp":   item["timestamp"],
                    "query":       item["query"],
                    "final_answer": (
                        item["result"].get("final_answer")
                        or item["result"].get("response", "")
                    ),
                    "citations":    item["result"].get("citations", []),
                    "safety_events": item["result"].get("safety_events", []),
                    "metadata":     item["result"].get("metadata", {}),
                }
                for item in self.history
            ],
        }

        with open(output_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        print(f"\n{_GREEN}✓ Session exported → {output_path}{_RESET}")

    def _print_goodbye(self) -> None:
        print(f"\n{_BOLD}{'═' * 70}{_RESET}")
        print("Thank you for using the Multi-Agent Research Assistant!")
        print(f"Queries this session: {self.query_count}")
        print(f"{_BOLD}{'═' * 70}{_RESET}\n")

    def _clear_screen(self) -> None:
        os.system("clear" if os.name == "posix" else "cls")

    def _print_stats(self) -> None:
        print(f"\n{_BOLD}System Statistics:{_RESET}")
        print(f"  Queries processed : {self.query_count}")
        print(f"  System            : {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic             : {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model             : {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")

    # ── backwards-compat helpers (kept for any external callers) ─────────────

    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Return structured citations; fall back to URL extraction."""
        citations = result.get("citations", [])
        if citations:
            return [c.get("url", c.get("title", "")) for c in citations]
        seen: List[str] = []
        for msg in result.get("conversation_history", []):
            for url in re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", msg.get("content", "")):
                if url not in seen:
                    seen.append(url)
        return seen[:10]

    def _should_show_traces(self) -> bool:
        return self.config.get("ui", {}).get("verbose", False)

    def _display_conversation_summary(self, conversation_history: list) -> None:
        if not conversation_history:
            return
        print(f"\n{'─' * 70}\n🔍 CONVERSATION SUMMARY\n{'─' * 70}")
        for i, msg in enumerate(conversation_history, 1):
            agent   = msg.get("source", "Unknown")
            content = msg.get("content", "")
            preview = (content[:150] + "…") if len(content) > 150 else content
            print(f"\n{i}. {agent}:\n   {preview.replace(chr(10), ' ')}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    try:
        cli = CLI(config_path=args.config)
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye!\n")
    except Exception as exc:
        print(f"\n{_RED}Fatal error: {exc}{_RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
