"""
Web Search Tool — Tavily-backed implementation with mock fallback.

Primary class : TavilySearchTool
  • Reads TAVILY_API_KEY from .env (loaded at module import via load_dotenv).
  • Returns top-N results as List[Dict] with keys: title, url, snippet,
    published_date.
  • Handles five error categories with clear user-facing messages:
      ImportError  – tavily-python package not installed
      Timeout      – API server did not respond in time
      HTTP 401     – invalid or expired API key
      HTTP 429     – rate limit exceeded
      HTTP 402     – monthly quota exhausted
      Other        – generic network or API failure
  • Activates MOCK MODE automatically when TAVILY_API_KEY is absent, returning
    three realistic HCI research results so the full pipeline can be tested
    without a live key.

AutoGen tool : web_search(query, provider, max_results) -> str
  • Fully synchronous — safe to call from both sync and AutoGen async contexts
    because it contains no asyncio.run() or event-loop manipulation.
  • AutoGen's FunctionTool wraps it and runs it in a thread pool executor.
  • Registered in ResearcherAgent via:
        FunctionTool(web_search, description="...")

Backwards-compatible export : WebSearchTool
  • Thin wrapper kept so src/tools/__init__.py continues to import
    WebSearchTool without changes.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Populate TAVILY_API_KEY (and any other .env vars) before the class reads them
load_dotenv()

logger = logging.getLogger("tools.web_search")

# Seconds to wait for a Tavily HTTP response before raising a timeout error
_REQUEST_TIMEOUT = 15


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class SearchError(Exception):
    """
    Raised by TavilySearchTool when a search cannot be completed.

    The message is always user-readable and suitable for returning directly
    to the agent as a tool-call result.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Mock results
# ─────────────────────────────────────────────────────────────────────────────

# Three realistic HCI research results served when TAVILY_API_KEY is absent.
# The query is ignored in mock mode; these results cover broadly relevant HCI
# topics so the full agent pipeline can be exercised end-to-end for testing.
_MOCK_RESULTS: List[Dict[str, str]] = [
    {
        "title": "Understanding User Mental Models in Interactive AI Systems",
        "url": "https://dl.acm.org/doi/10.1145/3544548.3580123",
        "snippet": (
            "CHI 2024 paper investigating how users form mental models when "
            "interacting with AI-driven interfaces. A mixed-methods study "
            "(n=48) identifies three distinct mental model types and shows "
            "that model accuracy correlates strongly with user trust "
            "calibration (r=0.71, p<0.001). Design implications include "
            "progressive disclosure of system capabilities and uncertainty "
            "signalling in AI outputs."
        ),
        "published_date": "2024-05-12",
    },
    {
        "title": (
            "Accessibility Barriers in Conversational Agents: "
            "A Systematic Review"
        ),
        "url": (
            "https://www.nngroup.com/articles/"
            "conversational-agents-accessibility-2023/"
        ),
        "snippet": (
            "Nielsen Norman Group review of 62 studies examining accessibility "
            "challenges faced by users with visual, motor, and cognitive "
            "disabilities when interacting with voice assistants and chatbots. "
            "Key findings: error-recovery failures appeared in 74% of studies; "
            "non-verbal feedback cues were inadequate in 81%; only 23% of "
            "tested agents met WCAG 2.1 Level AA criteria."
        ),
        "published_date": "2023-11-08",
    },
    {
        "title": (
            "Explainable AI in Practice: Design Patterns for Transparent UX"
        ),
        "url": "https://arxiv.org/abs/2401.12345",
        "snippet": (
            "Proposes a pattern language of 12 XAI design patterns derived "
            "from a card-sorting study with 30 UX practitioners. Patterns are "
            "organised into three tiers: surface-level explanations (what the "
            "system decided), interaction-level transparency (why it decided "
            "that), and system-level provenance (where the data came from). "
            "Validated through expert walkthroughs at three industry sites."
        ),
        "published_date": "2024-01-22",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TavilySearchTool
# ─────────────────────────────────────────────────────────────────────────────

class TavilySearchTool:
    """
    Web search tool backed by the Tavily Search API.

    Instantiation reads TAVILY_API_KEY from the environment.  If the key is
    absent the tool enters mock mode and returns static HCI research results
    instead of hitting the network.

    All public methods are synchronous — no asyncio required.

    Example::

        tool = TavilySearchTool(max_results=5)
        results = tool.search("explainable AI usability")
        for r in results:
            print(r["title"], r["url"])
    """

    def __init__(self, max_results: int = 5) -> None:
        """
        Initialise the tool and resolve the API key from the environment.

        Args:
            max_results: Maximum number of results to return per search.
                         Clamped to [1, 20] (Tavily's supported range).
        """
        self.max_results: int = max(1, min(max_results, 20))
        self.api_key: Optional[str] = os.getenv("TAVILY_API_KEY") or None
        self.mock_mode: bool = self.api_key is None

        if self.mock_mode:
            logger.warning(
                "TAVILY_API_KEY not found in environment — "
                "TavilySearchTool is running in MOCK MODE. "
                "Results are static HCI examples. "
                "Add TAVILY_API_KEY to .env for live search."
            )
        else:
            # Log only the last 4 characters to avoid exposing the key
            logger.info(
                "TavilySearchTool ready — live mode, key=...%s, max_results=%d",
                self.api_key[-4:],
                self.max_results,
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Search the web and return up to max_results structured results.

        Dispatches to the live Tavily API in normal mode or to the built-in
        mock dataset when TAVILY_API_KEY is absent.

        Args:
            query:           Search query string.
            search_depth:    "basic" (fast) or "advanced" (deeper, slower).
                             Ignored in mock mode.
            include_domains: Optional whitelist of domains to restrict results.
            exclude_domains: Optional blacklist of domains to exclude.

        Returns:
            List of dicts, each with keys:
              title          (str)  – page or article title
              url            (str)  – canonical URL
              snippet        (str)  – relevant excerpt from the page
              published_date (str)  – ISO date string or "" if unavailable

        Raises:
            SearchError: If the live API call fails (network error, bad key,
                         rate limit, timeout).  Mock mode never raises.
        """
        if not query or not query.strip():
            logger.warning("search() called with empty query — returning []")
            return []

        if self.mock_mode:
            return self._mock_search(query)

        return self._live_search(
            query=query.strip(),
            search_depth=search_depth,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
        )

    # ── Internal: live search ────────────────────────────────────────────────

    def _live_search(
        self,
        query: str,
        search_depth: str,
        include_domains: List[str],
        exclude_domains: List[str],
    ) -> List[Dict[str, str]]:
        """
        Call the Tavily REST API and return parsed results.

        Error classification:
          ImportError  → tavily-python not installed
          Timeout      → server did not respond within _REQUEST_TIMEOUT seconds
          HTTP 401     → API key invalid or expired
          HTTP 429     → rate limit hit
          HTTP 402     → monthly quota exhausted
          Other        → generic failure (logged with full traceback)

        Args:
            query:           Sanitised search string.
            search_depth:    "basic" or "advanced".
            include_domains: Domain whitelist (may be empty).
            exclude_domains: Domain blacklist (may be empty).

        Returns:
            List of result dicts in the standard format.

        Raises:
            SearchError: With a user-readable message on any failure.
        """
        # ── Import check ─────────────────────────────────────────────────────
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise SearchError(
                "The tavily-python package is not installed. "
                "Run:  pip install tavily-python"
            ) from exc

        # ── API call ─────────────────────────────────────────────────────────
        try:
            client = TavilyClient(api_key=self.api_key)

            # Build kwargs dict; only pass domain lists when non-empty to
            # avoid triggering Tavily validation errors on empty lists.
            call_kwargs: Dict[str, Any] = dict(
                query=query,
                max_results=self.max_results,
                search_depth=search_depth,
            )
            if include_domains:
                call_kwargs["include_domains"] = include_domains
            if exclude_domains:
                call_kwargs["exclude_domains"] = exclude_domains

            response: Dict[str, Any] = client.search(**call_kwargs)

        except Exception as exc:
            raise self._classify_error(exc, query) from exc

        results = self._parse_results(response)
        logger.info(
            "Tavily live search — query='%s', %d results returned",
            query[:80],
            len(results),
        )
        return results

    def _classify_error(self, exc: Exception, query: str) -> SearchError:
        """
        Map a raw Tavily / network exception to a user-readable SearchError.

        Inspects the exception message and any attached HTTP response for
        status codes.  Returns a SearchError with actionable guidance.

        Args:
            exc:   The original exception raised by the Tavily client.
            query: The search query (included in timeout messages).

        Returns:
            A SearchError with a user-facing message.
        """
        err_str = str(exc).lower()

        # HTTP status code attached by some HTTP libraries
        status: Optional[int] = None
        for attr in ("status_code", "status"):
            resp = getattr(exc, "response", None) or getattr(exc, "resp", None)
            code = getattr(resp, attr, None) or getattr(exc, attr, None)
            if isinstance(code, int):
                status = code
                break

        # ── Timeout ──────────────────────────────────────────────────────────
        if (
            "timeout" in err_str
            or "timed out" in err_str
            or isinstance(exc, TimeoutError)
        ):
            return SearchError(
                f"Tavily search timed out after {_REQUEST_TIMEOUT}s "
                f"for query: '{query}'. "
                "The API server may be under load. "
                "Try again in a moment or simplify the query."
            )

        # ── Invalid API key (HTTP 401) ────────────────────────────────────────
        if status == 401 or "401" in str(exc) or any(
            kw in err_str
            for kw in ("unauthorized", "invalid api key", "api key invalid")
        ):
            return SearchError(
                "Tavily API key is invalid or has expired. "
                "Check that TAVILY_API_KEY in .env matches your key at "
                "https://app.tavily.com/home — it starts with 'tvly-'."
            )

        # ── Rate limit (HTTP 429) ─────────────────────────────────────────────
        if status == 429 or "429" in str(exc) or "rate limit" in err_str:
            return SearchError(
                "Tavily rate limit reached. "
                "Wait a few seconds before retrying, or upgrade your Tavily "
                "plan at https://tavily.com."
            )

        # ── Quota exhausted (HTTP 402) ────────────────────────────────────────
        if status == 402 or "402" in str(exc) or any(
            kw in err_str for kw in ("quota", "payment required", "credits")
        ):
            return SearchError(
                "Tavily monthly search quota is exhausted. "
                "Upgrade your plan or wait for the monthly reset at "
                "https://app.tavily.com/home."
            )

        # ── Generic failure ───────────────────────────────────────────────────
        logger.error(
            "Unclassified Tavily error for query '%s': %s: %s",
            query[:80],
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return SearchError(
            f"Tavily search failed ({type(exc).__name__}): {exc}. "
            "Check your network connection and the Tavily service status."
        )

    def _parse_results(self, response: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert a raw Tavily JSON response into the standard result format.

        Keeps only the four keys the rest of the system depends on and
        normalises missing values to empty strings.

        Args:
            response: Dict returned directly by TavilyClient.search().

        Returns:
            List of result dicts with title, url, snippet, published_date.
        """
        results: List[Dict[str, str]] = []

        for item in response.get("results", [])[: self.max_results]:
            results.append(
                {
                    "title": (item.get("title") or "").strip(),
                    "url": (item.get("url") or "").strip(),
                    "snippet": (item.get("content") or "").strip(),
                    "published_date": (item.get("published_date") or ""),
                }
            )

        return results

    # ── Internal: mock search ────────────────────────────────────────────────

    def _mock_search(self, query: str) -> List[Dict[str, str]]:
        """
        Return static HCI results for testing without a live API key.

        The query is ignored; results are always the same three HCI research
        entries sliced to max_results.  A WARNING log is emitted so it is
        obvious in the console that mock mode is active.

        Args:
            query: Ignored; present only for logging context.

        Returns:
            Slice of _MOCK_RESULTS up to max_results entries.
        """
        count = min(len(_MOCK_RESULTS), self.max_results)
        logger.warning(
            "MOCK MODE — returning %d static HCI results (query ignored): '%s'",
            count,
            query[:80],
        )
        return _MOCK_RESULTS[:count]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — one initialisation per process
# ─────────────────────────────────────────────────────────────────────────────

# AutoGen calls web_search() on every tool invocation; caching the instance
# avoids re-reading the env var and re-logging the startup message each time.
_tool_cache: Dict[int, TavilySearchTool] = {}


def _get_tool(max_results: int) -> TavilySearchTool:
    """Return a cached TavilySearchTool for the given max_results value."""
    if max_results not in _tool_cache:
        _tool_cache[max_results] = TavilySearchTool(max_results=max_results)
    return _tool_cache[max_results]


# ─────────────────────────────────────────────────────────────────────────────
# AutoGen-compatible tool function
# ─────────────────────────────────────────────────────────────────────────────

def web_search(
    query: str,
    provider: str = "tavily",
    max_results: int = 5,
) -> str:
    """
    Search the web and return formatted results as a string.

    This is the function registered as an AutoGen FunctionTool in
    ResearcherAgent.  It is fully synchronous — no asyncio.run() or
    event-loop calls — so it is safe to call from within AutoGen's
    async team.run() context (AutoGen executes sync tools via a thread
    pool executor, which has its own event loop).

    Args:
        query:       Search query string.  Empty queries return immediately
                     with an explanatory message.
        provider:    Accepted for API compatibility; "tavily" is always used
                     regardless of this value (Tavily is the configured
                     provider via TAVILY_API_KEY in .env).
        max_results: Maximum number of results to return (1–20).

    Returns:
        Formatted multi-line string with numbered results suitable for an
        LLM to read as a tool call response.  Errors are reported as a
        single "[Web Search Error]" line rather than raising exceptions,
        so the agent can decide how to proceed.

    Output format::

        Web search results for '<query>' (N found):

        1. <Title>
           URL: <url>
           Published: <date>          ← omitted when not available
           <Snippet text>

        2. ...

    Error format::

        [Web Search Error] <user-readable explanation with next steps>
    """
    if not query or not query.strip():
        return "[Web Search] Empty query — please provide a search term."

    logger.info(
        "web_search called: query='%s', provider=%s, max_results=%d",
        query[:80],
        provider,
        max_results,
    )

    # provider parameter is accepted for backwards compatibility with code that
    # passes provider="tavily" explicitly, but TavilySearchTool is always used.
    if provider not in ("tavily", ""):
        logger.warning(
            "web_search: provider=%r is not supported; using Tavily.", provider
        )

    tool = _get_tool(max_results=max_results)

    # ── Execute search ────────────────────────────────────────────────────────
    try:
        results = tool.search(query.strip())
    except SearchError as exc:
        # Return the error as a readable string; do not raise so AutoGen can
        # continue the conversation rather than crashing the tool call.
        logger.warning("web_search returning error to agent: %s", exc)
        return f"[Web Search Error] {exc}"

    # ── Handle empty result set ───────────────────────────────────────────────
    if not results:
        return (
            f"No web search results found for: '{query}'. "
            "Suggestions: broaden the query, remove jargon, or try an "
            "alternative phrasing."
        )

    # ── Format results for the agent ─────────────────────────────────────────
    lines: List[str] = [
        f"Web search results for '{query}' ({len(results)} found):\n"
    ]

    for i, result in enumerate(results, 1):
        lines.append(f"{i}. {result['title']}")
        lines.append(f"   URL: {result['url']}")
        if result.get("published_date"):
            lines.append(f"   Published: {result['published_date']}")
        # Wrap long snippets at ~120 chars for readability in the agent context
        snippet = result.get("snippet", "").strip()
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")   # blank separator between results

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible WebSearchTool alias
# ─────────────────────────────────────────────────────────────────────────────

class WebSearchTool(TavilySearchTool):
    """
    Backwards-compatible wrapper so existing imports of WebSearchTool continue
    to work unchanged (used by src/tools/__init__.py and any legacy code).

    Accepts the old constructor signature:
        WebSearchTool(provider="tavily", max_results=5)

    The ``provider`` parameter is accepted but ignored; Tavily is always used.
    An async ``search()`` shim is provided for legacy callers that awaited it.
    """

    def __init__(self, provider: str = "tavily", max_results: int = 5) -> None:
        """
        Args:
            provider:    Ignored; kept for backwards compatibility.
            max_results: Maximum results per search.
        """
        if provider not in ("tavily", ""):
            logger.warning(
                "WebSearchTool: provider=%r is not supported — using Tavily.",
                provider,
            )
        super().__init__(max_results=max_results)

    async def search(self, query: str, **kwargs) -> List[Dict[str, str]]:  # type: ignore[override]
        """
        Async shim for legacy callers that awaited WebSearchTool.search().

        Delegates to the synchronous TavilySearchTool.search() method so
        existing async code that does ``await tool.search(query)`` continues
        to work without modification.

        Args:
            query:   Search query string.
            **kwargs: Forwarded to TavilySearchTool.search() (search_depth,
                      include_domains, exclude_domains).

        Returns:
            List of result dicts (same format as TavilySearchTool.search()).
        """
        # Run the synchronous search in the calling event loop's thread pool
        # so we don't block the loop.
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: super(WebSearchTool, self).search(query, **kwargs),
        )
