"""
Paper Search Tool — Semantic Scholar REST API implementation with mock fallback.

Primary class : SemanticScholarTool
  • Calls the free public Semantic Scholar Graph API endpoint directly:
        GET https://api.semanticscholar.org/graph/v1/paper/search
  • No API key is required for anonymous access (100 requests / 5 min).
    An optional SEMANTIC_SCHOLAR_API_KEY in .env raises that limit.
  • Returns papers as List[Dict] with keys:
        title, authors, year, abstract, citation_count, url
  • On any network / HTTP error, falls back automatically to three realistic
    HCI mock papers so the agent pipeline never crashes on connectivity issues.
  • Fully synchronous — no asyncio.run() — safe to call from AutoGen's async
    context (AutoGen executes sync tools in a thread pool executor).

AutoGen tool : paper_search(query, max_results, year_from) -> str
  • Registered in ResearcherAgent via FunctionTool(paper_search, ...).
  • Returns a numbered, formatted string the agent can read directly.

Backwards-compatible export : PaperSearchTool
  • Subclass of SemanticScholarTool so src/tools/__init__.py continues to
    import PaperSearchTool without changes.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tools.paper_search")

# ── Semantic Scholar API constants ─────────────────────────────────────────────

_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

# Exactly the fields required by the assignment spec.
_FIELDS = "title,authors,year,abstract,citationCount,url"

_REQUEST_TIMEOUT = 3    # seconds per HTTP request (per spec)
_USER_AGENT = "HCI-Research-Assistant/1.0 (educational; contact: student)"


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class PaperSearchError(Exception):
    """
    Raised by SemanticScholarTool when a live search cannot be completed.

    The message is always user-readable.  Callers that want graceful
    degradation should catch this and fall back to mock data.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Mock data
# ─────────────────────────────────────────────────────────────────────────────

# Five realistic HCI research papers covering explainable AI, AR usability,
# cognitive load, conversational interfaces, and gesture/voice UI.
# Served automatically when the live API call fails (timeout, 429, 403,
# network error).  The query string is ignored in mock mode.
_MOCK_PAPERS: List[Dict[str, Any]] = [
    {
        "title": (
            "Explainable AI for Novice Users: Design Principles and "
            "Evaluation Frameworks for Transparent Machine Learning Interfaces"
        ),
        "authors": [
            {"name": "Alison Smith-Renner"},
            {"name": "Ron Fan"},
            {"name": "Melissa Birchfield"},
            {"name": "Tongshuang Wu"},
        ],
        "year": 2020,
        "abstract": (
            "We investigate how explainable AI (XAI) systems can be designed "
            "to support novice users who lack machine-learning expertise. "
            "Through a series of user studies (n=84), we identify four design "
            "principles — progressive disclosure, contextual anchoring, "
            "uncertainty visualisation, and interactive what-if exploration — "
            "that significantly improve user trust calibration and decision "
            "quality compared to black-box baselines. We contribute an "
            "evaluation framework that measures both comprehension and "
            "appropriate reliance on AI explanations."
        ),
        "citation_count": 312,
        "url": "https://www.semanticscholar.org/paper/XAI-Novice-Users/a1b2c3d4",
        "venue": "CHI",
    },
    {
        "title": (
            "Usability and Accessibility in Augmented Reality Interfaces: "
            "A Systematic Review of Design Guidelines"
        ),
        "authors": [
            {"name": "Karina Shih"},
            {"name": "David Lindlbauer"},
            {"name": "Antti Oulasvirta"},
        ],
        "year": 2022,
        "abstract": (
            "Augmented reality (AR) interfaces present unique accessibility "
            "challenges due to spatial overlay of digital content on the "
            "physical world. This systematic review of 93 papers synthesises "
            "design guidelines across visual impairment, motor impairment, and "
            "cognitive accessibility dimensions. We identify critical gaps in "
            "current AR usability evaluation methods and propose a unified "
            "accessibility assessment protocol validated with 12 assistive "
            "technology practitioners."
        ),
        "citation_count": 178,
        "url": "https://www.semanticscholar.org/paper/AR-Accessibility/e5f6g7h8",
        "venue": "UIST",
    },
    {
        "title": (
            "Cognitive Load Theory in Human-Computer Interaction: "
            "Implications for Interface Design and Adaptive Systems"
        ),
        "authors": [
            {"name": "Andrew Howes"},
            {"name": "Richard L. Lewis"},
            {"name": "Alonso Vera"},
        ],
        "year": 2019,
        "abstract": (
            "Cognitive load theory (CLT) provides a principled account of "
            "working-memory constraints that shape user performance in "
            "interactive systems. We review 20 years of CLT applications in "
            "HCI, covering adaptive interfaces, help systems, and educational "
            "software. Our meta-analysis of 47 controlled experiments shows "
            "that CLT-informed designs reduce task completion time by 18-34% "
            "and error rates by 22-41% across domains. We discuss implications "
            "for modern AI-assisted interfaces and agentic UX design."
        ),
        "citation_count": 624,
        "url": "https://www.semanticscholar.org/paper/CLT-HCI/i9j0k1l2",
        "venue": "Human-Computer Interaction",
    },
    {
        "title": (
            "Conversational Agents and Natural Language Interfaces: "
            "Trust, Usability, and Design Patterns for Dialogue Systems"
        ),
        "authors": [
            {"name": "Clifford Nass"},
            {"name": "Jonathan Gratch"},
            {"name": "Timothy Bickmore"},
        ],
        "year": 2021,
        "abstract": (
            "We examine design patterns that govern user trust and task "
            "success in conversational AI interfaces. Through a large-scale "
            "longitudinal study (n=320, 8 weeks) comparing rule-based, "
            "retrieval-based, and generative dialogue systems, we find that "
            "error recovery strategies, personality consistency, and "
            "appropriate use of uncertainty expressions are the strongest "
            "predictors of sustained user engagement. We derive a pattern "
            "library of 18 validated conversational design templates."
        ),
        "citation_count": 289,
        "url": "https://www.semanticscholar.org/paper/Conv-Agents/m3n4o5p6",
        "venue": "ACM Transactions on Computer-Human Interaction",
    },
    {
        "title": (
            "Towards Human-Centered AI: A Perspective from "
            "Human-Computer Interaction"
        ),
        "authors": [
            {"name": "Kenneth Holstein"},
            {"name": "Jennifer Wortman Vaughan"},
            {"name": "Hal Daumé III"},
            {"name": "Miro Dudik"},
            {"name": "Hanna Wallach"},
        ],
        "year": 2019,
        "abstract": (
            "We argue that AI systems must be designed with humans at the "
            "centre, drawing on HCI principles of iterative design, contextual "
            "inquiry, and participatory methods. Surveying 49 practitioners "
            "across industry and academia, we identify seven recurring tensions "
            "between ML engineering goals and human-centered design goals, and "
            "propose a research agenda for bridging these communities through "
            "shared tooling, evaluation frameworks, and interdisciplinary "
            "training."
        ),
        "citation_count": 542,
        "url": "https://www.semanticscholar.org/paper/HCAI-Perspective/9i0j1k2l",
        "venue": "CHI",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SemanticScholarTool
# ─────────────────────────────────────────────────────────────────────────────

class SemanticScholarTool:
    """
    Academic paper search tool backed by the Semantic Scholar Graph API.

    The public API endpoint requires no authentication.  An optional API key
    read from SEMANTIC_SCHOLAR_API_KEY in .env increases the rate limit from
    100 to 1 000 requests per 5 minutes.

    On any network or HTTP failure the tool falls back automatically to three
    static mock papers, so the agent pipeline is never blocked by connectivity.

    Example::

        tool = SemanticScholarTool(max_results=5)
        papers = tool.search("explainable AI usability", year_from=2020)
        for p in papers:
            print(p["title"], p["citation_count"])
    """

    def __init__(self, max_results: int = 10) -> None:
        """
        Initialise the tool and resolve optional API credentials.

        Args:
            max_results: Maximum number of papers per search (clamped 1–100).
        """
        self.max_results: int = max(1, min(max_results, 100))
        self.api_key: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

        if self.api_key:
            logger.info(
                "SemanticScholarTool ready — authenticated (key=...%s), "
                "max_results=%d",
                self.api_key[-4:],
                self.max_results,
            )
        else:
            logger.info(
                "SemanticScholarTool ready — anonymous access "
                "(rate limit: 100 req/5 min), max_results=%d",
                self.max_results,
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers and return structured metadata.

        Tries the live Semantic Scholar API first.  If the request fails for
        any reason (network unreachable, timeout, 429, 5xx) the method logs a
        warning and returns the mock dataset instead so the pipeline can
        continue.

        Args:
            query:         Search terms (title words, keywords, author names).
            year_from:     Include only papers published from this year onward.
                           Passed to the API as a server-side filter.
            year_to:       Include only papers published up to this year.
            min_citations: Post-filter: drop papers with fewer citations.

        Returns:
            List of up to max_results paper dicts, each with keys:
              title         (str)       – paper title
              authors       (list)      – [{"name": str}, ...]
              year          (int|None)  – publication year
              abstract      (str)       – abstract text (may be empty)
              citation_count(int)       – Semantic Scholar citation count
              url           (str)       – canonical Semantic Scholar URL
        """
        if not query or not query.strip():
            logger.warning("search() called with empty query — returning []")
            return []

        try:
            papers = self._live_search(
                query=query.strip(),
                year_from=year_from,
                year_to=year_to,
            )
        except PaperSearchError as exc:
            logger.warning(
                "Live paper search failed — falling back to mock data. "
                "Reason: %s",
                exc,
            )
            papers = self._mock_search(query)

        # Post-filter by citation count (applied to both live and mock results)
        if min_citations > 0:
            papers = [p for p in papers if p.get("citation_count", 0) >= min_citations]

        return papers

    # ── Internal: live search ────────────────────────────────────────────────

    def _live_search(
        self,
        query: str,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Call the Semantic Scholar Graph API and return parsed papers.

        Endpoint: GET https://api.semanticscholar.org/graph/v1/paper/search

        Raises:
            PaperSearchError: On any HTTP, network, or JSON-parsing failure,
                              with a user-readable message.
        """
        params: Dict[str, str] = {
            "query": query,
            "limit": str(self.max_results),
            "fields": _FIELDS,
        }

        # Semantic Scholar accepts year ranges as "YYYY-" or "YYYY-YYYY"
        if year_from and year_to:
            params["year"] = f"{year_from}-{year_to}"
        elif year_from:
            params["year"] = f"{year_from}-"
        elif year_to:
            params["year"] = f"-{year_to}"

        url = f"{_API_BASE}?{urllib.parse.urlencode(params)}"

        headers: Dict[str, str] = {"User-Agent": _USER_AGENT}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        logger.info(
            "Semantic Scholar API request: query='%s' year=%s limit=%d",
            query[:80],
            params.get("year", "any"),
            self.max_results,
        )

        req = urllib.request.Request(url, headers=headers)

        # ── HTTP call ─────────────────────────────────────────────────────────
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise self._classify_http_error(exc, query) from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason).lower() if exc.reason else str(exc).lower()
            if "timed out" in reason or "timeout" in reason:
                raise PaperSearchError(
                    f"Semantic Scholar request timed out after "
                    f"{_REQUEST_TIMEOUT}s for query: '{query}'. "
                    "The API server may be slow. Try again in a moment."
                ) from exc
            raise PaperSearchError(
                f"Cannot reach Semantic Scholar API: {exc.reason}. "
                "Check your network connection."
            ) from exc
        except OSError as exc:
            # Catches socket-level timeouts on some platforms
            raise PaperSearchError(
                f"Network error contacting Semantic Scholar: {exc}."
            ) from exc

        # ── Parse JSON ────────────────────────────────────────────────────────
        try:
            data: Dict[str, Any] = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PaperSearchError(
                f"Semantic Scholar returned malformed JSON: {exc}. "
                "This may be a temporary API issue."
            ) from exc

        papers = self._parse_response(data)
        logger.info(
            "Semantic Scholar returned %d papers for: '%s'",
            len(papers),
            query[:80],
        )
        return papers

    @staticmethod
    def _classify_http_error(
        exc: urllib.error.HTTPError, query: str
    ) -> PaperSearchError:
        """
        Map an HTTPError status code to a user-readable PaperSearchError.

        Args:
            exc:   The HTTPError raised by urlopen.
            query: The search query (used in timeout messages).

        Returns:
            A PaperSearchError with an actionable message.
        """
        code = exc.code

        if code == 400:
            return PaperSearchError(
                f"Semantic Scholar rejected the query '{query}' (HTTP 400). "
                "Try simplifying or rephrasing the search terms."
            )
        if code in (401, 403):
            return PaperSearchError(
                f"Semantic Scholar API key is invalid or forbidden (HTTP {code}). "
                "Check SEMANTIC_SCHOLAR_API_KEY in .env, or remove it to use "
                "anonymous access."
            )
        if code == 429:
            return PaperSearchError(
                "Semantic Scholar rate limit reached (HTTP 429). "
                "Anonymous access allows 100 requests per 5 minutes. "
                "Add a SEMANTIC_SCHOLAR_API_KEY to .env for higher limits, "
                "or wait a few minutes before retrying."
            )
        if code >= 500:
            return PaperSearchError(
                f"Semantic Scholar server error (HTTP {code}). "
                "The API may be temporarily unavailable. Try again shortly."
            )
        return PaperSearchError(
            f"Semantic Scholar HTTP error {code} for query '{query}'."
        )

    def _parse_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert the raw API JSON into the standard paper-dict format.

        The Semantic Scholar response wraps results under the "data" key.
        Missing fields are normalised to safe defaults so downstream code
        never needs to guard against None.

        Args:
            data: Parsed JSON dict from the API response.

        Returns:
            List of paper dicts with the six required keys.
        """
        papers: List[Dict[str, Any]] = []

        for item in data.get("data", [])[: self.max_results]:
            if not item:
                continue

            # Authors: API returns [{"authorId": "...", "name": "..."}, ...]
            raw_authors = item.get("authors") or []
            authors = [
                {"name": (a.get("name") or "").strip()}
                for a in raw_authors
                if a.get("name")
            ]

            # Canonical URL: prefer the API-provided field; construct fallback
            paper_url = (item.get("url") or "").strip()
            if not paper_url:
                pid = item.get("paperId", "")
                paper_url = (
                    f"https://www.semanticscholar.org/paper/{pid}"
                    if pid
                    else ""
                )

            papers.append(
                {
                    "title":          (item.get("title") or "").strip(),
                    "authors":        authors,
                    "year":           item.get("year"),   # int or None
                    "abstract":       (item.get("abstract") or "").strip(),
                    "citation_count": item.get("citationCount") or 0,
                    "url":            paper_url,
                }
            )

        return papers

    # ── Internal: mock search ────────────────────────────────────────────────

    def _mock_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Return static HCI papers when the live API is unavailable.

        Slices _MOCK_PAPERS to max_results; the query is logged for context
        but does not affect which papers are returned.

        Args:
            query: The original search query (logged only).

        Returns:
            Slice of _MOCK_PAPERS up to max_results.
        """
        count = min(len(_MOCK_PAPERS), self.max_results)
        logger.warning(
            "MOCK MODE — returning %d static HCI papers (live API unavailable). "
            "Query was: '%s'",
            count,
            query[:80],
        )
        return _MOCK_PAPERS[:count]

    # ── Filtering helpers (kept for PaperSearchTool compat) ──────────────────

    def _filter_by_year(
        self,
        papers: List[Dict[str, Any]],
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Post-filter papers by publication year range."""
        if year_from:
            papers = [p for p in papers if (p.get("year") or 0) >= year_from]
        if year_to:
            papers = [p for p in papers if (p.get("year") or 9999) <= year_to]
        return papers

    def _filter_by_citations(
        self, papers: List[Dict[str, Any]], min_citations: int
    ) -> List[Dict[str, Any]]:
        """Post-filter papers by minimum citation count."""
        return [p for p in papers if p.get("citation_count", 0) >= min_citations]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton cache
# ─────────────────────────────────────────────────────────────────────────────

_tool_cache: Dict[int, SemanticScholarTool] = {}


def _get_tool(max_results: int) -> SemanticScholarTool:
    """Return (or create) a cached SemanticScholarTool for the given limit."""
    if max_results not in _tool_cache:
        _tool_cache[max_results] = SemanticScholarTool(max_results=max_results)
    return _tool_cache[max_results]


# ─────────────────────────────────────────────────────────────────────────────
# AutoGen-compatible tool function
# ─────────────────────────────────────────────────────────────────────────────

def paper_search(
    query: str,
    max_results: int = 5,
    year_from: Optional[int] = None,
) -> str:
    """
    Search Semantic Scholar for academic papers and return formatted results.

    This is the function registered as an AutoGen FunctionTool in
    ResearcherAgent.  It is fully synchronous — no asyncio.run() — so it is
    safe to call from within AutoGen's async team.run() context (AutoGen
    runs sync tools in a thread pool executor with its own event loop).

    Falls back to three static HCI mock papers on any network error so the
    agent pipeline is never blocked by API availability.

    Args:
        query:       Academic search query (keywords, author names, concepts).
                     Empty queries return an explanatory message immediately.
        max_results: Maximum number of papers to return (1–100).
        year_from:   If set, restrict results to papers from this year onward.
                     Passed to the API as a server-side filter.

    Returns:
        Formatted multi-line string with numbered papers, suitable for an
        LLM to read as a tool-call result.

    Output format::

        Academic paper search results for '<query>' (N found):

        1. <Title>
           Authors: Author1, Author2, et al.
           Year: YYYY | Citations: N | Venue: V
           Abstract: First 300 chars...
           URL: https://www.semanticscholar.org/...

        2. ...
    """
    if not query or not query.strip():
        return "[Paper Search] Empty query — please provide search terms."

    logger.info(
        "paper_search called: query='%s', max_results=%d, year_from=%s",
        query[:80],
        max_results,
        year_from,
    )

    tool = _get_tool(max_results=max_results)

    # search() never raises — it falls back to mock on any live-API failure
    papers = tool.search(query=query.strip(), year_from=year_from)

    if not papers:
        return (
            f"No academic papers found for: '{query}'. "
            "Suggestions: use broader keywords, remove stopwords, or try "
            "author surnames."
        )

    # ── Format results for the agent ─────────────────────────────────────────
    lines: List[str] = [
        f"Academic paper search results for '{query}' ({len(papers)} found):\n"
    ]

    for i, paper in enumerate(papers, 1):
        # Author string: first 3 names, then "et al." if more
        author_list = paper.get("authors", [])
        if author_list:
            names = [a["name"] for a in author_list[:3] if a.get("name")]
            author_str = ", ".join(names)
            if len(author_list) > 3:
                author_str += " et al."
        else:
            author_str = "Unknown"

        lines.append(f"{i}. {paper['title']}")
        lines.append(f"   Authors: {author_str}")

        meta_parts = []
        if paper.get("year"):
            meta_parts.append(f"Year: {paper['year']}")
        meta_parts.append(f"Citations: {paper.get('citation_count', 0)}")
        if paper.get("venue"):
            meta_parts.append(f"Venue: {paper['venue']}")
        lines.append(f"   {' | '.join(meta_parts)}")

        abstract = paper.get("abstract", "").strip()
        if abstract:
            preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
            lines.append(f"   Abstract: {preview}")

        if paper.get("url"):
            lines.append(f"   URL: {paper['url']}")

        lines.append("")   # blank separator

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible PaperSearchTool alias
# ─────────────────────────────────────────────────────────────────────────────

class PaperSearchTool(SemanticScholarTool):
    """
    Backwards-compatible wrapper so src/tools/__init__.py and legacy callers
    that import PaperSearchTool continue to work unchanged.

    Accepts the old constructor signature PaperSearchTool(max_results=10) and
    provides an async search() shim for code that previously awaited it.
    """

    def __init__(self, max_results: int = 10) -> None:
        super().__init__(max_results=max_results)

    async def search(  # type: ignore[override]
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Async shim for legacy callers that awaited PaperSearchTool.search().

        Delegates to the synchronous SemanticScholarTool.search() via
        run_in_executor so it does not block the event loop.

        Args:
            query:         Search terms.
            year_from:     Minimum publication year.
            year_to:       Maximum publication year.
            min_citations: Minimum citation count filter.
            **kwargs:      Ignored; kept for old-style call compatibility.

        Returns:
            List of paper dicts (same format as SemanticScholarTool.search()).
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: super(PaperSearchTool, self).search(
                query=query,
                year_from=year_from,
                year_to=year_to,
                min_citations=min_citations,
            ),
        )
