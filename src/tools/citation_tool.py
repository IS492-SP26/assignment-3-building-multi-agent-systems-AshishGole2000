"""
Citation Tool
Formats citations and manages citation lists.

This module provides two public classes:

CitationTool
    Low-level APA / MLA formatter for individual sources.  The rest of the
    system generally has no need to call this directly.

CitationManager
    Session-scoped tracker used by ResearcherAgent and WriterAgent.  It
    assigns each unique source a sequential index (1, 2, 3 …), deduplicates
    by URL, and formats a numbered APA bibliography on demand.

    Convenience module-level functions (add_source, get_inline_ref,
    format_bibliography, reset_session) delegate to a shared singleton
    so all agents within one query turn share the same registry.

Usage::

    from src.tools.citation_tool import CitationManager, add_source, \\
        get_inline_ref, format_bibliography, reset_session

    # Researcher adds sources while gathering evidence
    idx = add_source({"title": "...", "url": "https://...", "year": 2024})
    ref = get_inline_ref("https://...")   # "[1]"

    # Writer embeds inline refs in the answer, then appends bibliography
    bib = format_bibliography()

    # Orchestrator resets between user turns
    reset_session()
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re


class CitationTool:
    """
    Tool for formatting and managing citations.
    
    Features:
    - APA style formatting (7th edition)
    - Citation tracking and deduplication
    - Bibliography generation
    - Support for papers, articles, and web sources
    """

    def __init__(self, style: str = "apa"):
        """
        Initialize citation tool.

        Args:
            style: Citation style ("apa", "mla", "chicago", etc.)
        """
        self.style = style
        self.citations: List[Dict[str, Any]] = []
        self.citation_counter = 0

    def format_citation(self, source: Dict[str, Any]) -> str:
        """
        Format a source as a citation.

        Args:
            source: Source information dictionary with keys:
                - type: "article", "paper", "webpage", or "book"
                - authors: List of author dicts with "name" key
                - year: Publication year
                - title: Source title
                - venue: Journal/conference name (for papers)
                - url: Web URL
                - doi: DOI identifier (for papers)
                - site_name: Website name (for webpages)

        Returns:
            Formatted citation string in the specified style (default: APA)
        """
        source_type = source.get("type", "article")

        if self.style == "apa":
            return self._format_apa(source, source_type)
        elif self.style == "mla":
            return self._format_mla(source, source_type)
        else:
            return self._format_apa(source, source_type)

    def _format_apa(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Format citation in APA style (7th edition).
        
        Supports:
        - Academic papers/articles
        - Webpages
        - Generic sources
        
        Args:
            source: Source information dictionary
            source_type: Type of source ("article", "paper", "webpage", etc.)
            
        Returns:
            APA-formatted citation string
        """
        if source_type == "article" or source_type == "paper":
            # Journal article or academic paper
            authors = source.get("authors", [])
            year = source.get("year", "n.d.")
            title = source.get("title", "Untitled")
            venue = source.get("venue", "")

            # Format authors
            author_str = self._format_authors_apa(authors)

            # Basic APA format for article
            citation = f"{author_str} ({year}). {title}."
            if venue:
                citation += f" {venue}."

            # Add DOI or URL if available
            doi = source.get("doi")
            url = source.get("url")
            if doi:
                citation += f" https://doi.org/{doi}"
            elif url:
                citation += f" {url}"

            return citation

        elif source_type == "webpage":
            # Web page
            authors = source.get("authors", [])
            year = source.get("year", datetime.now().year)
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            site_name = source.get("site_name", "")

            author_str = self._format_authors_apa(authors) if authors else site_name

            citation = f"{author_str} ({year}). {title}."
            if url:
                citation += f" {url}"

            return citation

        else:
            # Generic fallback
            return f"{source.get('title', 'Unknown')} ({source.get('year', 'n.d.')})"

    def _format_mla(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Format citation in MLA style (9th edition).
        
        Args:
            source: Source information dictionary
            source_type: Type of source
            
        Returns:
            MLA-formatted citation string
        """
        if source_type == "article" or source_type == "paper":
            # Journal article or academic paper
            authors = source.get("authors", [])
            year = source.get("year", "n.d.")
            title = source.get("title", "Untitled")
            venue = source.get("venue", "")
            
            # Format authors for MLA (First Last, and Second Last)
            author_str = self._format_authors_mla(authors)
            
            # MLA format: Author(s). "Article Title." Journal Name, Year.
            citation = f'{author_str}. "{title}."'
            if venue:
                citation += f" {venue},"
            citation += f" {year}."
            
            # Add URL if available
            url = source.get("url")
            if url:
                citation += f" {url}."
            
            return citation
            
        elif source_type == "webpage":
            # Web page
            authors = source.get("authors", [])
            title = source.get("title", "Untitled")
            site_name = source.get("site_name", "")
            year = source.get("year", "n.d.")
            url = source.get("url", "")
            
            author_str = self._format_authors_mla(authors) if authors else site_name
            
            # MLA format for webpage
            citation = f'{author_str}. "{title}."'
            if site_name:
                citation += f" {site_name},"
            citation += f" {year}."
            if url:
                citation += f" {url}."
            
            return citation
            
        else:
            # Generic fallback
            return f'{source.get("title", "Unknown")}. {source.get("year", "n.d.")}.'
    
    def _format_authors_mla(self, authors: List[Dict[str, Any]]) -> str:
        """
        Format author list in MLA style.
        
        MLA format:
        - 1 author: Last, First
        - 2 authors: Last1, First1, and Last2, First2
        - 3+ authors: Last1, First1, et al.
        
        Args:
            authors: List of author dictionaries with "name" key
            
        Returns:
            MLA-formatted author string
        """
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            name = authors[0].get("name", "Unknown")
            return self._format_single_author_mla(name)
        
        elif len(authors) == 2:
            name1 = self._format_single_author_mla(authors[0].get("name", "Unknown"))
            name2 = self._format_single_author_mla(authors[1].get("name", "Unknown"))
            return f"{name1}, and {name2}"
        
        else:
            # 3+ authors - use et al.
            first_author = self._format_single_author_mla(authors[0].get("name", "Unknown"))
            return f"{first_author}, et al."
    
    def _format_single_author_mla(self, name: str) -> str:
        """
        Format a single author name in MLA style (Last, First).
        
        Args:
            name: Author's full name
            
        Returns:
            MLA-formatted name (Last, First)
        """
        if not name or name == "Unknown":
            return "Unknown"
        
        # If already in Last, First format, return as is
        if ',' in name:
            return name
        
        # Split name into parts
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        
        # Assume last part is surname, rest are given names
        surname = parts[-1]
        given_names = " ".join(parts[:-1])
        
        return f"{surname}, {given_names}"

    def _format_authors_apa(self, authors: List[Dict[str, Any]]) -> str:
        """
        Format author list in APA style.
        
        APA 7th edition:
        - 1-2 authors: List all
        - 3-20 authors: List all
        - 21+ authors: First 19, then ..., then last
        
        For simplicity, we use "et al." for 3+ authors
        """
        if not authors:
            return "Unknown Author"

        if len(authors) == 1:
            name = authors[0].get("name", "Unknown")
            return self._format_single_author(name)

        elif len(authors) == 2:
            name1 = self._format_single_author(authors[0].get("name", "Unknown"))
            name2 = self._format_single_author(authors[1].get("name", "Unknown"))
            return f"{name1}, & {name2}"

        else:
            # More than 2 authors - use et al. for brevity
            first_author = self._format_single_author(authors[0].get("name", "Unknown"))
            return f"{first_author}, et al."
    
    def _format_single_author(self, name: str) -> str:
        """
        Format a single author name in APA style (Last, F. M.)
        
        Handles various name formats and extracts last name and initials.
        """
        if not name or name == "Unknown":
            return "Unknown"
        
        # If already in Last, F. format, return as is
        if ',' in name:
            return name
        
        # Split name into parts
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        
        # Assume last part is surname, rest are given names
        surname = parts[-1]
        given_names = parts[:-1]
        
        # Create initials from given names
        initials = ". ".join([n[0].upper() for n in given_names if n]) + "."
        
        return f"{surname}, {initials}"

    def add_citation(self, source: Dict[str, Any]) -> int:
        """
        Add a source to the citation list with deduplication.
        
        Checks if a source with the same title already exists to avoid duplicates.

        Args:
            source: Source information dictionary

        Returns:
            Citation number/index (1-based)
        """
        # Check if already exists (deduplication by title)
        for i, existing in enumerate(self.citations):
            if existing.get("title") == source.get("title"):
                return i + 1

        # Add new citation
        self.citations.append(source)
        self.citation_counter += 1
        return self.citation_counter

    def get_citation_number(self, source: Dict[str, Any]) -> int:
        """Get the citation number for a source."""
        for i, existing in enumerate(self.citations):
            if existing.get("title") == source.get("title"):
                return i + 1
        return 0

    def generate_bibliography(self) -> List[str]:
        """
        Generate formatted bibliography from all citations.
        
        Citations are formatted according to the selected style and sorted
        alphabetically by the first author's last name (APA/MLA standard).

        Returns:
            List of formatted citation strings, sorted alphabetically
        """
        bibliography = []
        for source in self.citations:
            citation = self.format_citation(source)
            bibliography.append(citation)

        # Sort alphabetically (standard for APA and MLA)
        bibliography.sort()

        return bibliography

    def clear_citations(self):
        """Clear all citations."""
        self.citations = []
        self.citation_counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# CitationManager — session-scoped tracker with URL deduplication
# ─────────────────────────────────────────────────────────────────────────────

class CitationManager:
    """
    Session-scoped source registry for the multi-agent research pipeline.

    Design notes
    ────────────
    • Each unique source is assigned a 1-based sequential index that remains
      stable for the lifetime of the session.
    • Deduplication is performed on a normalised URL (trailing slashes and
      common UTM tracking params stripped).  Sources without a URL are
      deduplicated by exact title match as a fallback.
    • Two source shapes are accepted:
        Web result  — {title, url, snippet, published_date, ...}
        Paper       — {title, authors, year, abstract, citation_count,
                       url, venue, ...}
      Both are normalised to CitationTool's internal schema before storage.
    • APA formatting is delegated to CitationTool so formatting logic is
      not duplicated.

    The module-level singleton ``_session_manager`` is the instance shared
    across ResearcherAgent and WriterAgent within a single query turn.
    Use the module-level convenience functions (add_source, get_inline_ref,
    format_bibliography, reset_session) to access it.
    """

    def __init__(self) -> None:
        # Ordered list of normalised source dicts (index = position + 1)
        self._sources: List[Dict[str, Any]] = []
        # Canonical URL → 1-based index (primary dedup key)
        self._url_index: Dict[str, int] = {}
        # Title → 1-based index (fallback dedup when URL is absent)
        self._title_index: Dict[str, int] = {}
        # APA formatter
        self._formatter = CitationTool(style="apa")

    # ── Core public API ───────────────────────────────────────────────────────

    def add_source(self, source: Dict[str, Any]) -> int:
        """
        Register a source and return its 1-based sequential index.

        Deduplication order:
          1. Normalised URL (primary key — most reliable)
          2. Exact title match (fallback for sources without a URL)

        Accepts both web-search result dicts and paper dicts:
          Web:   ``{title, url, snippet, published_date, ...}``
          Paper: ``{title, authors, year, abstract, citation_count, url,
                    venue, ...}``

        Args:
            source: Raw result dict from a search tool (or any dict that
                    contains at least a ``url`` or ``title`` key).

        Returns:
            1-based index assigned to this source.  If the source is a
            duplicate the existing index is returned unchanged.
        """
        url_key = self._normalise_url(source.get("url", ""))
        title_key = (source.get("title") or "").strip().lower()

        # 1. Deduplicate by URL
        if url_key and url_key in self._url_index:
            return self._url_index[url_key]

        # 2. Deduplicate by title (when URL absent or non-unique)
        if title_key and title_key in self._title_index:
            # Back-fill the URL index so future URL lookups also hit the cache
            existing_idx = self._title_index[title_key]
            if url_key:
                self._url_index[url_key] = existing_idx
            return existing_idx

        # New source — normalise and store
        normalised = self._normalise_source(source)
        self._sources.append(normalised)
        idx = len(self._sources)

        if url_key:
            self._url_index[url_key] = idx
        if title_key:
            self._title_index[title_key] = idx

        return idx

    def get_inline_ref(self, url: str) -> str:
        """
        Return the ``[N]`` inline citation key for a URL.

        If the URL has not yet been added via :meth:`add_source` a minimal
        source record is created automatically so the reference remains valid
        in the final answer.

        Args:
            url: The source URL to look up or auto-register.

        Returns:
            Inline citation string, e.g. ``"[1]"`` or ``"[3]"``.
        """
        clean = self._normalise_url(url)
        if clean and clean in self._url_index:
            return f"[{self._url_index[clean]}]"

        # Auto-register so the Writer can cite URLs the Researcher mentioned
        # in prose without calling add_source explicitly
        idx = self.add_source({"url": url, "title": url})
        return f"[{idx}]"

    def format_bibliography(self) -> str:
        """
        Return all tracked sources as a numbered APA-style bibliography.

        Returns an empty string when no sources have been added yet.

        Output format::

            [1] Holstein, K., et al. (2019). Towards Human-Centered AI …
            [2] Santos, C., et al. (2020). Dark Patterns in User Interfaces …

        Returns:
            Multi-line string with one ``[N] APA-citation`` per line,
            in insertion order (i.e., in the order sources were discovered).
        """
        if not self._sources:
            return ""

        lines: List[str] = []
        for i, source in enumerate(self._sources, 1):
            apa = self._formatter.format_citation(source)
            lines.append(f"[{i}] {apa}")

        return "\n".join(lines)

    def get_source(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Return the stored source dict for a 1-based index.

        Args:
            index: 1-based citation index.

        Returns:
            Source dict, or ``None`` if the index is out of range.
        """
        if 1 <= index <= len(self._sources):
            return dict(self._sources[index - 1])
        return None

    def all_sources(self) -> List[Dict[str, Any]]:
        """Return a copy of all tracked sources in insertion order."""
        return [dict(s) for s in self._sources]

    def count(self) -> int:
        """Return the number of unique sources tracked in this session."""
        return len(self._sources)

    def reset(self) -> None:
        """
        Clear all tracked sources.

        Call this between user query sessions so citation indices start
        fresh from [1] for the next query.
        """
        self._sources.clear()
        self._url_index.clear()
        self._title_index.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_url(url: str) -> str:
        """
        Strip trailing slashes, fragments, and common tracking parameters
        so that semantically identical URLs map to the same dedup key.

        Args:
            url: Raw URL string (may be empty or None-coerced).

        Returns:
            Cleaned URL string, or ``""`` if the input is empty.
        """
        if not url:
            return ""
        url = url.strip()
        # Remove fragment
        url = url.split("#")[0]
        # Remove common UTM / tracking query params that don't change identity
        url = re.sub(r"[?&]utm_[^&]*", "", url)
        # Strip dangling ? or &
        url = url.rstrip("?&")
        # Strip trailing slash (but preserve root "https://example.com/")
        if url.endswith("/") and url.count("/") > 2:
            url = url.rstrip("/")
        return url.lower()

    def _normalise_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw search-result dict to CitationTool's internal schema.

        Detection heuristic:
          If the dict has an ``authors`` list (even empty) or a ``venue``
          string, it is treated as an academic paper; otherwise as a webpage.

        Args:
            source: Raw dict from web_search or paper_search.

        Returns:
            Normalised dict compatible with :meth:`CitationTool.format_citation`.
        """
        has_authors = bool(source.get("authors"))
        has_venue = bool(source.get("venue"))
        source_type = "paper" if (has_authors or has_venue) else "webpage"

        # Year: accept int, ISO date string ("2024-05-01"), or bare year string
        raw_year = source.get("year") or source.get("published_date", "")
        if isinstance(raw_year, int):
            year = raw_year
        elif isinstance(raw_year, str) and len(raw_year) >= 4:
            try:
                year = int(raw_year[:4])
            except ValueError:
                year = "n.d."
        else:
            year = "n.d."

        return {
            "type": source_type,
            "title": (source.get("title") or "Untitled").strip(),
            "url": (source.get("url") or "").strip(),
            "year": year,
            "authors": source.get("authors") or [],
            "venue": (source.get("venue") or "").strip(),
            "abstract": (
                source.get("abstract") or source.get("snippet") or ""
            ).strip(),
            "citation_count": source.get("citation_count", 0),
            # site_name used by CitationTool's webpage formatter
            "site_name": (source.get("site_name") or "").strip(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton and convenience functions
# ─────────────────────────────────────────────────────────────────────────────

# Shared instance — both ResearcherAgent and WriterAgent import and call the
# module-level functions below so they operate on the same registry without
# needing to pass the object around explicitly.
_session_manager: CitationManager = CitationManager()


def add_source(source: Dict[str, Any]) -> int:
    """
    Add a source to the session-wide citation registry.

    Thin wrapper around :meth:`CitationManager.add_source` on the shared
    singleton.  Suitable for registering sources as the ResearcherAgent
    discovers them.

    Args:
        source: Raw web or paper result dict.

    Returns:
        1-based citation index.
    """
    return _session_manager.add_source(source)


def get_inline_ref(url: str) -> str:
    """
    Return the ``[N]`` inline reference key for a URL.

    Thin wrapper around :meth:`CitationManager.get_inline_ref` on the
    shared singleton.  The WriterAgent calls this to embed inline
    citations such as ``[1]`` or ``[3]`` in the generated answer.

    Args:
        url: Source URL to look up (auto-registered if new).

    Returns:
        Inline citation string, e.g. ``"[1]"``.
    """
    return _session_manager.get_inline_ref(url)


def format_bibliography() -> str:
    """
    Return the full APA-formatted numbered bibliography for this session.

    Thin wrapper around :meth:`CitationManager.format_bibliography` on the
    shared singleton.  Typically called by the WriterAgent to append a
    ``## References`` section to the final answer.

    Returns:
        Multi-line bibliography string, or ``""`` if no sources were added.
    """
    return _session_manager.format_bibliography()


def reset_session() -> None:
    """
    Clear the session citation registry.

    The orchestrator calls this at the start of each new user query so that
    citation indices reset to [1] and stale sources from previous queries
    do not leak into the new response.
    """
    _session_manager.reset()
