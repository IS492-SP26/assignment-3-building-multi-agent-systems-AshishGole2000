"""
Citation Tool
Formats citations and manages citation lists.

This tool provides citation formatting in multiple styles (primarily APA)
and manages a bibliography for research outputs.
"""

from typing import Dict, Any, List
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
