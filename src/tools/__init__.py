"""
Research Tools Module
Contains tools for web search, paper search, citation extraction, etc.
"""

from .web_search import WebSearchTool
from .paper_search import PaperSearchTool
from .citation_tool import (
    CitationTool,
    CitationManager,
    add_source,
    get_inline_ref,
    format_bibliography,
    reset_session,
)

__all__ = [
    "WebSearchTool",
    "PaperSearchTool",
    "CitationTool",
    "CitationManager",
    "add_source",
    "get_inline_ref",
    "format_bibliography",
    "reset_session",
]
