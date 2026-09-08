"""Resilient web-search tools used by the multi-tool agent."""

from __future__ import annotations

import json

from ddgs import DDGS


def safe_duckduckgo_search(query: str) -> str:
    """Search the web without allowing provider failures to stop the agent."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return (
                "WEB_SEARCH_ERROR: No results were found. Tell the user briefly, "
                "then retry once with a shorter or differently worded query. If "
                "the retry also fails, continue with the other available tools."
            )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as error:
        return (
            f"WEB_SEARCH_ERROR: {type(error).__name__}: {error}. Tell the user "
            "briefly, then retry once with a shorter or differently worded query. "
            "If the retry also fails, continue with the other available tools."
        )
