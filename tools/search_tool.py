"""
tools/search_tool.py

Web search tool configuration and helpers.

The Anthropic API has a built-in web_search tool that handles
the actual searching. This module centralises the tool definition
so any agent that needs search can import it from one place rather
than duplicating the config dict.
"""

from __future__ import annotations

import os


MAX_USES = int(os.getenv("MAX_SEARCH_RESULTS", "5"))


# The tool definition dict passed directly to the Anthropic API.
# Keeping it here means if the tool type string or schema ever
# changes, there is exactly one place to update it.
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": MAX_USES,
}


def get_search_tool(max_uses: int | None = None) -> dict:
    """
    Return the web search tool definition.

    Optionally override max_uses for callers that need
    a different cap than the global default.
    """
    if max_uses is not None:
        return {**WEB_SEARCH_TOOL, "max_uses": max_uses}
    return WEB_SEARCH_TOOL


def extract_sources_from_response(content_blocks: list) -> list[str]:
    """
    Walk the content blocks returned by an Anthropic API response
    and extract any source URLs attached by the web search tool.

    Deduplicates and preserves order.
    """
    sources = []

    for block in content_blocks:
        # Tool result blocks carry the search metadata
        if getattr(block, "type", None) == "tool_result":
            for item in getattr(block, "content", []):
                source = getattr(item, "source", None)
                if source and isinstance(source, str):
                    sources.append(source)

        # Some API versions embed citations directly on text blocks
        citations = getattr(block, "citations", None)
        if citations:
            for citation in citations:
                url = getattr(citation, "url", None)
                if url and isinstance(url, str):
                    sources.append(url)

    return list(dict.fromkeys(sources))


def extract_text_from_response(content_blocks: list) -> str:
    """
    Concatenate all text content from an Anthropic API response,
    skipping tool use and tool result blocks.
    """
    parts = []

    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text.strip():
                parts.append(text.strip())

    return "\n\n".join(parts)
