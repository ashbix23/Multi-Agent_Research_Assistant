"""
agents/web_search.py

Runs a web search for each sub-task using the Anthropic API's
built-in web_search tool. Produces raw search results that the
summarizer will condense in the next stage.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from rich.console import Console

from state import ResearchState, SearchResult

console = Console()

_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-4-20250514")
_MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))


def _extract_sources(content_blocks: list) -> list[str]:
    """
    Pull source URLs out of the tool result blocks returned
    by the web search tool.
    """
    sources = []
    for block in content_blocks:
        if block.type == "tool_result":
            for item in getattr(block, "content", []):
                if hasattr(item, "source") and item.source:
                    sources.append(item.source)
    return list(dict.fromkeys(sources))  # deduplicate, preserve order


def _run_search_for_task(client: anthropic.Anthropic, query: str, focus: str) -> tuple[str, list[str]]:
    """
    Call the Anthropic API with the web_search tool enabled for a
    single query. Returns (raw_text, sources).
    """
    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": _MAX_RESULTS,
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Search for information about the following topic and return "
                    f"a thorough summary of what you find.\n\n"
                    f"Topic focus: {focus}\n"
                    f"Search query: {query}"
                ),
            }
        ],
    )

    # Collect all text across content blocks
    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text += block.text + "\n"

    sources = _extract_sources(response.content)

    return raw_text.strip(), sources


def web_search_node(state: ResearchState) -> dict:
    """
    LangGraph node: web_search.

    Reads:  state["sub_tasks"]
    Writes: state["search_results"], state["current_step"]

    Runs searches sequentially to avoid hammering the API.
    Each sub-task gets its own search call.
    """
    sub_tasks = state.get("sub_tasks", [])

    if not sub_tasks:
        return {
            "search_results": [],
            "current_step": "search_skipped",
            "errors": ["web_search: no sub-tasks found in state"],
        }

    client = anthropic.Anthropic()
    search_results: list[SearchResult] = []

    console.print(f"\n[bold teal]Web Search Agent[/] running {len(sub_tasks)} searches...")

    for task in sub_tasks:
        console.print(f"[teal]   Searching [{task['id']}]: {task['query']}[/]")

        try:
            raw_content, sources = _run_search_for_task(
                client=client,
                query=task["query"],
                focus=task["focus"],
            )

            search_results.append({
                "task_id": task["id"],
                "query": task["query"],
                "raw_content": raw_content,
                "sources": sources,
            })

            console.print(f"[teal]   Done [{task['id']}] — {len(sources)} sources found[/]")

        except Exception as e:
            error = f"web_search [{task['id']}] failed: {e}"
            console.print(f"[red]   Error: {error}[/]")
            # Still append a result so downstream agents have something to work with
            search_results.append({
                "task_id": task["id"],
                "query": task["query"],
                "raw_content": "",
                "sources": [],
            })

    return {
        "search_results": search_results,
        "current_step": "searched",
        "errors": [],
    }
