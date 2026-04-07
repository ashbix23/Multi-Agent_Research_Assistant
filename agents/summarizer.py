"""
agents/summarizer.py

Takes the raw search results and condenses each one into a
structured summary with key points. Runs once per search result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
from rich.console import Console

from state import ResearchState, Summary

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarizer.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()
_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-4-20250514")


def _summarize_one(
    client: anthropic.Anthropic,
    task_id: str,
    query: str,
    focus: str,
    raw_content: str,
) -> Summary:
    """
    Call the API to summarize a single search result.
    Returns a Summary TypedDict.
    """
    response = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Sub-task focus: {focus}\n"
                    f"Search query: {query}\n\n"
                    f"Raw search results:\n{raw_content}"
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    parsed = json.loads(raw.strip())

    return {
        "task_id": task_id,
        "query": query,
        "summary": parsed["summary"],
        "key_points": parsed.get("key_points", []),
    }


def summarizer_node(state: ResearchState) -> dict:
    """
    LangGraph node: summarizer.

    Reads:  state["search_results"], state["sub_tasks"]
    Writes: state["summaries"], state["current_step"]

    Matches each search result back to its sub-task to recover
    the focus description, then summarizes one by one.
    """
    search_results = state.get("search_results", [])
    sub_tasks = state.get("sub_tasks", [])

    if not search_results:
        return {
            "summaries": [],
            "current_step": "summarize_skipped",
            "errors": ["summarizer: no search results found in state"],
        }

    # Build a lookup so we can retrieve the focus for each task_id
    task_lookup = {t["id"]: t for t in sub_tasks}

    client = anthropic.Anthropic()
    summaries: list[Summary] = []

    console.print(f"\n[bold blue]Summarizer Agent[/] processing {len(search_results)} results...")

    for result in search_results:
        task_id = result["task_id"]
        task = task_lookup.get(task_id, {})
        focus = task.get("focus", result["query"])

        console.print(f"[blue]   Summarizing [{task_id}]: {focus}[/]")

        if not result["raw_content"]:
            console.print(f"[yellow]   Warning: [{task_id}] has no content to summarize, skipping[/]")
            summaries.append({
                "task_id": task_id,
                "query": result["query"],
                "summary": "No content was retrieved for this sub-task.",
                "key_points": [],
            })
            continue

        try:
            summary = _summarize_one(
                client=client,
                task_id=task_id,
                query=result["query"],
                focus=focus,
                raw_content=result["raw_content"],
            )
            summaries.append(summary)
            console.print(f"[blue]   Done [{task_id}] — {len(summary['key_points'])} key points extracted[/]")

        except json.JSONDecodeError as e:
            error = f"summarizer [{task_id}] JSON parse failed: {e}"
            console.print(f"[red]   Error: {error}[/]")
            summaries.append({
                "task_id": task_id,
                "query": result["query"],
                "summary": "Summary could not be generated due to a parsing error.",
                "key_points": [],
            })

        except Exception as e:
            error = f"summarizer [{task_id}] failed: {e}"
            console.print(f"[red]   Error: {error}[/]")
            summaries.append({
                "task_id": task_id,
                "query": result["query"],
                "summary": "Summary could not be generated due to an unexpected error.",
                "key_points": [],
            })

    return {
        "summaries": summaries,
        "current_step": "summarized",
        "errors": [],
    }
