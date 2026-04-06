"""
state.py — Shared state schema for the research assistant graph.

Every agent node reads from and writes to this TypedDict.
LangGraph passes it through the graph as an immutable snapshot,
merging updates at each node via the defined reducers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class SubTask(TypedDict):
    """A single research sub-task produced by the orchestrator."""
    id: str                  # e.g. "task_1"
    query: str               # The specific search query
    focus: str               # What aspect this sub-task covers
    status: str              # "pending" | "searching" | "summarized" | "done"


class SearchResult(TypedDict):
    """Raw result from the web search agent for one sub-task."""
    task_id: str
    query: str
    raw_content: str         # Full text returned by the search tool
    sources: list[str]       # URLs or source references extracted


class Summary(TypedDict):
    """Condensed output from the summarizer agent."""
    task_id: str
    query: str
    summary: str             # Condensed findings
    key_points: list[str]    # Bullet-point takeaways


class FactCheckResult(TypedDict):
    """Output from the fact-checker agent."""
    claims_checked: int
    verified: list[str]      # Claims that appear well-supported
    uncertain: list[str]     # Claims that conflict or lack sources
    verdict: str             # "high" | "medium" | "low"
    reliability_note: str    # One sentence explaining the verdict


class ResearchState(TypedDict):
    """
    Central state object threaded through every graph node.

    Reducers:
    - sub_tasks, search_results, summaries use operator.add so each
      agent can append without overwriting siblings running in parallel.
    - All other fields are last-write-wins (default TypedDict behaviour).
    """

    # ── Input ──────────────────────────────────────────────────────────
    research_question: str

    # ── Orchestrator output ───────────────────────────────────────────
    sub_tasks: Annotated[list[SubTask], operator.add]

    # ── Web search output ─────────────────────────────────────────────
    search_results: Annotated[list[SearchResult], operator.add]

    # ── Summarizer output ─────────────────────────────────────────────
    summaries: Annotated[list[Summary], operator.add]

    # ── Fact-checker output ───────────────────────────────────────────
    fact_check: FactCheckResult | None

    # ── Compiler output ───────────────────────────────────────────────
    final_report: str
    report_metadata: dict[str, Any]

    # ─ Control / observability ──────────────────────────────────────
    errors: Annotated[list[str], operator.add]
    current_step: str
