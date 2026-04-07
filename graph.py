"""
graph.py

Defines and compiles the LangGraph StateGraph that wires all
agent nodes together into a runnable research pipeline.

Graph topology:
    orchestrator
         |
    web_search
         |
    summarizer
         |
    fact_checker
         |
    compiler
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.compiler import compiler_node
from agents.fact_checker import fact_checker_node
from agents.orchestrator import orchestrator_node
from agents.summarizer import summarizer_node
from agents.web_search import web_search_node
from state import ResearchState


def should_continue_after_orchestrator(state: ResearchState) -> str:
    """
    Conditional edge after the orchestrator.
    If no sub-tasks were produced (e.g. a parse error), route
    straight to the compiler which will emit an error report.
    """
    if not state.get("sub_tasks"):
        return "compiler"
    return "web_search"


def should_continue_after_search(state: ResearchState) -> str:
    """
    Conditional edge after the web search agent.
    If no results came back, skip summarizer and fact-checker
    and go straight to compiler.
    """
    results = state.get("search_results", [])
    if not any(r["raw_content"] for r in results):
        return "compiler"
    return "summarizer"


def should_continue_after_summarizer(state: ResearchState) -> str:
    """
    Conditional edge after the summarizer.
    If no summaries were produced, skip fact-checker.
    """
    if not state.get("summaries"):
        return "compiler"
    return "fact_checker"


def build_graph() -> StateGraph:
    """
    Construct and compile the research assistant graph.

    Returns a compiled graph ready to invoke or stream.
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("compiler", compiler_node)

    # ── Entry point ───────────────────────────────────────────────────
    graph.add_edge(START, "orchestrator")

    # ── Conditional edges ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {
            "web_search": "web_search",
            "compiler": "compiler",
        },
    )

    graph.add_conditional_edges(
        "web_search",
        should_continue_after_search,
        {
            "summarizer": "summarizer",
            "compiler": "compiler",
        },
    )

    graph.add_conditional_edges(
        "summarizer",
        should_continue_after_summarizer,
        {
            "fact_checker": "fact_checker",
            "compiler": "compiler",
        },
    )

    # ── Unconditional edges ───────────────────────────────────────────
    graph.add_edge("fact_checker", "compiler")
    graph.add_edge("compiler", END)

    return graph.compile()


# Module-level compiled graph instance.
# Import this directly in main.py rather than calling build_graph() each time.
research_graph = build_graph()
