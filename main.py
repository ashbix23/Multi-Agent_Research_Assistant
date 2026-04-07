"""
main.py

CLI entry point for the multi-agent research assistant.

Usage:
    python main.py "What is the current state of nuclear fusion energy?"
    python main.py  # will prompt you interactively
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

from graph import research_graph
from state import ResearchState

console = Console()


def validate_environment() -> bool:
    """
    Check that required environment variables are set before
    attempting any API calls.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]Error:[/] ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key."
        )
        return False
    return True


def get_question(argv: list[str]) -> str:
    """
    Get the research question from CLI args or interactive prompt.
    """
    if len(argv) > 1:
        return " ".join(argv[1:])

    console.print(
        Panel(
            "[bold]Multi-Agent Research Assistant[/]\n"
            "[dim]Powered by LangGraph + Anthropic[/]",
            expand=False,
        )
    )
    console.print()
    question = console.input("[bold]Research question:[/] ").strip()

    if not question:
        console.print("[red]No question provided. Exiting.[/]")
        sys.exit(1)

    return question


def print_header(question: str) -> None:
    console.print()
    console.print(Rule("[bold]Research Assistant[/]"))
    console.print(f"[dim]Question:[/] {question}")
    console.print(f"[dim]Started: [/]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(Rule())


def print_report(state: ResearchState) -> None:
    """
    Render the final report and metadata to the terminal.
    """
    console.print()
    console.print(Rule("[bold green]Report Complete[/]"))

    metadata = state.get("report_metadata", {})
    final_report = state.get("final_report", "")

    if metadata:
        console.print(f"[dim]Words:[/]      {metadata.get('word_count', 'N/A')}")
        console.print(f"[dim]Saved to:[/]   {metadata.get('output_path', 'N/A')}")
        console.print(f"[dim]Generated:[/]  {metadata.get('generated_at', 'N/A')}")

    fact_check = state.get("fact_check")
    if fact_check:
        verdict = fact_check.get("verdict", "unknown")
        verdict_color = {"high": "green", "medium": "yellow", "low": "red"}.get(verdict, "white")
        console.print(f"[dim]Confidence:[/] [{verdict_color}]{verdict.upper()}[/]")

    console.print()
    console.print(Markdown(final_report))

    errors = state.get("errors", [])
    if errors:
        console.print()
        console.print(Rule("[bold red]Errors[/]"))
        for error in errors:
            console.print(f"[red]  - {error}[/]")


def run(question: str) -> ResearchState:
    """
    Execute the full research pipeline for a given question.
    Returns the final state so callers can inspect or test it.
    """
    print_header(question)

    initial_state: ResearchState = {
        "research_question": question,
        "sub_tasks": [],
        "search_results": [],
        "summaries": [],
        "fact_check": None,
        "final_report": "",
        "report_metadata": {},
        "errors": [],
        "current_step": "start",
    }

    final_state = research_graph.invoke(initial_state)

    print_report(final_state)

    return final_state


def main() -> None:
    if not validate_environment():
        sys.exit(1)

    question = get_question(sys.argv)
    final_state = run(question)

    # Exit with error code if the pipeline hit critical failures
    if final_state.get("current_step") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
