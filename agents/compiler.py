"""
agents/compiler.py

The final node in the pipeline. Receives all summaries, the
fact-check result, and metadata, then writes the full Markdown
research report.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import anthropic
from rich.console import Console

from state import ResearchState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compiler.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()
_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-4-20250514")
_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))


def _build_input(state: ResearchState) -> str:
    """
    Assemble all pipeline outputs into a single structured prompt
    for the compiler to write the report from.
    """
    summaries = state.get("summaries", [])
    sub_tasks = state.get("sub_tasks", [])
    fact_check = state.get("fact_check")
    metadata = state.get("report_metadata", {})
    search_results = state.get("search_results", [])
    task_lookup = {t["id"]: t for t in sub_tasks}

    # Collect all sources across all search results
    all_sources: list[str] = []
    for result in search_results:
        all_sources.extend(result.get("sources", []))
    all_sources = list(dict.fromkeys(all_sources))  # deduplicate

    lines = []
    lines.append(f"Research title: {metadata.get('title', 'Research Report')}")
    lines.append(f"Original question: {metadata.get('question', '')}")
    lines.append("")

    lines.append("=== SUMMARIES ===")
    for summary in summaries:
        task_id = summary["task_id"]
        focus = task_lookup.get(task_id, {}).get("focus", summary["query"])
        lines.append(f"\n[{task_id}] {focus}")
        lines.append(summary["summary"])
        if summary["key_points"]:
            lines.append("Key points:")
            for point in summary["key_points"]:
                lines.append(f"  - {point}")

    lines.append("\n=== FACT CHECK ===")
    if fact_check:
        lines.append(f"Claims checked: {fact_check['claims_checked']}")
        lines.append(f"Verdict: {fact_check['verdict']}")
        lines.append(f"Reliability note: {fact_check['reliability_note']}")

        if fact_check["verified"]:
            lines.append("Verified claims:")
            for claim in fact_check["verified"]:
                lines.append(f"  - {claim}")

        if fact_check["uncertain"]:
            lines.append("Uncertain claims:")
            for claim in fact_check["uncertain"]:
                lines.append(f"  - {claim}")

    lines.append("\n=== SOURCES ===")
    if all_sources:
        for source in all_sources:
            lines.append(f"  - {source}")
    else:
        lines.append("  No sources collected.")

    return "\n".join(lines)


def _save_report(title: str, content: str) -> Path:
    """
    Save the report to the output directory.
    Returns the path it was saved to.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:50]
    filename = f"{timestamp}_{safe_title}.md"

    output_path = _OUTPUT_DIR / filename
    output_path.write_text(content, encoding="utf-8")

    return output_path


def compiler_node(state: ResearchState) -> dict:
    """
    LangGraph node: compiler.

    Reads:  state["summaries"], state["fact_check"],
            state["search_results"], state["report_metadata"],
            state["sub_tasks"]
    Writes: state["final_report"], state["report_metadata"],
            state["current_step"]
    """
    summaries = state.get("summaries", [])

    if not summaries:
        return {
            "final_report": "# Report Error\n\nNo summaries were available to compile.",
            "report_metadata": state.get("report_metadata", {}),
            "current_step": "compile_skipped",
            "errors": ["compiler: no summaries found in state"],
        }

    console.print("\n[bold green]Compiler Agent[/] writing final report...")

    compiler_input = _build_input(state)
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Please compile a research report from the following "
                        f"findings:\n\n{compiler_input}"
                    ),
                }
            ],
        )

        report = response.content[0].text.strip()

        # Save to disk
        metadata = state.get("report_metadata", {})
        title = metadata.get("title", "research_report")
        output_path = _save_report(title, report)

        # Count words for metadata
        word_count = len(report.split())

        updated_metadata = {
            **metadata,
            "word_count": word_count,
            "output_path": str(output_path),
            "generated_at": datetime.now().isoformat(),
        }

        console.print(f"[green]   Done — {word_count} words, saved to {output_path}[/]")

        return {
            "final_report": report,
            "report_metadata": updated_metadata,
            "current_step": "complete",
            "errors": [],
        }

    except Exception as e:
        error = f"compiler failed: {e}"
        console.print(f"[red]   Error: {error}[/]")
        return {
            "final_report": f"# Report Error\n\nThe report could not be compiled.\n\nError: {e}",
            "report_metadata": state.get("report_metadata", {}),
            "current_step": "error",
            "errors": [error],
        }
