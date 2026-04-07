"""
agents/fact_checker.py

Receives all summaries at once and cross-checks them for
consistency, contradictions, and reliability. This is the only
agent that sees the full picture before the compiler.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
from rich.console import Console

from state import FactCheckResult, ResearchState

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "fact_checker.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()
_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-4-20250514")


def _build_input(state: ResearchState) -> str:
    """
    Flatten all summaries into a single structured string
    that the fact-checker can reason across.
    """
    summaries = state.get("summaries", [])
    sub_tasks = state.get("sub_tasks", [])
    task_lookup = {t["id"]: t for t in sub_tasks}

    lines = []
    for i, summary in enumerate(summaries, 1):
        task_id = summary["task_id"]
        focus = task_lookup.get(task_id, {}).get("focus", summary["query"])

        lines.append(f"--- Summary {i}: {focus} ---")
        lines.append(summary["summary"])
        lines.append("")

        if summary["key_points"]:
            lines.append("Key points:")
            for point in summary["key_points"]:
                lines.append(f"  - {point}")
            lines.append("")

    return "\n".join(lines)


def fact_checker_node(state: ResearchState) -> dict:
    """
    LangGraph node: fact_checker.

    Reads:  state["summaries"], state["sub_tasks"]
    Writes: state["fact_check"], state["current_step"]
    """
    summaries = state.get("summaries", [])

    if not summaries:
        return {
            "fact_check": {
                "claims_checked": 0,
                "verified": [],
                "uncertain": [],
                "verdict": "low",
                "reliability_note": "No summaries were available to fact-check.",
            },
            "current_step": "fact_check_skipped",
            "errors": ["fact_checker: no summaries found in state"],
        }

    console.print(f"\n[bold yellow]Fact Checker Agent[/] cross-checking {len(summaries)} summaries...")

    combined_input = _build_input(state)
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Please fact-check and cross-validate the following "
                        f"research summaries:\n\n{combined_input}"
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

        fact_check: FactCheckResult = {
            "claims_checked": parsed.get("claims_checked", 0),
            "verified": parsed.get("verified", []),
            "uncertain": parsed.get("uncertain", []),
            "verdict": parsed.get("verdict", "medium"),
            "reliability_note": parsed.get("reliability_note", ""),
        }

        console.print(
            f"[yellow]   Done — {fact_check['claims_checked']} claims checked, "
            f"verdict: {fact_check['verdict']}[/]"
        )

        return {
            "fact_check": fact_check,
            "current_step": "fact_checked",
            "errors": [],
        }

    except json.JSONDecodeError as e:
        error = f"fact_checker JSON parse failed: {e}"
        console.print(f"[red]   Error: {error}[/]")
        return {
            "fact_check": {
                "claims_checked": 0,
                "verified": [],
                "uncertain": [],
                "verdict": "low",
                "reliability_note": "Fact-check could not be completed due to a parsing error.",
            },
            "current_step": "error",
            "errors": [error],
        }

    except Exception as e:
        error = f"fact_checker failed: {e}"
        console.print(f"[red]   Error: {error}[/]")
        return {
            "fact_check": {
                "claims_checked": 0,
                "verified": [],
                "uncertain": [],
                "verdict": "low",
                "reliability_note": "Fact-check could not be completed due to an unexpected error.",
            },
            "current_step": "error",
            "errors": [error],
        }
