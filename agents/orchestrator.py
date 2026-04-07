"""
agents/orchestrator.py

The first node in the graph. Takes the raw research question and
produces 3 focused sub-tasks that drive the rest of the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
from rich.console import Console

from state import ResearchState, SubTask

console = Console()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "orchestrator.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()
_MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-4-20250514")


def orchestrator_node(state: ResearchState) -> dict:
    """
    LangGraph node: orchestrator.

    Reads:  state["research_question"]
    Writes: state["sub_tasks"], state["current_step"], state["report_metadata"]
    """
    question = state["research_question"]
    console.print(f"\n[bold purple] Orchestrator[/] decomposing: [italic]{question}[/]")

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Research question: {question}"
                }
            ]
        )

        raw = response.content[0].text.strip()

        # Strip accidental markdown fences the model might add
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        parsed = json.loads(raw.strip())

        sub_tasks: list[SubTask] = [
            {
                "id": t["id"],
                "query": t["query"],
                "focus": t["focus"],
                "status": "pending",
            }
            for t in parsed["sub_tasks"]
        ]

        console.print(f"[purple]   ✓ Decomposed into {len(sub_tasks)} sub-tasks:[/]")
        for task in sub_tasks:
            console.print(f"[purple]     • [{task['id']}] {task['focus']}[/]")

        return {
            "sub_tasks": sub_tasks,
            "current_step": "orchestrated",
            "report_metadata": {
                "title": parsed.get("title", question),
                "question": question,
            },
            "errors": [],
        }

    except json.JSONDecodeError as e:
        error = f"Orchestrator failed to parse JSON: {e}"
        console.print(f"[red]   ✗ {error}[/]")
        return {
            "sub_tasks": [],
            "current_step": "error",
            "report_metadata": {},
            "errors": [error],
        }

    except Exception as e:
        error = f"Orchestrator error: {e}"
        console.print(f"[red]   ✗ {error}[/]")
        return {
            "sub_tasks": [],
            "current_step": "error",
            "report_metadata": {},
            "errors": [error],
        }
