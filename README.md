# Multi-Agent Research Assistant

A LangGraph-based research pipeline that takes any question, breaks it into focused sub-tasks, runs parallel web searches, summarizes findings, fact-checks claims, and produces a structured Markdown report.

Built with LangGraph, the Anthropic API, and Python.

---

## Architecture

```
research_question
       |
  orchestrator        Decomposes the question into 3 focused sub-tasks
       |
  web_search          Runs a live web search for each sub-task
       |
  summarizer          Condenses raw results into summaries + key points
       |
  fact_checker        Cross-checks all summaries for consistency
       |
  compiler            Writes the final structured Markdown report
       |
  output/*.md         Report saved to disk
```

Each stage is a LangGraph node that reads from and writes to a shared `ResearchState` TypedDict. If any stage fails, the graph routes directly to the compiler which produces an error report rather than crashing.

---

## Project Structure

```
research_assistant/
│
├── main.py                  # CLI entry point
├── graph.py                 # LangGraph StateGraph definition
├── state.py                 # Shared TypedDict state schema
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py      # Decomposes question into 3 sub-tasks
│   ├── web_search.py        # Calls Anthropic API with web_search tool
│   ├── summarizer.py        # Condenses raw results per sub-task
│   ├── fact_checker.py      # Cross-checks claims across all summaries
│   └── compiler.py          # Assembles the final Markdown report
│
├── tools/
│   ├── __init__.py
│   └── search_tool.py       # Web search tool config and helpers
│
├── prompts/
│   ├── orchestrator.txt
│   ├── summarizer.txt
│   ├── fact_checker.txt
│   └── compiler.txt
│
├── output/                  # Generated reports saved here
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

**1. Clone and enter the project directory:**

```bash
git clone https://github.com/ashbix23/research_assistant
cd research_assistant
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables:**

```bash
cp .env.example .env
```

Open `.env` and add your Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
RESEARCH_MODEL=claude-sonnet-4-20250514
MAX_SEARCH_RESULTS=5
OUTPUT_DIR=output
```

Get your API key from [console.anthropic.com](https://console.anthropic.com).

---

## Usage

**Pass a question directly as a CLI argument:**

```bash
python main.py "What is the current state of nuclear fusion energy?"
```

**Or run interactively:**

```bash
python main.py
# Research question: <you type here>
```

**Import and run programmatically:**

```python
from main import run

state = run("What are the geopolitical implications of Arctic ice melt?")
print(state["final_report"])
```

---

## Output

Each run produces a Markdown report saved to the `output/` directory:

```
output/
└── 20250406_143022_Nuclear_Fusion_Energy.md
```

The report follows this structure:

```
# Title

## Executive Summary
## Findings
   ### Sub-task 1 focus
   ### Sub-task 2 focus
   ### Sub-task 3 focus
## Key Takeaways
## Reliability Assessment
## Sources
```

The terminal also displays:
- Live progress from each agent as it runs
- Word count and file path on completion
- Confidence verdict (HIGH / MEDIUM / LOW) from the fact-checker
- Any errors encountered during the pipeline

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key |
| `RESEARCH_MODEL` | `claude-sonnet-4-20250514` | Model used by all agents |
| `MAX_SEARCH_RESULTS` | `5` | Max web search calls per sub-task |
| `OUTPUT_DIR` | `output` | Directory where reports are saved |

---

## How Each Agent Works

### Orchestrator
Sends the raw research question to Claude and receives back a JSON object containing a title and 3 sub-tasks. Each sub-task has a specific search query and a focus description. This decomposition drives the rest of the pipeline.

### Web Search Agent
Calls the Anthropic API with the built-in `web_search_20250305` tool enabled. For each sub-task, it runs a targeted search and collects the raw response text and any source URLs. Searches run sequentially to avoid rate limits.

### Summarizer Agent
Takes each raw search result and sends it to Claude with instructions to return a structured JSON summary (150–250 words) plus 3–5 key points. Preserves specific facts, figures, and dates rather than paraphrasing them away.

### Fact Checker Agent
Receives all 3 summaries at once and cross-checks them for consistency. Returns a verdict of `high`, `medium`, or `low` confidence along with lists of verified and uncertain claims. This is the only agent that sees all summaries simultaneously.

### Compiler Agent
Receives all summaries, the fact-check result, and all collected sources, then writes the full Markdown report. Uses `max_tokens=4096` since it is producing a full document rather than compact JSON. Saves the report to disk with a timestamped filename.

---

---

## Requirements

- Python 3.11+
- An Anthropic API key with access to Claude Sonnet 4
- Internet access for the web search tool
