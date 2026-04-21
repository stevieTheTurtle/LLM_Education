# LLM Education

A hands-on learning path for building LLM-powered applications with LangGraph. Each folder is a self-contained example with its own guide and runnable code, ordered from simplest to most complex.

---

## Prerequisites

Before touching any code here, make sure you have:

- **Python 3.10 or newer** — [python.org/downloads](https://www.python.org/downloads/)
- **A virtual environment** — see setup instructions below
- **Ollama running locally** — the examples use local models via `ChatOllama`

If you are new to Python or have never set up a project environment before, read the two sections below first. They will save you a lot of confusion later.

---

## Python environment setup

### What is a virtual environment and why you need one

When you install a Python package (`pip install langgraph`), it gets placed in a global folder shared by every project on your machine. This causes problems: project A needs `langchain==0.1` while project B needs `langchain==0.3`, and they cannot both be satisfied globally.

A virtual environment (`.venv`) is an isolated copy of Python and its packages scoped to a single folder. Each project gets its own `.venv`, and they never interfere with each other.

### Creating and activating a virtual environment

Official documentation: [docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html)

```bash
# Create the environment inside the project folder
python -m venv .venv

# Activate it — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate it — Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate it — macOS / Linux
source .venv/bin/activate
```

Once activated, your terminal prompt will show `(.venv)`. Every `pip install` from that point on installs into the isolated environment, not globally.

To deactivate when you are done:

```bash
deactivate
```

### Installing dependencies

After activating the environment, install the packages used across this repo:

```bash
pip install langgraph langchain langchain-ollama
```

---

## Python basics — where to learn

If you are new to Python, these are the official resources:

- **Python tutorial (official)** — [docs.python.org/3/tutorial](https://docs.python.org/3/tutorial/) — covers variables, functions, classes, modules, and the standard library step by step
- **pip user guide** — [pip.pypa.io/en/stable/user_guide](https://pip.pypa.io/en/stable/user_guide/) — how to install, upgrade, and uninstall packages
- **venv documentation** — [docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) — full reference for virtual environments

---

## Repo structure

```
LLM_Education/
├── LangGraph_Basics/        ← start here
│   ├── guide.md             — LangGraph fundamentals, chat templates, sampling params
│   ├── chatbot.py           — single-node chatbot with short-term memory
│   └── langgraph.json       — VS Code extension config
│
└── LangGraph_Workflow/      ← once you understand the basics
    ├── guide.md             — conditional edges, fan-out/fan-in, structured output
    ├── workflow_example.py  — city research agent with parallel API nodes
    └── langgraph.json       — VS Code extension config
```

---

## Learning path

### 1. LangGraph_Basics

Start here. The example is a chatbot with short-term memory — the minimum possible LangGraph application. The guide covers:

- What LangGraph is and why graphs beat plain function chains
- The three building blocks: State, Nodes, Graph
- How `add_messages` and reducers work
- Chat message roles: `SystemMessage`, `HumanMessage`, `AIMessage`
- LLM sampling parameters: `temperature`, `top_k`, `top_p`

Run it:

```bash
cd LangGraph_Basics
python chatbot.py
```

### 2. LangGraph_Workflow

Once the basics are clear, move here. The example is a city research agent that fetches live data from public APIs. The guide covers:

- Designing a graph before writing code
- Pydantic schemas for structured LLM output
- Conditional edges and runtime branching
- Fan-out (parallel nodes) and fan-in (synchronization)
- API-grounded generation — facts from live data, not model memory

Run it:

```bash
cd LangGraph_Workflow
python workflow_example.py
```

---

## VS Code extension — VizLang Studio

Install **VizLang Studio** (`vkfolio.vizlangstudio`) from the VS Code marketplace. It is a standalone LangGraph development environment that lets you visualize, run, and debug your graphs without leaving the editor.

Each folder in this repo already contains a `langgraph.json` that tells the extension where to find the compiled graph:

```json
{
  "dependencies": ["."],
  "graphs": {
    "chatbot": "./chatbot.py:app"
  }
}
```

The extension imports the Python module and looks up the `app` variable at the path specified. No extra configuration is needed beyond what is already in place.

### Features

**Graph Visualization**
Renders the graph as a diagram — nodes as boxes, edges as arrows, conditional branches as labeled forks. This is static: it reflects the structure of `build_graph()` without running anything. Useful for confirming your edges are wired the way you think they are.

**Chat**
A built-in input panel that lets you invoke the graph directly from VS Code without opening a terminal. You type a message, it calls `app.invoke()`, and you see the output inline. Equivalent to running `python chatbot.py` but without leaving the editor.

**Tracing**
As the graph runs, the extension highlights the currently executing node and logs the state after each step. You can see exactly what each node received from state and what it wrote back — much easier than reading `print()` output scattered across the terminal.

**HITL — Human in the Loop**
A pattern where the graph **pauses mid-execution** and waits for a human to approve, modify, or reject an intermediate result before continuing. In LangGraph this is implemented with `interrupt()` inside a node. VizLang surfaces it as a prompt in the UI: the graph halts, shows you the current state, and lets you confirm or inject new values before it resumes. Useful when an LLM decision is high-stakes and needs a review step before proceeding.

**Time Travel**
After a run completes, you can scrub backwards through execution history — rewind to any previous state snapshot and re-run the graph from that point with different inputs or after a code fix. This requires LangGraph's checkpointer to be enabled, which saves a state snapshot at each node. Without it, time travel is not available.

### What the examples support out of the box

The examples in this repo support **Visualization**, **Chat**, and **Tracing** with no changes. HITL and Time Travel require adding a checkpointer to `build_graph()`:

```python
from langgraph.checkpoint.memory import MemorySaver

def build_graph():
    g = StateGraph(State)
    # ... nodes and edges ...
    return g.compile(checkpointer=MemorySaver())
```

This is a more advanced topic — not needed to follow the learning path here.

---

## Official documentation

- **LangGraph** — [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **LangChain** — [python.langchain.com](https://python.langchain.com/)
- **Ollama** — [ollama.com](https://ollama.com/)
