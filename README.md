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

## VS Code extension

Install the **LangGraph** extension from the VS Code marketplace. Each folder contains a `langgraph.json` that tells it where to find the compiled graph. Once installed, you can visualize the graph structure without running the code.

---

## Official documentation

- **LangGraph** — [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **LangChain** — [python.langchain.com](https://python.langchain.com/)
- **Ollama** — [ollama.com](https://ollama.com/)
