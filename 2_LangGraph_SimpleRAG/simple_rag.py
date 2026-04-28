"""
LangGraph Simple RAG — Researcher Paper Lookup
===============================================
Concepts demonstrated:

  1. Multi-node linear graph   — parse_query → fetch_papers → summarize
  2. State as data pipeline    — each node reads what the previous wrote
  3. Structured intermediate   — Pydantic object used mid-graph (not as final output)
  4. External API in a node    — SerpAPI Google Scholar called inside fetch_papers
  5. RAG pattern               — Retrieve → Augment prompt → Generate

Graph shape:

    START ──► parse_query ──► fetch_papers ──► summarize ──► END

RAG in one sentence:
    Fetch real data first, inject it into the LLM prompt, let the model
    reason over what was retrieved — not over what it memorized.

API used: SerpAPI Google Scholar (https://serpapi.com/google-scholar-api)
    Requires SERPAPI_API_KEY environment variable.
    Free tier: 100 searches/month.
"""
import requests, re
from typing import Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# ── 1. SCHEMAS ────────────────────────────────────────────────────────────────
# ResearcherQuery is a Pydantic model used as structured output from parse_query.
# It carries the identity extracted from the user's free-form input.
# It is an intermediate object — it lives in state but is never shown to the user.

class ResearcherQuery(BaseModel):
    first_name: str = Field(description="First name of the researcher")
    last_name: str = Field(description="Last name / family name of the researcher")
    institution: Optional[str] = Field(
        default=None,
        description="University or institution if explicitly mentioned, otherwise null"
    )


# ── 2. STATE ──────────────────────────────────────────────────────────────────
# State grows as data flows through the graph.
# Think of it as a shared document that each node fills in its section:
#
#   raw_query    → set by the caller before invoke()
#   researcher   → filled by parse_query   (structured identity)
#   author_name  → filled by fetch_papers  (canonical name from the API)
#   papers       → filled by fetch_papers  (list of paper dicts)
#   response     → filled by summarize     (final LLM output)
#
# No node needs to know about nodes before or after it — it only reads the
# fields it needs and writes the field it produces.

class State(TypedDict):
    raw_query: str
    researcher: Optional[ResearcherQuery]
    author_name: str
    papers: list[dict]
    response: str


# ── 3. LLMs ───────────────────────────────────────────────────────────────────
# Two separate LLM instances with different purposes:
#   parse_llm   — deterministic (temperature=0), structured output, small task
#   summary_llm — slight creativity (temperature=0.3), free-form prose response

parse_llm   = ChatOllama(model="qwen2.5:7b", temperature=0).with_structured_output(ResearcherQuery)
summary_llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)


# ── 4. NODES ──────────────────────────────────────────────────────────────────

_SERPAPI_URL = "https://serpapi.com/search"


def parse_query(state: State) -> dict:
    """
    Node 1 — Parse free-form user input into a structured researcher identity.

    The user might type "Yann LeCun", "Geoffrey Hinton from Toronto",
    or "show me papers by Andrej Karpathy". This node extracts the name
    (and optional institution) regardless of phrasing.

    Uses with_structured_output — no manual string parsing required.
    """
    result = parse_llm.invoke([
        HumanMessage(content=f"Extract the researcher's name from: {state['raw_query']}")
    ])
    return {"researcher": result}


def fetch_papers(state: State) -> dict:
    """
    Node 2 — Call SerpAPI Google Scholar to retrieve recent papers.

    Single API call: searches Google Scholar with the parsed researcher name.
    The author: operator narrows results to papers where the name appears as author.
    Year is extracted from the publication_info.summary string via regex.

    This node is a pure data-fetching side effect. It does not call an LLM.
    The fetched papers land in state["papers"] for the next node to use.
    """
    r = state["researcher"]
    author_name = f"{r.first_name} {r.last_name}"

    # author:"Name" restricts Google Scholar results to papers by that author
    query = f'author:"{author_name}"'
    if r.institution:
        query += f" {r.institution}"

    resp = requests.get(
        _SERPAPI_URL,
        params={
            "engine":  "google_scholar",
            "q":       query,
            "num":     10,
            "api_key": "29f4b71e9b8a37e4579f9285f99caf6891463bcc53d142b7126a7aa6c1a28437",
        },
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("organic_results", [])

    if not results:
        return {"author_name": author_name, "papers": []}

    papers = []
    for item in results:
        # Year is embedded in publication_info.summary: "... - Nature, 2023 - ..."
        summary = item.get("publication_info", {}).get("summary", "")
        year_match = re.search(r"\b(19|20)\d{2}\b", summary)
        year = int(year_match.group()) if year_match else None

        papers.append({
            "title":    item.get("title", "Unknown title"),
            "year":     year,
            "abstract": item.get("snippet"),
            "url":      item.get("link"),
        })

    # Sort by year descending — most recent first
    papers.sort(key=lambda p: p["year"] or 0, reverse=True)

    return {"author_name": author_name, "papers": papers}


_SUMMARY_SYSTEM = SystemMessage(content=(
    "You are a research assistant. Given a list of recent academic papers by a researcher, "
    "write a concise summary of: what topics they focus on, what methods or approaches "
    "they use, and what their most notable recent contribution appears to be. "
    "Be factual and grounded in the papers provided — do not speculate beyond them."
))

def summarize(state: State) -> dict:
    """
    Node 3 — Augment an LLM prompt with the retrieved papers and generate a summary.

    This is the 'A' and 'G' in RAG:
      Augment  — build a prompt that contains the fetched papers as context
      Generate — call the LLM, which reasons over retrieved data, not memory

    The LLM never searched the web or its training data for this researcher.
    Everything it says is grounded in what fetch_papers returned.
    """
    if not state["papers"]:
        return {"response": f"No papers found for '{state['author_name']}' on Google Scholar."}

    papers_text = "\n\n".join(
        f"[{p['year'] or 'n/a'}] {p['title']}\n{p['abstract'] or 'No abstract available.'}"
        for p in state["papers"]
    )

    prompt = (
        f"Researcher: {state['author_name']}\n\n"
        f"Their {len(state['papers'])} most recent papers:\n\n"
        f"{papers_text}"
    )

    response = summary_llm.invoke([_SUMMARY_SYSTEM, HumanMessage(content=prompt)])
    return {"response": response.content}


# ── 5. GRAPH ──────────────────────────────────────────────────────────────────
#
#   START ──► parse_query ──► fetch_papers ──► summarize ──► END
#
# Three nodes, four edges. Strictly linear — no branching, no memory.
# Data flows forward through state; nothing flows back.

def build_graph():
    g = StateGraph(State)

    g.add_node("parse_query",  parse_query)
    g.add_node("fetch_papers", fetch_papers)
    g.add_node("summarize",    summarize)

    g.add_edge(START,          "parse_query")
    g.add_edge("parse_query",  "fetch_papers")
    g.add_edge("fetch_papers", "summarize")
    g.add_edge("summarize",    END)

    return g.compile()


# ── 6. MODULE-LEVEL APP ───────────────────────────────────────────────────────

app = build_graph()


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CYAN = "\033[36m"
    GRAY = "\033[90m"
    RST  = "\033[0m"

    print("LangGraph Simple RAG — Researcher Paper Lookup")
    print(f"{GRAY}SerpAPI Google Scholar · set SERPAPI_API_KEY · qwen2.5:7b{RST}\n")
    print("Type a researcher name (e.g. 'Yann LeCun', 'Andrej Karpathy') or /exit.\n")

    while True:
        try:
            user_input = input("Researcher: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye!")
            break

        initial_state: State = {
            "raw_query":   user_input,
            "researcher":  None,
            "author_name": "",
            "papers":      [],
            "response":    "",
        }

        result = app.invoke(initial_state)

        r = result["researcher"]
        parsed_str = f"{r.first_name} {r.last_name}" + (f" @ {r.institution}" if r.institution else "")
        print(f"\n{GRAY}Parsed query : {parsed_str}{RST}")
        print(f"{GRAY}API matched  : {result['author_name']} — {len(result['papers'])} papers{RST}\n")

        for p in result["papers"]:
            year = p["year"] or "n/a"
            print(f"{GRAY}  [{year}] {p['title']}{RST}")

        print(f"\n{CYAN}Summary:{RST}")
        print(result["response"])
        print("\n" + "─" * 55 + "\n")
