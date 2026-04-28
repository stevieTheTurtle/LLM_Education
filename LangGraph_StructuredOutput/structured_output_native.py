"""
LangGraph Structured Output — Native (Tool-Calling Model)
=========================================================
Concepts demonstrated:

  1. Pydantic schema         — defines the exact shape of the LLM output
  2. with_structured_output  — binds the schema to the LLM
  3. No format instructions  — model's tool-calling mechanism handles JSON
  4. Graph with typed state  — state carries both raw input and parsed result

Use case: extract structured data from a free-text movie review.

Graph shape:

    START ──► extract ──► END

Why no system prompt for JSON format?
    Models with native tool-calling (qwen2.5, llama3.1, mistral-nemo, etc.)
    expose a "tools" API that lets the caller pass a JSON schema. The model
    fills it directly — no text-level formatting required. LangChain's
    with_structured_output uses this mechanism automatically when the model
    supports it.
"""

from pprint import pprint
from typing import Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# ── 1. OUTPUT SCHEMA ──────────────────────────────────────────────────────────
# Pydantic BaseModel defines the structure we want back from the LLM.
# Field(description=...) is passed to the model as part of the schema,
# giving it per-field context without polluting the system prompt.

class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie being reviewed")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    score: float = Field(description="Numeric score from 0.0 to 10.0")
    genre: str = Field(description="Primary genre of the movie")
    key_points: list[str] = Field(description="2 to 4 main points from the review")
    recommended: bool = Field(description="True if the reviewer recommends the movie")


# ── 2. STATE ──────────────────────────────────────────────────────────────────
# No add_messages reducer here — we are not building conversation history.
# Nodes overwrite state fields directly.
#
# review_text : raw string input from the user
# parsed      : the MovieReview object produced after extraction

class State(TypedDict):
    review_text: str
    parsed: Optional[MovieReview]


# ── 3. LLM — with_structured_output ──────────────────────────────────────────
# qwen2.5:7b has strong native tool-calling support.
# temperature=0 ensures deterministic, well-formed JSON output.
#
# with_structured_output(MovieReview) does two things:
#   a) Sends the Pydantic schema to the model as a tool definition
#   b) Parses the raw response back into a MovieReview instance
#
# The node receives a proper Python object, not a string to parse manually.

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)


# ── 4. NODE ───────────────────────────────────────────────────────────────────
# No system prompt instructing the model how to format JSON.
# The tool-calling mechanism injected by with_structured_output handles that.
# The prompt is just about the task — not about output format.

def extract(state: State) -> dict:
    prompt = f"Extract structured information from this movie review:\n\n{state['review_text']}"
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    return {"parsed": result}


# ── 5. GRAPH ──────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)

    g.add_node("extract", extract)

    g.add_edge(START, "extract")
    g.add_edge("extract", END)

    return g.compile()


# ── 6. MODULE-LEVEL APP ───────────────────────────────────────────────────────
# Must live outside __main__ so the VS Code LangGraph extension can find it.
# langgraph.json points to "structured_output_native.py:app".

app = build_graph()


# ── 7. DEMO LOOP ──────────────────────────────────────────────────────────────


CYAN = "\033[36m"
GRAY = "\033[90m"
SOUT  = "\033[32m"
RST  = "\033[0m"

MOVIE_REVIEW = """Inception is a mind-bending sci-fi thriller. Nolan crafts an intricate dream-within-a-dream story that keeps you on edge the whole time. The visuals are stunning and the score is unforgettable. I'd give it a 9 out of 10 — absolutely watch it."""

print("LangGraph Structured Output — Native Model (qwen2.5:7b)")
print(f"{GRAY}No JSON format instructions — model uses tool-calling natively{RST}\n")

print(f"{GRAY}{MOVIE_REVIEW}\n")

result = app.invoke({"review_text": MOVIE_REVIEW, "parsed": None})

print(f"\n{GRAY}RAW LLM Output:")
pprint(f"{result["parsed"]}")
parsed: MovieReview = result["parsed"]

print(f"\n{SOUT}Parsed Structured Output:")
pprint(f"{parsed}") #This is an actual instance of the Pydantic class MovieReview defined above

print(f"\n{CYAN}Extracted:{RST}")
print(f"  Title:       {parsed.title}")
print(f"  Sentiment:   {parsed.sentiment}")
print(f"  Score:       {parsed.score}/10")
print(f"  Genre:       {parsed.genre}")
print(f"  Recommended: {parsed.recommended}")
print(f"  Key points:")
for point in parsed.key_points:
    print(f"    • {point}")
print("\n" + "─" * 55 + "\n")
