"""
LangGraph Structured Output — Prompted (System Prompt Guidance)
===============================================================
Concepts demonstrated:

  1. Same Pydantic schema    — output shape is identical to native.py
  2. with_structured_output  — still used for parsing
  3. Explicit system prompt  — field-by-field instructions for the model
  4. When to use this        — smaller models, no tool-calling, ambiguous schemas

Use case: same movie review extractor, but the model needs explicit
          instructions to reliably populate every field.

Graph shape:

    START ──► extract ──► END

Key difference from native.py:
    The system prompt names every field, its type, and its allowed values.
    Without these hints, models that lack strong tool-calling support may:
      - skip optional fields entirely
      - return wrong types (e.g. score as "8/10" instead of 8.0)
      - hallucinate extra fields not in the schema
      - wrap the JSON in markdown fences that break parsing

When should you add system prompt instructions?
    Rule of thumb: start without them (native approach). Add a system prompt
    only when you observe the model producing malformed or incomplete output.
    The right amount of instruction is the minimum that makes output reliable.
"""

from pprint import pprint
from typing import Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# ── 1. OUTPUT SCHEMA ──────────────────────────────────────────────────────────
# Identical to native.py. The schema itself does not change — only the prompt
# strategy differs.

class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie being reviewed")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    score: float = Field(description="Numeric score from 0.0 to 10.0")
    genre: str = Field(description="Primary genre of the movie")
    key_points: list[str] = Field(description="2 to 4 main points from the review")
    recommended: bool = Field(description="True if the reviewer recommends the movie")


# ── 2. STATE ──────────────────────────────────────────────────────────────────

class State(TypedDict):
    review_text: str
    parsed: Optional[MovieReview]


# ── 3. LLM ────────────────────────────────────────────────────────────────────
# gpt-oss:20b-cloud is a capable model for text tasks but its JSON/tool-calling
# reliability is lower than dedicated models like qwen2.5.
# The system prompt below compensates for that gap.
#
# temperature=0 is mandatory for structured output — higher values introduce
# randomness that can corrupt field names or values.

llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)


# ── 4. SYSTEM PROMPT ──────────────────────────────────────────────────────────
# This is the core of the "prompted" approach.
#
# What to include in a structured output system prompt:
#   - The task (what the model is analyzing)
#   - Every field by name, type, and accepted values
#   - Constraints (e.g. exactly 2–4 items, no extra fields)
#   - Explicit "do not" rules for failure modes you have observed
#
# What NOT to include:
#   - Instructions to "respond in JSON" or "wrap in code fences" — those
#     interfere with with_structured_output, which injects its own format
#     directives. Describing the task and fields is enough.

SYSTEM_PROMPT = SystemMessage(content="""
You are a movie review analyzer. Your job is to extract structured data from movie reviews.

Fill every field listed below. Do not skip any field. Do not add extra fields.

Fields:
- title      (string)  : the exact movie name mentioned in the review
- sentiment  (string)  : one of exactly three values — "positive", "negative", or "mixed"
- score      (float)   : a decimal number between 0.0 and 10.0 representing the rating
                         If the reviewer says "9 out of 10", use 9.0. If they say "4/5", convert to 8.0.
- genre      (string)  : single primary genre, e.g. "thriller", "comedy", "horror", "drama"
- key_points (list)    : between 2 and 4 short strings, each summarizing one point from the review
- recommended (boolean): true if the reviewer recommends watching the movie, false otherwise

If any piece of information is not explicitly stated, make a reasonable inference from context.
""".strip())


# ── 5. NODE ───────────────────────────────────────────────────────────────────
# Both the system prompt and the user review are passed to the model.
# with_structured_output sits on top: it receives the model response and
# converts it into a MovieReview regardless of which model generated it.

def extract(state: State) -> dict:
    messages = [
        SYSTEM_PROMPT,
        HumanMessage(content=f"Extract structured data from this review:\n\n{state['review_text']}"),
    ]
    result = structured_llm.invoke(messages)
    return {"parsed": result}


# ── 6. GRAPH ──────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)

    g.add_node("extract", extract)

    g.add_edge(START, "extract")
    g.add_edge("extract", END)

    return g.compile()


# ── 7. MODULE-LEVEL APP ───────────────────────────────────────────────────────

app = build_graph()


# ── 8. DEMO LOOP ──────────────────────────────────────────────────────────────

CYAN = "\033[36m"
GRAY = "\033[90m"
SOUT  = "\033[32m"
RST  = "\033[0m"

MOVIE_REVIEW = """Inception is a mind-bending sci-fi thriller. Nolan crafts an intricate dream-within-a-dream story that keeps you on edge the whole time. The visuals are stunning and the score is unforgettable. I'd give it a 9 out of 10 — absolutely watch it."""

print("LangGraph Structured Output — Prompted Model (gpt-oss:20b-cloud)")
print(f"{GRAY}System prompt lists every field, type, and constraint explicitly{RST}\n")

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
