"""
LangGraph Minimal Example
=========================
Concepts demonstrated:

  1. Conditional edge    — check_intent gates the whole graph
  2. Fan-out / fan-in    — parse_city fans out to two parallel API nodes
  3. Structured output   — LLM returns Pydantic objects at check_intent,
                           parse_city, and final_answer
  4. API-grounded LLM    — final_answer is told to use ONLY the live data
                           injected into its prompt, not training knowledge

Graph shape:

    check_intent ──(not city)──► casual_response ──► END
         │
      (city)
         ▼
    parse_city
      │        │
      ▼        ▼
  fetch_      fetch_
  weather      geo
      │        │
      └────┬───┘   (fan-in: LangGraph waits for both before continuing)
           ▼
      final_answer ──► END
"""

import json
import requests
from typing import Optional, Literal, Type

# ── COLORS ────────────────────────────────────────────────────────────────────
RST   = "\033[0m"
GRAY  = "\033[90m"   # raw LLM output
NODE  = "\033[33m"   # node headers and API results
JSON  = "\033[32m"   # parsed structured output

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ── 1. STATE ──────────────────────────────────────────────────────────────────
# One shared dict that flows through every node.
# Nodes read from it and return only the keys they produce.

class State(dict):
    user_query:    str
    is_city_query: Optional[bool]
    casual_reply:  Optional[str]
    city_name:     Optional[str]
    weather:       Optional[str]
    latitude:      Optional[float]
    longitude:     Optional[float]
    country:       Optional[str]
    report:        Optional[str]


# ── 2. PYDANTIC SCHEMAS ───────────────────────────────────────────────────────
# Each schema is the contract between the node and the LLM.
# with_structured_output() forces the model to fill these fields.

class IntentCheck(BaseModel):
    is_city_query: bool = Field(
        description="True if the user is asking about a specific place or city."
    )
    reply: str = Field(
        description="If is_city_query is False, a short friendly reply. Otherwise empty string."
    )


class ParsedCity(BaseModel):
    city_name: str = Field(
        description="Canonical English city name extracted from the query, or empty string if none."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident you are in the extraction."
    )



# ── 3. PROMPT HELPER ─────────────────────────────────────────────────────────
# Generates a field-by-field instruction string from any Pydantic model.
# Appended to system prompts so the LLM knows the exact JSON shape to emit.

def schema_instruction(model: Type[BaseModel]) -> str:
    schema = model.model_json_schema()
    lines = [f"Respond with a JSON object with EXACTLY these fields for {model.__name__}:"]
    for name, info in schema.get("properties", {}).items():
        if "enum" in info:
            type_hint = " | ".join(f'"{v}"' for v in info["enum"])
        elif "type" in info:
            type_hint = info["type"]
        else:
            type_hint = "any"
        desc = f" — {info['description']}" if info.get("description") else ""
        lines.append(f'  "{name}" ({type_hint}){desc}')
    lines.append("Do NOT add, rename, or omit any field.")
    lines.append("Do NOT wrap the object under any outer key like the class name.")
    return "\n".join(lines)


# ── 4. LLM ────────────────────────────────────────────────────────────────────
# include_raw=True → result is {"raw": AIMessage, "parsed": <Model>|None, "parsing_error": ...}
# We always access result["parsed"] and guard against None.

llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)

intent_llm = llm.with_structured_output(IntentCheck, method="json_schema", include_raw=True)
city_llm   = llm.with_structured_output(ParsedCity,  method="json_schema", include_raw=True)


# ── 5. NODES ──────────────────────────────────────────────────────────────────
# Each node receives the full state and returns ONLY the keys it produces.
# Parallel nodes MUST return partial dicts — returning {**state, ...} causes
# InvalidUpdateError when LangGraph merges concurrent writes.

def check_intent(state: State) -> dict:
    """Gate: structured LLM decides if query is about a place."""
    print(f"{NODE}🤔  [check_intent]{RST}")
    raw = intent_llm.invoke([
        SystemMessage(content="You are an intent classifier.\n\n" + schema_instruction(IntentCheck)),
        HumanMessage(content=state["user_query"]),
    ])
    print(f"{GRAY}    [raw] {raw['raw'].content}{RST}")
    if raw["parsed"] is None:
        return {"is_city_query": False, "casual_reply": "Sorry, I couldn't understand that."}
    r: IntentCheck = raw["parsed"]
    print(f"{JSON}    → is_city_query={r.is_city_query}{RST}")
    return {"is_city_query": r.is_city_query, "casual_reply": r.reply}


def casual_response(state: State) -> dict:
    """Dead-end for non-city queries: just print the LLM's reply."""
    print(f"\n💬  {state.get('casual_reply', 'Ask me about any city!')}\n")
    return {}


def parse_city(state: State) -> dict:
    """Structured LLM extracts the city name from the query."""
    print(f"{NODE}🔍  [parse_city]{RST}")
    raw = city_llm.invoke([
        SystemMessage(content="Extract the city name from the message.\n\n" + schema_instruction(ParsedCity)),
        HumanMessage(content=state["user_query"]),
    ])
    print(f"{GRAY}    [raw] {raw['raw'].content}{RST}")
    if raw["parsed"] is None or not raw["parsed"].city_name:
        return {"city_name": None}
    r: ParsedCity = raw["parsed"]
    print(f"{JSON}    → '{r.city_name}' (confidence: {r.confidence}){RST}")
    return {"city_name": r.city_name}


def fetch_weather(state: State) -> dict:
    """Calls wttr.in — real data, zero LLM involvement."""
    city = state.get("city_name")
    if not city:
        return {"weather": "unavailable"}
    print(f"{NODE}🌤️   [fetch_weather] wttr.in → {city}{RST}")
    try:
        resp = requests.get(
            f"https://wttr.in/{requests.utils.quote(city)}?format=j1", timeout=10
        )
        c = resp.json()["current_condition"][0]
        weather = f"{c['weatherDesc'][0]['value']}, {c['temp_C']}°C, humidity {c['humidity']}%"
        print(f"{NODE}    → {weather}{RST}")
        return {"weather": weather}
    except Exception as e:
        return {"weather": f"unavailable ({e})"}


def fetch_geo(state: State) -> dict:
    """Calls Nominatim — real coordinates, zero LLM involvement."""
    city = state.get("city_name")
    if not city:
        return {"latitude": None, "longitude": None, "country": None}
    print(f"{NODE}🗺️   [fetch_geo] nominatim → {city}{RST}")
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "langgraph-example/1.0"},
            timeout=10,
        )
        hit = resp.json()[0]
        lat = float(hit["lat"])
        lon = float(hit["lon"])
        country = hit.get("display_name", "").split(",")[-1].strip()
        print(f"{NODE}    → {lat}, {lon}, {country}{RST}")
        return {"latitude": lat, "longitude": lon, "country": country}
    except Exception as e:
        return {"latitude": None, "longitude": None, "country": f"unavailable ({e})"}


def final_answer(state: State) -> dict:
    """LLM answers the user's question using ONLY the live API data."""
    print(f"{NODE}📝  [final_answer]{RST}")
    api_data = (
        f"City: {state.get('city_name')}\n"
        f"Country: {state.get('country')}\n"
        f"Coordinates: {state.get('latitude')}, {state.get('longitude')}\n"
        f"Current weather: {state.get('weather')}\n"
    )
    response = llm.invoke([
        SystemMessage(content=(
            "Answer the user's question using ONLY the API data provided. "
            #"Do NOT use any training knowledge about the city."
        )),
        HumanMessage(content=(
            f"Question: {state['user_query']}\n\nAPI data:\n{api_data}"
        )),
    ])
    print(f"\n{response.content}\n")
    return {"report": response.content}


# ── 6. ROUTING ────────────────────────────────────────────────────────────────

def route_intent(state: State) -> str:
    return "parse_city" if state.get("is_city_query") else "casual_response"


# ── 7. GRAPH ──────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)

    g.add_node("check_intent",    check_intent)
    g.add_node("casual_response", casual_response)
    g.add_node("parse_city",      parse_city)
    g.add_node("fetch_weather",   fetch_weather)   # ─┐ parallel
    g.add_node("fetch_geo",       fetch_geo)        # ─┘
    g.add_node("final_answer",    final_answer)

    g.set_entry_point("check_intent")

    # Conditional: route to city research or casual chat
    g.add_conditional_edges("check_intent", route_intent, {
        "parse_city":      "parse_city",
        "casual_response": "casual_response",
    })
    g.add_edge("casual_response", END)

    # Fan-out: one node → two parallel nodes
    g.add_edge("parse_city", "fetch_weather")
    g.add_edge("parse_city", "fetch_geo")

    # Fan-in: LangGraph waits for both before running final_answer
    g.add_edge("fetch_weather", "final_answer")
    g.add_edge("fetch_geo",     "final_answer")

    g.add_edge("final_answer", END)

    return g.compile()


# ── 8. CHAT LOOP ──────────────────────────────────────────────────────────────

# Module-level: required for the LangGraph VS Code extension to discover the graph.
# langgraph.json points to "example.py:app".
app = build_graph()

if __name__ == "__main__":
    print("City Research Agent — type a query or /exit to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query == "/exit":
            print("Bye!")
            break

        print()
        app.invoke({
            "user_query":    query,
            "is_city_query": None,
            "casual_reply":  None,
            "city_name":     None,
            "weather":       None,
            "latitude":      None,
            "longitude":     None,
            "country":       None,
            "report":        None,
        })
        print("─" * 55)
