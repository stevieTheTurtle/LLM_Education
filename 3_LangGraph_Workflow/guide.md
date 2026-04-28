# Building an LLM Graph System with LangGraph

A step-by-step guide to designing and implementing a non-linear, API-grounded agent using LangGraph, LangChain, and Pydantic.

---

## What we are building

A conversational city research agent. You type a question, and the system figures out whether you are asking about a place, looks up live data from free public APIs, and answers you using only that real data — not the model's pre-trained memory.

The interesting part is not the task itself but the architecture: the agent is a **graph**, not a pipeline. Different paths execute depending on what the LLM decides, two nodes run in parallel, and structured output is used to make LLM responses machine-readable at critical decision points.

Here is the final shape of the graph:

```
check_intent ──(not a city)──► casual_response ──► END
      │
   (city)
      ▼
 parse_city
   │       │
   ▼       ▼
fetch_   fetch_       ← these two run in parallel
weather   geo
   │       │
   └───┬───┘
       ▼
 final_answer ──► END
```

---

## Step 1 — Understand the core idea of LangGraph

Before writing a single line, it is worth understanding why LangGraph exists and what problem it solves.

A simple LLM call is stateless: you send a prompt, you get a response. For more complex tasks — where the system needs to make decisions, call tools, and pass information between steps — you need to chain things together. The naive approach is to write a Python function that calls other functions in sequence.

LangGraph formalises this as a **directed graph**:

- **Nodes** are the units of work (an LLM call, an API call, a print statement).
- **Edges** define which node runs after which.
- **Conditional edges** let you branch: run node A or node B depending on the current state.
- **The state** is a shared dictionary that flows through every node. Nodes read from it and write back to it.

Why bother? Because graphs can express things that linear code cannot do cleanly:

| Pattern | What it means |
|---|---|
| Conditional edge | Different nodes run depending on a decision made at runtime |
| Fan-out | One node triggers two nodes simultaneously |
| Fan-in | Two nodes both feed into a third, which waits for both |

All three of these appear in our script, and each one exists for a real reason that we will explain as we build.

---

## Step 2 — Design the graph before writing code

A common mistake when building graph systems is jumping straight to code. Instead, spend five minutes drawing the graph on paper (or in your head) and answering three questions for each node:

1. **What does this node need from the state?**
2. **What does it write back to the state?**
3. **Does it connect unconditionally, or does it branch?**

For our agent:

| Node | Reads | Writes | Edge type |
|---|---|---|---|
| `check_intent` | `user_query` | `is_city_query`, `casual_reply` | Conditional |
| `casual_response` | `casual_reply` | — | → END |
| `parse_city` | `user_query` | `city_name` | Fan-out |
| `fetch_weather` | `city_name` | `weather` | → `final_answer` |
| `fetch_geo` | `city_name` | `latitude`, `longitude`, `country` | → `final_answer` |
| `final_answer` | everything | `report` | → END |

Do this exercise first. The code practically writes itself once you have a clear table like this.

---

## Step 3 — Define the State

```python
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
```

The state is the single source of truth. Every node receives it, and every node returns a **partial dict** of only the keys it produces. LangGraph merges those partial dicts back into the state after each step.

Two things to notice:

**Why `Optional` for almost everything?** Because at the start of a run, most fields are unknown. `check_intent` has not run yet, so `city_name` is `None`. We initialize the state with `None` for all optional fields before invoking the graph.

**Why subclass `dict` instead of using a `TypedDict`?** Either works with LangGraph. Subclassing `dict` is slightly more flexible for our purposes because we can call `.get()` with a default value throughout the node functions. The type annotations are for documentation; Python does not enforce them at runtime.

---

## Step 4 — Define Pydantic schemas for structured LLM output

Not every node needs an LLM. But when an LLM is involved in a **decision** (should we go left or right? what is the city name?), we need its output to be machine-readable. Free-form text like "I think the city might be Tokyo" cannot drive a conditional edge.

This is what Pydantic schemas are for. We define the exact shape we want:

```python
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
```

The `description` on each field does double duty: it documents the code **and** gets included in the prompt sent to the LLM, telling it what each field should contain.

Notice that `final_answer` does **not** use a schema. That node produces text for the user to read, not data for the program to branch on. Structured output is only necessary where the program needs to act on the result programmatically.

---

## Step 5 — Write a prompt helper that stays in sync with your schemas

Here is a subtle but important problem: you have a Pydantic schema that defines your fields, and you also have a system prompt that tells the LLM what to output. These two things can silently get out of sync. You rename a field in the schema but forget to update the prompt, and now the LLM outputs the old name and parsing fails.

The fix is to generate the prompt instructions **from** the schema automatically:

```python
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
```

`model.model_json_schema()` is a Pydantic v2 method that returns the JSON Schema representation of the model, including all field names, types, and descriptions. We walk over `properties` and build a bullet list.

The last two lines are worth explaining:

- **"Do NOT add, rename, or omit any field."** — Models sometimes hallucinate extra fields or omit optional-looking ones.
- **"Do NOT wrap the object under any outer key like the class name."** — Some models output `{"ParsedCity": {"city_name": ...}}` instead of `{"city_name": ...}`. This wrapper breaks parsing because Pydantic expects a flat object.

When you call `schema_instruction(ParsedCity)`, it produces something like:

```
Respond with a JSON object with EXACTLY these fields for ParsedCity:
  "city_name" (string) — Canonical English city name extracted from the query, or empty string if none.
  "confidence" ("high" | "medium" | "low") — How confident you are in the extraction.
Do NOT add, rename, or omit any field.
Do NOT wrap the object under any outer key like the class name.
```

This gets appended to every system prompt that expects structured output.

---

## Step 6 — Set up the LLM instances

```python
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)

intent_llm = llm.with_structured_output(IntentCheck, method="json_schema", include_raw=True)
city_llm   = llm.with_structured_output(ParsedCity,  method="json_schema", include_raw=True)
```

**`temperature=0`** removes randomness. For structured output and intent classification, we want deterministic, consistent responses. Creative temperature is appropriate for the final prose answer, not for parsing a city name.

**`.with_structured_output()`** wraps the model so that it attempts to parse the response into the given Pydantic class. Under the hood it uses the model's JSON mode and validates the result against the schema.

**`include_raw=True`** is a debugging choice. Without it, a parsing failure raises an exception immediately. With it, the result is always a dict:

```python
{
    "raw":           AIMessage(...),   # the raw LLM response, always present
    "parsed":        ParsedCity(...),  # the Pydantic object, or None if parsing failed
    "parsing_error": None              # the exception, or None if parsing succeeded
}
```

This lets us inspect the raw LLM output when something goes wrong, and handle parse failures gracefully instead of crashing. We always guard against `parsed is None` in node code.

Note that `final_answer` uses the base `llm` directly — no structured output — because we want a free-form prose response there.

---

## Step 7 — Write the node functions

Every node follows the same contract:

> Receive the full state. Return a dict of **only the keys you produce**.

This last part is critical. When two nodes run in parallel (fan-out), LangGraph merges their return dicts back into the state at the same time. If both nodes return `{**state, "weather": ...}`, LangGraph sees two competing writes for `user_query`, `city_name`, and every other key — and raises `InvalidUpdateError`. Return only what you own.

### check_intent — the gate

```python
def check_intent(state: State) -> dict:
    raw = intent_llm.invoke([
        SystemMessage(content="You are an intent classifier.\n\n" + schema_instruction(IntentCheck)),
        HumanMessage(content=state["user_query"]),
    ])
    if raw["parsed"] is None:
        return {"is_city_query": False, "casual_reply": "Sorry, I couldn't understand that."}
    r: IntentCheck = raw["parsed"]
    return {"is_city_query": r.is_city_query, "casual_reply": r.reply}
```

This node decides whether the rest of the graph runs at all. If the user types "what's the weather in Rome?" the graph continues. If they type "what's the meaning of life?", the graph short-circuits to `casual_response` and ends.

Structured output is essential here because `is_city_query` is a boolean that drives a conditional edge. The routing function reads it directly:

```python
def route_intent(state: State) -> str:
    return "parse_city" if state.get("is_city_query") else "casual_response"
```

### parse_city — NLP extraction

```python
def parse_city(state: State) -> dict:
    raw = city_llm.invoke([
        SystemMessage(content="Extract the city name from the message.\n\n" + schema_instruction(ParsedCity)),
        HumanMessage(content=state["user_query"]),
    ])
    if raw["parsed"] is None or not raw["parsed"].city_name:
        return {"city_name": None}
    r: ParsedCity = raw["parsed"]
    return {"city_name": r.city_name}
```

The LLM is doing pure NLP here — extracting a structured piece of information from natural language. It is not being asked for any factual knowledge about the city. "What's the best time to visit Tokyo?" → city name is `"Tokyo"`. That is all this node cares about.

### fetch_weather and fetch_geo — pure API nodes

```python
def fetch_weather(state: State) -> dict:
    city = state.get("city_name")
    if not city:
        return {"weather": "unavailable"}
    resp = requests.get(f"https://wttr.in/{requests.utils.quote(city)}?format=j1", timeout=10)
    c = resp.json()["current_condition"][0]
    weather = f"{c['weatherDesc'][0]['value']}, {c['temp_C']}°C, humidity {c['humidity']}%"
    return {"weather": weather}
```

No LLM is involved in these nodes. They call free public APIs:

- **wttr.in** — a weather service with a clean JSON endpoint, no API key required.
- **Nominatim** (OpenStreetMap) — a geocoding service, also free and keyless.

These nodes demonstrate a key architectural principle: **the LLM should be told facts, not asked to recall them**. If you ask an LLM "what is the current weather in Tokyo?", it will either refuse or hallucinate. If you call an API, get the real data, and inject it into the prompt, the LLM can write accurate, useful prose about it.

These two nodes are also the fan-out: both run immediately after `parse_city`, in parallel. LangGraph does not guarantee ordering between them, and neither needs to wait for the other. This is the correct design — they are independent tasks.

### final_answer — prose generation grounded in API data

```python
def final_answer(state: State) -> dict:
    api_data = (
        f"City: {state.get('city_name')}\n"
        f"Country: {state.get('country')}\n"
        f"Coordinates: {state.get('latitude')}, {state.get('longitude')}\n"
        f"Current weather: {state.get('weather')}\n"
    )
    response = llm.invoke([
        SystemMessage(content="Answer the user's question using ONLY the API data provided."),
        HumanMessage(content=f"Question: {state['user_query']}\n\nAPI data:\n{api_data}"),
    ])
    print(f"\n{response.content}\n")
    return {"report": response.content}
```

This is the fan-in. LangGraph waits for **both** `fetch_weather` and `fetch_geo` to finish before running `final_answer`, because both contribute to the state that this node reads.

The LLM here is used as a prose writer, not a knowledge base. It receives:
1. The user's original question (so it knows what angle to take).
2. All the API data collected by the previous nodes.

It is told explicitly to use only the provided data. The question shapes the response — "what should I wear?" gets a weather-focused answer, "is it far from the coast?" gets a geography-focused one — but all facts come from the APIs.

---

## Step 8 — Assemble the graph

```python
def build_graph():
    g = StateGraph(State)

    g.add_node("check_intent",    check_intent)
    g.add_node("casual_response", casual_response)
    g.add_node("parse_city",      parse_city)
    g.add_node("fetch_weather",   fetch_weather)
    g.add_node("fetch_geo",       fetch_geo)
    g.add_node("final_answer",    final_answer)

    g.set_entry_point("check_intent")

    # Conditional branch
    g.add_conditional_edges("check_intent", route_intent, {
        "parse_city":      "parse_city",
        "casual_response": "casual_response",
    })
    g.add_edge("casual_response", END)

    # Fan-out
    g.add_edge("parse_city", "fetch_weather")
    g.add_edge("parse_city", "fetch_geo")

    # Fan-in (implicit: both edges point to the same target)
    g.add_edge("fetch_weather", "final_answer")
    g.add_edge("fetch_geo",     "final_answer")

    g.add_edge("final_answer", END)

    return g.compile()
```

Reading this top to bottom, the graph structure is immediately clear. Each `add_edge` or `add_conditional_edges` call is a single sentence: "after this, go there".

**The fan-in is implicit.** You do not write any special "wait for both" instruction. LangGraph infers it: `final_answer` has two incoming edges, so it waits until both source nodes have completed before it fires.

**`g.compile()`** validates the graph (checks for unreachable nodes, missing edges, etc.) and returns a `Runnable` — an object you can call `.invoke()` on.

---

## Step 9 — Expose the graph at module level

```python
app = build_graph()
```

This line lives **outside** `if __name__ == "__main__":`. The LangGraph VS Code extension discovers your graph by importing the module and looking up the variable named in `langgraph.json`. If `app` only exists inside the `__main__` block, the extension cannot find it.

The `langgraph.json` file in the project root tells the extension where to look:

```json
{
  "dependencies": ["."],
  "graphs": {
    "city_agent": "./example.py:app"
  }
}
```

---

## Step 10 — The chat loop

```python
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
```

Each invocation creates a fresh initial state with `None` for every optional field. This is important: the graph is stateless between turns. Nothing carries over from the previous question. If you wanted multi-turn memory (e.g. "tell me more about it" referring to the previous city), you would need to maintain and pass conversation history explicitly — but that is a different pattern.

The `EOFError` catch handles the case where stdin is closed (e.g. running in a pipe). `KeyboardInterrupt` handles `Ctrl+C`. Both exit gracefully.

---

## Concepts summary

| Concept | Where it appears | Why it is needed |
|---|---|---|
| **State as shared dict** | `class State(dict)` | Single source of truth across all nodes |
| **Partial return dicts** | Every node | Parallel nodes cannot write to the same key twice |
| **Pydantic structured output** | `check_intent`, `parse_city` | Machine-readable decisions from LLM responses |
| **`include_raw=True`** | All structured LLM chains | Allows graceful handling of parse failures |
| **`schema_instruction()`** | All structured nodes | Keeps prompt and schema in sync automatically |
| **Conditional edge** | `check_intent` → `route_intent` | Runtime branching without if/else in caller code |
| **Fan-out** | `parse_city` → two nodes | Independent tasks run in parallel |
| **Fan-in** | Two nodes → `final_answer` | Automatic synchronisation barrier |
| **API-grounded generation** | `final_answer` | Facts come from live data, not model memory |
| **Module-level `app`** | Below `build_graph()` | Required for VS Code extension discovery |
