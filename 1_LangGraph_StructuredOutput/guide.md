# LangGraph Structured Output

How to make an LLM return a Python object instead of a string — reliably, with a schema, inside a LangGraph graph.

---

## What is structured output

By default an LLM returns free text. Structured output is a technique that constrains the model to return data that matches a schema — a JSON object with specific fields, types, and constraints.

Instead of:

```
"The movie Inception gets a 9/10. It's a sci-fi thriller and I recommend it."
```

You get a Python object you can use directly:

```python
MovieReview(
    title="Inception",
    sentiment="positive",
    score=9.0,
    genre="sci-fi thriller",
    key_points=["Intricate plot", "Stunning visuals", "Memorable score"],
    recommended=True
)
```

No string parsing. No regex. No `json.loads` on a substring you hope is valid. The object is already there.

---

## Why use it

| Without structured output | With structured output |
|---|---|
| Parse the model's text manually | Receive a typed Python object |
| Fragile — prompt changes break parsing | Schema is enforced independently of prose |
| Type errors discovered at runtime | Type errors caught at definition time |
| Hard to validate optional fields | Pydantic validates everything automatically |

Structured output is the correct approach whenever you need the LLM to produce data that code will consume — data extraction, classification, form filling, entity recognition, API response generation.

---

## The two mechanisms

LangChain's `with_structured_output` works through two different underlying mechanisms depending on the model:

### 1. Tool-calling (native)

Models with built-in tool-calling support expose a special API endpoint that accepts a JSON schema alongside the messages. The model fills the schema directly at the generation level — it is not generating text that happens to look like JSON, it is constrained to produce valid JSON from the first token.

```
User message  ──►  Model  ──►  Tool call result (validated JSON)
              schema ──►┘
```

Examples: `qwen2.5`, `llama3.1`, `mistral-nemo`, `command-r`, GPT-4, Claude.

### 2. JSON mode / prompt-based

Models without tool-calling fall back to JSON mode (if available) or prompt-based extraction: the system prompt instructs the model to return JSON matching a schema, and the output parser validates the result.

```
System prompt (with schema description)  ──►  Model  ──►  Text  ──►  Parser  ──►  Object
```

This is less reliable because the model is producing text, not a structured response. The system prompt closes the gap — but only if it is specific enough.

---

## The Pydantic schema

Both examples use the same schema:

```python
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie being reviewed")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    score: float = Field(description="Numeric score from 0.0 to 10.0")
    genre: str = Field(description="Primary genre of the movie")
    key_points: list[str] = Field(description="2 to 4 main points from the review")
    recommended: bool = Field(description="True if the reviewer recommends the movie")
```

**`BaseModel`** — Pydantic's base class. Provides automatic type validation when the model's JSON response is parsed into this class.

**`Field(description=...)`** — the description is passed to the model as part of the schema. For tool-calling models this is included in the tool definition and helps the model understand what each field means. It is not a system prompt — it is metadata attached to the schema.

**Type annotations matter** — `float` means the model must return a number, not the string `"9/10"`. `list[str]` means an array, not a comma-separated string. Pydantic validates all of this on the way back.

---

## `with_structured_output`

This is the LangChain method that wires everything together:

```python
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)
```

`structured_llm` is now a new runnable. Call `.invoke()` on it exactly like the original `llm`, but the return value is a `MovieReview` instance instead of an `AIMessage`.

```python
result = structured_llm.invoke([HumanMessage(content="...")])
# result is a MovieReview — not a string, not an AIMessage
print(result.score)      # 9.0
print(result.recommended)  # True
```

Under the hood:
1. LangChain inspects the model to determine if it supports tool-calling
2. If yes — sends the Pydantic schema as a tool definition
3. If no — wraps the schema as JSON instructions in the prompt
4. Either way, the response is validated against `MovieReview` and returned as an object

---

## The State

These graphs use a simple `TypedDict` state without `add_messages`:

```python
class State(TypedDict):
    review_text: str
    parsed: Optional[MovieReview]
```

**No reducer** — both fields are replaced on each write, not appended. This is intentional: we are not building conversation history. We are running a single extraction pass.

Compare with the basics chatbot:

| Basics chatbot | Structured output graph |
|---|---|
| `messages: Annotated[list, add_messages]` | `review_text: str` |
| Reducer appends each turn | Node overwrites directly |
| Accumulates history | Single-pass extraction |

Use reducers when state needs to accumulate. Use plain fields when state is replaced.

---

## The graph

Both files use the simplest possible graph:

```
START ──► extract ──► END
```

Single node, two edges. The complexity is inside the node — not in the graph topology.

```python
def build_graph():
    g = StateGraph(State)
    g.add_node("extract", extract)
    g.add_edge(START, "extract")
    g.add_edge("extract", END)
    return g.compile()
```

A structured output pipeline is a good fit for a linear graph. More complex graphs come into play when you need to validate the output, retry on failure, or route to different extraction logic based on content type.

---

## Native approach — `structured_output_native.py`

### The node

```python
def extract(state: State) -> dict:
    prompt = f"Extract structured information from this movie review:\n\n{state['review_text']}"
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    return {"parsed": result}
```

No system prompt. The prompt is purely about the task — "extract structured information from this review." The JSON formatting is handled entirely by the tool-calling mechanism.

### Why this works without format instructions

When `with_structured_output` detects tool-calling support, it sends the Pydantic schema as a tool definition alongside every call. The model does not produce free text and then try to format it as JSON. It produces the JSON object directly, field by field, as a tool call response.

The model never "decides" to respond in JSON — it is constrained to do so at the API level.

### Model choice

`qwen2.5:7b` is a strong choice for this because:
- Native tool-calling support via Ollama
- Reliable JSON output even on complex schemas
- Fast at 7B parameters

Other models that work natively: `llama3.1:8b`, `mistral-nemo`, `command-r`.

---

## Prompted approach — `structured_output_prompted.py`

### The system prompt

```python
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
```

### The node

```python
def extract(state: State) -> dict:
    messages = [
        SYSTEM_PROMPT,
        HumanMessage(content=f"Extract structured data from this review:\n\n{state['review_text']}"),
    ]
    result = structured_llm.invoke(messages)
    return {"parsed": result}
```

The system prompt is included in every call. It names each field, its type, its accepted values, and edge cases (score conversion). This covers the failure modes that less capable models exhibit without guidance.

### What to include in a structured output system prompt

**Name every field** — models may invent field names if the schema is not reinforced in the prompt. Listing them explicitly prevents drift.

**Specify types and allowed values** — `"one of exactly three values — positive, negative, or mixed"` is better than `"the sentiment"`. Enumerating valid values removes ambiguity.

**Handle edge cases** — `score` can appear as `"9/10"`, `"four stars"`, or `"excellent"`. The system prompt should specify the expected format and how to convert common alternatives.

**Explicit prohibitions** — `"Do not skip any field"` prevents the model from omitting fields it considers ambiguous.

**What not to include** — do not add `"respond in JSON"` or `"wrap your response in code fences"`. `with_structured_output` injects its own format directives. Adding your own can conflict with them and break parsing.

### Model choice

`gpt-oss:20b` is a capable general-purpose model that benefits from explicit output instructions. Without the system prompt it may produce incomplete or malformed structured responses. With it, output quality is reliable.

Other models in this category: `phi4-mini`, `gemma3:4b`, older `mistral` versions.

---

## Native vs Prompted — when to use which

| Situation | Approach |
|---|---|
| Model supports tool-calling (qwen2.5, llama3.1, mistral-nemo) | Native — no system prompt for format |
| Model lacks tool-calling or behaves inconsistently | Prompted — add field-level instructions |
| Schema is complex with many optional fields | Prompted — enumerate fields explicitly |
| Schema is simple (2–3 fields, clear types) | Native — Field descriptions are enough |
| Model skips fields in output | Add system prompt listing all fields |
| Model returns wrong types (e.g. "9/10" for a float) | Add type/conversion instructions |
| Model produces extra fields | Add "do not add extra fields" |

**Decision rule:** start native. Run a few examples. If you see malformed or incomplete output, add a system prompt — starting with the specific failure you observed.

---

## `temperature=0` for structured output

Both files use `temperature=0`. This is not optional — it is the correct default for structured output.

Higher temperatures introduce randomness at the token level. For prose this adds creativity. For structured output it corrupts field names, values, and types in ways that break validation.

```python
llm = ChatOllama(model="qwen2.5:7b", temperature=0)   # correct
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7) # wrong — use this only for conversational output
```

See the LangGraph Basics guide for a full explanation of temperature, top_k, and top_p.

---

## State field initialization

When invoking the graph, initialize every state field:

```python
result = app.invoke({"review_text": review, "parsed": None})
```

`parsed` starts as `None` and is replaced by the `extract` node. Always pass all keys defined in `State` — missing keys may cause `KeyError` inside nodes that read them.

---

## Accessing the result

After `.invoke()`, the parsed object is in `result["parsed"]`:

```python
result = app.invoke({"review_text": review, "parsed": None})
parsed: MovieReview = result["parsed"]

print(parsed.title)        # str
print(parsed.score)        # float
print(parsed.recommended)  # bool
print(parsed.key_points)   # list[str]
```

Full type information is available because `MovieReview` is a Pydantic model. IDE autocomplete works. `isinstance(parsed, MovieReview)` is `True`.

---

## Building the graph step by step

Following both `.py` files, section by section.

---

### Step 1 — Define the schema

```python
class MovieReview(BaseModel):
    title: str = Field(description="...")
    # ...
```

Start here. The schema is the contract between your code and the LLM. Get this right before writing anything else.

---

### Step 2 — Define the state

```python
class State(TypedDict):
    review_text: str
    parsed: Optional[MovieReview]
```

Include the raw input the node will read and the structured output the node will write.

---

### Step 3 — Bind the schema to the LLM

```python
llm = ChatOllama(model="...", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)
```

`structured_llm` is the object you call `.invoke()` on — not the raw `llm`.

---

### Step 4 — Write the node

Native:
```python
def extract(state: State) -> dict:
    prompt = f"Extract structured information from this review:\n\n{state['review_text']}"
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    return {"parsed": result}
```

Prompted:
```python
def extract(state: State) -> dict:
    messages = [SYSTEM_PROMPT, HumanMessage(content=f"...\n\n{state['review_text']}")]
    result = structured_llm.invoke(messages)
    return {"parsed": result}
```

Same structure. The only difference is whether a `SystemMessage` is included.

---

### Step 5 — Build and compile the graph

```python
def build_graph():
    g = StateGraph(State)
    g.add_node("extract", extract)
    g.add_edge(START, "extract")
    g.add_edge("extract", END)
    return g.compile()

app = build_graph()
```

`app` must be at module level for the VS Code extension to visualize it.

---

### Step 6 — Invoke

```python
result = app.invoke({"review_text": "...", "parsed": None})
parsed = result["parsed"]  # MovieReview instance
```

---

## Common failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ValidationError` on `score` — got `"9/10"` | Model returned string, schema expects float | Add system prompt: "return score as a decimal number" |
| `key_points` is a string, not a list | Weak model concatenated items | Add system prompt: "key_points must be a JSON array of strings" |
| Field missing from result | Model skipped it | Add system prompt: "fill every field, do not skip any" |
| Extra fields in result | Model hallucinated fields | Add system prompt: "do not add extra fields" |
| `OutputParserException` | Model returned malformed JSON | Switch to tool-calling model or lower temperature |
| Result is `None` | Model returned empty response | Check temperature=0, check model name is correct |

---

## Try it: break the native approach on purpose

`structured_output_native.py` uses `qwen2.5:7b` — a model with strong tool-calling support. To see exactly why the prompted approach exists, swap it for a model that lacks reliable structured output:

```python
# structured_output_native.py — line 71
llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)  # swap in here
```

Run the script and observe the debug output:

```
RAW LLM Output:
'title=... sentiment=... score=...'   ← may be malformed, missing fields, or wrong types

Parsed Structured Output:
...                                   ← ValidationError or partial object
```

Common things you will see with a weaker model and no system prompt:

- `score` comes back as `"9 out of 10"` (string) instead of `9.0` (float) — Pydantic raises `ValidationError`
- `key_points` comes back as a single concatenated string instead of a list
- `sentiment` is something like `"mostly positive"` instead of one of the three allowed values
- Fields are missing entirely

Once you see a specific failure, open `structured_output_prompted.py` and look at how the system prompt handles that exact case. Then switch `structured_output_native.py` back to `qwen2.5:7b`.

This is the fastest way to build intuition for when and why system prompt guidance is needed.

---

## Summary

```
with_structured_output(Schema)
    │
    ├── Tool-calling model?  → schema sent as tool definition → JSON at API level
    │                                                            No format prompt needed
    │
    └── No tool-calling?     → schema sent via prompt         → text parsed to JSON
                                                               System prompt required for reliability
```

Both approaches return a Pydantic object. The difference is where the formatting constraint is enforced — at the API level (native) or at the text level (prompted). Use the native approach when the model supports it. Add system prompt guidance only when you observe output quality problems.
