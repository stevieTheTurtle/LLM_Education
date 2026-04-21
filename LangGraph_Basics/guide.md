# LangGraph Basics — Chatbot with Short-Term Memory

An introduction to LangGraph: what it is, how it works, and how to build a graph from scratch using the simplest possible example — a chatbot that remembers the conversation.

---

## What is LangGraph

LangGraph is a library that lets you build LLM applications as **directed graphs**. Each node in the graph is a Python function (an LLM call, an API call, a logic block). Edges define the execution order.

The advantage over a plain sequence of functions is that LangGraph handles:

- **Shared state** — a dictionary that flows through every node
- **Branching** — conditional edges that pick the next node at runtime
- **Parallelism** — multiple nodes running at the same time (fan-out / fan-in)
- **Reducers** — rules for how to update state fields when multiple nodes write to the same key

This example uses only the basics: a single node, a state with a reducer, and message memory.

---

## The three fundamental components

### 1. The State

The state is the **single source of truth** for the graph. It is a typed dictionary passed to every node. Nodes read from it and return a partial dict containing only the keys they produce.

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**`TypedDict`** — the standard way to type state in LangGraph. Subclassing `dict` also works, but `TypedDict` is cleaner for simple states.

**`Annotated[list[BaseMessage], add_messages]`** — this annotation does two things:
1. Declares the field type (`list[BaseMessage]`)
2. Registers a **reducer** (`add_messages`) that LangGraph calls when updating that field

Without the reducer, every node returning `{"messages": [...]}` would overwrite the entire list. With `add_messages`, LangGraph **appends** new messages to the existing ones instead of replacing them.

---

### 2. Nodes

A node is a plain Python function with this signature:

```python
def my_node(state: State) -> dict:
    # read from state
    # do something
    # return ONLY the keys you produced
    return {"messages": [new_message]}
```

Two fundamental rules:

**Rule 1 — Return only the keys you produce.** If you return `{**state, "messages": ...}`, you are also returning every other field, which will be written back to state even if you never touched them. In graphs with parallel nodes this causes `InvalidUpdateError`. Get into the habit of returning partial dicts immediately.

**Rule 2 — The reducer handles the merge.** You do not need to worry about blending your result with the existing state. LangGraph calls the reducer (in our case `add_messages`) and handles the update automatically.

---

### 3. The Graph

The graph is built with `StateGraph`, nodes and edges are added, and it is compiled:

```python
def build_graph():
    g = StateGraph(State)

    g.add_node("chat", chat)      # node name, function

    g.add_edge(START, "chat")     # START is the entry point
    g.add_edge("chat", END)       # END signals the graph is done

    return g.compile()
```

**`START` and `END`** — special LangGraph constants. `START` is the virtual node every execution begins from. `END` tells LangGraph the graph has finished. You never add them with `add_node`; they already exist.

**`g.compile()`** — validates the graph (unreachable nodes, missing edges, unhandled cycles) and returns a `Runnable` — an object you can call `.invoke()` on.

---

## Chat templates — System, Human, and AI messages

When you send a list of messages to a chat model, each message carries a **role** that tells the model who is speaking. This is the chat template format, and it is baked into every modern LLM API.

LangChain provides three message classes that map directly to these roles:

### SystemMessage

```python
SystemMessage(content="You are a helpful, concise assistant.")
```

A `SystemMessage` sets the **global instructions** for the model. It is not part of the conversation — it is a directive that shapes how the model behaves across all turns. Use it for things like: persona, tone, constraints, output format rules.

Key characteristics:
- Typically appears **once, at the top** of the message list
- The model follows it throughout the conversation but does not "reply" to it
- Not shown to the user — it is internal plumbing
- In our chatbot it is prepended inside the node at call time, not stored in state

### HumanMessage

```python
HumanMessage(content="What is the capital of France?")
```

A `HumanMessage` represents the **user's turn**. Each time the user types something, it becomes a `HumanMessage` appended to the history.

### AIMessage

```python
AIMessage(content="The capital of France is Paris.")
```

An `AIMessage` represents the **model's reply**. When you call `llm.invoke(messages)`, the return value is an `AIMessage`. You append it to history so the model can see its own previous answers on the next turn.

### Why the order matters

Models are trained to expect an alternating pattern. The full message list passed to the LLM on each turn looks like this:

```
SystemMessage  — "You are a helpful assistant."
HumanMessage   — "What is the capital of France?"
AIMessage      — "The capital of France is Paris."
HumanMessage   — "And Germany?"
AIMessage      — "Berlin."
HumanMessage   — "Thanks, one more — Italy?"
               ← model generates the next AIMessage here
```

This alternating history is how the model understands context. Without it, each question would be treated as brand new with no prior knowledge of what was said.

### In this chatbot

```python
SYSTEM_PROMPT = SystemMessage(content="You are a helpful, concise assistant. ...")

def chat(state: State) -> dict:
    history = [SYSTEM_PROMPT] + state["messages"]   # inject system at front
    response = llm.invoke(history)                  # response is an AIMessage
    return {"messages": [response]}                 # reducer appends it
```

`state["messages"]` already alternates HumanMessages and AIMessages from previous turns. The node prepends the `SystemMessage` and passes the full list. The model returns an `AIMessage`, which the reducer appends to state for the next turn.

---

## LLM sampling parameters — temperature, top_k, top_p

When you instantiate the model, you can pass parameters that control how it samples the next token. Understanding these is important because they directly affect output quality, creativity, and consistency.

```python
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7, top_k=40, top_p=0.9)
```

To understand what they do, you need to know what "sampling" means. At each step, the model produces a probability distribution over its entire vocabulary — tens of thousands of tokens, each with a score. Sampling parameters reshape that distribution before the model picks the next token.

---

### temperature

```python
ChatOllama(model="...", temperature=0.7)
```

Temperature **scales the probability distribution** before sampling. It is the most important parameter.

- **`temperature=0`** — the model always picks the single highest-probability token. Output is fully deterministic: same input always produces the same output. Use this for structured output, classification, data extraction — anything where you need consistency.
- **`temperature=1`** — probabilities are used as-is. The model samples naturally from the distribution. Balanced creativity and coherence.
- **`temperature > 1`** — the distribution is flattened: low-probability tokens become relatively more likely. Output becomes more creative, unexpected, and potentially incoherent.
- **`temperature < 1` (between 0 and 1)** — the distribution is sharpened: the top tokens become even more dominant. Output is more conservative and focused.

A useful mental model: temperature controls how much the model "trusts" its top guesses. At 0 it always takes the safest bet. At 2 it starts gambling.

For a conversational chatbot, values between `0.5` and `0.9` work well. The model responds naturally without becoming unpredictable.

---

### top_k

```python
ChatOllama(model="...", top_k=40)
```

Top-k **hard-limits the candidate pool** to the k highest-probability tokens before sampling. All other tokens are zeroed out and cannot be selected.

- **`top_k=1`** — same effect as `temperature=0`: only the single best token is considered, output is deterministic.
- **`top_k=40`** — the model picks from the 40 most likely tokens at each step. Common default.
- **`top_k=0` or disabled** — no hard cut-off, all tokens remain candidates.

Top-k is a blunt instrument: it cuts the tail of the distribution entirely, which prevents very unlikely tokens from ever appearing but can also eliminate legitimate low-probability words. It is usually combined with top_p, which is more adaptive.

---

### top_p (nucleus sampling)

```python
ChatOllama(model="...", top_p=0.9)
```

Top-p **keeps only the smallest set of tokens whose cumulative probability reaches p**, then samples from that set. It is also called nucleus sampling.

- **`top_p=0.9`** — find the fewest tokens that together cover 90% of the probability mass, then sample from that group. If one token has 91% probability on its own, only that token is considered. If the top 200 tokens are needed to reach 90%, all 200 are candidates.
- **`top_p=1.0`** — no filtering, all tokens are candidates (standard sampling).
- **`top_p=0.1`** — very narrow nucleus, almost always picks the single top token.

The key difference from top_k: top_p is **adaptive**. When the model is confident (one token dominates), the nucleus is tiny. When the model is uncertain (many tokens are plausible), the nucleus expands. This makes it more principled than top_k.

---

### How they interact

All three parameters are applied together before sampling:

1. **top_k** — remove all but the top k tokens
2. **top_p** — from what remains, keep the smallest nucleus that covers p probability mass
3. **temperature** — rescale the remaining probabilities, then sample

In practice:

| Use case | temperature | top_k | top_p |
|---|---|---|---|
| Structured output / classification | 0 | — | — |
| Factual Q&A, code generation | 0.2–0.4 | 40 | 0.9 |
| General conversation | 0.6–0.8 | 40 | 0.9 |
| Creative writing | 0.9–1.2 | 50–100 | 0.95 |
| Brainstorming / exploration | 1.2–1.5 | 0 | 1.0 |

When `temperature=0`, top_k and top_p have no effect — the argmax is taken directly.

---

## How add_messages works

`add_messages` is LangGraph's built-in reducer for message lists. It works like this:

```
Current state:   [HumanMessage("hi"), AIMessage("hello!")]
Node returns:    [AIMessage("how can I help?")]
State after:     [HumanMessage("hi"), AIMessage("hello!"), AIMessage("how can I help?")]
```

It simply appends the new messages to the existing list. This is exactly the behavior you want for a chatbot: each turn adds one human message and one AI message, building a growing transcript.

There is also a special behavior: if you pass a message with the same `id` as one already in the list, `add_messages` **replaces** it instead of appending. This is useful for updating in-progress messages during streaming, but you can ignore it for this example.

---

## Short-Term Memory

The STM in this chatbot is not a special LangGraph feature — it is simply the `history` list we maintain in the loop:

```python
history: list[BaseMessage] = []

while True:
    # ... read user input ...

    history.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": history})

    ai_reply = result["messages"][-1]
    history = result["messages"]   # save the full updated list

    print(f"Bot: {ai_reply.content}")
```

**How it works:**

1. Append the user's message to `history`
2. Invoke the graph passing the full history
3. The `chat` node receives all messages, sends them to the LLM, and returns the reply
4. The `add_messages` reducer appends the AI reply to the list
5. Save the updated list as the new `history`
6. On the next turn, the LLM sees both old and new messages

**Why short-term?** Because it lives only in memory for the duration of the script. When you exit and restart, `history` starts empty again. For persistent memory (long-term) you would need a database or LangGraph's checkpointer — but that is a more advanced pattern.

---

## Building the graph step by step

Following `chatbot.py` section by section.

---

### Step 1 — Imports

```python
from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
```

`add_messages` comes from `langgraph.graph.message`, not from `typing`. `START` and `END` come from `langgraph.graph` together with `StateGraph`.

---

### Step 2 — State

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

A single field. In LangGraph you do not need a giant state — only put in it what nodes need to share. The chatbot only needs the message list.

---

### Step 3 — The chat node

```python
def chat(state: State) -> dict:
    history = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(history)
    return {"messages": [response]}
```

Three lines. The node:
1. Prepends the system prompt to the history
2. Calls the LLM with the full list
3. Returns the response in a list (the reducer will append it)

The system prompt is not stored in state — it is injected at call time. This is intentional: we do not want the system prompt to accumulate in the history across turns.

---

### Step 4 — The graph

```python
def build_graph():
    g = StateGraph(State)
    g.add_node("chat", chat)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    return g.compile()
```

The simplest possible graph: one node, two edges.

```
START ──► chat ──► END
```

No branching, no parallelism, no conditional edges. This is the starting point from which everything else is built.

---

### Step 5 — Module-level `app`

```python
app = build_graph()
```

This line must live **outside** `if __name__ == "__main__":`. The LangGraph VS Code extension imports the module and looks up the variable named in `langgraph.json`. If `app` only exists inside `__main__`, the extension cannot find it and cannot visualize the graph.

`langgraph.json` tells the extension where to look:

```json
{
  "dependencies": ["."],
  "graphs": {
    "chatbot": "./chatbot.py:app"
  }
}
```

---

### Step 6 — The chat loop

```python
history: list[BaseMessage] = []

while True:
    user_input = input("You: ").strip()
    # ...
    history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": history})
    ai_reply = result["messages"][-1]
    history = result["messages"]
    print(f"Bot: {ai_reply.content}")
```

The loop keeps `history` between turns. `result["messages"]` is the full list after the reducer has done its merge — we save it as the new `history` so the next turn starts from the updated state.
