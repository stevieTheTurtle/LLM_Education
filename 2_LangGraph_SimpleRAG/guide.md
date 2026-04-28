# LangGraph Simple RAG — Researcher Paper Lookup

A three-node graph that takes a researcher's name, fetches their recent papers from a live API, and generates a grounded summary. Introduces the RAG pattern, multi-node state pipelines, and external API calls inside graph nodes.

---

## What is RAG

RAG stands for **Retrieval-Augmented Generation**. It is a pattern for grounding LLM output in real, external data instead of relying on what the model memorized during training.

The three stages map directly to the three nodes in this graph:

| Stage | What happens | Node |
|---|---|---|
| **Retrieve** | Fetch relevant data from an external source | `fetch_papers` |
| **Augment** | Inject the retrieved data into the LLM prompt | inside `summarize` |
| **Generate** | LLM reasons over retrieved data and produces output | `summarize` |

Without RAG, an LLM answering "what has Yann LeCun published recently?" is guessing from training data that may be months or years old. With RAG, the answer is built from papers retrieved right now.

---

## Why RAG in a graph

LangGraph is a good fit for RAG because the three stages are naturally separate nodes:

- **Retrieval** is a side effect — it calls an API, handles errors, returns data. No LLM involved.
- **Augmentation** is prompt construction — it formats retrieved data into a string the LLM can read.
- **Generation** is the LLM call — it has no knowledge of how the data was fetched.

Separating these into nodes makes each one testable, replaceable, and easy to reason about. You can swap the retrieval source (Semantic Scholar → arXiv → PubMed) without touching the generation node.

---

## The graph

```
START ──► parse_query ──► fetch_papers ──► summarize ──► END
```

Three nodes, four edges. Strictly linear — data flows forward only.

```python
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
```

This is the first graph in this series with more than one node. The shape is still simple — but the key lesson is how state acts as the data bus between nodes.

---

## State as a data pipeline

In the chatbot example, state held a message list that grew with each turn. Here, state holds separate typed fields that different nodes fill in sequence.

```python
class State(TypedDict):
    raw_query:   str                    # set by the caller — never modified
    researcher:  Optional[ResearcherQuery]  # filled by parse_query
    author_name: str                    # filled by fetch_papers
    papers:      list[dict]             # filled by fetch_papers
    response:    str                    # filled by summarize
```

Think of state as a shared document that moves through the pipeline. Each node opens the document, reads what it needs, writes its section, and passes it on.

```
invoke({raw_query: "Yann LeCun"})
         │
         ▼
   parse_query  →  state.researcher = ResearcherQuery(first_name="Yann", last_name="LeCun")
         │
         ▼
   fetch_papers →  state.author_name = "Yann LeCun"
                   state.papers = [{title: ..., year: ...}, ...]
         │
         ▼
   summarize    →  state.response = "Yann LeCun's recent work focuses on..."
         │
         ▼
   result = {raw_query, researcher, author_name, papers, response}
```

The final `result` from `.invoke()` is the complete state after all nodes have run.

---

## Node 1 — `parse_query`: structured input extraction

```python
parse_llm = ChatOllama(model="qwen2.5:7b", temperature=0).with_structured_output(ResearcherQuery)

def parse_query(state: State) -> dict:
    result = parse_llm.invoke([
        HumanMessage(content=f"Extract the researcher's name from: {state['raw_query']}")
    ])
    return {"researcher": result}
```

The user might type anything:

- `"Yann LeCun"` — clean name
- `"show me papers by Geoffrey Hinton from University of Toronto"` — embedded in prose, includes institution
- `"andrej karpathy recent work"` — lowercase, no structure

`parse_query` normalizes all of these into a `ResearcherQuery` object with `first_name`, `last_name`, and optional `institution`. The next node can then build a reliable API query string without any string manipulation.

### ResearcherQuery schema

```python
class ResearcherQuery(BaseModel):
    first_name:  str
    last_name:   str
    institution: Optional[str] = Field(default=None, description="...")
```

`institution` defaults to `None` — if the user does not mention one, the field is absent. This is used in `fetch_papers` to optionally narrow the API search.

### Why structured output here (not at the end)

In the StructuredOutput examples, `with_structured_output` produced the final result. Here it produces an **intermediate** — a typed object that lives in state and is consumed by the next node. This is a common pattern: use structured output wherever you need to convert messy input into a reliable shape for downstream logic.

---

## Node 2 — `fetch_papers`: external API call

```python
def fetch_papers(state: State) -> dict:
    r = state["researcher"]
    author_name = f"{r.first_name} {r.last_name}"

    query = f'author:"{author_name}"'
    if r.institution:
        query += f" {r.institution}"

    resp = requests.get(
        _SERPAPI_URL,
        params={
            "engine":  "google_scholar",
            "q":       query,
            "num":     10,
            "api_key": os.environ["SERPAPI_API_KEY"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("organic_results", [])

    if not results:
        return {"author_name": author_name, "papers": []}

    papers = []
    for item in results:
        summary = item.get("publication_info", {}).get("summary", "")
        year_match = re.search(r"\b(19|20)\d{2}\b", summary)
        year = int(year_match.group()) if year_match else None

        papers.append({
            "title":    item.get("title", "Unknown title"),
            "year":     year,
            "abstract": item.get("snippet"),
            "url":      item.get("link"),
        })

    papers.sort(key=lambda p: p["year"] or 0, reverse=True)
    return {"author_name": author_name, "papers": papers}
```

This node does **not** call an LLM. It is pure I/O: one HTTP call, JSON parsing, data normalization. This is the "R" in RAG.

### Single API call

SerpAPI handles the Google Scholar search in one request. The parsed `ResearcherQuery` from the previous node is used directly to build the query string — no intermediate ID resolution needed.

The `author:"Name"` operator in the query restricts results to papers where the name appears in the author list. If `institution` was extracted, it is appended to narrow ambiguous names (e.g. two researchers named "John Smith").

### API used: SerpAPI Google Scholar

```
https://serpapi.com/search?engine=google_scholar&q=author:"Name"&num=10&api_key=...
```

SerpAPI is a paid scraping API with a free tier of 100 searches/month. A free-to-use key is already in the code. 

The response contains `organic_results`, each with:

| Field | Content |
|---|---|
| `title` | Paper title |
| `link` | URL to the paper |
| `snippet` | Short abstract-like excerpt |
| `publication_info.summary` | Authors, venue, year as a string — e.g. `"Y LeCun - Nature, 2015"` |

Year is not a dedicated field — it is extracted from `publication_info.summary` with a regex that matches four-digit year patterns (`\b(19|20)\d{2}\b`).

### Error handling in nodes

`raise_for_status()` converts HTTP error codes (401 invalid key, 429 rate limit, 500 server error) into Python exceptions. If the search returns no results, the node returns an empty `papers` list — `summarize` handles that case gracefully instead of crashing.

For production code you would add retry logic and catch `requests.exceptions.Timeout` separately. For this educational example the explicit empty-list branch is enough.

---

## Node 3 — `summarize`: augmented generation

```python
def summarize(state: State) -> dict:
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
```

This is the "A" and "G" in RAG together:

**Augment** — `papers_text` is constructed from `state["papers"]`. The LLM prompt contains the actual titles and abstracts retrieved from the API. The model does not need to know who the researcher is from training — it reads the papers directly.

**Generate** — `summary_llm.invoke(...)` runs the LLM over the augmented prompt and produces a prose summary grounded in the retrieved content.

### Why this is better than just asking the LLM

```python
# Without RAG — the LLM uses training memory:
llm.invoke("What has Yann LeCun published recently?")
# → May be outdated, hallucinated titles, wrong years

# With RAG — the LLM uses retrieved data:
llm.invoke([system_prompt, HumanMessage(content=f"Papers:\n{papers_text}")])
# → Grounded in actual API results from right now
```

The model's job is to **reason and summarize**, not to **recall facts**. RAG separates these two responsibilities.

---

## Two LLM instances

```python
parse_llm   = ChatOllama(model="qwen2.5:7b", temperature=0).with_structured_output(ResearcherQuery)
summary_llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
```

Both use the same underlying model, but are configured differently:

| Instance | temperature | purpose |
|---|---|---|
| `parse_llm` | 0 | Deterministic extraction — always produce the same structured output for the same input |
| `summary_llm` | 0.3 | Slight variation allowed — summaries can differ slightly in phrasing without breaking anything |

This is a pattern you will use often: same model, different temperature per task.

You can also use different models per node — a small fast model for parsing, a larger smarter model for generation. The graph does not care.

---

## Accessing the full result

After `.invoke()` the entire populated state is returned:

```python
result = app.invoke({"raw_query": "Andrej Karpathy", "researcher": None, ...})

result["researcher"]   # ResearcherQuery(first_name="Andrej", last_name="Karpathy", institution=None)
result["author_name"]  # "Andrej Karpathy"  (canonical name from API)
result["papers"]       # list of 10 dicts — title, year, abstract, url
result["response"]     # str — the LLM's prose summary
```

Every intermediate result is preserved in state. You do not have to choose what to keep — everything is there if you need it for debugging, logging, or downstream use.

---

## State initialization

You must pass all state keys when calling `.invoke()`:

```python
initial_state: State = {
    "raw_query":   "Yann LeCun",
    "researcher":  None,
    "author_name": "",
    "papers":      [],
    "response":    "",
}
result = app.invoke(initial_state)
```

Fields that nodes have not written yet need placeholder values. `None` for objects, `""` for strings, `[]` for lists. Missing keys will cause `KeyError` inside nodes that read them before writing.

---

## Building the graph step by step

### Step 1 — Design state first

Identify what data flows through the pipeline:

- What does the caller provide? → `raw_query`
- What does parse produce? → `researcher`
- What does fetch produce? → `author_name`, `papers`
- What does generate produce? → `response`

State design drives node design. If you cannot name every field cleanly, the node boundaries may be wrong.

### Step 2 — Write nodes independently

Each node only reads its inputs and writes its outputs. `fetch_papers` does not care how `researcher` was produced — it just reads `state["researcher"]`. This makes nodes independently testable:

```python
# Test fetch_papers in isolation:
test_state = {
    "raw_query": "",
    "researcher": ResearcherQuery(first_name="Yann", last_name="LeCun"),
    "author_name": "",
    "papers": [],
    "response": "",
}
result = fetch_papers(test_state)
assert len(result["papers"]) > 0
```

### Step 3 — Wire the graph

```python
g.add_edge(START, "parse_query")
g.add_edge("parse_query", "fetch_papers")
g.add_edge("fetch_papers", "summarize")
g.add_edge("summarize", END)
```

Edge order defines execution order. LangGraph validates that every node is reachable and every non-END node has an outgoing edge.

### Step 4 — Invoke with full initial state

Pass all keys. Read from `result` what you need.

---

## Comparison with previous examples

| Feature | Basics chatbot | StructuredOutput | SimpleRAG |
|---|---|---|---|
| Nodes | 1 | 1 | 3 |
| State fields | messages (list) | review_text, parsed | 5 separate typed fields |
| Reducer | `add_messages` | none | none |
| LLM calls | 1 per turn | 1 | 2 (parse + summarize) |
| External I/O | none | none | HTTP API |
| Memory | short-term (history) | none | none |
| Structured output | no | final output | intermediate (parse node) |
| RAG | no | no | yes |

---

## Extending this graph

Common next steps from here:

**Add a validation node** — after `fetch_papers`, add a node that checks `state["papers"]` is non-empty and the author name matches what was searched. If not, return an error early.

**Add conditional routing** — if `papers` is empty, route to an `error` node instead of `summarize`. This requires `add_conditional_edges` — covered in the Workflow examples.

**Swap the retrieval source** — replace `fetch_papers` with a node that calls arXiv, PubMed, or a local vector database. The `parse_query` and `summarize` nodes are untouched.

**Add a second retrieval node** — after `summarize`, add a node that fetches citation counts or co-author networks. Fan out from one node to two, fan back in before the final response.

These are all graph topology changes. The nodes themselves stay simple and single-purpose.
