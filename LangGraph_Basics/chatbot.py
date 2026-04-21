"""
LangGraph Basics — Chatbot with Short-Term Memory
===================================================
Concepts demonstrated:

  1. Minimal graph        — a single node, two edges (START → chat → END)
  2. MessagesState        — built-in state type that holds the conversation
  3. add_messages reducer — LangGraph merges new messages instead of replacing
  4. Short-term memory    — conversation history passed on every turn

Graph shape:

    START ──► chat ──► END
"""

from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── 1. STATE ──────────────────────────────────────────────────────────────────
# The Annotated[list, add_messages] tells LangGraph to APPEND new messages
# to the list instead of overwriting it. Without this reducer, every node
# return would replace the whole messages list.

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. LLM ────────────────────────────────────────────────────────────────────

llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful, concise assistant. "
    "You remember what has been said earlier in the conversation."
))


# ── 3. NODE ───────────────────────────────────────────────────────────────────
# The only node in this graph. It receives the full message history,
# prepends the system prompt, calls the LLM, and returns the AI reply.
# LangGraph's add_messages reducer will append it to state["messages"].

def chat(state: State) -> dict:
    history = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(history)
    return {"messages": [response]}


# ── 4. GRAPH ──────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)

    g.add_node("chat", chat)

    g.add_edge(START, "chat")
    g.add_edge("chat", END)

    return g.compile()


# ── 5. MODULE-LEVEL APP ───────────────────────────────────────────────────────
# Must live outside __main__ so the VS Code extension can discover it.
# langgraph.json points to "chatbot.py:app".

app = build_graph()


# ── 6. CHAT LOOP ──────────────────────────────────────────────────────────────
# The STM lives here: `history` accumulates HumanMessages and AIMessages.
# Each .invoke() call passes the FULL history, so the LLM always has context.
# When you /exit and restart the script, the memory is gone — it is short-term.

if __name__ == "__main__":
    RST  = "\033[0m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    print("LangGraph Chatbot — type a message or /exit to quit.")
    print(f"{GRAY}(Memory resets when you restart the script){RST}\n")

    history: list[BaseMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye!")
            break

        history.append(HumanMessage(content=user_input))

        result = app.invoke({"messages": history})

        # result["messages"] is the full accumulated list after the reducer.
        # The last message is the AI reply we just got.
        ai_reply = result["messages"][-1]
        history = result["messages"]   # keep the full updated history

        print(f"{CYAN}Bot:{RST} {ai_reply.content}\n")
        print("─" * 55)
