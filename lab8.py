import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("✅ Groq Connected")


# -----------------------------
# State
# -----------------------------
class TeamState(TypedDict):
    task: str
    worker_result: str
    summary: str


# -----------------------------
# Worker Node
# -----------------------------
def worker(state: TeamState):

    task = state["task"]

    response = llm.invoke(
        f"""
You are a Worker Agent.

Solve the following task carefully.

Task:
{task}
"""
    )

    return {
        "worker_result": response.content
    }


# -----------------------------
# Supervisor Node
# -----------------------------
def supervisor(state: TeamState):

    worker_result = state["worker_result"]

    response = llm.invoke(
        f"""
You are a Supervisor Agent.

The worker completed this task:

{worker_result}

Write a short summary of the result.
"""
    )

    return {
        "summary": response.content
    }


# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(TeamState)

builder.add_node("worker", worker)
builder.add_node("supervisor", supervisor)

builder.add_edge(START, "worker")
builder.add_edge("worker", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()


# -----------------------------
# Run
# -----------------------------
state = {
    "task": "What is 144 divided by 12, then plus 5?",
    "worker_result": "",
    "summary": ""
}

result = graph.invoke(state)

print("\n==============================")
print("WORKER RESULT")
print("==============================")
print(result["worker_result"])

print("\n==============================")
print("SUPERVISOR SUMMARY")
print("==============================")
print(result["summary"])