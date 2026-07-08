from utils.llm import llm
from langchain.tools import tool
from langchain.agents import create_agent
import wikipedia

# -----------------------------
# Wikipedia Tool
# -----------------------------
@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and return a summary.
    """
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Wikipedia Error: {e}"


# -----------------------------
# Calculator Tool
# -----------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculator Error: {e}"


# -----------------------------
# Create Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=[wikipedia_search, calculator],
    system_prompt="""
You are a helpful AI assistant.

Rules:
1. Use wikipedia_search whenever factual information is needed.
2. Use calculator whenever mathematical calculations are required.
3. If both are needed, use Wikipedia first, then Calculator.
"""
)

print("🤖 Wikipedia + Calculator Agent Ready!")
print("Type 'exit' to quit.\n")

# -----------------------------
# Chat Loop
# -----------------------------
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print("\n🤖 AI:")
    print(response["messages"][-1].content)
    print("-" * 60)