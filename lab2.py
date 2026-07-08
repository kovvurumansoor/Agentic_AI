from utils.llm import llm
from langchain.tools import tool
from langchain.agents import create_agent


# -----------------------------
# Calculator Tool
# -----------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: 25*4
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# -----------------------------
# Create Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are a helpful AI assistant. Use the calculator tool whenever mathematical calculations are needed."
)

# -----------------------------
# Chat Loop
# -----------------------------
print("🤖 Agent Started!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("👋 Goodbye Buddy!")
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

    print("\n🤖 AI:", response["messages"][-1].content)
    print("-" * 50)