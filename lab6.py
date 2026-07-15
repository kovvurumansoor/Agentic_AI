import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# Load .env
load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "llama",
        "structured_output": True,
    },
)
# Writer Agent
writer = AssistantAgent(
    name="Writer",
    model_client=model_client,
    system_message="""
You are a professional blog writer.
Write clear, engaging, and informative blog posts.
"""
)
critic = AssistantAgent(
    name="Critic",
    model_client=model_client,
    system_message="""
You are a strict blog reviewer.
Review the writer's blog and provide constructive feedback.
Approve the blog only if it is clear, informative, and well-structured.
"""
)

async def main():
    topic = "Artificial Intelligence in Healthcare"

    # Writer writes the first draft
    writer_response = await writer.run(
        task=f"Write a blog on: {topic}"
    )

    blog = writer_response.messages[-1].content

    # Run 3 review rounds
    for i in range(3):
        print(f"\n========== ROUND {i+1} ==========\n")

        print("📝 Writer Draft:\n")
        print(blog)

        critic_response = await critic.run(
            task=f"Review this blog and provide suggestions:\n\n{blog}"
        )

        feedback = critic_response.messages[-1].content

        print("\n🔍 Critic Feedback:\n")
        print(feedback)

        writer_response = await writer.run(
            task=f"""
Improve the following blog using the critic's feedback.

BLOG:
{blog}

FEEDBACK:
{feedback}
"""
        )

        blog = writer_response.messages[-1].content

    print("\n==============================")
    print("✅ FINAL BLOG")
    print("==============================\n")
    print(blog)

asyncio.run(main())
