import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sqlite3

# Create or open database
conn = sqlite3.connect("memory.db")

# Create table
conn.execute("""
CREATE TABLE IF NOT EXISTS facts (
    key TEXT,
    value TEXT
)
""")

conn.commit()

print("✅ Database Ready!")
# Load Environment Variables
load_dotenv()

# Initialize Groq
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("✅ Groq Connected")

# Save information into the database
def remember(key, value):
    conn.execute(
        "DELETE FROM facts WHERE key = ?",
        (key,)
    )

    conn.execute(
        "INSERT INTO facts (key, value) VALUES (?, ?)",
        (key, value)
    )

    conn.commit()

    print(f"✅ Saved: {key} = {value}")
    print(f"✅ Saved: {key} = {value}")

# Retrieve information from the database
def recall(key):
    cursor = conn.execute(
        "SELECT value FROM facts WHERE key = ? ORDER BY rowid DESC LIMIT 1",
        (key,)
    )

    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        return "Not Found"

def ask_ai(question):
    name = recall("name")
    subject = recall("favourite_subject")

    prompt = f"""
You are a helpful AI assistant.

User Memory:
Name: {name}
Favourite Subject: {subject}

Answer the user's question using the stored memory whenever it is relevant.

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content

# Store some memory
remember("name", "Mansoor")
remember("favourite_subject", "Artificial Intelligence")

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("👋 Goodbye!")
        break

    answer = ask_ai(question)

    print("\n🤖 AI:")
    print(answer)