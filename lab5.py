from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load API Key
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# Split PDF
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "local_files_only": True
    }
)

# Vector Database
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})


# Function to answer from PDF
def answer_from_pdf(question):
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY from the given context.

If the answer is not available, reply:
I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# Adaptive RAG
def adaptive_answer(question):
    answer = answer_from_pdf(question)

    if "I don't know" in answer:
        print("\nRetrying with improved query...\n")

        new_question = f"Explain in detail: {question}"

        answer = answer_from_pdf(new_question)

    return answer


while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    final_answer = adaptive_answer(question)

    print("\n🤖 Final Answer:\n")
    print(final_answer)