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

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# User Question
question = input("Ask a question about the PDF: ")

# Retrieve relevant chunks
docs = retriever.invoke(question)

# Combine retrieved text
context = "\n\n".join([doc.page_content for doc in docs])

# Prompt
prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# Ask Groq
response = llm.invoke(prompt)

print("\n🤖 AI Answer:\n")
print(response.content)