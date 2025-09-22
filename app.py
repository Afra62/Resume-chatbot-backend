from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# -------- Load Resume --------
resume_path = r"c:\Users\User\Downloads\AI engineer Afra Irfan (1).pdf"

with open(resume_path, "rb") as f:
    pdf_reader = PyPDF2.PdfReader(f)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text() + "\n"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(resume_text)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_texts(texts, embedding=embeddings)

# -------- Question Answering --------
@app.post("/ask")
def ask_question(data: Question):
    docs = vector_store.similarity_search(data.question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")

    prompt = f"""
You are Afra Irfan, answering as yourself (not as an AI).
Use ONLY the information from this resume to answer questions:

{context}

Rules:
1. If the question is completely unrelated to your resume (like politics, government, weather, movies, drama, celebrity news, sports, random facts), politely say:
   "I’m here to discuss my professional background, education, and experience only."
2. If the question is related, answer naturally as if you are Afra, not an assistant.
3. Be conversational, avoid robotic or formal chatbot tone.
4. If greeting (hi, hello, how are you), respond politely and keep the flow.

Question: {data.question}
"""

    response = llm([HumanMessage(content=prompt)])
    return {"answer": response.content}

