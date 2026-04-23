import os
import re
import pdfplumber
import pytesseract
import hashlib
import numpy as np
from PIL import Image
from datetime import datetime
from docx import Document
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import faiss
import requests
from typing import Dict, Optional, List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
#done changes 
# ================================================================
# CONFIGURATION
# ================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "Add_Your_Groq_API_Key_Here")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "sqlite:///talktonic.db"
DEFAULT_TOP_K = 4
DEFAULT_WIDE_K = 12

app = FastAPI(
    title="TalkTonic API v3",
    description="High-performance RAG Chatbot using FAISS + Groq LLM",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# DATABASE SETUP
# ================================================================
Base = declarative_base()
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String(64), nullable=True)
    user_message = Column(Text)
    bot_reply = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ================================================================
# EMBEDDING MODEL
# ================================================================
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(vecs)
    return vecs.astype("float32")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])

# ================================================================
# TEXT & FILE HANDLING
# ================================================================
def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_file(uploaded_file: UploadFile) -> str:
    """Extract text from PDF, Image, DOCX, or TXT files."""
    try:
        if uploaded_file.content_type == "application/pdf":
            text = ""
            with pdfplumber.open(uploaded_file.file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if not t:
                        img = page.to_image(resolution=200).original
                        t = pytesseract.image_to_string(img)
                    text += (t or "") + "\n"
            return clean_text(text)

        elif uploaded_file.content_type.startswith("image/"):
            img = Image.open(uploaded_file.file)
            return clean_text(pytesseract.image_to_string(img))

        elif uploaded_file.content_type == "text/plain":
            return clean_text(uploaded_file.file.read().decode("utf-8"))

        elif uploaded_file.content_type.endswith("officedocument.wordprocessingml.document"):
            doc = Document(uploaded_file.file)
            return clean_text("\n".join(p.text for p in doc.paragraphs))

        raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

def chunk_text(text: str, chunk_words: int = 380, overlap: int = 60) -> List[str]:
    """Sentence-based chunking with overlap for better RAG recall."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, length = [], [], 0
    for s in sentences:
        words = s.split()
        if length + len(words) > chunk_words and current:
            chunks.append(" ".join(current).strip())
            tail = " ".join(current).split()[-overlap:]
            current = [(" ".join(tail) + " " + s).strip()]
            length = len(current[0].split())
        else:
            current.append(s)
            length += len(words)
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]

# ================================================================
# FAISS RAG MEMORY
# ================================================================
class RAGMemory:
    """Holds FAISS index and chunk map for each document."""
    def __init__(self):
        self.index = None
        self.chunk_map: Dict[int, str] = {}

    def build_index(self, chunks: List[str]):
        vectors = embed_texts(chunks)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self.chunk_map = {i: c for i, c in enumerate(chunks)}

    def retrieve(self, query: str, top_k=DEFAULT_WIDE_K):
        if not self.index:
            raise HTTPException(status_code=400, detail="No document context loaded.")
        qv = embed_query(query)
        D, I = self.index.search(qv, top_k)
        return [(i, float(D[0][n])) for n, i in enumerate(I[0]) if i in self.chunk_map]

def rerank(query: str, candidate_ids: List[int], chunk_map: Dict[int, str], top_k=DEFAULT_TOP_K) -> List[int]:
    """Semantic reranking of top FAISS hits."""
    if not candidate_ids:
        return []
    texts = [query] + [chunk_map[i] for i in candidate_ids]
    vecs = embed_texts(texts)
    qv, cvs = vecs[0], vecs[1:]
    scores = [float(np.dot(qv, cv)) for cv in cvs]
    pairs = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in pairs[:top_k]]

# ================================================================
# GROQ CLIENT
# ================================================================
class GroqClient:
    def __init__(self, api_key=GROQ_API_KEY):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def query(self, prompt: str) -> str:
        try:
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM error: {str(e)}"

llm_client = GroqClient()

# ================================================================
# UTILITIES
# ================================================================
def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_prompt(context: str, query: str) -> str:
    return (
        "You are a precise assistant. Use ONLY the context below. "
        "If the answer is not found, say 'I don’t find it in context.'\n\n"
        f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer precisely:"
    )

def limit_context(chunks: List[str], max_chars=12000) -> str:
    combined, size = [], 0
    for c in chunks:
        if size + len(c) > max_chars:
            break
        combined.append(c)
        size += len(c)
    return "\n\n".join(combined)

# ================================================================
# CHAT MEMORY PERSISTENCE
# ================================================================
def save_message(file_hash: Optional[str], user: str, bot: str):
    record = ChatHistory(file_hash=file_hash, user_message=user, bot_reply=bot)
    session.add(record)
    session.commit()

def get_history(file_hash: Optional[str] = None, limit=20):
    q = session.query(ChatHistory)
    if file_hash:
        q = q.filter(ChatHistory.file_hash == file_hash)
    q = q.order_by(ChatHistory.id.desc()).limit(limit)
    return [
        {"user": r.user_message, "bot": r.bot_reply, "time": r.timestamp.isoformat()}
        for r in q
    ]

# ================================================================
# MEMORY STORE
# ================================================================
memory_store: Dict[str, RAGMemory] = {}

# ================================================================
# REQUEST MODELS
# ================================================================
class ChatInput(BaseModel):
    message: str
    file_hash: Optional[str] = None

# ================================================================
# ENDPOINTS
# ================================================================
@app.post("/upload")
async def upload_file(file: UploadFile):
    raw_text = extract_text_from_file(file)
    text = clean_text(raw_text)
    chunks = chunk_text(text)

    file_hash = compute_hash(text)
    rag = RAGMemory()
    rag.build_index(chunks)
    memory_store[file_hash] = rag

    return {
        "file_hash": file_hash,
        "chunks": len(chunks),
        "message": "Document processed successfully."
    }

@app.post("/chat")
async def chat(input: ChatInput):
    query = input.message.strip()
    file_hash = input.file_hash

    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    try:
        if file_hash and file_hash in memory_store:
            rag = memory_store[file_hash]
            candidates = rag.retrieve(query)
            candidate_ids = [i for i, _ in candidates]
            top_ids = rerank(query, candidate_ids, rag.chunk_map)
            top_chunks = [rag.chunk_map[i] for i in top_ids]
            context = limit_context(top_chunks)
            prompt = build_prompt(context, query)
        else:
            prompt = query

        response = llm_client.query(prompt)
        save_message(file_hash, query, response)
        return {"response": response, "history": get_history(file_hash)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def history(file_hash: Optional[str] = None):
    return {"history": get_history(file_hash)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": len(memory_store)}
