import os
import re
import hashlib
import json
import numpy as np
import faiss
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# -------------------------- CONFIG --------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "Add_Your_Groq_API_Key_Here")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4
DEFAULT_WIDE_K = 12
MAX_CONTEXT_CHARS = 12000
EMBED_BATCH_SIZE = 24
SAVE_DIR = Path("./rag_store")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------- EMBEDDING MODEL --------------------------
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)

def _normalize(vecs: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vecs)
    return vecs

def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    model = get_embedder()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.append(vecs.astype("float32"))
    all_vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, 384), dtype="float32")
    return _normalize(all_vecs)

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])

# -------------------------- TEXT & FILE PROCESSING --------------------------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sentence_tokenize(text: str) -> List[str]:
    text = clean_text(text)
    return _SENT_SPLIT.split(text) if text else []

def chunk_text(text: str, chunk_words: int = 380, overlap_words: int = 60) -> List[str]:
    sentences = sentence_tokenize(text)
    chunks, cur, cur_len = [], [], 0
    for sent in sentences:
        words = sent.split()
        if cur_len + len(words) > chunk_words and cur:
            chunks.append(" ".join(cur).strip())
            tail = " ".join(cur).split()[-overlap_words:]
            cur = [(" ".join(tail) + " " + sent).strip()]
            cur_len = len(cur[0].split())
        else:
            cur.append(sent)
            cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]

def extract_text_from_file(file_path: str) -> str:
    """Extract text from file path (PDF, Image, DOCX, TXT)."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            text = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if not t:
                        img = page.to_image(resolution=200).original
                        t = pytesseract.image_to_string(img)
                    if t:
                        text.append(t)
            return clean_text("\n".join(text))

        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            img = Image.open(str(path))
            return clean_text(pytesseract.image_to_string(img))

        elif suffix == ".docx":
            doc = Document(str(path))
            return clean_text("\n".join([p.text for p in doc.paragraphs]))

        elif suffix in {".txt", ".md"}:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))

        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {e}")

# -------------------------- FAISS VECTOR STORE --------------------------
class VectorStore:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_map: Dict[int, str] = {}
        self._dim: int = 0

    @staticmethod
    def _make_index(dim: int) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(dim)

    def build(self, chunks: List[str]) -> None:
        if not chunks:
            self.index = None
            self.chunk_map = {}
            self._dim = 0
            return
        vecs = embed_texts(chunks)
        dim = vecs.shape[1]
        index = self._make_index(dim)
        index.add(vecs)
        self.index = index
        self.chunk_map = {i: c for i, c in enumerate(chunks)}
        self._dim = dim

    def search(self, query: str, top_k: int = DEFAULT_WIDE_K) -> List[Tuple[int, float]]:
        if not self.index:
            return []
        qv = embed_query(query)
        D, I = self.index.search(qv.astype("float32"), top_k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

# -------------------------- SEMANTIC RE-RANK --------------------------
def rerank(query: str, candidate_ids: List[int], chunk_map: Dict[int, str], top_k: int = DEFAULT_TOP_K) -> List[int]:
    if not candidate_ids:
        return []
    texts = [query] + [chunk_map[i] for i in candidate_ids]
    vecs = embed_texts(texts)
    qv, cvs = vecs[0], vecs[1:]
    scores = [float(np.dot(qv, c)) for c in cvs]
    pairs = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in pairs[:top_k]]

# -------------------------- PROMPT BUILDER --------------------------
def build_prompt(context: str, query: str) -> str:
    context = context.strip()
    return (
        "You are a precise assistant. Use ONLY the context below. "
        "If the answer is not found, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer concisely and cite short quotes from context if needed."
    )

def limit_context(chunks: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    combined = []
    total = 0
    for c in chunks:
        if total + len(c) > max_chars:
            break
        combined.append(c)
        total += len(c)
    return "\n\n".join(combined)

# -------------------------- LLM CLIENT --------------------------
class GroqClient:
    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    def query(self, prompt: str, timeout: int = 60) -> str:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            r = requests.post(self.url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM error: {e}"

# -------------------------- MAIN MODEL --------------------------
class TalkTonicModel:
    """
    Core RAG model for TalkTonic backend.
    Used directly by /upload and /chat endpoints.
    """
    def __init__(self, api_key: str = GROQ_API_KEY):
        self.store = VectorStore()
        self.client = GroqClient(api_key)
        self.doc_hash: Optional[str] = None

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def index_text(self, text: str) -> str:
        text = clean_text(text)
        chunks = chunk_text(text)
        self.store.build(chunks)
        self.doc_hash = self._hash_text(text)
        return self.doc_hash

    def index_file(self, file_path: str) -> str:
        text = extract_text_from_file(file_path)
        return self.index_text(text)

    def retrieve(self, query: str, wide_k: int = DEFAULT_WIDE_K, top_k: int = DEFAULT_TOP_K) -> List[str]:
        hits = self.store.search(query, wide_k)
        candidate_ids = [i for i, _ in hits]
        top_ids = rerank(query, candidate_ids, self.store.chunk_map, top_k)
        return [self.store.chunk_map[i] for i in top_ids]

    def ask(self, query: str) -> str:
        if not query.strip():
            return "Empty query."
        if not self.store.index:
            # Fallback to direct LLM mode
            prompt = build_prompt("", query)
            return self.client.query(prompt)

        top_chunks = self.retrieve(query)
        context = limit_context(top_chunks)
        prompt = build_prompt(context, query)
        return self.client.query(prompt)
