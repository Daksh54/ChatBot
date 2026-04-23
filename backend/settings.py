# Author: Daksh Sharma 26434

import os
from pathlib import Path


APP_NAME = "NexusRAG"
APP_VERSION = "1.1.0"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "nexusrag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "workspace_chunks")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "document_ingestion")

FILE_STORAGE_DIR = Path(os.getenv("FILE_STORAGE_DIR", "storage/uploads"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

DEFAULT_WORKSPACE_NAME = "Flagship Workspace"
DEFAULT_TOP_K = 6
MAX_CONTEXT_CHARS = 14000
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "240"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "24"))
