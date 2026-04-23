# Author: Daksh Sharma 26434

import hashlib
import io
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
from bson import ObjectId
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from PIL import Image
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi


APP_NAME = "NexusRAG"
APP_VERSION = "1.0.0"
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
FILE_STORAGE_DIR = Path(os.getenv("FILE_STORAGE_DIR", "storage/uploads"))
DEFAULT_WORKSPACE_NAME = "Flagship Workspace"
DEFAULT_TOP_K = 6
MAX_CONTEXT_CHARS = 14000
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


app_state: Dict[str, Any] = {
    "mongo_client": None,
    "database": None,
    "qdrant": None,
}


@dataclass
class SourcePage:
    page_number: int
    text: str


@dataclass
class ChunkPayload:
    text: str
    page_start: int
    page_end: int


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    description: str = ""


class ChatRequest(BaseModel):
    workspace_id: str
    message: str = Field(min_length=1)
    document_id: Optional[str] = None


def utc_now() -> datetime:
    return datetime.utcnow()


def normalize_text(value: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename).strip("-")
    return cleaned or "upload"


def object_id(value: str, field_name: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}.")
    return ObjectId(value)


def serialize_id(document: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(document)
    if "_id" in output:
        output["id"] = str(output.pop("_id"))
    if "workspace_id" in output and isinstance(output["workspace_id"], ObjectId):
        output["workspace_id"] = str(output["workspace_id"])
    if "document_id" in output and isinstance(output["document_id"], ObjectId):
        output["document_id"] = str(output["document_id"])
    return output


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embedder():
    from sentence_transformers import SentenceTransformer

    if "embedder" not in app_state:
        app_state["embedder"] = SentenceTransformer(EMBED_MODEL_NAME)
    return app_state["embedder"]


def get_reranker():
    from sentence_transformers import CrossEncoder

    if "reranker" not in app_state:
        try:
            app_state["reranker"] = CrossEncoder(RERANK_MODEL_NAME)
        except Exception:
            app_state["reranker"] = None
    return app_state["reranker"]


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    vectors = get_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [normalize_text(sentence) for sentence in sentences if normalize_text(sentence)]


def build_semantic_chunks(
    pages: List[SourcePage],
    max_words: int = 240,
    overlap_words: int = 50,
) -> List[ChunkPayload]:
    chunks: List[ChunkPayload] = []
    buffer: List[str] = []
    buffer_pages: List[int] = []
    word_count = 0

    for page in pages:
        for sentence in split_sentences(page.text):
            sentence_words = sentence.split()
            if not sentence_words:
                continue
            if buffer and word_count + len(sentence_words) > max_words:
                combined = " ".join(buffer).strip()
                chunks.append(
                    ChunkPayload(
                        text=combined,
                        page_start=min(buffer_pages),
                        page_end=max(buffer_pages),
                    )
                )
                overlap = combined.split()[-overlap_words:]
                buffer = [" ".join(overlap)] if overlap else []
                buffer_pages = [buffer_pages[-1]] if buffer_pages else []
                word_count = len(overlap)
            buffer.append(sentence)
            buffer_pages.append(page.page_number)
            word_count += len(sentence_words)

    if buffer:
        chunks.append(
            ChunkPayload(
                text=" ".join(buffer).strip(),
                page_start=min(buffer_pages),
                page_end=max(buffer_pages),
            )
        )

    return [chunk for chunk in chunks if chunk.text]


def extract_structured_dataframe(file_name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(payload))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(payload))
    raise HTTPException(status_code=400, detail="Unsupported structured file.")


def dataframe_to_context(frame: pd.DataFrame) -> List[SourcePage]:
    head_markdown = frame.head(12).to_markdown(index=False)
    numeric_summary = ""
    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty:
        numeric_summary = numeric.describe().round(2).to_markdown()

    text = (
        "Structured dataset preview:\n"
        f"{head_markdown}\n\n"
        f"Columns: {', '.join(frame.columns.astype(str).tolist())}\n\n"
    )
    if numeric_summary:
        text += f"Numeric summary:\n{numeric_summary}\n"
    return [SourcePage(page_number=1, text=text)]


def extract_pages_from_upload(file_name: str, content_type: str, payload: bytes) -> List[SourcePage]:
    try:
        if content_type == "application/pdf" or file_name.lower().endswith(".pdf"):
            pages: List[SourcePage] = []
            with pdfplumber.open(io.BytesIO(payload)) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not normalize_text(text):
                        image = page.to_image(resolution=200).original
                        text = pytesseract.image_to_string(image)
                    pages.append(SourcePage(page_number=index, text=normalize_text(text)))
            return [page for page in pages if page.text]

        if content_type.startswith("image/"):
            image = Image.open(io.BytesIO(payload))
            text = pytesseract.image_to_string(image)
            return [SourcePage(page_number=1, text=normalize_text(text))]

        if content_type == "text/plain" or file_name.lower().endswith(".txt"):
            return [SourcePage(page_number=1, text=normalize_text(payload.decode("utf-8")))]

        if file_name.lower().endswith(".docx"):
            document = Document(io.BytesIO(payload))
            paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return [SourcePage(page_number=1, text=normalize_text(paragraphs))]

        if file_name.lower().endswith((".csv", ".xlsx", ".xls")):
            dataframe = extract_structured_dataframe(file_name, payload)
            return dataframe_to_context(dataframe)

        raise HTTPException(status_code=400, detail="Unsupported file type.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File processing failed: {exc}") from exc


async def ensure_qdrant_collection() -> None:
    qdrant: QdrantClient = app_state["qdrant"]
    collections = qdrant.get_collections().collections
    existing = {collection.name for collection in collections}
    if QDRANT_COLLECTION in existing:
        return

    sample_vector_size = embed_texts(["NexusRAG bootstrap"]).shape[1]
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=sample_vector_size,
            distance=models.Distance.COSINE,
        ),
    )


async def get_database() -> AsyncIOMotorDatabase:
    database = app_state.get("database")
    if database is None:
        raise HTTPException(status_code=500, detail="Database connection not initialized.")
    return database


async def get_or_create_workspace(
    database: AsyncIOMotorDatabase,
    workspace_id: Optional[str],
    workspace_name: Optional[str],
) -> Dict[str, Any]:
    workspaces = database["workspaces"]

    if workspace_id:
        workspace = await workspaces.find_one({"_id": object_id(workspace_id, "workspace_id")})
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found.")
        return workspace

    target_name = normalize_text(workspace_name or DEFAULT_WORKSPACE_NAME)
    workspace = await workspaces.find_one({"name": target_name})
    if workspace:
        return workspace

    payload = {
        "name": target_name,
        "description": "Primary workspace for multimodal document research.",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    result = await workspaces.insert_one(payload)
    payload["_id"] = result.inserted_id
    return payload


async def store_document_chunks(
    database: AsyncIOMotorDatabase,
    workspace_id: ObjectId,
    document_id: ObjectId,
    chunks: List[ChunkPayload],
) -> None:
    chunk_records = []
    for index, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "workspace_id": workspace_id,
                "document_id": document_id,
                "chunk_index": index,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "created_at": utc_now(),
            }
        )

    if chunk_records:
        await database["chunks"].insert_many(chunk_records)

    qdrant: QdrantClient = app_state["qdrant"]
    embeddings = embed_texts([chunk.text for chunk in chunks])
    points = []
    for index, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=str(document_id) + f"-{index}",
                vector=embeddings[index].tolist(),
                payload={
                    "workspace_id": str(workspace_id),
                    "document_id": str(document_id),
                    "chunk_index": index,
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
            )
        )

    if points:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)


async def hybrid_retrieve(
    database: AsyncIOMotorDatabase,
    workspace_id: str,
    query: str,
    document_id: Optional[str] = None,
    limit: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    qdrant: QdrantClient = app_state["qdrant"]
    query_vector = embed_texts([query])[0].tolist()
    must_conditions = [
        models.FieldCondition(
            key="workspace_id",
            match=models.MatchValue(value=workspace_id),
        )
    ]
    if document_id:
        must_conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )

    dense_hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=models.Filter(must=must_conditions),
        limit=limit * 3,
        with_payload=True,
    )

    mongo_filter: Dict[str, Any] = {"workspace_id": object_id(workspace_id, "workspace_id")}
    if document_id:
        mongo_filter["document_id"] = object_id(document_id, "document_id")

    chunk_documents = await database["chunks"].find(mongo_filter).to_list(length=2500)
    if not chunk_documents:
        return []

    tokenized_corpus = [document["text"].lower().split() for document in chunk_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    sparse_scores = bm25.get_scores(query.lower().split())
    sparse_top = np.argsort(sparse_scores)[::-1][: limit * 3]

    merged: Dict[str, Dict[str, Any]] = {}

    for hit in dense_hits:
        payload = hit.payload or {}
        point_id = str(hit.id)
        merged[point_id] = {
            "point_id": point_id,
            "text": payload.get("text", ""),
            "document_id": payload.get("document_id"),
            "page_start": payload.get("page_start", 1),
            "page_end": payload.get("page_end", 1),
            "dense_score": float(hit.score),
            "sparse_score": 0.0,
        }

    for index in sparse_top:
        document = chunk_documents[int(index)]
        point_id = f"{document['document_id']}-{document['chunk_index']}"
        entry = merged.setdefault(
            point_id,
            {
                "point_id": point_id,
                "text": document["text"],
                "document_id": str(document["document_id"]),
                "page_start": document["page_start"],
                "page_end": document["page_end"],
                "dense_score": 0.0,
                "sparse_score": 0.0,
            },
        )
        entry["sparse_score"] = float(sparse_scores[int(index)])

    candidates = list(merged.values())
    if not candidates:
        return []

    reranker = get_reranker()
    if reranker:
        scores = reranker.predict([(query, candidate["text"]) for candidate in candidates])
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    else:
        for candidate in candidates:
            candidate["hybrid_score"] = candidate["dense_score"] + candidate["sparse_score"]
        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return candidates[:limit]


def build_context(candidates: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    total_chars = 0
    for index, candidate in enumerate(candidates, start=1):
        section = (
            f"[Source {index} | pages {candidate['page_start']}-{candidate['page_end']}]\n"
            f"{candidate['text']}"
        )
        if total_chars + len(section) > MAX_CONTEXT_CHARS:
            break
        sections.append(section)
        total_chars += len(section)
    return "\n\n".join(sections)


def build_prompt(user_message: str, context: str) -> str:
    return (
        "You are NexusRAG, an expert research copilot. "
        "Answer only from the retrieved context when a document context is provided. "
        "If the answer is partially supported, say so clearly. "
        "Cite useful evidence inline using [Source n]. "
        "Use markdown for tables or code when it improves clarity.\n\n"
        f"Retrieved Context:\n{context or 'No retrieved context supplied.'}\n\n"
        f"User Request:\n{user_message}"
    )


async def fetch_completion(prompt: str, stream: bool) -> Any:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": stream,
    }

    client = httpx.AsyncClient(timeout=90.0)
    if stream:
        return client, client.stream("POST", url, headers=headers, json=payload)

    try:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    finally:
        await client.aclose()


async def persist_message(
    database: AsyncIOMotorDatabase,
    workspace_id: str,
    user_message: str,
    assistant_message: str,
    citations: List[Dict[str, Any]],
    document_id: Optional[str] = None,
) -> None:
    await database["messages"].insert_one(
        {
            "workspace_id": object_id(workspace_id, "workspace_id"),
            "document_id": object_id(document_id, "document_id") if document_id else None,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "citations": citations,
            "created_at": utc_now(),
        }
    )


async def stream_chat_response(request: ChatRequest) -> AsyncIterator[str]:
    database = await get_database()
    citations = await hybrid_retrieve(database, request.workspace_id, request.message, request.document_id)
    context = build_context(citations)
    prompt = build_prompt(request.message, context)

    serializable_citations = [
        {
            "document_id": citation["document_id"],
            "page_start": citation["page_start"],
            "page_end": citation["page_end"],
            "excerpt": citation["text"][:240],
        }
        for citation in citations
    ]

    yield f"data: {json.dumps({'type': 'context', 'citations': serializable_citations})}\n\n"

    full_text = ""
    client, stream_context = await fetch_completion(prompt, stream=True)
    try:
        async with stream_context as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if not delta:
                    continue
                full_text += delta
                yield f"data: {json.dumps({'type': 'assistant_chunk', 'content': delta})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return
    finally:
        await client.aclose()

    await persist_message(
        database,
        workspace_id=request.workspace_id,
        document_id=request.document_id,
        user_message=request.message,
        assistant_message=full_text,
        citations=serializable_citations,
    )
    yield f"data: {json.dumps({'type': 'assistant_done'})}\n\n"


@asynccontextmanager
async def lifespan(_: FastAPI):
    FILE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    app_state["mongo_client"] = AsyncIOMotorClient(MONGODB_URI)
    app_state["database"] = app_state["mongo_client"][MONGODB_DB]
    app_state["qdrant"] = QdrantClient(url=QDRANT_URL)
    await ensure_qdrant_collection()
    yield
    if app_state.get("mongo_client") is not None:
        app_state["mongo_client"].close()


app = FastAPI(
    title=f"{APP_NAME} API",
    description="Workspace-native hybrid RAG platform with MongoDB, Qdrant, and SSE streaming.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    database = await get_database()
    workspace_count = await database["workspaces"].count_documents({})
    document_count = await database["documents"].count_documents({})
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "workspace_count": workspace_count,
        "document_count": document_count,
        "dependencies": {
            "mongodb": MONGODB_URI,
            "qdrant": QDRANT_URL,
            "redis": REDIS_URL,
        },
    }


@app.get("/workspaces")
async def list_workspaces() -> Dict[str, Any]:
    database = await get_database()
    workspaces = await database["workspaces"].find().sort("updated_at", -1).to_list(length=100)
    return {"items": [serialize_id(workspace) for workspace in workspaces]}


@app.post("/workspaces")
async def create_workspace(payload: WorkspaceCreate) -> Dict[str, Any]:
    database = await get_database()
    workspace = await get_or_create_workspace(database, None, payload.name)
    if payload.description and workspace.get("description") != payload.description:
        await database["workspaces"].update_one(
            {"_id": workspace["_id"]},
            {"$set": {"description": payload.description, "updated_at": utc_now()}},
        )
        workspace["description"] = payload.description
        workspace["updated_at"] = utc_now()
    return serialize_id(workspace)


@app.get("/workspaces/{workspace_id}/documents")
async def list_documents(workspace_id: str) -> Dict[str, Any]:
    database = await get_database()
    documents = await database["documents"].find(
        {"workspace_id": object_id(workspace_id, "workspace_id")}
    ).sort("created_at", -1).to_list(length=200)
    return {"items": [serialize_id(document) for document in documents]}


@app.get("/workspaces/{workspace_id}/messages")
async def list_messages(workspace_id: str) -> Dict[str, Any]:
    database = await get_database()
    messages = await database["messages"].find(
        {"workspace_id": object_id(workspace_id, "workspace_id")}
    ).sort("created_at", 1).to_list(length=200)

    items = []
    for message in messages:
        item = serialize_id(message)
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat()
        items.append(item)

    return {"items": items}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(default=None),
    workspace_name: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    database = await get_database()
    workspace = await get_or_create_workspace(database, workspace_id, workspace_name)
    raw_payload = await file.read()
    file_name = sanitize_filename(file.filename or "upload")
    pages = extract_pages_from_upload(file_name, file.content_type or "", raw_payload)
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found in document.")

    combined_text = "\n".join(page.text for page in pages)
    file_hash = compute_hash(combined_text)
    chunks = build_semantic_chunks(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks.")

    document_record = {
        "workspace_id": workspace["_id"],
        "name": file_name,
        "mime_type": file.content_type or "application/octet-stream",
        "file_hash": file_hash,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    insert_result = await database["documents"].insert_one(document_record)
    document_id = insert_result.inserted_id

    storage_dir = FILE_STORAGE_DIR / str(workspace["_id"])
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{document_id}-{file_name}"
    storage_path.write_bytes(raw_payload)

    await database["documents"].update_one(
        {"_id": document_id},
        {"$set": {"storage_path": str(storage_path), "updated_at": utc_now()}},
    )

    await store_document_chunks(database, workspace["_id"], document_id, chunks)
    await database["workspaces"].update_one(
        {"_id": workspace["_id"]},
        {"$set": {"updated_at": utc_now()}},
    )

    document_record["_id"] = document_id
    document_record["storage_path"] = str(storage_path)
    return {
        "workspace": serialize_id(workspace),
        "document": serialize_id(document_record),
        "message": "Document indexed successfully.",
    }


@app.get("/documents/{document_id}/content")
async def get_document_content(document_id: str) -> FileResponse:
    database = await get_database()
    document = await database["documents"].find_one({"_id": object_id(document_id, "document_id")})
    if not document or not document.get("storage_path"):
        raise HTTPException(status_code=404, detail="Document file not found.")

    path = Path(document["storage_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stored file missing on disk.")

    return FileResponse(path=path, media_type=document.get("mime_type"), filename=document.get("name"))


@app.post("/chat")
async def chat_once(request: ChatRequest) -> Dict[str, Any]:
    database = await get_database()
    citations = await hybrid_retrieve(database, request.workspace_id, request.message, request.document_id)
    context = build_context(citations)
    prompt = build_prompt(request.message, context)
    answer = await fetch_completion(prompt, stream=False)
    serializable_citations = [
        {
            "document_id": citation["document_id"],
            "page_start": citation["page_start"],
            "page_end": citation["page_end"],
            "excerpt": citation["text"][:240],
        }
        for citation in citations
    ]
    await persist_message(
        database,
        workspace_id=request.workspace_id,
        document_id=request.document_id,
        user_message=request.message,
        assistant_message=answer,
        citations=serializable_citations,
    )
    return {
        "response": answer,
        "citations": serializable_citations,
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
