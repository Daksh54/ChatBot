# Author: Daksh Sharma 26434

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import numpy as np
from bson import ObjectId
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from redis import Redis
from rq import Queue

from jobs import process_document_task
from services import (
    embed_texts,
    ensure_qdrant_collection,
    get_reranker,
    sanitize_filename,
    utc_now,
)
from settings import (
    APP_NAME,
    APP_VERSION,
    CHAT_HISTORY_TURNS,
    DEFAULT_TOP_K,
    DEFAULT_WORKSPACE_NAME,
    FILE_STORAGE_DIR,
    FRONTEND_ORIGIN,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_CONTEXT_CHARS,
    MONGODB_DB,
    MONGODB_URI,
    QDRANT_COLLECTION,
    QDRANT_URL,
    REDIS_URL,
    TASK_QUEUE_NAME,
)


app_state: Dict[str, Any] = {
    "mongo_client": None,
    "database": None,
    "qdrant": None,
    "redis": None,
    "queue": None,
}


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    description: str = ""


class ChatRequest(BaseModel):
    workspace_id: str
    message: str = Field(min_length=1)
    document_id: Optional[str] = None


def object_id(value: str, field_name: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}.")
    return ObjectId(value)


def serialize_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return value


def serialize_document(document: Dict[str, Any]) -> Dict[str, Any]:
    output = {}
    for key, value in document.items():
        if key == "_id":
            output["id"] = str(value)
        else:
            output[key] = serialize_value(value)
    return output


async def get_database() -> AsyncIOMotorDatabase:
    database = app_state.get("database")
    if database is None:
        raise HTTPException(status_code=500, detail="Database connection not initialized.")
    return database


def get_queue() -> Queue:
    queue = app_state.get("queue")
    if queue is None:
        raise HTTPException(status_code=500, detail="Task queue not initialized.")
    return queue


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

    target_name = (workspace_name or DEFAULT_WORKSPACE_NAME).strip()
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


def build_system_prompt(context: str) -> str:
    return (
        "You are NexusRAG, an expert research copilot. "
        "Maintain conversational continuity using the prior chat history. "
        "When retrieved context is supplied, ground your answer in that evidence and cite it inline as [Source n]. "
        "If the context is incomplete, say so clearly instead of inventing details. "
        "Use markdown when it improves readability.\n\n"
        f"Retrieved Context:\n{context or 'No retrieved context supplied.'}"
    )


async def get_recent_history_messages(
    database: AsyncIOMotorDatabase,
    workspace_id: str,
    document_id: Optional[str],
    limit: int = CHAT_HISTORY_TURNS,
) -> List[Dict[str, str]]:
    query: Dict[str, Any] = {"workspace_id": object_id(workspace_id, "workspace_id")}
    if document_id:
        query["$or"] = [
            {"document_id": object_id(document_id, "document_id")},
            {"document_id": None},
        ]

    records = await database["messages"].find(query).sort("created_at", -1).limit(limit).to_list(length=limit)
    records.reverse()

    history: List[Dict[str, str]] = []
    for record in records:
        history.append({"role": "user", "content": record["user_message"]})
        history.append({"role": "assistant", "content": record["assistant_message"]})
    return history


def build_llm_messages(user_message: str, context: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(context)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


async def fetch_completion(messages: List[Dict[str, str]], stream: bool) -> Any:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
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
    history = await get_recent_history_messages(database, request.workspace_id, request.document_id)
    llm_messages = build_llm_messages(request.message, context, history)

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
    client, stream_context = await fetch_completion(llm_messages, stream=True)
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
    mongo_client = AsyncIOMotorClient(MONGODB_URI)
    qdrant = QdrantClient(url=QDRANT_URL)
    redis = Redis.from_url(REDIS_URL)

    ensure_qdrant_collection(qdrant)

    app_state["mongo_client"] = mongo_client
    app_state["database"] = mongo_client[MONGODB_DB]
    app_state["qdrant"] = qdrant
    app_state["redis"] = redis
    app_state["queue"] = Queue(name=TASK_QUEUE_NAME, connection=redis, default_timeout=3600)

    yield

    mongo_client.close()
    redis.close()


app = FastAPI(
    title=f"{APP_NAME} API",
    description="Workspace-native hybrid RAG platform with conversational memory, queued ingestion, and SSE streaming.",
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
    queued_tasks = await database["ingestion_tasks"].count_documents({"status": {"$in": ["QUEUED", "PROCESSING"]}})
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "workspace_count": workspace_count,
        "document_count": document_count,
        "queued_tasks": queued_tasks,
    }


@app.get("/workspaces")
async def list_workspaces() -> Dict[str, Any]:
    database = await get_database()
    workspaces = await database["workspaces"].find().sort("updated_at", -1).to_list(length=100)
    return {"items": [serialize_document(workspace) for workspace in workspaces]}


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
    return serialize_document(workspace)


@app.get("/workspaces/{workspace_id}/documents")
async def list_documents(workspace_id: str) -> Dict[str, Any]:
    database = await get_database()
    documents = await database["documents"].find(
        {"workspace_id": object_id(workspace_id, "workspace_id")}
    ).sort("created_at", -1).to_list(length=200)
    return {"items": [serialize_document(document) for document in documents]}


@app.get("/workspaces/{workspace_id}/messages")
async def list_messages(workspace_id: str) -> Dict[str, Any]:
    database = await get_database()
    messages = await database["messages"].find(
        {"workspace_id": object_id(workspace_id, "workspace_id")}
    ).sort("created_at", 1).to_list(length=200)
    return {"items": [serialize_document(message) for message in messages]}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(default=None),
    workspace_name: Optional[str] = Form(default=None),
) -> JSONResponse:
    database = await get_database()
    queue = get_queue()
    workspace = await get_or_create_workspace(database, workspace_id, workspace_name)

    raw_payload = await file.read()
    file_name = sanitize_filename(file.filename or "upload")
    document_record = {
        "workspace_id": workspace["_id"],
        "name": file_name,
        "mime_type": file.content_type or "application/octet-stream",
        "status": "QUEUED",
        "page_count": 0,
        "chunk_count": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "error": None,
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

    task_id = str(uuid.uuid4())
    task_record = {
        "_id": task_id,
        "workspace_id": str(workspace["_id"]),
        "document_id": str(document_id),
        "file_name": file_name,
        "mime_type": file.content_type or "application/octet-stream",
        "storage_path": str(storage_path),
        "status": "QUEUED",
        "progress": 0,
        "phase": "Queued for background indexing",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed_at": None,
        "error": None,
    }
    await database["ingestion_tasks"].insert_one(task_record)

    try:
        queue.enqueue(process_document_task, task_id, job_timeout=3600, result_ttl=3600)
    except Exception as exc:
        error_text = str(exc)
        await database["ingestion_tasks"].update_one(
            {"_id": task_id},
            {"$set": {"status": "FAILED", "error": error_text, "updated_at": utc_now()}},
        )
        await database["documents"].update_one(
            {"_id": document_id},
            {"$set": {"status": "FAILED", "error": error_text, "updated_at": utc_now()}},
        )
        raise HTTPException(status_code=500, detail=f"Failed to enqueue indexing task: {error_text}") from exc

    document_record["_id"] = document_id
    document_record["storage_path"] = str(storage_path)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "workspace": serialize_document(workspace),
            "document": serialize_document(document_record),
            "task": serialize_document(task_record),
            "message": "Document accepted for background indexing.",
        },
    )


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    database = await get_database()
    task = await database["ingestion_tasks"].find_one({"_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    document = None
    if task.get("document_id") and ObjectId.is_valid(task["document_id"]):
        document = await database["documents"].find_one({"_id": ObjectId(task["document_id"])})

    return {
        "task": serialize_document(task),
        "document": serialize_document(document) if document else None,
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
    history = await get_recent_history_messages(database, request.workspace_id, request.document_id)
    llm_messages = build_llm_messages(request.message, context, history)
    answer = await fetch_completion(llm_messages, stream=False)

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
    return {"response": answer, "citations": serializable_citations}


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
