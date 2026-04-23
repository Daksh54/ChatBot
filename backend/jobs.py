# Author: Daksh Sharma 26434

from typing import Any, Dict

from bson import ObjectId
from pymongo import MongoClient
from qdrant_client import QdrantClient

from services import ensure_qdrant_collection, process_document_indexing, utc_now
from settings import MONGODB_DB, MONGODB_URI, QDRANT_URL


def process_document_task(task_id: str) -> Dict[str, Any]:
    mongo_client = MongoClient(MONGODB_URI)
    database = mongo_client[MONGODB_DB]
    qdrant = QdrantClient(url=QDRANT_URL)

    tasks = database["ingestion_tasks"]
    documents = database["documents"]
    workspaces = database["workspaces"]

    task = tasks.find_one({"_id": task_id})
    if not task:
        mongo_client.close()
        raise ValueError(f"Task {task_id} not found.")

    document_id = task["document_id"]
    workspace_id = task["workspace_id"]

    def set_progress(progress: int, phase: str) -> None:
        tasks.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": "PROCESSING",
                    "progress": progress,
                    "phase": phase,
                    "updated_at": utc_now(),
                }
            },
        )

    try:
        ensure_qdrant_collection(qdrant)
        set_progress(5, "Worker picked up task")

        result = process_document_indexing(
            database=database,
            qdrant=qdrant,
            document_id=document_id,
            workspace_id=workspace_id,
            storage_path=task["storage_path"],
            file_name=task["file_name"],
            mime_type=task.get("mime_type", "application/octet-stream"),
            progress_callback=set_progress,
        )

        documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "READY",
                    "file_hash": result["file_hash"],
                    "page_count": result["page_count"],
                    "chunk_count": result["chunk_count"],
                    "updated_at": utc_now(),
                    "error": None,
                }
            },
        )
        workspaces.update_one(
            {"_id": ObjectId(workspace_id)},
            {"$set": {"updated_at": utc_now()}},
        )
        tasks.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": "SUCCEEDED",
                    "progress": 100,
                    "phase": "Indexing complete",
                    "updated_at": utc_now(),
                    "completed_at": utc_now(),
                    "result": result,
                    "error": None,
                }
            },
        )
        return {"task_id": task_id, **result}
    except Exception as exc:
        error_text = str(exc)
        tasks.update_one(
            {"_id": task_id},
            {
                "$set": {
                    "status": "FAILED",
                    "phase": "Indexing failed",
                    "updated_at": utc_now(),
                    "completed_at": utc_now(),
                    "error": error_text,
                }
            },
        )
        documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "FAILED",
                    "updated_at": utc_now(),
                    "error": error_text,
                }
            },
        )
        raise
    finally:
        mongo_client.close()
