# Author: Daksh Sharma 26434

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
from bson import ObjectId
from docx import Document
from PIL import Image
from qdrant_client import QdrantClient, models

from settings import (
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    EMBED_BATCH_SIZE,
    EMBED_MODEL_NAME,
    QDRANT_COLLECTION,
    RERANK_MODEL_NAME,
)


@dataclass
class SourcePage:
    page_number: int
    text: str


@dataclass
class ChunkPayload:
    text: str
    page_start: int
    page_end: int


@dataclass
class SentenceSegment:
    page_number: int
    text: str
    word_count: int


_EMBEDDER = None
_RERANKER = None


def utc_now() -> datetime:
    return datetime.utcnow()


def normalize_text(value: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename).strip("-")
    return cleaned or "upload"


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBEDDER


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        try:
            _RERANKER = CrossEncoder(RERANK_MODEL_NAME)
        except Exception:
            _RERANKER = False
    return None if _RERANKER is False else _RERANKER


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


def ensure_qdrant_collection(qdrant: QdrantClient) -> None:
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


def extract_structured_dataframe(file_name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(payload))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(payload))
    raise ValueError("Unsupported structured file.")


def dataframe_to_context(frame: pd.DataFrame) -> str:
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
    return text


def iter_sentences(text: str) -> Iterator[str]:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        cleaned = normalize_text(sentence)
        if cleaned:
            yield cleaned


def iter_source_pages(file_name: str, content_type: str, payload: bytes) -> Iterator[SourcePage]:
    if content_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not normalize_text(text):
                    image = page.to_image(resolution=200).original
                    text = pytesseract.image_to_string(image)
                cleaned = normalize_text(text)
                if cleaned:
                    yield SourcePage(page_number=index, text=cleaned)
        return

    if content_type.startswith("image/"):
        image = Image.open(io.BytesIO(payload))
        cleaned = normalize_text(pytesseract.image_to_string(image))
        if cleaned:
            yield SourcePage(page_number=1, text=cleaned)
        return

    if content_type == "text/plain" or file_name.lower().endswith(".txt"):
        cleaned = normalize_text(payload.decode("utf-8", errors="ignore"))
        if cleaned:
            yield SourcePage(page_number=1, text=cleaned)
        return

    if file_name.lower().endswith(".docx"):
        document = Document(io.BytesIO(payload))
        paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)
        cleaned = normalize_text(paragraphs)
        if cleaned:
            yield SourcePage(page_number=1, text=cleaned)
        return

    if file_name.lower().endswith((".csv", ".xlsx", ".xls")):
        dataframe = extract_structured_dataframe(file_name, payload)
        yield SourcePage(page_number=1, text=dataframe_to_context(dataframe))
        return

    raise ValueError("Unsupported file type.")


def build_overlap_segments(segments: List[SentenceSegment], overlap_words: int) -> List[SentenceSegment]:
    if overlap_words <= 0 or not segments:
        return []

    selected: List[SentenceSegment] = []
    words_kept = 0
    for segment in reversed(segments):
        selected.append(segment)
        words_kept += segment.word_count
        if words_kept >= overlap_words:
            break

    return list(reversed(selected))


def iter_semantic_chunks(
    pages: Iterable[SourcePage],
    max_words: int = CHUNK_MAX_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> Iterator[ChunkPayload]:
    current_segments: List[SentenceSegment] = []
    current_words = 0

    for page in pages:
        for sentence in iter_sentences(page.text):
            word_count = len(sentence.split())
            if word_count == 0:
                continue

            if current_segments and current_words + word_count > max_words:
                yield ChunkPayload(
                    text=" ".join(segment.text for segment in current_segments),
                    page_start=current_segments[0].page_number,
                    page_end=current_segments[-1].page_number,
                )
                current_segments = build_overlap_segments(current_segments, overlap_words)
                current_words = sum(segment.word_count for segment in current_segments)

            current_segments.append(
                SentenceSegment(
                    page_number=page.page_number,
                    text=sentence,
                    word_count=word_count,
                )
            )
            current_words += word_count

    if current_segments:
        yield ChunkPayload(
            text=" ".join(segment.text for segment in current_segments),
            page_start=current_segments[0].page_number,
            page_end=current_segments[-1].page_number,
        )


def flush_chunk_batch(
    database: Any,
    qdrant: QdrantClient,
    workspace_id: str,
    document_id: str,
    chunks: List[ChunkPayload],
    start_index: int,
) -> int:
    if not chunks:
        return 0

    embeddings = embed_texts([chunk.text for chunk in chunks])
    records: List[Dict[str, Any]] = []
    points: List[models.PointStruct] = []

    for offset, chunk in enumerate(chunks):
        chunk_index = start_index + offset
        records.append(
            {
                "workspace_id": ObjectId(workspace_id),
                "document_id": ObjectId(document_id),
                "chunk_index": chunk_index,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "created_at": utc_now(),
            }
        )
        points.append(
            models.PointStruct(
                id=f"{document_id}-{chunk_index}",
                vector=embeddings[offset].tolist(),
                payload={
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
            )
        )

    database["chunks"].insert_many(records)
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(chunks)


def process_document_indexing(
    database: Any,
    qdrant: QdrantClient,
    document_id: str,
    workspace_id: str,
    storage_path: str,
    file_name: str,
    mime_type: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    ensure_qdrant_collection(qdrant)
    payload = Path(storage_path).read_bytes()
    hasher = hashlib.sha256()
    page_count = 0
    chunk_count = 0
    next_chunk_index = 0
    batches_processed = 0
    chunk_batch: List[ChunkPayload] = []

    if progress_callback:
        progress_callback(10, "Extracting source pages")

    def tracked_pages() -> Iterator[SourcePage]:
        nonlocal page_count
        for page in iter_source_pages(file_name, mime_type, payload):
            page_count += 1
            hasher.update(page.text.encode("utf-8", errors="ignore"))
            if progress_callback and page_count % 20 == 0:
                progress_callback(min(35, 10 + page_count // 2), f"Parsed {page_count} pages")
            yield page

    for chunk in iter_semantic_chunks(tracked_pages()):
        chunk_batch.append(chunk)
        if len(chunk_batch) >= EMBED_BATCH_SIZE:
            inserted = flush_chunk_batch(
                database=database,
                qdrant=qdrant,
                workspace_id=workspace_id,
                document_id=document_id,
                chunks=chunk_batch,
                start_index=next_chunk_index,
            )
            chunk_count += inserted
            next_chunk_index += inserted
            batches_processed += 1
            chunk_batch = []
            if progress_callback:
                progress_callback(
                    min(92, 35 + batches_processed * 8),
                    f"Indexed {chunk_count} chunks",
                )

    if chunk_batch:
        inserted = flush_chunk_batch(
            database=database,
            qdrant=qdrant,
            workspace_id=workspace_id,
            document_id=document_id,
            chunks=chunk_batch,
            start_index=next_chunk_index,
        )
        chunk_count += inserted

    if page_count == 0:
        raise ValueError("No extractable text found in document.")
    if chunk_count == 0:
        raise ValueError("Document produced no semantic chunks.")

    return {
        "file_hash": hasher.hexdigest(),
        "page_count": page_count,
        "chunk_count": chunk_count,
    }
