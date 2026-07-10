"""Document ingestion: loading, chunking, embedding, and upsert into Qdrant."""

import hashlib
import logging
import uuid
from typing import List

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

from src.config import settings
from src.core.embedder import embed_texts
from src.core.vector_store import get_qdrant_client
from src.models.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)


def _load_content(req: IngestRequest) -> str:
    if req.content:
        return req.content
    if req.source_url:
        resp = httpx.get(req.source_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    raise ValueError("Either content or source_url must be provided.")


def _chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


async def ingest_document(req: IngestRequest) -> IngestResponse:
    content = _load_content(req)
    chunks = _chunk_text(content)

    if len(chunks) > settings.MAX_CHUNKS_PER_DOC:
        chunks = chunks[: settings.MAX_CHUNKS_PER_DOC]
        logger.warning("Document truncated to %d chunks.", settings.MAX_CHUNKS_PER_DOC)

    doc_id = hashlib.sha256(content[:512].encode()).hexdigest()[:16]
    embeddings = await embed_texts(chunks)

    client = get_qdrant_client()
    points = []
    from qdrant_client.models import PointStruct

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "source": req.source_url or "direct",
                    **req.metadata,
                },
            )
        )

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    logger.info("Ingested doc_id=%s with %d chunks.", doc_id, len(chunks))
    return IngestResponse(doc_id=doc_id, chunks_created=len(chunks), message="Success")
