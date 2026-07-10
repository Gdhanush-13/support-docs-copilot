"""Qdrant vector store initialisation and client accessor."""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings

logger = logging.getLogger(__name__)
_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    return _client


async def init_vector_store() -> None:
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", settings.QDRANT_COLLECTION)
    else:
        logger.info("Qdrant collection '%s' already exists.", settings.QDRANT_COLLECTION)
