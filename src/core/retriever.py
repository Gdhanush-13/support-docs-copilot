"""Hybrid retrieval: dense vector search + BM25 sparse, fused with RRF."""

import logging
from typing import List

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import settings
from src.core.embedder import embed_query
from src.core.vector_store import get_qdrant_client
from src.models.schemas import Citation

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


async def retrieve(question: str, top_k: int = 5, filters: dict = None) -> List[Citation]:
    """Hybrid retrieval: dense + sparse merged via Reciprocal Rank Fusion."""
    query_vec = await embed_query(question)
    client = get_qdrant_client()

    qdrant_filter = None
    if filters:
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        qdrant_filter = Filter(must=conditions)

    # Dense search
    dense_hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=settings.TOP_K_DENSE,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    # Build citations with RRF fusion scores
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit.id
        scores[cid] = scores.get(cid, 0) + _rrf_score(rank) * settings.HYBRID_ALPHA
        payloads[cid] = hit.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    citations = []
    for cid, score in ranked:
        p = payloads[cid]
        citations.append(
            Citation(
                doc_id=p.get("doc_id", ""),
                chunk_id=str(cid),
                score=round(score, 4),
                text=p.get("text", ""),
                source=p.get("source", ""),
                page=p.get("page"),
            )
        )

    logger.debug("Retrieved %d chunks for query.", len(citations))
    return citations
