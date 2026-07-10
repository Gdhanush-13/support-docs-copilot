"""Admin endpoints — stats, collection management."""
from fastapi import APIRouter
from src.core.vector_store import get_qdrant_client
from src.config import settings
from src.models.schemas import AdminStatsResponse

router = APIRouter()


@router.get("/stats", response_model=AdminStatsResponse)
async def stats():
    client = get_qdrant_client()
    info = client.get_collection(settings.QDRANT_COLLECTION)
    count = info.points_count or 0
    return AdminStatsResponse(
        total_documents=0,
        total_chunks=count,
        total_queries=0,
        avg_latency_ms=0.0,
        collection_size_mb=round((info.segments_count or 0) * 0.1, 2),
        last_ingestion=None,
    )


@router.delete("/collection")
async def reset_collection():
    client = get_qdrant_client()
    client.delete_collection(settings.QDRANT_COLLECTION)
    from src.core.vector_store import init_vector_store
    await init_vector_store()
    return {"message": "Collection reset successfully."}
