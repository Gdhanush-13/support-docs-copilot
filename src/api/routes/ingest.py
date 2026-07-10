"""Ingestion endpoints."""
from fastapi import APIRouter, HTTPException
from src.core.ingestion import ingest_document
from src.models.schemas import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/", response_model=IngestResponse, status_code=201)
async def ingest(req: IngestRequest):
    try:
        return await ingest_document(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
