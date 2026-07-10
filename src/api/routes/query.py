"""Query endpoint."""
from fastapi import APIRouter, HTTPException
from src.core.rag_chain import answer
from src.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        return await answer(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
