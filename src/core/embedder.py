"""Async embedding generation using OpenAI text-embedding-3-small."""

from typing import List
from openai import AsyncOpenAI
from src.config import settings

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, respecting OpenAI's batch size limit."""
    BATCH = 100
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = await _client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
    return all_embeddings


async def embed_query(text: str) -> List[float]:
    resp = await _client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding
