"""RAG chain: memory-aware conversation + citation-grounded answer generation."""

import logging
import time
import uuid
from typing import Optional

import redis.asyncio as aioredis
from openai import AsyncOpenAI

from src.config import settings
from src.core.retriever import retrieve
from src.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

_openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def _load_history(session_id: str) -> str:
    r = await get_redis()
    history = await r.lrange(f"history:{session_id}", -6, -1)
    return "\n".join(history) if history else ""


async def _save_turn(session_id: str, question: str, answer: str) -> None:
    r = await get_redis()
    key = f"history:{session_id}"
    await r.rpush(key, f"User: {question}", f"Assistant: {answer}")
    await r.expire(key, settings.MEMORY_TTL_SECONDS)


SYSTEM_PROMPT = """You are a helpful support documentation assistant.
Answer ONLY from the provided context chunks. If the answer is not in the context, say so clearly.
Always cite your sources using [source: <source_name>] inline.
Be concise and accurate."""


async def answer(req: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()
    session_id = req.session_id or str(uuid.uuid4())

    citations = await retrieve(req.question, top_k=req.top_k, filters=req.filters)
    history = await _load_history(session_id)

    context = "\n\n".join(
        f"[{i+1}] (source: {c.source})\n{c.text}" for i, c in enumerate(citations)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.append({"role": "assistant", "content": f"Previous conversation:\n{history}"})
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {req.question}",
        }
    )

    completion = await _openai.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    answer_text = completion.choices[0].message.content

    await _save_turn(session_id, req.question, answer_text)
    latency = (time.monotonic() - t0) * 1000

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        session_id=session_id,
        model=settings.OPENAI_CHAT_MODEL,
        latency_ms=round(latency, 2),
    )
