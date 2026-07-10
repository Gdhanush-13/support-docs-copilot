"""Unit tests for RAG chain and retriever (mocked dependencies)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.models.schemas import QueryRequest, Citation


@pytest.fixture
def sample_citations():
    return [
        Citation(doc_id="abc", chunk_id="1", score=0.9, text="Foo is a bar.", source="docs/foo.md"),
        Citation(doc_id="abc", chunk_id="2", score=0.8, text="Bar does baz.", source="docs/bar.md"),
    ]


@pytest.mark.asyncio
async def test_answer_returns_response(sample_citations):
    req = QueryRequest(question="What is foo?", top_k=2)
    mock_answer = "Foo is a bar [source: docs/foo.md]."

    with (
        patch("src.core.rag_chain.retrieve", new_callable=AsyncMock, return_value=sample_citations),
        patch("src.core.rag_chain._load_history", new_callable=AsyncMock, return_value=""),
        patch("src.core.rag_chain._save_turn", new_callable=AsyncMock),
        patch("src.core.rag_chain._openai") as mock_openai,
    ):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = mock_answer
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)

        from src.core.rag_chain import answer
        result = await answer(req)

    assert result.answer == mock_answer
    assert len(result.citations) == 2
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_retrieve_returns_citations(sample_citations):
    with (
        patch("src.core.retriever.embed_query", new_callable=AsyncMock, return_value=[0.1] * 1536),
        patch("src.core.retriever.get_qdrant_client") as mock_client,
    ):
        mock_hit = MagicMock()
        mock_hit.id = "chunk-1"
        mock_hit.payload = {"doc_id": "abc", "text": "Foo is a bar.", "source": "docs/foo.md"}
        mock_client.return_value.search.return_value = [mock_hit]

        from src.core.retriever import retrieve
        results = await retrieve("What is foo?", top_k=1)

    assert len(results) == 1
    assert results[0].source == "docs/foo.md"
