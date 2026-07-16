"""Mock heavy external deps so unit tests run without full install."""
import sys
from unittest.mock import MagicMock

_HEAVY = [
    "redis", "redis.asyncio",
    "qdrant_client", "qdrant_client.models",
    "langchain", "langchain.text_splitter", "langchain_text_splitters",
    "langchain_openai",
    "unstructured", "unstructured.partition", "unstructured.partition.md",
    "asyncpg",
    "sqlalchemy", "sqlalchemy.ext", "sqlalchemy.ext.asyncio", "sqlalchemy.orm",
    "openai",
]

for _mod in _HEAVY:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make openai.AsyncOpenAI available as a class
import openai  # noqa: E402 (already mocked above)
openai.AsyncOpenAI = MagicMock
