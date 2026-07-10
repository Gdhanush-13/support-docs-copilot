"""
Support Docs Copilot — FastAPI application entry point.
RAG-powered documentation assistant with hybrid search, citations, and memory.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api.routes import ingest, query, admin, health
from src.config import settings
from src.core.vector_store import init_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising vector store…")
    await init_vector_store()
    logger.info("Support Docs Copilot ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Support Docs Copilot",
    description="RAG-powered documentation assistant with hybrid search and citations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
