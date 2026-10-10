import logging
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

import uvicorn
from src.routers import health


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")
    yield

app = FastAPI(
    title="Production RAG API",
    description="Personal assistant with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(health.router, prefix="/api/v1")  # Health check endpoint
# app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid



if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")