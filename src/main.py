import os
import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager


from src.services.ollama.factory import make_ollama_client
from src.config import get_settings
from src.db.factory import make_database
from src.routers import health, ollama


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

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # app.state.embeddings_service = make_embeddings_service()
    app.state.ollama_client = make_ollama_client()
    # logger.info("Services initialized: PDF parser, OpenSearch, Embeddings, Ollama, Cache")
    yield

app = FastAPI(
    title="Production RAG API",
    description="Personal assistant with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(health.router, prefix="/api/v1")  # Health check endpoint
app.include_router(ollama.router, prefix="/api/v1")  # Health check endpoint
# app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid



if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")