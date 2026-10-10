from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session


from src.config import Settings
from src.db.interfaces.base import BaseDatabase
# from src.services.cache.client import CacheClient
# from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.ollama.client import OllamaClient
# from src.services.opensearch.client import OpenSearchClient
# from src.services.pdf_parser.parser import PDFParserService
# from src.services.agents.agentic_rag import AgenticRAGService
# from src.services.agents.factory import make_agentic_rag_service


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


def get_ollama_client(request: Request) -> OllamaClient:
    """Get Ollama client from the request state."""
    return request.app.state.ollama_client



# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
