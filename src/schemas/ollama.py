"""Pydantic models for Ollama structured outputs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Structured response model for RAG queries."""

    answer: str = Field(
        description="Comprehensive answer based on the provided paper excerpts"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of PDF URLs from papers used in the answer",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: high, medium, or low based on excerpt relevance",
    )
    citations: Optional[List[str]] = Field(
        default=None,
        description="Specific IDs or paper titles referenced in the answer",
    )


class OllamaResponse(BaseModel):
    """Default Ollama response model."""

    model: str = Field(
        description="Comprehensive answer based on the provided paper excerpts"
    )
    created_at: str = Field(
        description="List of PDF URLs from papers used in the answer",
    )
    response: str = Field(
        description="Confidence level: high, medium, or low based on excerpt relevance",
    )
    usage_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Specific IDs or paper titles referenced in the answer",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "model": "ollama:3.1:8b",
                "created_at": "2026-10-10T14:18:17.907067Z",
                "response": "Test Response message",
                "usage_metadata": {
                    "prompt_tokens": 508,
                    "total_tokens": 491,
                },
            }
        }


class OllamaChatRequest(BaseModel):
    """Request model for Ollama question answering."""

    query: str = Field(
        ..., description="User's question", min_length=1, max_length=1000
    )
    model: Optional[str] = Field(
        "llama3.2:8b", description="Ollama model to use for generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "model": "llama3.1:8b",
            }
        }
