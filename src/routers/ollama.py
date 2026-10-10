import datetime
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException

from src.schemas.ollama import OllamaChatRequest, OllamaResponse
from src.dependencies import SettingsDep
from src.services.ollama.client import OllamaClient

router = APIRouter()


@router.post("/ollama_chat", response_model=OllamaResponse, tags=["Ollama"])
async def ollama_chat(
    request: OllamaChatRequest, settings: SettingsDep
) -> OllamaResponse:
    try:
        ollama_client = OllamaClient(settings)
        ollama_model = request.model or settings.ollama_model
        model_response = await ollama_client.generate(ollama_model, request.query)
        
        return OllamaResponse.model_validate(model_response)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
