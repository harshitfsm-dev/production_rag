
import json
import logging
from fastapi.responses import StreamingResponse
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
    

@router.post("/ollama_chat_stream", tags=["Ollama"])
async def ollama_chat(
    request: OllamaChatRequest, settings: SettingsDep
) -> StreamingResponse:
    try:
        ollama_client = OllamaClient(settings)
        ollama_model = request.model or settings.ollama_model
        async def generate():
            async for chunk in ollama_client.generate_stream(
                ollama_model,
                request.query,
            ):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
