from fastapi import APIRouter
from sqlalchemy import text

from src.dependencies import SettingsDep, DatabaseDep
from src.schemas.api.health import HealthResponse, ServiceStatus
from src.services.ollama.client import OllamaClient

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: SettingsDep, database: DatabaseDep) -> HealthResponse:
    services = {}
    overall_status = "ok"
    try:
        ollama_client = OllamaClient(settings)
        ollama_health = await ollama_client.health_check()
        services["ollama"] = ServiceStatus(status=ollama_health["status"], message=ollama_health["message"])
        if ollama_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["ollama"] = ServiceStatus(status="unhealthy", message=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
