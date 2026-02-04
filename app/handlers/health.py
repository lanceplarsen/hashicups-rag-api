from fastapi import APIRouter, Response, status
from typing import TYPE_CHECKING

from app.models import HealthStats
from app.metrics import service_health

if TYPE_CHECKING:
    from app.services.indexer import IndexerService
    from app.client.public_api import PublicAPIClient
    from app.client.anthropic_client import AnthropicClient

router = APIRouter()

# These will be set by main.py
_indexer_service: "IndexerService" = None
_public_api_client: "PublicAPIClient" = None
_anthropic_client: "AnthropicClient" = None


def set_dependencies(
    indexer_service: "IndexerService",
    public_api_client: "PublicAPIClient",
    anthropic_client: "AnthropicClient"
):
    """Set the dependencies for health checks."""
    global _indexer_service, _public_api_client, _anthropic_client
    _indexer_service = indexer_service
    _public_api_client = public_api_client
    _anthropic_client = anthropic_client


@router.get("/health/livez")
async def liveness():
    """Liveness check - returns OK if service is running."""
    service_health.set(1)
    return {"status": "ok"}


@router.get("/health/readyz")
async def readiness():
    """Readiness check - verifies dependencies are available."""
    is_ready = True
    reasons = []

    # Check if indexer has run at least once
    if _indexer_service:
        if _indexer_service.total_documents == 0:
            is_ready = False
            reasons.append("no documents indexed")
    else:
        is_ready = False
        reasons.append("indexer not initialized")

    if is_ready:
        service_health.set(1)
        return {"status": "ready"}
    else:
        service_health.set(0)
        return Response(
            content=f'{{"status": "not ready", "reasons": {reasons}}}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json"
        )


@router.get("/health/stats", response_model=HealthStats)
async def stats():
    """Get service statistics."""
    public_api_healthy = False
    anthropic_api_healthy = False

    # Check Public API health
    if _public_api_client:
        try:
            from app.client.public_api import PublicAPIClient
            async with PublicAPIClient() as client:
                public_api_healthy = await client.health_check()
        except Exception:
            pass

    # Check Anthropic API health
    if _anthropic_client:
        anthropic_api_healthy = await _anthropic_client.health_check()

    return HealthStats(
        total_documents=_indexer_service.total_documents if _indexer_service else 0,
        last_index_time=_indexer_service.last_index_time if _indexer_service else None,
        index_status=_indexer_service.index_status if _indexer_service else "not_initialized",
        public_api_healthy=public_api_healthy,
        anthropic_api_healthy=anthropic_api_healthy
    )
