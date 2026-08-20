from fastapi import APIRouter, Response, status
from loguru import logger

from financial_crime.api.dependencies import _load_pipeline

router = APIRouter(prefix="/health")


@router.get("/live", status_code=status.HTTP_200_OK)
def liveness_check():
    """Confirms the FastAPI application process is up and running."""
    return {"status": "alive"}


@router.get("/ready")
def readiness_check(response: Response):
    """Verify that the persisted inference pipeline can be loaded."""
    try:
        _load_pipeline()
    except:  # noqa E722
        logger.exception("Inference pipeline is not ready")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "reason": "Inference pipeline could not be loaded"}

    return {"status": "healthy"}
