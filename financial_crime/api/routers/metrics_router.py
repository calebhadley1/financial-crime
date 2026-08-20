from fastapi import APIRouter, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

router = APIRouter()
metrics_registry = CollectorRegistry()


def instrument_app(app: FastAPI) -> None:
    Instrumentator(registry=metrics_registry).instrument(app)


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(
        content=generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST,
    )
