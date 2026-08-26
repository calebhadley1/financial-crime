from fastapi import FastAPI
from prometheus_client import make_asgi_app

from financial_crime.api.routers import health_router, metrics_router, prediction_router

app = FastAPI()

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(prediction_router.router)
app.include_router(health_router.router)
