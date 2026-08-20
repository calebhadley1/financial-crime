from fastapi import FastAPI

from financial_crime.api.routers import health_router, metrics_router, prediction_router

app = FastAPI()
metrics_router.instrument_app(app)

app.include_router(metrics_router.router)
app.include_router(prediction_router.router)
app.include_router(health_router.router)
