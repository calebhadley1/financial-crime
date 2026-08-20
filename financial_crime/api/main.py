from fastapi import FastAPI

from financial_crime.api.routers import health_router, prediction_router

app = FastAPI()

app.include_router(prediction_router.router)
app.include_router(health_router.router)
