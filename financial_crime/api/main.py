from datetime import datetime
from functools import lru_cache
from typing import Literal
from uuid import UUID

from fastapi import FastAPI
from feast import FeatureStore
from loguru import logger
import pandas as pd
from pydantic import BaseModel

from financial_crime.config import MODELS_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline

app = FastAPI()


class PredictionRequest(BaseModel):
    """
    All requests must have a corresponding entry in the feature store
    """

    ID: UUID
    event_timestamp: datetime


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float


@lru_cache(maxsize=1)
def load_pipeline():
    pipeline_dir = MODELS_DIR / "pipeline"
    logger.info(f"Loading InferencePipeline from {pipeline_dir}")
    pipeline = InferencePipeline.load(pipeline_dir)
    return pipeline


@app.post("/predict")
def predict(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    """
    Perform inference using the trained ML pipeline.

    The pipeline handles the complete transformation:
    engineered features → preprocessing → model prediction
    """
    # Load the trained pipeline
    logger.info("Loading pipeline")
    pipeline = load_pipeline()

    # Pull engineered features from the feature store using the entity DataFrame
    feature_store = FeatureStore("financial_crime/feature_store/feature_repo")
    feature_service = feature_store.get_feature_service("transaction_v1")
    entity_rows = [request.model_dump(mode="json") for request in requests]
    inference_data = feature_store.get_online_features(
        features=feature_service, entity_rows=entity_rows
    ).to_df()
    logger.info(f"Loaded {len(inference_data)} rows from Feature Store")

    # Perform inference
    logger.info("Making prediction")
    probs = pipeline.predict_proba(inference_data)
    probs_df = pd.DataFrame(probs, columns=pipeline.model.classes_)
    results = pd.DataFrame(
        {"prediction": probs_df.idxmax(axis=1), "probability": probs_df.max(axis=1)}
    )
    response = [
        PredictionResponse.model_validate(row) for row in results.to_dict(orient="records")
    ]
    logger.info(f"{response=}")
    return response
