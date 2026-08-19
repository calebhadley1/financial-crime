from datetime import datetime
from functools import lru_cache
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI
from feast import FeatureStore
from loguru import logger
import numpy as np
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
    predictions: list[Literal[0, 1]]


@lru_cache(maxsize=1)
def load_pipeline():
    pipeline_dir = MODELS_DIR / "pipeline"
    logger.info(f"Loading InferencePipeline from {pipeline_dir}")
    pipeline = InferencePipeline.load(pipeline_dir)
    return pipeline


@app.post("/predict")
def predict(requests: list[PredictionRequest]) -> PredictionResponse:
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
    y_pred = pipeline.predict(inference_data)
    y_pred_list = y_pred.tolist()

    # Summary stat on # fraud vs non-fraud predictions
    values, counts = np.unique(y_pred_list, return_counts=True)
    value_counts = dict(zip(values, counts))
    logger.info(f"Predictions complete: {value_counts=}")

    return PredictionResponse(predictions=y_pred_list)
