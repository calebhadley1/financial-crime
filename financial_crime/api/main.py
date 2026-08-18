from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel

from financial_crime.config import MODELS_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline

app = FastAPI()


class PredictionRequest(BaseModel):
    features: list[dict[Any, Any]]


class PredictionResponse(BaseModel):
    predictions: list[int]


@lru_cache(maxsize=1)
def load_pipeline():
    pipeline_dir = MODELS_DIR / "pipeline"
    logger.info(f"Loading InferencePipeline from {pipeline_dir}")
    pipeline = InferencePipeline.load(pipeline_dir)
    return pipeline


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Perform inference using the trained ML pipeline.

    The pipeline handles the complete transformation:
    engineered features → preprocessing → model prediction
    """
    # Convert list of dicts to DataFrame
    logger.info("Transforming engineered features request into DataFrame")
    df = pd.DataFrame(request.features)

    # Load the trained pipeline
    logger.info("Loading pipeline")
    pipeline = load_pipeline()

    # Perform inference
    logger.info("Making prediction")
    preds = pipeline.predict(df)
    predictions = preds.tolist()

    # Summary stat on # fraud vs non-fraud predictions
    values, counts = np.unique(predictions, return_counts=True)
    value_counts = dict(zip(values, counts))
    logger.info(f"Predictions complete: {value_counts=}")

    return PredictionResponse(predictions=preds.tolist())
