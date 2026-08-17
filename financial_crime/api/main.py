from functools import lru_cache
from typing import Any

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

from financial_crime.config import MODELS_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline

app = FastAPI()


class PredictionRequest(BaseModel):
    raw_features: list[dict[Any, Any]]


class PredictionResponse(BaseModel):
    predictions: list[int]


@lru_cache(maxsize=1)
def load_pipeline():
    pipeline_dir = MODELS_DIR / "pipeline"
    pipeline = InferencePipeline.load(pipeline_dir)
    return pipeline


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Perform inference using the trained ML pipeline.

    The pipeline handles the complete transformation:
    raw data → feature engineering → preprocessing → model prediction
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(request.raw_features)

    # Load the trained pipeline
    pipeline = load_pipeline()

    # Perform inference
    preds = pipeline.predict(df)

    return PredictionResponse(predictions=preds.tolist())
