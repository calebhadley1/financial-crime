from functools import lru_cache

from fastapi import Depends, FastAPI
from feast import FeatureStore
from loguru import logger
import pandas as pd

from financial_crime.api.clients.feature_client import FeatureClient
from financial_crime.api.clients.inference_client import InferenceClient
from financial_crime.config import PIPELINE_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline
from financial_crime.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI()


@lru_cache(maxsize=1)
def _load_pipeline():
    logger.info(f"Loading InferencePipeline from {PIPELINE_DIR}")
    pipeline = InferencePipeline.load(PIPELINE_DIR)
    return pipeline

def get_feature_client() -> FeatureClient:
    return FeatureClient(
        feature_store_repo_path="financial_crime/feature_store/feature_repo",
        feature_service_name="transaction_v1"
    )

def get_inference_client() -> InferenceClient:
    return InferenceClient(
        pipeline=_load_pipeline()
    )


@app.post("/predict")
def predict(
    requests: list[PredictionRequest],
    feature_client: FeatureClient = Depends(get_feature_client),
    inference_client: InferenceClient = Depends(get_inference_client)
) -> list[PredictionResponse]:
    """
    Perform inference using the trained ML pipeline.

    The pipeline handles the complete transformation:
    engineered features → preprocessing → model prediction
    """
    features_df = feature_client.get(requests=requests)
    results_df = inference_client.predict(features_df=features_df)
    response = [
        PredictionResponse.model_validate(row) for row in results_df.to_dict(orient="records")
    ]
    logger.debug(f"Inference response: {response}")
   
    return response
