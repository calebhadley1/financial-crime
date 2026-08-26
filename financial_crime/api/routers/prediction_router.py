from typing import Annotated

from fastapi import APIRouter, Depends
from loguru import logger
from time import time
from prometheus_client import Counter, Histogram

from financial_crime.api.clients.feature_client import FeatureClient
from financial_crime.api.clients.inference_client import InferenceClient
from financial_crime.api.dependencies import get_feature_client, get_inference_client
from financial_crime.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict")

MODEL_VERSION = "v0.1.0"

# Track the number of predictions and split them by model version and classification outcome
PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of ML predictions made",
    ["model_version", "predicted_class"]
)

# Track the distribution of model confidence scores (probabilities)
MODEL_CONFIDENCE = Histogram(
    "ml_model_confidence_scores",
    "Distribution of model confidence scores",
    ["model_version"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Custom bins for probabilities
)

# Track internal model processing time (excluding network overhead)
MODEL_INFERENCE_TIME = Histogram(
    "ml_inference_duration_seconds",
    "Time taken strictly by the ML model to run inference",
    ["model_version"]
)

@router.post("")
def predict(
    requests: list[PredictionRequest],
    feature_client: Annotated[FeatureClient, Depends(get_feature_client)],
    inference_client: Annotated[InferenceClient, Depends(get_inference_client)],
) -> list[PredictionResponse]:
    """
    Perform inference using the trained ML pipeline.

    The pipeline handles the complete transformation:
    engineered features → preprocessing → model prediction
    """
    start_time = time()

    features_df = feature_client.get(requests=requests)
    results_df = inference_client.predict(features_df=features_df)
    response = [
        PredictionResponse.model_validate(row) for row in results_df.to_dict(orient="records")
    ]
    
    inference_duration = time() - start_time

    logger.debug("Updating Prometheus metrics")
    for pred in response:
        PREDICTION_COUNTER.labels(model_version=MODEL_VERSION, predicted_class=pred.prediction).inc()
        MODEL_CONFIDENCE.labels(model_version=MODEL_VERSION).observe(pred.probability)
        MODEL_INFERENCE_TIME.labels(model_version=MODEL_VERSION).observe(inference_duration)

    logger.debug(f"Inference response: {response}")
    

    return response
