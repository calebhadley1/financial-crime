from typing import Annotated

from fastapi import APIRouter, Depends
from loguru import logger

from financial_crime.api.clients.feature_client import FeatureClient
from financial_crime.api.clients.inference_client import InferenceClient
from financial_crime.api.dependencies import get_feature_client, get_inference_client
from financial_crime.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict")


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
    features_df = feature_client.get(requests=requests)
    results_df = inference_client.predict(features_df=features_df)
    response = [
        PredictionResponse.model_validate(row) for row in results_df.to_dict(orient="records")
    ]
    logger.debug(f"Inference response: {response}")

    return response
