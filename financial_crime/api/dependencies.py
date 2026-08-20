from functools import lru_cache

from loguru import logger

from financial_crime.api.clients.feature_client import FeatureClient
from financial_crime.api.clients.inference_client import InferenceClient
from financial_crime.config import PIPELINE_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline


@lru_cache(maxsize=1)
def _load_pipeline():
    logger.info(f"Loading InferencePipeline from {PIPELINE_DIR}")
    pipeline = InferencePipeline.load(PIPELINE_DIR)
    return pipeline


def get_feature_client() -> FeatureClient:
    return FeatureClient(
        feature_store_repo_path="financial_crime/feature_store/feature_repo",
        feature_service_name="transaction_v1",
    )


def get_inference_client() -> InferenceClient:
    return InferenceClient(pipeline=_load_pipeline())
