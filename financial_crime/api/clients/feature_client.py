from feast import FeatureStore
from loguru import logger
from pandas import DataFrame

from financial_crime.api.schemas import PredictionRequest


class FeatureClient:
    """
    Encapsulates communication with Feast Feature Store
    """

    def __init__(self, feature_store_repo_path, feature_service_name):
        self.feature_store = FeatureStore(feature_store_repo_path)
        self.feature_service = self.feature_store.get_feature_service(feature_service_name)

    def get(self, requests: list[PredictionRequest]) -> DataFrame:
        """
        Loads Online Engineered Features from Feast
        """
        entity_rows = [request.model_dump(mode="json") for request in requests]
        inference_data = self.feature_store.get_online_features(
            features=self.feature_service, entity_rows=entity_rows
        ).to_df()
        logger.info(f"Loaded {len(inference_data)} rows from Feature Store")
        return inference_data
