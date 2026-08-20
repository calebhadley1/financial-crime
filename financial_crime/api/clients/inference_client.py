from loguru import logger
from pandas import DataFrame


class InferenceClient:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, features_df: DataFrame) -> DataFrame:
        logger.debug("Making prediction")
        probs = self.pipeline.predict_proba(features_df)
        probs_df = DataFrame(probs, columns=self.pipeline.model.classes_)
        results = DataFrame(
            {"prediction": probs_df.idxmax(axis=1), "probability": probs_df.max(axis=1)}
        )
        logger.debug(f"Inference results: {results}")
        return results
