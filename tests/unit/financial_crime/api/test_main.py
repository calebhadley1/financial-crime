from datetime import datetime
from uuid import uuid4

import pandas as pd

from financial_crime.api import main as api


class FakeFeatureClient:
    def get(self, requests):
        assert len(requests) == 2
        return pd.DataFrame({"feature": [1, 2]})


class FakeInferenceClient:
    def predict(self, features_df):
        assert list(features_df["feature"]) == [1, 2]
        return pd.DataFrame({"prediction": [0, 1], "probability": [0.8, 0.9]})


def test_predict_returns_highest_probability():
    requests = [
        api.PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 1)),
        api.PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 2)),
    ]

    response = api.predict(
        requests,
        feature_client=FakeFeatureClient(),
        inference_client=FakeInferenceClient(),
    )

    assert [item.prediction for item in response] == [0, 1]
    assert [item.probability for item in response] == [0.8, 0.9]