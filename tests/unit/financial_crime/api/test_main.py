from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pandas as pd

from financial_crime.api.main import app
from financial_crime.api.routers.prediction_router import predict
from financial_crime.api.schemas import PredictionRequest


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
        PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 1, tzinfo=ZoneInfo("America/New_York"))),
        PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 2, tzinfo=ZoneInfo("America/New_York"))),
    ]

    response = predict(
        requests,
        feature_client=FakeFeatureClient(),
        inference_client=FakeInferenceClient(),
    )

    assert [item.prediction for item in response] == [0, 1]
    assert [item.probability for item in response] == [0.8, 0.9]


def test_metrics_endpoint_exposes_prometheus_metrics():
    client = TestClient(app)
    client.get("/health/live")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
