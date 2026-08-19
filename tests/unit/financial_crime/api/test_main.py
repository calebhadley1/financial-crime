from datetime import datetime
from uuid import uuid4

import pandas as pd

from financial_crime.api import main as api


class FakePipeline:
    model = type("Model", (), {"classes_": [0, 1]})()

    def predict_proba(self, dataframe):
        assert len(dataframe) == 2
        return [[0.8, 0.2], [0.1, 0.9]]


class FakeOnlineFeatures:
    def to_df(self):
        return pd.DataFrame({"feature": [1, 2]})


class FakeStore:
    def get_feature_service(self, name):
        assert name == "transaction_v1"
        return "service"

    def get_online_features(self, features, entity_rows):
        assert features == "service"
        assert len(entity_rows) == 2
        return FakeOnlineFeatures()


def test_predict_returns_highest_probability(monkeypatch):
    monkeypatch.setattr(api, "load_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(api, "FeatureStore", lambda path: FakeStore())
    requests = [
        api.PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 1)),
        api.PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 2)),
    ]

    response = api.predict(requests)

    assert [item.prediction for item in response] == [0, 1]
    assert [item.probability for item in response] == [0.8, 0.9]