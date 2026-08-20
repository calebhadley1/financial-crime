from datetime import datetime
from uuid import uuid4

import pandas as pd
import pandas.testing as tm

from financial_crime.api.clients.feature_client import FeatureClient
from financial_crime.api.schemas import PredictionRequest


class FakeOnlineFeatures:
    def __init__(self, rows):
        self.rows = rows

    def to_df(self):
        return pd.DataFrame(self.rows)


class FakeFeatureStore:
    def __init__(self, repo_path):
        assert repo_path == "feature_repo"

    def get_feature_service(self, name):
        assert name == "transaction_v1"
        return "service"

    def get_online_features(self, features, entity_rows):
        assert features == "service"
        assert len(entity_rows) == 2
        assert set(entity_rows[0].keys()) == {"ID", "event_timestamp"}
        return FakeOnlineFeatures({"feature": [1, 2]})


def test_feature_client_get_returns_online_features(monkeypatch):
    monkeypatch.setattr(
        "financial_crime.api.clients.feature_client.FeatureStore",
        FakeFeatureStore,
    )

    requests = [
        PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 1)),
        PredictionRequest(ID=uuid4(), event_timestamp=datetime(2022, 1, 2)),
    ]

    client = FeatureClient(
        feature_store_repo_path="feature_repo",
        feature_service_name="transaction_v1",
    )

    result = client.get(requests)

    expected = pd.DataFrame({"feature": [1, 2]})
    tm.assert_frame_equal(result, expected)
