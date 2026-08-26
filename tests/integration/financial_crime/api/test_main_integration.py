from fastapi.testclient import TestClient
import pandas as pd
import pytest
from feast import FeatureStore

from financial_crime.api.main import app
from tests.factory.feature_factory import EngineeredFeatureFactory

client = TestClient(app)


@pytest.fixture
def setup():
    engineered_feature = EngineeredFeatureFactory.build()

    df_engineered = pd.DataFrame([
        engineered_feature.model_dump(mode="python", by_alias=True)
    ])
    feature_store = FeatureStore("financial_crime/feature_store/feature_repo")
    feature_store.push("transaction_push_source", df_engineered)

    return engineered_feature


def test_predict(setup):
    payload = [setup.model_dump(mode="json")]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert type(response_json) == list
    assert len(response_json) == 1
    assert response_json[0]['prediction'] == 0
    assert 0.0 <= response_json[0]['probability'] <= 1.0
