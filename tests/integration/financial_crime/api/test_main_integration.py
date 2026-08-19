import factory
from fastapi.testclient import TestClient
from loguru import logger
import pandas as pd
import pytest
from financial_crime.api.main import app
from tests.factory.feature_factory import EngineeredFeatureFactory
from feast import FeatureStore

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
    json = [setup.model_dump(mode="json")]
    logger.error("Sending data to API: ", json)
    response = client.post("/predict", json=json)
    assert response.status_code == 200
    assert response.json() == [{"prediction": 0, "probability": 0.9990280506216597}]
