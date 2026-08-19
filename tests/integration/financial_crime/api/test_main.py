from fastapi.testclient import TestClient
import pytest
from financial_crime.api.main import app

client = TestClient(app)

@pytest.fixture
def setup():
    # TODO I need to add factoryboy, create a model for an engineered feature and then put it into the feature store and invoke the endpoint
    return

def test_predict(setup):
    json = {""}
    response = client.post("/predict", json=json)
    assert response.status_code == 200
    assert response.json() == {"item_id": 42, "valid": True}
