"""
Quick API checks using the saved model and one real dataset row.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.load_data import load_raw_data


@pytest.fixture(scope="module")
def client():
    # Context manager makes sure app startup runs.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_payload():
    # Use one real row so the endpoint gets the full feature set.
    df = load_raw_data()
    row = df.drop(columns=["Class"]).iloc[0]
    return row.to_dict()


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_name"] == "XGBoost"
    assert data["model_version"] == "v1"


def test_score_transaction_valid(client, valid_payload):
    response = client.post("/score_transaction", json=valid_payload)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["risk_score"], float)
    assert data["decision"] in {"approve", "review", "block"}
    assert data["model_name"] == "XGBoost"
    assert data["model_version"] == "v1"
    assert "timestamp" in data


def test_score_transaction_missing_feature(client, valid_payload):
    payload = dict(valid_payload)
    del payload["V28"]

    response = client.post("/score_transaction", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"][-1] == "V28" for error in errors)
