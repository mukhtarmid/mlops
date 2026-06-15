from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_metrics():

    response = client.get(
        "/metrics"
    )

    assert response.status_code == 200