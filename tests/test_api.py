from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

def test_health():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction():

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200