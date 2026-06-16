from fastapi.testclient import TestClient
from unittest.mock import patch

from src.app import app

client = TestClient(app)


def test_health():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


@patch("src.app.predict")
def test_prediction(mock_predict):

    mock_predict.return_value = "setosa"

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    assert response.json() == {
        "prediction": "setosa"
    }

    mock_predict.assert_called_once()