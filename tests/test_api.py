from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "I am having problems with my mortgage payment."
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert "category" in data
        assert "confidence" in data

        assert isinstance(data["category"], str)
        assert isinstance(data["confidence"], float)

        assert 0 <= data["confidence"] <= 1


def test_batch_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={
                "text": [
                    "I have a problem with my mortgage payment.",
                    "Someone opened a credit card in my name.",
                    "The debt collector keeps calling me."
                ]
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert "predictions" in data
        assert len(data["predictions"]) == 3

        for prediction in data["predictions"]:
            assert "category" in prediction
            assert "confidence" in prediction

            assert isinstance(prediction["category"], str)
            assert isinstance(prediction["confidence"], float)

            assert 0 <= prediction["confidence"] <= 1


def test_predict_empty_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": ""
            }
        )

        assert response.status_code == 422


def test_batch_predict_empty():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"text": []}
        )

        assert response.status_code == 422


def test_batch_predict_text_too_long():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={
                "text": ["a" * 601]
            }
        )

        assert response.status_code == 422