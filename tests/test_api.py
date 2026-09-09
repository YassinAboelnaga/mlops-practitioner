def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_happy_path(client, sample_features):
    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "correlation_id" in body
    assert "latency_ms" in body


def test_predict_invalid_payload_returns_422(client):
    response = client.post("/predict", json={"PU_DO": "74_42", "trip_distance": -5})
    assert response.status_code == 422


def test_predict_mocked_model(client, sample_features, monkeypatch):
    from prodml.api import main as main_module

    class FakePredictor:
        def predict_one(self, features):
            return 12.5

    monkeypatch.setitem(main_module.ml_state, "predictor", FakePredictor())

    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200
    assert response.json()["prediction"] == 12.5
