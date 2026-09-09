import pytest
from fastapi.testclient import TestClient

from prodml.predict import DurationPredictor
from prodml.config import settings
from prodml.api.main import app


@pytest.fixture
def sample_features():
    return {"PU_DO": "74_42", "trip_distance": 3.2}


@pytest.fixture(scope="session")
def trained_model():
    return DurationPredictor.load(settings.model_path)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
