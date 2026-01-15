import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
import jwt
import time
from app.main import app, state
from app.config import settings

# Disable the limiter for these tests, because we are not testing rate limiting here
app.state.limiter.enabled = False

@pytest.fixture
def client():
    with patch("app.main.lifespan", side_effect=lambda app: MagicMock()):
        with TestClient(app) as c:
            yield c

@pytest.fixture
def mock_auth_token():
    payload = {"sub": "test_user", "isTemp": False, "exp": time.time() + 3600}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

@pytest.fixture
def mock_state_data():
    """Zorgt voor alle kolommen die predictor.py en loader.py nodig hebben."""
    data = {
        "_id": ["1", "2"],
        "name_en": ["AI Basics", "Web Dev"],
        "name_nl": ["AI Basis", "Web Ontwikkeling"],
        "description_en": ["Intro to AI.", "Web dev course."],
        "description_nl": ["Intro AI.", "Web dev cursus."],
        "shortdescription_en": ["AI info", "Web info"],
        "shortdescription_nl": ["AI kort", "Web kort"],
        "studycredit": [15, 30],
        "location": ["Eindhoven", "Breda"],
        "module_tags_en": [["ai", "tech"], ["web", "design"]],
        "module_tags_nl": [["ai", "tech"], ["web", "design"]],
        "ai_context": ["context1", "context2"]
    }
    df = pd.DataFrame(data)
    # Mock the numeric column if the predictor uses it
    df['studycredit_num'] = df['studycredit'].astype(int)
    
    state.df = df
    state.model = MagicMock()
    # Return an embedding vector
    state.model.encode.return_value = torch.rand((1, 384))
    state.module_embeddings = torch.rand((2, 384))
    yield
    state.df = None

def test_predict_endpoint_success(client, mock_state_data, mock_auth_token):
    payload = {
        "description": "I want to learn about artificial intelligence.",
        "current_ects": 15,
        "tags": ["python", "coding", "AI", "tech", "logic"]
    }
    headers = {"Authorization": f"Bearer {mock_auth_token}"}
    response = client.post("/api/predict", json=payload, headers=headers)
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_language_parameter_en(client, mock_state_data, mock_auth_token):
    payload = {
        "description": "I want to study in English.",
        "current_ects": 30,
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
    }
    headers = {"Authorization": f"Bearer {mock_auth_token}"}
    response = client.post("/api/predict?language=EN", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["language"] == "EN"

def test_predict_invalid_ects(client, mock_auth_token):
    # ECTS must be 15 or 30 according to your validator
    payload = {"description": "Long description...", "current_ects": 20, "tags": ["a","b","c","d","e"]}
    headers = {"Authorization": f"Bearer {mock_auth_token}"}
    response = client.post("/api/predict", json=payload, headers=headers)
    assert response.status_code == 422

def test_predict_missing_auth(client):
    # Test without Authorization header
    response = client.post("/api/predict", json={})
    assert response.status_code in [401, 403]

def test_model_not_ready_503(client, mock_auth_token):
    # Force the state to 'not ready'
    state.df = None
    payload = {"description": "Lange omschrijving...", "current_ects": 15, "tags": ["a","b","c","d","e"]}
    headers = {"Authorization": f"Bearer {mock_auth_token}"}
    response = client.post("/api/predict", json=payload, headers=headers)
    assert response.status_code == 503