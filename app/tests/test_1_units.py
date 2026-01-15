import pytest
from fastapi.testclient import TestClient
import pandas as pd
import torch
import jwt
from unittest.mock import MagicMock, patch

# Import app components
from app.services.auth import verify_token
from app.main import app
from app.config import settings
from app.services.state import state
from app.utils import sanitize_text

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_mock_state():
    """
    Fixture to populate the 'state' with mock data before each test.
    This prevents the need to load the actual heavy AI model.
    """
    # 1. Mock the AI model
    mock_model = MagicMock()
    # Simulate an embedding vector (e.g., 1x384)
    mock_model.encode.return_value = torch.zeros((1, 384))
    state.model = mock_model

    # 2. Mock the DataFrame (simulated database content)
    state.df = pd.DataFrame([{
        "_id": "123",
        "name_en": "Test Module",
        "name_nl": "Test Module NL",
        "description_en": "This is a test description",
        "description_nl": "Dit is een test beschrijving",
        "shortdescription_en": "Short test",
        "shortdescription_nl": "Korte test",
        "module_tags_en": ["test", "ai"],
        "module_tags_nl": ["test", "ai"],
        "studycredit": 15,
        "location": "Breda",
        "ai_context": "Test context"
    }])

    # 3. Mock the database embeddings
    state.module_embeddings = torch.zeros((1, 384))
    
    yield

def test_predict_study_success():
    """Test if the predict endpoint works with a valid token and correct filters."""
    # Override authentication dependency
    app.dependency_overrides[verify_token] = lambda: {"sub": "test_user"}

    test_payload = {
        "description": "I want to learn about AI and data science.",
        "preferred_location": "Breda",
        "current_ects": 15,
        "tags": ["ai", "data science", "machine learning", "python", "statistics"]
    }

    response = client.post("/api/predict?language=NL", json=test_payload)

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    assert data["recommendations"][0]["Module_Name"] == "Test Module NL"

    app.dependency_overrides = {}

def test_predict_rejects_temp_token():
    """Test that the API rejects a temporary 2FA token (isTemp: true)."""
    # We DO NOT override dependencies here because we want to test the real verify_token logic
    
    # 1. Create a fake temp token
    payload = {"userId": "user123", "isTemp": True}
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    
    test_payload = {
        "description": "Test",
        "current_ects": 15
    }

    # 2. Use the token in a cookie (as per auth.py logic)
    client.cookies.set("token", token)
    
    response = client.post("/api/predict", json=test_payload)
    
    # 3. Expect 403 Forbidden
    assert response.status_code == 403
    assert response.json()["detail"] == "Full authentication required (2FA not completed)"

def test_predict_study_no_results_for_ects():
    """Test if the API returns 422 on a unvalid ECT number."""
    app.dependency_overrides[verify_token] = lambda: {"sub": "test_user"}

    test_payload = {
        "description": "Random description",
        "current_ects": 24,
    }

    response = client.post("/api/predict?language=NL", json=test_payload)
    
    assert response.status_code == 422
    
    app.dependency_overrides = {}

def test_predict_invalid_language_fallback():
    """Test how the API handles unsupported language parameters."""
    app.dependency_overrides[verify_token] = lambda: {"sub": "test_user"}

    test_payload = {"description": "Test", "current_ects": 5}
    
    # Request with an unsupported language (e.g., French 'FR')
    response = client.post("/api/predict?language=FR", json=test_payload)
    
    assert response.status_code == 422
    
    app.dependency_overrides = {}

def test_predict_description_too_long():
    """Test if the API rejects descriptions exceeding the character limit."""
    app.dependency_overrides[verify_token] = lambda: {"sub": "test_user"}
    
    long_payload = {
        "description": "a" * 1001, 
        "current_ects": 5
    }
    response = client.post("/api/predict", json=long_payload)
    
    assert response.status_code == 422
    app.dependency_overrides = {}

def test_sanitize_text_strips_html():
    """Test if the utility function correctly removes HTML tags."""
    dirty_html = "<script>alert('xss')</script>Hello <b>World</b>"
    clean_text = sanitize_text(dirty_html)
    assert clean_text == "alert('xss')Hello World"
    assert "<script>" not in clean_text