import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
import jwt
import time
from app.main import app, state
from app.config import settings

# Disable rate limiting for integration tests
app.state.limiter.enabled = False

@pytest.fixture # client fixture for reuse
def client():
    with patch("app.main.lifespan", side_effect=lambda app: MagicMock()):
        with TestClient(app) as c:
            yield c

@pytest.fixture # mock authentication token
def mock_auth_token():
    payload = {"sub": "test_user", "isTemp": False, "exp": time.time() + 3600}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

@pytest.fixture # mock state data
def mock_state_data():
    data = {
        "_id": ["1", "2"], "name_en": ["AI Basics", "Web Dev"], "name_nl": ["AI Basis", "Web Ontwikkeling"],
        "description_en": ["D1", "D2"], "description_nl": ["D1 NL", "D2 NL"],
        "shortdescription_en": ["S1", "S2"], "shortdescription_nl": ["S1 NL", "S2 NL"],
        "studycredit": [15, 30], "location": ["Breda", "Eindhoven"],
        "module_tags_en": [["ai"], ["web"]], "module_tags_nl": [["ai"], ["web"]],
        "ai_context": ["c1", "c2"], "studycredit_num": [15, 30]
    }
    state.df = pd.DataFrame(data)
    state.model = MagicMock()
    state.model.encode.return_value = torch.rand((1, 384))
    state.module_embeddings = torch.rand((2, 384))
    yield
    state.df = None

def test_predict_endpoint_success(client, mock_state_data, mock_auth_token):
    """Integration test: Test successful request and language output."""
    payload = {"description": "Learning about AI.", "current_ects": 15, "tags": ["t1","t2","t3","t4","t5"]}
    headers = {"Authorization": f"Bearer {mock_auth_token}"}
    response = client.post("/api/predict?language=NL", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["recommendations"][0]["Module_Name"] == "AI Basis"

def test_predict_validation_errors(client, mock_auth_token):
    """Integration test: Bundles all 422 validation errors (ECTS, Language, Length)."""
    headers = {"Authorization": f"Bearer {mock_auth_token}"}
    
    # 1. Wrong ECTS
    res1 = client.post("/api/predict", json={"description": "Test", "current_ects": 24}, headers=headers)
    assert res1.status_code == 422
    
    # 2. Invalid language
    res2 = client.post("/api/predict?language=FR", json={"description": "Test", "current_ects": 15}, headers=headers)
    assert res2.status_code == 422
    
    # 3. Too long description
    res3 = client.post("/api/predict", json={"description": "a" * 1001, "current_ects": 15}, headers=headers)
    assert res3.status_code == 422

    # 4. Too short description
    res4 = client.post("/api/predict", json={"description": "Short", "current_ects": 15}, headers=headers)
    assert res4.status_code == 422

    # 5. Too few tags
    res5 = client.post("/api/predict", json={"description": "Valid description.", "current_ects": 15, "tags": ["t1"]}, headers=headers)
    assert res5.status_code == 422

    # 6. Too many tags
    res6 = client.post("/api/predict", json={"description": "Valid description.", "current_ects": 15, "tags": ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12","t13","t14","t15","t16"]}, headers=headers)
    assert res6.status_code == 422

    # 7. Missing description
    res7 = client.post("/api/predict", json={"current_ects": 15}, headers=headers)
    assert res7.status_code == 422

    # 8. Missing current_ects
    res8 = client.post("/api/predict", json={"description": "Valid description."}, headers=headers)
    assert res8.status_code == 422

    # 9. Missing tags
    res9 = client.post("/api/predict", json={"description": "Valid description.", "current_ects": 15}, headers=headers)
    assert res9.status_code == 422

    # 10. Non-integer current_ects
    res10 = client.post("/api/predict", json={"description": "Valid description.", "current_ects": "fifteen"}, headers=headers)
    assert res10.status_code == 422

    # 11. Non-list tags
    res11 = client.post("/api/predict", json={"description": "Valid description.", "current_ects": 15, "tags": "notalist"}, headers=headers)
    assert res11.status_code == 422

    # 12. Non-string description
    res12 = client.post("/api/predict", json={"description": 12345, "current_ects": 15}, headers=headers)
    assert res12.status_code == 422

def test_predict_missing_auth(client):
    """Integration test: Test blocking when authentication is missing."""
    response = client.post("/api/predict", json={})
    assert response.status_code in [401, 403]

def test_model_not_ready_503(client, mock_auth_token):
    """Integration test: Test behavior when database/model is not loaded."""
    state.df = None
    payload = {"description": "Valid desc", "current_ects": 15, "tags": ["t1","t2","t3","t4","t5"]}
    response = client.post("/api/predict", json=payload, headers={"Authorization": f"Bearer {mock_auth_token}"})
    assert response.status_code == 503