import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
import jwt
import time
from app.main import app, state
from app.config import settings

app.state.limiter.enabled = True

@pytest.fixture
def client():
    with patch("app.main.lifespan", side_effect=lambda app: MagicMock()):
        with TestClient(app) as c:
            yield c

def test_predict_rejects_temp_token(client):
    """Security: Test if requests with temporary tokens are rejected."""
    payload = {"sub": "user123", "isTemp": True}
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    
    client.cookies.set("token", token)
    
    test_payload = {"description": "Valid description length.", "current_ects": 15}
    response = client.post("/api/predict", json=test_payload)
    
    assert response.status_code == 403
    assert "2FA not completed" in response.json()["detail"]

def test_rate_limiting_triggered(client):
    """Security: Test if the 4 requests per minute limit works."""
    # Setup mock data
    state.df = pd.DataFrame({"_id":["1"],"name_en":["T"],"name_nl":["T"],"description_en":["D"],"description_nl":["D"],"shortdescription_en":["S"],"shortdescription_nl":["S"],"studycredit":[15],"location":["X"],"module_tags_en":[["a"]],"module_tags_nl":[["a"]],"ai_context":["C"],"studycredit_num":[15]})
    state.model = MagicMock()
    state.model.encode.return_value = torch.rand((1, 384))
    state.module_embeddings = torch.rand((1, 384))

    token = jwt.encode({"sub":"test","isTemp":False,"exp":time.time()+3600}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"description": "Genoeg tekst om validatie te passeren.", "current_ects": 15, "tags": ["t1","t2","t3","t4","t5"]}

    status_codes = []
    for _ in range(6):
        res = client.post("/api/predict", json=payload, headers=headers)
        status_codes.append(res.status_code)
    
    assert 429 in status_codes