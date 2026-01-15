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

def test_rate_limiting_triggered(client):
    data = {
        "_id": ["1"],
        "name_en": ["Test Module"],
        "name_nl": ["Test Module NL"],
        "description_en": ["Description"],
        "description_nl": ["Beschrijving"],
        "shortdescription_en": ["Short"],
        "shortdescription_nl": ["Kort"],
        "studycredit": [15],
        "location": ["Breda"],
        "module_tags_en": [["tag"]],
        "module_tags_nl": [["tag"]],
        "ai_context": ["context"],
        "studycredit_num": [15]
    }
    state.df = pd.DataFrame(data)
    state.model = MagicMock()
    state.model.encode.return_value = torch.rand((1, 384))
    state.module_embeddings = torch.rand((1, 384))

    token = jwt.encode({"sub":"test","isTemp":False,"exp":time.time()+3600}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}
    
    payload = {
        "description": "Dit is een test omschrijving van voldoende lengte.",
        "current_ects": 15,
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
    }

    # Perform 6 requests (limit is 4 per minute)
    status_codes = []
    for _ in range(6):
        res = client.post("/api/predict", json=payload, headers=headers)
        status_codes.append(res.status_code)
    
    # Check if we received a 429 at least once
    assert 429 in status_codes