import pytest
from fastapi.testclient import TestClient
from tags_suggester.api import main  # à adapter selon l'emplacement de ton app FastAPI
from tags_suggester.api.main import app


def test_api_predict_endpoint():
    client = TestClient(app)
    response = client.post("/predict/", json={
        "title": "How to reverse a string in Python?",
        "body": "I want to reverse a string using built-in functions"
    })
    assert response.status_code == 200
    data = response.json()
    assert "suggested_tags" in data
    assert isinstance(data["suggested_tags"], list)
    assert all(isinstance(tag, str) for tag in data["suggested_tags"])
