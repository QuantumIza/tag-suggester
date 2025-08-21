import pytest
from fastapi.testclient import TestClient
from tags_suggester.api import main  # à adapter selon l'emplacement de ton app FastAPI
from tags_suggester.api.main import app

# --- TEST BON FONCTIONNEMENT ENDPOINT
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

# ---------------------
# --- TESTs ROBUSTESSE
# ---------------------
# --- teste que l'api gere bien les champs manquants
def test_missing_fields():
    client = TestClient(app)
    response = client.post("/predict/", json={"title": "Only title"})
    assert response.status_code == 422  # Unprocessable Entity

# --- teste que l’API gère les inputs vides correctement
def test_empty_fields():
    client = TestClient(app)
    response = client.post("/predict/", json={"title": "", "body": ""})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["suggested_tags"], list)

# --- Simule une question très détaillée pour tester les limites
def test_long_input():
    client = TestClient(app)
    long_body = " ".join(["This is a long body."] * 1000)
    response = client.post("/predict/", json={
        "title": "Long question",
        "body": long_body
    })
    assert response.status_code == 200
    data = response.json()
    assert "suggested_tags" in data

# --- teste que le modèle gère bien les encodages
def test_unicode_input():
    client = TestClient(app)
    response = client.post("/predict/", json={
        "title": "Comment inverser une chaîne ? 🤔",
        "body": "J'utilise Python pour manipuler des chaînes de caractères."
    })
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["suggested_tags"], list)
