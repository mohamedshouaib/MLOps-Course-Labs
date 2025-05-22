from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)



sample_input = {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000.0,
    "Gender": "Female",
    "Geography": "Spain"
}

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    print("Home Test Passed:", response.json())

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
    print("Health Test Passed:", response.json())

def test_predict(client):
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    print("Predict Test Passed:", response.json())
