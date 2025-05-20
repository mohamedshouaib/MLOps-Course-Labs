import requests

base_url = "http://127.0.0.1:8000"

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

def test_home():
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200
    print("Home Test Passed:", response.json())

def test_health():
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
    print("Health Test Passed:", response.json())

def test_predict():
    response = requests.post(f"{base_url}/predict", json=sample_input)
    assert response.status_code == 200
    print("Predict Test Passed:", response.json())

if __name__ == "__main__":
    test_home()
    test_health()
    test_predict()