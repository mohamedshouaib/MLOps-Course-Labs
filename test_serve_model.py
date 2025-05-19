import pytest
from fastapi.testclient import TestClient
import mlflow
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from app import app, load_model

client = TestClient(app)

# Mock model for testing
class MockModel:
    def predict(self, df):
        # Always predict 0 for testing
        return np.array([0])
    
    def predict_proba(self, df):
        # Return fake probabilities for testing
        return np.array([[0.8, 0.2]])

@pytest.fixture
def mock_mlflow(monkeypatch):
    """Fixture to mock MLflow interactions."""
    # Create mock experiment
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "fake_exp_id"
    
    # Create mock runs DataFrame
    fake_runs = pd.DataFrame({
        'run_id': ['fake_run_id'],
        'metrics.accuracy': [0.95]
    })
    
    # Mock the MLflow functions
    with patch('mlflow.get_experiment_by_name') as mock_get_exp:
        mock_get_exp.return_value = mock_experiment
        with patch('mlflow.search_runs') as mock_search:
            mock_search.return_value = fake_runs
            with patch('mlflow.pyfunc.load_model') as mock_load:
                mock_load.return_value = MockModel()
                yield

def test_home():
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Churn Prediction API" in response.json()["message"]

def test_health(mock_mlflow):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]

def test_predict(mock_mlflow):
    """Test the predict endpoint."""
    # Create test input data
    test_data = {
        "CreditScore": 700,
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0,
        "Gender": 1,
        "Geography_Germany": 0,
        "Geography_Spain": 1
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], float)

def test_predict_missing_data():
    """Test the predict endpoint with missing data."""
    # Missing required fields
    incomplete_data = {
        "CreditScore": 700,
        "Age": 35
        # Missing other required fields
    }
    
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_invalid_data_types():
    """Test the predict endpoint with invalid data types."""
    # Invalid data types
    invalid_data = {
        "CreditScore": "not_a_number",  # Should be an integer
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0,
        "Gender": 1,
        "Geography_Germany": 0,
        "Geography_Spain": 1
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity
