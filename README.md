# Churn Prediction API

This repository contains a FastAPI application that serves a machine learning model for customer churn prediction.

## Project Structure

```
.
├── app.py                  # FastAPI application
├── test.py                 # Tests for the API
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker file
└── Models                  # Production_model
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Running the API

1. Make sure MLflow tracking server is running
   ```
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. Start the FastAPI application
   ```
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

3. Access the API documentation at http://localhost:8000/docs

## API Endpoints

- **/** - Home endpoint
- **/health** - Health check endpoint
- **/predict** - POST endpoint for making predictions

## Making Predictions

Send a POST request to `/predict` with JSON data like:

```json
{
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
```

## Running Tests

Run the tests using pytest:

```
pytest test.py -v
```

## Model Information

This API serves the best performing model from the MLflow experiment "Churn Prediction."
It automatically loads the model with the highest accuracy from the tracking server.