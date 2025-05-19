import logging
import os
import json
from typing import Dict, Any, List, Union
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import mlflow
import pandas as pd
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_logs.log")
    ]
)
logger = logging.getLogger("churn-prediction-api")

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using machine learning model",
    version="1.0.0"
)

# Path to your MLflow tracking server
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Define the input schema based on your data
class ChurnPredictionInput(BaseModel):
    CreditScore: int = Field(..., description="Customer's credit score")
    Age: int = Field(..., description="Customer's age")
    Tenure: int = Field(..., description="Number of years as a customer")
    Balance: float = Field(..., description="Account balance")
    NumOfProducts: int = Field(..., description="Number of bank products used")
    HasCrCard: int = Field(..., example=1, description="Has credit card (1=Yes, 0=No)")
    IsActiveMember: int = Field(..., example=1, description="Is active member (1=Yes, 0=No)")
    EstimatedSalary: float = Field(..., description="Estimated salary")
    Gender: int = Field(..., example=1, description="Gender (1=Male, 0=Female)")
    Geography_Germany: int = Field(..., example=0, description="Customer from Germany (1=Yes, 0=No)")
    Geography_Spain: int = Field(..., example=1, description="Customer from Spain (1=Yes, 0=No)")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Churn prediction (1=Will Churn, 0=Won't Churn)")
    probability: float = Field(..., description="Probability of churn")

# Model loading function
def load_model():
    """Load the best model from MLflow."""
    logger.info("Loading model from MLflow")
    
    try:
        # Connect to MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # The run_id should be retrieved from the best model - this is a placeholder
        # You should find your best model's run ID from the MLflow UI or API
        # For now, let's assume XGBoost is the best model based on your metrics
        
        # Listing all runs to find the best model
        experiment = mlflow.get_experiment_by_name("Churn Prediction")
        if experiment is None:
            error_msg = "Experiment 'Churn Prediction' not found"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            error_msg = "No runs found in experiment 'Churn Prediction'"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Find the run with the best accuracy (you might use a different metric)
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        best_run_id = best_run.run_id
        
        logger.info(f"Found best model run_id: {best_run_id}")
        
        # Load the model from the best run
        model_uri = f"runs:/{best_run_id}/xgboost_model" 
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info("Model loaded successfully")
        return loaded_model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

# Initialize model at startup
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    logger.info("Starting up the API server")
    try:
        model = load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        # Continue startup but model will be None
        # We'll handle this in the predict endpoint

@app.get("/", tags=["General"])
async def home():
    """Home endpoint for the API."""
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Churn Prediction API", 
            "docs": "/docs", 
            "health": "/health", 
            "predict": "/predict"}

@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    global model
    
    if model is None:
        try:
            model = load_model()
            status = "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            status = "unhealthy"
    else:
        status = "healthy"
        
    return {"status": status, "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: ChurnPredictionInput):
    """Predict churn probability for a customer."""
    logger.info("Predict endpoint accessed")
    
    # Ensure model is loaded
    global model
    if model is None:
        try:
            logger.info("Model not loaded, attempting to load model")
            model = load_model()
        except Exception as e:
            error_msg = f"Model could not be loaded: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
    
    try:
        # Convert input data to DataFrame for prediction
        input_dict = input_data.dict()
        logger.info(f"Received prediction request with data: {input_dict}")
        
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(df)
        
        # For this example, we'll assume model.predict gives us the class label
        # If your model can return probabilities, you should extract those too
        prediction_result = int(prediction[0])
        
        # Try to get probability if model supports it
        try:
            # This works if model.predict_proba is available
            probability = float(model.predict_proba(df)[0][1])
        except:
            # Fallback to a dummy probability if not available
            probability = 1.0 if prediction_result == 1 else 0.0
            
        logger.info(f"Prediction result: {prediction_result}, probability: {probability}")
        
        return {"prediction": prediction_result, "probability": probability}
    
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
