import os
import logging
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

# === FastAPI Setup ===
app = FastAPI(title="Churn API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Input/Output Schemas ===
class ChurnRequest(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Gender: str
    Geography: str

class ChurnResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    model_name: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# === Global Variables ===
model = None
model_name = ""
model_version = ""
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# === Preprocessing ===
def preprocess(data: ChurnRequest) -> np.ndarray:
    d = data.dict()
    d["Gender"] = 1 if d["Gender"].lower() == "male" else 0
    d["Geography_Germany"] = 1 if d["Geography"] == "Germany" else 0
    d["Geography_Spain"] = 1 if d["Geography"] == "Spain" else 0
    del d["Geography"]
    df = pd.DataFrame([d])
    ordered = ["CreditScore", "Gender", "Age", "Tenure", "Balance",
               "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
               "Geography_Germany", "Geography_Spain"]
    return df[ordered].values.astype(np.float32)

# === Load Best Model ===
def load_model():
    global model, model_name, model_version
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("Churn Prediction")
        runs = client.search_runs([exp.experiment_id], order_by=["metrics.accuracy DESC"])
        best_run = runs[0]
        run_id = best_run.info.run_id

        for name in ["xgboost", "random_forest", "logistic"]:
            try:
                uri = f"runs:/{run_id}/{name}_model"
                model = mlflow.pyfunc.load_model(uri)
                model_name = name
                model_version = run_id
                logger.info(f"Loaded {name} model.")
                return
            except:
                continue
        logger.error("No model loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# === Events ===
@app.on_event("startup")
async def startup_event():
    load_model()

# === Endpoints ===
@app.get("/", response_model=dict)
async def home():
    return {"msg": "Welcome to Churn Prediction API"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=ChurnResponse)
async def predict(req: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        X = preprocess(req)
        prob = float(model.predict(X)[0])
        return {
            "churn_probability": prob,
            "will_churn": prob > 0.5,
            "model_name": model_name,
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
