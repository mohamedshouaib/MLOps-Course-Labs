import os
import logging
import numpy as np
import pandas as pd
import pickle
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
    
    model_config = {"protected_namespaces": ()}

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
    model_config = {"protected_namespaces": ()}

# === Global Variables ===
model = None
model_name = ""
model_version = ""
MODEL_DIR = "models"  

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

# === Load Model from Local File ===
def load_model():
    global model, model_name, model_version
    priority_models = ["random_forest_model.pkl"]

    for file_name in priority_models:
        model_path = os.path.join(MODEL_DIR, file_name)
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                model_name = file_name.split("_model.pkl")[0]
                model_version = "local"
                logger.info(f"Loaded local model: {file_name}")
                return
            except Exception as e:
                logger.error(f"Failed to load model {file_name}: {e}")
    
    logger.error("No valid model found in local directory.")

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
        prob = float(model.predict_proba(X)[0][1])  
        return {
            "churn_probability": prob,
            "will_churn": prob > 0.5,
            "model_name": model_name,
            "model_version": model_version
        }
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))