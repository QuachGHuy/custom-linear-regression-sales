import sys
import os
import numpy as np

from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schemas import Predict, Response
from src.data_loader import data_scaler, add_bias


model_artifacts = {
    "w": None,
    "mean": None,
    "std": None
}

@asynccontextmanager
async def lifespan(app : FastAPI):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "models")
    w_path = os.path.join(model_dir, "weights.npy")
    mean_path = os.path.join(model_dir, "mean.npy")
    std_path = os.path.join(model_dir, "std.npy")

    try:
        print("Loading model artifacts...")
        model_artifacts["w"] = np.load(w_path)
        model_artifacts["mean"] = np.load(mean_path)
        model_artifacts["std"] = np.load(std_path)
        print("Model loaded successfully!")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Missing model file: {e.filename}")
        print("Please run 'python scripts/train.py' first.")
        
        raise e
    
    yield
    
app = FastAPI(title="Sales Prediction API", lifespan=lifespan)

@app.get("/")
def heath_check():
    # Check if server started successfully
    return {"status": "ok", "message": "Sales Prediction API is running"}

@app.post("/predict", response_model=Response, status_code=status.HTTP_200_OK)
def predict(data: Predict):
    # Check if model artifacts not exists: 
    if model_artifacts["w"] is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Model not loaded. Check server logs."
            )
    try:
        # 1. Get model artifacts
        w = model_artifacts["w"]
        mean = model_artifacts["mean"]
        std = model_artifacts["std"]

        # 2. Get input data
        X_input = np.array([[data.tv, data.radio, data.newspaper]])

        # 3. Data Scaling
        X_input, _, _ = data_scaler(X_input,mean,std)

        # 4. Add Bias
        X_input = add_bias(X_input)

        # 5. Predict
        y_predict = X_input @ w

        # 6. Get a result
        result = float(y_predict[0])

        return Response(sales_prediction=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

        

