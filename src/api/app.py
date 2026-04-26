from fastapi import FastAPI, HTTPException
from src.api.schema import ChurnRequest
from src.pipelines.inference_pipeline import run_inference

app = FastAPI()

MODEL_PATH = "artifacts/models/model.pkl"


@app.get("/")
def home():
    return {"message": "Churn Prediction API"}


@app.post("/predict")
def predict(data: ChurnRequest):
    try:
        return run_inference(data.dict(), MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
