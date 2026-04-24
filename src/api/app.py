from fastapi import FastAPI
from src.pipelines.inference_pipeline import run_inference

app = FastAPI()

MODEL_PATH = "artifacts/models/model.pkl"


@app.get("/")
def home():
    return {"message": "Churn Prediction API"}


@app.post("/predict")
def predict(data: dict):
    prediction = run_inference(data, MODEL_PATH)
    return {"churn_prediction": prediction}
