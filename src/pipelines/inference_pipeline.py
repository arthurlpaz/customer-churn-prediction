import joblib
import pandas as pd


def run_inference(input_data: dict, model_path: str):
    model = joblib.load(model_path)

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {"prediction": int(prediction), "probability": float(probability)}
