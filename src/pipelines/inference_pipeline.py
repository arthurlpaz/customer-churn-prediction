# src/pipelines/inference_pipeline.py
import joblib
import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

def run_inference(input_data: dict, model_path: str):
    model = joblib.load(model_path)

    df = pd.DataFrame([input_data])
    df = preprocess_data(df)
    df = build_features(df)

    df = pd.get_dummies(df)

    expected_cols = model.feature_names_in_
    df = df.reindex(columns=expected_cols, fill_value=False)

    prediction = model.predict(df)

    return int(prediction[0])