import joblib
import os
import yaml

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.validate import validate_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate
from src.models.train import train_model


def run_training_pipeline(config_path="src/config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = load_data(config["data"]["path"])
    df = preprocess_data(df)
    df = build_features(df)

    validate_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    model, X_test, y_test, best_threshold = train_model(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )

    evaluate(model, X_test, y_test, threshold=best_threshold)

    os.makedirs(os.path.dirname(config["output"]["model_path"]), exist_ok=True)
    joblib.dump(model, config["output"]["model_path"])

    print("--- Model saved ---")
