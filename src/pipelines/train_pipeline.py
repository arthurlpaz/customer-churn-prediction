import os
import yaml
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.data.load_data import load_data
from src.models.train import train_model
from src.models.evaluate import evaluate


def run_training_pipeline(config_path="src/config/config.yaml", data_path_override=None):

    mlflow.set_experiment("churn-prediction")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if data_path_override:
        config["data"]["path"] = data_path_override

    df = load_data(config["data"]["path"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    with mlflow.start_run(run_name="production_pipeline"):

        model, X_test, y_test = train_model(df, config)

        threshold = config["model"]["threshold"]

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > threshold).astype(int)

        roc_auc = roc_auc_score(y_test, probs)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(config["model"]["xgboost"])
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("recall_class_1", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        output_path = config["output"]["model_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        joblib.dump(model, output_path)

        evaluate(model, X_test, y_test, threshold)
