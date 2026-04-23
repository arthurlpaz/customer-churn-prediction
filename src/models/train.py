from __future__ import annotations

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import train_test_split

config_path = "src/config/config.yaml"


def _find_best_threshold(y_true, probs, beta: float = 1.0):
    thresholds = np.linspace(0.2, 0.8, 61)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        score = fbeta_score(y_true, preds, beta=beta) if beta != 1 else f1_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def train_model(X, y, test_size=0.2, random_state=42):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    val_size = config["model"].get("validation_size", 0.2)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["random_forest"]["n_estimators"],
        max_depth=config["model"]["random_forest"]["max_depth"],
        min_samples_leaf=config["model"]["random_forest"].get("min_samples_leaf", 1),
        class_weight=config["model"]["random_forest"].get("class_weight", None),
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold_beta = config["model"].get("threshold_beta", 1.0)
    best_threshold, best_score = _find_best_threshold(y_val, val_probs, beta=threshold_beta)

    final_model = RandomForestClassifier(
        n_estimators=config["model"]["random_forest"]["n_estimators"],
        max_depth=config["model"]["random_forest"]["max_depth"],
        min_samples_leaf=config["model"]["random_forest"].get("min_samples_leaf", 1),
        class_weight=config["model"]["random_forest"].get("class_weight", None),
        random_state=random_state,
    )
    final_model.fit(X_train_full, y_train_full)

    print(f"Best threshold on validation (beta={threshold_beta}): {best_threshold:.2f}")
    print(f"Best validation score: {best_score:.4f}")

    return final_model, X_test, y_test, best_threshold
