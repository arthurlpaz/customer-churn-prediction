from __future__ import annotations

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

config_path = "src/config/config.yaml"


def _threshold_objective_score(y_true, preds, metric: str, beta: float):
    if metric == "precision":
        return precision_score(y_true, preds, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, preds, zero_division=0)
    if metric == "fbeta":
        return fbeta_score(y_true, preds, beta=beta, zero_division=0)
    return f1_score(y_true, preds, zero_division=0)


def _find_best_threshold(
    y_true,
    probs,
    metric: str = "f1",
    beta: float = 1.0,
    min_recall: float = 0.0,
):
    thresholds = np.linspace(0.2, 0.9, 71)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        recall = recall_score(y_true, preds, zero_division=0)

        if recall < min_recall:
            continue

        score = _threshold_objective_score(y_true, preds, metric=metric, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    if best_score < 0:
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            score = _threshold_objective_score(y_true, preds, metric=metric, beta=beta)
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

    rf_cfg = config["model"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 1),
        class_weight=rf_cfg.get("class_weight", None),
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold_metric = config["model"].get("threshold_metric", "f1")
    threshold_beta = config["model"].get("threshold_beta", 1.0)
    min_recall = config["model"].get("min_recall_for_threshold", 0.0)

    best_threshold, best_score = _find_best_threshold(
        y_val,
        val_probs,
        metric=threshold_metric,
        beta=threshold_beta,
        min_recall=min_recall,
    )

    final_model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 1),
        class_weight=rf_cfg.get("class_weight", None),
        random_state=random_state,
    )
    final_model.fit(X_train_full, y_train_full)

    print(
        f"Best threshold on validation (metric={threshold_metric}, beta={threshold_beta}, min_recall={min_recall}): {best_threshold:.2f}"
    )
    print(f"Best validation score: {best_score:.4f}")

    return final_model, X_test, y_test, best_threshold
