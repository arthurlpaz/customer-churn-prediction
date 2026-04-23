from __future__ import annotations

from itertools import product

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
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
    thresholds = np.linspace(0.2, 0.95, 76)
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


def _build_rf(params: dict, random_state: int):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        max_features=params["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )


def _fit_model(params: dict, calibration_method: str, calibration_cv: int, random_state: int):
    base_model = _build_rf(params, random_state=random_state)
    if calibration_method == "none":
        return base_model

    return CalibratedClassifierCV(
        estimator=base_model,
        method=calibration_method,
        cv=calibration_cv,
        n_jobs=-1,
    )


def _search_best_rf_params(X_train, y_train, X_val, y_val, random_state, config):
    search_cfg = config["model"].get("random_forest_search", {})
    threshold_metric = config["model"].get("threshold_metric", "f1")
    threshold_beta = config["model"].get("threshold_beta", 1.0)
    min_recall = config["model"].get("min_recall_for_threshold", 0.0)

    if not search_cfg.get("enabled", False):
        rf_cfg = config["model"]["random_forest"]
        params = {
            "n_estimators": rf_cfg["n_estimators"],
            "max_depth": rf_cfg["max_depth"],
            "min_samples_leaf": rf_cfg.get("min_samples_leaf", 1),
            "class_weight": rf_cfg.get("class_weight", None),
            "max_features": rf_cfg.get("max_features", "sqrt"),
        }
        calibration_method = config["model"].get("calibration", {}).get("method", "none")
        calibration_cv = config["model"].get("calibration", {}).get("cv", 3)

        model = _fit_model(
            params,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        best_threshold, best_score = _find_best_threshold(
            y_val,
            val_probs,
            metric=threshold_metric,
            beta=threshold_beta,
            min_recall=min_recall,
        )
        return params, calibration_method, best_threshold, best_score

    n_estimators_grid = search_cfg.get("n_estimators", [200, 300, 500])
    max_depth_grid = search_cfg.get("max_depth", [10, 12, 16])
    min_samples_leaf_grid = search_cfg.get("min_samples_leaf", [1, 2, 4])
    class_weight_grid = search_cfg.get("class_weight", [None, "balanced"])
    max_features_grid = search_cfg.get("max_features", ["sqrt", "log2", None])
    calibration_methods = search_cfg.get("calibration_method", ["none", "sigmoid"])
    calibration_cv = config["model"].get("calibration", {}).get("cv", 3)

    best_params = None
    best_calibration = "none"
    best_threshold = 0.5
    best_score = -1.0

    for values in product(
        n_estimators_grid,
        max_depth_grid,
        min_samples_leaf_grid,
        class_weight_grid,
        max_features_grid,
        calibration_methods,
    ):
        params = {
            "n_estimators": values[0],
            "max_depth": values[1],
            "min_samples_leaf": values[2],
            "class_weight": values[3],
            "max_features": values[4],
        }
        calibration_method = values[5]

        model = _fit_model(
            params,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]

        threshold, score = _find_best_threshold(
            y_val,
            val_probs,
            metric=threshold_metric,
            beta=threshold_beta,
            min_recall=min_recall,
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_params = params
            best_calibration = calibration_method

    return best_params, best_calibration, best_threshold, best_score


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

    best_params, best_calibration, best_threshold, best_score = _search_best_rf_params(
        X_train, y_train, X_val, y_val, random_state=random_state, config=config
    )

    calibration_cv = config["model"].get("calibration", {}).get("cv", 3)
    final_model = _fit_model(
        best_params,
        calibration_method=best_calibration,
        calibration_cv=calibration_cv,
        random_state=random_state,
    )
    final_model.fit(X_train_full, y_train_full)

    threshold_metric = config["model"].get("threshold_metric", "f1")
    threshold_beta = config["model"].get("threshold_beta", 1.0)
    min_recall = config["model"].get("min_recall_for_threshold", 0.0)

    print(f"Best RF params on validation: {best_params}")
    print(f"Best calibration on validation: {best_calibration}")
    print(
        f"Best threshold on validation (metric={threshold_metric}, beta={threshold_beta}, min_recall={min_recall}): {best_threshold:.2f}"
    )
    print(f"Best validation score: {best_score:.4f}")

    return final_model, X_test, y_test, best_threshold
