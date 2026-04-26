from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds))

    print("\n=== ROC-AUC ===")
    print(roc_auc_score(y_test, probs))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    print(f"\nClass 1 precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Class 1 recall: {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"Threshold used for class 1: {threshold:.2f}")
