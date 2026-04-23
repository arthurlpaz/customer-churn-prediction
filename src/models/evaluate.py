from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds))

    print("\n=== ROC-AUC ===")
    print(roc_auc_score(y_test, probs))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    print(f"\nThreshold used for class 1: {threshold:.2f}")
