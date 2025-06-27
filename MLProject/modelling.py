import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes_processing.csv")
    data = pd.read_csv(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("isDiabetes", axis=1),
        data["isDiabetes"],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]

    # Read CLI parameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)

        # Log model with input example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Log metrics
        acc = model.score(X_test, y_test)
        prec = precision_score(y_test, predicted, average='weighted')
        rec = recall_score(y_test, predicted, average='weighted')
        f1 = f1_score(y_test, predicted, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Save confusion matrix
        os.makedirs("cm", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        plt.title("Confusion Matrix")
        cm_path = os.path.join("cm", "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # Save metrics to JSON
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        json_path = os.path.join("cm", "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f)
        mlflow.log_artifact(json_path)

        print(f"Run complete. Accuracy: {acc:.4f}")
