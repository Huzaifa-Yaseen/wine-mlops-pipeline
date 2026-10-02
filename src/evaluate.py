"""Evaluation and Model Registry Champion registration script."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, log_loss
from src.data import get_train_test_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
DB_URI = "sqlite:///mlflow.db"


def register_and_promote_champion():
    """Find best run by validation macro F1-score and assign champion alias."""
    mlflow.set_tracking_uri(DB_URI)
    client = MlflowClient(tracking_uri=DB_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_macro_f1 DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No MLflow runs found to evaluate.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("cv_macro_f1")
    model_family = best_run.data.tags.get("model_family")

    print(f"Top CV Run ID: {best_run_id}")
    print(f"Model Family: {model_family}")
    print(f"Best CV Macro F1: {best_f1:.4f}")

    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=reg_model.version,
    )
    print(f"Successfully registered version {reg_model.version} with alias '{CHAMPION_ALIAS}'.")
    return reg_model.version


def evaluate_champion():
    """Load the champion model by alias and evaluate on test set."""
    mlflow.set_tracking_uri(DB_URI)
    _, X_test, _, y_test = get_train_test_data()

    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    champion_model = mlflow.sklearn.load_model(model_uri)

    preds = champion_model.predict(X_test)
    probs = champion_model.predict_proba(X_test)

    test_f1 = f1_score(y_test, preds, average="macro")
    test_acc = accuracy_score(y_test, preds)
    test_loss = log_loss(y_test, probs, labels=[0, 1, 2])

    print("\n--- Final Champion Test Split Metrics ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test Log Loss: {test_loss:.4f}")

    return test_f1, test_acc, test_loss


if __name__ == "__main__":
    register_and_promote_champion()
    evaluate_champion()
