"""Automated MLOps Quality Gate checks for champion model validation."""
import time
import numpy as np
import pytest
import mlflow
from src.data import get_train_test_data

DB_URI = "sqlite:///mlflow.db"
MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"


@pytest.fixture(scope="module")
def champion_model():
    """Load the champion model version from MLflow Model Registry."""
    mlflow.set_tracking_uri(DB_URI)
    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


@pytest.fixture(scope="module")
def evaluation_data():
    """Load test features and ground truth labels."""
    _, X_test, _, y_test = get_train_test_data()
    return X_test, y_test


def test_metric_threshold_gate(champion_model, evaluation_data):
    """Quality Gate 1: Ensure validation/test Macro F1-score >= 0.88."""
    from sklearn.metrics import f1_score
    X_test, y_test = evaluation_data
    preds = champion_model.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro")
    assert macro_f1 >= 0.88, f"F1 score {macro_f1:.4f} below quality threshold of 0.88"


def test_inference_latency_gate(champion_model, evaluation_data):
    """Quality Gate 2: Ensure batch inference latency <= 30 ms."""
    X_test, _ = evaluation_data

    # Warmup invocation
    _ = champion_model.predict(X_test)

    start_time = time.perf_counter()
    _ = champion_model.predict(X_test)
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0

    assert elapsed_time_ms <= 30.0, f"Latency {elapsed_time_ms:.2f}ms exceeded 30ms limit"


def test_output_schema_integrity_gate(champion_model, evaluation_data):
    """Quality Gate 3: Output must contain valid class indices strictly in {0, 1, 2}."""
    X_test, _ = evaluation_data
    preds = champion_model.predict(X_test)

    valid_classes = {0, 1, 2}
    unique_preds = set(np.unique(preds))

    assert unique_preds.issubset(valid_classes), f"Unexpected classes predicted: {unique_preds}"
    assert len(preds) == len(X_test), "Prediction shape mismatch with input test batch"
