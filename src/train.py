"""Training routine for Wine classification with dual model families and MLflow."""
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from src.data import RANDOM_STATE, get_train_test_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
DB_URI = "sqlite:///mlflow.db"


def evaluate_cv(model_cls, params, x_train, y_train):
    """Run 5-fold stratified cross-validation and compute mean metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1_scores, acc_scores, loss_scores = [], [], []

    for train_idx, val_idx in skf.split(x_train, y_train):
        X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        clf = model_cls(**params)
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)

        f1_scores.append(f1_score(y_val, preds, average="macro"))
        acc_scores.append(accuracy_score(y_val, preds))
        loss_scores.append(log_loss(y_val, probs, labels=[0, 1, 2]))

    return np.mean(f1_scores), np.mean(acc_scores), np.mean(loss_scores)


def train_and_track():
    """Execute hyperparameter grid evaluation for RF and GBM."""
    mlflow.set_tracking_uri(DB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    x_train, _, y_train, _ = get_train_test_data()

    configs = [
        ("RandomForest", RandomForestClassifier, {
            "n_estimators": 50, "max_depth": 3, "random_state": RANDOM_STATE
        }),
        ("RandomForest", RandomForestClassifier, {
            "n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE
        }),
        ("RandomForest", RandomForestClassifier, {
            "n_estimators": 150, "max_depth": None, "random_state": RANDOM_STATE
        }),
        ("GradientBoosting", GradientBoostingClassifier, {
            "n_estimators": 50, "learning_rate": 0.05, "max_depth": 3,
            "random_state": RANDOM_STATE
        }),
        ("GradientBoosting", GradientBoostingClassifier, {
            "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
            "random_state": RANDOM_STATE
        }),
        ("GradientBoosting", GradientBoostingClassifier, {
            "n_estimators": 150, "learning_rate": 0.2, "max_depth": 4,
            "random_state": RANDOM_STATE
        }),
    ]

    for model_family, model_cls, params in configs:
        with mlflow.start_run():
            mlflow.set_tag("model_family", model_family)
            mlflow.log_params(params)

            mean_f1, mean_acc, mean_loss = evaluate_cv(
                model_cls, params, x_train, y_train
            )

            mlflow.log_metric("cv_macro_f1", float(mean_f1))
            mlflow.log_metric("cv_accuracy", float(mean_acc))
            mlflow.log_metric("cv_log_loss", float(mean_loss))

            final_model = model_cls(**params)
            final_model.fit(x_train, y_train)

            input_example = x_train.iloc[:2]
            signature = mlflow.models.infer_signature(x_train, final_model.predict(x_train))

            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            print(f"Logged {model_family} {params} -> F1: {mean_f1:.4f}, Acc: {mean_acc:.4f}")


if __name__ == "__main__":
    train_and_track()
