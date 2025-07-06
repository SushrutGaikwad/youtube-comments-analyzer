import pytest
import pickle
import mlflow
import dagshub

import pandas as pd

from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="youtube-comments-analyzer",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/SushrutGaikwad/youtube-comments-analyzer.mlflow"
)


@pytest.mark.parametrize(
    "model_name, stage, X_holdout_path, y_holdout_path",
    [
        (
            "yt_chrome_plugin_model",
            "Staging",
            "data/processed/X_test.csv",
            "data/processed/y_test.csv",
        )
    ],
)
def test_model_performance(model_name, stage, X_holdout_path, y_holdout_path):
    try:
        logger.info(
            f"Starting performance test for model '{model_name}' in stage '{stage}'"
        )
        client = MlflowClient()

        # Getting the latest version of the model in the specified stage
        latest_version_info = client.get_latest_versions(
            name=model_name, stages=[stage]
        )
        if latest_version_info:
            latest_version = latest_version_info[0].version
            logger.info(f"Found model version: {latest_version}")
        else:
            latest_version = None
            logger.warning(
                f"No model found in stage '{stage}' for model '{model_name}'"
            )

        assert (
            latest_version is not None
        ), f"No model found in the '{stage}' stage for '{model_name}' model name."

        # Loading the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # Loading the holdout data
        logger.info(f"Reading X_holdout from {X_holdout_path}")
        X_holdout = pd.read_csv(X_holdout_path)

        logger.info(f"Reading y_holdout from {y_holdout_path}")
        y_holdout = pd.read_csv(y_holdout_path)

        # Predictions
        logger.info("Generating predictions on holdout data")
        y_pred = model.predict(X_holdout)

        # Calculating performance metrics
        logger.info("Calculating performance metrics")
        acc = accuracy_score(y_true=y_holdout, y_pred=y_pred)
        prec = precision_score(
            y_true=y_holdout, y_pred=y_pred, average="weighted", zero_division=1
        )
        rec = recall_score(
            y_true=y_holdout, y_pred=y_pred, average="weighted", zero_division=1
        )
        f1 = f1_score(
            y_true=y_holdout, y_pred=y_pred, average="weighted", zero_division=1
        )
        logger.info(
            f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        # Thresholds
        expected_acc = 0.75
        expected_prec = 0.75
        expected_rec = 0.75
        expected_f1 = 0.75

        # Asserts
        assert (
            acc >= expected_acc
        ), f"Accuracy should at least be {expected_acc}, instead got {acc}."
        assert (
            prec >= expected_prec
        ), f"Precision should at least be {expected_prec}, instead got {prec}."
        assert (
            rec >= expected_rec
        ), f"Recall should at least be {expected_rec}, instead got {rec}."
        assert (
            f1 >= expected_f1
        ), f"F1 score should at least be {expected_f1}, instead got {f1}."

        logger.success(
            f"Performance test passed for model '{model_name}' version {latest_version}."
        )

    except Exception as e:
        logger.error(f"Model performance test failed with error: {e}")
        pytest.fail(f"Model performance test failed with the error: {e}")
