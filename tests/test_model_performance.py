import pytest
import pickle
import mlflow
import dagshub

import pandas as pd

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
    "model_name, stage, holdout_data_path, vectorizer_path",
    [
        (
            "yt_chrome_plugin_model",
            "Staging",
            "data/interim/test.csv",
            "models/vectorizer.pkl",
        )
    ],
)
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):
    try:
        client = MlflowClient()

        # Getting the latest version of the model in the specified stage
        latest_version_info = client.get_latest_versions(
            name=model_name, stages=[stage]
        )
        if latest_version_info:
            latest_version = latest_version_info[0].version
        else:
            latest_version = None

        assert (
            latest_version is not None
        ), f"No model found in the '{stage}' stage for '{model_name}' model name."

        # Loading the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # Loading the vectorizer
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Loading the holdout data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze()
        y_holdout = holdout_data.iloc[:, -1]

        # Handling NaNs
        X_holdout_raw = X_holdout_raw.fillna("")

        # Applying vectorizer
        X_holdout = vectorizer.transform(X_holdout_raw)
        X_holdout_df = pd.DataFrame(X_holdout.toarray())
        X_holdout_df.columns = [str(i) for i in range(X_holdout_df.shape[1])]

        # Predictions
        y_pred = model.predict(X_holdout_df)

        # Calculating performance metrics
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

        # Thresholds
        expected_acc = 0.60
        expected_prec = 0.60
        expected_rec = 0.60
        expected_f1 = 0.60

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

        print(
            f"Performance test passed for model '{model_name}' version {latest_version}."
        )

    except Exception as e:
        pytest.fail(f"Model performance test failed with the error: {e}")
