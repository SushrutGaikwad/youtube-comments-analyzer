import pytest
import mlflow
import dagshub
import mlflow.pyfunc

from loguru import logger
from mlflow.tracking import MlflowClient

dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="youtube-comments-analyzer",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/SushrutGaikwad/youtube-comments-analyzer.mlflow"
)


@pytest.mark.parametrize(
    "model_name, stage",
    [("yt_chrome_plugin_model", "Staging")],
)
def test_load_latest_staging_model(model_name, stage):
    logger.info(f"Starting model loading test for '{model_name}' in stage '{stage}'")
    client = MlflowClient()

    # Getting the latest version of the model in the specified stage
    latest_version_info = client.get_latest_versions(name=model_name, stages=[stage])
    if latest_version_info:
        latest_version = latest_version_info[0].version
        logger.info(f"Found model version: {latest_version}")
    else:
        latest_version = None
        logger.warning(f"No model found in stage '{stage}' for model '{model_name}'")

    assert (
        latest_version is not None
    ), f"No model found in the '{stage}' stage for '{model_name}' model name."

    try:
        # Loading the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # Ensuring that the model loads successfully
        assert model is not None, "Model failed to load."
        logger.success(
            f"Model '{model_name}' version {latest_version} loaded successfully from the '{stage}' stage."
        )
    except Exception as e:
        logger.error(f"Model loading failed with error: {e}")
        pytest.fail(f"Model failed to load with error: {e}")
