import pytest
import pickle
import mlflow
import dagshub
import mlflow.pyfunc

import pandas as pd

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
    "model_name, stage, vectorizer_path",
    [("yt_chrome_plugin_model", "Staging", "models/vectorizer.pkl")],
)
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    logger.info(f"Starting test for model '{model_name}' in stage '{stage}'")
    client = MlflowClient()

    # Getting the latest version of the model in the specified stage
    latest_version_info = client.get_latest_versions(name=model_name, stages=[stage])
    if latest_version_info:
        latest_version = latest_version_info[0].version
        logger.info(f"Found model version: {latest_version}")
    else:
        latest_version = None
        logger.warning(f"No model found in stage '{stage}' for '{model_name}'")

    assert (
        latest_version is not None
    ), f"No model found in the '{stage}' stage for '{model_name}' model name."

    try:
        # Loading the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # Loading the vectorizer
        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Creating a dummy input for the model
        input_text = "hi how are you"
        logger.info(f"Transforming input text: {input_text}")
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray())
        input_df.columns = [str(i) for i in range(input_df.shape[1])]

        # Prediction of the dummy input
        logger.info("Running model prediction")
        prediction = model.predict(input_df)
        logger.info(f"Prediction result: {prediction}")

        # Verifying that the input shape matches the number of vectorizer's features
        assert input_df.shape[1] == len(
            vectorizer.get_feature_names_out()
        ), "Number of input features of the model are not the same as required."
        logger.info("Input features check passed")

        # Verifying that the output shape matches
        assert (
            len(prediction) == input_df.shape[0]
        ), "Output shape is not the same as what is required."
        logger.info("Output shape check passed")

        logger.success(
            f"Model '{model_name}' version {latest_version} successfully processed the dummy input."
        )
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        pytest.fail("Model signature test failed with error: {e}")
