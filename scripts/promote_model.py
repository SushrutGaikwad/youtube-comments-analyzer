import mlflow
import dagshub

from loguru import logger

dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="youtube-comments-analyzer",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/SushrutGaikwad/youtube-comments-analyzer.mlflow"
)


def promote_model():
    try:
        logger.info("Starting model promotion process...")
        client = mlflow.MlflowClient()

        model_name = "yt_chrome_plugin_model"

        # Getting the latest version of the model in the staging stage
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        assert staging_versions, "No model found in Staging stage."
        latest_version_staging = staging_versions[0].version
        logger.info(f"Latest staging version: {latest_version_staging}")

        # Archiving the current model in the production stage
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            logger.info("Archiving current Production versions...")
            for version in prod_versions:
                logger.info(f"Archiving version: {version.version}")
                client.transition_model_version_stage(
                    name=model_name, version=version.version, stage="Archived"
                )
        else:
            logger.info("No existing models in Production to archive.")

        # Promoting the new model (in staging stage) to production stage
        client.transition_model_version_stage(
            name=model_name, version=latest_version_staging, stage="Production"
        )
        logger.success(
            f"Model version {latest_version_staging} promoted to Production."
        )
    except Exception as e:
        logger.error(f"Model promotion failed: {e}")


if __name__ == "__main__":
    promote_model()
