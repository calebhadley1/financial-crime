from pathlib import Path

from feast import FeatureStore
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import typer

from financial_crime.config import DECISION_THRESHOLD, MODELS_DIR, PROCESSED_DATA_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer

app = typer.Typer()


@app.command()
def main(
    # Input
    input_path: Path = PROCESSED_DATA_DIR / "dataset_50k.csv",
    pipeline_dir: Path = MODELS_DIR / "pipeline",
    # Output
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Perform inference using the trained ML pipeline.

    The pipeline handles the complete transformation:
    raw data → feature engineering → preprocessing → model prediction

    For evaluation: By default, runs predictions on test data that was held out during
    training and compares against true labels.
    """
    logger.info("Performing inference with ML pipeline...")

    logger.info(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info("Loading FeatureEngineer")
    feature_engineer = FeatureEngineer.load(pipeline_dir / "feature_engineer.pkl")

    logger.info("Applying FE to holdout set...")
    df_engineered = feature_engineer.fit_transform(df)

    # TODO: Create a class for feature store interaction. We replicate this code a bit.

    logger.info("Pushing Data into the FS...")
    feature_store = FeatureStore(
        "financial_crime/feature_store/feature_repo"
    )  # Initialize the feature store
    feature_store.push("transaction_push_source", df_engineered)

    logger.info("Retrieving Features from Feature Store...")
    logger.info(f"Loading {len(df_engineered)} rows from FS")

    entity_df = df_engineered[["ID", "event_timestamp"]]
    entity_rows = entity_df.to_dict(orient="records")

    feature_service = feature_store.get_feature_service("transaction_v1")
    # Pull engineered features and labels from the feature store using the entity DataFrame
    # SQLite throws OperationalError: too many SQL variables at full entity rows, so batching is performed
    SQLITE_CHUNK_SIZE = 10_000
    df_chunks = []
    # Fetch from Redis in safe chunks
    for i in range(0, len(entity_rows), SQLITE_CHUNK_SIZE):
        chunk = entity_rows[i : i + SQLITE_CHUNK_SIZE]

        response = feature_store.get_online_features(
            features=feature_service,
            entity_rows=chunk,
        )

        # Convert directly to DataFrame and store the reference
        df_chunks.append(response.to_df())

    # Concatenate all fragments into one master DataFrame
    inference_data = pd.concat(df_chunks, ignore_index=True)

    logger.info(f"Loaded {len(inference_data)} rows from Feature Store")

    # Guard against accidental label leakage when reusing a raw dataset.
    X = inference_data.drop(columns="Is Laundering", errors="ignore")
    y = df.get("Is Laundering")

    logger.info(f"Loading pipeline from {pipeline_dir}...")
    pipeline = InferencePipeline.load(pipeline_dir)

    logger.info("Performing inference...")
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    if y is not None:
        logger.info("Generating classification report...")
        logger.info(y)
        logger.info(y_pred)
        report = classification_report(y, y_pred)
        logger.info(f"\n{report}")
    else:
        logger.info("No labels found in input; skipping classification report.")

    logger.info(f"Saving predictions to {predictions_path}...")
    np.savetxt(predictions_path, y_pred, delimiter=",")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
