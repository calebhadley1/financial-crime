from pathlib import Path
from typing import Optional

from feast import FeatureStore
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import classification_report

from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer

app = typer.Typer()


@app.command()
def main(
    # Input
    input_path: Path = PROCESSED_DATA_DIR / "dataset_50k.csv",
    pipeline_dir: Path = MODELS_DIR / "pipeline",
    # Output
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.parquet",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.parquet",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Perform inference using the trained ML pipeline.
    
    The pipeline handles the complete transformation:
    raw data → feature engineering → preprocessing → model prediction
    
    For evaluation: By default, runs predictions on test data that was held out during
    training and compares against true labels.
    
    For inference on new data: Pass raw_features_path to the raw CSV file and set
    raw_labels_path=None to skip evaluation.
    
    NOTE: test_features.csv contains RAW data from train.py (not preprocessed).
    The pipeline applies all transformations automatically.
    """
    logger.info("Performing inference with ML pipeline...")

    # In the real world we would not be doing the insertion into the feature store here
    # We would instead have a feature eng job that does this and loads into the store separately
    # The API will only be responsible for preprocessing/predicting the engineered features

    logger.info(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info('-' * 50)
    logger.info("TODO: Extract this into a standalone orchestrator it should not be in predict file")
    
    logger.info("Loading FeatureEngineer")
    feature_engineer = FeatureEngineer.load(pipeline_dir / "feature_engineer.pkl")

    logger.info("Applying FE to holdout set...")
    df_engineered = feature_engineer.fit_transform(df)    

    logger.info("Pushing Data into the FS...")
    feature_store = FeatureStore('financial_crime/feature_store/feature_repo')  # Initialize the feature store
    feature_store.push("transaction_push_source", df_engineered)
    logger.info('-' * 50)

    logger.info("Retrieving Features from Feature Store...")
    logger.info(f"Loading {len(df_engineered)} rows from FS")

    entity_df = df_engineered[
        ["ID", "event_timestamp"]
    ]
    entity_rows = entity_df.to_dict(orient="records")

    feature_service = feature_store.get_feature_service("transaction_v1")
    # Pull engineered features and labels from the feature store using the entity DataFrame
    inference_data = feature_store.get_online_features(features=feature_service, entity_rows=entity_rows[:100]).to_df()
    logger.info(f"Loaded {len(inference_data)} rows from Feature Store")
    
    # Guard against accidental label leakage when reusing a raw dataset.
    X = inference_data.drop(columns="Is Laundering")
    y = inference_data["Is Laundering"] # TODO: Y isnt coming through

    logger.info(f"Loading pipeline from {pipeline_dir}...")
    pipeline = InferencePipeline.load(pipeline_dir)

    logger.info("Performing inference...")
    y_pred = pipeline.predict(X)

    logger.info("Generating classification report...")
    logger.info(y)
    logger.info(y_pred)
    report = classification_report(y, y_pred)
    logger.info(f"\n{report}")

    logger.info(f"Saving predictions to {predictions_path}...")
    np.savetxt(predictions_path, y_pred, delimiter=",")
    
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
