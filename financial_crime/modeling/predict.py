from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import classification_report

from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline

app = typer.Typer()


@app.command()
def main(
    # Input
    raw_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    raw_labels_path: Optional[Path] = PROCESSED_DATA_DIR / "test_labels.csv",
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
    
    For inference on new data: Pass raw_features_path to the raw CSV file and set
    raw_labels_path=None to skip evaluation.
    
    NOTE: test_features.csv contains RAW data from train.py (not preprocessed).
    The pipeline applies all transformations automatically.
    """
    logger.info("Performing inference with ML pipeline...")

    logger.info(f"Loading raw features from {raw_features_path}...")
    df = pd.read_csv(raw_features_path)

    # Guard against accidental label leakage when reusing a raw dataset.
    if "Is Laundering" in df.columns:
        logger.warning("Dropping 'Is Laundering' from input features before inference to avoid target leakage.")
        df = df.drop(columns=["Is Laundering"])

    logger.info(f"Loading pipeline from {pipeline_dir}...")
    pipeline = InferencePipeline.load(pipeline_dir)

    logger.info("Performing inference...")
    y_pred = pipeline.predict(df)

    if raw_labels_path is not None:
        logger.info(f"Loading true labels from {raw_labels_path}...")
        y_true = pd.read_csv(raw_labels_path)

        logger.info("Generating classification report...")
        report = classification_report(y_true, y_pred)
        logger.info(f"\n{report}")

    logger.info(f"Saving predictions to {predictions_path}...")
    np.savetxt(predictions_path, y_pred, delimiter=",")
    
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
