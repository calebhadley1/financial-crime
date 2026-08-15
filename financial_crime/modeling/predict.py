from pathlib import Path
from typing import Optional
import pandas as pd
import pickle
from loguru import logger
from sklearn.metrics import classification_report
from tqdm import tqdm
import typer
import numpy as np

from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # Input
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    labels_path: Optional[Path] = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # Output
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    logger.info("Performing inference for model...")

    logger.info(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)

    logger.info(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    logger.info("Performing inference...")
    y_pred = model.predict(df)

    if labels_path is not None:
        logger.info(f"Loading true labels from {labels_path}...")
        y_true = pd.read_csv(labels_path)

        logger.info("Generating classification report...")
        report = classification_report(y_true, y_pred)
        logger.info(f"\n{report}")

    logger.info(f"Saving predictions to {predictions_path}...")
    np.savetxt(predictions_path, y_pred, delimiter=",")
    
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
