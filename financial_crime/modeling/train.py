from pathlib import Path
import pickle

import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier

from financial_crime.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    MODEL_N_ESTIMATORS,
    MODEL_MAX_DEPTH,
    SAMPLING_STRATEGY,
    TEST_SIZE,
)
from financial_crime.modeling.pipelines.training_pipeline import TrainingPipeline

app = typer.Typer()


@app.command()
def main(
    # Input
    raw_features_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    raw_labels_path: Path = None,
    # Output
    pipeline_dir: Path = MODELS_DIR / "pipeline",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
):
    """
    Train the complete ML pipeline: split → feature engineering → preprocessing → model.
    
    This script ensures no data leakage by using TrainingPipeline orchestration:
    1. Splitting train/test first (on raw data)
    2. Fitting all transformers only on training data
    3. Applying transformations to both train and test consistently
    4. Resampling only on training data
    
    Args:
        raw_features_path: Path to raw dataset CSV (must include "Is Laundering" column)
        raw_labels_path: Optional path to pre-extracted labels CSV. If not provided,
                        "Is Laundering" column will be extracted from raw_features_path.
        pipeline_dir: Directory to save trained pipeline
        test_features_path: Path to save raw test features
        test_labels_path: Path to save test labels
    """
        
    logger.info("Training ML pipeline with feature engineering, preprocessing, and model...")

    logger.info(f"Loading raw features from {raw_features_path}...")
    X_raw = pd.read_csv(raw_features_path)

    # Extract or load labels
    if raw_labels_path is not None and Path(raw_labels_path).exists():
        logger.info(f"Loading pre-extracted labels from {raw_labels_path}...")
        y = pd.read_csv(raw_labels_path)
    else:
        logger.info("Extracting labels from 'Is Laundering' column in raw features...")
        if "Is Laundering" not in X_raw.columns:
            raise ValueError(
                "Column 'Is Laundering' not found in raw features. "
                "Provide raw_labels_path or ensure dataset contains this column."
            )
        y = X_raw[["Is Laundering"]]

    # Initialize model
    logger.info(
        f"Initializing GradientBoostingClassifier with "
        f"n_estimators={MODEL_N_ESTIMATORS}, max_depth={MODEL_MAX_DEPTH}..."
    )
    model = GradientBoostingClassifier(
        n_estimators=MODEL_N_ESTIMATORS,
        max_depth=MODEL_MAX_DEPTH,
        random_state=RANDOM_STATE
    )

    # Execute training pipeline orchestration
    training_pipeline = TrainingPipeline(
        test_size=TEST_SIZE,
        sampling_strategy=SAMPLING_STRATEGY,
        random_state=RANDOM_STATE
    )
    
    (
        ml_pipeline,
        X_train_preprocessed,
        y_train_resampled,
        X_test_raw,
        X_test_preprocessed,
        y_test,
    ) = training_pipeline.train(X_raw, y, model)

    # Persist raw test data for evaluation (not preprocessed)
    logger.info(f"Saving raw test features to {test_features_path}")
    X_test_raw.to_csv(test_features_path, index=False)
    logger.info(f"Saving test labels to {test_labels_path}")
    y_test.to_csv(test_labels_path, index=False)

    # Save complete pipeline
    logger.info("Saving complete ML pipeline...")
    ml_pipeline.save(pipeline_dir)

    logger.success("ML pipeline training complete.")


if __name__ == "__main__":
    app()
