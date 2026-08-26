from pathlib import Path

from feast import FeatureStore
from loguru import logger
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import typer

from financial_crime.config import (
    MODEL_MAX_DEPTH,
    MODEL_N_ESTIMATORS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SAMPLING_STRATEGY,
    TEST_SIZE,
)
from financial_crime.modeling.pipelines.training_pipeline import TrainingPipeline

app = typer.Typer()


@app.command()
def main(
    # Input
    features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    labels_path: Path | None = None,
    feature_engineer_path: Path = MODELS_DIR / "pipeline" / "feature_engineer.pkl",
    feature_repo_path: Path = Path("financial_crime/feature_store/feature_repo"),
    # Output
    pipeline_dir: Path = MODELS_DIR / "pipeline",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
):
    """
    Train the complete ML pipeline: split → preprocessing → model.

    This script ensures no data leakage by using TrainingPipeline orchestration:
    1. Splitting train/test first (on raw data)
    2. Fitting all transformers only on training data
    3. Applying transformations to both train and test consistently
    4. Resampling only on training data

    Args:
        features_path: Path to engineered features Parquet file
        labels_path: Path to labels CSV file
        pipeline_dir: Directory to save trained pipeline
        test_features_path: Path to save raw test features
        test_labels_path: Path to save test labels
    """

    logger.info("Training ML pipeline with preprocessing and model...")

    logger.info(f"Loading engineered features from {features_path}...")
    # Retrieve entity DataFrame for historical feature retrieval
    entity_df = pd.read_parquet(features_path)[["ID", "event_timestamp"]]
    logger.info(f"Loaded {len(entity_df)} rows")

    logger.info("Retrieving Features from Feature Store...")
    feature_store = FeatureStore(str(feature_repo_path))
    feature_service = feature_store.get_feature_service("transaction_v1")
    # Pull engineered features and labels from the feature store using the entity DataFrame
    training_data = feature_store.get_historical_features(
        features=feature_service, entity_df=entity_df
    ).to_df()
    logger.info(f"Loaded {len(training_data)} rows from Feature Store")

    X = training_data.drop(columns=["Is Laundering"])
    y = training_data[["Is Laundering"]]

    # Initialize model
    logger.info(
        f"Initializing GradientBoostingClassifier with "
        f"n_estimators={MODEL_N_ESTIMATORS}, max_depth={MODEL_MAX_DEPTH}..."
    )
    model = GradientBoostingClassifier(
        n_estimators=MODEL_N_ESTIMATORS, max_depth=MODEL_MAX_DEPTH, random_state=RANDOM_STATE
    )

    # Execute training pipeline orchestration
    training_pipeline = TrainingPipeline(
        test_size=TEST_SIZE, sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE
    )

    ml_pipeline = training_pipeline.train(X, y, model)

    # Save complete pipeline
    logger.info("Saving complete ML pipeline...")
    ml_pipeline.save(pipeline_dir)

    logger.success("ML pipeline training complete.")


if __name__ == "__main__":
    app()
