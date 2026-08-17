"""
Feature extraction entry point for the financial crime detection pipeline.

This script loads raw data and extracts features/labels for downstream modeling.

IMPORTANT: This script is for compatibility and should generally not be used in
production workflows. Instead, use train.py which handles the complete pipeline
(split → engineer → preprocess → train) with proper data leakage prevention.

This script is useful for:
- Exploratory data analysis on engineered features
- Debugging feature engineering independently
- Inspecting the intermediate feature representation

For the standard training workflow, use:
    python -m financial_crime.modeling.train
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer

app = typer.Typer()


@app.command()
def main(
    # input
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # output
    output_features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    output_labels_path: Path = PROCESSED_DATA_DIR / "labels.parquet",
    output_feature_engineer_path: Path = MODELS_DIR / "pipeline" / "feature_engineer.pkl",
):
    """
    Extract engineered features and labels from raw dataset. Used prior to training/inference
    to ensure features are ready for modeling

    This produces feature and label CSVs which can be:
    1. Loaded into the Feature Store
    2. Used for exploratory data analysis

    Original notebooks:
    - Research: papers/bu_omds/1_ai_for_leaders/milestone_3/notebooks/ibm_eda.ipynb
    - Reduced notebook: notebooks/1.01-cjjh-ibm.ipynb

    Args:
        input_path: Path to raw dataset CSV
        output_features_path: Path to save engineered features CSV
        output_labels_path: Path to save labels CSV
    """
    logger.info("Generating engineered features from raw dataset...")

    df = pd.read_csv(input_path)

    logger.info("Applying feature engineering...")
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.fit_transform(df)

    X = df_engineered.drop(columns=["Is Laundering", "labeler"])
    y = df_engineered[["ID", "event_timestamp", "Is Laundering", "labeler"]]

    logger.info(f"Saving engineered features dataset to {output_features_path}...")
    X.to_parquet(output_features_path, index=False)

    logger.info(f"Saving labels dataset to {output_labels_path}...")
    y.to_parquet(output_labels_path, index=False)

    logger.info(f"Saving feature engineer to {output_feature_engineer_path}...")
    feature_engineer.save(output_feature_engineer_path)

    logger.success("Feature extraction complete.")


if __name__ == "__main__":
    app()
