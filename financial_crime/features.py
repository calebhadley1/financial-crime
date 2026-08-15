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

import pandas as pd
from loguru import logger
import typer

from financial_crime.config import PROCESSED_DATA_DIR
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_labels_path: Path = PROCESSED_DATA_DIR / "labels.csv"
):
    """
    Extract engineered features and labels from raw dataset.
    
    This creates intermediate CSV files for exploratory analysis.
    
    WARNING: These intermediate features.csv and labels.csv files should NOT be used
    with train.py, as train.py expects raw data and applies all transformations
    internally to prevent data leakage.
    
    Original notebooks:
    - Feature research: papers/bu_omds/1_ai_for_leaders/milestone_3/notebooks/ibm_eda.ipynb
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

    logger.info(f"Saving engineered features dataset to {output_features_path}...")
    df_engineered.to_csv(output_features_path, index=False)

    logger.info(f"Saving labels dataset to {output_labels_path}...")
    df[["Is Laundering"]].to_csv(output_labels_path, index=False)
    
    logger.success("Feature extraction complete.")
    logger.warning("These intermediate files are for EDA only. Use train.py for the full pipeline.")


if __name__ == "__main__":
    app()
