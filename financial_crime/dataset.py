from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import typer

from financial_crime.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "HI-Small_Trans.csv",
    output_holdout_set_path: Path = PROCESSED_DATA_DIR / "dataset_50k.csv",
    output_remainder_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info(f"Processing dataset from {input_path}...")
    logger.info("TODO: Add automatic downloading of the dataset from Kaggle using Kaggle SDK")
    df = pd.read_csv(input_path)

    logger.info("Splitting dataset into a 50k final holdout set and remaining...")

    df_last_10k = df.iloc[-50000:]
    df_remainder = df.iloc[:-50000]

    logger.info(f"Saving processed dataset with {len(df_remainder)} rows to {output_remainder_path}...")
    df_remainder.to_csv(output_remainder_path, index=False)

    logger.info(f"Saving processed holdout set with {len(df_last_10k)} rows to {output_holdout_set_path}...")
    df_last_10k.to_csv(output_holdout_set_path, index=False)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
