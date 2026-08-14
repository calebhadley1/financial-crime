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
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info(f"Processing dataset from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info("For now we are just duplicating the dataset into processed")
    logger.info("TODO: Add automatic downloading of the dataset from Kaggle using Kaggle SDK")
    logger.info("TODO: Create an initial train/test split of the dataset so that we can simulate real-time inference using data sources like kafka")
    
    logger.info(f"Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
