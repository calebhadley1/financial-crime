from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """
    This feature engineering notebook was created through research from my master's program. 
    The original notebook can be found at `papers/bu_omds/1_ai_for_leaders/milestone_3/notebooks/ibm_eda.ipynb`. 
    """
        
    logger.info("Training some model...")


    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
