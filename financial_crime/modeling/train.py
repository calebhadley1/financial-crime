from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import pickle
from sklearn.pipeline import make_pipeline
from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

random_state = 42


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """
    This feature engineering notebook was created through research from my master's program. 
    The original notebook can be found at `papers/bu_omds/2_capstone/milestone_2/notebooks/Week9.ipynb`.
    I have further reduced it to only the required code in `notebooks/3.01-cjjh-ibm.ipynb` and converted that into a script here.
    """
        
    logger.info("Training hyperparameter-tuned GradientBoostingClassifier with StandardScaling...")

    logger.info(f"Loading features from {features_path}...")
    df = pd.read_csv(f'{PROCESSED_DATA_DIR}/features.csv')

    X = df.drop(columns='Is Laundering')
    y = df['Is Laundering']

    logger.info(f"Original dataset shape: {Counter(y)}")
    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    logger.info(f"Resampled dataset shape: {Counter(y_resampled)}")

    logger.info("Creating 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, 
        y_resampled, 
        test_size=0.2, 
        stratify=y_resampled,  # Maintains class distribution
        random_state=random_state
    )

    logger.info("Fitting StandardScaler + GradientBoostingClassifier pipeline with n_estimators=150, max_depth=3...")
    pipeline = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            random_state=random_state
        )
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    logger.info("Classification Report:")
    logger.info(classification_report(y_true=y_test, y_pred=y_pred))

    logger.info("Saving trained pipeline...")
    with open(f"{MODELS_DIR}/pipeline.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
