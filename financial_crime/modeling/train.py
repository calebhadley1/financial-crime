from pathlib import Path

from loguru import logger
from sklearn.compose import make_column_transformer
from tqdm import tqdm
import typer
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    # Input
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # Output
    model_path: Path = MODELS_DIR / "model.pkl",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
):
    """
    This feature engineering notebook was created through research from my master's program. 
    The original notebook can be found at `papers/bu_omds/2_capstone/milestone_2/notebooks/Week9.ipynb`.
    I have further reduced it to only the required code in `notebooks/3.01-cjjh-ibm.ipynb` and converted that into a script here.
    """
        
    logger.info("Training hyperparameter-tuned GradientBoostingClassifier with StandardScaling...")

    logger.info(f"Loading features from {features_path}...")
    X = pd.read_csv(features_path)

    logger.info(f"Loading labels from {labels_path}...")
    y = pd.read_csv(labels_path)

    logger.info("Undersampling the majority class to address significant class imbalance...")
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

    # Persisting test data for later evaluation
    logger.info(f"Saving test features to {test_features_path}")
    X_test.to_csv(test_features_path, index=False)
    logger.info(f"Saving test labels to {test_labels_path}")
    y_test.to_csv(test_labels_path, index=False)

    # One Hot encoding for categorical features
    categorical_features = ["Receiving Currency", "Payment Currency", "Payment Format"]
    preprocessor = make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
        remainder="passthrough"
    )

    logger.info("Fitting StandardScaler + GradientBoostingClassifier pipeline with n_estimators=150, max_depth=3...")
    pipeline = make_pipeline(
        preprocessor,
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            random_state=random_state
        )
    )
    pipeline.fit(X_train, y_train)

    logger.info("Saving trained pipeline...")
    with open(model_path, "wb") as file:
        pickle.dump(pipeline, file)

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
