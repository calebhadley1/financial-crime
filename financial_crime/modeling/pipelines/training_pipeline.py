"""
Training pipeline for financial crime detection model.

This module provides a unified orchestration layer for the complete training workflow,
mirroring the inference pipeline but handling data splitting, fitting, and resampling.
This ensures consistency between training and inference conditions.
"""

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from financial_crime.config import COLUMNS_TO_DROP, DECISION_THRESHOLD, RANDOM_STATE, SAMPLING_STRATEGY, TEST_SIZE
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline
from financial_crime.modeling.transformers.feature_preprocessing import FeaturePreprocessor


class TrainingPipeline:
    """
    Orchestrates the complete training workflow with proper separation of concerns.

    Ensures no data leakage by:
    1. Splitting train/test first (on raw data)
    2. Fitting all transformers only on training data
    3. Applying transformations to both train and test consistently
    4. Resampling only on training data

    This class provides a clear, auditable training process that mirrors the inference
    pipeline structure.
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        sampling_strategy: float = SAMPLING_STRATEGY,
        random_state: int = RANDOM_STATE,
    ):
        """Initialize training pipeline configuration.

        Args:
            test_size: Proportion of data to use for testing (default from config)
            sampling_strategy: Undersampling ratio for majority class
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.preprocessor = None
        self.resampler = None

    def split_data(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/test sets on raw data.

        Prevents data leakage by splitting before any transformations.

        Args:
            X: Raw features DataFrame
            y: Labels DataFrame

        Returns:
            Tuple of (X_train_raw, X_test_raw, y_train, y_test)
        """
        logger.info("Splitting data into chronological train/test sets on raw data...")
        if "event_timestamp" not in X.columns:
            raise ValueError("Chronological splitting requires an 'event_timestamp' column")

        sort_order = pd.to_datetime(X["event_timestamp"]).sort_values().index
        X_ordered = X.loc[sort_order]
        y_ordered = y.loc[sort_order]
        split_index = int(len(X_ordered) * (1 - self.test_size))
        X_train_raw, X_test_raw = X_ordered.iloc[:split_index], X_ordered.iloc[split_index:]
        y_train, y_test = y_ordered.iloc[:split_index], y_ordered.iloc[split_index:]
        logger.info(f"Training set shape: {X_train_raw.shape}, Test set shape: {X_test_raw.shape}")
        return X_train_raw, X_test_raw, y_train, y_test

    def handle_class_imbalance(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Resample training data to handle class imbalance.

        Only resamples training data - never touch test data.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (X_train_resampled, y_train_resampled)
        """
        logger.info("Handling class imbalance with undersampling on training data only...")
        logger.info(f"Original class distribution: {Counter(y_train.values.flatten())}")

        self.resampler = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy, random_state=self.random_state
        )
        X_train_resampled, y_train_resampled = self.resampler.fit_resample(X_train, y_train)
        X_train_resampled = X_train_resampled.sort_index()
        y_train_resampled = y_train_resampled.sort_index()
        y_train_resampled_flattened = y_train_resampled.values.flatten()

        logger.info(f"Resampled class distribution: {Counter(y_train_resampled_flattened)}")
        return X_train_resampled, y_train_resampled_flattened

    def preprocess_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, fit: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply preprocessing (encoding + scaling) to train and test data.

        Args:
            X_train: Training features (already engineered and resampled)
            X_test: Test features (already engineered, NOT resampled)
            fit: Whether to fit the preprocessor on training data (default True)

        Returns:
            Tuple of (X_train_preprocessed, X_test_preprocessed)
        """
        logger.info("Applying feature preprocessing (encoding + scaling)...")

        # Drop any columns that are not needed for modeling (e.g., identifiers)
        X_train = X_train.drop(columns=COLUMNS_TO_DROP, errors="ignore")
        X_test = X_test.drop(columns=COLUMNS_TO_DROP, errors="ignore")

        if fit:
            self.preprocessor = FeaturePreprocessor()
            X_train_preprocessed = self.preprocessor.fit_transform(X_train)
            logger.debug("FeaturePreprocessor fitted on training data")
        else:
            if self.preprocessor is None:
                raise ValueError(
                    "FeaturePreprocessor not fitted. Set fit=True or provide a fitted preprocessor."
                )
            X_train_preprocessed = self.preprocessor.transform(X_train)

        X_test_preprocessed = self.preprocessor.transform(X_test)
        logger.info(f"Features after preprocessing: shape={X_train_preprocessed.shape}")

        return X_train_preprocessed, X_test_preprocessed

    def train(
        self, X: pd.DataFrame, y: pd.DataFrame, model
    ) -> tuple[
        InferencePipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        """Execute complete training workflow.

        Orchestrates: split → engineer → resample → preprocess → train model

        Args:
            X: Features DataFrame
            y: Labels DataFrame
            model: Unfitted sklearn model to train

        Returns:
            Tuple of (pipeline, X_train_preprocessed, y_train_resampled, X_test, X_test_preprocessed, y_test)

        Note:
            X_test is returned for saving to disk (for model evaluation/reproduction)
            X_test_preprocessed is preprocessed but NOT resampled.
            y_test is the original (un-resampled) test labels.
        """
        logger.info("=" * 60)
        logger.info("Starting complete training workflow")
        logger.info("=" * 60)

        # Step 1: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Step 3: Handle class imbalance (training data only)
        X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_train, y_train)

        # Step 4: Preprocessing
        X_train_preprocessed, _ = self.preprocess_features(X_train_resampled, X_test, fit=True)

        # Step 5: Train model
        logger.info(f"Training model: {type(model).__name__}...")
        model.fit(X_train_preprocessed, y_train_resampled)
        logger.success("Model training complete")

        # Step 6: Create pipeline for inference
        logger.info("Creating inference pipeline...")
        pipeline = InferencePipeline(self.preprocessor, model)

        logger.info("Testing pipeline on holdout set")
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        logger.info("Generating classification report...")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")

        logger.info("=" * 60)
        logger.info("Training workflow complete")
        logger.info("=" * 60)

        return pipeline
